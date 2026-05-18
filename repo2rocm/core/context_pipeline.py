"""Four-layer compaction pipeline. Run in this fixed order before every model call.

Layers:
  1. tool-result budget   (enforce per-message size caps)
  2. snip compact          (physically drop old messages, with boundary marker)
  3. microcompact          (drop tool results no longer needed by tool_use_id)
  4. context collapse      (replace spans with summaries — uses LLM)
  5. auto-compact          (full conversation summarization — heaviest)

Each layer is a pure function: `(messages, ctx) -> (messages, freed_tokens)`.
Auto-compact has a circuit breaker (3 consecutive failures).
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from repo2rocm.core.messages import (
    AssistantMessage,
    Message,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from repo2rocm.core.state import LoopState
from repo2rocm.core.token_count import (
    estimate_block_tokens,
    estimate_message_tokens,
    estimate_messages_tokens,
)
from repo2rocm.observability.logging import get_logger
from repo2rocm.observability.metrics import METRICS
from repo2rocm.observability.tracing import span

log = get_logger(__name__)

AUTOCOMPACT_BUFFER_TOKENS = 13_000
MANUAL_COMPACT_BUFFER_TOKENS = 3_000
MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES = 3


@dataclass
class CompactionResult:
    messages: list[Message]
    freed_tokens: int
    layers_applied: list[str]
    blocking: bool = False


SummarizeFn = Callable[[list[Message]], Awaitable[str]]


async def compress_if_needed(
    *,
    state: LoopState,
    context_window: int,
    summarize: SummarizeFn | None = None,
    tool_max_result_chars: dict[str, int] | None = None,
) -> CompactionResult:
    """Run the four-layer pipeline. Returns the messages to send to the API."""
    effective_window = context_window - min(state.max_output_tokens, 20_000)
    autocompact_threshold = effective_window - AUTOCOMPACT_BUFFER_TOKENS
    blocking_threshold = effective_window - MANUAL_COMPACT_BUFFER_TOKENS

    messages = list(state.messages)
    freed_total = 0
    layers: list[str] = []

    # Layer 0: tool-result budget per-tool
    if tool_max_result_chars:
        messages, f = _apply_tool_result_budget(messages, tool_max_result_chars)
        freed_total += f
        if f:
            METRICS.context_compactions.labels(layer="tool_result_budget", outcome="ok").inc()
            layers.append("tool_result_budget")

    tokens_now = estimate_messages_tokens(messages)
    if tokens_now < autocompact_threshold:
        return CompactionResult(messages=messages, freed_tokens=freed_total, layers_applied=layers)

    # Layer 1: snip — drop oldest non-system turn pair, insert boundary marker
    if tokens_now >= autocompact_threshold:
        messages, f = _snip(messages, target_free=tokens_now - autocompact_threshold + 2_000)
        freed_total += f
        if f:
            layers.append("snip")
            METRICS.context_compactions.labels(layer="snip", outcome="ok").inc()
        tokens_now = estimate_messages_tokens(messages)

    # Layer 2: microcompact — drop tool results without matching tool_use (orphans)
    if tokens_now >= autocompact_threshold:
        messages, f = _microcompact(messages)
        freed_total += f
        if f:
            layers.append("microcompact")
            METRICS.context_compactions.labels(layer="microcompact", outcome="ok").inc()
        tokens_now = estimate_messages_tokens(messages)

    # Layer 3: context collapse — replace spans with summaries (LLM-backed)
    if tokens_now >= autocompact_threshold and summarize is not None:
        with span("context.collapse"):
            try:
                messages, f = await _collapse(messages, summarize)
                freed_total += f
                if f:
                    layers.append("collapse")
                    METRICS.context_compactions.labels(layer="collapse", outcome="ok").inc()
                tokens_now = estimate_messages_tokens(messages)
            except Exception as exc:  # noqa: BLE001
                METRICS.context_compactions.labels(layer="collapse", outcome="error").inc()
                log.warning("context collapse failed", error=str(exc))

    # Layer 4: auto-compact — full summarization (heaviest, last resort)
    if (
        tokens_now >= autocompact_threshold
        and summarize is not None
        and state.consecutive_compact_failures < MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES
    ):
        with span("context.auto_compact"):
            try:
                messages, f = await _auto_compact(messages, summarize)
                freed_total += f
                if f:
                    layers.append("auto_compact")
                    METRICS.context_compactions.labels(layer="auto_compact", outcome="ok").inc()
                tokens_now = estimate_messages_tokens(messages)
            except Exception as exc:  # noqa: BLE001
                METRICS.context_compactions.labels(layer="auto_compact", outcome="error").inc()
                log.warning("auto-compact failed", error=str(exc))

    blocking = tokens_now >= blocking_threshold
    return CompactionResult(
        messages=messages,
        freed_tokens=freed_total,
        layers_applied=layers,
        blocking=blocking,
    )


# ── Layer implementations ────────────────────────────────────────────────────


def _apply_tool_result_budget(
    messages: list[Message], caps: dict[str, int]
) -> tuple[list[Message], int]:
    freed = 0
    out: list[Message] = []
    for m in messages:
        if isinstance(m, UserMessage) and isinstance(m.content, list):
            new_content = []
            for b in m.content:
                if isinstance(b, ToolResultBlock):
                    text = b.content if isinstance(b.content, str) else str(b.content)
                    # we cap at the largest of any known cap (we don't know which tool produced it
                    # without scanning the prior tool_use); use the max as a soft cap
                    cap = max(caps.values(), default=100_000)
                    if len(text) > cap:
                        freed += (len(text) - cap) // 4
                        text = (
                            text[:cap]
                            + f"\n... [truncated by tool_result_budget; {len(text) - cap} more chars]"
                        )
                        b = ToolResultBlock(
                            tool_use_id=b.tool_use_id, content=text, is_error=b.is_error
                        )
                new_content.append(b)
            out.append(UserMessage(content=new_content))
        else:
            out.append(m)
    return out, freed


def _snip(messages: list[Message], *, target_free: int) -> tuple[list[Message], int]:
    """Drop oldest message-pairs until we've freed `target_free` tokens.

    Always preserves the first user message (the task prompt).
    """
    if len(messages) <= 4:
        return messages, 0

    keep_head = 1  # the initial user prompt
    keep_tail = 4  # keep recent turns intact

    freed = 0
    drop_until = keep_head
    cursor = keep_head
    while cursor < len(messages) - keep_tail and freed < target_free:
        freed += estimate_message_tokens(messages[cursor])
        cursor += 1
    drop_until = cursor

    if drop_until <= keep_head:
        return messages, 0

    boundary = SystemMessage(
        content=f"[context-snip: removed {drop_until - keep_head} older messages "
        f"freeing ~{freed} tokens]",
        kind="boundary",
    )
    new = messages[:keep_head] + [boundary] + messages[drop_until:]
    return new, freed


def _microcompact(messages: list[Message]) -> tuple[list[Message], int]:
    """Drop tool_result blocks whose tool_use_id no longer appears in any assistant tool_use."""
    live_tool_use_ids: set[str] = set()
    for m in messages:
        if isinstance(m, AssistantMessage):
            for b in m.content:
                if isinstance(b, ToolUseBlock):
                    live_tool_use_ids.add(b.id)

    freed = 0
    out: list[Message] = []
    for m in messages:
        if isinstance(m, UserMessage) and isinstance(m.content, list):
            new_content = []
            for b in m.content:
                if isinstance(b, ToolResultBlock) and b.tool_use_id not in live_tool_use_ids:
                    freed += estimate_block_tokens(b)
                    continue
                new_content.append(b)
            if new_content:
                out.append(UserMessage(content=new_content))
        else:
            out.append(m)
    return out, freed


async def _collapse(
    messages: list[Message], summarize: SummarizeFn
) -> tuple[list[Message], int]:
    """Replace a middle span (between keep_head and keep_tail) with one summary."""
    if len(messages) <= 6:
        return messages, 0

    keep_head, keep_tail = 2, 4
    middle = messages[keep_head : len(messages) - keep_tail]
    if not middle:
        return messages, 0

    before_tokens = estimate_messages_tokens(middle)
    try:
        summary = await summarize(middle)
    except Exception:
        return messages, 0

    new = (
        messages[:keep_head]
        + [SystemMessage(content=f"[context-collapse summary]\n{summary}", kind="boundary")]
        + messages[-keep_tail:]
    )
    freed = before_tokens - estimate_tokens_str(summary)
    return new, max(0, freed)


async def _auto_compact(
    messages: list[Message], summarize: SummarizeFn
) -> tuple[list[Message], int]:
    """Heavy compaction: replace the entire history (except the first user msg + last 2) with a summary."""
    if len(messages) <= 4:
        return messages, 0
    target = messages[1:-2]
    before_tokens = estimate_messages_tokens(target)
    try:
        summary = await summarize(target)
    except Exception:
        return messages, 0
    new = (
        [messages[0]]
        + [SystemMessage(content=f"[auto-compact summary]\n{summary}", kind="boundary")]
        + messages[-2:]
    )
    freed = before_tokens - estimate_tokens_str(summary)
    return new, max(0, freed)


def estimate_tokens_str(s: str) -> int:
    return max(1, len(s) // 4)
