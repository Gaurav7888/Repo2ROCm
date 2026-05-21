"""The async-generator agent loop — Repo2ROCm's only model+tool driver.

Yields `LoopEvent` objects (messages + executor updates). The terminal reason is
exposed via the `QueryRun.terminal` attribute once iteration completes.

Per Ch. 5 of the Claude Code book: this is the single function that touches every
other subsystem.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from repo2rocm.core.api import (
    ChunkDone,
    ChunkError,
    ChunkText,
    ChunkThinking,
    ChunkToolUse,
    ChunkUsage,
    ModelClient,
    ToolSpec,
    stream_model,
)
from repo2rocm.core.context_pipeline import compress_if_needed
from repo2rocm.core.hooks import HookEvent, execute_hooks
from repo2rocm.core.messages import (
    AssistantMessage,
    Message,
    SystemPrompt,
    TextBlock,
    ToolResultBlock,
    UserMessage,
)
from repo2rocm.core.state import LoopState
from repo2rocm.core.terminal import (
    AbortedTools,
    Completed,
    MaxTurns,
    ModelError,
    PromptTooLong,
    Terminal,
)
from repo2rocm.observability.logging import get_logger
from repo2rocm.observability.metrics import METRICS
from repo2rocm.observability.tracing import span
from repo2rocm.tools.base import BaseTool, ToolUseContext
from repo2rocm.tools.executor import StreamingToolExecutor

log = get_logger(__name__)


@dataclass
class LoopEvent:
    """Stream of events the caller can observe."""

    kind: str  # "text" | "tool_use" | "tool_result" | "thinking" | "usage" | "error" | "boundary"
    payload: Any


SummarizeFn = Callable[[list[Message]], Awaitable[str]]


class QueryRun:
    """Wrapper around the agent loop. Iterate to consume events; read `.terminal` when done.

    The wrapper exists because Python async generators have no first-class return value.
    """

    def __init__(
        self,
        *,
        messages: list[Message],
        system_prompt: SystemPrompt,
        tools: list[BaseTool],
        tool_specs: list[ToolSpec],
        client: ModelClient,
        tool_use_context: ToolUseContext,
        max_turns: int = 100,
        context_window: int = 200_000,
        agent_type: str = "default",
        summarize: SummarizeFn | None = None,
        workspace_trusted: bool = True,
    ):
        self.messages = messages
        self.system_prompt = system_prompt
        self.tools = tools
        self.tool_specs = tool_specs
        self.client = client
        self.tool_use_context = tool_use_context
        self.max_turns = max_turns
        self.context_window = context_window
        self.agent_type = agent_type
        self.summarize = summarize
        self.workspace_trusted = workspace_trusted
        self.terminal: Terminal | None = None
        self.final_state: LoopState | None = None

    def __aiter__(self) -> AsyncIterator[LoopEvent]:
        return self._run()

    async def _run(self) -> AsyncIterator[LoopEvent]:
        state = LoopState.init(self.messages, max_turns=self.max_turns)
        terminal: Terminal | None = None

        try:
            while True:
                if self.tool_use_context.abort_event.is_set():
                    terminal = AbortedTools(turns=state.turn_count, usage=state.usage)
                    break

                with METRICS.time_turn(self.agent_type):
                    with span(
                        "agent.turn",
                        turn=state.turn_count,
                        agent_type=self.agent_type,
                        agent_id=self.tool_use_context.agent_id,
                    ):
                        # 1. Compress
                        comp = await compress_if_needed(
                            state=state,
                            context_window=self.context_window,
                            summarize=self.summarize,
                        )
                        messages_for_api = comp.messages
                        if comp.layers_applied and self.tool_use_context.transcript is not None:
                            self.tool_use_context.transcript.append(
                                {
                                    "kind": "context_compaction",
                                    "turn": state.turn_count,
                                    "layers": comp.layers_applied,
                                    "freed_tokens": comp.freed_tokens,
                                }
                            )

                        # 2. Stream the model + speculative tool execution
                        executor = StreamingToolExecutor(self.tool_use_context)
                        assistant_msg: AssistantMessage | None = None
                        stream_error: ChunkError | None = None

                        try:
                            async for chunk in stream_model(
                                client=self.client,
                                messages=messages_for_api,
                                system=self.system_prompt,
                                tools=self.tool_specs,
                                max_tokens=state.max_output_tokens,
                                stop_event=self.tool_use_context.abort_event,
                            ):
                                if isinstance(chunk, ChunkText):
                                    yield LoopEvent("text", chunk)
                                elif isinstance(chunk, ChunkThinking):
                                    yield LoopEvent("thinking", chunk)
                                elif isinstance(chunk, ChunkToolUse):
                                    yield LoopEvent("tool_use", chunk)
                                    executor.add_tool(chunk.tool_use)
                                    for t in executor.get_completed_results():
                                        yield LoopEvent("tool_result", t)
                                elif isinstance(chunk, ChunkUsage):
                                    yield LoopEvent("usage", chunk)
                                    state = state.with_(usage=state.usage + chunk.usage)
                                elif isinstance(chunk, ChunkError):
                                    stream_error = chunk
                                    if not chunk.recoverable:
                                        yield LoopEvent("error", chunk)
                                elif isinstance(chunk, ChunkDone):
                                    assistant_msg = chunk.assistant_message
                        except asyncio.CancelledError:
                            executor.discard()
                            terminal = AbortedTools(turns=state.turn_count, usage=state.usage)
                            raise

                        # 3. Drain remaining tools
                        tool_result_blocks: list[ToolResultBlock] = []
                        seen_tool_result_ids: set[str] = set()
                        async for t in executor.get_remaining_results():
                            yield LoopEvent("tool_result", t)
                            if t.result is not None:
                                text = t.result.text
                                if not isinstance(text, str):
                                    text = str(text)
                                if not text.strip():
                                    text = "[internal-empty-tool-result]"
                                tool_result_blocks.append(
                                    ToolResultBlock(
                                        tool_use_id=t.tool_use.id,
                                        content=text,
                                        is_error=t.result.is_error,
                                    )
                                )
                                seen_tool_result_ids.add(t.tool_use.id)

                        # 4. Error handling for the API call itself
                        if assistant_msg is None:
                            if stream_error is not None and stream_error.recoverable:
                                if not state.has_attempted_reactive_compact:
                                    state = state.with_(
                                        has_attempted_reactive_compact=True,
                                        transition_reason="reactive_compact_retry",
                                    )
                                    continue
                            msg = stream_error.message if stream_error else "no response"
                            err_class = stream_error.error_class if stream_error else "unknown"
                            if self.tool_use_context.transcript is not None:
                                try:
                                    self.tool_use_context.transcript.append(
                                        {
                                            "kind": "stream_error",
                                            "turn": state.turn_count,
                                            "error_class": err_class,
                                            "message": msg[:2000],
                                            "recoverable": (
                                                stream_error.recoverable if stream_error else False
                                            ),
                                        }
                                    )
                                except Exception:
                                    pass
                            log.error(
                                "model error",
                                turn=state.turn_count,
                                error_class=err_class,
                                msg=msg[:500],
                            )
                            if "prompt is too long" in msg.lower() or err_class.startswith(
                                "http_413"
                            ):
                                terminal = PromptTooLong(
                                    turns=state.turn_count, usage=state.usage, message=msg
                                )
                            else:
                                terminal = ModelError(
                                    turns=state.turn_count,
                                    usage=state.usage,
                                    message=msg,
                                    error_class=err_class,
                                )
                            break

                        # 4b. Record the assistant message every turn (for debugging)
                        if self.tool_use_context.transcript is not None:
                            try:
                                self.tool_use_context.transcript.append(
                                    {
                                        "kind": "assistant_turn",
                                        "turn": state.turn_count,
                                        "text": assistant_msg.text()[:4000],
                                        "tool_uses": [
                                            {"id": t.id, "name": t.name, "input": t.input}
                                            for t in assistant_msg.tool_uses()
                                        ],
                                        "stop_reason": assistant_msg.stop_reason,
                                    }
                                )
                            except Exception:
                                pass

                        # 5. Decision: tool_use? continue. Otherwise done.
                        tool_uses = assistant_msg.tool_uses()
                        if not tool_uses:
                            hook_outcome = await execute_hooks(
                                event=HookEvent.STOP,
                                input_data={"final_text": assistant_msg.text()},
                                workspace_trusted=self.workspace_trusted,
                            )
                            if hook_outcome.blocked or hook_outcome.prevent_continuation:
                                new_messages = [
                                    *messages_for_api,
                                    assistant_msg,
                                    UserMessage(
                                        content=[
                                            TextBlock(
                                                text=f"[stop-hook-blocking]\n{hook_outcome.block_reason or hook_outcome.additional_context}"
                                            )
                                        ]
                                    ),
                                ]
                                state = state.with_(
                                    messages=new_messages,
                                    stop_hook_active=True,
                                    turn_count=state.turn_count + 1,
                                    transition_reason="stop_hook_blocking",
                                )
                                continue
                            terminal = Completed(
                                turns=state.turn_count,
                                usage=state.usage,
                                final_text=assistant_msg.text(),
                            )
                            break

                        # Some provider / executor edge-cases can leave us with
                        # tool_use blocks but no corresponding tool_result
                        # blocks. Sending an empty follow-up user message causes
                        # Vertex/OpenAI-compatible APIs to reject the request.
                        # Synthesize explicit error tool_results so the next turn
                        # is always well-formed and the model can recover.
                        missing_tool_results = [
                            tu for tu in tool_uses if tu.id not in seen_tool_result_ids
                        ]
                        for tu in missing_tool_results:
                            tool_result_blocks.append(
                                ToolResultBlock(
                                    tool_use_id=tu.id,
                                    content=(
                                        "[internal-tool-error] The tool call produced no "
                                        "result. Treat this as a failed tool invocation "
                                        "and either retry with different input or choose "
                                        "a different tool."
                                    ),
                                    is_error=True,
                                )
                            )

                        # 6. Reconstruct state for next turn
                        new_messages = [
                            *messages_for_api,
                            assistant_msg,
                            UserMessage(content=list(tool_result_blocks)),
                        ]
                        state = state.with_(
                            messages=new_messages,
                            turn_count=state.turn_count + 1,
                            transition_reason="next_turn",
                            has_attempted_reactive_compact=False,
                        )

                        if state.turn_count >= self.max_turns:
                            terminal = MaxTurns(turns=state.turn_count, usage=state.usage)
                            break
        finally:
            self.terminal = terminal or Completed(turns=state.turn_count, usage=state.usage)
            self.final_state = state
            METRICS.subagent_completions.labels(
                agent_type=self.agent_type, reason=self.terminal.reason
            ).inc()
            if self.tool_use_context.transcript is not None:
                try:
                    rec = {
                        "kind": "terminal",
                        "reason": self.terminal.reason,
                        "turns": self.terminal.turns,
                        "usage": self.terminal.usage.model_dump(),
                    }
                    # surface message+error_class for failure terminals
                    for attr in ("message", "error_class", "final_text", "hook_name"):
                        v = getattr(self.terminal, attr, None)
                        if v:
                            rec[attr] = v[:1000] if isinstance(v, str) else v
                    self.tool_use_context.transcript.append(rec)
                except Exception:
                    pass


def query(**kwargs: Any) -> QueryRun:
    """Convenience constructor."""
    return QueryRun(**kwargs)
