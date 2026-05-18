"""Token counting — API-anchored with conservative estimation for new messages.

Mirror of `tokenCountWithEstimation` from Claude Code: prefer the API's authoritative
`usage` field for everything up to the last response, then add a conservative
estimate (chars/4) for messages added after that.
"""
from __future__ import annotations

from typing import Iterable

from repo2rocm.core.messages import (
    AssistantMessage,
    ContentBlock,
    Message,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

_CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: str | None) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def estimate_message_tokens(msg: Message) -> int:
    if isinstance(msg, SystemMessage):
        return estimate_tokens(msg.content)
    if isinstance(msg, UserMessage) and isinstance(msg.content, str):
        return estimate_tokens(msg.content)
    blocks: Iterable[ContentBlock]
    if isinstance(msg, UserMessage):
        blocks = msg.content  # type: ignore[assignment]
    else:
        blocks = msg.content
    return sum(estimate_block_tokens(b) for b in blocks)


def estimate_block_tokens(block: ContentBlock) -> int:
    if isinstance(block, TextBlock):
        return estimate_tokens(block.text)
    if isinstance(block, ThinkingBlock):
        return estimate_tokens(block.thinking)
    if isinstance(block, ToolUseBlock):
        return estimate_tokens(str(block.input)) + 32
    if isinstance(block, ToolResultBlock):
        if isinstance(block.content, str):
            return estimate_tokens(block.content)
        return sum(estimate_tokens(str(c)) for c in block.content)
    return 32  # ImageBlock or future block types


def estimate_messages_tokens(messages: list[Message]) -> int:
    return sum(estimate_message_tokens(m) for m in messages)


def token_count_with_estimation(
    *,
    messages: list[Message],
    last_api_usage_input: int,
    boundary_index: int,
) -> int:
    """Treat messages up to `boundary_index` as the API-anchored count;
    estimate everything after that.

    Conservative: errs toward higher counts so auto-compact fires slightly early.
    """
    if boundary_index <= 0:
        return estimate_messages_tokens(messages)
    after = messages[boundary_index:]
    return last_api_usage_input + estimate_messages_tokens(after)
