"""Context pipeline: snip + microcompact."""
from __future__ import annotations

import pytest

from repo2rocm.core.context_pipeline import compress_if_needed
from repo2rocm.core.messages import (
    AssistantMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from repo2rocm.core.state import LoopState


def _make_long_messages(n: int) -> list:
    msgs = [UserMessage(content="task: do the thing")]
    for i in range(n):
        msgs.append(AssistantMessage(content=[TextBlock(text=("x" * 500) + f" step {i}")]))
        msgs.append(UserMessage(content=[TextBlock(text=("y" * 500) + f" reply {i}")]))
    return msgs


@pytest.mark.asyncio
async def test_snip_drops_old_messages():
    state = LoopState.init(_make_long_messages(40))
    result = await compress_if_needed(state=state, context_window=8_000)
    assert "snip" in result.layers_applied
    assert result.freed_tokens > 0
    # boundary marker present
    text_blob = " ".join(
        m.content if isinstance(m.content, str) else " ".join(getattr(b, "text", "") for b in m.content)
        for m in result.messages
    )
    assert "context-snip" in text_blob


@pytest.mark.asyncio
async def test_microcompact_drops_orphan_tool_results():
    # one tool_use with id t1 -> followed by tool_result for t1 AND an orphan t2 (no tool_use exists)
    msgs = [
        UserMessage(content="task"),
        AssistantMessage(content=[ToolUseBlock(id="t1", name="Read", input={"file_path": "x"})]),
        UserMessage(
            content=[
                ToolResultBlock(tool_use_id="t1", content="ok"),
                ToolResultBlock(tool_use_id="t2_orphan", content="orphan"),
            ]
        ),
    ]
    state = LoopState.init(msgs)
    # force compaction by setting low budget
    result = await compress_if_needed(state=state, context_window=200)
    # orphan removed
    user_blocks = [b for m in result.messages if isinstance(m, UserMessage) and isinstance(m.content, list) for b in m.content]
    orphan_ids = [b.tool_use_id for b in user_blocks if isinstance(b, ToolResultBlock)]
    assert "t2_orphan" not in orphan_ids
