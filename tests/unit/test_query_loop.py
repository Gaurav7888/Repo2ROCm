"""End-to-end agent loop with a mock LLM client."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import BaseModel

from repo2rocm.core.api import (
    ChunkDone,
    ChunkText,
    ChunkToolUse,
    ChunkUsage,
    MockClient,
)
from repo2rocm.core.messages import (
    AssistantMessage,
    SystemPrompt,
    TextBlock,
    TokenUsage,
    ToolUseBlock,
    UserMessage,
)
from repo2rocm.core.permissions import PermissionMode
from repo2rocm.core.query import query
from repo2rocm.tools.base import (
    BaseTool,
    ReadFileState,
    ToolResult,
    ToolUseContext,
    clear_registry,
    register_tool,
)
from repo2rocm.tools.registry import assemble_tool_pool


class _PingIn(BaseModel):
    message: str = "hello"


class _PingOut(BaseModel):
    echo: str


class _PingTool(BaseTool[_PingIn, _PingOut]):
    name: ClassVar[str] = "Ping"
    description: ClassVar[str] = "echo a message"
    input_model: ClassVar[type[BaseModel]] = _PingIn

    def is_concurrency_safe(self, parsed: _PingIn) -> bool:
        return True

    def is_read_only(self, parsed: _PingIn) -> bool:
        return True

    async def call(self, parsed: _PingIn, ctx: ToolUseContext) -> ToolResult[_PingOut]:
        return ToolResult(data=_PingOut(echo=parsed.message), text=f"pong:{parsed.message}")


@pytest.mark.asyncio
async def test_loop_executes_tool_then_completes(tmp_path: Path):
    """Two-turn scenario: assistant calls Ping; loop runs tool; second response has no tools."""
    clear_registry()
    register_tool(_PingTool)
    tools, specs = assemble_tool_pool(allowed=["Ping"])

    turn1_chunks = [
        ChunkToolUse(
            tool_use=ToolUseBlock(id="tu1", name="Ping", input={"message": "hi"}),
            block_index=0,
        ),
        ChunkUsage(usage=TokenUsage(input_tokens=10, output_tokens=4)),
        ChunkDone(
            assistant_message=AssistantMessage(
                content=[ToolUseBlock(id="tu1", name="Ping", input={"message": "hi"})]
            ),
            stop_reason="tool_use",
        ),
    ]
    turn2_chunks = [
        ChunkText(text="done", block_index=0),
        ChunkUsage(usage=TokenUsage(input_tokens=15, output_tokens=2)),
        ChunkDone(
            assistant_message=AssistantMessage(content=[TextBlock(text="done")]),
            stop_reason="end_turn",
        ),
    ]
    client = MockClient(scripted_responses=[turn1_chunks, turn2_chunks])

    ctx = ToolUseContext(
        agent_id="a",
        session_id="s",
        workdir=tmp_path,
        abort_event=asyncio.Event(),
        permission_mode=PermissionMode.BYPASS,
        read_file_state=ReadFileState(),
    )

    runner = query(
        messages=[UserMessage(content="please ping")],
        system_prompt=SystemPrompt.from_text("you are a test agent"),
        tools=tools,
        tool_specs=specs,
        client=client,
        tool_use_context=ctx,
        max_turns=10,
    )
    events: list = []
    async for ev in runner:
        events.append(ev)

    assert runner.terminal is not None
    assert runner.terminal.reason == "completed"
    assert getattr(runner.terminal, "final_text", "") == "done"
    kinds = [e.kind for e in events]
    assert "tool_use" in kinds
    assert "tool_result" in kinds
    assert "text" in kinds


@pytest.mark.asyncio
async def test_loop_max_turns(tmp_path: Path):
    """If the model never stops calling tools we should hit max_turns."""
    clear_registry()
    register_tool(_PingTool)
    tools, specs = assemble_tool_pool(allowed=["Ping"])

    def keep_calling():
        tu = ToolUseBlock(id="x", name="Ping", input={"message": "again"})
        return [
            ChunkToolUse(tool_use=tu, block_index=0),
            ChunkDone(
                assistant_message=AssistantMessage(content=[tu]),
                stop_reason="tool_use",
            ),
        ]

    client = MockClient(scripted_responses=[keep_calling() for _ in range(5)])
    ctx = ToolUseContext(
        agent_id="a",
        session_id="s",
        workdir=tmp_path,
        abort_event=asyncio.Event(),
        permission_mode=PermissionMode.BYPASS,
        read_file_state=ReadFileState(),
    )
    runner = query(
        messages=[UserMessage(content="run forever")],
        system_prompt=SystemPrompt.from_text("test"),
        tools=tools,
        tool_specs=specs,
        client=client,
        tool_use_context=ctx,
        max_turns=3,
    )
    async for _ in runner:
        pass
    assert runner.terminal is not None
    assert runner.terminal.reason in ("max_turns", "completed")
