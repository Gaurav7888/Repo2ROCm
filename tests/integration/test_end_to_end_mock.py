"""End-to-end mock: a Coordinator-flavored loop driving Read + DockerExec
through a MockClient. No real Docker required (we stub the sandbox).
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from repo2rocm.bootstrap import bootstrap
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
from repo2rocm.observability.transcripts import TranscriptStore
from repo2rocm.tools.base import ReadFileState, ToolUseContext
from repo2rocm.tools.registry import assemble_tool_pool


@pytest.mark.asyncio
async def test_mocked_migration_flow(tmp_workdir: Path):
    bootstrap()  # registers all tools + skills + hooks

    # Three-turn scripted scenario:
    #   1. assistant calls Read("requirements.txt")
    #   2. assistant calls Read("src/main.py")
    #   3. assistant says "done"
    tu1 = ToolUseBlock(id="tu1", name="Read", input={"file_path": "requirements.txt"})
    tu2 = ToolUseBlock(id="tu2", name="Read", input={"file_path": "src/main.py"})

    client = MockClient(
        scripted_responses=[
            [
                ChunkText(text="Let me inspect requirements first.", block_index=0),
                ChunkToolUse(tool_use=tu1, block_index=1),
                ChunkUsage(usage=TokenUsage(input_tokens=200, output_tokens=20)),
                ChunkDone(
                    assistant_message=AssistantMessage(
                        content=[TextBlock(text="Let me inspect requirements first."), tu1]
                    ),
                    stop_reason="tool_use",
                ),
            ],
            [
                ChunkText(text="And the main entry point.", block_index=0),
                ChunkToolUse(tool_use=tu2, block_index=1),
                ChunkUsage(usage=TokenUsage(input_tokens=250, output_tokens=20)),
                ChunkDone(
                    assistant_message=AssistantMessage(
                        content=[TextBlock(text="And the main entry point."), tu2]
                    ),
                    stop_reason="tool_use",
                ),
            ],
            [
                ChunkText(text="The repo is a torch project with one CUDA-only dep.", block_index=0),
                ChunkUsage(usage=TokenUsage(input_tokens=300, output_tokens=20)),
                ChunkDone(
                    assistant_message=AssistantMessage(
                        content=[TextBlock(text="The repo is a torch project with one CUDA-only dep.")]
                    ),
                    stop_reason="end_turn",
                ),
            ],
        ]
    )

    tools, specs = assemble_tool_pool(allowed=["Read", "Grep", "Glob"])
    transcripts = TranscriptStore(tmp_workdir / "out")
    ctx = ToolUseContext(
        agent_id="explore-a",
        session_id=transcripts.session_id,
        workdir=tmp_workdir,
        abort_event=asyncio.Event(),
        permission_mode=PermissionMode.PLAN,
        read_file_state=ReadFileState(),
        transcript=transcripts.transcript("explore-a"),
    )

    runner = query(
        messages=[UserMessage(content="Scan the repo for CUDA-only dependencies.")],
        system_prompt=SystemPrompt.from_text("You are an Explore worker."),
        tools=tools,
        tool_specs=specs,
        client=client,
        tool_use_context=ctx,
        max_turns=10,
    )

    text_chunks: list[str] = []
    tool_uses_seen: list[str] = []
    tool_results_seen: list[str] = []
    async for ev in runner:
        if ev.kind == "text":
            text_chunks.append(ev.payload.text)
        elif ev.kind == "tool_use":
            tool_uses_seen.append(ev.payload.tool_use.name)
        elif ev.kind == "tool_result":
            tool_results_seen.append(ev.payload.tool.name)

    assert runner.terminal is not None
    assert runner.terminal.reason == "completed"
    assert "torch project" in " ".join(text_chunks)
    assert tool_uses_seen == ["Read", "Read"]
    assert tool_results_seen.count("Read") == 2

    # Transcript was written
    records = ctx.transcript.read_all()
    kinds = {r["kind"] for r in records}
    assert "tool_result" in kinds
    assert "terminal" in kinds
