"""Run an Explore-style agent against a real local repo using the MockClient.

No API key, no Docker. Demonstrates the full pipeline end-to-end:
  bootstrap → assemble_tool_pool → query loop → tool execution → terminal.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

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


async def main() -> None:
    boot = bootstrap()
    workdir = Path.cwd()

    # Script the assistant: glob for .py, then read first match, then summarize.
    tu_glob = ToolUseBlock(id="t1", name="Glob", input={"pattern": "**/*.py", "head_limit": 5})
    tu_read = ToolUseBlock(id="t2", name="Read", input={"file_path": "repo2rocm/cli.py", "limit": 20})

    client = MockClient(
        scripted_responses=[
            [
                ChunkText(text="Let me see what's here.", block_index=0),
                ChunkToolUse(tool_use=tu_glob, block_index=1),
                ChunkUsage(usage=TokenUsage(input_tokens=200, output_tokens=10)),
                ChunkDone(
                    assistant_message=AssistantMessage(
                        content=[TextBlock(text="Let me see what's here."), tu_glob]
                    ),
                    stop_reason="tool_use",
                ),
            ],
            [
                ChunkText(text="Now reading the CLI.", block_index=0),
                ChunkToolUse(tool_use=tu_read, block_index=1),
                ChunkUsage(usage=TokenUsage(input_tokens=300, output_tokens=10)),
                ChunkDone(
                    assistant_message=AssistantMessage(
                        content=[TextBlock(text="Now reading the CLI."), tu_read]
                    ),
                    stop_reason="tool_use",
                ),
            ],
            [
                ChunkText(text="FINDINGS: this is a Typer CLI exposing migrate/batch/mcp/doctor.", block_index=0),
                ChunkUsage(usage=TokenUsage(input_tokens=400, output_tokens=20)),
                ChunkDone(
                    assistant_message=AssistantMessage(
                        content=[TextBlock(text="FINDINGS: this is a Typer CLI exposing migrate/batch/mcp/doctor.")]
                    ),
                    stop_reason="end_turn",
                ),
            ],
        ]
    )

    tools, specs = assemble_tool_pool(allowed=["Read", "Grep", "Glob"])
    transcripts = TranscriptStore(workdir / "examples" / "out")
    ctx = ToolUseContext(
        agent_id="explore-demo",
        session_id=transcripts.session_id,
        workdir=workdir,
        abort_event=asyncio.Event(),
        permission_mode=PermissionMode.PLAN,
        read_file_state=ReadFileState(),
        transcript=transcripts.transcript("explore-demo"),
    )

    runner = query(
        messages=[UserMessage(content="Scan the repo and tell me what it is.")],
        system_prompt=SystemPrompt.from_text("You are an Explore worker."),
        tools=tools,
        tool_specs=specs,
        client=client,
        tool_use_context=ctx,
        max_turns=10,
    )

    print(f"session_id = {transcripts.session_id}")
    print("---")
    async for ev in runner:
        if ev.kind == "text":
            print(f"[TEXT] {ev.payload.text}")
        elif ev.kind == "tool_use":
            print(f"[TOOL_USE] {ev.payload.tool_use.name}({ev.payload.tool_use.input})")
        elif ev.kind == "tool_result":
            tt = ev.payload
            text = (tt.result.text if tt.result else "")[:120]
            print(f"[TOOL_RESULT] {tt.tool.name} -> {text!r}")
        elif ev.kind == "usage":
            u = ev.payload.usage
            print(f"[USAGE] in={u.input_tokens} out={u.output_tokens}")
    print("---")
    print(f"terminal = {runner.terminal.reason}, turns={runner.terminal.turns}")
    print(f"transcript at {ctx.transcript.path}")


if __name__ == "__main__":
    asyncio.run(main())
