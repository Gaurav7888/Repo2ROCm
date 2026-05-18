"""StreamingToolExecutor: speculative concurrency + sibling cascade + order preservation."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import BaseModel

from repo2rocm.core.messages import ToolUseBlock
from repo2rocm.core.permissions import PermissionMode
from repo2rocm.tools.base import (
    BaseTool,
    ReadFileState,
    ToolResult,
    ToolUseContext,
    clear_registry,
    register_tool,
)
from repo2rocm.tools.executor.streaming import StreamingToolExecutor, ToolStatus


class _SlowIn(BaseModel):
    delay: float = 0.0
    name: str = "anon"
    fail: bool = False


class _SlowOut(BaseModel):
    name: str


class _SlowSafeTool(BaseTool[_SlowIn, _SlowOut]):
    name: ClassVar[str] = "Slow"
    description: ClassVar[str] = "slow but safe"
    input_model: ClassVar[type[BaseModel]] = _SlowIn

    def is_concurrency_safe(self, parsed: _SlowIn) -> bool:
        return True

    def is_read_only(self, parsed: _SlowIn) -> bool:
        return True

    async def call(self, parsed: _SlowIn, ctx: ToolUseContext) -> ToolResult[_SlowOut]:
        await asyncio.sleep(parsed.delay)
        if parsed.fail:
            return ToolResult(data=_SlowOut(name=parsed.name), text=f"fail:{parsed.name}", is_error=True)
        return ToolResult(data=_SlowOut(name=parsed.name), text=parsed.name)


def _ctx(tmp_path: Path) -> ToolUseContext:
    return ToolUseContext(
        agent_id="a",
        session_id="s",
        workdir=tmp_path,
        abort_event=asyncio.Event(),
        permission_mode=PermissionMode.BYPASS,
        read_file_state=ReadFileState(),
    )


@pytest.mark.asyncio
async def test_unknown_tool_yields_error_without_crashing(tmp_path: Path):
    """Regression: model calls a tool name we don't have; must produce an error result,
    not raise TypeError. The Coordinator needs the failure surfaced so it can correct."""
    clear_registry()
    ctx = _ctx(tmp_path)
    executor = StreamingToolExecutor(ctx)
    executor.add_tool(ToolUseBlock(id="t-x", name="DefinitelyNotARealTool", input={}))
    out = []
    async for t in executor.get_remaining_results():
        out.append(t)
    assert len(out) == 1
    assert out[0].result is not None
    assert out[0].result.is_error
    assert "DefinitelyNotARealTool" in out[0].result.text or "Unknown tool" in out[0].result.text


@pytest.mark.asyncio
async def test_safe_tools_run_concurrently_preserve_order(tmp_path: Path):
    clear_registry()
    register_tool(_SlowSafeTool)
    ctx = _ctx(tmp_path)
    executor = StreamingToolExecutor(ctx)
    for i, delay in enumerate((0.10, 0.02, 0.04)):
        executor.add_tool(ToolUseBlock(id=f"t{i}", name="Slow", input={"delay": delay, "name": str(i)}))
    out_order: list[str] = []
    async for t in executor.get_remaining_results():
        out_order.append(t.tool.name + ":" + (t.result.text if t.result else ""))
    # tool-use IDs were t0, t1, t2; results must be in that submission order
    assert out_order == ["Slow:0", "Slow:1", "Slow:2"]
    assert all(t.status == ToolStatus.YIELDED for t in executor.tools)
