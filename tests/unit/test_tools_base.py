"""BaseTool defaults, registry, and validation."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import BaseModel

from repo2rocm.core.permissions import PermissionMode
from repo2rocm.tools.base import (
    BaseTool,
    ReadFileState,
    ToolResult,
    ToolUseContext,
    clear_registry,
    get_all_tools,
    get_tool,
    register_tool,
)


class _SampleInput(BaseModel):
    x: int


class _SampleOutput(BaseModel):
    y: int


class _SampleTool(BaseTool[_SampleInput, _SampleOutput]):
    name: ClassVar[str] = "Sample"
    description: ClassVar[str] = "double x"
    input_model: ClassVar[type[BaseModel]] = _SampleInput

    def is_concurrency_safe(self, parsed: _SampleInput) -> bool:
        return True

    def is_read_only(self, parsed: _SampleInput) -> bool:
        return True

    async def call(self, parsed: _SampleInput, ctx: ToolUseContext) -> ToolResult[_SampleOutput]:
        return ToolResult(data=_SampleOutput(y=parsed.x * 2), text=str(parsed.x * 2))


@pytest.fixture
def ctx(tmp_path: Path) -> ToolUseContext:
    return ToolUseContext(
        agent_id="t1",
        session_id="s1",
        workdir=tmp_path,
        abort_event=asyncio.Event(),
        permission_mode=PermissionMode.DEFAULT,
        read_file_state=ReadFileState(),
    )


def test_registry_round_trip():
    clear_registry()
    register_tool(_SampleTool)
    assert get_tool("Sample") is not None
    assert any(t.name == "Sample" for t in get_all_tools())


@pytest.mark.asyncio
async def test_invoke_validates_input(ctx: ToolUseContext):
    clear_registry()
    tool = register_tool(_SampleTool)
    bad = await tool.invoke({"x": "not an int"}, ctx)
    assert bad.is_error
    ok = await tool.invoke({"x": 21}, ctx)
    assert not ok.is_error
    assert ok.text == "42"


@pytest.mark.asyncio
async def test_invoke_persists_overflow_to_disk(ctx: ToolUseContext):
    class _Big(BaseTool[_SampleInput, _SampleOutput]):
        name: ClassVar[str] = "Big"
        description: ClassVar[str] = "huge"
        input_model: ClassVar[type[BaseModel]] = _SampleInput
        max_result_size_chars: ClassVar[int] = 100

        def is_concurrency_safe(self, parsed: _SampleInput) -> bool:
            return True

        async def call(self, parsed: _SampleInput, ctx: ToolUseContext) -> ToolResult[_SampleOutput]:
            return ToolResult(data=_SampleOutput(y=0), text="x" * 1000)

    clear_registry()
    tool = register_tool(_Big)
    res = await tool.invoke({"x": 1}, ctx)
    assert not res.is_error
    assert "<persisted-output" in res.text
