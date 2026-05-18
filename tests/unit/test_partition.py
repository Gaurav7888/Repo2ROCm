"""Partition algorithm: consecutive safe tools batch together."""
from __future__ import annotations

from repo2rocm.core.messages import ToolUseBlock
from repo2rocm.tools.base import clear_registry, register_tool
from repo2rocm.tools.executor.partition import partition_tool_calls
from repo2rocm.tools.repo.edit import Edit
from repo2rocm.tools.repo.glob import Glob
from repo2rocm.tools.repo.grep import Grep
from repo2rocm.tools.repo.read import Read


def _setup():
    clear_registry()
    for cls in (Read, Grep, Glob, Edit):
        register_tool(cls)


def _call(name: str, idx: int, input_: dict) -> ToolUseBlock:
    return ToolUseBlock(id=f"t{idx}", name=name, input=input_)


def test_safe_tools_merge_into_one_batch():
    _setup()
    calls = [
        _call("Read", 1, {"file_path": "a"}),
        _call("Read", 2, {"file_path": "b"}),
        _call("Grep", 3, {"pattern": "x"}),
    ]
    batches = partition_tool_calls(calls)
    assert len(batches) == 1
    assert batches[0].parallel is True
    assert len(batches[0].calls) == 3


def test_unsafe_tool_breaks_batch():
    _setup()
    calls = [
        _call("Read", 1, {"file_path": "a"}),
        _call("Edit", 2, {"file_path": "a", "old_string": "x", "new_string": "y"}),
        _call("Read", 3, {"file_path": "b"}),
    ]
    batches = partition_tool_calls(calls)
    assert len(batches) == 3
    assert batches[0].parallel and len(batches[0].calls) == 1
    assert not batches[1].parallel
    assert batches[2].parallel and len(batches[2].calls) == 1


def test_unknown_tool_defaults_serial():
    _setup()
    calls = [
        _call("UnknownThing", 1, {}),
        _call("Read", 2, {"file_path": "a"}),
    ]
    batches = partition_tool_calls(calls)
    assert not batches[0].parallel
