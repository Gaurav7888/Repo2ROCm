"""Greedy partition algorithm. Mirror of `partitionToolCalls` in Ch. 7 of the book.

Walks the tool_use list left-to-right; merges consecutive concurrency-safe tools into one
batch; any unsafe tool starts a new (serial) batch. Fail-closed: parse failures or
exceptions inside `is_concurrency_safe` default to serial.
"""
from __future__ import annotations

from dataclasses import dataclass

from repo2rocm.core.messages import ToolUseBlock
from repo2rocm.tools.base import BaseTool, get_tool


@dataclass
class Batch:
    parallel: bool
    calls: list[ToolUseBlock]


def partition_tool_calls(calls: list[ToolUseBlock]) -> list[Batch]:
    batches: list[Batch] = []
    for call in calls:
        tool = get_tool(call.name)
        safe = False
        if tool is not None:
            try:
                parsed = tool.input_model.model_validate(call.input)
                safe = bool(tool.is_concurrency_safe(parsed))
            except Exception:
                safe = False
        if safe and batches and batches[-1].parallel:
            batches[-1].calls.append(call)
        else:
            batches.append(Batch(parallel=safe, calls=[call]))
    return batches


def can_execute_alongside(new_tool: BaseTool, new_input: dict, running_tools: list[BaseTool]) -> bool:
    """Streaming executor admission check."""
    if not running_tools:
        return True
    try:
        new_parsed = new_tool.input_model.model_validate(new_input)
        if not new_tool.is_concurrency_safe(new_parsed):
            return False
    except Exception:
        return False
    return all(getattr(t, "_is_safe_cached", False) for t in running_tools)
