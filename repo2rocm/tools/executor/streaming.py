"""Streaming tool executor.

Mirrors Claude Code's `StreamingToolExecutor` (Ch. 7):
  * `add_tool(tu, asst_msg)` registers a tool while the model streams
  * `process_queue()` admits new tools according to the mutual-exclusion rule:
       can_run = no_tools_running OR (new_is_safe AND all_running_are_safe)
  * `get_completed_results()` yields results in SUBMISSION order (not completion order)
  * `get_remaining_results()` is the post-stream drain
  * Bash-only sibling-error cascade
  * order-preserving result buffer
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator

from repo2rocm.core.messages import ToolUseBlock
from repo2rocm.observability.logging import get_logger
from repo2rocm.observability.metrics import METRICS
from repo2rocm.observability.tracing import span
from repo2rocm.tools.base import BaseTool, ToolResult, ToolUseContext, get_tool

log = get_logger(__name__)

MAX_CONCURRENCY = 10


class ToolStatus(str, Enum):
    QUEUED = "queued"
    EXECUTING = "executing"
    COMPLETED = "completed"
    YIELDED = "yielded"


@dataclass
class TrackedTool:
    tool_use: ToolUseBlock
    tool: BaseTool
    status: ToolStatus = ToolStatus.QUEUED
    result: ToolResult | None = None
    error: Exception | None = None
    is_safe: bool = False
    task: asyncio.Task | None = None
    abort_event: asyncio.Event = field(default_factory=asyncio.Event)


class StreamingToolExecutor:
    def __init__(self, ctx: ToolUseContext):
        self.ctx = ctx
        self.tools: list[TrackedTool] = []
        self.sibling_event = asyncio.Event()  # fires on Bash error
        self._sem = asyncio.Semaphore(MAX_CONCURRENCY)
        self._wake = asyncio.Event()
        self._discarded = False

    # ── External API ─────────────────────────────────────────────────────────

    def add_tool(self, tu: ToolUseBlock) -> None:
        """Called by the streaming parser when a tool_use block completes."""
        tool = get_tool(tu.name)
        if tool is None:
            tracked = TrackedTool(
                tool_use=tu,
                tool=_UnknownTool(tu.name),
                status=ToolStatus.COMPLETED,
                result=ToolResult(
                    data=None,
                    text=f"Unknown tool: {tu.name}",
                    is_error=True,
                ),
            )
            self.tools.append(tracked)
            self._wake.set()
            return

        is_safe = False
        try:
            parsed = tool.input_model.model_validate(tu.input)
            is_safe = bool(tool.is_concurrency_safe(parsed))
        except Exception:
            is_safe = False

        tracked = TrackedTool(tool_use=tu, tool=tool, is_safe=is_safe)
        self.tools.append(tracked)
        # fire-and-forget admission
        asyncio.ensure_future(self.process_queue())

    async def process_queue(self) -> None:
        """Start queued tools that can now run."""
        if self._discarded:
            return
        for t in self.tools:
            if t.status != ToolStatus.QUEUED:
                continue
            running = [x for x in self.tools if x.status == ToolStatus.EXECUTING]
            if not running:
                self._start_tool(t)
                continue
            # mutual exclusion: new safe + all running safe
            if t.is_safe and all(x.is_safe for x in running):
                self._start_tool(t)
                continue
            # cannot admit; stop scanning (preserves order for serial tools)
            if not t.is_safe:
                break

    def _start_tool(self, t: TrackedTool) -> None:
        t.status = ToolStatus.EXECUTING
        t.task = asyncio.ensure_future(self._run(t))

    async def _run(self, t: TrackedTool) -> None:
        async with self._sem:
            with span(
                "executor.run_tool",
                tool=t.tool.name,
                is_safe=t.is_safe,
                tool_use_id=t.tool_use.id,
            ):
                try:
                    if self._discarded:
                        t.error = RuntimeError("executor discarded")
                        t.result = ToolResult(
                            data=None,
                            text="Tool execution discarded (streaming fallback).",
                            is_error=True,
                        )
                        t.status = ToolStatus.COMPLETED
                        return
                    t.result = await t.tool.invoke(t.tool_use.input, self.ctx)
                except asyncio.CancelledError:
                    t.error = asyncio.CancelledError()
                    t.result = ToolResult(
                        data=None,
                        text=f"Cancelled: parallel tool call {t.tool.name} aborted",
                        is_error=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    t.error = exc
                    t.result = ToolResult(
                        data=None,
                        text=f"Tool {t.tool.name} raised {type(exc).__name__}: {exc}",
                        is_error=True,
                    )
                t.status = ToolStatus.COMPLETED

                # Sibling error cascade: only for Bash-style failures.
                if t.result and t.result.is_error and _should_cascade(t.tool.name):
                    self._cascade_sibling_error(t)

                self._wake.set()
                # try to admit the next batch
                asyncio.ensure_future(self.process_queue())

    def _cascade_sibling_error(self, errored: TrackedTool) -> None:
        for other in self.tools:
            if other is errored:
                continue
            if other.status == ToolStatus.EXECUTING and other.task is not None:
                other.task.cancel()
        self.sibling_event.set()

    def discard(self) -> None:
        self._discarded = True
        for t in self.tools:
            if t.status == ToolStatus.EXECUTING and t.task is not None:
                t.task.cancel()
        self._wake.set()

    # ── Result harvesting (order-preserving) ────────────────────────────────

    def get_completed_results(self) -> list[TrackedTool]:
        """Synchronously yield results in STRICT submission order.

        Walks left-to-right and stops at the first non-yielded slot that is
        still in progress. This matches the invariant from Ch. 7 of the
        Claude Code book: the model sees tool_results in the order it emitted
        the corresponding tool_use blocks.
        """
        out: list[TrackedTool] = []
        for t in self.tools:
            if t.status == ToolStatus.YIELDED:
                continue
            if t.status == ToolStatus.COMPLETED:
                t.status = ToolStatus.YIELDED
                out.append(t)
                continue
            # tool[i] is still executing — nothing after it can be yielded
            break
        return out

    async def get_remaining_results(self) -> AsyncIterator[TrackedTool]:
        """Async drain: yield in submission order. Wait for tool[i] before tool[i+1]."""
        for t in self.tools:
            if t.status == ToolStatus.YIELDED:
                continue
            # Ensure queued tools eventually start
            await self.process_queue()
            if t.task is not None:
                try:
                    await t.task
                except (asyncio.CancelledError, Exception):
                    pass
            if t.status == ToolStatus.COMPLETED:
                t.status = ToolStatus.YIELDED
                yield t
            # else: cancelled / discarded — skip silently


def _should_cascade(tool_name: str) -> bool:
    """Only Bash/Docker exec errors cascade. Reads stay independent."""
    return tool_name in {"DockerExec", "Bash"}


class _UnknownTool:
    name = "<unknown>"

    def is_concurrency_safe(self, _: object) -> bool:
        return False
