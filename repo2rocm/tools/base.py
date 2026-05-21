"""Tool interface, base class, and registry.

Five methods matter (per Ch. 6 of the Claude Code book):
  * `call()`             — execute the tool
  * `input_model`        — Pydantic model serving double duty as schema + validator
  * `is_concurrency_safe()` — can this run in parallel?
  * `check_permissions()` — tool-specific permission opinion
  * `validate_semantic()` — beyond-schema sanity (e.g. reject no-op edits)
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel

from repo2rocm.core.permissions import (
    PermissionDecision,
    PermissionMode,
    passthrough,
)
from repo2rocm.observability.logging import get_logger
from repo2rocm.observability.metrics import METRICS
from repo2rocm.observability.tracing import span

I = TypeVar("I", bound=BaseModel)
O = TypeVar("O", bound=BaseModel)

log = get_logger(__name__)


@dataclass
class FileStateEntry:
    path: Path
    mtime_ns: int
    sha: str


@dataclass
class ReadFileState:
    """LRU cache of files the agent has read, with mtime for staleness checks."""

    capacity: int = 256
    _entries: dict[Path, FileStateEntry] = field(default_factory=dict)

    def record(self, path: Path, mtime_ns: int, sha: str) -> None:
        if len(self._entries) >= self.capacity:
            self._entries.pop(next(iter(self._entries)))
        self._entries[path] = FileStateEntry(path, mtime_ns, sha)

    def get(self, path: Path) -> FileStateEntry | None:
        return self._entries.get(path)

    def clear(self) -> None:
        self._entries.clear()


@dataclass
class ToolUseContext:
    """The "god object" passed to every tool — but with deliberate sub-objects per concern."""

    # identity
    agent_id: str
    session_id: str
    workdir: Path

    # execution
    abort_event: asyncio.Event
    permission_mode: PermissionMode
    read_file_state: ReadFileState

    # sandbox handle (set by agents that need it)
    sandbox: Any | None = None  # repo2rocm.sandbox.manager.Sandbox

    # transcript (set by run_agent)
    transcript: Any | None = None

    # message history (read-only inside tools)
    messages: list[Any] = field(default_factory=list)

    # bookkeeping
    options: dict[str, Any] = field(default_factory=dict)

    # quality-gate state shared with hooks
    gate_state: Any | None = None


@dataclass
class ToolResult(Generic[O]):
    """What `BaseTool.call()` returns."""

    data: O
    text: str  # what the model sees as the `tool_result.content`
    is_error: bool = False
    bytes_used: int = 0
    new_messages: list[Any] = field(default_factory=list)
    context_modifier: Any | None = None  # only honored for non-concurrent tools


# ── BaseTool ─────────────────────────────────────────────────────────────────


class BaseTool(Generic[I, O]):
    """Fail-closed defaults: a new tool that forgets to mark itself concurrency-safe
    runs serially; one that forgets `is_read_only` is treated as a write."""

    # class attrs every tool must set
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    input_model: ClassVar[type[BaseModel]]
    max_result_size_chars: ClassVar[int] = 30_000
    interrupt_behavior: ClassVar[str] = "block"  # "cancel" | "block"

    def is_concurrency_safe(self, parsed: I) -> bool:
        return False

    def is_read_only(self, parsed: I) -> bool:
        return False

    def check_permissions(self, parsed: I, ctx: ToolUseContext) -> PermissionDecision:
        return passthrough()

    def validate_semantic(self, parsed: I, ctx: ToolUseContext) -> str | None:
        """Return error string to reject, or None to accept."""
        return None

    async def call(self, parsed: I, ctx: ToolUseContext) -> ToolResult[O]:
        raise NotImplementedError

    async def progress(
        self, parsed: I, ctx: ToolUseContext
    ) -> AsyncIterator[Any]:
        """Optional streaming progress; default yields nothing."""
        if False:  # pragma: no cover — never executes; satisfies AsyncIterator typing
            yield None

    # provider-agnostic JSON Schema for the model
    @classmethod
    def schema(cls) -> dict[str, Any]:
        return cls.input_model.model_json_schema()

    # ── instrumented entry point used by the executor ──
    async def invoke(self, raw_input: dict[str, Any], ctx: ToolUseContext) -> ToolResult[O]:
        """Validate, permission-check, and call. Instrumented end-to-end."""
        from repo2rocm.core.permissions import (
            PermissionDecisionKind,
            PermissionRuleSet,
            resolve_permission,
        )

        start = time.perf_counter()
        outcome = "ok"
        with span("tool.invoke", tool=self.name, agent_id=ctx.agent_id):
            try:
                parsed = self.input_model.model_validate(raw_input)  # type: ignore[assignment]
            except Exception as exc:
                outcome = "validation_error"
                METRICS.tool_calls.labels(tool=self.name, outcome=outcome).inc()
                return ToolResult(
                    data=self._empty_output(),  # type: ignore[arg-type]
                    text=f"Input validation failed: {exc}",
                    is_error=True,
                )

            sem = self.validate_semantic(parsed, ctx)  # type: ignore[arg-type]
            if sem:
                outcome = "semantic_error"
                METRICS.tool_calls.labels(tool=self.name, outcome=outcome).inc()
                return ToolResult(
                    data=self._empty_output(),  # type: ignore[arg-type]
                    text=sem,
                    is_error=True,
                )

            decision = resolve_permission(
                self, raw_input, ctx, rules=PermissionRuleSet.empty()
            )
            if decision.kind == PermissionDecisionKind.DENY:
                outcome = "permission_denied"
                METRICS.tool_calls.labels(tool=self.name, outcome=outcome).inc()
                # Surface denies in the transcript — otherwise an agent that retries
                # a denied tool 20× looks identical to a silent agent in the log,
                # which is exactly what caused the original "planner just hung" bug.
                if ctx.transcript is not None:
                    try:
                        ctx.transcript.append(
                            {
                                "kind": "tool_result",
                                "tool": self.name,
                                "outcome": outcome,
                                "reason": decision.reason,
                                "input": raw_input,
                                "permission_mode": ctx.permission_mode.value,
                            }
                        )
                    except Exception:
                        pass
                return ToolResult(
                    data=self._empty_output(),  # type: ignore[arg-type]
                    text=f"Permission denied: {decision.reason}",
                    is_error=True,
                )

            try:
                with METRICS.time_tool(self.name):
                    result = await self.call(parsed, ctx)  # type: ignore[arg-type]
            except asyncio.CancelledError:
                outcome = "cancelled"
                raise
            except Exception as exc:  # noqa: BLE001
                outcome = "error"
                log.exception("tool call failed", tool=self.name)
                METRICS.tool_calls.labels(tool=self.name, outcome=outcome).inc()
                return ToolResult(
                    data=self._empty_output(),  # type: ignore[arg-type]
                    text=f"Tool {self.name} raised {type(exc).__name__}: {exc}",
                    is_error=True,
                )

            # Result budget
            if len(result.text) > self.max_result_size_chars:
                result = self._persist_overflow(result, ctx)
                outcome = "truncated"
            elapsed = time.perf_counter() - start
            METRICS.tool_calls.labels(tool=self.name, outcome=outcome).inc()
            METRICS.tool_result_bytes.labels(tool=self.name).observe(len(result.text))
            if ctx.transcript is not None:
                try:
                    ctx.transcript.append(
                        {
                            "kind": "tool_result",
                            "tool": self.name,
                            "outcome": outcome,
                            "elapsed_s": round(elapsed, 4),
                            "bytes": len(result.text),
                            "input": raw_input,
                        }
                    )
                except Exception:
                    pass
            return result

    def _empty_output(self) -> Any:
        """Construct an empty instance of the output model for error returns."""
        # Most output models accept all-default values; if not, override per tool.
        try:
            return getattr(self, "output_model", BaseModel)().model_dump() if False else None
        except Exception:
            return None

    def _persist_overflow(self, result: ToolResult[O], ctx: ToolUseContext) -> ToolResult[O]:
        """Save oversized result to disk and replace with a preview pointer."""
        out_dir = ctx.workdir / "tool-results"
        out_dir.mkdir(parents=True, exist_ok=True)
        import hashlib

        digest = hashlib.sha1(result.text.encode("utf-8")).hexdigest()[:16]
        path = out_dir / f"{self.name}-{digest}.txt"
        path.write_text(result.text, encoding="utf-8")
        preview = result.text[: self.max_result_size_chars]
        result.text = (
            f"<persisted-output tool='{self.name}' bytes={len(result.text)} "
            f"path='{path}'>\n{preview}\n... (truncated, full output at {path})\n"
            f"</persisted-output>"
        )
        return result


# ── Registry ─────────────────────────────────────────────────────────────────


_REGISTRY: dict[str, BaseTool] = {}


def register_tool(tool: BaseTool | type[BaseTool]) -> BaseTool:
    inst = tool() if isinstance(tool, type) else tool
    if not inst.name:
        raise ValueError(f"Tool {inst.__class__.__name__} has no .name")
    _REGISTRY[inst.name] = inst
    return inst


def get_tool(name: str) -> BaseTool | None:
    return _REGISTRY.get(name)


def get_all_tools() -> list[BaseTool]:
    return list(_REGISTRY.values())


def clear_registry() -> None:
    """For tests only."""
    _REGISTRY.clear()
