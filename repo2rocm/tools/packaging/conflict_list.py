"""ConflictList — surface and resolve version conflicts before bulk install."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import ClassVar

from pydantic import BaseModel

from repo2rocm.tools.base import BaseTool, ToolResult, ToolUseContext


@dataclass
class ConflictEntry:
    name: str
    tool: str  # pip | apt
    constraints: list[str]


@dataclass
class ConflictList:
    items: list[ConflictEntry] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, entry: ConflictEntry) -> None:
        with self._lock:
            self.items.append(entry)

    def solve(self, *, constraint: str | None = None, keep_original: bool = False) -> str:
        with self._lock:
            if not self.items:
                return "(conflict list is empty)"
            head = self.items.pop(0)
            if keep_original:
                return f"kept original constraint for {head.name}"
            if constraint is None:
                return f"set {head.name} to latest"
            return f"set {head.name} to {constraint}"

    def clear(self) -> None:
        with self._lock:
            self.items.clear()

    def show(self) -> str:
        with self._lock:
            if not self.items:
                return "(conflict list is empty)"
            return "\n".join(
                f"  - [{i.tool}] {i.name}: {', '.join(i.constraints)}" for i in self.items
            )


def _get_or_create_cl(ctx: ToolUseContext) -> ConflictList:
    cl = ctx.options.get("conflict_list")
    if cl is None:
        cl = ConflictList()
        ctx.options["conflict_list"] = cl
    return cl


class CShowInput(BaseModel):
    pass


class CShowOutput(BaseModel):
    items: list[str]


class ConflictListShow(BaseTool[CShowInput, CShowOutput]):
    name: ClassVar[str] = "ConflictListShow"
    description: ClassVar[str] = "Show pending dependency conflicts."
    input_model: ClassVar[type[BaseModel]] = CShowInput
    max_result_size_chars: ClassVar[int] = 4_000

    def is_concurrency_safe(self, parsed: CShowInput) -> bool:
        return True

    def is_read_only(self, parsed: CShowInput) -> bool:
        return True

    async def call(self, parsed: CShowInput, ctx: ToolUseContext) -> ToolResult[CShowOutput]:
        cl = _get_or_create_cl(ctx)
        return ToolResult(
            data=CShowOutput(items=[f"{i.name}:{','.join(i.constraints)}" for i in cl.items]),
            text=cl.show(),
        )


class CSolveInput(BaseModel):
    constraint: str | None = None
    keep_original: bool = False


class CSolveOutput(BaseModel):
    message: str


class ConflictListSolve(BaseTool[CSolveInput, CSolveOutput]):
    name: ClassVar[str] = "ConflictListSolve"
    description: ClassVar[str] = (
        "Resolve the first conflict. Pass `constraint` (e.g. '==2.0') or `keep_original=true`."
    )
    input_model: ClassVar[type[BaseModel]] = CSolveInput
    max_result_size_chars: ClassVar[int] = 1_000

    def is_concurrency_safe(self, parsed: CSolveInput) -> bool:
        return False

    def is_read_only(self, parsed: CSolveInput) -> bool:
        return False

    async def call(self, parsed: CSolveInput, ctx: ToolUseContext) -> ToolResult[CSolveOutput]:
        cl = _get_or_create_cl(ctx)
        msg = cl.solve(constraint=parsed.constraint, keep_original=parsed.keep_original)
        return ToolResult(data=CSolveOutput(message=msg), text=msg)


class CClearInput(BaseModel):
    pass


class CClearOutput(BaseModel):
    cleared: bool


class ConflictListClear(BaseTool[CClearInput, CClearOutput]):
    name: ClassVar[str] = "ConflictListClear"
    description: ClassVar[str] = "Clear the conflict list."
    input_model: ClassVar[type[BaseModel]] = CClearInput
    max_result_size_chars: ClassVar[int] = 500

    def is_concurrency_safe(self, parsed: CClearInput) -> bool:
        return False

    def is_read_only(self, parsed: CClearInput) -> bool:
        return False

    async def call(self, parsed: CClearInput, ctx: ToolUseContext) -> ToolResult[CClearOutput]:
        _get_or_create_cl(ctx).clear()
        return ToolResult(data=CClearOutput(cleared=True), text="conflict list cleared")
