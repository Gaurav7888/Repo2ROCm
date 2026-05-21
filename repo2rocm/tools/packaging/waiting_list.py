"""WaitingList — collect dependencies before bulk install. Replaces the regex parser version.

We keep the same semantics from the old Repo2ROCm: add packages, addfile from a
requirements-style file, show, clear. Conflict detection lives in `ConflictList`.

State lives on `ctx.options["waiting_list"]` so multiple migrator workers share one queue.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from repo2rocm.tools.base import BaseTool, ToolResult, ToolUseContext
from repo2rocm.tools.repo.pathing import RepoPathResolutionError, resolve_repo_path


@dataclass
class PackageSpec:
    name: str
    version_constraint: str = ""
    tool: str = "pip"  # pip | apt

    def normalized(self) -> str:
        return f"{self.name}{self.version_constraint}".strip()


@dataclass
class WaitingList:
    items: list[PackageSpec] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, spec: PackageSpec) -> tuple[bool, str]:
        with self._lock:
            for existing in self.items:
                if existing.name == spec.name and existing.tool == spec.tool:
                    if existing.version_constraint == spec.version_constraint:
                        return False, f"already in waiting list: {spec.normalized()}"
                    return False, (
                        f"CONFLICT: {spec.name} has constraint {existing.version_constraint!r} "
                        f"vs new {spec.version_constraint!r}"
                    )
            self.items.append(spec)
            return True, f"added {spec.normalized()}"

    def add_from_file(self, lines: list[str]) -> list[str]:
        msgs: list[str] = []
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^([A-Za-z0-9_.\-]+)([<>=!~][^;]*)?", line)
            if not m:
                msgs.append(f"could not parse: {raw}")
                continue
            name = m.group(1)
            constraint = (m.group(2) or "").strip()
            ok, msg = self.add(PackageSpec(name=name, version_constraint=constraint))
            msgs.append(msg)
        return msgs

    def clear(self) -> None:
        with self._lock:
            self.items.clear()

    def show(self) -> str:
        with self._lock:
            if not self.items:
                return "(waiting list is empty)"
            return "\n".join(f"  - [{i.tool}] {i.normalized()}" for i in self.items)


def _get_or_create_wl(ctx: ToolUseContext) -> WaitingList:
    wl = ctx.options.get("waiting_list")
    if wl is None:
        wl = WaitingList()
        ctx.options["waiting_list"] = wl
    return wl


# ── Tools ────────────────────────────────────────────────────────────────────


class WLAddInput(BaseModel):
    name: str = Field(..., description="Package name.")
    version_constraint: str = Field(
        "", description='Version spec like ">=1.0", "==2.0". No spaces.'
    )
    tool: str = Field("pip", description="pip | apt")


class WLAddOutput(BaseModel):
    added: bool
    message: str


class WaitingListAdd(BaseTool[WLAddInput, WLAddOutput]):
    name: ClassVar[str] = "WaitingListAdd"
    description: ClassVar[str] = "Queue a package for batched install."
    input_model: ClassVar[type[BaseModel]] = WLAddInput
    max_result_size_chars: ClassVar[int] = 1_000

    def is_concurrency_safe(self, parsed: WLAddInput) -> bool:
        return False  # writes the shared waiting list

    def is_read_only(self, parsed: WLAddInput) -> bool:
        return False

    async def call(self, parsed: WLAddInput, ctx: ToolUseContext) -> ToolResult[WLAddOutput]:
        wl = _get_or_create_wl(ctx)
        added, msg = wl.add(PackageSpec(parsed.name, parsed.version_constraint, parsed.tool))
        return ToolResult(data=WLAddOutput(added=added, message=msg), text=msg)


class WLFileInput(BaseModel):
    file_path: str


class WLFileOutput(BaseModel):
    added_count: int
    messages: list[str]


class WaitingListAddFile(BaseTool[WLFileInput, WLFileOutput]):
    name: ClassVar[str] = "WaitingListAddFile"
    description: ClassVar[str] = "Queue all requirements from a requirements.txt-style file."
    input_model: ClassVar[type[BaseModel]] = WLFileInput
    max_result_size_chars: ClassVar[int] = 8_000

    def is_concurrency_safe(self, parsed: WLFileInput) -> bool:
        return False

    def is_read_only(self, parsed: WLFileInput) -> bool:
        return False

    async def call(self, parsed: WLFileInput, ctx: ToolUseContext) -> ToolResult[WLFileOutput]:
        try:
            path = resolve_repo_path(ctx, parsed.file_path)
        except RepoPathResolutionError as exc:
            return ToolResult(
                data=WLFileOutput(added_count=0, messages=[]),
                text=str(exc),
                is_error=True,
            )
        if not path.is_file():
            return ToolResult(
                data=WLFileOutput(added_count=0, messages=[]),
                text=f"file not found: {parsed.file_path}",
                is_error=True,
            )
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        wl = _get_or_create_wl(ctx)
        msgs = wl.add_from_file(lines)
        added = sum(1 for m in msgs if m.startswith("added"))
        return ToolResult(
            data=WLFileOutput(added_count=added, messages=msgs),
            text="\n".join(msgs),
        )


class WLShowInput(BaseModel):
    pass


class WLShowOutput(BaseModel):
    items: list[str]


class WaitingListShow(BaseTool[WLShowInput, WLShowOutput]):
    name: ClassVar[str] = "WaitingListShow"
    description: ClassVar[str] = "Show the current waiting list."
    input_model: ClassVar[type[BaseModel]] = WLShowInput
    max_result_size_chars: ClassVar[int] = 4_000

    def is_concurrency_safe(self, parsed: WLShowInput) -> bool:
        return True

    def is_read_only(self, parsed: WLShowInput) -> bool:
        return True

    async def call(self, parsed: WLShowInput, ctx: ToolUseContext) -> ToolResult[WLShowOutput]:
        wl = _get_or_create_wl(ctx)
        text = wl.show()
        items = [i.normalized() for i in wl.items]
        return ToolResult(data=WLShowOutput(items=items), text=text)


class WLClearInput(BaseModel):
    pass


class WLClearOutput(BaseModel):
    cleared: bool


class WaitingListClear(BaseTool[WLClearInput, WLClearOutput]):
    name: ClassVar[str] = "WaitingListClear"
    description: ClassVar[str] = "Clear the waiting list."
    input_model: ClassVar[type[BaseModel]] = WLClearInput
    max_result_size_chars: ClassVar[int] = 500

    def is_concurrency_safe(self, parsed: WLClearInput) -> bool:
        return False

    def is_read_only(self, parsed: WLClearInput) -> bool:
        return False

    async def call(self, parsed: WLClearInput, ctx: ToolUseContext) -> ToolResult[WLClearOutput]:
        _get_or_create_wl(ctx).clear()
        return ToolResult(data=WLClearOutput(cleared=True), text="waiting list cleared")
