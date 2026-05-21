"""Edit — replace old_string with new_string in a single file, with staleness check."""
from __future__ import annotations

import hashlib
import os
from typing import ClassVar

from pydantic import BaseModel, Field

from repo2rocm.tools.base import BaseTool, ToolResult, ToolUseContext
from repo2rocm.tools.repo.pathing import RepoPathResolutionError, resolve_repo_path


class EditInput(BaseModel):
    file_path: str
    old_string: str = Field(..., description="Exact text to replace. Must be unique unless replace_all=True.")
    new_string: str
    replace_all: bool = False


class EditOutput(BaseModel):
    file_path: str
    replacements: int


class Edit(BaseTool[EditInput, EditOutput]):
    name: ClassVar[str] = "Edit"
    description: ClassVar[str] = (
        "Search-and-replace edit. Rejects no-op edits and stale files (modified since last read)."
    )
    input_model: ClassVar[type[BaseModel]] = EditInput
    max_result_size_chars: ClassVar[int] = 4_000
    interrupt_behavior: ClassVar[str] = "block"

    def is_concurrency_safe(self, parsed: EditInput) -> bool:
        return False  # writes; serialize

    def is_read_only(self, parsed: EditInput) -> bool:
        return False

    def validate_semantic(self, parsed: EditInput, ctx: ToolUseContext) -> str | None:
        if parsed.old_string == parsed.new_string:
            return "old_string == new_string (no-op edit rejected)"
        if str(ctx.options.get("run_mode") or "").lower() == "reproduce":
            low_path = parsed.file_path.lower()
            if "paper_experiment" in low_path and low_path.endswith(".log"):
                return (
                    "Reproduce-mode guard: refusing to edit a paper experiment log. "
                    "PaperVerify must read the real experiment output."
                )
        return None

    async def call(self, parsed: EditInput, ctx: ToolUseContext) -> ToolResult[EditOutput]:
        try:
            path = resolve_repo_path(ctx, parsed.file_path)
        except RepoPathResolutionError as exc:
            return ToolResult(
                data=EditOutput(file_path=parsed.file_path, replacements=0),
                text=str(exc),
                is_error=True,
            )
        if not path.is_file():
            return ToolResult(
                data=EditOutput(file_path=parsed.file_path, replacements=0),
                text=f"File not found: {parsed.file_path}",
                is_error=True,
            )
        # Staleness check
        cached = ctx.read_file_state.get(path)
        try:
            current = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return ToolResult(
                data=EditOutput(file_path=parsed.file_path, replacements=0),
                text=f"read failed: {exc}",
                is_error=True,
            )
        sha = hashlib.sha1(current.encode()).hexdigest()
        if cached is not None and cached.sha != sha:
            return ToolResult(
                data=EditOutput(file_path=parsed.file_path, replacements=0),
                text=(
                    f"{parsed.file_path} has been modified since last read. Re-read it first."
                ),
                is_error=True,
            )

        if parsed.old_string not in current:
            return ToolResult(
                data=EditOutput(file_path=parsed.file_path, replacements=0),
                text=(
                    f"old_string not found in {parsed.file_path}. "
                    "Re-read the file and include exact (whitespace-sensitive) context."
                ),
                is_error=True,
            )
        if not parsed.replace_all and current.count(parsed.old_string) > 1:
            return ToolResult(
                data=EditOutput(file_path=parsed.file_path, replacements=0),
                text=(
                    f"old_string is not unique in {parsed.file_path} "
                    f"({current.count(parsed.old_string)} occurrences). "
                    "Provide more context or set replace_all=true."
                ),
                is_error=True,
            )

        if parsed.replace_all:
            new_content = current.replace(parsed.old_string, parsed.new_string)
            replacements = current.count(parsed.old_string)
        else:
            new_content = current.replace(parsed.old_string, parsed.new_string, 1)
            replacements = 1

        path.write_text(new_content, encoding="utf-8")
        # update file-state cache so subsequent edits don't trip the staleness check
        ctx.read_file_state.record(
            path=path,
            mtime_ns=os.stat(path).st_mtime_ns,
            sha=hashlib.sha1(new_content.encode()).hexdigest(),
        )

        return ToolResult(
            data=EditOutput(file_path=parsed.file_path, replacements=replacements),
            text=f"Edited {parsed.file_path}: {replacements} replacement(s).",
        )
