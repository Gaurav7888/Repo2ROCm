"""Read a file from the working directory. Always concurrency-safe."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from repo2rocm.tools.base import BaseTool, ToolResult, ToolUseContext


class ReadInput(BaseModel):
    file_path: str = Field(..., description="Path relative to the working directory.")
    offset: int = Field(0, description="Start at this line (0-indexed).")
    limit: int | None = Field(None, description="Read at most this many lines.")


class ReadOutput(BaseModel):
    file_path: str
    total_lines: int
    returned_lines: int
    content: str
    sha: str


class Read(BaseTool[ReadInput, ReadOutput]):
    name: ClassVar[str] = "Read"
    description: ClassVar[str] = (
        "Read a file. Always concurrency-safe. Returns numbered lines so the model can cite them."
    )
    input_model: ClassVar[type[BaseModel]] = ReadInput
    max_result_size_chars: ClassVar[int] = 200_000  # FileReadTool is generous; bounded by lines
    interrupt_behavior: ClassVar[str] = "cancel"

    def is_concurrency_safe(self, parsed: ReadInput) -> bool:
        return True

    def is_read_only(self, parsed: ReadInput) -> bool:
        return True

    async def call(self, parsed: ReadInput, ctx: ToolUseContext) -> ToolResult[ReadOutput]:
        path = (ctx.workdir / parsed.file_path).resolve()
        if not path.is_file():
            return ToolResult(
                data=ReadOutput(
                    file_path=str(parsed.file_path),
                    total_lines=0,
                    returned_lines=0,
                    content="",
                    sha="",
                ),
                text=f"File not found: {parsed.file_path}",
                is_error=True,
            )
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return ToolResult(
                data=ReadOutput(
                    file_path=str(parsed.file_path),
                    total_lines=0,
                    returned_lines=0,
                    content="",
                    sha="",
                ),
                text=f"Failed to read {parsed.file_path}: {exc}",
                is_error=True,
            )

        lines = raw.splitlines()
        start = max(0, parsed.offset)
        end = len(lines) if parsed.limit is None else min(len(lines), start + parsed.limit)
        slice_ = lines[start:end]
        numbered = "\n".join(f"{i + 1:6d}|{line}" for i, line in enumerate(slice_, start=start))

        sha = hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()
        ctx.read_file_state.record(
            path=path,
            mtime_ns=os.stat(path).st_mtime_ns,
            sha=sha,
        )

        out = ReadOutput(
            file_path=str(parsed.file_path),
            total_lines=len(lines),
            returned_lines=len(slice_),
            content=numbered,
            sha=sha,
        )
        return ToolResult(data=out, text=numbered)
