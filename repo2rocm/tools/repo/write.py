"""Write — overwrite/create a file."""
from __future__ import annotations

import hashlib
import os
from typing import ClassVar

from pydantic import BaseModel

from repo2rocm.tools.base import BaseTool, ToolResult, ToolUseContext


class WriteInput(BaseModel):
    file_path: str
    content: str
    create_parent_dirs: bool = True


class WriteOutput(BaseModel):
    file_path: str
    bytes_written: int


class Write(BaseTool[WriteInput, WriteOutput]):
    name: ClassVar[str] = "Write"
    description: ClassVar[str] = "Write/overwrite a file. Creates parent dirs by default."
    input_model: ClassVar[type[BaseModel]] = WriteInput
    max_result_size_chars: ClassVar[int] = 2_000

    def is_concurrency_safe(self, parsed: WriteInput) -> bool:
        return False

    def is_read_only(self, parsed: WriteInput) -> bool:
        return False

    async def call(self, parsed: WriteInput, ctx: ToolUseContext) -> ToolResult[WriteOutput]:
        path = (ctx.workdir / parsed.file_path).resolve()
        if parsed.create_parent_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(parsed.content, encoding="utf-8")
        ctx.read_file_state.record(
            path=path,
            mtime_ns=os.stat(path).st_mtime_ns,
            sha=hashlib.sha1(parsed.content.encode()).hexdigest(),
        )
        return ToolResult(
            data=WriteOutput(file_path=parsed.file_path, bytes_written=len(parsed.content)),
            text=f"Wrote {len(parsed.content)} bytes to {parsed.file_path}",
        )
