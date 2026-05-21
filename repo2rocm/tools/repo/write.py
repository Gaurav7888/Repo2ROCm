"""Write — overwrite/create a file."""
from __future__ import annotations

import hashlib
import os
from typing import ClassVar

from pydantic import BaseModel

from repo2rocm.tools.base import BaseTool, ToolResult, ToolUseContext
from repo2rocm.tools.repo.pathing import RepoPathResolutionError, resolve_repo_path

_PLACEHOLDER_REPRO_MARKERS = (
    "test_data.json",
    "test_image.jpg",
    "test_image.png",
    "What is in this image?",
    "A test image",
)
_SYNTHETIC_REPRO_LOG_MARKERS = (
    "paper_experiment_formatted.log",
    "synthetic",
    "formatted log",
    "perplexity:",
    "accuracy:",
    "speedup:",
)


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

    def validate_semantic(self, parsed: WriteInput, ctx: ToolUseContext) -> str | None:
        if str(ctx.options.get("run_mode") or "").lower() == "reproduce":
            haystack = f"{parsed.file_path}\n{parsed.content}"
            if any(marker in haystack for marker in _PLACEHOLDER_REPRO_MARKERS):
                return (
                    "Reproduce-mode guard: refusing to write synthetic placeholder paper inputs "
                    "(for example `test_data.json` / `test_image.*`). Use authoritative repo/paper "
                    "artifacts, or stop with `PAPER_RUN_FAILED`."
                )
            lower_path = parsed.file_path.lower()
            if "paper_experiment" in lower_path and any(
                marker in haystack.lower() for marker in _SYNTHETIC_REPRO_LOG_MARKERS
            ):
                return (
                    "Reproduce-mode guard: refusing to write a synthetic verification log. "
                    "PaperVerify must read the real experiment output."
                )
        return None

    async def call(self, parsed: WriteInput, ctx: ToolUseContext) -> ToolResult[WriteOutput]:
        try:
            path = resolve_repo_path(ctx, parsed.file_path)
        except RepoPathResolutionError as exc:
            return ToolResult(
                data=WriteOutput(file_path=parsed.file_path, bytes_written=0),
                text=str(exc),
                is_error=True,
            )
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
