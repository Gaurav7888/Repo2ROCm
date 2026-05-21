"""Grep tool — wraps ripgrep when available, falls back to Python re."""
from __future__ import annotations

import asyncio
import re
import shutil
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from repo2rocm.tools.base import BaseTool, ToolResult, ToolUseContext
from repo2rocm.tools.repo.pathing import (
    RepoPathResolutionError,
    display_repo_path,
    resolve_repo_path,
)


class GrepInput(BaseModel):
    pattern: str = Field(..., description="Regex pattern.")
    path: str = Field(".", description="Subdirectory or file under workdir.")
    glob: str | None = Field(None, description="Optional glob filter (e.g. '*.py').")
    case_insensitive: bool = False
    head_limit: int = Field(250, description="Max matches to return.")


class GrepMatch(BaseModel):
    file: str
    line: int
    text: str


class GrepOutput(BaseModel):
    matches: list[GrepMatch]
    truncated: bool


_EXCLUDED_DIRS = {".git", ".svn", ".hg", ".bzr", ".jj", ".sl", "__pycache__", "node_modules"}


class Grep(BaseTool[GrepInput, GrepOutput]):
    name: ClassVar[str] = "Grep"
    description: ClassVar[str] = (
        "Search files with regex. Excludes VCS/build dirs. Default limit 250 matches."
    )
    input_model: ClassVar[type[BaseModel]] = GrepInput
    max_result_size_chars: ClassVar[int] = 100_000
    interrupt_behavior: ClassVar[str] = "cancel"

    def is_concurrency_safe(self, parsed: GrepInput) -> bool:
        return True

    def is_read_only(self, parsed: GrepInput) -> bool:
        return True

    async def call(self, parsed: GrepInput, ctx: ToolUseContext) -> ToolResult[GrepOutput]:
        try:
            root = resolve_repo_path(ctx, parsed.path)
        except RepoPathResolutionError as exc:
            return ToolResult(
                data=GrepOutput(matches=[], truncated=False),
                text=str(exc),
                is_error=True,
            )
        if not root.exists():
            return ToolResult(
                data=GrepOutput(matches=[], truncated=False),
                text=f"Path not found: {parsed.path}",
                is_error=True,
            )

        # try ripgrep first
        if shutil.which("rg"):
            return await self._rg(parsed, root, ctx)
        return self._py_re(parsed, root, ctx)

    async def _rg(
        self, parsed: GrepInput, root: Path, ctx: ToolUseContext
    ) -> ToolResult[GrepOutput]:
        # -H/--with-filename forces ripgrep to always prefix matches with the
        # filename. Without it, rg omits the filename when given a single file
        # path, which breaks our `file:line:body` parser (we'd misread the
        # line number as a filename and try int() on the match body).
        args = [
            "rg", "-H", "-n", "--no-heading",
            "--max-count", str(parsed.head_limit),
        ]
        if parsed.case_insensitive:
            args.append("-i")
        if parsed.glob:
            args.extend(["-g", parsed.glob])
        args.append("--")
        args.append(parsed.pattern)
        args.append(str(root))
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        text = stdout.decode("utf-8", errors="replace")
        matches: list[GrepMatch] = []
        for line in text.splitlines():
            if matches and len(matches) >= parsed.head_limit:
                break
            try:
                file_part, line_no, body = line.split(":", 2)
            except ValueError:
                continue
            try:
                lineno_int = int(line_no)
            except ValueError:
                # Defense in depth: if rg ever emits an unexpected prefix
                # (e.g. a path containing ':' on Windows-style hosts), skip
                # the line instead of crashing the whole tool call.
                continue
            matches.append(
                GrepMatch(
                    file=display_repo_path(ctx, Path(file_part)),
                    line=lineno_int,
                    text=body[:500],
                )
            )
        truncated = len(matches) >= parsed.head_limit
        rendered = "\n".join(f"{m.file}:{m.line}:{m.text}" for m in matches) or "(no matches)"
        return ToolResult(data=GrepOutput(matches=matches, truncated=truncated), text=rendered)

    def _py_re(self, parsed: GrepInput, root: Path, ctx: ToolUseContext) -> ToolResult[GrepOutput]:
        flags = re.IGNORECASE if parsed.case_insensitive else 0
        try:
            regex = re.compile(parsed.pattern, flags)
        except re.error as exc:
            return ToolResult(
                data=GrepOutput(matches=[], truncated=False),
                text=f"invalid regex: {exc}",
                is_error=True,
            )
        matches: list[GrepMatch] = []
        for fp in self._walk(root, parsed.glob):
            if len(matches) >= parsed.head_limit:
                break
            try:
                with fp.open("r", encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f, start=1):
                        if regex.search(line):
                            matches.append(
                                GrepMatch(
                                    file=display_repo_path(ctx, fp),
                                    line=i,
                                    text=line.rstrip()[:500],
                                )
                            )
                            if len(matches) >= parsed.head_limit:
                                break
            except OSError:
                continue
        truncated = len(matches) >= parsed.head_limit
        text = "\n".join(f"{m.file}:{m.line}:{m.text}" for m in matches) or "(no matches)"
        return ToolResult(data=GrepOutput(matches=matches, truncated=truncated), text=text)

    def _walk(self, root: Path, glob: str | None):
        if root.is_file():
            yield root
            return
        for p in root.rglob("*"):
            if p.is_dir():
                if p.name in _EXCLUDED_DIRS:
                    continue
                continue
            if any(part in _EXCLUDED_DIRS for part in p.parts):
                continue
            if glob and not p.match(glob):
                continue
            yield p
