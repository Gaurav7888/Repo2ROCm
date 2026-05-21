"""Glob — recursive file listing with glob pattern."""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from repo2rocm.tools.base import BaseTool, ToolResult, ToolUseContext
from repo2rocm.tools.repo.pathing import normalize_glob_pattern, repo_root


class GlobInput(BaseModel):
    pattern: str = Field(..., description="Glob like 'src/**/*.py'.")
    head_limit: int = Field(500, description="Max paths to return.")


class GlobOutput(BaseModel):
    paths: list[str]
    truncated: bool


class Glob(BaseTool[GlobInput, GlobOutput]):
    name: ClassVar[str] = "Glob"
    description: ClassVar[str] = "List files matching a glob (recursive)."
    input_model: ClassVar[type[BaseModel]] = GlobInput
    max_result_size_chars: ClassVar[int] = 64_000
    interrupt_behavior: ClassVar[str] = "cancel"

    def is_concurrency_safe(self, parsed: GlobInput) -> bool:
        return True

    def is_read_only(self, parsed: GlobInput) -> bool:
        return True

    async def call(self, parsed: GlobInput, ctx: ToolUseContext) -> ToolResult[GlobOutput]:
        root = repo_root(ctx)
        pattern = normalize_glob_pattern(ctx, parsed.pattern)
        try:
            paths = [
                str(p.relative_to(root))
                for p in sorted(root.glob(pattern))
                if p.is_file()
            ]
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                data=GlobOutput(paths=[], truncated=False),
                text=f"glob failed: {exc}",
                is_error=True,
            )
        truncated = len(paths) > parsed.head_limit
        paths = paths[: parsed.head_limit]
        text = "\n".join(paths) if paths else "(no matches)"
        return ToolResult(data=GlobOutput(paths=paths, truncated=truncated), text=text)
