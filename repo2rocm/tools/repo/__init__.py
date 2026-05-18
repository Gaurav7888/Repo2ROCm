"""Repo-local file tools. All file paths are resolved relative to ctx.workdir."""
from repo2rocm.tools.repo.read import Read
from repo2rocm.tools.repo.grep import Grep
from repo2rocm.tools.repo.glob import Glob
from repo2rocm.tools.repo.edit import Edit
from repo2rocm.tools.repo.write import Write
from repo2rocm.tools.repo.apply_diff import ApplyDiff
from repo2rocm.tools.base import register_tool


def register_repo_tools() -> None:
    """Register all repo tools (idempotent)."""
    for cls in (Read, Grep, Glob, Edit, Write, ApplyDiff):
        register_tool(cls)


__all__ = ["Read", "Grep", "Glob", "Edit", "Write", "ApplyDiff", "register_repo_tools"]
