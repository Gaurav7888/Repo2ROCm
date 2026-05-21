"""Shared path resolution helpers for repo-oriented tools.

Repo/code tools should operate on the mounted repository, not on the
session/output workdir. In sandboxed runs the agent sees the repo as `/repo`,
while the host path lives under `ctx.options["repo_path"]`.
"""
from __future__ import annotations

from pathlib import Path

from repo2rocm.tools.base import ToolUseContext


class RepoPathResolutionError(ValueError):
    """Raised when a user-supplied path cannot be mapped into the repo."""


def repo_root(ctx: ToolUseContext) -> Path:
    raw = ctx.options.get("repo_path")
    if raw:
        return Path(str(raw)).resolve()
    return Path(ctx.workdir).resolve()


def repo_container_root(ctx: ToolUseContext) -> Path:
    raw = ctx.options.get("repo_container_path") or "/repo"
    return Path(str(raw))


def resolve_repo_path(ctx: ToolUseContext, user_path: str) -> Path:
    """Resolve a repo path safely against the mounted repository.

    Accepted forms:
      * relative repo paths, e.g. `requirements.txt`
      * container paths, e.g. `/repo/eval_ppl.py`
      * host-absolute paths under the repo root
    """
    root = repo_root(ctx)
    container_root = repo_container_root(ctx)
    raw = Path(user_path)

    if raw.is_absolute():
        try:
            rel = raw.relative_to(container_root)
        except ValueError:
            candidate = raw.resolve()
        else:
            candidate = (root / rel).resolve()
    else:
        candidate = (root / raw).resolve()

    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise RepoPathResolutionError(
            f"path escapes repo root: {user_path!r} (repo_root={root})"
        ) from exc
    return candidate


def normalize_glob_pattern(ctx: ToolUseContext, pattern: str) -> str:
    """Strip `/repo/` or host-root prefixes from glob patterns."""
    root = repo_root(ctx)
    container_root = repo_container_root(ctx)
    p = pattern.strip()
    if not p:
        return p

    container_prefix = str(container_root).rstrip("/") + "/"
    if p == str(container_root):
        return "*"
    if p.startswith(container_prefix):
        return p[len(container_prefix) :]

    host_prefix = str(root).rstrip("/") + "/"
    if p == str(root):
        return "*"
    if p.startswith(host_prefix):
        return p[len(host_prefix) :]
    return p


def display_repo_path(ctx: ToolUseContext, resolved_path: Path) -> str:
    """Render a host path back into the container-visible `/repo/...` form."""
    root = repo_root(ctx)
    container_root = repo_container_root(ctx)
    try:
        rel = resolved_path.resolve().relative_to(root)
    except ValueError:
        return str(resolved_path)
    return str(container_root / rel)
