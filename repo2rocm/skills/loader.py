"""Skill discovery and loading."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

try:
    import frontmatter
except ImportError:  # pragma: no cover
    frontmatter = None  # type: ignore[assignment]


@dataclass
class SkillManifest:
    """Frontmatter-only summary; the body is loaded lazily."""

    name: str
    description: str
    when_to_use: str
    path: Path
    source: str  # "builtin" | "user" | "project" | "policy"
    allowed_tools: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)  # conditional activation globs
    disable_model_invocation: bool = False
    hooks: dict[str, list[dict]] = field(default_factory=dict)


@dataclass
class SkillCatalog:
    manifests: dict[str, SkillManifest] = field(default_factory=dict)

    def menu_text(self) -> str:
        """The compact text injected into system prompts at startup."""
        if not self.manifests:
            return "(no skills loaded)"
        lines = ["## Available Skills (invoke with /<name>)"]
        for m in sorted(self.manifests.values(), key=lambda x: x.name):
            lines.append(f"- /{m.name}: {m.description}")
        return "\n".join(lines)


_SOURCES = [
    ("policy", Path("/etc/repo2rocm/skills")),
    ("user", Path.home() / ".repo2rocm" / "skills"),
    ("project", Path(".") / ".repo2rocm" / "skills"),
]


def _builtin_dir() -> Path:
    return Path(__file__).parent / "builtin"


def discover_skills(extra_dirs: Iterable[Path] = ()) -> SkillCatalog:
    cat = SkillCatalog()
    for name, p in _SOURCES:
        for sk in _walk(p, source=name):
            cat.manifests.setdefault(sk.name, sk)
    for sk in _walk(_builtin_dir(), source="builtin"):
        cat.manifests.setdefault(sk.name, sk)
    for d in extra_dirs:
        for sk in _walk(d, source="extra"):
            cat.manifests.setdefault(sk.name, sk)
    return cat


def _walk(root: Path, *, source: str) -> Iterable[SkillManifest]:
    if not root.exists() or frontmatter is None:
        return
    for skill_md in root.rglob("SKILL.md"):
        try:
            post = frontmatter.load(str(skill_md))
            meta = post.metadata
            name = str(meta.get("name") or skill_md.parent.name)
            yield SkillManifest(
                name=name,
                description=str(meta.get("description") or ""),
                when_to_use=str(meta.get("when_to_use") or ""),
                path=skill_md,
                source=source,
                allowed_tools=list(meta.get("allowed_tools") or []),
                paths=list(meta.get("paths") or []),
                disable_model_invocation=bool(meta.get("disable_model_invocation", False)),
                hooks=dict(meta.get("hooks") or {}),
            )
        except Exception:
            continue


def load_skill_body(manifest: SkillManifest) -> str:
    """Load the full skill body (Phase 2)."""
    if frontmatter is None:
        return manifest.path.read_text(encoding="utf-8", errors="replace")
    post = frontmatter.load(str(manifest.path))
    return post.content
