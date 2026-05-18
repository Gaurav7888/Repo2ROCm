"""File-based memory: Markdown + YAML frontmatter, per-project directory.

Two-tier:
  * `MEMORY.md` — always-loaded index (≤200 lines, ≤25KB)
  * Topic files (`{type}_{topic}.md`) — surfaced on demand by RecallSelector

Path resolution: gitroot → sanitized → `~/.repo2rocm/projects/<slug>/memory/`.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

try:
    import frontmatter
except ImportError:  # pragma: no cover
    frontmatter = None  # type: ignore[assignment]


MEMORY_INDEX = "MEMORY.md"

MemoryType = Literal["user", "feedback", "project", "reference"]


@dataclass
class MemoryFile:
    path: Path
    name: str
    description: str
    type: MemoryType
    body: str = ""
    mtime: float = 0.0

    def age_days(self) -> float:
        return max(0.0, (time.time() - self.mtime) / 86_400.0)


def sanitize_project_path(p: Path) -> str:
    s = str(p).replace(os.sep, "-").replace(":", "")
    # restrict character set
    s = re.sub(r"[^A-Za-z0-9._\-]", "", s)
    return s.lstrip("-")[:200] or "unknown"


@dataclass
class MemoryStore:
    base_dir: Path

    @classmethod
    def for_project(cls, project_root: Path, *, root: Path | None = None) -> MemoryStore:
        root = root or (Path.home() / ".repo2rocm")
        # find git root or fall back to project_root
        gitroot = _find_git_root(project_root) or project_root
        slug = sanitize_project_path(gitroot)
        d = root / "projects" / slug / "memory"
        d.mkdir(parents=True, exist_ok=True)
        return cls(d)

    def list_files(self) -> list[MemoryFile]:
        out: list[MemoryFile] = []
        if frontmatter is None:
            return out
        for p in self.base_dir.glob("*.md"):
            if p.name == MEMORY_INDEX:
                continue
            try:
                post = frontmatter.load(str(p))
                meta = post.metadata
                out.append(
                    MemoryFile(
                        path=p,
                        name=str(meta.get("name") or p.stem),
                        description=str(meta.get("description") or ""),
                        type=str(meta.get("type") or "reference"),  # type: ignore[arg-type]
                        body="",  # body loaded on demand
                        mtime=p.stat().st_mtime,
                    )
                )
            except Exception:
                continue
        return out

    def load_body(self, mf: MemoryFile) -> str:
        if frontmatter is None:
            return mf.path.read_text(encoding="utf-8", errors="replace")
        post = frontmatter.load(str(mf.path))
        return post.content

    def index_text(self) -> str:
        idx = self.base_dir / MEMORY_INDEX
        if idx.exists():
            return idx.read_text(encoding="utf-8", errors="replace")
        return ""

    def manifest_for_recall(self) -> str:
        """Lightweight summary the recall selector LLM sees."""
        files = self.list_files()
        lines = []
        for f in files:
            age = int(f.age_days())
            lines.append(
                f"- type={f.type} name={f.name!r} age_days={age} "
                f"file={f.path.name} :: {f.description[:160]}"
            )
        return "\n".join(lines) or "(no memories)"

    def write_topic(
        self,
        *,
        slug: str,
        type: MemoryType,
        name: str,
        description: str,
        body: str,
    ) -> Path:
        path = self.base_dir / f"{type}_{slug}.md"
        text = (
            f"---\n"
            f"name: {name}\n"
            f"description: {description}\n"
            f"type: {type}\n"
            f"---\n\n{body}\n"
        )
        path.write_text(text, encoding="utf-8")
        return path

    def update_index(self, entries: Iterable[tuple[str, str, str]]) -> None:
        """Rewrite MEMORY.md from (name, filename, description) tuples."""
        idx = self.base_dir / MEMORY_INDEX
        lines = ["# Memory Index", "", "Each line: `[name](file) -- description`", ""]
        for name, file, desc in entries:
            lines.append(f"- [{name}]({file}) -- {desc[:150]}")
        idx.write_text("\n".join(lines), encoding="utf-8")


def _find_git_root(start: Path) -> Path | None:
    cur = start.resolve()
    for _ in range(20):
        if (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            return None
        cur = cur.parent
    return None
