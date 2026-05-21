"""File-system helpers for recon."""
from __future__ import annotations

import os
from pathlib import Path

CONFIG_FILES: list[str] = [
    "requirements.txt", "requirements_dev.txt", "requirements-dev.txt",
    "requirements_test.txt", "requirements-test.txt",
    "setup.py", "setup.cfg", "pyproject.toml",
    "Pipfile", "Pipfile.lock",
    "environment.yml", "environment.yaml",
    "poetry.lock", "tox.ini", "Makefile",
    ".python-version",
]

README_NAMES: list[str] = ["README.md", "readme.md", "README.rst", "README.txt", "README"]


def read_file(path: Path | str, *, max_chars: int | None = 4000) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
        return text if max_chars is None else text[:max_chars]
    except Exception:
        return None


def find_python_files(repo_path: Path | str, *, limit: int | None = None) -> list[str]:
    out: list[str] = []
    skip = {"__pycache__", "node_modules", ".venv", "venv", ".git", "site-packages"}
    for root, dirs, files in os.walk(str(repo_path)):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in skip]
        for f in files:
            if f.endswith(".py"):
                out.append(os.path.join(root, f))
                if limit is not None and len(out) >= limit:
                    return out
    return out


def read_readme(repo_path: Path | str) -> tuple[str, str]:
    """Return (filename, content) for the first README found, or ("", "")."""
    repo_path = Path(repo_path)
    for name in README_NAMES:
        p = repo_path / name
        if p.is_file():
            text = read_file(p, max_chars=None) or ""
            return name, text
    return "", ""


def list_top_level(repo_path: Path | str, *, limit: int = 40) -> list[str]:
    try:
        return sorted(os.listdir(str(repo_path)))[:limit]
    except OSError:
        return []
