"""Config-file scanning, Python-version + install-mechanism detection."""
from __future__ import annotations

import os
import re
from pathlib import Path

from repo2rocm.recon.files import CONFIG_FILES, read_file


def read_config_files(repo_path: Path | str) -> dict[str, str]:
    """Return {filename: content[:4k]} for every config file present."""
    out: dict[str, str] = {}
    repo_path = Path(repo_path)
    for fname in CONFIG_FILES:
        content = read_file(repo_path / fname)
        if content is not None:
            out[fname] = content

    workflows = repo_path / ".github" / "workflows"
    if workflows.is_dir():
        try:
            entries = sorted(os.listdir(workflows))[:3]
        except OSError:
            entries = []
        for wf in entries:
            content = read_file(workflows / wf, max_chars=3000)
            if content:
                out[f".github/workflows/{wf}"] = content
    return out


def detect_python_version(config_contents: dict[str, str]) -> str:
    if ".python-version" in config_contents:
        return config_contents[".python-version"].strip().splitlines()[0].strip()

    pyproject = config_contents.get("pyproject.toml", "")
    m = re.search(r"requires-python\s*=\s*['\"]([^'\"]+)['\"]", pyproject)
    if m:
        return m.group(1).strip()
    m = re.search(r"python\s*=\s*['\"]([^'\"]+)['\"]", pyproject)
    if m:
        return m.group(1).strip()

    setup_py = config_contents.get("setup.py", "")
    m = re.search(r"python_requires\s*=\s*['\"]([^'\"]+)['\"]", setup_py)
    if m:
        return m.group(1).strip()
    return ""


def detect_install_mechanisms(config_contents: dict[str, str]) -> list[str]:
    out: list[str] = []
    if "poetry.lock" in config_contents or (
        "pyproject.toml" in config_contents
        and "poetry" in config_contents.get("pyproject.toml", "").lower()
    ):
        out.append("poetry install")
    if "setup.py" in config_contents:
        out.append("pip install -e .")
    if "setup.cfg" in config_contents and "pyproject.toml" in config_contents:
        out.append("pip install -e .")
    if "Pipfile" in config_contents:
        out.append("pipenv install")
    if "environment.yml" in config_contents or "environment.yaml" in config_contents:
        out.append("conda env create -f environment.yml")
    for fname in config_contents:
        if fname.startswith("requirements") and fname.endswith(".txt"):
            out.append(f"pip install -r {fname}")
    if not out:
        out.append("pipreqs + manual install")
    return out


def parse_requirements(content: str) -> list[tuple[str, str]]:
    """Return list of (package, version_spec). Skips comments and -options."""
    out: list[tuple[str, str]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        m = re.match(r"^([a-zA-Z0-9_\-.]+)\s*(.*)$", line)
        if m:
            out.append((m.group(1).strip(), m.group(2).strip()))
    return out
