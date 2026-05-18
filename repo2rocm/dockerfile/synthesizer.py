"""Synthesize a reproducible Dockerfile from a sandbox run.

Mirrors the original Repo2ROCm `integrate_dockerfile.py` shape:

    FROM <base_image>
    WORKDIR /
    # pre-install (apt, curl, git, poetry, pytest, pipdeptree, pipreqs)
    RUN apt-get update && apt-get install -y curl git
    RUN curl -sSL https://install.python-poetry.org | python3 -
    ENV PATH="/root/.local/bin:$PATH"
    RUN pip install pytest pipdeptree pipreqs

    # bring in the repo at the right SHA
    RUN git clone https://github.com/<owner>/<name>.git
    RUN mkdir /repo
    RUN git config --global --add safe.directory /repo
    RUN cp -r /<name>/. /repo && rm -rf /<name>/
    RUN cd /repo && git checkout <sha>

    # any code patches the agent applied
    COPY patches/patch_001.diff /tmp/patches/patch_001.diff
    RUN cd /repo && git apply --reject /tmp/patches/patch_001.diff --allow-empty

    # everything the agent successfully ran in the sandbox
    RUN <command 1>
    RUN cd <subdir> && <command 2>
    ...

Inputs:
  * sandbox.commands : list of ExecResult — every command, with exit_code + cwd
  * repo_host_path   : path on disk to the cloned repo (we'll `git diff` it to
                       extract patches for any host-side file edits)
  * base_image       : the ROCm base image the sandbox last ran on
  * repo_full_name   : "owner/name" — for the `git clone` line
  * sha              : the commit SHA we checked out
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from repo2rocm.sandbox import Sandbox
from repo2rocm.sandbox.manager import ExecResult

# Commands that don't belong in the Dockerfile.
_SKIP_PREFIXES = (
    "ls", "cat", "pwd", "echo", "head", "tail", "grep", "find", "wc",
    "stat", "file", "which", "whereis", "rocm-smi", "nvidia-smi",
    "history", "type", "env", "printenv", "true", "false",
)

# Hosting-tool commands the agent uses to probe — never bake into the image.
_INTERNAL_TOOL_PREFIXES = (
    "python /home/tools/", "/home/tools/",
)

_SAFE_INSPECTIONS = re.compile(
    r"^(python\s+-c\s+[\"'].*torch\.cuda\.is_available)|"
    r"^(rocm-smi)|"
    r"^(nvidia-smi)|"
    r"^(pipdeptree)|"
    r"^(pip\s+show)|"
    r"^(pip\s+list)|"
    r"^(pip\s+freeze)|"
    r"^(conda\s+list)|"
    r"^(which\b)|"
    r"^(whereis\b)|"
    r"^(env\s*$)|"
    r"^(printenv\b)",
)

# Multi-line heredoc probes (e.g. the EnvVerify script). These are diagnostic-only
# and should never be baked into the production image.
_HEREDOC_PROBE_RE = re.compile(
    r"<<\s*['\"]?(?:PY|EOF|PYEOF|END|SH)['\"]?",
    re.IGNORECASE,
)
_ENV_VERIFY_FINGERPRINT = re.compile(
    r"torch\.cuda\.is_available\(\)|ENV_VERIFY_JSON:",
)


@dataclass
class DockerfileSynthesis:
    dockerfile_text: str
    successful_commands: list[str]
    base_image: str
    patches: list[Path] = field(default_factory=list)  # Paths to .diff files written to disk


def _is_skippable(c: ExecResult) -> bool:
    """A command that shouldn't be replayed in the Dockerfile.

    Rules:
      * Read-only inspection commands (ls, cat, grep, find, pip list, …) → skip
      * Internal tool wrappers (/home/tools/*) → skip
      * Success/marker echoes (ROCM_ENV_VERIFIED, PAPER_RESULT_*) → skip
      * Multi-line heredoc probes — especially the EnvVerify script — never bake
      * Empty / sentinel placeholders → skip
    """
    cmd = c.command.strip()
    if not cmd:
        return True

    # Heredoc probes (the killer case from the live run): an EnvVerify
    # `python - <<'PY' ... PY` script is a probe, not a build step.
    if _HEREDOC_PROBE_RE.search(cmd) and _ENV_VERIFY_FINGERPRINT.search(cmd):
        return True
    # Generic single-shot probe heredocs that print a single env-check value —
    # if the heredoc body is short (<300 chars) and ends with a sys.exit(int),
    # treat it as a probe rather than a build step.
    if _HEREDOC_PROBE_RE.search(cmd):
        body_len = len(cmd)
        if body_len < 600 and re.search(r"sys\.exit\(\s*\d+", cmd):
            return True

    head = cmd.split()[0]
    if head in _SKIP_PREFIXES:
        return True
    if cmd.startswith(_INTERNAL_TOOL_PREFIXES):
        return True
    if _SAFE_INSPECTIONS.match(cmd):
        return True
    # the env-verify echo + the success marker
    if "ROCM_ENV_VERIFIED" in cmd or "PAPER_RESULT_" in cmd:
        return True
    if cmd in ("$pwd$", "$pip list --format json$"):
        return True
    # bare `set -e` on its own line — fragment, not a useful RUN
    if cmd in ("set -e", "set -ex", "set -eux"):
        return True
    return False


def _is_rocm_base_image(image: str) -> bool:
    s = (image or "").lower().strip()
    return s.startswith("rocm/") or "rocm" in s.split(":", 1)[0]


def _pre_install_block(base_image: str) -> list[str]:
    """Mirror integrate_dockerfile._get_pre_download."""
    if _is_rocm_base_image(base_image):
        return [
            "RUN apt-get update && apt-get install -y curl git",
            "RUN curl -sSL https://install.python-poetry.org | python3 -",
            'ENV PATH="/root/.local/bin:$PATH"',
            "RUN pip install pytest pytest-xdist pipdeptree pipreqs",
        ]
    return [
        "RUN apt-get update && apt-get install -y curl git",
        "RUN curl -sSL https://install.python-poetry.org | python3 -",
        'ENV PATH="/root/.local/bin:$PATH"',
        "RUN pip install pytest pytest-xdist pipdeptree",
    ]


def _run_line(c: ExecResult) -> str:
    """Render a single ExecResult as a Dockerfile RUN line, with cwd if needed."""
    cmd = c.command.strip()
    cwd = (c.cwd or "/repo").rstrip("/")
    if cwd in ("", "/", "/repo"):
        return f"RUN {cmd}"
    return f"RUN cd {cwd} && {cmd}"


def _normalize_run(line: str) -> str:
    """Loose-normalize a `RUN ...` line for duplicate detection."""
    s = line.strip()
    if s.startswith("RUN "):
        s = s[4:].strip()
    s = re.sub(r"^cd\s+\S+\s*&&\s*", "", s, count=1)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*\|\s*tee\s+\S+\s*$", "", s)
    s = re.sub(r"\s*2>&1\s*$", "", s)
    return s.strip()


def _dedupe_runs(lines: list[str]) -> list[str]:
    """Keep the LAST occurrence of each normalized RUN line."""
    if not lines:
        return lines
    seen: dict[str, int] = {}
    for i, ln in enumerate(lines):
        seen[_normalize_run(ln)] = i
    return [ln for i, ln in enumerate(lines) if seen[_normalize_run(ln)] == i]


def _extract_repo_patches(
    repo_host_path: Path, patches_dir: Path
) -> list[Path]:
    """Run `git diff` in the cloned repo to capture all host-side edits.

    Returns a list of .diff files written into `patches_dir`. Always returns
    at most one combined patch for simplicity (Repo2ROCm's original wrote one
    per agent edit; we collapse into one — easier to read and apply).
    """
    if not (repo_host_path / ".git").exists():
        return []
    try:
        out = subprocess.run(
            ["git", "diff", "--no-color"],
            cwd=str(repo_host_path),
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except Exception:
        return []
    diff_text = (out.stdout or "").strip()
    if not diff_text:
        return []
    patches_dir.mkdir(parents=True, exist_ok=True)
    patch_path = patches_dir / "agent_edits.diff"
    patch_path.write_text(diff_text + "\n", encoding="utf-8")
    return [patch_path]


def synthesize_dockerfile(
    sandbox: Sandbox,
    *,
    repo_full_name: str = "",
    sha: str = "",
    repo_host_path: Path | None = None,
    patches_dir: Path | None = None,
) -> DockerfileSynthesis:
    """Produce a reproducible Dockerfile that recreates the sandbox's final state.

    Args:
        sandbox: the live or stopped Sandbox whose `.commands` list will be replayed.
        repo_full_name: 'owner/name'. If given, emits `git clone` + `git checkout sha`
                        so the Dockerfile is fully reproducible from a fresh clone.
        sha: commit SHA to check out after clone. Required if `repo_full_name` is set.
        repo_host_path: host-side path to the cloned repo. If set, we'll `git diff`
                        there to capture any agent-applied edits as a .diff file.
        patches_dir: where to write extracted .diff files (usually <output>/patches/).
    """
    base_image = sandbox.cfg.base_image
    lines: list[str] = [f"FROM {base_image}", "", "WORKDIR /", ""]
    lines.extend(_pre_install_block(base_image))
    lines.append("")

    # Bring in the repo
    if repo_full_name and "/" in repo_full_name:
        owner, name = repo_full_name.split("/", 1)
        lines.extend([
            f"RUN git clone https://github.com/{owner}/{name}.git",
            "RUN mkdir -p /repo",
            "RUN git config --global --add safe.directory /repo",
            f"RUN cp -r /{name}/. /repo && rm -rf /{name}",
        ])
        if sha:
            lines.append(f"RUN cd /repo && git checkout {sha}")
        lines.append("")

    # Capture any code edits the agent applied to the host clone
    patches: list[Path] = []
    if repo_host_path and patches_dir:
        patches = _extract_repo_patches(repo_host_path, patches_dir)
    if patches:
        lines.append("# Code patches the agent applied (from `git diff` of the host clone)")
        for p in patches:
            lines.append(f"COPY {p.name} /tmp/{p.name}")
            lines.append(f"RUN cd /repo && git apply --reject /tmp/{p.name} --allow-empty || true")
        lines.append("")

    # Replay every successful command
    successful = [c for c in sandbox.commands if c.exit_code == 0 and not _is_skippable(c)]
    run_lines = [_run_line(c) for c in successful]
    run_lines = _dedupe_runs(run_lines)
    if run_lines:
        lines.append("# Commands the agent ran inside the sandbox (deduplicated)")
        lines.extend(run_lines)
        lines.append("")

    # Default working directory for `docker run`
    lines.append("WORKDIR /repo")
    lines.append('CMD ["/bin/bash"]')

    return DockerfileSynthesis(
        dockerfile_text="\n".join(lines).rstrip() + "\n",
        successful_commands=[c.command for c in successful],
        base_image=base_image,
        patches=patches,
    )


def write_dockerfile(synth: DockerfileSynthesis, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(synth.dockerfile_text, encoding="utf-8")
    # also copy patches next to the Dockerfile for `docker build .` to find them
    for p in synth.patches:
        dst = target.parent / p.name
        if str(p.resolve()) != str(dst.resolve()):
            shutil.copy2(p, dst)
