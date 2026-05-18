"""Convert the sandbox's commit-DAG trunk into a reproducible Dockerfile."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from repo2rocm.sandbox import Sandbox
from repo2rocm.sandbox.manager import ExecResult


@dataclass
class DockerfileSynthesis:
    dockerfile_text: str
    successful_commands: list[str]
    base_image: str


_SKIP_COMMANDS = (
    "ls", "cat", "pwd", "echo", "head", "tail", "grep", "find", "wc",
    "stat", "file", "which", "whereis", "rocm-smi", "nvidia-smi",
)


def synthesize_dockerfile(sandbox: Sandbox) -> DockerfileSynthesis:
    """Build a Dockerfile from the sandbox's successful inner commands."""
    base = sandbox.cfg.base_image
    successful = [
        c for c in sandbox.commands
        if c.exit_code == 0 and not _is_skippable(c)
    ]
    lines = [
        f"FROM {base}",
        "",
        "ENV DEBIAN_FRONTEND=noninteractive",
        "WORKDIR /repo",
        "COPY . /repo",
        "",
    ]
    for c in successful:
        lines.append(f"RUN {c.command}")
    lines.append("")
    return DockerfileSynthesis(
        dockerfile_text="\n".join(lines),
        successful_commands=[c.command for c in successful],
        base_image=base,
    )


def _is_skippable(c: ExecResult) -> bool:
    tokens = c.command.strip().split()
    if not tokens:
        return True
    head = tokens[0]
    if head in _SKIP_COMMANDS:
        return True
    if c.command.strip().startswith("python -c"):
        return True
    return False


def write_dockerfile(synth: DockerfileSynthesis, target: Path) -> None:
    target.write_text(synth.dockerfile_text, encoding="utf-8")
