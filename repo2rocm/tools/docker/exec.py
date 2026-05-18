"""DockerExec — run a shell command inside the sandbox container.

Per-input safety: read-only commands (ls, cat, grep, …) are concurrency-safe.
Compound commands are safe only if every non-neutral subcommand is read-only.
"""
from __future__ import annotations

import re
from typing import ClassVar

from pydantic import BaseModel, Field

from repo2rocm.tools.base import BaseTool, ToolResult, ToolUseContext


class DockerExecInput(BaseModel):
    command: str = Field(..., description="Shell command to execute via `bash -lc`.")
    cwd: str | None = Field(None, description="Working dir inside container; default /repo.")
    timeout_s: float = Field(1800.0, description="Hard timeout in seconds.")
    env: dict[str, str] | None = Field(None, description="Extra env vars.")


class DockerExecOutput(BaseModel):
    exit_code: int
    stdout: str
    stderr: str
    elapsed_s: float
    truncated: bool


# ── concurrency-safety classification ────────────────────────────────────────

_SEARCH_COMMANDS = {"grep", "rg", "find", "fd", "ag", "ack", "locate", "whereis", "which"}
_READ_COMMANDS = {
    "cat", "head", "tail", "wc", "jq", "less", "more", "file", "stat",
    "md5sum", "sha256sum", "diff", "cmp",
}
_LIST_COMMANDS = {"ls", "tree", "du", "df", "lsblk", "blkid"}
_NEUTRAL_COMMANDS = {"echo", "printf", "true", "false", ":"}

_SAFE_COMMANDS = _SEARCH_COMMANDS | _READ_COMMANDS | _LIST_COMMANDS

_SPLIT_RE = re.compile(r"(?:&&|\|\||;|\|)")


def _classify_command_str(cmd: str) -> bool:
    """Return True if `cmd` is read-only and concurrency-safe."""
    if not cmd or not cmd.strip():
        return False
    # reject if contains output redirection
    if re.search(r"(?<!2)>(?!\s*&\s*\d)", cmd):
        return False
    if re.search(r"\b(rm|mv|cp|mkdir|rmdir|chmod|chown|touch|tar|gzip|gunzip|zip|unzip|"
                 r"dd|truncate|sed\s+-i|apt|apt-get|yum|dnf|pip|pip3|poetry|conda|"
                 r"git\s+(commit|push|pull|clone|checkout|reset|reflog|fetch|merge|rebase)|"
                 r"docker|kubectl|systemctl|service|kill|killall|reboot|halt|shutdown|"
                 r"make|cmake|ninja|cargo|go|npm|yarn|pnpm|node|deno|"
                 r"python\s*-c|python3\s*-c|bash\s*-c|sh\s*-c)\b", cmd):
        return False
    # split by &&, ||, ;, |
    parts = _SPLIT_RE.split(cmd)
    for part in parts:
        tokens = part.strip().split()
        if not tokens:
            continue
        head = tokens[0].split("/")[-1]  # strip path
        # allow `cd` and `pwd` as neutral
        if head in ("cd", "pwd"):
            continue
        if head in _NEUTRAL_COMMANDS:
            continue
        if head in _SAFE_COMMANDS:
            continue
        # `git status`, `git log`, `git diff`, `git show` are read-only
        if head == "git":
            if len(tokens) >= 2 and tokens[1] in {"status", "log", "diff", "show", "branch", "remote", "config", "rev-parse"}:
                continue
            return False
        return False
    return True


class DockerExec(BaseTool[DockerExecInput, DockerExecOutput]):
    name: ClassVar[str] = "DockerExec"
    description: ClassVar[str] = (
        "Execute a shell command inside the ROCm sandbox container. Compound commands "
        "(&&, ||, ;, |) supported. Read-only commands run concurrently."
    )
    input_model: ClassVar[type[BaseModel]] = DockerExecInput
    max_result_size_chars: ClassVar[int] = 60_000
    interrupt_behavior: ClassVar[str] = "block"

    def is_concurrency_safe(self, parsed: DockerExecInput) -> bool:
        return _classify_command_str(parsed.command)

    def is_read_only(self, parsed: DockerExecInput) -> bool:
        return _classify_command_str(parsed.command)

    async def call(
        self, parsed: DockerExecInput, ctx: ToolUseContext
    ) -> ToolResult[DockerExecOutput]:
        if ctx.sandbox is None:
            return ToolResult(
                data=DockerExecOutput(
                    exit_code=-1, stdout="", stderr="no sandbox attached",
                    elapsed_s=0.0, truncated=False,
                ),
                text="No sandbox attached to context.",
                is_error=True,
            )
        result = await ctx.sandbox.exec(
            parsed.command,
            timeout_s=parsed.timeout_s,
            cwd=parsed.cwd,
            env=parsed.env,
        )
        body = f"$ {parsed.command}\n[exit {result.exit_code} in {result.elapsed_s:.2f}s]\n"
        if result.stdout:
            body += result.stdout
        if result.stderr:
            body += f"\n--- stderr ---\n{result.stderr}"
        return ToolResult(
            data=DockerExecOutput(
                exit_code=result.exit_code,
                stdout=result.stdout,
                stderr=result.stderr,
                elapsed_s=result.elapsed_s,
                truncated=result.truncated,
            ),
            text=body,
            is_error=result.exit_code != 0,
        )
