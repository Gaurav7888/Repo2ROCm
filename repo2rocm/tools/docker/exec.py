"""DockerExec — run a shell command inside the sandbox container.

Per-input safety: read-only commands (ls, cat, grep, …) are concurrency-safe.
Compound commands are safe only if every non-neutral subcommand is read-only.
"""
from __future__ import annotations

import re
from typing import ClassVar

from pydantic import BaseModel, Field

from repo2rocm.paper.types import CommandSpec, PaperContext
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
_REPRO_PLACEHOLDER_PATTERNS = (
    r"test_data\.json",
    r"test_image\.(?:jpg|jpeg|png)",
    r"What is in this image\?",
    r"A test image",
    r"Image\.fromarray\(np\.random",
)

_EXEC_LAUNCHERS = ("accelerate launch", "torchrun", "python", "python3", "bash", "sh")


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


def _coerce_paper_context(obj) -> PaperContext | None:
    if obj is None:
        return None
    if isinstance(obj, PaperContext):
        return obj
    try:
        return PaperContext.model_validate(obj)
    except Exception:
        return None


def _norm_script(script: str) -> str:
    return script.strip().lstrip("./")


def _norm_value(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().strip("\"'"))


def _looks_like_experiment_run(cmd: str, chosen_script: str) -> bool:
    low = cmd.lower()
    if any(low.startswith(prefix) for prefix in ("cat ", "rg ", "grep ", "ls ", "head ", "tail ", "sed ", "wc ")):
        return False
    return (
        "paper_experiment.log" in low
        or _norm_script(chosen_script).lower() in low
    ) and any(tok in low for tok in _EXEC_LAUNCHERS)


def _reproduce_command_guard(cmd: str, paper_ctx_obj) -> str | None:
    paper_ctx = _coerce_paper_context(paper_ctx_obj)
    chosen = paper_ctx.chosen() if paper_ctx is not None else None
    if chosen is None:
        return None
    # The new pipeline emits a fully-bound `suggested_command`; parse it on the
    # fly here so we can drift-check whatever the reproducer actually runs.
    expected = CommandSpec.from_command(chosen.suggested_command)
    if expected is None or not expected.script:
        return None
    if not _looks_like_experiment_run(cmd, expected.script):
        return None
    actual = CommandSpec.from_command(cmd)
    if actual is None:
        return (
            "Reproduce-mode guard: could not parse the final experiment command. "
            "Run the chosen PaperContext command exactly instead of improvising."
        )
    if _norm_script(actual.script) != _norm_script(expected.script):
        return (
            "Reproduce-mode guard: final run script drifted from the chosen "
            f"experiment (`{expected.script}` expected, got `{actual.script}`)."
        )
    expected_launcher = expected.launcher.replace("python3", "python")
    actual_launcher = actual.launcher.replace("python3", "python")
    if expected_launcher and actual_launcher and expected_launcher != actual_launcher:
        return (
            "Reproduce-mode guard: final run launcher drifted from the chosen "
            f"experiment (`{expected.launcher}` expected, got `{actual.launcher}`)."
        )
    mismatches: list[str] = []
    for key, exp_value in expected.args.items():
        act_value = actual.args.get(key)
        if act_value is None:
            mismatches.append(f"{key}=<missing>")
            continue
        if _norm_value(act_value) != _norm_value(exp_value):
            mismatches.append(f"{key}={act_value} (expected {exp_value})")
    if mismatches:
        return (
            "Reproduce-mode guard: final run drifted from the chosen experiment "
            "arguments: " + ", ".join(mismatches[:6]) + "."
        )
    return None


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

    def validate_semantic(self, parsed: DockerExecInput, ctx: ToolUseContext) -> str | None:
        if str(ctx.options.get("run_mode") or "").lower() == "reproduce":
            mutates_placeholder = bool(
                re.search(r"(?<!2)>(?!\s*&\s*\d)", parsed.command)
                or "Image.fromarray(np.random" in parsed.command
                or ".save('/repo/data/test_image" in parsed.command
                or '.save("/repo/data/test_image' in parsed.command
            )
            if mutates_placeholder and any(
                re.search(pat, parsed.command) for pat in _REPRO_PLACEHOLDER_PATTERNS
            ):
                return (
                    "Reproduce-mode guard: refusing to create synthetic placeholder paper "
                    "inputs inside the sandbox. Use authoritative repo/paper artifacts, "
                    "or stop with `PAPER_RUN_FAILED`."
                )
            drift_msg = _reproduce_command_guard(
                parsed.command,
                ctx.options.get("paper_context"),
            )
            if drift_msg:
                return drift_msg
        return None

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

        # ── Bug 2: auto-restart if the container died (e.g. OOM-killed earlier) ──
        # Don't let the agent waste turns on a dead container. If we have a known-good
        # commit, transparently re-start the sandbox from it and tell the model.
        restart_note = await _maybe_restart_dead_container(ctx.sandbox)

        try:
            result = await ctx.sandbox.exec(
                parsed.command,
                timeout_s=parsed.timeout_s,
                cwd=parsed.cwd,
                env=parsed.env,
            )
        except RuntimeError as exc:
            # sandbox.exec raises if container is None even after restart attempt
            return ToolResult(
                data=DockerExecOutput(
                    exit_code=-1, stdout="", stderr=str(exc),
                    elapsed_s=0.0, truncated=False,
                ),
                text=(
                    f"Sandbox is unavailable: {exc}\n"
                    f"{restart_note}\n"
                    "No prior commit exists to recover from; the operator must restart the migration."
                ),
                is_error=True,
            )

        # ── Bug 3: OOM detection + actionable hint ──────────────────────────────
        oom_hint = _diagnose_crash_exit(result.exit_code, result.stderr)

        body = f"$ {parsed.command}\n[exit {result.exit_code} in {result.elapsed_s:.2f}s]\n"
        if restart_note:
            body = restart_note + "\n\n" + body
        if result.stdout:
            body += result.stdout
        if result.stderr:
            body += f"\n--- stderr ---\n{result.stderr}"
        if oom_hint:
            body += f"\n\n--- diagnostic ---\n{oom_hint}"

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


# ── Container resurrection ───────────────────────────────────────────────────


async def _maybe_restart_dead_container(sandbox) -> str:
    """If the sandbox's container is None/exited (e.g. OOM-killed), rebuild it from
    the most recent commit. Returns a human-readable note describing what happened
    (empty string if no restart was needed). Surfaced into the tool_result so the
    model sees the recovery explicitly.
    """
    # First check: is the container object missing entirely?
    container_dead = sandbox.container is None
    # Second check: is it a zombie? (object exists but `docker inspect` says exited)
    if not container_dead and sandbox.container is not None:
        try:
            sandbox.container.reload()
            status = sandbox.container.status
            if status in ("exited", "dead", "removing"):
                container_dead = True
        except Exception:
            # If we can't even inspect it, treat as dead
            container_dead = True

    if not container_dead:
        return ""

    target = sandbox.latest_commit() if hasattr(sandbox, "latest_commit") else None
    if target is None:
        # Try a cold restart from the original base image
        try:
            await sandbox.stop()
        except Exception:
            pass
        try:
            await sandbox.start()
            return (
                "[container was dead — restarted from the original base image; "
                "any uncommitted in-container state is lost]"
            )
        except Exception as exc:
            return f"[container was dead and cold restart failed: {exc}]"

    try:
        await sandbox.rollback(to_commit=target)
        node = sandbox.commit_log.nodes.get(target)
        label = (node.label if node else "") or target
        return (
            f"[container was dead — automatically restored to commit "
            f"'{label}' (id={target}). Any work after that commit is lost.]"
        )
    except Exception as exc:
        return f"[container was dead; rollback to {target} also failed: {exc}]"


# ── Crash diagnostics ────────────────────────────────────────────────────────


_OOM_STDERR_HINTS = (
    "out of memory",
    "outofmemoryerror",
    "killed",
    "cannot allocate",
    "memoryerror",
    "torch.cuda.OutOfMemoryError",
)


def _diagnose_crash_exit(exit_code: int, stderr: str) -> str:
    """Translate cryptic exit codes into actionable model-facing hints."""
    s_low = (stderr or "").lower()
    if exit_code == 137:
        return (
            "Exit 137 = SIGKILL — the process was killed by the kernel OOM killer "
            "(or by Docker hitting the container's memory limit). The model load + "
            "KV-cache + activations exceeded RAM. Mitigations, in order of preference:\n"
            "  1. Drop the model size / sequence length / batch size and rerun "
            "     (e.g. `python -m turboquant.test_turboquant` is a smoke test that "
            "     does NOT load Qwen-3B; try that first).\n"
            "  2. Disable bitsandbytes 4-bit quantization on ROCm — bnb's ROCm support "
            "     is partial and may silently fall back to FP16/FP32 (doubling memory).\n"
            "     Use plain `dtype=torch.float16` without `quantization_config`.\n"
            "  3. If the previous commit is intact, call `DockerRollback` then retry "
            "     with a SMALLER test command. Do NOT keep re-running the same big test."
        )
    if exit_code == 139:
        return (
            "Exit 139 = SIGSEGV — segfault inside the container. Likely a native "
            "extension (bitsandbytes, flash-attn, custom CUDA kernel) crashing on "
            "ROCm. Try uninstalling that wheel and using the pure-Python or AMD-fork "
            "replacement from the /cuda_to_rocm_mapping skill."
        )
    if exit_code == 124:
        return "Exit 124 = command timed out. Increase `timeout_s` or run a shorter operation."
    if exit_code != 0:
        for hint in _OOM_STDERR_HINTS:
            if hint in s_low:
                return (
                    f"Stderr contains a memory-error keyword ('{hint}'); this is "
                    "likely an OOM. See exit-137 mitigations above (smaller model / "
                    "drop bnb quantization / rollback + smaller test)."
                )
    return ""
