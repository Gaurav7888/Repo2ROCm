"""Migrator — write-heavy worker. One per plan step in coordinator mode."""
from __future__ import annotations

from typing import Any

from repo2rocm.agents.definition import AgentDefinition
from repo2rocm.core.permissions import PermissionMode


_STATIC_HEADER = """You are a Migrator worker. The Coordinator handed you ONE focused plan step
with exact file paths and exact intended changes. Execute it. Do NOT explore freely.

You have the full ROCm tool set:
  - Read, Grep, Glob, Edit, Write, ApplyDiff
  - DockerExec, DockerCommit, DockerRollback, ChangeBaseImage, ChangePythonVersion
  - WaitingListAdd/AddFile/Show/Clear, ConflictListShow/Solve/Clear, Download
  - PyPIVersions, DockerHubTags, Fetch, WebSearch
  - EnvVerify, InvokeSkill

You do NOT have the Agent tool. You cannot spawn sub-workers. If the step is wider
than expected, finish what you can and report the gap in your final message —
the Coordinator will dispatch additional workers.

Workflow:
  1. Read the target file(s) FIRST so the staleness cache is warm.
  2. Invoke any skill referenced in your step (`/banned_nvidia_packages`,
     `/nvidia_alternatives`, `/pin_hazards`, ...) via `InvokeSkill`.
  3. For each install: `PyPIVersions` / `DockerHubTags` FIRST, then `Download`.
  4. After each successful sub-action, call `DockerCommit` with a short label.
  5. On a non-recoverable failure, `DockerRollback` and report.
  6. Call `EnvVerify` only if the step asked for it; otherwise return.

Output:
  A single concise summary message: what you did, what failed, the last
  commit_id (so the Coordinator can rollback safely).
"""


def _build_migrator_prompt(*, agent_def, ctx, skill_catalog, tools) -> str:
    parts = [_STATIC_HEADER]
    if ctx is not None:
        opts: dict[str, Any] = getattr(ctx, "options", {}) or {}
        step = opts.get("plan_step")
        if step is not None:
            parts.append("# Your assigned step")
            try:
                parts.append(f"  id={step.id}  title={step.title!r}")
                parts.append(f"  inputs={step.inputs}")
                if step.success_marker:
                    parts.append(f"  success_marker={step.success_marker}")
                if step.skills:
                    parts.append(f"  skills={step.skills}")
            except Exception:
                pass
    return "\n\n".join(parts)


MIGRATOR = AgentDefinition(
    name="migrator",
    description="Write-heavy worker. One step at a time, dispatched by the coordinator.",
    allowed_tools=None,
    disallowed_tools=["Agent", "SendMessage", "TaskStop"],
    permission_mode=PermissionMode.BYPASS,
    max_turns=60,
    max_tokens=8_192,
    preload_skills=[
        "nvidia_alternatives",
        "banned_nvidia_packages",
        "pin_hazards",
        "amd_dependencies",
    ],
    system_prompt_builder=_build_migrator_prompt,
    system_prompt_template="",
    color="yellow",
)
