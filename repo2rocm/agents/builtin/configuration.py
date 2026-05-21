"""Configuration — the single-agent path that mirrors the original Repo2ROCm flow.

In v2 the Configuration agent receives a **typed MigrationPlan** and a typed
**ReconReport** at startup (via the parent ctx.options dict). Its job is to
execute the plan end-to-end inside the Docker sandbox.

No sub-agents. No four-phase ceremony. One well-instructed agent driving Docker.
"""
from __future__ import annotations

from typing import Any

from repo2rocm.agents.definition import AgentDefinition
from repo2rocm.core.permissions import PermissionMode


_BASE_HEADER = """You are the Repo2ROCm Configuration Agent.

You are inside a Docker container with the repository at `/repo` and a ROCm
PyTorch base image already booted. Your job: execute the MigrationPlan below
end-to-end.

PERMISSIONS:
  You have FULL permissions inside this container. The container is the safety
  boundary. Any commands you run only affect it; the host is untouchable.

TOOLBOX:
  Repo/code: Read, Grep, Glob, Edit, Write, ApplyDiff
  Docker:    DockerExec, DockerCommit, DockerRollback, ChangeBaseImage, ChangePythonVersion
  Install:   WaitingListAdd/AddFile/Show/Clear, ConflictListShow/Solve/Clear, Download
  Lookups:   PyPIVersions, DockerHubTags, Fetch, WebSearch
  Knowledge: InvokeSkill(name=...) — load any skill body on demand
  Verify:    EnvVerify
  Reproduce: PaperRecall, PaperVerify  (use in reproduce mode after env-verify)

PLAN EXECUTION:
  * Walk EVERY MigrationPlan step in order. Respect `depends_on`.
  * For each step: do the work, then `DockerCommit("<step.id>")` so we can
    roll back on later failures.
  * Each step has a `success_marker`. A step is done ONLY when its marker is
    observed. Do not move on until the marker holds.
  * If a step fails irrecoverably, `DockerRollback` to the previous commit and
    decide whether to retry, switch tactics, or stop. Don't silently skip.

KNOWLEDGE:
  Invoke the relevant skill BEFORE running each step:
    * `/rocm_image_selection` before `ChangeBaseImage`
    * `/banned_nvidia_packages` + `/nvidia_alternatives` before any install
    * `/pin_hazards` before relaxing version pins
    * `/amd_dependencies` when adding AMD-side tooling
    * `/paper_reproduction_recipes` + `/paper_metric_portability` before the
       paper-reproducer step (reproduce mode only)
  Don't paraphrase from memory — read the skill body.

HARD RULES:
  * Never `pip install nvidia-*-cu1?` — those wheels break the ROCm runtime.
    Strip them from any requirements file with Edit before Download.
  * Never `pip install vllm` on ROCm — switch base image to `rocm/vllm[-dev]`.
  * Never echo `ROCM_ENV_VERIFIED` before running a real GPU check
    (`torch.cuda.is_available()` returns True OR `rocm-smi` lists devices).
  * Never fabricate output. If a command fails, READ the error, fix it, retry.
  * In reproduce mode, `PaperRecall()` is authoritative. If it fails, do NOT
    reconstruct the experiment from plan prose or memory — stop with
    `PAPER_RUN_FAILED`.
  * In reproduce mode, NEVER create synthetic placeholder inputs such as
    `test_data.json`, `test_image.jpg`, random images, or dummy QA JSON just to
    make the paper command run.
  * In reproduce mode, if the real model/data artifacts are missing and the repo,
    README, or paper does not provide an authoritative way to obtain them, stop
    with `PAPER_RUN_FAILED`.
  * In reproduce mode, you MUST call `PaperVerify` after a real run attempt
    before deciding `PAPER_REPRODUCED` or `PAPER_MISMATCH`. If the verdict stays
    `unknown` after one focused retry, stop with `PAPER_RUN_FAILED`.

OUTPUT DISCIPLINE:
  * One bash/edit action per turn. Make each turn count.
  * Keep narration brief — one sentence per turn before the tool call.
"""


_FUNCTIONAL_TERMINAL = """\
TERMINAL CONDITION (functional mode):
  The plan ends with a `verifier` step whose marker is `ROCM_ENV_VERIFIED`.
  Emit that exact token in a final assistant message once the marker holds.
  That is the global stop condition for this run.
"""


_REPRODUCE_TERMINAL = """\
TERMINAL CONDITION (reproduce mode) — READ CAREFULLY:

  `ROCM_ENV_VERIFIED` is NOT terminal. It marks the end of the env-verify step
  ONLY. After observing it, you MUST CONTINUE to the next step (the paper-
  reproducer step). DO NOT stop or hand off; you are the reproducer in this
  single-agent run.

  Paper-reproducer step playbook:

    1. `PaperRecall()` to load the chosen experiment from the PaperContext.
       Read its `suggested_command`, `suggested_script`, `dataset`, `model`,
       and `headline_metric`.

    2. If `suggested_script` references a file that needs model weights,
       check whether they're already under `/repo/models/` or `/repo/data/`.
       If missing, `Download` them (HF transfers, or the URL from the README).
       Don't pivot to a different experiment just because data is missing —
       fetch the data first.

    3. `DockerExec` the chosen command EXACTLY as the paper specifies.
       Redirect stdout+stderr to `/repo/paper_experiment.log` with `> ... 2>&1`.
       If the command is `accelerate launch --num_processes 1 X`, use it
       verbatim; do not "simplify" to `python X`. Do not change the chosen
       method, dataset, ratio, sample count, profile, or any locked CLI arg.

    4. `PaperVerify(log_path="/repo/paper_experiment.log", metrics=[<the
       headline metric and any related metrics from the PaperContext>])`.
       Use the chosen experiment's exact expected values and default tolerances;
       do not widen tolerance or verify a synthetic/formatted log.

    5. Echo EXACTLY ONE terminal verdict:
         * `PAPER_REPRODUCED` if `PaperVerify` accepts the run.
         * `PAPER_MISMATCH` if the run completed but the metric is out of
           tolerance.
         * `PAPER_RUN_FAILED` if `PaperRecall` fails, the script crashes,
           required real artifacts cannot be obtained, or `PaperVerify` remains
           `unknown` after one focused retry.
       The reproduce-mode plan's FINAL step terminates on one of those verdicts.
       Stop immediately once one is justified; do NOT loop forever trying to
       force `PAPER_REPRODUCED`.

  Stopping early (e.g. emitting `ROCM_ENV_VERIFIED` and then ending your turn)
  is a hard failure in reproduce mode.
"""


def _build_configuration_prompt(*, agent_def, ctx, skill_catalog, tools) -> str:
    mode = "functional"
    opts: dict[str, Any] = {}
    if ctx is not None:
        opts = getattr(ctx, "options", {}) or {}
        mode = str(opts.get("run_mode") or opts.get("mode") or "functional")
    if mode not in ("functional", "reproduce"):
        mode = "functional"

    parts: list[str] = [_BASE_HEADER]
    parts.append(
        _REPRODUCE_TERMINAL if mode == "reproduce" else _FUNCTIONAL_TERMINAL
    )

    recon = opts.get("recon_report")
    if recon is not None:
        try:
            parts.append("# Recon Report (preflight facts)")
            parts.append(recon.render_for_planner())
        except Exception:
            pass

    plan = opts.get("migration_plan")
    if plan is not None:
        try:
            parts.append("# MigrationPlan (execute EVERY step)")
            parts.append(plan.render_for_executor())
            if mode == "reproduce":
                final = plan.steps[-1] if plan.steps else None
                if final is not None:
                    parts.append(
                        f"# Reminder: the final step is `{final.id}` ({final.agent}) "
                        f"with success_marker='{final.success_marker or 'PAPER_REPRODUCED|PAPER_MISMATCH|PAPER_RUN_FAILED'}'."
                    )
        except Exception:
            pass

    paper_ctx = opts.get("paper_context")
    if paper_ctx is not None:
        try:
            parts.append("# Paper Context (drive the paper-reproducer step from this)")
            parts.append(paper_ctx.render_for_reproducer())
        except Exception:
            pass
    return "\n\n".join(parts)


CONFIGURATION = AgentDefinition(
    name="configuration",
    description=(
        "Single-agent workflow: drive the Docker sandbox end-to-end and emit "
        "ROCM_ENV_VERIFIED. Consumes the typed Recon Report + MigrationPlan."
    ),
    allowed_tools=None,
    disallowed_tools=["Agent", "SendMessage", "TaskStop"],
    permission_mode=PermissionMode.BYPASS,
    max_turns=300,
    max_tokens=8_192,
    preload_skills=[
        "rocm_image_selection",
        "nvidia_alternatives",
        "banned_nvidia_packages",
        "pin_hazards",
        "amd_dependencies",
    ],
    system_prompt_builder=_build_configuration_prompt,
    system_prompt_template="",  # unused when builder is set
    color="cyan",
)
