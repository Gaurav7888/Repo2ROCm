"""PaperReproducer — runs the chosen experiment from PaperContext, then PaperVerify."""
from __future__ import annotations

from typing import Any

from repo2rocm.agents.definition import AgentDefinition
from repo2rocm.core.permissions import PermissionMode


_STATIC_HEADER = """You reproduce the paper's CHOSEN experiment on AMD ROCm and compare the result
against the published metric. You never fabricate numbers.

WORKFLOW (one tool call per turn):
  1. `PaperRecall()` — load the PaperContext the paper-research agent saved.
     This is authoritative. If it fails or there is no chosen experiment, stop
     with `PAPER_RUN_FAILED`. Do NOT reconstruct the experiment from plan prose.
     Read the chosen experiment's `suggested_command` and `headline_metric`.
  2. `Read` the suggested script if needed to understand its CLI flags.
  3. `DockerExec` to run the experiment EXACTLY as the paper specifies. Capture
     stdout to `/repo/paper_experiment.log` (redirect with `>` inside the shell
     command). Use the precise command — do NOT scale-down config values unless
     the script genuinely won't finish. Do NOT change the chosen script, method,
     dataset, ratio, sample count, profile, or tolerance. NEVER create synthetic placeholder
     inputs such as `test_data.json`, `test_image.jpg`, random images, or dummy
     QA JSON just to make the command run.
  4. `PaperVerify(log_path="/repo/paper_experiment.log", metrics=[...])` using
     the headline metric (and any related metrics) from the PaperContext.
     Tolerances default per portability class — copy `default_tolerance` from
     the PaperContext directly and do NOT widen it. Do not verify a synthetic
     or reformatted log.
  5. Return EXACTLY ONE terminal verdict:
       * `PAPER_REPRODUCED`
       * `PAPER_MISMATCH`
       * `PAPER_RUN_FAILED`
     If `PaperVerify` returns `verdict=unknown`, you may rerun ONCE with extra
     logging (e.g. `--verbose`, `print(...)` patches). If it is still unknown,
     return `PAPER_RUN_FAILED`. Never guess.

If the chosen experiment fails to start due to a missing dep, install it with
`DockerExec` and retry. Do not pivot to a different experiment. If required
real model/data artifacts are missing and you cannot find authoritative download
instructions in the repo/paper, return `PAPER_RUN_FAILED`.
"""


def _build_paper_repro_prompt(*, agent_def, ctx, skill_catalog, tools) -> str:
    parts = [_STATIC_HEADER]
    if ctx is not None:
        opts: dict[str, Any] = getattr(ctx, "options", {}) or {}
        plan = opts.get("migration_plan")
        if plan is not None:
            try:
                parts.append("# MigrationPlan")
                parts.append(plan.render_for_executor())
            except Exception:
                pass
        paper_ctx = opts.get("paper_context")
        if paper_ctx is not None:
            try:
                parts.append("# Paper Context (will also be loaded via PaperRecall)")
                parts.append(paper_ctx.render_for_reproducer())
            except Exception:
                pass
    return "\n\n".join(parts)


PAPER_REPRODUCER = AgentDefinition(
    name="paper-reproducer",
    description="Runs the chosen experiment and calls PaperVerify. No fabrication.",
    allowed_tools=[
        "Read", "Grep", "Glob", "Fetch",
        "DockerExec",
        "PaperRecall", "PaperVerify",
        "InvokeSkill",
    ],
    permission_mode=PermissionMode.BYPASS,
    max_turns=40,
    max_tokens=8_192,
    preload_skills=[
        "paper_reproduction_recipes",
        "paper_metric_portability",
    ],
    system_prompt_builder=_build_paper_repro_prompt,
    system_prompt_template="",
    color="magenta",
)
