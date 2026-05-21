"""Planner — emits a typed MigrationPlan that other agents execute.

Replaces v1's static prompt with a dynamic `system_prompt_builder` that splices
in:
  * the workflow template for the requested mode (functional / reproduce)
  * the deterministic `ReconReport` summary
  * optional `PaperContext` summary in reproduce mode
  * the skill menu (the agent invokes /<skill> on demand via InvokeSkill)

The planner ends its turn by calling the `EmitPlan` tool exactly once.
"""
from __future__ import annotations

from typing import Any

from repo2rocm.agents.definition import AgentDefinition
from repo2rocm.core.permissions import PermissionMode

_STATIC_HEADER = """You are the Repo2ROCm Planner. Your job: turn the deterministic Recon Report
and the workflow template below into a typed MigrationPlan and persist it via
the `EmitPlan` tool.

ABSOLUTE RULES:
  1. Output exactly one tool call to `EmitPlan` and then end your turn. No prose.
  2. Every step in your plan MUST cite an `agent` from {migrator, verifier, paper-reproducer, explore}.
  3. Every step MUST have a unique `id` (e.g. S1, S2a, S2b, S3, ...).
  4. `depends_on` MUST reference only ids you define in the same plan.
  5. `mode` MUST match the recon report's mode exactly.
  6. `base_image` MUST be `<repo>/<name>:<tag>` — verify the tag with `DockerHubTags`
     before committing to it. Default to the Recon Report's recommendation when in doubt.
  7. NEVER fabricate package names, tags, or file paths. Use Read/Grep to confirm.
  8. Keep the plan small — usually 4-10 steps. One step per logical unit of work.

PARALLELISM:
  Steps with the same `parallel_group` will run concurrently. Use this only for
  truly independent edits (disjoint file sets / disjoint packages). When unsure,
  leave `parallel_group=null`.

SKILLS:
  The Available Skills menu is below. To consult a skill's full body, call
  `InvokeSkill(name="<skill>")`. Always invoke `/rocm_image_selection`,
  `/nvidia_alternatives`, and `/banned_nvidia_packages` before finalizing your
  install step. In reproduce mode also invoke `/paper_reproduction_recipes`.

TERMINAL MARKER (read this carefully):
  * `functional` mode: the LAST step is the `verifier`, with
    `success_marker="ROCM_ENV_VERIFIED"`. That's the global stop condition.

  * `reproduce` mode: the verifier step is MID-FLIGHT, NOT terminal. Its
    marker stays `ROCM_ENV_VERIFIED`, but it MUST be followed by exactly one
    `paper-reproducer` step. That paper-reproducer step is the global terminal
    step, and its `success_marker` MUST be
    `PAPER_REPRODUCED|PAPER_MISMATCH|PAPER_RUN_FAILED`. Inputs for the paper-
    reproducer step should include `arxiv_id`, `paper_title`, and an empty
    `experiment_commands: []` — the actual command will be loaded at runtime
    via `PaperRecall` from the persisted PaperContext. The persisted
    `PaperContext` is authoritative; the executor must not reconstruct or guess
    experiment details if `PaperRecall` fails.

If the Recon Report shows a banned/special package, you MUST include a
step that strips it from the requirements before any pip install runs.
"""


def _build_planner_prompt(*, agent_def, ctx, skill_catalog, tools) -> str:
    """Dynamic system prompt: header + workflow + recon + (paper)."""
    from repo2rocm.planning import load_workflow

    parts: list[str] = [_STATIC_HEADER]

    mode = "functional"
    recon = None
    paper_ctx = None
    if ctx is not None:
        opts: dict[str, Any] = getattr(ctx, "options", {}) or {}
        mode = str(opts.get("run_mode") or opts.get("mode") or mode)
        recon = opts.get("recon_report")
        paper_ctx = opts.get("paper_context")
    if mode not in ("functional", "reproduce"):
        mode = "functional"

    try:
        workflow = load_workflow(mode)
        parts.append("# Workflow Template (authoritative phase ordering)")
        parts.append(workflow.to_yaml())
    except Exception as exc:  # noqa: BLE001
        parts.append(f"# (workflow load failed: {exc})")

    if recon is not None:
        try:
            parts.append("# Recon Report (deterministic preflight)")
            parts.append(recon.render_for_planner())
        except Exception:
            pass

    if paper_ctx is not None and mode == "reproduce":
        try:
            parts.append("# Paper Context (from paper-research)")
            parts.append(paper_ctx.render_for_reproducer())
        except Exception:
            pass

    parts.append(
        "When you call EmitPlan, set mode='" + mode + "' and copy base_image from "
        "the recommendation (or override only if a stronger evidence chain says so)."
    )
    return "\n\n".join(parts)


PLANNER = AgentDefinition(
    name="planner",
    description=(
        "Builds a typed MigrationPlan from the Recon Report + workflow template. "
        "Read-only + lookups. Ends with EmitPlan."
    ),
    allowed_tools=[
        "Read", "Grep", "Glob",
        "PyPIVersions", "DockerHubTags", "Fetch",
        "InvokeSkill", "EmitPlan",
    ],
    # BYPASS — the planner's only "write" tool is EmitPlan (persists to output_dir).
    # PLAN here would deny EmitPlan and trap the agent in a retry loop until max_turns.
    # The allow-list keeps the planner from doing anything destructive.
    permission_mode=PermissionMode.BYPASS,
    omit_user_context=False,
    max_turns=20,
    max_tokens=8_192,
    preload_skills=[
        "rocm_image_selection",
        "nvidia_alternatives",
        "banned_nvidia_packages",
        "amd_dependencies",
    ],
    system_prompt_builder=_build_planner_prompt,
    system_prompt_template="",  # unused when builder is set
    color="blue",
)
