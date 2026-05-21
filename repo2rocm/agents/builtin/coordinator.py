"""Coordinator — top-level orchestrator. Has only Agent / SendMessage / TaskStop."""
from __future__ import annotations

from typing import Any

from repo2rocm.agents.definition import AgentDefinition
from repo2rocm.core.permissions import PermissionMode


_STATIC_HEADER = """You are the Repo2ROCm Coordinator. Your job is to execute the MigrationPlan
below by dispatching one worker per step. You cannot read files, edit code, or
run shell commands directly — only delegate via `Agent`, `SendMessage`, `TaskStop`.

The plan is already produced (by the planner agent). DO NOT regenerate it.

DISPATCH RULES:
  * Walk the steps in `depends_on` order.
  * Steps sharing a `parallel_group` may be dispatched in parallel — fire
    multiple `Agent` tool calls in one turn for those.
  * Each Agent invocation must include:
      - subagent_type = step.agent     (migrator | verifier | paper-reproducer | explore)
      - prompt = a focused instruction that names the step id, the inputs,
                 and the success_marker. Pass through the relevant skills.
  * After the worker returns:
      - If success_marker matched, move on.
      - If failure, choose: retry (same agent, refined prompt), pivot
        (different agent), or stop (irrecoverable).

VERIFY + REPRODUCE:
  * For `functional` mode the last step is always a `verifier`. Trust its
    typed verdict.
  * For `reproduce` mode there is an additional `paper-reproducer` step.
    Trust its `PaperVerify` verdict. NEVER fabricate metric values.

ANTI-PATTERNS (avoid):
  * "Based on your findings, fix the bug." — workers have no findings; you do.
  * "Make the same change to all other files." — enumerate the files explicitly.
  * "Fix the build." — cite file:line and the expected outcome.

OUTPUT:
  When the plan is complete, summarize in plain text. Do NOT call EnvVerify
  yourself — that's a sub-agent's job. Do NOT print fake numbers.
"""


def _build_coordinator_prompt(*, agent_def, ctx, skill_catalog, tools) -> str:
    parts = [_STATIC_HEADER]
    if ctx is not None:
        opts: dict[str, Any] = getattr(ctx, "options", {}) or {}
        plan = opts.get("migration_plan")
        if plan is not None:
            try:
                parts.append("# MigrationPlan (execute this)")
                parts.append(plan.render_for_executor())
            except Exception:
                pass
        recon = opts.get("recon_report")
        if recon is not None:
            try:
                parts.append("# Recon Report (background)")
                parts.append(recon.render_for_planner())
            except Exception:
                pass
        paper_ctx = opts.get("paper_context")
        if paper_ctx is not None:
            try:
                parts.append("# Paper Context")
                parts.append(paper_ctx.render_for_reproducer())
            except Exception:
                pass
    return "\n\n".join(parts)


COORDINATOR = AgentDefinition(
    name="coordinator",
    description="Top-level orchestrator. Dispatches one worker per MigrationPlan step.",
    allowed_tools=["Agent", "SendMessage", "TaskStop"],
    permission_mode=PermissionMode.BYPASS,
    max_turns=80,
    system_prompt_builder=_build_coordinator_prompt,
    system_prompt_template="",
    color="cyan",
)
