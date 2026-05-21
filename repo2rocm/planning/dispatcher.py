"""Convert MigrationPlan steps into agent invocations.

Used by the coordinator agent (and the single-agent CONFIGURATION as guidance).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from repo2rocm.planning.types import MigrationPlan, PlanStep


@dataclass
class StepBatch:
    """A group of steps that can execute concurrently."""

    parallel: bool
    steps: list[PlanStep]


def topo_order(plan: MigrationPlan) -> list[PlanStep]:
    """Return steps in dependency order. Cycles raise ValueError."""
    by_id = {s.id: s for s in plan.steps}
    in_deg: dict[str, int] = {s.id: 0 for s in plan.steps}
    succ: dict[str, list[str]] = defaultdict(list)
    for s in plan.steps:
        for d in s.depends_on:
            if d not in by_id:
                continue
            in_deg[s.id] += 1
            succ[d].append(s.id)

    queue = [sid for sid, deg in in_deg.items() if deg == 0]
    order: list[PlanStep] = []
    while queue:
        sid = queue.pop(0)
        order.append(by_id[sid])
        for n in succ[sid]:
            in_deg[n] -= 1
            if in_deg[n] == 0:
                queue.append(n)
    if len(order) != len(plan.steps):
        raise ValueError("dependency cycle detected in plan")
    return order


def batch_by_parallel_group(plan: MigrationPlan) -> list[StepBatch]:
    """Group steps by parallel_group, respecting topo order."""
    ordered = topo_order(plan)
    batches: list[StepBatch] = []
    i = 0
    while i < len(ordered):
        s = ordered[i]
        if s.parallel_group is None:
            batches.append(StepBatch(parallel=False, steps=[s]))
            i += 1
            continue
        group = s.parallel_group
        run = [s]
        j = i + 1
        while j < len(ordered) and ordered[j].parallel_group == group:
            run.append(ordered[j])
            j += 1
        batches.append(StepBatch(parallel=True, steps=run))
        i = j
    return batches


def dispatch_plan(plan: MigrationPlan) -> list[StepBatch]:
    """Public entry: ordered batches ready for the coordinator to dispatch."""
    return batch_by_parallel_group(plan)


def render_step_prompt(step: PlanStep, plan: MigrationPlan) -> str:
    """Build the prompt the coordinator hands to a worker for ONE step."""
    parts = [
        f"You are executing plan step [{step.id}] {step.title!r} (mode={plan.mode}).",
        f"Goal: {step.notes or step.title}",
        "Inputs (treat as authoritative):",
    ]
    for k, v in step.inputs.items():
        parts.append(f"  - {k} = {v!r}")
    if step.success_marker:
        parts.append(f"Success criterion: {step.success_marker}")
    if step.skills:
        parts.append(f"Relevant skills (invoke when useful): {', '.join('/' + s for s in step.skills)}")
    parts.append(
        "When done, return ONE concise summary line: "
        "what you changed, what passed, the last DockerCommit id if any."
    )
    return "\n".join(parts)
