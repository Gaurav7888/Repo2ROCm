"""Plan-as-data.

This package owns:
  * `MigrationPlan` / `PlanStep` — the typed contract the planner emits.
  * `WorkflowTemplate` — the canonical phase sequence per mode, loaded from YAML.
  * `load_workflow(mode)` — return the parsed workflow.
  * `dispatch_plan(plan, ...)` — convert plan steps into Agent invocations.

The planner agent uses these to produce a `MigrationPlan` that other agents
(configuration / coordinator / migrator / verifier / paper-reproducer) execute.
"""
from repo2rocm.planning.dispatcher import dispatch_plan
from repo2rocm.planning.loader import WorkflowPhase, WorkflowTemplate, load_workflow
from repo2rocm.planning.types import MigrationPlan, PlanStep

__all__ = [
    "MigrationPlan",
    "PlanStep",
    "WorkflowPhase",
    "WorkflowTemplate",
    "load_workflow",
    "dispatch_plan",
]
