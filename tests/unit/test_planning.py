"""Planning: types, workflow loader, dispatcher."""
from __future__ import annotations

import pytest

from repo2rocm.planning import (
    MigrationPlan,
    PlanStep,
    load_workflow,
)
from repo2rocm.planning.dispatcher import (
    batch_by_parallel_group,
    dispatch_plan,
    render_step_prompt,
    topo_order,
)


def _plan(*steps: PlanStep) -> MigrationPlan:
    return MigrationPlan(
        repo="owner/repo",
        sha="",
        mode="functional",
        base_image="rocm/pytorch:latest",
        steps=list(steps),
    )


def test_workflow_load_functional():
    wf = load_workflow("functional")
    assert wf.mode == "functional"
    assert any(p.id == "P4" for p in wf.phases)
    assert any(p.agent == "verifier" for p in wf.phases)


def test_workflow_load_reproduce():
    wf = load_workflow("reproduce")
    assert wf.mode == "reproduce"
    assert any(p.agent == "paper-reproducer" for p in wf.phases)
    final = next(p for p in wf.phases if p.agent == "paper-reproducer")
    assert "PAPER_RUN_FAILED" in (final.success_marker or "")


def test_workflow_rejects_unknown_mode():
    with pytest.raises(ValueError):
        load_workflow("env")


def test_topo_order_respects_dependencies():
    plan = _plan(
        PlanStep(id="S3", title="c", agent="migrator", depends_on=["S2"]),
        PlanStep(id="S1", title="a", agent="migrator"),
        PlanStep(id="S2", title="b", agent="migrator", depends_on=["S1"]),
    )
    order = [s.id for s in topo_order(plan)]
    assert order.index("S1") < order.index("S2") < order.index("S3")


def test_topo_detects_cycle():
    plan = _plan(
        PlanStep(id="A", title="a", agent="migrator", depends_on=["B"]),
        PlanStep(id="B", title="b", agent="migrator", depends_on=["A"]),
    )
    with pytest.raises(ValueError):
        topo_order(plan)


def test_parallel_grouping():
    plan = _plan(
        PlanStep(id="S1", title="setup", agent="migrator"),
        PlanStep(id="S2a", title="a", agent="migrator", depends_on=["S1"], parallel_group="g1"),
        PlanStep(id="S2b", title="b", agent="migrator", depends_on=["S1"], parallel_group="g1"),
        PlanStep(id="S3", title="verify", agent="verifier", depends_on=["S2a", "S2b"]),
    )
    batches = batch_by_parallel_group(plan)
    assert any(b.parallel and {s.id for s in b.steps} == {"S2a", "S2b"} for b in batches)


def test_dispatch_plan_is_alias():
    plan = _plan(PlanStep(id="S1", title="a", agent="migrator"))
    assert dispatch_plan(plan) == batch_by_parallel_group(plan)


def test_render_step_prompt_includes_inputs_and_marker():
    step = PlanStep(
        id="S2",
        title="install deps",
        agent="migrator",
        inputs={"file": "requirements.txt"},
        success_marker="all_installed",
        skills=["nvidia_alternatives"],
        notes="strip banned wheels first",
    )
    plan = _plan(step)
    txt = render_step_prompt(step, plan)
    assert "[S2]" in txt
    assert "requirements.txt" in txt
    assert "all_installed" in txt
    assert "/nvidia_alternatives" in txt


def test_migration_plan_render():
    plan = _plan(
        PlanStep(id="S1", title="setup", agent="migrator", success_marker="image_ready"),
        PlanStep(id="S2", title="verify", agent="verifier", depends_on=["S1"]),
    )
    txt = plan.render_for_executor()
    assert "Migration Plan (functional)" in txt
    assert "[S1]" in txt
    assert "deps=S1" in txt


def test_migration_plan_step_lookup():
    s = PlanStep(id="S1", title="x", agent="migrator")
    plan = _plan(s)
    assert plan.step("S1") is s
    assert plan.step("missing") is None
