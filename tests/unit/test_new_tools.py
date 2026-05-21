"""Coverage for the new tools: InvokeSkill, EmitPlan, PaperRecall."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from repo2rocm.core.permissions import PermissionMode
from repo2rocm.paper import PaperCorpus
from repo2rocm.paper.types import (
    Experiment,
    MetricRow,
    PaperContext,
    PaperMetadata,
)
from repo2rocm.planning import MigrationPlan, PlanStep
from repo2rocm.tools.base import ReadFileState, ToolUseContext
from repo2rocm.tools.paper.paper_recall import PaperRecall
from repo2rocm.tools.planning.emit_plan import EmitPlan
from repo2rocm.tools.skills.invoke_skill import InvokeSkill


def _ctx(tmp_path: Path) -> ToolUseContext:
    return ToolUseContext(
        agent_id="t",
        session_id="s",
        workdir=tmp_path,
        abort_event=asyncio.Event(),
        permission_mode=PermissionMode.DEFAULT,
        read_file_state=ReadFileState(),
    )


@pytest.mark.asyncio
async def test_invoke_skill_loads_builtin(tmp_path: Path):
    tool = InvokeSkill()
    res = await tool.invoke({"name": "nvidia_alternatives"}, _ctx(tmp_path))
    assert not res.is_error
    assert "flash-attn" in res.text.lower() or "flash_attn" in res.text.lower()
    assert res.data.found is True


@pytest.mark.asyncio
async def test_invoke_skill_unknown(tmp_path: Path):
    tool = InvokeSkill()
    res = await tool.invoke({"name": "definitely-not-real"}, _ctx(tmp_path))
    assert res.is_error
    assert res.data.found is False


@pytest.mark.asyncio
async def test_emit_plan_validates_and_persists(tmp_path: Path):
    tool = EmitPlan()
    ctx = _ctx(tmp_path)
    valid_plan = {
        "repo": "owner/repo",
        "mode": "functional",
        "base_image": "rocm/pytorch:latest",
        "steps": [
            {
                "id": "S1",
                "title": "install",
                "agent": "migrator",
                "inputs": {},
                "depends_on": [],
            },
            {
                "id": "S2",
                "title": "verify",
                "agent": "verifier",
                "inputs": {},
                "depends_on": ["S1"],
                "success_marker": "ROCM_ENV_VERIFIED",
            },
        ],
    }
    res = await tool.invoke({"plan": valid_plan}, ctx)
    assert not res.is_error
    assert res.data.ok is True
    assert res.data.step_count == 2
    path = Path(res.data.path)
    assert path.is_file()
    assert isinstance(ctx.options["migration_plan"], MigrationPlan)


@pytest.mark.asyncio
async def test_emit_plan_rejects_bad_agent(tmp_path: Path):
    tool = EmitPlan()
    res = await tool.invoke(
        {
            "plan": {
                "repo": "x",
                "mode": "functional",
                "base_image": "rocm/pytorch:latest",
                "steps": [
                    {"id": "S1", "title": "t", "agent": "nonexistent", "inputs": {}},
                ],
            }
        },
        _ctx(tmp_path),
    )
    assert res.is_error


@pytest.mark.asyncio
async def test_paper_recall_returns_latest(tmp_path: Path):
    corpus = PaperCorpus(tmp_path / "papers")
    ctx_obj = PaperContext(
        metadata=PaperMetadata(arxiv_id="2401.00099", title="T"),
        experiments=[
            Experiment(
                id="E1",
                title="x",
                headline_metric=MetricRow(
                    name="acc", value=1.0, portability="accuracy", default_tolerance=0.03,
                ),
            ),
        ],
        chosen_experiment_id="E1",
    )
    corpus.save(ctx_obj)
    # Regression: PaperRecall should ignore auxiliary JSON artifacts in the same
    # directory and still return the real PaperContext.
    (tmp_path / "papers" / "repo_entry_points.json").write_text("{}", encoding="utf-8")
    (tmp_path / "papers" / "2401.00099.experiments.json").write_text("[]", encoding="utf-8")
    tool = PaperRecall()
    res = await tool.invoke({"arxiv_id": ""}, _ctx(tmp_path))
    assert not res.is_error
    assert res.data.found is True
    assert res.data.context["chosen_experiment_id"] == "E1"


@pytest.mark.asyncio
async def test_paper_recall_missing(tmp_path: Path):
    tool = PaperRecall()
    res = await tool.invoke({"arxiv_id": "9999.00000"}, _ctx(tmp_path))
    assert res.is_error
    assert res.data.found is False


@pytest.mark.asyncio
async def test_paper_recall_prefers_context_already_in_ctx(tmp_path: Path):
    ctx = _ctx(tmp_path)
    ctx_obj = PaperContext(
        metadata=PaperMetadata(arxiv_id="2412.03409", title="PrefixKV"),
        experiments=[
            Experiment(
                id="E1",
                title="chosen",
                headline_metric=MetricRow(
                    name="ppl",
                    value=5.5,
                    portability="other",
                    default_tolerance=0.15,
                ),
            ),
        ],
        chosen_experiment_id="E1",
    )
    ctx.options["paper_context"] = ctx_obj
    (tmp_path / "papers").mkdir(parents=True, exist_ok=True)
    (tmp_path / "papers" / "repo_entry_points.json").write_text("{}", encoding="utf-8")

    tool = PaperRecall()
    res = await tool.invoke({"arxiv_id": ""}, ctx)

    assert not res.is_error
    assert res.data.found is True
    assert res.data.context["metadata"]["arxiv_id"] == "2412.03409"
    assert "Chosen experiment: E1" in res.text


@pytest.mark.asyncio
async def test_emit_plan_works_in_plan_mode(tmp_path: Path):
    """Regression: previously the planner ran in PermissionMode.PLAN and every
    EmitPlan call was denied with `plan mode denies mutations`, leaving no
    artifact on disk and aborting the whole pipeline. EmitPlan now has an
    explicit allow() in check_permissions, so PLAN-mode callers also succeed."""
    tool = EmitPlan()
    ctx = ToolUseContext(
        agent_id="t",
        session_id="s",
        workdir=tmp_path,
        abort_event=asyncio.Event(),
        permission_mode=PermissionMode.PLAN,
        read_file_state=ReadFileState(),
    )
    plan = {
        "repo": "o/r",
        "mode": "functional",
        "base_image": "rocm/pytorch:latest",
        "steps": [
            {"id": "S1", "title": "x", "agent": "migrator", "inputs": {}, "depends_on": []},
        ],
    }
    res = await tool.invoke({"plan": plan}, ctx)
    assert not res.is_error, res.text
    assert (tmp_path / "plans" / "migration_plan.json").is_file()


def test_plan_step_renders_for_executor_without_paper_metric_keys():
    plan = MigrationPlan(
        repo="o/r",
        mode="functional",
        base_image="rocm/pytorch:latest",
        steps=[
            PlanStep(
                id="S1",
                title="install",
                agent="migrator",
                inputs={"path": "requirements.txt"},
            ),
        ],
    )
    txt = plan.render_for_executor()
    assert "S1" in txt and "migrator" in txt
