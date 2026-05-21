"""EmitPaperContext \u2014 terminal tool for the paper-research agent.

The agent calls this EXACTLY ONCE at the end of its turn with a fully-bound
`PaperContext`. The tool:

  * validates the dict against the `PaperContext` pydantic model
  * enforces a minimum reproducibility bar (chosen experiment exists, has a
    headline metric with a target value, has a runnable script or command)
  * persists `papers/<arxiv_id>.json` so `PaperRecall` can load it from any
    downstream agent (planner, reproducer)
  * stores the typed object on `ctx.options["paper_context"]` so the same
    process keeps a fast path

Mirrors `EmitPlan` for the planner agent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from repo2rocm.core.permissions import PermissionDecision, allow
from repo2rocm.paper import PaperCorpus
from repo2rocm.paper.types import Experiment, PaperContext
from repo2rocm.tools.base import BaseTool, ToolResult, ToolUseContext


class EmitPaperContextInput(BaseModel):
    context: dict[str, Any] = Field(
        ...,
        description=(
            "The full PaperContext as a JSON object: "
            "{metadata{arxiv_id,title,authors,abstract,hardware_claimed,libraries,pdf_path,html_path,text_path,headline_metrics}, "
            "experiments[]{id,title,description,model_checkpoint,dataset,prompt_template,"
            "metric{name,value,unit,portability,default_tolerance,paper_source,repo_eval_source,is_baseline}, "
            "related_metrics[], hyperparameters[]{name,value,unit,paper_source}, "
            "repo_bindings[]{hyperparam_name,kind,location,default}, "
            "unbound_hyperparameters[], suggested_script, suggested_command, "
            "runtime_class, estimated_runtime_min, portability_score, repo_match_confidence, rationale}, "
            "chosen_experiment_id}."
        ),
    )


class EmitPaperContextOutput(BaseModel):
    ok: bool
    path: str = ""
    chosen_experiment_id: str = ""
    experiment_count: int = 0
    chosen_summary: str = ""
    warnings: list[str] = Field(default_factory=list)
    error: str = ""


class EmitPaperContext(BaseTool[EmitPaperContextInput, EmitPaperContextOutput]):
    name: ClassVar[str] = "EmitPaperContext"
    description: ClassVar[str] = (
        "Validate and persist the PaperContext. The paper-research agent calls this "
        "EXACTLY ONCE at the end of its turn. The context must include a chosen "
        "experiment with a headline metric (name+value+portability+paper_source) and "
        "either a suggested_command OR a suggested_script. Every hyperparameter you "
        "extract from the paper must either appear in repo_bindings or in "
        "unbound_hyperparameters \u2014 the binding can't be silently omitted. After "
        "EmitPaperContext returns ok=true, end your turn."
    )
    input_model: ClassVar[type[BaseModel]] = EmitPaperContextInput
    max_result_size_chars: ClassVar[int] = 12_000

    def is_concurrency_safe(self, parsed: EmitPaperContextInput) -> bool:
        return False

    def is_read_only(self, parsed: EmitPaperContextInput) -> bool:
        return False

    def check_permissions(
        self, parsed: EmitPaperContextInput, ctx: ToolUseContext
    ) -> PermissionDecision:
        return allow("EmitPaperContext only writes a PaperContext JSON under workdir/papers/")

    async def call(
        self, parsed: EmitPaperContextInput, ctx: ToolUseContext
    ) -> ToolResult[EmitPaperContextOutput]:
        try:
            paper_ctx = PaperContext.model_validate(parsed.context)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                data=EmitPaperContextOutput(ok=False, error=str(exc)),
                text=f"PaperContext validation failed: {exc}",
                is_error=True,
            )

        warnings: list[str] = []
        chosen = paper_ctx.chosen()
        if not paper_ctx.experiments:
            return ToolResult(
                data=EmitPaperContextOutput(ok=False, error="no experiments"),
                text=(
                    "PaperContext.experiments is empty. Emit at least one "
                    "Experiment with a headline metric."
                ),
                is_error=True,
            )
        if chosen is None:
            return ToolResult(
                data=EmitPaperContextOutput(
                    ok=False, error="chosen_experiment_id not found"
                ),
                text=(
                    f"chosen_experiment_id={paper_ctx.chosen_experiment_id!r} does "
                    "not match any experiment.id"
                ),
                is_error=True,
            )

        err = _require_reproducibility(chosen)
        if err is not None:
            return ToolResult(
                data=EmitPaperContextOutput(ok=False, error=err),
                text=err,
                is_error=True,
            )

        # Soft checks \u2014 returned as warnings rather than failing the emit.
        for w in _collect_warnings(chosen):
            warnings.append(w)

        # Back-fill legacy fields so older consumers (PaperVerify, render_for_reproducer)
        # don't need to read the new shape.
        for exp in paper_ctx.experiments:
            exp.ensure_back_compat()

        # Persist.
        corpus = PaperCorpus(Path(ctx.workdir) / "papers")
        path = corpus.save(paper_ctx)

        # Stash on the context so same-process consumers don't need to round-trip.
        ctx.options["paper_context"] = paper_ctx
        ctx.options.setdefault("paper", {})["context"] = paper_ctx.model_dump()

        chosen_summary = (
            f"[{chosen.id}] {chosen.title}"
            + (f" on {chosen.dataset}" if chosen.dataset else "")
            + (f" \u2014 script {chosen.suggested_script}" if chosen.suggested_script else "")
            + (
                f" \u2014 metric {chosen.metric.display()}"
                if chosen.metric is not None
                else ""
            )
        )

        lines = [
            f"PaperContext saved: {path}",
            f"Chosen experiment: {chosen_summary}",
            f"Hyperparameters bound: "
            f"{len(chosen.repo_bindings)}/{len(chosen.hyperparameters)}",
        ]
        if chosen.unbound_hyperparameters:
            lines.append(
                "Unbound hyperparameters (require a code patch before running): "
                + ", ".join(chosen.unbound_hyperparameters)
            )
        for w in warnings:
            lines.append(f"warning: {w}")
        lines.append("End your turn now.")

        return ToolResult(
            data=EmitPaperContextOutput(
                ok=True,
                path=str(path),
                chosen_experiment_id=paper_ctx.chosen_experiment_id,
                experiment_count=len(paper_ctx.experiments),
                chosen_summary=chosen_summary,
                warnings=warnings,
            ),
            text="\n".join(lines),
        )


def _require_reproducibility(exp: Experiment) -> str | None:
    """Reject experiments that can't possibly be verified or run."""
    if exp.metric is None:
        return (
            f"Experiment {exp.id} has no `metric`. The reproducer needs a "
            f"headline metric with a target value to verify against."
        )
    if exp.metric.value is None:
        return (
            f"Experiment {exp.id}.metric.value is null. Set the published "
            f"numeric target (e.g. 35.6)."
        )
    if not exp.metric.paper_source:
        return (
            f"Experiment {exp.id}.metric.paper_source is empty. Record where "
            f"in the paper the number came from (e.g. 'Table 1, row SnapKV, col qasper')."
        )
    if not (exp.suggested_command or exp.suggested_script):
        return (
            f"Experiment {exp.id} has neither `suggested_command` nor "
            f"`suggested_script`. The reproducer needs something to run."
        )
    bound_names = {b.hyperparam_name for b in exp.repo_bindings}
    unbound_names = set(exp.unbound_hyperparameters)
    for hp in exp.hyperparameters:
        if hp.name not in bound_names and hp.name not in unbound_names:
            return (
                f"Hyperparameter {hp.name!r} is neither in repo_bindings nor in "
                f"unbound_hyperparameters. Every paper hyperparameter must be "
                f"accounted for \u2014 silently omitting bindings is forbidden."
            )
    return None


def _collect_warnings(exp: Experiment) -> list[str]:
    out: list[str] = []
    if not exp.model_checkpoint:
        out.append(
            f"Experiment {exp.id}.model_checkpoint is empty \u2014 the reproducer may "
            "have to guess. Prefer a concrete checkpoint id."
        )
    if not exp.dataset:
        out.append(
            f"Experiment {exp.id}.dataset is empty \u2014 specify dataset/subset "
            "(e.g. 'LongBench/qasper')."
        )
    if exp.metric is not None and exp.metric.portability == "absolute_perf":
        out.append(
            f"metric portability is 'absolute_perf' \u2014 numbers don't port across "
            "GPUs. Prefer a ratio_speedup / accuracy / quality metric if one exists."
        )
    if exp.unbound_hyperparameters:
        out.append(
            f"{len(exp.unbound_hyperparameters)} hyperparameter(s) require a code "
            "patch \u2014 the reproducer must apply the patch before DockerExec."
        )
    return out
