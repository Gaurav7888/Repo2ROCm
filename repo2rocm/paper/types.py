"""Typed contracts for paper research.



The pipeline is intentionally LLM-driven (see `agents/builtin/paper_research.py`
and the `paper_*` skills). These types are the schema the agent must populate
via `EmitPaperContext` so the downstream `paper-reproducer` agent and the
`PaperVerify` tool can consume a fully-specified, reproducible experiment.

Design rule: every numeric claim must carry its `paper_source` (where in the
paper it came from) and every hyperparameter must carry a `RepoBinding` (how to
set it via the repo's actual config surface). If neither is available, the
agent must record it in `unbound_hyperparameters` instead of fabricating one.
"""
from __future__ import annotations

import re
import shlex
from typing import Literal

from pydantic import BaseModel, Field

MetricClass = Literal["ratio_speedup", "accuracy", "quality", "absolute_perf", "other"]

RuntimeClass = Literal["smoke", "short", "medium", "long", "unknown"]
_RUNTIME_MIN = {
    "smoke": 2,
    "short": 15,
    "medium": 90,
    "long": 480,
    "unknown": 60,
}


def runtime_class_to_min(rc: str) -> int:
    return _RUNTIME_MIN.get(rc, 60)


# ── Command parsing (used by the docker reproducer-mode drift guard) ───────


class CommandSpec(BaseModel):
    """Normalized shell command shape \u2014 launcher + script + flag dict.

    Parsed from a plain shell string via `CommandSpec.from_command(...)`.
    Used by the DockerExec drift guard to confirm the reproducer didn't
    silently swap script / flags vs. the chosen experiment.
    """

    raw: str = ""
    launcher: str = ""
    script: str = ""
    args: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_command(cls, command: str) -> "CommandSpec | None":
        raw = re.sub(r"\s+", " ", command or "").strip()
        if not raw:
            return None
        try:
            tokens = shlex.split(raw)
        except ValueError:
            tokens = raw.split()
        if not tokens:
            return None

        launcher = ""
        args: dict[str, str] = {}
        i = 0
        # Skip env-var prefixes ("CUDA_VISIBLE_DEVICES=0").
        while i < len(tokens) and re.match(r"[A-Za-z_][A-Za-z0-9_]*=.*", tokens[i]):
            i += 1
        if i >= len(tokens):
            return None

        if tokens[i : i + 2] == ["accelerate", "launch"]:
            launcher = "accelerate launch"
            i += 2
        elif tokens[i : i + 1] == ["torchrun"]:
            launcher = "torchrun"
            i += 1
        elif tokens[i : i + 2] == ["python", "-m"] and len(tokens) >= i + 3:
            launcher = "python -m"
            return cls(raw=raw, launcher=launcher, script=tokens[i + 2], args={})
        elif tokens[i] in {"python", "python3", "bash", "sh"}:
            launcher = tokens[i]
            i += 1

        # Launcher-level flags (accelerate/torchrun) before the script.
        while (
            i < len(tokens)
            and tokens[i].startswith("-")
            and launcher in {"accelerate launch", "torchrun"}
        ):
            key = tokens[i]
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                args[key] = tokens[i + 1]
                i += 2
            else:
                args[key] = "true"
                i += 1

        script = ""
        while i < len(tokens):
            tok = tokens[i]
            if _looks_like_script_token(tok):
                script = tok.lstrip("./")
                i += 1
                break
            if not script and launcher in {"python", "python3", "bash", "sh", ""}:
                script = tok.lstrip("./")
                i += 1
                break
            i += 1
        if not script:
            return None

        while i < len(tokens):
            tok = tokens[i]
            if tok.startswith("-"):
                if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                    args[tok] = tokens[i + 1]
                    i += 2
                else:
                    args[tok] = "true"
                    i += 1
            else:
                i += 1

        return cls(raw=raw, launcher=launcher, script=script, args=args)


def _looks_like_script_token(token: str) -> bool:
    low = token.lower()
    return (
        low.endswith((".py", ".sh"))
        or low.startswith("./")
        or "/" in token
    )


# ── Metric records ─────────────────────────────────────────────────────────


class MetricRow(BaseModel):
    """A simple (name, value, unit) tuple. Kept for back-compat with
    `PaperMetadata.headline_metrics` and `PaperVerify`'s metric specs.

    For new code prefer `MetricDefinition`, which carries provenance.
    """

    name: str
    value: float | None = None
    unit: str = ""
    is_baseline: bool = False
    portability: MetricClass = "other"
    default_tolerance: float = 0.15
    raw_text: str = ""
    dataset: str = ""
    method: str = ""

    def display(self) -> str:
        v = f"{self.value:g}" if self.value is not None else "?"
        ds = f" on {self.dataset}" if self.dataset else ""
        return f"{self.name}={v}{self.unit}{ds} ({self.portability})"


class MetricDefinition(BaseModel):
    """A reproducibility-grade metric record.

    The reproducer compares the measured value against `value` within
    `default_tolerance`. The verifier refuses to accept a MetricSpec whose
    `expected_value` doesn't match `value` in the chosen experiment.
    """

    name: str = Field(..., description="Metric name as the paper reports it, e.g. 'qasper_f1'.")
    value: float = Field(..., description="The published target value to compare against.")
    unit: str = ""
    portability: MetricClass = "other"
    default_tolerance: float = 0.15
    is_baseline: bool = False
    paper_source: str = Field(
        "",
        description=(
            "Where this number lives in the paper, verbatim. "
            "Example: 'Table 1, row \"SnapKV\", column \"qasper\"' or '\u00a74.2 \u00b63'."
        ),
    )
    repo_eval_source: str = Field(
        "",
        description=(
            "Where the measured value is produced in the repo, verbatim. "
            "Example: 'experiments/LongBench/eval.py:42 scorer_token_f1'."
        ),
    )
    notes: str = ""

    def display(self) -> str:
        v = f"{self.value:g}{self.unit}"
        return f"{self.name}={v} ({self.portability}, tol={self.default_tolerance:g})"

    def as_metric_row(self) -> MetricRow:
        """Down-convert to the legacy MetricRow so existing consumers
        (PaperVerify, PaperRecall rendering) keep working."""
        return MetricRow(
            name=self.name,
            value=self.value,
            unit=self.unit,
            is_baseline=self.is_baseline,
            portability=self.portability,
            default_tolerance=self.default_tolerance,
            raw_text=self.paper_source,
        )


# ── Hyperparameters and repo bindings ──────────────────────────────────────


class Hyperparameter(BaseModel):
    """A single knob the paper specifies for this experiment row.

    `value` is always serialized as a string; the consumer parses it for the
    CLI / config-file form the script expects. This lets us round-trip lists,
    paths, and enum-like values without pinning a type.
    """

    name: str = Field(..., description="Paper-side name, e.g. 'max_capacity_prompt'.")
    value: str = Field(..., description="Value verbatim from the paper, as a string.")
    unit: str = ""
    paper_source: str = Field(
        "",
        description="Where this hyperparameter appears, e.g. 'Appendix B, Setup' or 'Table 3 caption'.",
    )
    notes: str = ""


class RepoBinding(BaseModel):
    """How a paper-side hyperparameter maps to the repo's actual config surface."""

    hyperparam_name: str = Field(..., description="Must match a Hyperparameter.name.")
    kind: Literal["cli_flag", "json_key", "yaml_key", "constant", "env_var", "code_patch"] = Field(
        ...,
        description=(
            "How the value is set in the repo. `code_patch` means the value is "
            "currently hardcoded and the reproducer must patch the source."
        ),
    )
    location: str = Field(
        ...,
        description=(
            "Concrete location, verbatim. Examples:\n"
            "  - 'experiments/LongBench/pred_snap.py --max_capacity_prompt'\n"
            "  - 'experiments/LongBench/config/run.json::max_capacity_prompt'\n"
            "  - 'snapkv/monkeypatch/snapkv_utils.py:88 window_size'"
        ),
    )
    default: str = Field("", description="Default value at that location, if observed.")
    notes: str = ""


# ── Experiment + context ───────────────────────────────────────────────────


class Experiment(BaseModel):
    """One concrete, reproducible experiment.

    The agent populates this via `EmitPaperContext`. The reproducer agent
    consumes it via `PaperRecall` and runs it via `DockerExec`, then verifies
    with `PaperVerify`.
    """

    id: str = Field(..., description="Stable id; the reproducer keys off this.")
    title: str
    description: str = ""

    # Identity of the experiment
    model_checkpoint: str = Field(
        "",
        description="Concrete model checkpoint, e.g. 'mistralai/Mistral-7B-Instruct-v0.2'.",
    )
    dataset: str = Field(
        "",
        description=(
            "Dataset and subset, fully specified. Example: 'LongBench/qasper' "
            "or 'NIAH/16k-depth-50%'."
        ),
    )
    prompt_template: str = Field(
        "",
        description="Prompt or chat template used by the paper, verbatim if reproducible.",
    )

    # Reproducibility spec
    metric: MetricDefinition | None = Field(
        None, description="The single headline metric to verify against."
    )
    related_metrics: list[MetricDefinition] = Field(
        default_factory=list,
        description="Other metrics from the same run (e.g. the FullKV baseline row).",
    )
    hyperparameters: list[Hyperparameter] = Field(
        default_factory=list,
        description="Every knob the paper specifies for this row. Must include provenance.",
    )
    repo_bindings: list[RepoBinding] = Field(
        default_factory=list,
        description="One binding per Hyperparameter that the repo can express.",
    )
    unbound_hyperparameters: list[str] = Field(
        default_factory=list,
        description=(
            "Hyperparameter names the paper specifies but the repo does NOT "
            "currently expose. Non-empty means the reproducer must apply a "
            "code patch before running."
        ),
    )

    # Execution
    suggested_script: str = Field(
        "",
        description="Repo-relative path to the entry-point script, e.g. 'experiments/LongBench/pred_snap.py'.",
    )
    suggested_command: str = Field(
        "",
        description=(
            "Full shell command, fully bound: launcher + script + every flag with "
            "its paper-specified value. The reproducer runs this verbatim."
        ),
    )
    runtime_class: RuntimeClass = "unknown"
    estimated_runtime_min: int = 0

    # Scoring + rationale (agent-supplied; no scoring formula in code)
    portability_score: float = 0.0
    repo_match_confidence: float = 0.0
    rationale: str = Field(
        "",
        description=(
            "Free-text rationale for why this experiment was chosen. Should cite "
            "the relevant skill (e.g. portability class, runtime cost, binding fit)."
        ),
    )

    # Back-compat surface so PaperVerify / PaperRecall / older renderers keep
    # working without churn. Auto-populated from `metric` if not set explicitly.
    headline_metric: MetricRow | None = None
    code_available: bool = False

    def ensure_back_compat(self) -> None:
        """Populate legacy fields from the new ones if the agent didn't."""
        if self.headline_metric is None and self.metric is not None:
            self.headline_metric = self.metric.as_metric_row()
        if not self.code_available:
            self.code_available = bool(self.suggested_script or self.suggested_command)
        if not self.estimated_runtime_min:
            self.estimated_runtime_min = runtime_class_to_min(self.runtime_class)


class PaperMetadata(BaseModel):
    arxiv_id: str = ""
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    hardware_claimed: list[str] = Field(default_factory=list)
    libraries: list[str] = Field(default_factory=list)
    headline_metrics: list[MetricRow] = Field(
        default_factory=list,
        description="Back-compat: a flat bag of metrics. New code should look at Experiment.metric.",
    )
    pdf_path: str = ""
    text_path: str = ""
    html_path: str = ""


class PaperContext(BaseModel):
    """Output of the paper-research agent. Consumed by paper-reproducer."""

    metadata: PaperMetadata
    experiments: list[Experiment] = Field(default_factory=list)
    chosen_experiment_id: str = ""

    def chosen(self) -> Experiment | None:
        for e in self.experiments:
            if e.id == self.chosen_experiment_id:
                return e
        return None

    def render_for_reproducer(self) -> str:
        lines = ["# Paper Context", f"Title: {self.metadata.title}"]
        if self.metadata.arxiv_id:
            lines.append(f"arXiv: {self.metadata.arxiv_id}")
        if self.metadata.hardware_claimed:
            lines.append(f"HW (paper): {', '.join(self.metadata.hardware_claimed)}")
        if self.metadata.libraries:
            lines.append(f"Libraries: {', '.join(self.metadata.libraries)}")
        chosen = self.chosen()
        if chosen:
            chosen.ensure_back_compat()
            lines.append("")
            lines.append(f"## Chosen experiment: {chosen.id} \u2014 {chosen.title}")
            if chosen.model_checkpoint:
                lines.append(f"Model: {chosen.model_checkpoint}")
            elif chosen.description:
                lines.append(chosen.description)
            if chosen.dataset:
                lines.append(f"Dataset: {chosen.dataset}")
            if chosen.runtime_class and chosen.runtime_class != "unknown":
                lines.append(
                    f"Runtime class: {chosen.runtime_class} "
                    f"(~{chosen.estimated_runtime_min} min)"
                )
            if chosen.metric is not None:
                lines.append(f"Headline metric: {chosen.metric.display()}")
                if chosen.metric.paper_source:
                    lines.append(f"  source: {chosen.metric.paper_source}")
                if chosen.metric.repo_eval_source:
                    lines.append(f"  repo eval: {chosen.metric.repo_eval_source}")
            if chosen.related_metrics:
                lines.append("Related metrics (same run):")
                for m in chosen.related_metrics:
                    lines.append(f"  - {m.display()}  [{m.paper_source}]")
            if chosen.hyperparameters:
                lines.append("Hyperparameters (paper-side):")
                for hp in chosen.hyperparameters:
                    extra = f" [{hp.paper_source}]" if hp.paper_source else ""
                    unit = hp.unit or ""
                    lines.append(f"  - {hp.name} = {hp.value}{unit}{extra}")
            if chosen.repo_bindings:
                lines.append("Repo bindings (paper hyperparam \u2192 repo knob):")
                for b in chosen.repo_bindings:
                    default = f" (default={b.default})" if b.default else ""
                    lines.append(f"  - {b.hyperparam_name}: {b.kind} @ {b.location}{default}")
            if chosen.unbound_hyperparameters:
                lines.append(
                    "Unbound hyperparameters (NOT exposed by the repo \u2014 "
                    "reproducer must patch before running):"
                )
                for name in chosen.unbound_hyperparameters:
                    lines.append(f"  - {name}")
            if chosen.prompt_template:
                lines.append("Prompt template (verbatim):")
                lines.append("```")
                lines.append(chosen.prompt_template)
                lines.append("```")
            if chosen.suggested_script:
                lines.append(f"Suggested script: {chosen.suggested_script}")
            if chosen.suggested_command:
                lines.append(f"Suggested command: $ {chosen.suggested_command}")
            if chosen.rationale:
                lines.append(f"Rationale: {chosen.rationale}")
        if len(self.experiments) > 1:
            lines.append("")
            lines.append("## Other candidates (in case the chosen one is infeasible)")
            for e in self.experiments[:6]:
                if e.id == self.chosen_experiment_id:
                    continue
                m = e.metric.display() if e.metric else "(no metric)"
                bits = [f"  - [{e.id}] {e.title}", f"| {m}"]
                if e.suggested_script:
                    bits.append(f"| script={e.suggested_script}")
                if e.portability_score:
                    bits.append(f"| score={e.portability_score:g}")
                lines.append(" ".join(bits))
        return "\n".join(lines)
