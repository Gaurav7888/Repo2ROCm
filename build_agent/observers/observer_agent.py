"""
Async, proactive observer sidecar.

Pipeline (per turn snapshot):
  1. State Interpreter  -> TurnState (semantic summary)
  2. Hazard Ledger      -> static risks discovered at run_started
  3. Trajectory Forecaster -> predicts the next action family + likely failures
  4. Preparation Planner   -> decides what to research now and what to defer
  5. Readiness Packs       -> compact advice the main loop consumes at turn boundaries

The main loop never imports the sidecar directly. Communication is one-way:
  - main loop appends events to observer_events.jsonl
  - sidecar appends advice rows to observer_advice.jsonl
  - main loop reads only fresh advice at safe turn boundaries
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_AGENT_ROOT = os.path.dirname(CURRENT_DIR)
if BUILD_AGENT_ROOT not in sys.path:
    sys.path.insert(0, BUILD_AGENT_ROOT)

from observers.types import (  # noqa: E402
    HazardLedger,
    HazardSignal,
    ObserverAdvice,
    ObserverEvent,
    TrajectoryForecast,
    TurnState,
    append_jsonl,
    read_jsonl_from_offset,
)


# ── Skill catalogue ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ObserverSkill:
    name: str
    description: str


_SKILLS: List[ObserverSkill] = [
    ObserverSkill("progressOK",
                  "Recent turns show healthy progress; no advice needed."),
    ObserverSkill("dependencyPreflight",
                  "Predict missing/incompatible packages before installs run."),
    ObserverSkill("modelAssetReadiness",
                  "Predict missing model/dataset assets before benchmarks launch."),
    ObserverSkill("rocmRuntimeCompatibility",
                  "Predict ROCm/HIP runtime, image, or kernel compatibility issues."),
    ObserverSkill("frameworkApiDrift",
                  "Predict drift between repo code and current framework APIs."),
    ObserverSkill("benchmarkPathing",
                  "Predict pathing/CLI issues before running benchmark scripts."),
    ObserverSkill("paperMetricPath",
                  "Predict metric/path mismatches before paper verifier runs."),
    ObserverSkill("paperExperimentFidelity",
                  "Predict deviations from paper-reported experiment fidelity."),
    ObserverSkill("repoEntrypointAlignment",
                  "Predict entrypoint/config alignment problems vs the plan."),
    ObserverSkill("repoExplorationStuck",
                  "Detect when the run is circling without convergence."),
    ObserverSkill("dependencyRepair",
                  "Reactive fix for an active dependency failure loop."),
    ObserverSkill("paperReproduction",
                  "Reactive fix when Stage 2 verifier discipline is drifting."),
]


def _skills_text() -> str:
    return "\n".join(f"- {s.name}: {s.description}" for s in _SKILLS)


# ── Helpers ──────────────────────────────────────────────────────────────────


_AMD_TERMS = ("rocm", "hip", "rocblas", "miopen", "gfx", "kfd", "amd", "amdgpu")
_DEP_TERMS = (
    "pip", "install", "wheel", "pyproject", "setup.py", "setup.cfg",
    "requirements", "version", "dependency", "deepspeed", "transformers",
    "flash_attn", "flash-attn", "xformers", "bitsandbytes", "triton",
)
_BENCHMARK_TERMS = (
    "benchmark", "eval", "predict", "pred.py", "run.py", "main.py",
    "torchrun", "python", "model_path", "tokenizer_path", "task",
)
_PAPER_TERMS = (
    "paper", "metric", "table", "appendix", "figure", "experiment",
    "tolerance", "primary_metric", "verify_paper_result",
)
_LOCAL_RETRIEVAL = (
    "graphify_query", "paper_recall", "mem_recall", "verify_paper_result",
)


def _decision_fingerprint(decision: Dict[str, Any]) -> str:
    base = {
        "profile_used": decision.get("profile_used"),
        "diagnosis": decision.get("diagnosis"),
        "recommended_strategy": decision.get("recommended_strategy"),
        "suggested_questions_or_tools": decision.get("suggested_questions_or_tools"),
        "applies_before": decision.get("applies_before"),
        "predicted_failure": decision.get("predicted_failure"),
    }
    return hashlib.sha1(json.dumps(base, sort_keys=True).encode("utf-8")).hexdigest()[:20]


def _contains_any(haystack: str, needles) -> bool:
    if not haystack:
        return False
    h = haystack.lower()
    return any(n in h for n in needles)


# ── State Interpreter ────────────────────────────────────────────────────────


class StateInterpreter:
    """Convert raw turn snapshots into a typed TurnState."""

    def interpret(self, snapshot: Dict[str, Any]) -> TurnState:
        commands_text = " ".join(str(c) for c in (snapshot.get("commands") or []))
        observation = str(snapshot.get("observation_excerpt") or "")
        return_codes = [int(rc) for rc in (snapshot.get("return_codes") or [])
                        if isinstance(rc, (int, float))]
        succeeded = bool(return_codes) and all(rc == 0 for rc in return_codes)

        action_family, action_target = self._classify_action(commands_text, observation)
        repo_areas = self._extract_repo_areas(commands_text)
        dep_signals = self._signals(commands_text + " " + observation, _DEP_TERMS)
        paper_signals = self._signals(commands_text + " " + observation, _PAPER_TERMS)
        runtime_signals = self._signals(commands_text + " " + observation, _AMD_TERMS)

        used_local = _contains_any(commands_text, _LOCAL_RETRIEVAL)
        paper_retrieval = bool(snapshot.get("paper_retrieval_used")) or _contains_any(
            commands_text, ("paper_recall",)
        )

        blocked_on = ""
        if not succeeded:
            blocked_on = self._blocked_on(observation, snapshot.get("error_class") or "")

        return TurnState(
            turn=int(snapshot.get("turn", 0) or 0),
            stage=str(snapshot.get("stage") or "stage1"),
            action_family=action_family,
            action_target=action_target,
            succeeded=succeeded,
            return_codes=return_codes,
            error_class=str(snapshot.get("error_class") or ""),
            duration_s=float(snapshot.get("duration_s") or 0.0),
            repo_areas_touched=repo_areas,
            dependency_signals=dep_signals,
            paper_signals=paper_signals,
            runtime_signals=runtime_signals,
            blocked_on=blocked_on,
            used_local_retrieval=used_local,
            paper_retrieval_used=bool(paper_retrieval),
        )

    @staticmethod
    def _classify_action(commands_text: str, observation: str) -> Tuple[str, str]:
        c = commands_text.lower()
        if not c:
            return "none", ""
        if "verify_paper_result" in c:
            return "verify", "paper_metric"
        if c.startswith("pip install") or " pip install" in c or "pip download" in c:
            return "dependency_install", _first_match(c, _DEP_TERMS)
        if "huggingface_hub" in c or "snapshot_download" in c:
            return "model_download", "hf_snapshot"
        if "torch.cuda.is_available" in c or "rocm-smi" in c:
            return "runtime_check", "gpu"
        if "graphify_query" in c or "paper_recall" in c or "mem_recall" in c:
            return "local_retrieval", _first_match(c, _LOCAL_RETRIEVAL)
        if any(tok in c for tok in ("benchmark", "pred.py", "eval.py", "torchrun")):
            return "benchmark_run", _first_match(c, _BENCHMARK_TERMS)
        if c.startswith("sed ") or " sed -i" in c or "code_edit" in c:
            return "code_patch", _first_match(c, ("sed", "code_edit", "diff"))
        if c.startswith("git ") or " git " in c:
            return "vcs", "git"
        if c.startswith("cat ") or c.startswith("ls ") or c.startswith("grep ") or c.startswith("find "):
            return "inspect", _first_match(c, ("cat", "ls", "grep", "find"))
        return "shell", c.split()[0] if c.split() else ""

    @staticmethod
    def _extract_repo_areas(commands_text: str) -> List[str]:
        areas: List[str] = []
        for match in re.findall(r"/repo/([\w\./\-]+)", commands_text):
            top = match.strip("/").split("/", 1)[0]
            if top and top not in areas:
                areas.append(top)
        return areas[:6]

    @staticmethod
    def _signals(text: str, vocab: tuple) -> List[str]:
        if not text:
            return []
        t = text.lower()
        seen: List[str] = []
        for tok in vocab:
            if tok in t and tok not in seen:
                seen.append(tok)
        return seen[:8]

    @staticmethod
    def _blocked_on(observation: str, error_class: str) -> str:
        text = (observation or "").lower()
        if "modulenotfounderror" in text:
            return "missing_python_module"
        if "no module named" in text:
            return "missing_python_module"
        if "no such file or directory" in text:
            return "missing_path_or_asset"
        if "is not a directory" in text or "filenotfounderror" in text:
            return "missing_path_or_asset"
        if "attributeerror" in text:
            return "framework_api_drift"
        if "assertionerror" in text:
            return "assertion_failure"
        if "could not find a version that satisfies" in text:
            return "pip_version_conflict"
        if error_class:
            return error_class.lower()
        return ""


def _first_match(text: str, vocab) -> str:
    t = text.lower()
    for tok in vocab:
        if tok in t:
            return tok
    return ""


# ── Hazard Ledger Builder ────────────────────────────────────────────────────


class HazardLedgerBuilder:
    """Static hazard discovery from run-context (plan, paper, repo signals)."""

    def build(self, run_context: Dict[str, Any]) -> HazardLedger:
        ledger = HazardLedger(
            repo_id=str(run_context.get("repo") or ""),
            plan_excerpt=str(run_context.get("plan_excerpt") or "")[:4000],
        )
        plan_lower = (run_context.get("plan_excerpt") or "").lower()
        repo = str(run_context.get("repo") or "")

        if any(tok in plan_lower for tok in ("flash-attn", "flash_attn",
                                              "bitsandbytes", "xformers",
                                              "deepspeed", "triton")):
            ledger.add(HazardSignal.create(
                skill="dependencyPreflight",
                title="CUDA-leaning wheels in plan",
                description=(
                    "The strategic plan references one of flash-attn / bitsandbytes / "
                    "xformers / deepspeed / triton. PyPI wheels for these are typically "
                    "CUDA-only and should not be installed unmodified on AMD ROCm. Plan "
                    "for source builds with FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE or "
                    "ROCm-native equivalents."
                ),
                triggers_when="action_family=dependency_install AND target matches "
                              "flash-attn|bitsandbytes|xformers|triton|deepspeed",
                suggested_prep=(
                    "Use pypi_versions before pinning, prefer ROCm-native or "
                    "Triton-AMD source builds, and confirm the install command "
                    "with verified compatibility evidence before retrying."
                ),
                confidence=0.7,
                evidence_refs=["plan"],
            ))

        if any(tok in plan_lower for tok in ("rocm/pytorch", "rocm/vllm",
                                              "rocm/sgl", "rocm/jax",
                                              "rocm/tensorflow")):
            ledger.add(HazardSignal.create(
                skill="rocmRuntimeCompatibility",
                title="ROCm base image may need a verified tag",
                description=(
                    "The plan selects a `rocm/*` image. Tag drift between planner "
                    "memory and Docker Hub is common. Before any change_base_image "
                    "or torch import, ensure the actual pulled tag is supported and "
                    "compatible with the repo's framework expectations."
                ),
                triggers_when="action_family=runtime_check OR command mentions "
                              "change_base_image OR torch.cuda.is_available",
                suggested_prep=(
                    "Use dockerhub_tags to confirm the tag is current and pair it "
                    "with the repo's torch/transformers expectations."
                ),
                confidence=0.55,
                evidence_refs=["plan"],
            ))

        if "model_path" in plan_lower or "snapshot_download" in plan_lower or "/models/" in plan_lower:
            ledger.add(HazardSignal.create(
                skill="modelAssetReadiness",
                title="Benchmark expects a local model checkpoint",
                description=(
                    "The plan/benchmark path references a local model directory. "
                    "Containers from prior runs may not contain that path. Predict "
                    "an asset-missing failure on first benchmark launch."
                ),
                triggers_when="action_family=benchmark_run AND target involves "
                              "model_path|/models/|snapshot_download",
                suggested_prep=(
                    "Confirm model availability or pre-download a compatible checkpoint "
                    "matching the repo's model family (e.g., Qwen2 architecture) before "
                    "kicking off the benchmark."
                ),
                confidence=0.6,
                evidence_refs=["plan"],
            ))

        if any(tok in plan_lower for tok in ("transformers", "qwen", "llama")):
            ledger.add(HazardSignal.create(
                skill="frameworkApiDrift",
                title="Custom model code may drift from current transformers",
                description=(
                    "Repos that ship their own Qwen/Llama implementations frequently "
                    "drift from upstream `transformers` (e.g., `rope_theta` vs "
                    "`rope_parameters`). Predict an AttributeError on first model "
                    "instantiation against the latest release."
                ),
                triggers_when="action_family=benchmark_run OR target imports "
                              "Qwen2ForCausalLM/LlamaForCausalLM",
                suggested_prep=(
                    "Pin transformers to the version the repo expects, or apply a "
                    "narrow getattr() patch around drifted attributes before launch."
                ),
                confidence=0.6,
                evidence_refs=["plan"],
            ))

        if bool(run_context.get("reproduce_results")):
            ledger.add(HazardSignal.create(
                skill="paperMetricPath",
                title="Stage 2 metric path must align with verifier",
                description=(
                    "Paper reproduction is enabled. Predict that benchmark output "
                    "won't match `verify_paper_result` expectations on the first try "
                    "(missing log path, wrong metric name, partial results)."
                ),
                triggers_when="action_family=benchmark_run AND stage=stage2 "
                              "OR command mentions verify_paper_result",
                suggested_prep=(
                    "Tee stdout+stderr to /repo/paper_experiment.log explicitly and "
                    "use paper_recall to lock the metric name/tolerance before "
                    "calling verify_paper_result."
                ),
                confidence=0.7,
                evidence_refs=["plan"],
            ))

        if repo:
            ledger.add(HazardSignal.create(
                skill="repoEntrypointAlignment",
                title="Entrypoint must match the chosen experiment",
                description=(
                    "Plans frequently reference scripts that don't exactly exist at "
                    "the assumed path. Predict a 'missing entrypoint' or wrong-flag "
                    "failure at the first benchmark launch."
                ),
                triggers_when="action_family=benchmark_run AND target involves a "
                              "script path",
                suggested_prep=(
                    "Use graphify_query --scope code to confirm the true entrypoint "
                    "and supported flags before running the benchmark."
                ),
                confidence=0.5,
                evidence_refs=["plan"],
            ))

        return ledger


# ── Trajectory Forecaster ────────────────────────────────────────────────────


class TrajectoryForecaster:
    """Predict the next action family and likely failures based on history."""

    def forecast(self, history: List[TurnState],
                 ledger: Optional[HazardLedger]) -> Optional[TrajectoryForecast]:
        if not history:
            return None
        last = history[-1]

        next_family, next_target = self._predict_next(history)
        predicted_failures, probability = self._predict_failures(
            history, ledger, next_family, next_target
        )
        if not predicted_failures and probability < 0.4:
            return None
        cost = "high" if last.stage == "stage2" else "medium"
        notes = ""
        if last.blocked_on:
            notes = f"Last turn blocked_on={last.blocked_on}; "
        if last.error_class:
            notes += f"error_class={last.error_class}; "
        return TrajectoryForecast.create(
            turn_seen=last.turn,
            predicted_next_action_family=next_family,
            predicted_next_target=next_target,
            predicted_failures=predicted_failures,
            failure_probability=probability,
            failure_cost=cost,
            horizon="short",
            notes=notes.strip(),
        )

    @staticmethod
    def _predict_next(history: List[TurnState]) -> Tuple[str, str]:
        last = history[-1]
        if last.action_family == "dependency_install" and not last.succeeded:
            return "dependency_install", last.action_target
        if last.action_family == "code_patch":
            # After a patch, the agent typically re-runs the broken thing.
            for prior in reversed(history[:-1]):
                if prior.action_family in {"benchmark_run", "runtime_check"}:
                    return prior.action_family, prior.action_target
            return "benchmark_run", ""
        if last.action_family == "inspect":
            # Inspections usually precede a real attempt.
            return "benchmark_run", last.action_target
        if last.action_family == "model_download":
            return "benchmark_run", last.action_target
        if last.stage == "stage2" and last.action_family == "benchmark_run":
            return "verify", "paper_metric"
        return last.action_family or "benchmark_run", last.action_target

    @staticmethod
    def _predict_failures(history: List[TurnState],
                          ledger: Optional[HazardLedger],
                          next_family: str, next_target: str) -> Tuple[List[str], float]:
        last = history[-1]
        failures: List[str] = []
        score = 0.0

        if next_family == "benchmark_run":
            if not any(h.action_family == "model_download" or "model_path" in h.action_target
                       for h in history):
                failures.append("missing_model_asset_on_benchmark_launch")
                score += 0.4
            if any(("transformers" in s) or ("qwen" in s) for h in history
                   for s in h.dependency_signals):
                failures.append("framework_api_drift_on_model_init")
                score += 0.25
            if last.stage == "stage2" and not last.paper_retrieval_used:
                failures.append("verifier_metric_path_mismatch")
                score += 0.3

        if next_family == "dependency_install":
            target = (next_target or "").lower()
            if any(t in target for t in ("flash", "bitsandbytes", "xformers", "triton", "deepspeed")):
                failures.append("cuda_only_wheel_on_amd")
                score += 0.45
            recent_installs = [h for h in history[-3:]
                               if h.action_family == "dependency_install" and not h.succeeded]
            if len(recent_installs) >= 2:
                failures.append("dependency_install_loop")
                score += 0.35

        if next_family == "verify":
            if last.stage == "stage2" and not last.paper_retrieval_used:
                failures.append("verify_paper_result_missing_metric_args")
                score += 0.4

        if last.action_family == "code_patch":
            failures.append("post_patch_regression")
            score += 0.15

        # Hazard ledger boosts probability when a static hazard matches.
        if ledger is not None and ledger.hazards:
            for hazard in ledger.hazards:
                trigger = hazard.triggers_when.lower()
                if next_family in trigger or any(
                    f in trigger for f in failures
                ):
                    score += 0.1 * float(hazard.confidence or 0.0)

        score = min(score, 0.95)
        # Deduplicate failures while preserving order.
        seen = set()
        deduped: List[str] = []
        for failure in failures:
            if failure not in seen:
                seen.add(failure)
                deduped.append(failure)
        return deduped, score


# ── Preparation Planner ──────────────────────────────────────────────────────


class PreparationPlanner:
    """Decide which forecast/hazard becomes a researched ReadinessPack now."""

    def __init__(self, sidecar: "ObserverSidecar") -> None:
        self.sidecar = sidecar

    def plan(self, history: List[TurnState], ledger: Optional[HazardLedger],
             forecast: Optional[TrajectoryForecast]) -> List[ObserverAdvice]:
        emitted: List[ObserverAdvice] = []
        last = history[-1] if history else None

        # 1. Preventive packs from forecast
        if forecast and forecast.predicted_failures:
            skill = self._skill_for_failure(forecast.predicted_failures[0],
                                            forecast.predicted_next_action_family,
                                            last)
            advice = self.sidecar.research_pack(
                skill=skill,
                kind="preventive",
                turn_seen=forecast.turn_seen,
                applies_before=forecast.predicted_next_action_family,
                predicted_failure=forecast.predicted_failures[0],
                forecast=forecast,
                hazard=None,
                priority="high" if forecast.failure_probability >= 0.6 else "normal",
            )
            if advice is not None:
                emitted.append(advice)

        # 2. Reactive packs when something is genuinely stuck
        if last and self._is_reactive_trigger(history):
            skill = self._reactive_skill(last)
            advice = self.sidecar.research_pack(
                skill=skill,
                kind="reactive",
                turn_seen=last.turn,
                applies_before=last.action_family,
                predicted_failure=last.blocked_on or last.error_class or "stalled_progress",
                forecast=None,
                hazard=None,
                priority="high",
            )
            if advice is not None:
                emitted.append(advice)

        # 3. Hazard-led prep on slow turns (no need to wait for failure)
        if ledger and last:
            for hazard in ledger.hazards:
                if self._hazard_imminent(hazard, history):
                    advice = self.sidecar.research_pack(
                        skill=hazard.skill,
                        kind="preventive",
                        turn_seen=last.turn,
                        applies_before=last.action_family or "next_turn",
                        predicted_failure=hazard.title,
                        forecast=None,
                        hazard=hazard,
                        priority="normal",
                    )
                    if advice is not None:
                        emitted.append(advice)

        return emitted

    @staticmethod
    def _skill_for_failure(failure: str, action_family: str,
                           last: Optional[TurnState]) -> str:
        f = failure.lower()
        if "model_asset" in f:
            return "modelAssetReadiness"
        if "framework_api_drift" in f or "rope_theta" in f:
            return "frameworkApiDrift"
        if "verifier" in f or "paper" in f:
            if last and last.stage == "stage2":
                return "paperMetricPath"
            return "paperExperimentFidelity"
        if "cuda_only_wheel" in f or "dependency_install_loop" in f:
            return "dependencyPreflight"
        if "post_patch_regression" in f:
            return "repoEntrypointAlignment"
        if action_family == "runtime_check":
            return "rocmRuntimeCompatibility"
        return "repoExplorationStuck"

    @staticmethod
    def _is_reactive_trigger(history: List[TurnState]) -> bool:
        recent = history[-3:]
        if len(recent) < 2:
            return False
        # repeated same error class
        errs = [h.error_class for h in recent if h.error_class]
        if len(errs) >= 2 and len(set(errs[-2:])) == 1:
            return True
        # repeated identical action targets without success
        same_target = [h for h in recent
                       if h.action_target and h.action_target == recent[-1].action_target]
        if len(same_target) >= 3 and not any(h.succeeded for h in same_target):
            return True
        # benchmark loop without retrieval discipline
        last = recent[-1]
        if last.stage == "stage2" and last.action_family in {"benchmark_run", "verify"} and not last.paper_retrieval_used:
            return True
        return False

    @staticmethod
    def _reactive_skill(last: TurnState) -> str:
        if last.stage == "stage2":
            return "paperReproduction"
        if last.action_family == "dependency_install":
            return "dependencyRepair"
        if any(tok in (last.runtime_signals or []) for tok in _AMD_TERMS):
            return "rocmRuntimeCompatibility"
        return "repoExplorationStuck"

    @staticmethod
    def _hazard_imminent(hazard: HazardSignal, history: List[TurnState]) -> bool:
        if not history:
            return False
        last = history[-1]
        trigger = (hazard.triggers_when or "").lower()
        if hazard.skill == "modelAssetReadiness" and last.action_family in {
            "inspect", "code_patch", "dependency_install"
        }:
            return True
        if hazard.skill == "frameworkApiDrift" and last.action_family in {
            "dependency_install", "code_patch"
        }:
            return True
        if hazard.skill == "paperMetricPath" and last.stage == "stage2":
            return True
        if hazard.skill == "rocmRuntimeCompatibility" and last.action_family in {
            "runtime_check", "dependency_install"
        }:
            return True
        if hazard.skill == "dependencyPreflight" and last.action_family == "dependency_install":
            return True
        if "next_turn" in trigger:
            return True
        return False


# ── Sidecar ──────────────────────────────────────────────────────────────────


class ObserverSidecar:
    def __init__(self, events_path: str, advice_path: str, llm: str,
                 poll_interval_s: float = 1.0, max_history: int = 10) -> None:
        self.events_path = events_path
        self.advice_path = advice_path
        self.llm = llm
        self.poll_interval_s = max(0.2, poll_interval_s)
        self.max_history = max(2, max_history)
        self._events_offset = 0
        self._run_context: Dict[str, Any] = {}
        self._states: List[TurnState] = []
        self._raw_snapshots: List[Dict[str, Any]] = []
        self._ledger: Optional[HazardLedger] = None
        self._done = False
        self._emitted_fingerprints: Dict[str, int] = {}
        self._emitted_hazards: Dict[str, int] = {}
        self._budget_window_s = 60.0
        self._last_research_at = 0.0
        self.interpreter = StateInterpreter()
        self.ledger_builder = HazardLedgerBuilder()
        self.forecaster = TrajectoryForecaster()
        self.planner = PreparationPlanner(self)

    # -- Event handling ------------------------------------------------------

    def _on_run_started(self, payload: Dict[str, Any]) -> None:
        self._run_context = dict(payload or {})
        self._ledger = self.ledger_builder.build(self._run_context)

    def _on_turn_snapshot(self, payload: Dict[str, Any]) -> None:
        snapshot = dict(payload or {})
        self._raw_snapshots.append(snapshot)
        self._raw_snapshots = self._raw_snapshots[-self.max_history:]
        try:
            state = self.interpreter.interpret(snapshot)
        except Exception:
            return
        self._states.append(state)
        self._states = self._states[-self.max_history:]

        forecast = self.forecaster.forecast(self._states, self._ledger)
        advice_rows = self.planner.plan(self._states, self._ledger, forecast)
        for advice in advice_rows:
            if self._should_emit(advice):
                append_jsonl(self.advice_path, advice)
                self._emitted_fingerprints[_decision_fingerprint(self._advice_to_decision(advice))] = state.turn
                if advice.hazard_id:
                    self._emitted_hazards[advice.hazard_id] = state.turn

    @staticmethod
    def _advice_to_decision(advice: ObserverAdvice) -> Dict[str, Any]:
        return {
            "profile_used": advice.profile_used,
            "diagnosis": advice.diagnosis,
            "recommended_strategy": advice.recommended_strategy,
            "suggested_questions_or_tools": advice.suggested_questions_or_tools,
            "applies_before": advice.applies_before,
            "predicted_failure": advice.predicted_failure,
        }

    def _should_emit(self, advice: ObserverAdvice) -> bool:
        if not advice:
            return False
        decision = self._advice_to_decision(advice)
        fp = _decision_fingerprint(decision)
        last_seen = self._emitted_fingerprints.get(fp, -10)
        if advice.turn_seen - last_seen <= 2:
            return False
        if advice.hazard_id:
            last_hazard = self._emitted_hazards.get(advice.hazard_id, -10)
            if advice.turn_seen - last_hazard <= 4:
                return False
        return True

    # -- Research path -------------------------------------------------------

    def _build_research_question(self, skill: str, kind: str,
                                 predicted_failure: str,
                                 applies_before: str) -> str:
        if kind == "preventive":
            return (
                f"As an external observer, you predict the executor's next action "
                f"family is `{applies_before}`. The most likely failure is "
                f"`{predicted_failure}`. Using only retrieval and external evidence, "
                f"recommend a compact strategic preparation note that lets the next "
                f"turn avoid this failure. Do NOT execute commands."
            )
        if skill == "paperReproduction":
            return (
                "Stage 2 paper reproduction is drifting. Recommend a strategic "
                "correction (metric path, paper retrieval, verifier discipline)."
            )
        if skill == "dependencyRepair":
            return (
                "The executor is in a dependency-install loop. Recommend the "
                "verified next strategy backed by external package evidence."
            )
        if skill == "rocmRuntimeCompatibility":
            return (
                "An AMD ROCm runtime issue is active. Recommend the next high-level "
                "corrective strategy backed by external evidence."
            )
        return (
            "The executor's progress is stalling. Recommend a strategic reframing "
            "of the next action backed by external evidence."
        )

    def _build_research_context(self, skill: str,
                                forecast: Optional[TrajectoryForecast],
                                hazard: Optional[HazardSignal]) -> Dict[str, Any]:
        context: Dict[str, Any] = {
            "skill": skill,
            "skills": [s.name for s in _SKILLS],
            "run_context": self._run_context,
            "recent_states": [self._state_to_dict(s) for s in self._states[-self.max_history:]],
        }
        if forecast is not None:
            context["forecast"] = {
                "predicted_next_action_family": forecast.predicted_next_action_family,
                "predicted_next_target": forecast.predicted_next_target,
                "predicted_failures": forecast.predicted_failures,
                "failure_probability": forecast.failure_probability,
                "failure_cost": forecast.failure_cost,
                "horizon": forecast.horizon,
                "notes": forecast.notes,
            }
        if hazard is not None:
            context["hazard"] = {
                "skill": hazard.skill,
                "title": hazard.title,
                "description": hazard.description,
                "triggers_when": hazard.triggers_when,
                "suggested_prep": hazard.suggested_prep,
                "confidence": hazard.confidence,
            }
        if self._ledger is not None:
            context["hazard_ledger_size"] = len(self._ledger.hazards)
        return context

    @staticmethod
    def _state_to_dict(state: TurnState) -> Dict[str, Any]:
        return {
            "turn": state.turn,
            "stage": state.stage,
            "action_family": state.action_family,
            "action_target": state.action_target,
            "succeeded": state.succeeded,
            "return_codes": state.return_codes,
            "error_class": state.error_class,
            "blocked_on": state.blocked_on,
            "duration_s": state.duration_s,
            "repo_areas_touched": state.repo_areas_touched,
            "dependency_signals": state.dependency_signals,
            "paper_signals": state.paper_signals,
            "runtime_signals": state.runtime_signals,
            "used_local_retrieval": state.used_local_retrieval,
            "paper_retrieval_used": state.paper_retrieval_used,
        }

    def research_pack(self, *, skill: str, kind: str, turn_seen: int,
                      applies_before: str, predicted_failure: str,
                      forecast: Optional[TrajectoryForecast],
                      hazard: Optional[HazardSignal],
                      priority: str) -> Optional[ObserverAdvice]:
        if not self.llm:
            return None
        # rate-limit external research to avoid flooding the LLM gateway.
        now = time.time()
        if (now - self._last_research_at) < 4.0 and kind == "preventive":
            return None
        try:
            from agents.researcher import research
        except Exception:
            return None
        question = self._build_research_question(
            skill=skill,
            kind=kind,
            predicted_failure=predicted_failure,
            applies_before=applies_before,
        )
        context = self._build_research_context(skill, forecast, hazard)
        try:
            note = research(
                question,
                llm=self.llm,
                use_cache=True,
                budget_s=20.0 if kind == "preventive" else 25.0,
                profile="observerCritic",
                context=context,
                extra_evidence=[json.dumps({
                    "predicted_failure": predicted_failure,
                    "applies_before": applies_before,
                })],
                max_search_hits=4,
                max_visits=2,
            )
        except Exception:
            return None
        self._last_research_at = time.time()

        answer = str(note.get("answer") or "").strip()
        followups = [str(x).strip() for x in (note.get("followups") or []) if str(x).strip()]
        if not answer and not followups:
            return None
        confidence = float(note.get("confidence", 0.0) or 0.0)
        if kind == "reactive" and "progress is healthy" in answer.lower() and confidence < 0.3:
            return None

        diagnosis = predicted_failure or (answer.split(".")[0].strip() if answer else "")
        suggestions = followups[:4]
        for cmd in (note.get("suggested_commands") or [])[:3]:
            cmd_s = str(cmd).strip()
            if cmd_s and cmd_s not in suggestions:
                suggestions.append(cmd_s)
        evidence = []
        for citation in (note.get("citations") or [])[:3]:
            if isinstance(citation, dict):
                title = str(citation.get("title") or "").strip()
                url = str(citation.get("url") or "").strip()
                if title or url:
                    evidence.append(f"{title[:100]} {url[:180]}".strip())

        expires = turn_seen + (3 if kind == "preventive" else 2)
        return ObserverAdvice.create(
            turn_seen=turn_seen,
            profile_used=skill,
            diagnosis=(diagnosis or "")[:220],
            recommended_strategy=answer or "",
            suggested_questions_or_tools=suggestions,
            confidence=confidence,
            evidence=evidence,
            kind=kind,
            predicted_failure=predicted_failure,
            applies_before=applies_before,
            expires_after_turn=expires,
            priority=priority,
            forecast_id=forecast.forecast_id if forecast else "",
            hazard_id=hazard.hazard_id if hazard else "",
        )

    # -- Loop ----------------------------------------------------------------

    def _handle_event(self, row: Dict[str, Any]) -> None:
        event_type = str(row.get("event_type") or "")
        payload = row.get("payload") or {}
        if event_type == "run_started":
            self._on_run_started(payload)
            return
        if event_type == "turn_snapshot":
            self._on_turn_snapshot(payload)
            return
        if event_type == "run_finished":
            self._done = True

    def run(self) -> None:
        while not self._done:
            rows, self._events_offset = read_jsonl_from_offset(
                self.events_path,
                self._events_offset,
            )
            if not rows:
                time.sleep(self.poll_interval_s)
                continue
            for row in rows:
                self._handle_event(row)
                if self._done:
                    break


# ── Client (used by main loop) ───────────────────────────────────────────────


class ObserverClient:
    def __init__(self, output_dir: str, llm: str,
                 api_key: str = "", enabled: bool = True) -> None:
        self.output_dir = output_dir
        self.llm = llm
        self.api_key = api_key or ""
        self.enabled = bool(enabled and llm)
        self.events_path = os.path.join(output_dir, "observer_events.jsonl")
        self.advice_path = os.path.join(output_dir, "observer_advice.jsonl")
        self.log_path = os.path.join(output_dir, "observer_sidecar.log")
        self._advice_offset = 0
        self._consumed_ids: set = set()
        self._proc: Optional[subprocess.Popen] = None
        self._log_handle = None

    def start(self) -> None:
        if not self.enabled:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        open(self.events_path, "w", encoding="utf-8").close()
        open(self.advice_path, "w", encoding="utf-8").close()
        self._log_handle = open(self.log_path, "a", encoding="utf-8")
        env = os.environ.copy()
        if self.api_key and not env.get("AMD_LLM_API_KEY"):
            env["AMD_LLM_API_KEY"] = self.api_key
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--events", self.events_path,
            "--advice", self.advice_path,
            "--llm", self.llm,
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=self._log_handle,
            stderr=self._log_handle,
            cwd=BUILD_AGENT_ROOT,
            env=env,
        )

    def emit_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        append_jsonl(self.events_path, ObserverEvent.create(event_type, payload))

    def consume_new_advice(self, current_turn: Optional[int] = None,
                           applies_to_action: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        rows, self._advice_offset = read_jsonl_from_offset(
            self.advice_path,
            self._advice_offset,
        )
        fresh: List[Dict[str, Any]] = []
        for row in rows:
            advice_id = str(row.get("advice_id") or "")
            if not advice_id or advice_id in self._consumed_ids:
                continue
            # Honor expires_after_turn so stale predictions don't leak in.
            expires = row.get("expires_after_turn")
            if (
                current_turn is not None
                and isinstance(expires, (int, float))
                and int(expires) >= 0
                and int(expires) < int(current_turn)
            ):
                self._consumed_ids.add(advice_id)
                continue
            self._consumed_ids.add(advice_id)
            fresh.append(row)
        # Optional filtering: prefer rows that match the action family the
        # main loop is about to execute.
        if applies_to_action and fresh:
            preferred = [r for r in fresh
                         if not r.get("applies_before")
                         or str(r.get("applies_before")).lower() == applies_to_action.lower()]
            if preferred:
                fresh = preferred
        # Sort by priority then by turn_seen so high-priority preventive packs
        # are surfaced first.
        priority_rank = {"high": 0, "normal": 1, "low": 2}
        fresh.sort(key=lambda r: (
            priority_rank.get(str(r.get("priority") or "normal"), 1),
            -int(r.get("turn_seen") or 0),
        ))
        return fresh

    def shutdown(self) -> None:
        if self.enabled:
            self.emit_event("run_finished", {})
        if self._proc is not None:
            try:
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.terminate()
            self._proc = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None


# ── CLI entrypoint (used when started by ObserverClient.start) ───────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repo2ROCm observer sidecar")
    parser.add_argument("--events", required=True)
    parser.add_argument("--advice", required=True)
    parser.add_argument("--llm", required=True)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--max-history", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    sidecar = ObserverSidecar(
        events_path=args.events,
        advice_path=args.advice,
        llm=args.llm,
        poll_interval_s=args.poll_interval,
        max_history=args.max_history,
    )
    sidecar.run()


if __name__ == "__main__":
    main()
