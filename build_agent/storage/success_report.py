# Copyright (2025) Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""SuccessReport: a single, numerically explainable verdict for a run.

The Configuration agent emits a `success_report` dict and embeds it in
`paper_reproduction.json`.  The schema is intentionally small so it is easy
to consume from notebooks, dashboards, or downstream learners.

The score formula matches the spec:

    overall = w_goal    * goal_score
            + w_env     * env_score
            + w_process * process_score

with the default weights `(0.6, 0.2, 0.2)`.  All sub-scores are in `[0, 1]`.

Sub-scores:
- `goal_score`      – Did we actually reproduce the chosen experiment?
                      Driven by the deterministic `verify_paper_result`
                      record (per-metric pass/fail with tolerance).
- `env_score`       – Did the runtime environment hold up?
                      Driven by Stage-1 (ROCM_ENV_VERIFIED) and the
                      observed GPU check.
- `process_score`   – Did the agent behave well regardless of outcome?
                      Driven by tool-calling discipline (did we use
                      retrieval/verification before risky actions),
                      not by the verdict.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    if den <= 0:
        return default
    return float(num) / float(den)


def _clamp01(x: float) -> float:
    if x is None:
        return 0.0
    try:
        v = float(x)
    except Exception:
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _score_from_verifier(verifier_record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Translate the per-metric verifier record into a single goal score."""
    if not verifier_record or not isinstance(verifier_record, dict):
        return {
            "score": 0.0,
            "passed_metrics": 0,
            "total_metrics": 0,
            "metric_scores": [],
            "verdict": "unknown",
            "explanation": "no verify_paper_result record present",
        }

    metrics = verifier_record.get("metric_results") or verifier_record.get("metrics") or []
    metric_scores: List[Dict[str, Any]] = []
    passed = 0
    for m in metrics:
        if not isinstance(m, dict):
            continue
        # The deterministic verifier uses {"verdict": "reproduced"|"not_reproduced"}
        # per metric. Older shape used a boolean "passed"; accept both.
        if "passed" in m:
            ok = bool(m.get("passed"))
        else:
            ok = (str(m.get("verdict", "")).lower() == "reproduced")
        if ok:
            passed += 1
        metric_scores.append({
            "name": m.get("name", ""),
            "passed": ok,
            "expected": m.get("expected") if "expected" in m else m.get("expected_value"),
            "observed": m.get("actual") if "actual" in m else m.get("observed_value"),
            "delta_pct": m.get("delta_pct"),
            "within_tolerance": m.get("within_tolerance"),
            "direction": m.get("direction"),
            "direction_match": m.get("direction_match"),
        })

    total = len(metric_scores)
    score = _safe_div(passed, total, 0.0)
    verdict = verifier_record.get("verdict", "unknown")
    explanation = verifier_record.get("observation") or verifier_record.get("summary") or ""

    return {
        "score": _clamp01(score),
        "passed_metrics": passed,
        "total_metrics": total,
        "metric_scores": metric_scores,
        "verdict": verdict,
        "explanation": explanation,
    }


def _env_score(stage1_marker_emitted: bool, gpu_check_seen: bool) -> Dict[str, Any]:
    parts = []
    if stage1_marker_emitted:
        parts.append(1.0)
    else:
        parts.append(0.0)
    if gpu_check_seen:
        parts.append(1.0)
    else:
        parts.append(0.0)
    return {
        "score": _clamp01(sum(parts) / len(parts)),
        "stage1_marker_emitted": bool(stage1_marker_emitted),
        "gpu_check_seen": bool(gpu_check_seen),
    }


def _process_score(
    tool_calls: Dict[str, int],
    outer_commands: List[Dict[str, Any]],
    turns_used: int,
) -> Dict[str, Any]:
    """Reward tool-discipline; penalise turn-spend and repeated failures.

    The score is the unweighted mean of:
      * `evidence_score`     – did we call retrieval/verify tools at all?
      * `verify_score`       – did we verify the paper before claiming success?
      * `efficiency_score`   – fewer turns = better (saturates at 70 turns).
    """
    tc = tool_calls or {}

    retrieval_hits = (
        tc.get("mem_recall", 0)
        + tc.get("paper_recall", 0)
        + tc.get("graphify_query", 0)
        + tc.get("web_search", 0)
        + tc.get("visit_url", 0)
        + tc.get("deep_research", 0)
    )
    lookup_hits = (
        tc.get("pypi_versions", 0)
        + tc.get("dockerhub_tags", 0)
    )

    # Saturating mapping: 0 → 0.0, 1 → 0.4, 3 → 0.8, ≥6 → 1.0
    def _sat(n: int) -> float:
        if n <= 0:
            return 0.0
        if n >= 6:
            return 1.0
        return min(1.0, 0.4 + (n - 1) * 0.15)

    evidence_score = _clamp01(0.5 * _sat(retrieval_hits) + 0.5 * _sat(lookup_hits))

    verify_calls = tc.get("verify_paper_result", 0)
    verify_score = 1.0 if verify_calls >= 1 else 0.0

    # Efficiency saturates at 70 turns (matches default `--max-turn`).
    if turns_used <= 0:
        efficiency_score = 1.0
    else:
        efficiency_score = _clamp01(1.0 - min(1.0, max(0, turns_used - 10) / 60.0))

    score = (evidence_score + verify_score + efficiency_score) / 3.0
    return {
        "score": _clamp01(score),
        "evidence_score": evidence_score,
        "verify_score": verify_score,
        "efficiency_score": efficiency_score,
        "turns_used": int(turns_used or 0),
        "tool_calls": dict(tc),
    }


def build_success_report(
    *,
    final_verdict: str,
    verifier_record: Optional[Dict[str, Any]],
    chosen_experiment: Optional[Dict[str, Any]],
    gpu_check_seen: bool,
    stage1_marker_emitted: bool,
    turns_used: int,
    tool_calls: Dict[str, int],
    outer_commands: List[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Assemble the SuccessReport dict.

    The output is JSON-serialisable and self-describing.  Callers should
    persist it under `paper_reproduction.json["success_report"]`.
    """
    w = {"goal": 0.6, "env": 0.2, "process": 0.2}
    if isinstance(weights, dict):
        for k in ("goal", "env", "process"):
            if k in weights:
                w[k] = float(weights[k])

    goal = _score_from_verifier(verifier_record)
    env = _env_score(stage1_marker_emitted, gpu_check_seen)
    proc = _process_score(tool_calls or {}, outer_commands or [], turns_used)

    overall = (
        w["goal"] * goal["score"]
        + w["env"] * env["score"]
        + w["process"] * proc["score"]
    )

    chosen_expected = (chosen_experiment or {}).get("expected_metric_name", "")
    return {
        "weights": w,
        "overall": _clamp01(overall),
        "final_verdict": final_verdict,
        "goal": goal,
        "env": env,
        "process": proc,
        "chosen_experiment_metric": chosen_expected,
    }
