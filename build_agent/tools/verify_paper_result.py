"""
Deterministic paper-reproduction verifier.

This is the trusted source of truth for STAGE 2 verdicts. The configuration
agent MUST call `verify_paper_result --log <path>` before echoing any
`PAPER_RESULT_REPRODUCED` / `PAPER_RESULT_NOT_REPRODUCED` marker. The
`_handle_paper_marker` guard refuses to accept a verdict that has not been
backed by a verifier run captured in the same turn / a previous turn.

Why this exists:
- The previous flow let the LLM free-form a `PAPER_RESULT_REPRODUCED metric=X
  actual=Y expected=Z delta_pct=W` line. Nothing checked that Y appeared in
  the log, nothing checked that the metric name matched the chosen
  experiment, nothing applied the tolerance correctly, and nothing handled
  multi-metric experiments where one metric matched and another did not (the
  EARTH "RMSE better but PCC much worse" case).
- This module replaces that with a strict, side-effect-free pipeline:
    1) Read the log file from the host (NOT inside the sandbox).
    2) For each `(name, expected_value)` pair, extract every plausible actual
       value from the log via a small set of robust regexes.
    3) Compare actual vs expected with the supplied tolerance rule, taking
       direction (`higher_is_better` / `lower_is_better` / `equal`) into
       account.
    4) Emit a structured JSON block that the LLM is required to echo
       verbatim into the marker line.

The verifier never decides the verdict on its own; it returns a per-metric
verdict and a single overall `verdict` field that is `reproduced` only when
EVERY metric individually reproduces. Anything else is `not_reproduced`. The
LLM may still annotate the marker with a one-line reason, but the numeric
fields come from this verifier.
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple


_NUMBER_RE = re.compile(
    r"[-+]?\d{1,3}(?:[,_]\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?"
    r"|[-+]?\.\d+(?:[eE][-+]?\d+)?"
)

# Common "<metric_name><sep><value>[<unit>]" patterns. Captures the value.
_KV_TEMPLATES = [
    # name = 0.123 / name : 0.123
    r"(?<![A-Za-z_0-9]){name}\s*[:=]\s*({num})",
    # name is 0.123
    r"(?<![A-Za-z_0-9]){name}\s+is\s+({num})",
    # "name": 0.123  (JSON style)
    r"\"{name}\"\s*[:=]\s*({num})",
    # name (units): 0.123
    r"(?<![A-Za-z_0-9]){name}\s*\([^)]*\)\s*[:=]\s*({num})",
    # 0.123 name (number first)
    r"({num})\s+{name}(?![A-Za-z_0-9])",
]


def _aliases_for(name: str) -> List[str]:
    """Generate sensible alias spellings of a metric name to widen extraction."""
    n = (name or "").strip()
    if not n:
        return []
    seen: List[str] = []
    bases = {n, n.lower(), n.upper(), n.replace("_", " "), n.replace("-", " "),
             n.replace(" ", "_"), n.replace(" ", "-")}
    for b in bases:
        b2 = b.strip()
        if b2 and b2 not in seen:
            seen.append(b2)
    return seen


def _safe_float(s: Any) -> Optional[float]:
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        return float(s)
    if not isinstance(s, str):
        return None
    s2 = s.strip().strip(",").strip()
    s2 = s2.replace(",", "") if re.match(r"^[-+]?\d{1,3}(,\d{3})+(\.\d+)?$", s2) else s2
    try:
        return float(s2)
    except (TypeError, ValueError):
        return None


def _extract_values(log_text: str, metric_name: str) -> List[float]:
    """
    Extract every plausible numeric value tagged with `metric_name` from the
    log. Order is preserved (file order) so callers can pick the LAST hit,
    which is usually the final reported metric.
    """
    if not log_text or not metric_name:
        return []
    out: List[float] = []
    for alias in _aliases_for(metric_name):
        # Build a non-greedy regex per template/alias.
        esc = re.escape(alias)
        for tpl in _KV_TEMPLATES:
            pat = tpl.format(name=esc, num=_NUMBER_RE.pattern)
            for m in re.finditer(pat, log_text, re.IGNORECASE):
                v = _safe_float(m.group(1))
                if v is not None and not math.isnan(v) and not math.isinf(v):
                    out.append(v)
    seen: set = set()
    dedup: List[float] = []
    for v in out:
        if v not in seen:
            dedup.append(v)
            seen.add(v)
    return dedup


# ── Tolerance parsing ────────────────────────────────────────────────────────

_TOL_REL_RE = re.compile(r"(<=|<|≤|≦|<=)?\s*(\d+(?:\.\d+)?)\s*%")
_TOL_ABS_RE = re.compile(
    r"(<=|<|≤|≦|<=)?\s*(\d+(?:\.\d+)?)\s*(abs|absolute|points?|pts?)\b",
    re.IGNORECASE,
)


def _parse_tolerance(rule: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse a free-form tolerance string into (rel_pct, abs_pts).

    Examples:
      "<=15% for ratios"      -> (15.0, None)
      "<=3 abs pts"           -> (None, 3.0)
      "5% or 0.5 absolute"    -> (5.0, 0.5)
      ""                      -> (10.0, None)  # fallback default
    """
    if not rule:
        return (10.0, None)
    rel = None
    abs_ = None
    rel_m = _TOL_REL_RE.search(rule)
    if rel_m:
        try:
            rel = float(rel_m.group(2))
        except (TypeError, ValueError):
            rel = None
    abs_m = _TOL_ABS_RE.search(rule)
    if abs_m:
        try:
            abs_ = float(abs_m.group(2))
        except (TypeError, ValueError):
            abs_ = None
    if rel is None and abs_ is None:
        return (10.0, None)
    return (rel, abs_)


def _direction_signature(direction: str) -> str:
    d = (direction or "").strip().lower()
    if d in ("higher_is_better", "higher", "max", "maximize"):
        return "higher_is_better"
    if d in ("lower_is_better", "lower", "min", "minimize"):
        return "lower_is_better"
    return "equal"


def _verdict_for_metric(actual: Optional[float],
                        expected: Any,
                        rel_pct: Optional[float],
                        abs_pts: Optional[float],
                        direction: str) -> Dict[str, Any]:
    """Compute per-metric verdict. Returns a dict ready to JSON-serialise."""
    exp_f = _safe_float(expected)
    out: Dict[str, Any] = {
        "actual": actual,
        "expected": expected,
        "tolerance_rel_pct": rel_pct,
        "tolerance_abs_pts": abs_pts,
        "direction": direction,
        "delta_abs": None,
        "delta_pct": None,
        "within_tolerance": False,
        "direction_match": None,
        "verdict": "not_reproduced",
    }
    if actual is None:
        out["reason"] = "actual value not found in log"
        return out
    if exp_f is None:
        out["reason"] = "expected value is non-numeric; deterministic verifier cannot judge"
        return out

    out["delta_abs"] = actual - exp_f
    if exp_f != 0:
        out["delta_pct"] = (actual - exp_f) / abs(exp_f) * 100.0
    else:
        out["delta_pct"] = float("inf") if actual != 0 else 0.0

    within_rel = (rel_pct is None) or (
        out["delta_pct"] is not None
        and not math.isinf(out["delta_pct"])
        and abs(out["delta_pct"]) <= rel_pct
    )
    within_abs = (abs_pts is None) or (abs(out["delta_abs"]) <= abs_pts)
    out["within_tolerance"] = bool(within_rel and within_abs)

    sig = _direction_signature(direction)
    if sig == "higher_is_better":
        out["direction_match"] = bool(actual >= exp_f or out["within_tolerance"])
    elif sig == "lower_is_better":
        out["direction_match"] = bool(actual <= exp_f or out["within_tolerance"])
    else:
        out["direction_match"] = bool(out["within_tolerance"])

    out["verdict"] = "reproduced" if (out["within_tolerance"] and out["direction_match"]) else "not_reproduced"
    return out


# ── Public entrypoint (called from configuration._maybe_run_retrieval_tool) ──

def verify_paper_result(log_path: str,
                        metrics: List[Dict[str, Any]],
                        tolerance: str = "",
                        direction: str = "",
                        max_log_chars: int = 200_000) -> Tuple[str, int, Dict[str, Any]]:
    """
    Deterministic verifier.

    Args:
        log_path: absolute or repo-relative path to the experiment log
                  (typically `/repo/paper_experiment.log`).
        metrics:  list of {"name": str, "expected_value": float | str,
                            "direction"?: str, "tolerance"?: str} dicts.
        tolerance: default tolerance rule applied to every metric without
                   its own override.
        direction: default direction applied to every metric without its
                   own override.

    Returns:
        (observation_text, return_code, structured_record)
        - observation_text is what the LLM sees.
        - return_code is 0 when every metric reproduces, 1 otherwise.
        - structured_record is the JSON dict that `_handle_paper_marker`
          stores in `paper_reproduction.json["verifier"]`.
    """
    record: Dict[str, Any] = {
        "log_path": log_path,
        "log_chars_read": 0,
        "metrics": [],
        "tolerance_default": tolerance or "",
        "direction_default": direction or "",
        "verdict": "not_reproduced",
    }

    if not log_path:
        msg = "verify_paper_result: --log <path> is required."
        record["error"] = "missing log path"
        return msg, 1, record

    # Allow callers to point at a path inside the docker mount; if the host
    # copy exists side-by-side under /repo we prefer that. The dispatcher will
    # also rewrite typical /repo paths to the host location before calling.
    candidates = [log_path]
    if log_path.startswith("/repo/"):
        # Common host-side mount layout (set by the dispatcher).
        host_hint = os.environ.get("REPO2ROCM_HOST_REPO_PATH", "")
        if host_hint:
            candidates.insert(0, os.path.join(host_hint, log_path[len("/repo/"):]))

    text = ""
    used_path = ""
    last_err = ""
    for cand in candidates:
        try:
            with open(cand, "r", encoding="utf-8", errors="replace") as f:
                text = f.read(max_log_chars + 1)
            used_path = cand
            break
        except FileNotFoundError as e:
            last_err = str(e)
        except OSError as e:
            last_err = f"{type(e).__name__}: {e}"
    if not used_path:
        record["error"] = f"could not read log: {last_err or 'not found'}"
        return (
            f"verify_paper_result: could not read log at {log_path}: "
            f"{last_err or 'not found'}\n",
            1,
            record,
        )

    truncated = len(text) > max_log_chars
    if truncated:
        text = text[:max_log_chars]
    record["log_chars_read"] = len(text)
    record["log_truncated"] = truncated
    record["resolved_log_path"] = used_path

    if not metrics:
        record["error"] = "no metrics supplied"
        return (
            "verify_paper_result: no metrics supplied. Pass --metric name=value "
            "(repeatable) or rely on the dispatcher to fill from the chosen "
            "experiment.\n",
            1,
            record,
        )

    rel_def, abs_def = _parse_tolerance(tolerance)
    direction_def = _direction_signature(direction)

    all_reproduced = True
    metric_records: List[Dict[str, Any]] = []
    for spec in metrics:
        name = str(spec.get("name") or "").strip()
        if not name:
            continue
        rel_p, abs_p = (rel_def, abs_def)
        if spec.get("tolerance"):
            rel_p, abs_p = _parse_tolerance(str(spec["tolerance"]))
        m_dir = _direction_signature(str(spec.get("direction") or direction_def))
        actuals = _extract_values(text, name)
        actual = actuals[-1] if actuals else None
        v = _verdict_for_metric(
            actual=actual,
            expected=spec.get("expected_value"),
            rel_pct=rel_p,
            abs_pts=abs_p,
            direction=m_dir,
        )
        v["name"] = name
        v["candidates"] = actuals[-5:]
        if v["verdict"] != "reproduced":
            all_reproduced = False
        metric_records.append(v)

    record["metrics"] = metric_records
    record["verdict"] = "reproduced" if (all_reproduced and metric_records) else "not_reproduced"

    obs_lines: List[str] = []
    obs_lines.append(
        f"verify_paper_result: read {record['log_chars_read']:,} chars from "
        f"{used_path}{' (truncated)' if truncated else ''}"
    )
    obs_lines.append(f"overall_verdict: {record['verdict']}")
    for v in metric_records:
        actual_s = (
            f"{v['actual']:.6g}"
            if isinstance(v["actual"], (int, float)) and v["actual"] is not None
            else "MISSING"
        )
        delta_s = (
            f"delta_pct={v['delta_pct']:+.2f}%"
            if isinstance(v["delta_pct"], (int, float)) and not math.isinf(v["delta_pct"])
            else "delta_pct=NA"
        )
        obs_lines.append(
            f"  - {v['name']}: actual={actual_s} expected={v['expected']} "
            f"{delta_s} within_tol={v['within_tolerance']} "
            f"dir={v['direction']} dir_match={v['direction_match']} "
            f"-> {v['verdict']}"
        )
    obs_lines.append("")
    obs_lines.append("STRUCTURED_VERDICT_JSON:")
    obs_lines.append(json.dumps(record, default=str, indent=2))
    obs_lines.append("")
    if record["verdict"] == "reproduced":
        obs_lines.append(
            "Verifier OK -> you may now echo "
            "`PAPER_RESULT_REPRODUCED metric=<name> actual=<v> expected=<v> "
            "delta_pct=<x>` using EXACTLY the numbers above."
        )
    else:
        obs_lines.append(
            "Verifier did NOT reproduce -> you MUST echo "
            "`PAPER_RESULT_NOT_REPRODUCED <reason>` citing which metric failed "
            "and (if known) the caveat that explains it. Do NOT invent numbers."
        )

    return "\n".join(obs_lines) + "\n", (0 if record["verdict"] == "reproduced" else 1), record
