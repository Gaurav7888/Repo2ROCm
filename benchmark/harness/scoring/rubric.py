"""
0-5 quality rubric and degradation-flag detector.

Rubric (headline integer score per (paper, approach)):

  5 - ROCM_ENV_VERIFIED + PAPER_RESULT_REPRODUCED + zero degradation flags.
  4 - ROCM_ENV_VERIFIED + PAPER_RESULT_REPRODUCED, but >=1 degradation flag.
  3 - ROCM_ENV_VERIFIED only.
  2 - Dockerfile produced and 'Generate success!' written, but no
      ROCM_ENV_VERIFIED.  (Repo2ROCm only; baseline cannot reach 2.)
  1 - Agent loop ran to completion; no Dockerfile / no ROCM_ENV_VERIFIED.
  0 - Hard failure (clone failed, container never started, timeout with no
      useful artifact).

Degradation flags (zero or more per task):
  - flash_attn_triton_amd_install
  - sdpa_fallback
  - base_image_changed
  - scale_down_engaged
  - loose_tolerance_pass
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


# --------------------------------------------------------------------------- #
# Degradation flags
# --------------------------------------------------------------------------- #

_FLASH_ATTN_TRITON_RE = re.compile(
    r"FLASH_ATTENTION_TRITON_AMD_ENABLE\s*=\s*TRUE",
    re.IGNORECASE,
)
_SDPA_RE = re.compile(
    r"""(?xi)
        attn_implementation\s*=\s*['"]?sdpa['"]?
      | scaled_dot_product_attention
      | (?:disable|remove|skip|patch).{0,40}flash[_\-\s]*attn
      | use_flash_attn\s*=\s*False
    """
)
_BASE_IMAGE_CHANGE_RE = re.compile(r"\bchange_base_image\b")
_SCALE_DOWN_RE = re.compile(
    r"""(?xi)
        \b(num_train_epochs|num_epochs|max_epochs|epochs)\s*=\s*[1-3]\b
      | \bbatch[_\-\s]*size\s*=\s*[12]\b
      | \bdataset.{0,40}\.select\(\s*range\(\s*\d{1,3}\s*\)\)
      | \bmock[_\-]?(data|dataset)\b
      | \bsmoke[_\-]?test\b
    """
)
_LOOSE_TOL_DELTA_RE = re.compile(r"delta_pct\s*=\s*([0-9]+\.?[0-9]*)")


def _has(text: str, pattern: re.Pattern) -> bool:
    return bool(pattern.search(text))


def detect_flags(record: Dict[str, Any]) -> List[str]:
    blob: str = record.get("track_blob") or ""
    repro_line: str = record.get("reproduced_line") or ""

    flags: List[str] = []
    if _has(blob, _FLASH_ATTN_TRITON_RE):
        flags.append("flash_attn_triton_amd_install")
    if _has(blob, _SDPA_RE):
        flags.append("sdpa_fallback")
    if _has(blob, _BASE_IMAGE_CHANGE_RE):
        flags.append("base_image_changed")
    if _has(blob, _SCALE_DOWN_RE):
        flags.append("scale_down_engaged")

    # Loose-tolerance: paper reproduced, but delta_pct between 5% and 25%.
    if record.get("paper_reproduced"):
        m = _LOOSE_TOL_DELTA_RE.search(repro_line) or _LOOSE_TOL_DELTA_RE.search(blob)
        if m:
            try:
                pct = abs(float(m.group(1)))
                if 5.0 < pct <= 25.0:
                    flags.append("loose_tolerance_pass")
            except Exception:
                pass

    return flags


# --------------------------------------------------------------------------- #
# 0-5 score
# --------------------------------------------------------------------------- #

def score_record(record: Dict[str, Any]) -> Tuple[int, str, List[str]]:
    """Return (score_0_5, verdict_marker, degradation_flags)."""
    flags = detect_flags(record)

    rocm_verified = bool(record.get("rocm_env_verified"))
    paper_repro = bool(record.get("paper_reproduced"))
    build_ok = bool(record.get("build_success"))
    timed_out = bool(record.get("timed_out"))
    exit_code = record.get("exit_code")

    if timed_out and not (rocm_verified or build_ok):
        return 0, "failure", flags

    if rocm_verified and paper_repro:
        if flags:
            return 4, "paper_reproduced", flags
        return 5, "paper_reproduced", flags

    if rocm_verified:
        return 3, "env_verified_only", flags

    if build_ok:
        return 2, "build_only", flags

    # No useful marker but the loop did run (we have an exit_code).
    if exit_code is not None and exit_code != 127:
        return 1, "loop_complete", flags

    return 0, "failure", flags


def score_record_full(record: Dict[str, Any]) -> Dict[str, Any]:
    """Score plus a copy of the input record's headline metrics."""
    score, marker, flags = score_record(record)
    return {
        "score_0_5": score,
        "verdict_marker": marker,
        "degradation_flags": flags,
        # Pass through metrics so downstream report.py can group/aggregate.
        "approach": record.get("approach"),
        "rocm_env_verified": bool(record.get("rocm_env_verified")),
        "paper_reproduced": bool(record.get("paper_reproduced")),
        "paper_not_reproduced": bool(record.get("paper_not_reproduced")),
        "build_success": bool(record.get("build_success")),
        "dockerfile_present": bool(record.get("dockerfile_present")),
        "timed_out": bool(record.get("timed_out")),
        "exit_code": record.get("exit_code"),
        "elapsed_s": float(record.get("elapsed_s") or 0.0),
        "prompt_tokens": int(record.get("prompt_tokens") or 0),
        "completion_tokens": int(record.get("completion_tokens") or 0),
        "total_tokens": int(record.get("total_tokens") or 0),
        "n_llm_calls": int(record.get("n_llm_calls") or 0),
        "chosen_base_image": record.get("chosen_base_image") or "",
        "success_report_overall": float(record.get("success_report_overall") or 0.0),
    }
