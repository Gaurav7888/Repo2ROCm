"""
Extract a flat record per (paper, approach) from the raw run artifacts.

Inputs (per-task `task_dir`):

  Repo2ROCm tasks
    - artifacts/test.txt                  (markers: ROCM_ENV_VERIFIED, PAPER_RESULT_*)
    - artifacts/paper_reproduction.json   (success_report + verdict)
    - artifacts/outer_commands.json       (per-LLM-call usage + timings)
    - artifacts/track.json                (full conversation history)
    - artifacts/track.txt                 ("Generate success!" indicator)
    - artifacts/Dockerfile                (existence => env build done)

  Claude-CLI tasks
    - claude_summary.json                 (markers, usage, num_turns)
    - claude_response.json                (raw stdout)
    - run.log                             (clone + git log)

The extractor returns a dict with ALL fields we know about; missing
artifacts produce None / 0 / [] rather than crashing, so partial runs are
still scoreable.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _safe_load(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return json.load(f)
    except Exception:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception:
            return None


def _read_text(path: str) -> str:
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# Repo2ROCm
# --------------------------------------------------------------------------- #

def _sum_outer_commands_usage(outer: Any) -> Dict[str, int]:
    out = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
           "n_llm_calls": 0, "gpt_time_s": 0.0}
    if not isinstance(outer, list):
        return out
    for entry in outer:
        if not isinstance(entry, dict):
            continue
        usage = entry.get("usage")
        if isinstance(usage, dict):
            out["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
            out["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
            out["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
            out["n_llm_calls"] += 1
        if "GPT_time" in entry:
            try:
                out["gpt_time_s"] += float(entry["GPT_time"])
            except Exception:
                pass
    if out["total_tokens"] == 0:
        out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]
    return out


def _track_text(track: Any) -> str:
    """Flatten track.json content into one searchable string.

    Only `role=assistant` turns are included. User-role messages contain
    injected prompt context (env reminders, memory-provider advisories,
    seeded [CAUSAL] transitions, etc.) that would false-positive-match
    the rubric's marker regexes (ROCM_ENV_VERIFIED, PAPER_RESULT_*,
    FLASH_ATTENTION_TRITON_AMD_ENABLE, ...) if grepped over the whole
    conversation.
    """
    if track is None:
        return ""
    if isinstance(track, str):
        return track
    try:
        if isinstance(track, list):
            parts = []
            for msg in track:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    parts.append(str(msg.get("content", "")))
            return "\n".join(parts)
        return json.dumps(track, ensure_ascii=False)
    except Exception:
        return str(track)


def _detect_chosen_image(track_blob: str, dockerfile_text: str,
                         repro: Optional[Dict[str, Any]]) -> str:
    # 1) Dockerfile FROM line is most reliable
    if dockerfile_text:
        m = re.search(r"^\s*FROM\s+(\S+)", dockerfile_text, re.MULTILINE)
        if m:
            return m.group(1)
    # 2) `change_base_image` invocation in the trajectory
    m = re.search(r"change_base_image\s+(\S+)", track_blob)
    if m:
        return m.group(1)
    # 3) Anything that looks like a rocm/* image reference
    m = re.search(r"\b(rocm/[A-Za-z0-9_.\-]+(?::[A-Za-z0-9_.\-]+)?)", track_blob)
    if m:
        return m.group(1)
    return ""


def extract_repo2rocm(task_dir: str) -> Dict[str, Any]:
    art = os.path.join(task_dir, "artifacts")
    metadata = _safe_load(os.path.join(task_dir, "metadata.json")) or {}
    test_txt = _read_text(os.path.join(art, "test.txt"))
    track_json = _safe_load(os.path.join(art, "track.json"))
    track_txt = _read_text(os.path.join(art, "track.txt"))
    outer_json = _safe_load(os.path.join(art, "outer_commands.json"))
    repro_json = _safe_load(os.path.join(art, "paper_reproduction.json"))
    dockerfile = _read_text(os.path.join(art, "Dockerfile"))

    track_blob = _track_text(track_json)
    usage = _sum_outer_commands_usage(outer_json)

    rocm_verified = "ROCM_ENV_VERIFIED" in test_txt or "ROCM_ENV_VERIFIED" in track_blob
    paper_reproduced = "PAPER_RESULT_REPRODUCED" in test_txt or "PAPER_RESULT_REPRODUCED" in track_blob
    paper_not_reproduced = (
        "PAPER_RESULT_NOT_REPRODUCED" in test_txt
        or "PAPER_RESULT_NOT_REPRODUCED" in track_blob
    )
    build_success = ("Generate success!" in track_txt) or bool(dockerfile.strip())

    success_report: Dict[str, Any] = {}
    verdict_marker_extra: Dict[str, Any] = {}
    if isinstance(repro_json, dict):
        success_report = repro_json.get("success_report") or {}
        verdict_marker_extra = {
            "repro_verdict": repro_json.get("verdict"),
            "reproduced_line": repro_json.get("reproduced_line"),
            "not_reproduced_line": repro_json.get("not_reproduced_line"),
        }

    return {
        "approach": "repo2rocm",
        "exit_code": metadata.get("exit_code"),
        "timed_out": bool(metadata.get("timed_out")),
        "elapsed_s": float(metadata.get("elapsed_s") or 0.0),
        "rocm_env_verified": rocm_verified,
        "paper_reproduced": paper_reproduced,
        "paper_not_reproduced": paper_not_reproduced,
        "build_success": build_success,
        "dockerfile_present": bool(dockerfile.strip()),
        "chosen_base_image": _detect_chosen_image(track_blob, dockerfile, repro_json),
        "success_report_overall": float((success_report or {}).get("overall") or 0.0),
        "success_report_goal": float(((success_report or {}).get("goal") or {}).get("score") or 0.0),
        "success_report_env": float(((success_report or {}).get("env") or {}).get("score") or 0.0),
        "success_report_process": float(((success_report or {}).get("process") or {}).get("score") or 0.0),
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "n_llm_calls": usage["n_llm_calls"],
        "gpt_time_s": round(usage["gpt_time_s"], 2),
        "track_blob": track_blob,
        **verdict_marker_extra,
    }


# --------------------------------------------------------------------------- #
# Claude CLI baseline
# --------------------------------------------------------------------------- #

def extract_claude_cli(task_dir: str) -> Dict[str, Any]:
    metadata = _safe_load(os.path.join(task_dir, "metadata.json")) or {}
    summary = _safe_load(os.path.join(task_dir, "claude_summary.json")) or {}
    run_log = _read_text(os.path.join(task_dir, "run.log"))

    final_text = summary.get("final_text") or ""
    markers = summary.get("markers") or {}
    usage = summary.get("usage") or {}

    # The marker contract is identical to Repo2ROCm so the rubric stays uniform.
    rocm_verified = bool(markers.get("rocm_env_verified")) or "ROCM_ENV_VERIFIED" in final_text
    paper_reproduced = bool(markers.get("paper_reproduced")) or "PAPER_RESULT_REPRODUCED" in final_text
    paper_not_reproduced = bool(markers.get("paper_not_reproduced")) or "PAPER_RESULT_NOT_REPRODUCED" in final_text

    # The baseline doesn't produce a Dockerfile of its own, but if it left a
    # working container around we count that as build_success when ROCM_ENV_VERIFIED
    # was emitted.
    build_success = rocm_verified

    return {
        "approach": "claude_cli",
        "exit_code": metadata.get("exit_code"),
        "timed_out": bool(metadata.get("timed_out") or summary.get("timed_out")),
        "elapsed_s": float(metadata.get("elapsed_s") or 0.0),
        "rocm_env_verified": rocm_verified,
        "paper_reproduced": paper_reproduced,
        "paper_not_reproduced": paper_not_reproduced,
        "build_success": build_success,
        "dockerfile_present": False,
        "chosen_base_image": _detect_chosen_image(final_text + "\n" + run_log, "", None),
        "success_report_overall": 0.0,
        "success_report_goal": 0.0,
        "success_report_env": 0.0,
        "success_report_process": 0.0,
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
        "n_llm_calls": int(summary.get("num_turns") or 0),
        "gpt_time_s": float(metadata.get("elapsed_s") or 0.0),
        "track_blob": final_text + "\n" + run_log,
        "is_error": bool(summary.get("is_error")),
        "terminal_reason": summary.get("terminal_reason"),
    }


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #

def extract(task_dir: str, approach: str) -> Optional[Dict[str, Any]]:
    if not os.path.isdir(task_dir):
        return None
    if approach == "repo2rocm":
        return extract_repo2rocm(task_dir)
    if approach == "claude_cli":
        return extract_claude_cli(task_dir)
    return None
