"""
Repo2ROCm task runner.

Invokes `python3 -u build_agent/main.py` with the same flags as the user's
example command, plus `--gpu-index <i>` so the underlying Sandbox pins itself
to the worker's GPU. Writes everything (run.log + copied artifacts) into the
caller-supplied `task_dir`.

The scheduler imports this module and calls `run_task(...)` once per task.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from typing import Any, Dict, Optional


# Repo root containing build_agent/ and benchmark/.
REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _load_task_record(tasks_json: str, paper_id: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(tasks_json):
        return None
    with open(tasks_json, "r", encoding="utf-8") as f:
        records = json.load(f)
    for r in records:
        if r.get("paper_id") == paper_id:
            return r
    return None


def _copy_if_exists(src: str, dst: str) -> bool:
    if not os.path.exists(src):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isdir(src):
        if os.path.exists(dst):
            shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst, symlinks=True, ignore_dangling_symlinks=True)
    else:
        shutil.copy2(src, dst)
    return True


def _terminate_process_group(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #

VALID_MODES = ("env", "reproduce", "full")


def run_task(*, paper_id: str, task_dir: str, gpu_index: int,
             container_name_prefix: str, timeout_s: int,
             tasks_json: str,
             llm: str = "claude-sonnet-4",
             claude_code_model: str = "sonnet",
             paper_source_mode: str = "html",
             mode: str = "full",
             extra_args: Optional[list] = None,
             **_unused) -> Dict[str, Any]:
    """Run a single Repo2ROCm task.

    `mode` selects the agent run-mode and maps to build_agent/main.py flags:
      * env       -> --mode env  (Mode 1: ROCM_ENV_VERIFIED only, scale-down OK)
      * reproduce -> --mode reproduce  (Mode 2: paper reproduce; auto no-scale-down)
      * full      -> --reproduce-results  (Mode 3: env then paper; explicit no-scale-down)

    Returns a dict like:
        {"exit_code": int, "timed_out": bool, "notes": str, "elapsed_s": float}
    """
    if mode not in VALID_MODES:
        return {"exit_code": 2, "timed_out": False,
                "notes": f"invalid mode {mode!r}; expected one of {VALID_MODES}"}

    rec = _load_task_record(tasks_json, paper_id)
    if not rec:
        return {"exit_code": 2, "timed_out": False,
                "notes": f"task record not found for {paper_id} in {tasks_json}"}

    full_name = rec.get("repo_full_name") or ""
    sha = rec.get("repo_sha") or ""
    if not full_name or not sha:
        return {"exit_code": 2, "timed_out": False,
                "notes": f"missing repo or sha for {paper_id}"}

    # Per-task root directory keeps each Repo2ROCm run's `output/`, `kb/`,
    # and `utils/repo/` isolated so parallel runs do not collide.
    # Use an absolute path so the build_agent subprocess (cwd=REPO_ROOT)
    # and the post-run copy step (harness cwd) resolve to the same place.
    root_path = os.path.abspath(os.path.join(task_dir, "root"))
    os.makedirs(root_path, exist_ok=True)

    api_key = os.environ.get("AMD_LLM_API_KEY", "")
    cmd = [
        "python3", "-u", os.path.join(REPO_ROOT, "build_agent", "main.py"),
        "--full_name", full_name,
        "--sha", sha,
        "--root_path", root_path,
        "--llm", llm,
        "--rocm",
        "--use-claude-code",
        "--claude-code-model", claude_code_model,
        "--gpu-index", str(gpu_index),
    ]
    if mode == "full":
        # Legacy "full" path: env then reproduce, with explicit no-scale-down.
        cmd.extend(["--no-scale-down", "--reproduce-results"])
    elif mode == "reproduce":
        # build_agent/main.py auto-sets no_scale_down for --mode reproduce.
        cmd.extend(["--mode", "reproduce"])
    else:
        cmd.extend(["--mode", "env"])

    # Paper source / URL only matter when the agent actually reads the paper.
    if mode in ("reproduce", "full"):
        cmd.extend(["--paper-source-mode", paper_source_mode])
        if rec.get("paper_url"):
            cmd.extend(["--paper-url", rec["paper_url"]])

    if api_key:
        cmd.extend(["--api-key", api_key])
    if extra_args:
        cmd.extend(list(extra_args))

    log_path = os.path.join(task_dir, "run.log")
    metadata = {
        "paper_id": paper_id,
        "approach": "repo2rocm",
        "mode": mode,
        "gpu_index": gpu_index,
        "container_name_prefix": container_name_prefix,
        "full_name": full_name,
        "sha": sha,
        "paper_url": rec.get("paper_url"),
        "started_at": time.time(),
        "cmd": cmd,
    }
    with open(os.path.join(task_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Provide a hint to the inner agent for nicer container labels (used by
    # learners/observers). The actual GPU isolation is done by --gpu-index
    # which propagates to Sandbox.
    env["AMD60_PAPER_ID"] = paper_id
    env["AMD60_GPU_INDEX"] = str(gpu_index)
    # Per-process Triton/Inductor caches keep parallel runs from clobbering.
    cache_root = os.path.join(task_dir, "caches")
    os.makedirs(cache_root, exist_ok=True)
    env["TRITON_CACHE_DIR"] = os.path.join(cache_root, "triton")
    env["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(cache_root, "inductor")

    timed_out = False
    exit_code: Optional[int] = None
    start = time.time()

    with open(log_path, "wb") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid,
        )
        try:
            exit_code = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_group(proc)
            exit_code = proc.poll() if proc.poll() is not None else 124

    elapsed = time.time() - start

    artifacts_root = os.path.join(task_dir, "artifacts")
    out_repo_dir = os.path.join(root_path, "output", full_name)
    copied = []
    for fname in (
        "test.txt",
        "track.json",
        "track.txt",
        "outer_commands.json",
        "inner_commands.json",
        "paper_reproduction.json",
        "plan.txt",
        "Dockerfile",
        "agent_debug_log.txt",
        "sha.txt",
    ):
        src = os.path.join(out_repo_dir, fname)
        dst = os.path.join(artifacts_root, fname)
        if _copy_if_exists(src, dst):
            copied.append(fname)

    metadata["finished_at"] = time.time()
    metadata["elapsed_s"] = round(elapsed, 2)
    metadata["exit_code"] = exit_code
    metadata["timed_out"] = timed_out
    metadata["copied_artifacts"] = copied
    with open(os.path.join(task_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    notes = (
        f"timeout after {timeout_s}s" if timed_out
        else f"exit={exit_code} artifacts={len(copied)}"
    )
    return {
        "exit_code": int(exit_code if exit_code is not None else 1),
        "timed_out": timed_out,
        "notes": notes,
        "elapsed_s": elapsed,
    }
