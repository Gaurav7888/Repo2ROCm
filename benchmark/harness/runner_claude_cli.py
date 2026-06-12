"""
Claude-CLI baseline runner.

Drives a one-shot agentic Claude CLI session per paper. The CLI is given
Bash + Read + Edit tools and a system prompt that contains:

  * a compact ROCm knowledge dump (image catalog, CUDA->ROCm mapping,
    banned NVIDIA packages, code patterns, pre-installed lists), and
  * a deterministic workflow derived from rocm_knowledge.py:
       1. clone repo, choose ROCm image from the catalog,
       2. start a single ROCm container pinned to the worker's GPU,
       3. install deps with the documented fallbacks
          (flash-attn -> Triton-AMD; if that fails -> SDPA patch),
       4. echo `ROCM_ENV_VERIFIED` after a successful smoke run,
       5. run the shortest paper experiment from the PDF,
       6. echo `PAPER_RESULT_REPRODUCED ...` or `PAPER_RESULT_NOT_REPRODUCED ...`.

This is the cheapest possible baseline that still gets `rocm_knowledge.py`
in the loop. It does NOT use Repo2ROCm's planner, KB, observer, waiting
list, conflict list, or trajectory store.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional


HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "build_agent"))


# --------------------------------------------------------------------------- #
# ROCm knowledge digest (compact prompt-friendly view of rocm_knowledge.py)
# --------------------------------------------------------------------------- #

def _build_rocm_knowledge_digest() -> str:
    """Render a compact, prompt-friendly subset of rocm_knowledge.py."""
    try:
        from knowledge.rocm_knowledge import (
            ROCM_IMAGE_CATALOG,
            ROCM_PREINSTALLED_PACKAGES,
            CUDA_TO_ROCM_MAPPING,
            BANNED_NVIDIA_PACKAGES,
            CUDA_CODE_PATTERNS,
        )
    except Exception as e:
        return f"# rocm_knowledge import failed: {e}"

    lines: List[str] = ["# ROCm Knowledge Base (digest)"]
    lines.append("\n## Image catalog (pick one as base image)")
    for key, info in ROCM_IMAGE_CATALOG.items():
        tags = ", ".join(info.get("tags", []) or [])
        lines.append(
            f"- {key}: image={info.get('image')} default_tag={info.get('default_tag')} "
            f"tags=[{tags}]\n  use_when: {info.get('description', '').strip()}"
        )

    lines.append("\n## Pre-installed packages (do NOT reinstall these per image)")
    for img, pkgs in ROCM_PREINSTALLED_PACKAGES.items():
        lines.append(f"- {img}: {', '.join(pkgs)}")

    lines.append("\n## CUDA -> ROCm package mapping (with fallbacks)")
    for cuda_pkg, info in CUDA_TO_ROCM_MAPPING.items():
        rocm_pkg = info.get("rocm_package") or "(no direct equivalent)"
        cmd = info.get("install_cmd") or ""
        notes = (info.get("notes") or "").strip()
        lines.append(f"- {cuda_pkg} -> {rocm_pkg}")
        if cmd:
            lines.append(f"  install: {cmd}")
        if notes:
            lines.append(f"  notes: {notes}")

    lines.append("\n## Banned NVIDIA packages (NEVER install on ROCm)")
    lines.append(", ".join(BANNED_NVIDIA_PACKAGES))

    lines.append("\n## CUDA code-pattern equivalents")
    for key, info in CUDA_CODE_PATTERNS.items():
        lines.append(
            f"- {info.get('cuda_pattern')} -> {info.get('rocm_replacement')}: "
            f"{info.get('notes', '').strip()}"
        )

    return "\n".join(lines)


_BASELINE_SYSTEM_TEMPLATE = """You are a ROCm-migration baseline agent. Your job is to take a single
GitHub repository and reproduce one experiment from its companion paper on
an AMD ROCm GPU.

You have access to Bash, Read, Edit, Glob, Grep. Network is available.
You may freely run `docker`, `git`, `pip`, etc. on the host. The host has
8 ROCm GPUs (indices 0..7). You MUST pin all GPU work to the GPU index
provided in the task description by setting HIP_VISIBLE_DEVICES (and
ROCR_VISIBLE_DEVICES) on every `docker run`.

Workflow (follow in order; do not skip steps):

1. Read the README of the cloned repo and the paper PDF.
2. Pick a base image from the IMAGE CATALOG below. Use the catalog's
   `use_when` hints. Default to rocm/pytorch:latest if uncertain.
3. Start ONE long-running container, named exactly as instructed in the
   task description, with /dev/kfd and /dev/dri devices, group_add=video,
   shm_size=8g, network=host, the repo mounted at /repo, paper.pdf mounted
   read-only at /repo/paper.pdf, and HIP_VISIBLE_DEVICES set to the
   assigned GPU index. From here on, drive that container with
   `docker exec -w /repo <name> bash -c '...'`.
4. Install dependencies. Honor the CUDA -> ROCm mapping below and
   the BANNED NVIDIA package list (NEVER install those). For
   `flash-attn`: try the Dao-AILab Triton-AMD recipe (FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE);
   if that fails, fall back to patching the source to use
   `attn_implementation="sdpa"` and disabling flash-attn imports.
5. Run the project's main script (or a minimal smoke test) on the GPU.
   Verify with `rocm-smi` and `torch.cuda.is_available()` that the GPU
   is actually being used. When the environment is fully working, echo:
       echo ROCM_ENV_VERIFIED
   exactly once.
6. Identify the SHORTEST experiment in the paper whose code is present
   in the repo. Run it with the EXACT paper/README config (no scale-down,
   no mock data). Capture stdout to /repo/paper_experiment.log.
7. Compare the produced metric to the paper's reported value. Then echo
   exactly ONE of:
       echo PAPER_RESULT_REPRODUCED metric=<name> actual=<v> expected=<v> delta_pct=<x>
       echo PAPER_RESULT_NOT_REPRODUCED <one-line reason>
   Do NOT fabricate numbers. If parsing fails after one retry, echo
   PAPER_RESULT_NOT_REPRODUCED with a brief parsing note.

When degradation occurs (SDPA fallback, alternate flash-attn install,
base-image change, scale-down), say so explicitly in your final message
so the benchmark scorer can pick it up.

{rocm_digest}
"""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _check_claude_cli() -> bool:
    return shutil.which("claude") is not None


def _load_task_record(tasks_json: str, paper_id: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(tasks_json):
        return None
    with open(tasks_json, "r", encoding="utf-8") as f:
        records = json.load(f)
    for r in records:
        if r.get("paper_id") == paper_id:
            return r
    return None


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


def _clone_repo(full_name: str, sha: str, dst: str, log_path: str) -> bool:
    """git clone + checkout. Returns True on success."""
    if os.path.exists(dst):
        shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    with open(log_path, "ab") as logf:
        logf.write(b"\n--- git clone ---\n")
        rc = subprocess.run(
            ["git", "clone", "--depth", "200", f"https://github.com/{full_name}.git", dst],
            stdout=logf, stderr=subprocess.STDOUT,
        ).returncode
        if rc != 0:
            return False

        logf.write(f"\n--- git checkout {sha} ---\n".encode())
        rc = subprocess.run(
            ["git", "-C", dst, "checkout", sha],
            stdout=logf, stderr=subprocess.STDOUT,
        ).returncode
        if rc != 0:
            # fall back to full fetch
            subprocess.run(["git", "-C", dst, "fetch", "--unshallow"],
                           stdout=logf, stderr=subprocess.STDOUT)
            rc = subprocess.run(
                ["git", "-C", dst, "checkout", sha],
                stdout=logf, stderr=subprocess.STDOUT,
            ).returncode
    return rc == 0


def _download_paper(url: str, dst: str, timeout: int = 120) -> bool:
    if not url:
        return False
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Repo2ROCm-Benchmark/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(dst, "wb") as f:
            shutil.copyfileobj(resp, f)
        return os.path.getsize(dst) > 1024
    except Exception:
        return False


def _build_task_prompt(*, paper_id: str, full_name: str, repo_path: str,
                       paper_pdf_path: Optional[str], paper_url: Optional[str],
                       gpu_index: int, container_name: str,
                       paper_title: str, mode: str = "full") -> str:
    paper_clause = (
        f"- Paper PDF (mount as /repo/paper.pdf): {paper_pdf_path}"
        if paper_pdf_path else
        f"- Paper PDF: not pre-downloaded; if needed fetch from {paper_url or '(unknown)'}"
    )
    if mode == "env":
        goal_clause = (
            "GOAL: ROCm environment verification ONLY (Mode 1 / functional correctness).\n"
            "Stop after step 5 of the system-prompt workflow. Do NOT attempt the paper\n"
            "experiment (steps 6-7). Echo ROCM_ENV_VERIFIED exactly once when the\n"
            "smoke run succeeds. Do NOT echo any PAPER_RESULT_* marker."
        )
    elif mode == "reproduce":
        goal_clause = (
            "GOAL: Paper experiment reproduction (Mode 2). Env setup is a prerequisite;\n"
            "echo ROCM_ENV_VERIFIED when the env works, then proceed to steps 6-7 and\n"
            "echo a PAPER_RESULT_* marker."
        )
    else:
        goal_clause = (
            "GOAL: Full Mode 3 - env setup then paper reproduction. Echo both\n"
            "ROCM_ENV_VERIFIED and a PAPER_RESULT_* marker."
        )
    return (
        f"Reproduce one experiment from the paper for repository '{full_name}'.\n\n"
        f"{goal_clause}\n\n"
        f"Inputs:\n"
        f"- Paper title: {paper_title}\n"
        f"- Local clone path: {repo_path}\n"
        f"{paper_clause}\n"
        f"- Assigned GPU index (set HIP_VISIBLE_DEVICES): {gpu_index}\n"
        f"- Container name to use: {container_name}\n\n"
        f"Follow the workflow in the system prompt exactly, subject to the GOAL above.\n"
        f"Echo all markers as plain shell echo statements so a downstream scorer can\n"
        f"grep them out of your final message and the container logs."
    )


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #

def run_task(*, paper_id: str, task_dir: str, gpu_index: int,
             container_name_prefix: str, timeout_s: int,
             tasks_json: str,
             claude_model: str = "sonnet",
             max_turns: int = 100,
             mode: str = "full",
             **_unused) -> Dict[str, Any]:
    if not _check_claude_cli():
        return {"exit_code": 127, "timed_out": False,
                "notes": "claude CLI not on PATH"}

    rec = _load_task_record(tasks_json, paper_id)
    if not rec:
        return {"exit_code": 2, "timed_out": False,
                "notes": f"task record not found for {paper_id}"}

    full_name = rec.get("repo_full_name") or ""
    sha = rec.get("repo_sha") or ""
    paper_url = rec.get("paper_url") or ""
    paper_title = rec.get("paper_title") or paper_id

    if not full_name or not sha:
        return {"exit_code": 2, "timed_out": False,
                "notes": f"missing repo or sha for {paper_id}"}

    repo_path = os.path.join(task_dir, "repo")
    paper_pdf_path = os.path.join(task_dir, "paper.pdf")
    log_path = os.path.join(task_dir, "run.log")

    # Step 1: clone the repo on the host so the CLI can mount it into the
    # container with read-write access.
    if not _clone_repo(full_name, sha, repo_path, log_path):
        return {"exit_code": 1, "timed_out": False, "notes": "clone_failed"}

    # Step 2: pre-fetch the paper PDF (best-effort).
    have_paper = _download_paper(paper_url, paper_pdf_path) if paper_url else False
    if not have_paper:
        paper_pdf_path = None

    # Step 3: assemble system + task prompts.
    rocm_digest = _build_rocm_knowledge_digest()
    system_prompt = _BASELINE_SYSTEM_TEMPLATE.format(rocm_digest=rocm_digest)
    container_name = f"{container_name_prefix}-cc"
    task_prompt = _build_task_prompt(
        paper_id=paper_id,
        full_name=full_name,
        repo_path=os.path.abspath(repo_path),
        paper_pdf_path=os.path.abspath(paper_pdf_path) if paper_pdf_path else None,
        paper_url=paper_url,
        gpu_index=gpu_index,
        container_name=container_name,
        paper_title=paper_title,
        mode=mode,
    )

    cmd = [
        "claude", "-p",
        "--output-format", "json",
        "--dangerously-skip-permissions",
        "--system-prompt", system_prompt,
        "--max-turns", str(max_turns),
    ]
    if claude_model:
        cmd.extend(["--model", claude_model])
    cmd.append(task_prompt)

    metadata = {
        "paper_id": paper_id,
        "approach": "claude_cli",
        "mode": mode,
        "gpu_index": gpu_index,
        "container_name": container_name,
        "full_name": full_name,
        "sha": sha,
        "paper_url": paper_url,
        "paper_pdf_local": paper_pdf_path,
        "paper_pdf_downloaded": have_paper,
        "started_at": time.time(),
        "claude_model": claude_model,
        "max_turns": max_turns,
    }
    with open(os.path.join(task_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    timed_out = False
    exit_code: Optional[int] = None
    stdout_bytes = b""
    start = time.time()

    with open(log_path, "ab") as logf:
        logf.write(b"\n--- claude CLI invocation ---\n")
        proc = subprocess.Popen(
            cmd,
            cwd=task_dir,
            stdout=subprocess.PIPE,
            stderr=logf,
            env=env,
            preexec_fn=os.setsid,
        )
        try:
            stdout_bytes, _ = proc.communicate(timeout=timeout_s)
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_group(proc)
            try:
                stdout_bytes, _ = proc.communicate(timeout=10)
            except Exception:
                stdout_bytes = b""
            exit_code = proc.returncode if proc.returncode is not None else 124

    # Always persist the raw stdout (may be JSON or partial) for scoring.
    raw_path = os.path.join(task_dir, "claude_response.json")
    with open(raw_path, "wb") as f:
        f.write(stdout_bytes or b"")

    # Parse usage and final text.
    parsed: Dict[str, Any] = {}
    final_text = ""
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    num_turns = 0
    try:
        parsed = json.loads(stdout_bytes.decode("utf-8", errors="replace"))
        final_text = str(parsed.get("result", ""))
        num_turns = int(parsed.get("num_turns", 0) or 0)
        u = parsed.get("usage") or {}
        prompt_tokens = (
            u.get("input_tokens", 0)
            + u.get("cache_read_input_tokens", 0)
            + u.get("cache_creation_input_tokens", 0)
        )
        if not prompt_tokens:
            prompt_tokens = parsed.get("input_tokens", 0)
        completion_tokens = u.get("output_tokens", 0) or parsed.get("output_tokens", 0)
        usage = {
            "prompt_tokens": int(prompt_tokens or 0),
            "completion_tokens": int(completion_tokens or 0),
            "total_tokens": int((prompt_tokens or 0) + (completion_tokens or 0)),
        }
    except Exception:
        final_text = stdout_bytes.decode("utf-8", errors="replace")

    # Mine markers out of the final text (the agent is told to echo them).
    markers = {
        "rocm_env_verified": "ROCM_ENV_VERIFIED" in final_text,
        "paper_reproduced": "PAPER_RESULT_REPRODUCED" in final_text,
        "paper_not_reproduced": "PAPER_RESULT_NOT_REPRODUCED" in final_text,
    }

    # Persist a normalized verdict file the scorer can read directly.
    summary = {
        "final_text": final_text,
        "final_text_tail": final_text[-4000:],
        "markers": markers,
        "usage": usage,
        "num_turns": num_turns,
        "timed_out": timed_out,
        "exit_code": exit_code,
        "is_error": bool(parsed.get("is_error")) if parsed else False,
        "terminal_reason": parsed.get("terminal_reason") if parsed else None,
    }
    with open(os.path.join(task_dir, "claude_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - start
    metadata["finished_at"] = time.time()
    metadata["elapsed_s"] = round(elapsed, 2)
    metadata["exit_code"] = exit_code
    metadata["timed_out"] = timed_out
    metadata["usage"] = usage
    metadata["markers"] = markers
    with open(os.path.join(task_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    notes = (
        f"timeout after {timeout_s}s" if timed_out
        else f"exit={exit_code} markers={markers} tokens={usage['total_tokens']}"
    )
    return {
        "exit_code": int(exit_code if exit_code is not None else 1),
        "timed_out": timed_out,
        "notes": notes,
        "elapsed_s": elapsed,
    }
