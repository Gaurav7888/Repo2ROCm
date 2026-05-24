"""Deeper scan: also flag repos that import CUDAExtension / cpp_extension or
have CUDA-ish source files."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS_JSON = os.path.join(HERE, "tasks.json")
OUT_JSON = os.path.join(HERE, "cuda_classification_v2.json")
OUT_TXT = os.path.join(HERE, "kernel_subset_paper_ids.txt")

CUDA_HINTS = [
    "CUDAExtension",
    "cpp_extension",
    "torch.utils.cpp_extension",
    "load_inline",
    "nvcc",
    "FLASH_ATTENTION",
    "flash_attn",
    "xformers",
    "deepspeed",
    "vllm",
    "triton.jit",
    "@triton.jit",
]


def fetch_tree(repo: str, sha: str) -> List[Dict[str, Any]]:
    res = subprocess.run(
        ["gh", "api", f"repos/{repo}/git/trees/{sha}?recursive=1"],
        capture_output=True, text=True, timeout=60,
    )
    if res.returncode != 0:
        return []
    try:
        return json.loads(res.stdout).get("tree", []) or []
    except Exception:
        return []


def fetch_file(repo: str, sha: str, path: str) -> str:
    res = subprocess.run(
        ["gh", "api", f"repos/{repo}/contents/{path}?ref={sha}",
         "-H", "Accept: application/vnd.github.raw"],
        capture_output=True, text=True, timeout=60,
    )
    if res.returncode != 0:
        return ""
    return res.stdout or ""


def classify(repo: str, sha: str, tree: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_cu = 0
    n_cuh = 0
    has_csrc = False
    has_cuda_dir = False
    has_setup_py = False
    has_pyproject = False
    has_requirements = False
    cuda_extension_signal = False
    triton_signal = False
    setup_paths: List[str] = []
    req_paths: List[str] = []
    for entry in tree:
        path = entry.get("path", "")
        if entry.get("type") != "blob":
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext == ".cu":
            n_cu += 1
        elif ext == ".cuh":
            n_cuh += 1
        if "csrc/" in path or path.startswith("csrc/"):
            has_csrc = True
        if "/cuda/" in path.lower() or path.lower().startswith("cuda/"):
            has_cuda_dir = True
        if path.endswith("setup.py"):
            has_setup_py = True
            setup_paths.append(path)
        if path.endswith("pyproject.toml"):
            has_pyproject = True
            setup_paths.append(path)
        if path.endswith("requirements.txt") or "requirements" in path and path.endswith(".txt"):
            has_requirements = True
            req_paths.append(path)

    # Probe a handful of files for CUDA / Triton signals.
    probe_paths: List[str] = []
    for p in setup_paths[:3]:
        probe_paths.append(p)
    for p in req_paths[:3]:
        probe_paths.append(p)
    probe_paths.append("README.md")

    blob = ""
    for p in probe_paths:
        try:
            content = fetch_file(repo, sha, p)
        except Exception:
            content = ""
        if content:
            blob += "\n" + content[:50_000]
        time.sleep(0.05)

    blob_low = blob.lower()
    for hint in ("cudaextension", "cpp_extension", "torch.utils.cpp_extension", "load_inline", "nvcc"):
        if hint in blob_low:
            cuda_extension_signal = True
            break
    for hint in ("triton.jit", "@triton.jit"):
        if hint in blob_low:
            triton_signal = True
            break

    score = (
        4 * n_cu
        + 2 * n_cuh
        + (3 if has_csrc else 0)
        + (2 if has_cuda_dir else 0)
        + (3 if cuda_extension_signal else 0)
        + (1 if triton_signal else 0)
    )
    return {
        "n_cu": n_cu,
        "n_cuh": n_cuh,
        "has_csrc": has_csrc,
        "has_cuda_dir": has_cuda_dir,
        "cuda_extension_signal": cuda_extension_signal,
        "triton_signal": triton_signal,
        "has_cuda": (n_cu + n_cuh) > 0 or cuda_extension_signal,
        "score": score,
    }


def main() -> int:
    with open(TASKS_JSON, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    runnable = [t for t in tasks if t.get("repo_full_name") and t.get("repo_sha")]

    # Reuse v1 classification for fast keys, but also probe content for signals.
    out: Dict[str, Dict[str, Any]] = {}
    for i, t in enumerate(runnable, 1):
        pid = t["paper_id"]
        repo = t["repo_full_name"]
        sha = t["repo_sha"]
        print(f"[{i:02d}/{len(runnable)}] {repo}@{sha[:8]} ...", file=sys.stderr)
        try:
            tree = fetch_tree(repo, sha)
        except Exception as e:
            print(f"  ! tree fetch failed: {e}", file=sys.stderr)
            tree = []
        info = classify(repo, sha, tree)
        info["paper_id"] = pid
        info["repo_full_name"] = repo
        info["paper_title"] = t.get("paper_title", "")
        info["tags"] = t.get("tags", "")
        out[pid] = info

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    cuda_repos = sorted(
        [v for v in out.values() if v["has_cuda"] or v["score"] >= 3],
        key=lambda v: -v["score"],
    )
    print(f"\nRepos with CUDA hints: {len(cuda_repos)}", file=sys.stderr)
    for v in cuda_repos:
        print(f"  score={v['score']:3d} cu={v['n_cu']} cuh={v['n_cuh']} "
              f"csrc={v['has_csrc']} cudaext={v['cuda_extension_signal']} "
              f"triton={v['triton_signal']}  {v['repo_full_name']}  ({v['paper_id']})",
              file=sys.stderr)

    # Build subset: top CUDA-ish + a few controls.
    n_cuda = min(7, len(cuda_repos))
    selected = [v["paper_id"] for v in cuda_repos[:n_cuda]]
    no_cuda = sorted(
        [v for v in out.values() if not v["has_cuda"] and v["score"] < 3],
        key=lambda v: v["paper_id"],
    )
    n_controls = min(3, len(no_cuda))
    selected += [v["paper_id"] for v in no_cuda[:n_controls]]

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        for pid in selected:
            f.write(pid + "\n")
    print(f"\nWrote subset of {len(selected)} paper_ids to {OUT_TXT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
