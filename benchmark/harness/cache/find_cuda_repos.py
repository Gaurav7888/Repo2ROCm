"""Scan AMD-60 tasks.json and classify repos by CUDA-kernel content.

For each runnable task, fetches the GitHub tree at the task's SHA via gh CLI
(uses authenticated quota) and counts .cu / .cuh / cuda_*.cpp files.

Outputs:
    cache/cuda_classification.json — { paper_id: { has_cuda, n_cu, n_cuh,
        has_torch_cpp_extension, score } }
    cache/kernel_subset_paper_ids.txt — selected ~10 paper_ids for the
        "kernel-only" benchmark subset.
"""
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
OUT_JSON = os.path.join(HERE, "cuda_classification.json")
OUT_TXT = os.path.join(HERE, "kernel_subset_paper_ids.txt")

CUDA_EXTS = {".cu", ".cuh"}
HINT_FILES = ("setup.py", "csrc", "kernels", "cuda")


def fetch_tree(repo_full_name: str, sha: str) -> List[Dict[str, Any]]:
    if shutil.which("gh") is None:
        raise RuntimeError("gh CLI not available")
    cmd = ["gh", "api", f"repos/{repo_full_name}/git/trees/{sha}?recursive=1"]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if res.returncode != 0:
        return []
    try:
        data = json.loads(res.stdout)
        return data.get("tree", []) or []
    except Exception:
        return []


def classify(tree: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_cu = 0
    n_cuh = 0
    has_torch_ext = False
    has_csrc = False
    has_cuda_dir = False
    has_pybind = False
    for entry in tree:
        path = entry.get("path", "")
        kind = entry.get("type", "")
        if kind == "blob":
            ext = os.path.splitext(path)[1].lower()
            if ext == ".cu":
                n_cu += 1
            elif ext == ".cuh":
                n_cuh += 1
            elif path.endswith("setup.py"):
                pass
            if "csrc/" in path or path.startswith("csrc/"):
                has_csrc = True
            if "/cuda/" in path or path.startswith("cuda/"):
                has_cuda_dir = True
            if path.endswith(("pybind.cpp", "bindings.cpp")):
                has_pybind = True
    score = (
        4 * n_cu
        + 2 * n_cuh
        + (3 if has_csrc else 0)
        + (2 if has_cuda_dir else 0)
        + (1 if has_pybind else 0)
    )
    return {
        "n_cu": n_cu,
        "n_cuh": n_cuh,
        "has_csrc": has_csrc,
        "has_cuda_dir": has_cuda_dir,
        "has_pybind": has_pybind,
        "has_cuda": (n_cu + n_cuh) > 0,
        "score": score,
    }


def main() -> int:
    with open(TASKS_JSON, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    runnable = [t for t in tasks if t.get("repo_full_name") and t.get("repo_sha")]
    print(f"Scanning {len(runnable)} runnable repos...", file=sys.stderr)

    out: Dict[str, Dict[str, Any]] = {}
    for i, t in enumerate(runnable, 1):
        pid = t["paper_id"]
        repo = t["repo_full_name"]
        sha = t["repo_sha"]
        print(f"[{i:02d}/{len(runnable)}] {repo}@{sha[:8]} ...", file=sys.stderr)
        try:
            tree = fetch_tree(repo, sha)
        except Exception as e:
            print(f"  ! {e}", file=sys.stderr)
            tree = []
        info = classify(tree)
        info["paper_id"] = pid
        info["repo_full_name"] = repo
        info["paper_title"] = t.get("paper_title", "")
        out[pid] = info
        time.sleep(0.05)  # polite pacing

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    cuda_repos = [v for v in out.values() if v["has_cuda"]]
    cuda_repos.sort(key=lambda v: -v["score"])
    print(f"\nRepos with .cu/.cuh: {len(cuda_repos)}", file=sys.stderr)
    for v in cuda_repos:
        print(f"  score={v['score']:3d} cu={v['n_cu']} cuh={v['n_cuh']}  "
              f"{v['repo_full_name']}  ({v['paper_id']})", file=sys.stderr)

    # Pick top ~7 cuda repos + 3 controls (no .cu, well-known PyTorch / vLLM-ish)
    n_cuda = min(7, len(cuda_repos))
    selected = [v["paper_id"] for v in cuda_repos[:n_cuda]]
    no_cuda = [v for v in out.values() if not v["has_cuda"]]
    no_cuda.sort(key=lambda v: v["paper_id"])  # stable
    n_controls = min(3, len(no_cuda))
    selected += [v["paper_id"] for v in no_cuda[:n_controls]]

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        for pid in selected:
            f.write(pid + "\n")
    print(f"\nWrote subset of {len(selected)} paper_ids to {OUT_TXT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
