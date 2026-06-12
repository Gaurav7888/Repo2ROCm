"""Pick 10 papers: 2 CUDA-likely + 8 deterministic controls."""
from __future__ import annotations

import json
import os
from typing import Dict, Any

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = os.path.join(HERE, "tasks.json")
CLASS = os.path.join(HERE, "cuda_classification_v2.json")
OUT = os.path.join(HERE, "kernel_subset_paper_ids.txt")
SUBSET_TASKS = os.path.join(HERE, "tasks_kernel_subset.json")

with open(TASKS, "r", encoding="utf-8") as f:
    tasks = json.load(f)
with open(CLASS, "r", encoding="utf-8") as f:
    classification: Dict[str, Any] = json.load(f)

runnable_pids = {
    t["paper_id"] for t in tasks
    if t.get("repo_full_name") and t.get("repo_sha")
}

cuda_pids = sorted(
    [pid for pid, v in classification.items()
     if pid in runnable_pids and (v["has_cuda"] or v["score"] >= 3)],
    key=lambda pid: -classification[pid]["score"],
)
print("CUDA-likely:", cuda_pids)

# Pick 8 controls: prefer variety across tags, deterministic by paper_id.
controls_pool = sorted(
    [pid for pid in runnable_pids if pid not in cuda_pids],
    key=lambda p: p,
)
# Pick 8 spread evenly across the sorted control pool to get a variety.
n_ctrl = 8
step = max(1, len(controls_pool) // n_ctrl)
controls = []
for i in range(n_ctrl):
    idx = min(i * step, len(controls_pool) - 1)
    if controls_pool[idx] not in controls:
        controls.append(controls_pool[idx])
# Top up if any duplicates.
for pid in controls_pool:
    if len(controls) >= n_ctrl:
        break
    if pid not in controls:
        controls.append(pid)

selected = list(cuda_pids[:2]) + controls[:n_ctrl]
selected = selected[:10]
print(f"Selected {len(selected)} papers:")
for pid in selected:
    repo = next((t["repo_full_name"] for t in tasks if t["paper_id"] == pid), "?")
    cls = classification.get(pid, {})
    cu = cls.get("n_cu", 0)
    ext = cls.get("cuda_extension_signal", False)
    print(f"  {pid}  ({repo})  cu={cu} cudaext={ext}")

with open(OUT, "w", encoding="utf-8") as f:
    for pid in selected:
        f.write(pid + "\n")

# Also write a tasks.json filtered to the subset, for runner_repo2rocm.
subset = [t for t in tasks if t["paper_id"] in selected]
with open(SUBSET_TASKS, "w", encoding="utf-8") as f:
    json.dump(subset, f, indent=2)

print(f"\nWrote {OUT} and {SUBSET_TASKS}")
