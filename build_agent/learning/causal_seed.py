"""
Seed causal-migration transitions for the KB.

The five transition classes called out in the research plan are encoded here
with concrete, executable actions and counterfactuals.  These give Repo2ROCm a
useful prior on the very first run — before any trajectory has been distilled
— so retrieval in `--mode env` can immediately surface guidance for known
failure modes (CUDA-only wheels, wrong base images, missing GPU runtime,
custom CUDA compile errors, paper-metric mismatches).

Mirrors the `seed_if_empty` shape used by `errors/seed_patterns.py`: the
function is a no-op once any causal transitions exist in the KB.

Cross-clone persistence: when this module is loaded, `seed_causal_transitions`
also hydrates the KB from a committed-in-git SQLite file at
`build_agent/learning/causal_kb.db` (if it exists), so distilled
transitions written by past benchmark runs survive a `git pull` / `git clone`.
The harness writes that file after each run via `cmd_run` (checkpoints the
shared WAL and copies the consolidated `.db` into the tracked path).
"""

from __future__ import annotations

import json
import os
import sqlite3

from storage.models import (
    CausalAction, CausalOutcome, CausalState, CausalTransition,
)
from storage.kb_store import KBStore


# Canonical tracked-in-git path for committed causal-memory persistence.
# Lives next to this file so it travels with the build_agent code.
TRACKED_CAUSAL_KB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "causal_kb.db",
)


# Stable IDs make seed transitions idempotent across runs (they are only
# inserted when the table is empty, but a stable id also lets a later
# `INSERT OR REPLACE` safely refresh them).
_SEEDS = [
    {
        "id": "seed_cuda_only_wheel_to_rocm_source_build",
        "transition_class": "cuda_only_wheel_to_rocm_source_build",
        "state": {
            "repo_fingerprint": "torch+flash_attn+custom_cuda",
            "image": "rocm/pytorch",
            "gpu_arch": "gfx942",
            "error_class": "FLASH_ATTN_CUDA_WHEEL",
            "error_signature": "No module named flash_attn_2_cuda",
            "degradation_policy": "strict",
        },
        "action": {
            "type": "package_strategy",
            "command": (
                "pip uninstall -y flash-attn && "
                "git clone https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention && "
                "cd /tmp/flash-attention && "
                "FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE python setup.py install"
            ),
            "evidence": [
                "pypi_versions:flash-attn shows CUDA-only wheels",
                "rocm_package_guidance: prefer Triton AMD backend",
                "torch.version.hip is non-None (ROCm build)",
            ],
        },
        "outcome": {
            "return_code": 0,
            "verification": [
                "import flash_attn passed",
                "GPU smoke test passed",
            ],
            "degradation": "D1",   # functionally equivalent, perf may differ
            "confidence": 0.82,
        },
        "counterfactuals": [
            {
                "action": "pip install flash-attn",
                "expected_outcome": "fail",
                "reason": "PyPI wheel for flash-attn is CUDA-only on this stack.",
            },
        ],
    },
    {
        "id": "seed_wrong_image_to_ranked_image_switch",
        "transition_class": "wrong_image_to_ranked_image_switch",
        "state": {
            "repo_fingerprint": "torch+rocm_required",
            "image": "python:3.10",
            "gpu_arch": "unknown",
            "error_class": "TORCH_CUDA_NOT_AVAILABLE",
            "error_signature": "torch.cuda.is_available() == False",
            "degradation_policy": "strict",
        },
        "action": {
            "type": "image_switch",
            "command": (
                "change_base_image rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.1"
            ),
            "evidence": [
                "dockerhub_tags: rocm/pytorch has matching py3.10 release tag",
                "rocm_image_ranker: highest-ranked tag for torch repo",
                "torch.version.hip is None on the previous image",
            ],
        },
        "outcome": {
            "return_code": 0,
            "verification": [
                "torch.cuda.is_available() == True",
                "torch.version.hip is non-None",
            ],
            "degradation": "D0",
            "confidence": 0.88,
        },
        "counterfactuals": [
            {
                "action": "pip install torch --index-url https://download.pytorch.org/whl/rocm",
                "expected_outcome": "fail",
                "reason": "non-ROCm base image lacks libamdhip64; ROCm wheel cannot find runtime.",
            },
        ],
    },
    {
        "id": "seed_missing_gpu_runtime_to_rocm_base_image",
        "transition_class": "missing_gpu_runtime_to_rocm_base_image",
        "state": {
            "repo_fingerprint": "any+gpu_required",
            "image": "ubuntu:22.04",
            "gpu_arch": "unknown",
            "error_class": "HIPBLAS_NOT_INITIALIZED",
            "error_signature": "hipErrorNotInitialized | libamdhip64.so not found",
            "degradation_policy": "strict",
        },
        "action": {
            "type": "image_switch",
            "command": "change_base_image rocm/dev-ubuntu-22.04:latest",
            "evidence": [
                "ls /opt/rocm/lib/libamdhip64.so* shows missing runtime",
                "rocm-smi command not found on the original image",
                "dockerhub_tags: rocm/dev-ubuntu-22.04 has supported tags",
            ],
        },
        "outcome": {
            "return_code": 0,
            "verification": [
                "rocm-smi reports devices",
                "/opt/rocm/lib/libamdhip64.so* present",
            ],
            "degradation": "D0",
            "confidence": 0.85,
        },
        "counterfactuals": [
            {
                "action": "apt-get install rocm-libs",
                "expected_outcome": "fail",
                "reason": "rocm-libs package alone does not configure /dev/kfd or kernel driver bindings.",
            },
        ],
    },
    {
        "id": "seed_custom_cuda_compile_error_to_hipify_fix",
        "transition_class": "custom_cuda_compile_error_to_hipify_fix",
        "state": {
            "repo_fingerprint": "torch+custom_cuda_kernels",
            "image": "rocm/pytorch",
            "gpu_arch": "gfx942",
            "error_class": "SETUPTOOLS_BUILD_FAIL",
            "error_signature": (
                "fatal error: cuda_runtime.h: No such file or directory"
            ),
            "degradation_policy": "strict",
        },
        "action": {
            "type": "kernel_fix",
            "command": (
                "hipify-clang --inplace $(find . -name '*.cu' -o -name '*.cuh') && "
                "python setup.py build_ext --inplace"
            ),
            "evidence": [
                "hipify-clang available at /opt/rocm/bin/hipify-clang",
                "kernel_migration scaffold: cuda_runtime_header_to_hip_runtime_header",
                "compile error mentions cuda_runtime.h header",
            ],
        },
        "outcome": {
            "return_code": 0,
            "verification": [
                "hipcc compile passed",
                "python -c 'import <ext>' passed",
            ],
            "degradation": "D1",   # warp-size / perf semantics may shift
            "confidence": 0.7,
        },
        "counterfactuals": [
            {
                "action": "pip install --no-build-isolation .",
                "expected_outcome": "fail",
                "reason": "no-build-isolation does not translate CUDA headers to HIP equivalents.",
            },
        ],
    },
    {
        "id": "seed_paper_metric_mismatch_to_not_reproduced",
        "transition_class": "paper_metric_mismatch_to_not_reproduced",
        "state": {
            "repo_fingerprint": "any+paper_reproduction",
            "image": "rocm/pytorch",
            "gpu_arch": "gfx942",
            "error_class": "PAPER_METRIC_MISMATCH",
            "error_signature": "delta_pct exceeds tolerance after verify_paper_result",
            "degradation_policy": "strict",
        },
        "action": {
            "type": "verdict_emit",
            "command": (
                "echo 'PAPER_RESULT_NOT_REPRODUCED metric=<name> "
                "actual=<v> expected=<v> delta_pct=<x>'"
            ),
            "evidence": [
                "verify_paper_result returned verdict=not_reproduced",
                "metric_results show >tolerance delta",
                "no further reasonable scale-down available without changing paper config",
            ],
        },
        "outcome": {
            "return_code": 0,
            "verification": [
                "honest verdict emitted; no fabricated numbers",
            ],
            "degradation": "D2",   # truthful failure beats false-success
            "confidence": 0.95,
        },
        "counterfactuals": [
            {
                "action": "loosen tolerance to >=25% and emit PAPER_RESULT_REPRODUCED",
                "expected_outcome": "false_success",
                "reason": "loose-tolerance pass is flagged by the rubric; reduces benchmark signal.",
            },
        ],
    },
]


def _hydrate_from_tracked_kb(kb: KBStore,
                             tracked_path: str = TRACKED_CAUSAL_KB_PATH) -> int:
    """Copy every transition from the committed-in-git KB into `kb`.

    `INSERT OR REPLACE` is keyed on `transition.id`, so re-hydrating into a
    KB that already has the same row is a no-op (seeds keep their stable
    ids; learned transitions keep their UUIDs). Returns the count of rows
    we attempted to copy (so callers can log it). Best-effort: any
    failure on the tracked file (missing, old schema, corrupt) is
    silently skipped.
    """
    if not tracked_path or not os.path.exists(tracked_path):
        return 0
    try:
        src = sqlite3.connect(f"file:{tracked_path}?mode=ro", uri=True)
    except Exception:
        return 0
    try:
        try:
            rows = src.execute(
                "SELECT data_json FROM causal_transitions"
            ).fetchall()
        except sqlite3.OperationalError:
            return 0  # table missing or schema incompatible
        copied = 0
        for (data_json,) in rows:
            try:
                payload = json.loads(data_json)
                t = CausalTransition.from_dict(payload)
                kb.insert_transition(t, source_attempt=t.source_attempt_id or "tracked")
                copied += 1
            except Exception:
                continue
        return copied
    finally:
        src.close()


def seed_causal_transitions(kb: KBStore) -> int:
    """Insert the seed transitions if `causal_transitions` is empty, then
    hydrate the KB from the git-tracked causal_kb.db (if present).

    Returns the total number of rows inserted (seeds + tracked).
    Mirrors `errors/seed_patterns.seed_if_empty` so it can be safely called
    on every startup. Hydration is unconditional (not gated on the table
    being empty), so re-runs across `git pull` keep newly-distilled
    transitions visible — `INSERT OR REPLACE` makes this idempotent.
    """
    try:
        was_empty = kb.count_transitions() == 0
    except Exception:
        return 0

    inserted = 0
    if was_empty:
        for spec in _SEEDS:
            t = CausalTransition(
                id=spec["id"],
                transition_class=spec["transition_class"],
                state=CausalState(**spec["state"]),
                action=CausalAction(**spec["action"]),
                outcome=CausalOutcome(**spec["outcome"]),
                counterfactuals=list(spec.get("counterfactuals", [])),
                source="seed",
                source_attempt_id="seed",
                evidence_count=5 if spec["outcome"].get("confidence", 0.5) >= 0.8 else 3,
            )
            kb.insert_transition(t, source_attempt="seed")
            inserted += 1

    # Always try to hydrate from the tracked file, even when the KB already
    # had data: harmless re-INSERT (idempotent by id) and lets a `git pull`
    # propagate new transitions into an already-seeded KB.
    try:
        inserted += _hydrate_from_tracked_kb(kb)
    except Exception:
        pass
    return inserted
