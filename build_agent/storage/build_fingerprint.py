"""
Build fingerprint computation — produces a canonical BuildFingerprint
from a cloned repository for nearest-neighbor lookup in the KB.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Set

from storage.models import BuildFingerprint


_FRAMEWORK_PACKAGES = {
    "torch": "pytorch",
    "pytorch": "pytorch",
    "tensorflow": "tensorflow",
    "keras": "tensorflow",
    "jax": "jax",
    "flax": "jax",
    "vllm": "vllm",
    "sglang": "sglang",
    "onnxruntime": "onnxruntime",
    "onnx": "onnxruntime",
}

_CUDA_INDICATORS = {
    "flash_attn", "flash_attn_2", "apex", "xformers", "triton",
    "bitsandbytes", "cuda", "cudnn", "nccl", "pynvml",
    "nvidia", "nvml",
}

_DISTRIBUTED_INDICATORS = {
    "deepspeed", "accelerate", "pytorch_lightning", "lightning",
    "horovod", "torch.distributed", "colossalai",
}


def compute_fingerprint(
    repo_path: str,
    repo_id: str = "",
    import_counts: Optional[Dict[str, int]] = None,
    config_contents: Optional[Dict[str, str]] = None,
    python_version: str = "",
) -> BuildFingerprint:
    """
    Build a canonical fingerprint from a cloned repo.

    Can accept pre-computed imports and configs from the planner
    to avoid redundant filesystem scans.
    """
    fp = BuildFingerprint(repo_id=repo_id, python_version=python_version)

    if import_counts is None:
        import_counts = _scan_imports(repo_path)

    # frameworks
    for pkg, framework in _FRAMEWORK_PACKAGES.items():
        if pkg in import_counts:
            fp.frameworks.add(framework)

    # CUDA deps
    for pkg in import_counts:
        pkg_lower = pkg.lower()
        if pkg_lower in _CUDA_INDICATORS or "cuda" in pkg_lower or "nvidia" in pkg_lower:
            fp.cuda_deps.add(pkg_lower)
    if config_contents:
        all_config = "\n".join(config_contents.values())
        for indicator in _CUDA_INDICATORS:
            if indicator in all_config:
                fp.cuda_deps.add(indicator)

    # build system
    fp.build_system = _detect_build_system(repo_path, config_contents)

    # config files present
    if config_contents:
        fp.config_files_present = sorted(config_contents.keys())

    # custom CUDA kernels
    fp.has_custom_cuda_kernels = _has_cuda_files(repo_path)

    # Triton kernels
    fp.has_triton_kernels = _has_triton_kernels(repo_path, import_counts)

    # distributed
    for pkg in _DISTRIBUTED_INDICATORS:
        if pkg in import_counts:
            fp.has_distributed = True
            break

    # workload type
    fp.workload_type = _infer_workload_type(repo_path, import_counts, config_contents)

    # top imports
    sorted_imports = sorted(import_counts.items(), key=lambda x: -x[1])
    fp.top_imports = [pkg for pkg, _ in sorted_imports[:20]]

    return fp


def _scan_imports(repo_path: str) -> Dict[str, int]:
    """Quick import scan when planner hasn't provided one."""
    import_re = re.compile(r"^\s*(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_]*)")
    counts: Dict[str, int] = {}
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "node_modules")]
        for f in files:
            if not f.endswith(".py"):
                continue
            fpath = os.path.join(root, f)
            try:
                with open(fpath, errors="ignore") as fh:
                    for line in fh:
                        m = import_re.match(line)
                        if m:
                            pkg = m.group(1)
                            counts[pkg] = counts.get(pkg, 0) + 1
            except Exception:
                continue
    return counts


def _detect_build_system(repo_path: str,
                         config_contents: Optional[Dict[str, str]]) -> str:
    if config_contents:
        files = set(config_contents.keys())
    else:
        try:
            files = set(os.listdir(repo_path))
        except Exception:
            files = set()

    if "poetry.lock" in files:
        return "poetry"
    if "Pipfile" in files:
        return "pipenv"
    if "environment.yml" in files or "environment.yaml" in files:
        return "conda"
    if "setup.py" in files:
        return "setuptools"
    if "pyproject.toml" in files:
        pyproject = config_contents.get("pyproject.toml", "") if config_contents else ""
        if "poetry" in pyproject.lower():
            return "poetry"
        return "pyproject"
    if any(f.startswith("requirements") for f in files):
        return "pip"
    return "unknown"


def _has_cuda_files(repo_path: str) -> bool:
    """Check for .cu / .cuh files."""
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "node_modules")]
        for f in files:
            if f.endswith((".cu", ".cuh")):
                return True
    return False


def _has_triton_kernels(repo_path: str,
                        import_counts: Dict[str, int]) -> bool:
    if "triton" in import_counts:
        return True
    autotune_re = re.compile(r"@triton\.(?:autotune|jit)")
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "node_modules")]
        for f in files:
            if not f.endswith(".py"):
                continue
            fpath = os.path.join(root, f)
            try:
                with open(fpath, errors="ignore") as fh:
                    content = fh.read(10000)
                if autotune_re.search(content):
                    return True
            except Exception:
                continue
    return False


def _infer_workload_type(repo_path: str,
                         import_counts: Dict[str, int],
                         config_contents: Optional[Dict[str, str]]) -> str:
    """Classify as inference/training/finetuning/serving."""
    all_text = ""
    if config_contents:
        all_text = "\n".join(config_contents.values()).lower()

    readme_path = os.path.join(repo_path, "README.md")
    if os.path.isfile(readme_path):
        try:
            with open(readme_path, errors="ignore") as f:
                all_text += "\n" + f.read(5000).lower()
        except Exception:
            pass

    serving_signals = {"vllm" in import_counts, "sglang" in import_counts,
                       "serve" in all_text, "endpoint" in all_text,
                       "api server" in all_text}
    finetune_signals = {"finetun" in all_text, "lora" in all_text,
                        "qlora" in all_text, "peft" in import_counts}
    training_signals = {"train" in all_text and "pretrain" in all_text,
                        "deepspeed" in import_counts,
                        "trainer" in all_text}
    inference_signals = {"infer" in all_text, "generat" in all_text,
                         "predict" in all_text, "demo" in all_text}

    if any(serving_signals):
        return "serving"
    if any(finetune_signals):
        return "finetuning"
    if any(training_signals):
        return "training"
    if any(inference_signals):
        return "inference"
    return "unknown"
