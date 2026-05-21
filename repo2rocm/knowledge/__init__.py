"""Structured ROCm domain knowledge.

This package holds the single source of truth for everything CUDA→ROCm:
  * ROCm Docker image catalog (`ROCM_IMAGE_CATALOG`)
  * Preinstalled packages per image (`ROCM_PREINSTALLED_PACKAGES`)
  * NVIDIA → AMD package mappings (`CUDA_TO_ROCM_MAPPING`)
  * Banned NVIDIA packages (`BANNED_NVIDIA_PACKAGES`)
  * Image-scoring signals (`IMAGE_SIGNALS`) for the deterministic preflight selector
  * Supported GPU arches (`SUPPORTED_GPU_ARCHITECTURES`)

The recon pipeline imports the tables directly (deterministic, fast).
The skills under `repo2rocm/skills/builtin/` are kept in lockstep with this data
so the LLM consumes the same facts in markdown form.
"""
from repo2rocm.knowledge.rocm_data import (
    BANNED_NVIDIA_PACKAGES,
    CUDA_CODE_PATTERNS,
    CUDA_TO_ROCM_MAPPING,
    IMAGE_SIGNALS,
    ROCM_IMAGE_CATALOG,
    ROCM_PREINSTALLED_PACKAGES,
    SUPPORTED_GPU_ARCHITECTURES,
    get_preinstalled_packages,
    get_rocm_alternative,
    is_banned_package,
)

__all__ = [
    "BANNED_NVIDIA_PACKAGES",
    "CUDA_CODE_PATTERNS",
    "CUDA_TO_ROCM_MAPPING",
    "IMAGE_SIGNALS",
    "ROCM_IMAGE_CATALOG",
    "ROCM_PREINSTALLED_PACKAGES",
    "SUPPORTED_GPU_ARCHITECTURES",
    "get_preinstalled_packages",
    "get_rocm_alternative",
    "is_banned_package",
]
