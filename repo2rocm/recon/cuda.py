"""CUDA dependency detection."""
from __future__ import annotations

from repo2rocm.knowledge import BANNED_NVIDIA_PACKAGES, CUDA_TO_ROCM_MAPPING

_CUDA_KEYWORDS = {
    "cuda", "cudnn", "nccl", "nvidia", "nvml", "pynvml",
    "bitsandbytes", "flash_attn", "flash_attn_2", "apex", "xformers",
    "triton", "cupy", "pycuda", "tensorrt",
}


def detect_cuda_deps(
    import_counts: dict[str, int],
    config_contents: dict[str, str],
) -> list[str]:
    """Return the sorted list of CUDA/NVIDIA-flavored deps observed."""
    found: set[str] = set()
    for pkg in import_counts:
        low = pkg.lower()
        if low in _CUDA_KEYWORDS or "cuda" in low or "nvidia" in low:
            found.add(pkg)
    config_blob = "\n".join(config_contents.values())
    for banned in BANNED_NVIDIA_PACKAGES:
        if banned in config_blob:
            found.add(banned)
    for mapped in CUDA_TO_ROCM_MAPPING:
        if mapped in config_blob or mapped in import_counts:
            found.add(mapped)
    return sorted(found)
