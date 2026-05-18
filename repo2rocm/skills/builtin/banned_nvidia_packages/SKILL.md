---
name: banned_nvidia_packages
description: List of NVIDIA-only PyPI packages that must be removed from requirements before any ROCm install.
---

# Banned NVIDIA packages (PyPI)

Strip these from `requirements*.txt`, `pyproject.toml`, `setup.py` before `Download`:

- `nvidia-cuda-runtime-cu11`, `nvidia-cuda-runtime-cu12`
- `nvidia-cuda-cupti-cu11`, `nvidia-cuda-cupti-cu12`
- `nvidia-cuda-nvrtc-cu11`, `nvidia-cuda-nvrtc-cu12`
- `nvidia-cublas-cu11`, `nvidia-cublas-cu12`
- `nvidia-cudnn-cu11`, `nvidia-cudnn-cu12`
- `nvidia-curand-cu11`, `nvidia-curand-cu12`
- `nvidia-cusparse-cu11`, `nvidia-cusparse-cu12`
- `nvidia-cusolver-cu11`, `nvidia-cusolver-cu12`
- `nvidia-cufft-cu11`, `nvidia-cufft-cu12`
- `nvidia-nccl-cu11`, `nvidia-nccl-cu12`
- `nvidia-nvjitlink-cu12`
- `nvidia-nvtx-cu11`, `nvidia-nvtx-cu12`
- `tensorrt`
- `cupy-cuda*` (use `cupy-rocm-*`)
- `pycuda`
- `numba` (when used for CUDA kernels — Triton AMD is the alternative)
