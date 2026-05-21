---
name: banned_nvidia_packages
description: NVIDIA-only PyPI packages that must be stripped from requirements before install on ROCm
when_to_use: Before running Download / pip install, or when interpreting an install failure that mentions a CUDA runtime wheel
paths: ["**/requirements*.txt", "**/setup.py", "**/setup.cfg", "**/pyproject.toml"]
---

# Banned NVIDIA packages on ROCm

These wheels are CUDA-only and have NO ROCm equivalent. They MUST be removed from
any requirements before `Download` runs — installing them breaks the ROCm runtime.

```
nvidia-cuda-runtime-cu11      nvidia-cuda-runtime-cu12
nvidia-cuda-cupti-cu11        nvidia-cuda-cupti-cu12
nvidia-cuda-nvrtc-cu11        nvidia-cuda-nvrtc-cu12
nvidia-cublas-cu11            nvidia-cublas-cu12
nvidia-cufft-cu11             nvidia-cufft-cu12
nvidia-curand-cu11            nvidia-curand-cu12
nvidia-cusolver-cu11          nvidia-cusolver-cu12
nvidia-cusparse-cu11          nvidia-cusparse-cu12
nvidia-nccl-cu11              nvidia-nccl-cu12
nvidia-nvjitlink-cu11         nvidia-nvjitlink-cu12
nvidia-nvtx-cu11              nvidia-nvtx-cu12
nvidia-cudnn-cu11             nvidia-cudnn-cu12
nvidia-dali-cuda110           nvidia-dali-cuda120
torch+cu118                   torch+cu121
```

Why each is gone:

- `nvidia-cuda-runtime-*`, `nvidia-cublas-*`, `nvidia-cufft-*`, `nvidia-curand-*`,
  `nvidia-cusolver-*`, `nvidia-cusparse-*`, `nvidia-nvjitlink-*`, `nvidia-nvtx-*` —
  provided by the ROCm base image's `torch` install via ROCm-native libraries.
- `nvidia-cudnn-*` — replaced by MIOpen, used transparently by PyTorch.
- `nvidia-nccl-*` — replaced by RCCL, used transparently by `torch.distributed`.
- `nvidia-dali-*` — no ROCm port; use plain `DataLoader` or `webdataset`.
- `torch+cu1??` — wrong wheel; rocm/pytorch images ship `torch+rocm`.

Strip pattern: any requirement starting with `nvidia-` and ending with `-cuNN`, or
`torch+cuNN`. Keep regular `nvidia-`-prefixed tooling unrelated to runtime (rare).
