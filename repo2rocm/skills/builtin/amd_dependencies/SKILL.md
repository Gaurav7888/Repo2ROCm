---
name: amd_dependencies
description: Preferred AMD/ROCm packages, install hints, and pitfalls beyond the simple CUDA→ROCm mapping
when_to_use: When the repo's task involves an AMD ecosystem component (RCCL, MIOpen, MIGraphX, ROCm libraries, profilers)
paths: ["**/Dockerfile*", "**/Makefile", "**/CMakeLists.txt"]
---

# AMD / ROCm preferred dependencies

## Core runtime

- **HIP** — comes with the ROCm base images; no separate install.
- **RCCL** — NCCL counterpart; transparent via `torch.distributed(backend="nccl")`.
- **MIOpen** — cuDNN counterpart; transparent for PyTorch ops.
- **rocBLAS / hipBLAS** — preinstalled.

## Math + science

- **MIGraphX** — `pip install migraphx`; inference acceleration alternative to TensorRT.
- **rocFFT / hipFFT** — preinstalled.
- **rocSPARSE / hipSPARSE** — preinstalled.

## Profilers

- **rocprof** — bundled with ROCm; CLI at `rocprof`.
- **omniperf** — install via `pip install omniperf` for kernel-level profiling.
- **omnitrace** — system-wide tracing.

## GPU monitoring

- CLI: `rocm-smi` (already installed in every ROCm image).
- Python: `pyrsmi` (replaces `pynvml` / `nvidia-ml-py`).

## Distributed training building blocks

- DeepSpeed: `pip install deepspeed` works on ROCm; set
  `DS_BUILD_OPS=1 DS_BUILD_AIO=0` for source builds.
- FSDP: native PyTorch, no extra package.
- Accelerate: `pip install accelerate` (no change from CUDA).
- Megatron-LM: use the `rocm/megatron-lm` base image; do not pip install.

## GFX architectures

| GFX | AMD GPU |
|---|---|
| `gfx908` | MI100 |
| `gfx90a` | MI210 / MI250 / MI250x |
| `gfx942` | MI300A / MI300X / MI325 |
| `gfx950` | MI350 / MI355 (ROCm 7.0+) |
| `gfx1030` | Navi21 consumer |
| `gfx1100` | Navi31 consumer |

Set `PYTORCH_ROCM_ARCH` only when **explicitly required by a source build** and only
to the target arch — usually unnecessary on the prebuilt PyTorch images.
