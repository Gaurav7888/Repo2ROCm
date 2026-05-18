---
name: cuda_to_rocm_mapping
description: Mapping table from CUDA-only PyPI wheels to AMD/ROCm alternatives.
when_to_use: Use when installing a package that historically requires CUDA, or when a `pip install` fails with a CUDA-arch error.
---

# CUDA → ROCm mapping

| CUDA PyPI wheel | ROCm/AMD equivalent | Notes |
|---|---|---|
| `flash-attn` | `git clone Dao-AILab/flash-attention && FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE python setup.py install` | Use Triton backend on AMD. |
| `bitsandbytes` | `pip install bitsandbytes` from the ROCm fork (`bitsandbytes-rocm`) or build from source with `ROCM_HOME` set. |
| `xformers` | Not supported on ROCm in many versions. Use PyTorch SDPA (`torch.nn.functional.scaled_dot_product_attention`). |
| `nvidia-cuda-runtime-cu12`, `nvidia-cublas-cu12`, `nvidia-cudnn-cu12` | **Remove** — provided by the ROCm base image's `torch`. |
| `triton` | Pre-installed on ROCm pytorch images. Do NOT `pip install triton`. |
| `apex` | Use ROCm fork: `git clone https://github.com/ROCm/apex && python setup.py install --cpp_ext --cuda_ext`. |
| `deepspeed` | Pip-installs cleanly on `rocm/pytorch-training`; verify with `ds_report`. |
| `vllm` | Use `rocm/vllm` base image; do NOT `pip install vllm` (CUDA wheels). |

## Banned at install time

These NEVER work on ROCm and must be stripped from `requirements*.txt`:

- `nvidia-cuda-runtime-cu*`
- `nvidia-cuda-cupti-cu*`
- `nvidia-cudnn-cu*`
- `nvidia-curand-cu*`
- `nvidia-cusparse-cu*`
- `cupy-cuda*` (use `cupy-rocm-*` instead)
- `tensorrt`
- `pycuda`
