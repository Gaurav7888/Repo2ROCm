---
name: nvidia_alternatives
description: CUDA-only PyPI wheels and their AMD/ROCm replacements with exact install commands
when_to_use: When a pip install of a CUDA-flavored package fails, or before installing any package that historically requires CUDA
paths: ["**/requirements*.txt", "**/pyproject.toml", "**/setup.py", "**/setup.cfg", "**/environment.yml"]
allowed_tools: ["PyPIVersions", "DockerExec"]
---

# CUDA → ROCm package alternatives

## flash-attn (a.k.a. flash_attn)

**Do NOT** `pip install flash-attn` — PyPI ships CUDA-only wheels.

```bash
git clone https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention
cd /tmp/flash-attention
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE python setup.py install
echo 'export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE' >> /root/.bashrc
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
```

Use the MAIN Dao-AILab repo (not the ROCm fork). The Triton backend works
out-of-the-box on rocm/pytorch. The env var is required at install AND runtime.
Never set `HSA_OVERRIDE_GFX_VERSION`, `PYTORCH_ROCM_ARCH`, or `MAX_JOBS` for this.

If install still fails, fall back to `torch.nn.functional.scaled_dot_product_attention`.

## bitsandbytes

```bash
git clone https://github.com/ROCm/bitsandbytes.git /tmp/bnb
cd /tmp/bnb && pip install -e .
```

PyPI wheels are CUDA-only.

## xformers

```bash
pip install xformers --index-url https://download.pytorch.org/whl/rocm6.2
```

If that fails, fall back to `torch.nn.functional.scaled_dot_product_attention`.

## nvidia-ml-py / pynvml

```bash
pip install pyrsmi
```

Also consider the `rocm-smi` CLI directly.

## triton

Already preinstalled on ROCm PyTorch images. **Do NOT** reinstall.

## deepspeed

```bash
pip install deepspeed
```

If building from source, set `DS_BUILD_OPS=1 DS_BUILD_AIO=0`.

## apex

```bash
git clone https://github.com/ROCm/apex /tmp/apex
cd /tmp/apex && python setup.py install --cpp_ext --cuda_ext
```

Preinstalled in most ROCm PyTorch images already.

## vllm

**Do NOT** `pip install vllm` — CUDA-only wheels. Switch base image to `rocm/vllm`
or `rocm/vllm-dev` instead. There is no pip-install path on ROCm.

## cupy

```bash
pip install cupy-rocm-5-0
```

If incompatible, fall back to PyTorch tensors or numpy.

## No-go (no ROCm equivalent)

- `pycuda` — use HIP Python bindings or PyTorch.
- `tensorrt` — use MIGraphX, ONNXRuntime-ROCm, or vLLM-ROCm for inference.

## Code patterns

- `nvidia-smi` → `rocm-smi`
- `CUDA_VISIBLE_DEVICES` → also `HIP_VISIBLE_DEVICES` (both work on ROCm PyTorch)
- `nccl` backend → still call it `nccl` in `torch.distributed.init_process_group`; ROCm uses RCCL under the hood
- `torch.device('cuda')` → keep `'cuda'`, do NOT change to `'rocm'` or `'hip'`
- `torch.cuda.is_available()` → returns True on ROCm; reuse the API
