---
name: flash_attn_amd_install
description: How to install flash-attention on AMD ROCm (Triton backend).
when_to_use: Use when the repo imports `flash_attn` or pins `flash-attn` in requirements.
---

# Installing flash-attention on AMD ROCm

The PyPI wheel `flash-attn` is CUDA-only. On AMD use the upstream repo's Triton backend.

```bash
# Inside the sandbox container
git clone https://github.com/Dao-AILab/flash-attention /tmp/flash-attention
cd /tmp/flash-attention
git checkout v2.6.3  # pick a recent stable tag
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
python setup.py install
```

Verification:

```bash
python -c "import flash_attn; print(flash_attn.__version__)"
```

If it imports cleanly, you're good. If you see a CUDA arch error, the
`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` env var was not set during build.
