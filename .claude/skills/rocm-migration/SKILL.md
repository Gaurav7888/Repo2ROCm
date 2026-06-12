---
name: rocm-migration
description: Step-by-step guide for migrating CUDA-based repositories to AMD ROCm GPUs
---

# ROCm Migration Skill

When migrating a CUDA-based repository to ROCm, follow this structured workflow.

## Phase 1: Environment Setup

1. Start from a ROCm Docker image (e.g., `rocm/pytorch:latest`)
2. Verify GPU visibility:
   ```bash
   rocm-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```
3. Set environment variables:
   ```bash
   export ROCM_HOME=/opt/rocm
   export HIP_VISIBLE_DEVICES=0
   ```

## Phase 2: Dependency Migration

1. Parse requirements.txt/setup.py for CUDA-specific packages
2. Apply package mappings:
   - `torch` → install from ROCm wheel index
   - `nvidia-*` → skip entirely
   - `flash-attn` → install from ROCm-compatible source
   - `bitsandbytes` → `bitsandbytes-rocm`
   - `triton` → `pytorch-triton-rocm`
3. Install remaining non-CUDA dependencies normally
4. Run `pipdeptree` to verify no conflicts

## Phase 3: Code Patches

1. Replace `nvidia-smi` calls with `rocm-smi`
2. Guard `torch.backends.cudnn.*`:
   ```python
   if not getattr(torch.version, 'hip', None):
       torch.backends.cudnn.benchmark = True
   ```
3. Set `WANDB_MODE=offline` if wandb is used
4. Fix deprecated `torch.cuda.amp` → `torch.amp.autocast('cuda')`

## Phase 4: Custom Kernel Migration (if applicable)

1. Inventory `.cu`/`.cuh` files
2. Run `hipify-clang` for automated conversion
3. Fix compilation errors:
   - Warp size 64 vs 32
   - cuBLAS → hipBLAS
   - cuDNN → MIOpen
4. Verify numerical equivalence

## Phase 5: Triton Kernel Fixes (if applicable)

1. Find Triton kernels: `grep -rl "@triton.jit" --include="*.py"`
2. Fix `num_warps` for AMD wavefront64
3. Remove `allow_tf32=True` from `tl.dot`
4. Install `pytorch-triton-rocm`

## Phase 6: Verification

1. Run `python -c "import <main_package>"` to verify imports
2. Create minimal mock data if needed (unless --no-scale-down)
3. Run the project's main script with mock/real data
4. Verify output shows CUDA device usage (not CPU)
5. Signal success: `echo ROCM_ENV_VERIFIED`

## Common Error Patterns and Fixes

| Error | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'nvidia'` | Skip nvidia-* packages |
| `hipErrorNoBinaryForGpu` | Set `HSA_OVERRIDE_GFX_VERSION` |
| `undefined symbol: _ZN...` (ABI mismatch) | Rebuild extension with matching PyTorch |
| `RuntimeError: HIP error` | Check ROCm version compatibility |
| `ImportError: libamdhip64.so` | Add `/opt/rocm/lib` to LD_LIBRARY_PATH |
