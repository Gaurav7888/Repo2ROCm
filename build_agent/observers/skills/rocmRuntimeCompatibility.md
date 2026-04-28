# Skill: rocmRuntimeCompatibility

**Use when:** the agent is hitting ROCm/HIP runtime, image, or kernel
compatibility issues — `torch.cuda.is_available() == False` despite a
ROCm image, `HIP error: invalid device function`, missing `gfx` arch,
`miopen` / `rocblas` / `rccl` import errors, or driver/runtime mismatch.

**What to research:**
- The exact `rocm/pytorch:*` tag the run is using and what GPU arches it
  supports (`gfx942` MI300X, `gfx90a` MI250, `gfx1100` RDNA3, etc.).
- Whether the repo's pinned `torch` is ROCm-compatible at all (anything
  before 2.0 typically isn't on modern ROCm).
- Whether `PYTORCH_ROCM_ARCH` or `HSA_OVERRIDE_GFX_VERSION` need to be
  set explicitly.

**What to recommend:**
- Move to a verified `rocm/pytorch` tag from Docker Hub (cite the tag).
- Set the right env vars before importing torch.
- Replace the repo's pinned old torch with the container's pre-installed
  ROCm torch and skip `pip install torch` entirely.

**Tone:** crisp; runtime issues are usually fixed by one strategic
choice (right image, right env var) not by code edits.
