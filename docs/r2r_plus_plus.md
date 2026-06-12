# Repo2ROCm++ Additions and Roadmap

This branch, `r2r++`, turns the original Repo2ROCm flow into a more evidence-driven ROCm migration prototype. The normal branch already has the main Repo2Run-derived loop: clone, fingerprint, plan, configure in Docker, observe, export Dockerfile, and distill trajectories. This branch adds early pieces for research-grade ROCm image selection and correctness-first CUDA kernel migration.

## What This Branch Adds

### Dynamic ROCm Image Selection

New module:

- `build_agent/images/rocm_ranker.py`

The ranker treats ROCm Docker selection as a retrieval/ranking problem. Given repo imports, config files, optional Python version, and optional GPU architecture hint, it ranks ROCm image candidates such as:

- `rocm/pytorch`
- `rocm/pytorch-training`
- `rocm/vllm`
- `rocm/sgl-dev`
- `rocm/jax`
- `rocm/tensorflow`
- `rocm/onnxruntime`
- `rocm/megatron-lm`

The ranker uses cheap, no-pull metadata first:

- DockerHub live tags
- tag dates and digests
- compressed image size
- tag-parsed ROCm, Python, PyTorch, vLLM, JAX, or TensorFlow versions
- static known preinstalled package hints
- GPU architecture tokens encoded in image tags

It does **not** pretend to know the full installed package list unless a future expensive probe supplies it. That keeps planning fast for 10-25GB ROCm images.

### Jaccard/Compatibility Scoring

The image ranker builds a rough desired package/feature set from repo signals:

- imports
- `requirements.txt`
- `pyproject.toml`
- config text
- framework markers such as `torch`, `vllm`, `sglang`, `jax`, `tensorflow`, `deepspeed`, `flash_attn`, `xformers`, and `triton`

It then compares those against inferred image inventory tokens using Jaccard overlap and adds scoring terms for:

- preferred workload match
- live tag availability
- Python version match
- GPU architecture match
- image size penalty
- floating-tag penalty in strict mode

The planner now surfaces the top image candidates, overlap/missing tokens, risks, live tag evidence, and final tag choice in `plan.txt`.

### Context-Aware ROCm Package Guidance

New module:

- `build_agent/knowledge/rocm_dynamic.py`

CUDA-sensitive package advice now branches by:

- model stack, such as LLM serving, LLM training, diffusion, JAX, TensorFlow, or generic PyTorch
- GPU architecture hint, such as CDNA/MI300/gfx942 or RDNA/gfx11/gfx12
- degradation policy, such as strict paper reproduction or permissive environment smoke test

Example behavior:

- `flash-attn` on unknown/RDNA architecture prefers Triton AMD backend first.
- strict mode refuses silent SDPA/eager fallback unless the result records degradation.
- permissive env mode allows SDPA fallback if the goal is only ROCm environment verification.

### Structured DockerHub Tag Lookup

Updated module:

- `build_agent/tools/external_lookups.py`

Added `dockerhub_tags_structured()`, which returns raw tag records for planner-side ranking instead of only prompt-facing text. This is used by the image ranker.

### Correctness-Only Kernel Migration Scaffold

New package:

- `build_agent/kernel_migration/`

The scaffold introduces a future specialist lane for CUDA-to-HIP migration:

1. Discover `.cu`, `.cuh`, and CUDA-like C++ files.
2. Classify kernel purpose, such as attention, normalization, quantization, optimizer, or fused operation.
3. Generate `hipify-clang --examine` / `hipify-perl --examine` commands.
4. Generate conservative hipify apply commands.
5. Surface granular post-hipify repair work items:
   - CUDA runtime headers
   - CUDA fp16 headers
   - cuBLAS/cuSPARSE/cuRAND/NCCL library mappings
   - inline PTX
   - warp-size assumptions
   - CUDA preprocessor guards
   - warp shuffle/vote intrinsics
   - texture/surface APIs
6. Generate correctness-only `hipcc` compile checks.

This is a scaffold for a future "CUDA Surgeon" sub-agent. It does not optimize kernels yet and intentionally returns patch hints rather than blindly editing source code.

### Tests Added

New focused tests:

- `tests/test_rocm_dynamic.py`
- `tests/test_rocm_image_ranker.py`
- `tests/test_kernel_migration_scaffold.py`

These cover:

- live tag selection logic
- model stack detection
- dynamic package guidance by arch/degradation policy
- ROCm image ranking for vLLM and generic PyTorch repos
- CUDA source discovery and granular fix suggestions
- dry-run hipify/compile planning

## Planned Research Additions

### 1. Cost-Aware Image Introspection

Full image probing is expensive because ROCm images are often 10-25GB. The planned policy is tiered:

1. Use no-pull metadata for all candidates.
2. Rank all candidates cheaply.
3. Pull/probe only the top candidate if uncertainty is high or the run is strict paper reproduction.
4. Cache probe results by image digest.

Research claim:

> Cost-aware environment introspection reduces wrong-image retries without paying the full image-pull cost for every candidate.

### 2. Image Inventory Cache

Future expensive probes should write a digest-keyed inventory:

```json
{
  "image": "rocm/pytorch:rocm7.2.3_ubuntu22.04_py3.10_pytorch_release_2.10.0",
  "digest": "sha256:...",
  "python": "3.10",
  "rocm": "7.2.3",
  "torch": "2.10.0",
  "packages": ["torch", "triton", "numpy"],
  "probe_cost_seconds": 812
}
```

This should feed future Jaccard scoring as true inventory rather than inferred inventory.

### 3. CUDA Surgeon Sub-Agent

The kernel migration scaffold should become an actual specialist agent. Its job:

1. Read hipify output and compile errors.
2. Inspect exact local code.
3. Apply minimal correctness patches.
4. Re-run compile/import checks.
5. Stop once correctness is reached.

No performance optimization in this phase.

Research claim:

> Granular post-transpilation repair improves CUDA-to-HIP success over hipify-only migration.

### 4. Migration Strategy Tournament

Instead of one migration strategy, run a small competition between branches:

- hipify-clang first
- hipify-perl first
- source ROCm fork first
- framework fallback first
- base-image switch first

Each branch is scored by correctness, degradation, and cost. Only the best branch continues.

Research claim:

> Test-time strategy search improves hardware migration reliability under uncertainty.

### 5. Degradation Ledger

Every fallback should receive a degradation class:

- `D0`: no degradation
- `D1`: equivalent ROCm backend
- `D2`: slower but same semantics
- `D3`: custom kernel replaced with framework op
- `D4`: acceleration disabled
- `D5`: experiment semantics changed

This prevents false success where a repo "runs" but no longer reproduces the original GPU path.

### 6. Causal Trajectory Memory

Store state-action-outcome triples rather than raw memories:

```json
{
  "state": {
    "image": "rocm/pytorch:...",
    "gpu_arch": "gfx942",
    "error": "No module named flash_attn_2_cuda"
  },
  "action": "install flash-attn with FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE",
  "outcome": {
    "import_success": true,
    "degradation": "D1"
  }
}
```

Research claim:

> Causal memory transfers migration fixes better than free-form trajectory summaries.

### 7. Evidence Passport

Every final Dockerfile or patch should include a compact evidence bundle:

- chosen base image and why
- live tag evidence
- image ranking score
- package guidance evidence
- CUDA migration fixes applied
- degradation level
- verification commands and outputs

Research claim:

> Evidence-carrying agent actions make software migration auditable and reduce hallucinated configuration claims.

## Near-Term Engineering Checklist

- Add optional CLI flags for GPU architecture hints.
- Add a cache table for image ranker metadata by image digest.
- Add a planner section that explicitly says whether image ranking used inferred or probed inventory.
- Wire `KernelMigrationAgent` as a callable specialist after env setup detects custom CUDA kernels.
- Keep optimization out of the first kernel lane; correctness first.
