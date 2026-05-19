# Next Research Plan: Causal Memory and Kernel Converter Agent

This plan captures the next two `r2r++` workstreams after the ROCm image ranker and hipify scaffold:

1. Causal memory for migration trajectories.
2. Making the kernel converter agent actually run inside the Repo2ROCm loop.

The goal is to keep these research-grade, not just engineering cleanup.

## Track 1: Causal Migration Memory

### Research Question

Can Repo2ROCm improve across repositories by storing **causal migration transitions** instead of generic memories or successful command snippets?

The current memory stack can store plans, decisions, failures, fixes, and compact observations. The next step is to store typed transitions:

```json
{
  "state": {
    "repo_fingerprint": "torch+flash_attn+custom_cuda",
    "image": "rocm/pytorch:rocm7.2.3_ubuntu22.04_py3.10_pytorch_release_2.10.0",
    "gpu_arch": "gfx942",
    "error_class": "cuda_only_wheel",
    "error_signature": "No module named flash_attn_2_cuda",
    "degradation_policy": "strict"
  },
  "action": {
    "type": "package_strategy",
    "command": "FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE python setup.py install",
    "evidence": ["pypi_versions", "rocm_package_guidance", "torch.version.hip"]
  },
  "outcome": {
    "return_code": 0,
    "verification": ["import flash_attn passed", "GPU smoke test passed"],
    "degradation": "D1",
    "confidence": 0.82
  },
  "counterfactuals": [
    {
      "action": "pip install flash-attn",
      "expected_outcome": "fail",
      "reason": "PyPI wheel is CUDA-only for this stack"
    }
  ]
}
```

### Why This Solves A Real Problem

Free-form memory answers “what happened before?” Causal memory answers:

- What state was I in?
- What action changed that state?
- What evidence justified the action?
- What actually improved?
- What alternative should be avoided next time?

This is the difference between:

> “Install flash-attn from source worked once.”

and:

> “When `flash_attn_2_cuda` appears in a ROCm PyTorch image on `gfx942`, the Triton AMD backend resolves import failure without semantic degradation; direct PyPI install is predicted to fail.”

### Module Plan

Add new models, storage, and distillation around existing learning infrastructure:

- `storage/models.py`
  - Add `CausalState`, `CausalAction`, `CausalOutcome`, `CausalTransition`.
- `storage/kb_store.py`
  - Add a `causal_transitions` table.
  - Query by fingerprint, error class, package, GPU arch, ROCm version, and degradation policy.
- `learning/distiller.py`
  - Extract causal transitions from successful and failed trajectories.
  - Start conservative: only extract transitions when a failure is followed by a successful action and verifier evidence.
- `learning/memory_provider.py`
  - Retrieve top causal transitions during BEGIN and IN phases.
  - Format as structured guidance, not prose.
- `agents/configuration.py`
  - Record richer per-turn state snapshots: image, GPU check, package action, error class, degradation, verifier status.

### First Implementation Slice

Do not try to infer everything at once.

Start with package-level transitions:

- failed import/install error
- action command
- next successful import or GPU smoke test
- degradation flag if any

Initial supported transition classes:

- `cuda_only_wheel_to_rocm_source_build`
- `wrong_image_to_ranked_image_switch`
- `missing_gpu_runtime_to_rocm_base_image`
- `custom_cuda_compile_error_to_hipify_fix`
- `paper_metric_mismatch_to_not_reproduced`

### Acceptance Criteria

- A run can write at least one causal transition JSON record.
- A later run with a similar error can retrieve it.
- The retrieved transition includes evidence and degradation, not just a command.
- Unit tests cover:
  - transition serialization
  - KB insert/query
  - distiller extraction from a small synthetic trajectory
  - memory provider formatting

### Research Metric

Measure:

- reduced repeated failed commands
- fewer turns to resolve repeated error class
- lower false-success rate
- forward transfer across related repos

## Track 2: Kernel Converter Agent

### Research Question

Can a specialist sub-agent improve CUDA-to-HIP migration beyond raw hipify by doing granular correctness repair after transpilation?

This is not a performance optimizer yet. It is a correctness converter:

```text
discover CUDA kernels
run hipify examine/apply
read hipify warnings and compile errors
repair granular correctness issues
re-run compile/import checks
emit evidence and degradation status
```

### Current Starting Point

The scaffold exists in:

- `build_agent/kernel_migration/scaffold.py`

It already supports:

- CUDA source discovery
- kernel purpose classification
- hipify command planning
- granular fix suggestions
- `hipcc` compile-check planning
- dry-run reporting

It is not yet integrated into:

- `main.py`
- `agents/configuration.py`
- `executor/scheduler.py`
- `storage/success_report.py`

### Module Plan

Add a runnable specialist path:

- `kernel_migration/scaffold.py`
  - Keep core dry-run planning.
  - Add sandbox executor adapter.
  - Add JSON report writer.
- `agents/kernel_converter_agent.py`
  - New placeholder name for now.
  - Wraps `KernelMigrationAgent`.
  - Calls an LLM only for suggestions marked `requires_subagent=True`.
  - Applies minimal patches through existing edit mechanisms.
- `agents/configuration.py`
  - Trigger after environment setup when fingerprint has custom CUDA kernels.
  - Run only if repo has `.cu/.cuh` or custom extension build errors.
  - Keep optimization disabled.
- `storage/models.py`
  - Add `KernelMigrationReport` model or reuse scaffold report schema.
- `storage/success_report.py`
  - Include kernel migration status:
    - no kernels
    - hipify planned
    - hipify applied
    - compile passed
    - manual fixes required
    - unsupported

### Execution Policy

The kernel converter should run in phases:

1. **Inventory**
   - Find `.cu`, `.cuh`, CUDA-like `.cpp/.h`.
   - Classify risk flags.

2. **Examine**
   - Run `hipify-clang --examine`.
   - Fall back to `hipify-perl --examine`.
   - Store converted refs, warnings, unsupported APIs.

3. **Apply**
   - Prefer non-in-place output first.
   - Only in-place patch after compile strategy is clear.

4. **Granular Fix**
   - Headers.
   - CUDA macro guards.
   - CUDA library mappings.
   - Warp intrinsics.
   - Inline PTX isolation.
   - Texture/surface APIs.

5. **Verify**
   - Compile with `hipcc`.
   - If PyTorch extension, attempt import/build.
   - Run smallest available smoke test.

6. **Report**
   - Write `kernel_migration_report.json`.
   - Write degradation class.
   - Feed result to causal memory.

### What The Sub-Agent Should Do

The future `KernelConverterAgent` should receive a structured task packet:

```json
{
  "candidate": "src/kernels/attention.cu",
  "hipify_output": "...",
  "compile_error": "...",
  "risk_flags": ["inline_ptx", "warp_size_assumption"],
  "allowed_scope": "correctness_only",
  "forbidden": ["performance tuning", "large rewrites", "mock success"]
}
```

It should return:

```json
{
  "patches": [
    {
      "file": "src/kernels/attention.cu",
      "reason": "cuda_runtime.h must become hip/hip_runtime.h",
      "patch": "..."
    }
  ],
  "verification_command": "hipcc -c ...",
  "expected_risk": "warp-size semantics still require smoke test"
}
```

### Acceptance Criteria

- The main agent can invoke the kernel converter when custom CUDA files are detected.
- The converter produces a report even in dry-run mode.
- At least one real sandbox command path exists for:
  - hipify examine
  - hipify apply
  - compile check
- Manual fixes are emitted as structured suggestions when not safely patchable.
- Unit tests cover:
  - trigger conditions
  - sandbox executor adapter
  - report serialization
  - sub-agent task packet generation

### Research Metric

Measure:

- hipify-only compile success
- hipify + converter compile success
- number of manual fixes required
- degradation class
- turns/cost added by converter

## Combined Story

These two tracks should connect.

Kernel conversion results should become causal memory:

```json
{
  "state": {
    "kernel_risk": "warp_size_assumption",
    "compile_error": "identifier cudaError_t is undefined"
  },
  "action": {
    "type": "kernel_fix",
    "patch_type": "cuda_runtime_header_to_hip_runtime_header"
  },
  "outcome": {
    "compile_passed": true,
    "degradation": "D1"
  }
}
```

This gives the paper a coherent claim:

> Repo2ROCm++ learns causal migration repairs across package, image, and kernel failures, and uses them to guide future hardware-portability experiments.

## Suggested Order

1. Add causal transition schema and storage.
2. Add kernel migration report schema.
3. Wire `KernelMigrationAgent` in dry-run mode after planning.
4. Add sandbox executor adapter for hipify examine only.
5. Add compile-check execution.
6. Add LLM sub-agent task packets for `requires_subagent=True` fixes.
7. Distill successful kernel repairs into causal memory.
