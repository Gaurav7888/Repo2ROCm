# Repo2ROCm

**An agentic system for reproducing research code on AMD ROCm GPUs — with zero manual intervention.**

Reproducing a published ML result from its GitHub repository is rarely a clone-and-run
exercise, and it gets dramatically harder when the target hardware differs from the
platform the code was written for. Most public research code assumes CUDA implicitly;
running it on AMD ROCm surfaces dependencies with no packaged equivalent, incompatible
builds, embedded device assumptions, and the common failure mode where installation
*appears* to succeed while execution silently falls back to the CPU.

**Repo2ROCm** takes a repository URL (and, for result reproduction, the accompanying
paper) and autonomously produces a working ROCm environment, verifies that the intended
workload actually runs **on the GPU**, and — when asked — executes the paper's experiment
and issues a deterministic verdict on whether the reported result reproduces. Every run
emits an auditable artifact bundle: a reproducible ROCm `Dockerfile`, a source-code patch
in unified-diff format, execution logs, extracted metrics, and the full agent trajectory.

On a benchmark of **60 repositories** from ICLR, ICML, ACL, and NeurIPS, only **18/60**
run out of the box on ROCm hardware. Repo2ROCm raises functional reproduction to
**59/60**.

> Repo2ROCm extends the [Repo2Run](https://github.com/bytedance/Repo2Run) environment-
> reconstruction framework (Bytedance, 2025) into a hardware-aware, evidence-gated
> reproducibility system for heterogeneous GPU ecosystems.

---

## Table of Contents

- [Why this is hard](#why-this-is-hard)
- [What Repo2ROCm does differently](#what-repo2rocm-does-differently)
- [Operating modes](#operating-modes)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Batch Processing](#batch-processing)
- [Architecture](#architecture)
- [Artifact bundle](#artifact-bundle)
- [Evaluation & Results](#evaluation--results)
- [Benchmark harness](#benchmark-harness)
- [Repository layout](#repository-layout)
- [Design decisions](#design-decisions)
- [Limitations](#limitations)
- [License](#license)

---

## Why this is hard

Reproduction failures arise at three escalating tiers:

1. **Software reproducibility.** Incomplete docs, stale dependency pins, conflicting
   build configs, and hidden external-service requirements.
2. **Research reproducibility.** A single project may ship `requirements.txt`,
   `setup.py`, `pyproject.toml`, conda envs, Dockerfiles, *and* lockfiles that disagree.
   Experimental configs are scattered, the correct entry point is rarely obvious, and
   multiple frameworks (PyTorch, JAX, TensorFlow, vLLM, SGLang) may coexist.
3. **Cross-platform migration.** When the target is AMD ROCm but the code assumes CUDA,
   platform gaps appear: packages with no packaged ROCm equivalent (`flash-attn`,
   `bitsandbytes`, `xformers`, `pynvml`), a non-trivial base-image choice, and source
   code that hard-codes `device="cuda"`, custom `.cu` kernels, or `torch.backends.cudnn`
   tweaks.

The subtle problem is **verification**: successful installation does **not** imply
successful migration. A repo can install cleanly yet silently run on the CPU. Establishing
a valid migration requires *evidence of actual GPU execution*, not just dependency
resolution.

---

## What Repo2ROCm does differently

General-purpose coding agents can usually make a repo "run" on AMD hardware, but they tend
to prioritize completing the task over preserving the **original execution path**. Observed
failure modes include treating a successful `pip install` (or a `--help` screen) as
verification, silently disabling FlashAttention instead of installing a ROCm-compatible
backend, and falling back to a slower CPU/SDPA path. The result looks runnable but no longer
matches the experiment the paper evaluated.

Repo2ROCm treats migration as an **evidence-driven reproducibility task**:

- **Two-phase design** — a cheap static planner (outside Docker) followed by a guarded
  ReAct executor (inside an isolated Docker sandbox).
- **Evidence-gated execution** — irreversible actions (switching images, replacing
  CUDA-dependent packages, declaring GPU success, declaring a paper reproduced) are
  **blocked until supporting runtime evidence is collected**.
- **Deterministic verification** — final reproduction verdicts are owned by a
  side-effect-free verifier, not the language model. The LLM cannot override it.
- **Auditable artifacts** — every decision is recorded as a reproducible Dockerfile,
  unified-diff patch, command log, metrics, and trajectory.

---

## Operating modes

The paper describes three settings; the CLI exposes them via `--mode`:

| Paper term | CLI | Goal | Success marker |
|------------|-----|------|----------------|
| **Mode 0** | *(no agent)* | Out-of-the-box baseline: clone & run per the repo's own docs | — |
| **Mode 1** | `--mode env` *(default)* | **Functional reproduction** — reconstruct the env and verify a real workload runs on the GPU | `ROCM_ENV_VERIFIED` |
| **Mode 2** | `--mode reproduce` | **Result reproduction** — run the paper's exact experiment and compare metrics under tolerance | `PAPER_RESULT_REPRODUCED` / `PAPER_RESULT_NOT_REPRODUCED` |
| Mode 3 | `--mode full` | Mode 1 **then** Mode 2 (both outputs produced) | both markers |

In `env` mode the agent may use scaled-down smoke parameters; `reproduce` automatically
enables `--no-scale-down` so the experiment runs the exact paper/README configuration. The
legacy flag `--reproduce-results` is an alias for `--mode full`.

---

## Quick Start

### Prerequisites

- A host with **AMD ROCm GPUs** and the `amdgpu` driver (ROCm 6.x/7.x).
- **Docker** with ROCm device passthrough (`--device=/dev/kfd --device=/dev/dri`).
- **Python 3.8+** on the host that drives the agent.
- An **AMD LLM API Gateway key** (Claude backbone) — or an OpenAI key, or the `claude` CLI
  / `ANTHROPIC_API_KEY` for Claude Code mode.

### Installation

```bash
git clone <repo-url> Repo2ROCm
cd Repo2ROCm
pip install -r requirements.txt
```

`requirements.txt` covers the core stack (`docker`, `pexpect`, `openai`, `requests`,
`tenacity`, `rich`, `pandas`). Optional extras degrade gracefully if missing:
`pymupdf` (paper PDF to text), `mempalace` + `chromadb` (per-run memory layer), and
`ddgs` + `html2text` (in-loop web search / URL fetch).

### Run on a single repository (Mode 1 — functional)

```bash
export AMD_LLM_API_KEY=...                 # or use --api-key

SHA=$(git ls-remote https://github.com/user/repo HEAD | cut -f1)

python build_agent/main.py \
  --full_name "user/repo" \
  --sha "$SHA" \
  --root_path . \
  --llm "claude-sonnet-4" \
  --rocm \
  --mode env \
  --gpu-index 0
```

### Reproduce a paper result (Mode 2)

```bash
python build_agent/main.py \
  --full_name "user/repo" \
  --sha "$SHA" \
  --root_path . \
  --llm "claude-sonnet-4" \
  --rocm \
  --mode reproduce \
  --paper-url "https://arxiv.org/pdf/2505.XXXXX.pdf" \
  --gpu-index 0
```

If `--paper-url` / `--paper-pdf` is omitted, the planner auto-discovers the arXiv link
from the repo's README.

---

## CLI Reference

`build_agent/main.py` flags:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--full_name` | string | *required* | GitHub repo (`owner/name`) |
| `--sha` | string | *required* | Git commit SHA to check out |
| `--root_path` | string | *required* | Working dir for clones, output, and the KB |
| `--llm` | string | `gpt-4o-2024-05-13` | LLM model name (use `claude-sonnet-4` for the AMD gateway) |
| `--rocm` | flag | `false` | Enable AMD ROCm migration mode |
| `--rocm-base-image` | string | auto-selected | Override the planner's ROCm base image (e.g. `rocm/pytorch:latest`) |
| `--api-key` | string | `$AMD_LLM_API_KEY` | AMD LLM API Gateway key |
| `--mode` | `env`/`reproduce`/`full` | `env` | Run mode (see [Operating modes](#operating-modes)) |
| `--reproduce-results` | flag | `false` | Legacy alias for `--mode full` |
| `--paper-url` | string | auto | Direct URL to the paper PDF (e.g. an arXiv PDF link) |
| `--paper-pdf` | string | — | Local path to the paper PDF (overrides `--paper-url`) |
| `--paper-source-mode` | `pdf`/`html`/`both` | `both` | How the paper corpus is built for planning/retrieval |
| `--no-scale-down` | flag | `false` | Follow the README exactly — no param scale-down, no mock data (implied by `reproduce`) |
| `--gpu-index` | int | — | GPU slot for sandbox isolation (sets `HIP_VISIBLE_DEVICES` etc.) |
| `--optimize-kernels` | flag | `false` | Reserve a kernel-optimization phase for CUDA/Triton kernels *(experimental)* |
| `--kb-path` | string | `<root_path>/kb/repo2rocm.db` | Path to the SQLite knowledge base |
| `--use-claude-code` | flag | `false` | Use the Claude Code Agent SDK / CLI instead of the AMD gateway (sub-agents, skills, built-in tools) |
| `--claude-code-model` | string | — | Model for Claude Code mode (`sonnet`, `opus`, `haiku`) |
| `--claude-code-agentic` | flag | `false` | Let Claude Code drive the whole configuration autonomously |
| `--verbose` | flag | `false` | Print full LLM prompts, responses, and context tables |

---

## Batch Processing

Run many repositories with one process per GPU:

```bash
cat > batch.sh << 'EOF'
python -u build_agent/main.py --full_name "org/repo1" --sha "abc123" --root_path . --llm "claude-sonnet-4" --rocm --gpu-index 0
python -u build_agent/main.py --full_name "org/repo2" --sha "def456" --root_path . --llm "claude-sonnet-4" --rocm --gpu-index 1
EOF

python build_agent/multi_main.py batch.sh
```

`multi_main.py` runs commands concurrently, monitors host disk usage, halts above a safe
threshold, and cleans up dangling containers between runs. For full benchmark sweeps with
per-GPU scheduling, progress tracking, scoring, and reporting, use the
[benchmark harness](#benchmark-harness) instead.

---

## Architecture

### Pipeline overview

```
Repository URL (+ SHA)  [+ paper URL/PDF for Mode 2]
        |
        v
  Clone & fingerprint     git clone @ SHA, pipreqs, BuildFingerprint
        |                 (frameworks, CUDA deps, build system, custom kernels)
        v
  PHASE 1 - Static Planner   (outside Docker, one shot)
        |   analyze imports/configs/README; reconstruct & classify deps;
        |   rank ROCm base images; detect hazards; [Mode 2] extract experiment+metrics
        |   -> plan.txt + recommended image + paper experiment list
        v
  PHASE 2 - Guarded Executor   (ReAct, 1 action/turn, GPU-pinned Docker sandbox)
        |   Hard Guards . ROCm KB . Causal Memory . Observer Sidecar . Deterministic Tools
        |   checkpoint (docker commit) on success / rollback on failure
        v
   Mode 1: ROCM_ENV_VERIFIED        Mode 2: Deterministic Verifier ->
   (real workload on the GPU)       PAPER_RESULT_REPRODUCED / _NOT_REPRODUCED
        |
        v
  integrate_dockerfile() + generate_diff()
  -> Dockerfile . patch . logs . metrics . trajectory
        |
        v
  Post-run learning (KB rules + causal transitions)
```

Separating planning from execution lowers agent cost, avoids unnecessary container
launches, and resolves many migration decisions before any code runs.

### Phase 1 — Static Planner

`build_agent/agents/planner.py` → `generate_plan(...)` performs deep static analysis
**before any Docker command runs** and returns `(plan_text, recommended_image,
paper_context)`. It reads config files (`requirements*.txt`, `setup.py`,
`pyproject.toml`, `Pipfile`, `environment.yml`, CI workflows, `.python-version`), the
README, and every Python file's imports. From this it:

- detects the dominant framework/runtime stack and workload type;
- reconstructs and **classifies dependencies** by ROCm compatibility, dropping banned
  NVIDIA wheels and mapping CUDA-only packages to ROCm install recipes;
- flags migration hazards (Python 3.12 stdlib removals, stale version pins, `cudnn`
  flags, hard-coded `device="cuda"`, wandb, large epochs, external asset downloads);
- ranks and recommends a ROCm base image (see [ROCm knowledge base](#rocm-knowledge-base));
- for **Mode 2**, uses `PaperAgent` + the indexed paper corpus to extract the target
  experiment, expected metrics, and evaluation criteria.

The plan is printed, saved to `output/<owner>/<repo>/plan.txt`, and injected into the
executor's system prompt so it can act from turn 1.

### Phase 2 — Guarded Executor

`build_agent/agents/configuration.py` → `Configuration` is a ReAct-style control loop
running **inside the Docker sandbox**. Each turn the LLM emits a **Thought** and exactly
**one Action** (one ` ```bash ``` ` block *or* one ` ```diff ``` ` block); the sandbox
executes it and feeds back the observation.

| Parameter | Value |
|-----------|-------|
| Max turns | **100** (passed by `main.py`; constructor default 70) |
| Actions per turn | **1** (only the first bash block runs; mixed bash+diff is rejected) |
| Token budget | ~150,000 (older messages trimmed FIFO) |
| Verification (ROCm) | `runtest`/`poetryruntest` are **disabled** — the agent must run the repo's real entry point and prove GPU use |

The executor has a deterministic tool suite (advertised in `utils/tools_config.py`):

| Tool | Purpose |
|------|---------|
| `waitinglist add/addfile/clear/show` | Queue pip/apt dependencies for batch install |
| `conflictlist solve/clear/show` | Resolve version-constraint conflicts |
| `download` | Batch-install everything queued in the waiting list |
| `change_python_version` / `change_base_image` / `clear_configuration` | Rebuild the container on a new Python/image or reset |
| `pypi_versions` / `dockerhub_tags` | Live PyPI / Docker Hub metadata lookups (7-day cache) |
| `web_search` / `visit_url` / `deep_research` | DuckDuckGo search, URL→markdown, bounded multi-step research |
| `graphify_query` / `mem_recall` / `paper_recall` | Code-graph search, per-run memory recall, paper-context recall |
| `verify_paper_result` | **Deterministic** Mode 2 metric verifier |
| ` ```diff ``` ` blocks | SEARCH/REPLACE source patches applied via `tools/code_edit.py` |

### Evidence-gated execution (hard guards)

The core novelty: irreversible or high-stakes actions are **blocked until the agent has
collected supporting runtime evidence**. Enforced in `Configuration._maybe_run_hard_guard()`:

| Guard | Blocked action | Required evidence first |
|-------|----------------|-------------------------|
| **A** | `change_base_image` | a prior `dockerhub_tags` lookup for that image |
| **B** | risky `pip install` (`flash-attn`, `xformers`, …) | a prior `pypi_versions` lookup |
| **C** | echo `ROCM_ENV_VERIFIED` | an observed GPU check (`rocm-smi` / `torch.cuda.is_available()`) |
| **D** | echo `PAPER_RESULT_*` | a prior `verify_paper_result` run |
| **E** | run the Mode 2 experiment | prior paper-evidence retrieval (`paper_recall` / `graphify_query --scope paper`) |
| **F** | broad `find` / `grep` discovery | a prior `graphify_query --scope code` |

This turns tool usage from a *recommendation* into an *enforceable requirement*, and is
the mechanism that prevents silent CPU fallback, SDPA substitution, and "install ≠ success"
gaming. Deterministic and advisory **rules** (`rules/engine.py`; thresholds 0.85
deterministic / 0.4 advisory, ≥3 evidence) and an **error classifier**
(`errors/classifier.py`) supply known fixes alongside the guards.

### Observer sidecar

`build_agent/observers/` runs an **asynchronous reviewer** in a separate subprocess that
communicates with the main loop over JSONL buses — no direct coupling. Each turn snapshot
passes a cheap heuristic gate (`Reviewer.should_consider`); if something looks wrong, a
single structured LLM call (`Reviewer.decide`) selects **one skill** and emits advice with
cooldowns. Skills are markdown cards in `observers/skills/`: `progressOK` (stay silent when
healthy), `dependencyRepair`, `externalAssetDownload`, `modelAssetReadiness`,
`benchmarkPathing`, `explorationStuck`, `frameworkApiDrift`, `rocmRuntimeCompatibility`,
and `paperReproduction`.

### Causal memory & cross-run learning

Repo2ROCm accumulates **typed causal transitions** — `state → action → outcome` priors —
in a SQLite knowledge base (`storage/kb_store.py`, table `causal_transitions`), rather than
free-form prose:

- **State** — repo fingerprint, image, GPU arch, error class/signature, degradation policy.
- **Action** — type (`package_strategy`, `image_switch`, `kernel_fix`, …), command, evidence.
- **Outcome** — return code, verification status, degradation level (`D0`–`D3`), confidence.

At plan time the `BuildMemoryProvider` BEGIN phase injects similar successful builds (by
fingerprint), applicable rules, package compatibility, and relevant causal transitions; the
IN phase does the same per-turn after classifying the current error. After each run, the
`TrajectoryDistiller` conservatively extracts new transitions (requires `ROCM_ENV_VERIFIED`
and a failure→success pair within ≤3 turns) and updates install paths and rule confidence.
The KB is seeded with expert error patterns, rules, and causal classes on first use.

> Supporting layers: a per-run trace store (`learning/mempalace_provider.py`, optional
> ChromaDB), an observation compactor (`learning/observation_compactor.py`), and a static
> code-/paper-graph index (`learning/graphify_provider.py`).

### Deterministic verifier

`build_agent/tools/verify_paper_result.py` is a side-effect-free verifier that **owns the
Mode 2 verdict**:

```python
verify_paper_result(log_path, metrics, tolerance="", direction="", max_log_chars=200_000)
# returns (verdict_json, return_code, details); rc == 0 iff ALL metrics reproduced
```

It reads the experiment log on the host, extracts each metric via regex templates + aliases
(using the *last* matching value), parses the tolerance (`<=15%`, `<=3 abs pts`; default
**10% relative**), applies a direction (`higher_is_better` / `lower_is_better` / `equal`),
and emits a structured verdict. The agent **cannot override** a `reproduced`/`not_reproduced`
decision. A `SuccessReport` (`storage/success_report.py`) summarizes each run as
`0.6·goal + 0.2·env + 0.2·process`, distinguishing "paper didn't reproduce" from
"environment failed" from "agent was inefficient".

### ROCm knowledge base

`build_agent/knowledge/` encodes AMD-specific migration knowledge injected into the prompt
(avoiding noisy/slow web search for well-understood failures):

- **Image catalog** (`rocm_knowledge.py`) — specialized ROCm images keyed by workload:
  `rocm/sgl-dev` (SGLang), `rocm/vllm-dev`, `rocm/vllm`, `rocm/jax`, `rocm/tensorflow`,
  `rocm/onnxruntime`, `rocm/pytorch-training` (DeepSpeed/FSDP), `rocm/megatron-lm`, and
  `rocm/pytorch` (default).
- **CUDA→ROCm mappings** — install recipes/notes for `flash-attn` (AMD Triton backend via
  `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`), `bitsandbytes`, `xformers`, `pynvml`, etc.
- **Banned NVIDIA packages**, **per-image pre-installed package lists**, **code patterns**
  (`nvidia-smi`→`rocm-smi`), and **supported GPU archs** (gfx908/90a/942/950, RDNA).
- **Live signals** (`rocm_dynamic.py`) — GPU-arch detection, Jaccard image selection, and a
  per-dependency degradation policy (`strict`/`moderate`/`permissive`) chosen by run mode.
- **AMD ecosystem catalog** (`amd_rocm_repos.py`, `amd_rocm_ecosystem.md`) — rocBLAS,
  hipBLAS, HIPIFY, etc., filtered by the repo's imports.

### Sandbox, checkpoint & rollback

`build_agent/utils/sandbox.py` manages the container lifecycle over an interactive
`pexpect` shell that persists across turns. Before each **non-safe** command the sandbox
checkpoints via `docker commit`; if a **destructive** command fails, it restores from the
last good image (`switch_to_pre_image`). Read-only commands, heredoc writes, and plain
`python` runs are exempt from rollback. `docker commit` is used (not full snapshots) for
low-latency recovery. `main.py` also retries up to 3 alternative base images if a chosen
image's container crashes at startup, recording the bad image in the KB per host arch.

### Kernel migration

For repositories with custom GPU kernels, the fingerprinter flags `.cu/.cuh` files and
`@triton.jit` usage and the planner notes hipification in the plan. Dedicated agents exist —
`agents/cuda_kernel_agent.py` (inventory → `hipify-clang`/`hipify-perl` → `hipcc` compile →
numerical check) and `agents/triton_kernel_agent.py` (AMD warp-size/block/stage checks and
autotune-config patches), plus a dry-run pipeline scaffold in `kernel_migration/scaffold.py`.

> **Status:** kernel hipification and the `--optimize-kernels` performance phase are
> experimental — the agents and scaffold are implemented but not yet auto-invoked from the
> main loop, and the CUDA→HIP path prioritizes functional correctness over runtime
> optimization.

---

## Artifact bundle

Every run writes `output/<owner>/<repo>/` (the paper refers to this as
`verification/<repo>/`):

```
output/<owner>/<repo>/
├── Dockerfile               # reproducible ROCm environment — docker build -t myenv .
├── plan.txt                 # the static planner's strategic plan
├── run_mode.txt             # which mode this run used
├── patch/                   # source-code modifications as unified diffs
├── inner_commands.json      # every command executed inside the container
├── outer_commands.json      # LLM-call timings / metadata
├── track.json               # full agent conversation history
├── test.txt                 # success markers (ROCM_ENV_VERIFIED, PAPER_RESULT_*)
├── paper_reproduction.json  # verifier verdict + SuccessReport (Mode 2 only)
├── sha.txt                  # git SHA used
├── trajectory.jsonl         # post-run learning trace
└── agent_debug_log.txt      # rich-formatted debug log
```

This auditable record means every environment change, patch, and decision is inspectable —
in contrast to manual setups where important changes go undocumented.

---

## Evaluation & Results

All experiments run on a host with eight **AMD Instinct MI250X/MI250** GCDs (`gfx90a`,
64 GB HBM2e, ~1.6 TB/s peak bandwidth each), **ROCm 7.1.2** (amdgpu driver 6.16.6). The
base image is selected per repository by the planner. The agent uses **Claude Sonnet 4** as
its backbone, capped at **100 turns** per run with no fixed token budget. Runs are scored
with a deterministic rubric plus degradation indicators (attention fallbacks, environment
substitutions, reduced execution scale). *Functional reproduction* = a verified GPU-backed
execution environment, regardless of whether the headline research result reproduces.

### Functional reproduction (60 repositories)

The benchmark spans 60 repositories from **ICLR, ICML, ACL, and NeurIPS**. Only **18/60**
run out of the box (**Mode 0**); Repo2ROCm (**Mode 1**) raises this to **59/60** — more than
tripling the baseline. The sole remaining failure, *DiffusionVeteran*, fails under both.

✅ = verified GPU-backed execution &nbsp;·&nbsp; ❌ = failed

| Repository | M0 | M1 | Repository | M0 | M1 | Repository | M0 | M1 |
|------------|:--:|:--:|------------|:--:|:--:|------------|:--:|:--:|
| GCTM | ❌ | ✅ | LayerDAG | ❌ | ✅ | HARDMath | ✅ | ✅ |
| mm-argfallacy | ❌ | ✅ | EARTH | ❌ | ✅ | ELABORATION | ✅ | ✅ |
| understanding_mcqa | ✅ | ✅ | Knowledge-Entropy | ❌ | ✅ | VTI | ❌ | ✅ |
| WildBench | ❌ | ✅ | convolutional_diffusion | ❌ | ✅ | modelmap | ✅ | ✅ |
| DiffusionVeteran | ❌ | ❌ | AlphaRec | ❌ | ✅ | TimeEmb | ❌ | ✅ |
| Self-Certainty | ❌ | ✅ | pcx | ✅ | ✅ | CEB | ❌ | ✅ |
| Ladder | ❌ | ✅ | paperbanana | ✅ | ✅ | ACL25-CoPE | ❌ | ✅ |
| TOP_ERL_ICLR25 | ✅ | ✅ | SeqSNN | ✅ | ✅ | PrefixKV | ❌ | ✅ |
| DataEnvGym | ❌ | ✅ | MCNC | ❌ | ✅ | CLEME | ✅ | ✅ |
| m-rewardbench | ❌ | ✅ | Risk-Sensitive-CMDP | ✅ | ✅ | FR-Spec | ❌ | ✅ |
| askqe | ✅ | ✅ | HarmoniCa | ❌ | ✅ | moa | ❌ | ✅ |
| HealthGPT | ❌ | ✅ | topoloss | ✅ | ✅ | SemEval2025-EAMT | ❌ | ✅ |
| CATCH | ❌ | ✅ | llm_reproducibility | ❌ | ✅ | ORMind | ❌ | ✅ |
| CSL-Mem | ❌ | ✅ | torch_brain | ❌ | ✅ | AIRMVC | ❌ | ✅ |
| TemporalHead | ❌ | ✅ | IoA | ✅ | ✅ | Stacey | ❌ | ✅ |
| NLPromptEval | ❌ | ✅ | c-seo-bench | ✅ | ✅ | BRIGHT | ❌ | ✅ |
| luckmatters | ❌ | ✅ | LCPO | ✅ | ✅ | CoMRes | ❌ | ✅ |
| AdaKV | ❌ | ✅ | gated_attention | ✅ | ✅ | Effort-AIGI-Detection | ❌ | ✅ |
| VerbalizED | ❌ | ✅ | jl-metric | ✅ | ✅ | SelfElicit | ❌ | ✅ |
| HeadKV | ❌ | ✅ | bookcoref | ✅ | ✅ | NeurIPS2025-KFF | ❌ | ✅ |

**Totals:&nbsp; Mode 0 (out of the box) 18/60 &nbsp;→&nbsp; Repo2ROCm (Mode 1) 59/60.**

Four recurring failure modes that affect general-purpose agents motivated the
evidence-gated design: (1) picking images that don't support the target ROCm/GPU; (2)
silently disabling accelerator components (e.g. FlashAttention) instead of installing a
ROCm-compatible alternative; (3) incorrect API adaptations from version mismatches or
hallucinated signatures; (4) falling back to alternative implementations (e.g. SDPA) that
change the evaluated code path. When a repo exceeds GPU memory, Repo2ROCm applies controlled
scaling (smaller batch / shorter generation) while preserving the model and execution path
where possible.

### Comparison with general-purpose agents

On five repositories, general-purpose agents (Cursor, Claude Code, Mini-SWE-Agent) run
nearly everything — but several runs complete along an execution path that differs from the
paper's. Repo2ROCm completes all five while **preserving the original execution path**
(installing the AMD Triton FlashAttention backend rather than substituting SDPA, and
surfacing incorrect-signature calls before they're accepted).

✅ = ran &nbsp;·&nbsp; ❌ = failed

| Repository | Out of the box | Cursor | Claude Code | Mini-SWE-Agent | Repo2ROCm |
|------------|:--:|:--:|:--:|:--:|:--:|
| Knowledge Entropy | ❌ | ✅ ᵃ | ✅ ᵃ | ✅ ᵃ | ✅ |
| AlphaRec | ❌ | ✅ | ✅ | ✅ | ✅ |
| AnyEdit | ❌ | ✅ | ✅ | ✅ | ✅ |
| PrefixKV | ❌ | ✅ ᵃ | ✅ ᵃ | ✅ ᵃ′ᵇ | ✅ |
| UniZyme | ✅ | ✅ | ✅ | ✅ ᶜ | ✅ |

Superscripts mark runs whose execution path differs from the paper's:
ᵃ FlashAttention replaced with PyTorch SDPA (AMD Triton FlashAttention backend not used);
ᵇ a library function called with an incorrect signature;
ᶜ a module (frustratometer) left disabled. Repo2ROCm preserved the original path on all five.

### Result reproduction (Mode 2)

Mode 2 is evaluated on two repositories with a single, clearly measurable headline metric.
Verdicts are owned by the deterministic verifier, not the LLM.

| Repository | Reported | Measured (MI250X) | Verifier verdict |
|------------|----------|-------------------|------------------|
| **TurboQuant** | compression 5.1 · cosine 0.9996 · top-1 88.9 · top-5 97.2 | identical (0.00% deviation, all 4 within tolerance) | `PAPER_RESULT_REPRODUCED` |
| **dKV-Cache** | 2.09× speedup | 1.02× speedup | `PAPER_RESULT_NOT_REPRODUCED` |

For **TurboQuant**, all four reported metrics reproduce within tolerance at 0.00% deviation.
For **dKV-Cache**, the method runs correctly but the headline *speedup* does not reproduce on
our configuration — and the verifier flags it rather than reporting a false success. This is
not a migration artifact: in Mode 2 the agent runs under a strict policy that forbids SDPA
fallback and scale-down, so the intended code path executed. The original speedup was reported
on an NVIDIA H20 (~4.0 TB/s peak bandwidth), roughly 2.5× the ~1.6 TB/s of a single
MI250X/MI250 GCD; since the mechanism is bandwidth-sensitive, this gap is a likely
contributing factor. The key point: Repo2ROCm produced an **auditable refutation**, not a
fabricated pass or a generic failure.

### Causal-memory ablation

Ablating causal memory over 20 CUDA repositories (full system vs. an identical configuration
with causal memory disabled):

✅ ran &nbsp;·&nbsp; ❌ failed &nbsp;·&nbsp; † timeout

| Repository | Baseline | Causal-mem | Repository | Baseline | Causal-mem |
|------------|:--:|:--:|------------|:--:|:--:|
| luckmatters | ✅ | ✅ | LLaDA | ✅ | ✅ |
| llm_reproducibility | ❌ | ❌ | Show-o | ✅ | ✅ |
| torch_brain | ✅ | ✅ | PrefixKV | ✅ | ❌ † |
| HarmoniCa | ✅ | ✅ | paperbanana (4KAgent) | ✅ | ✅ |
| Ladder | ✅ | ✅ | TemporalHead | ✅ | ✅ |
| TimeEmb | ✅ | ✅ | gated_attention | ❌ † | ✅ |
| askqe | ✅ | ✅ | Self-Certainty | ✅ | ✅ |
| GCTM | ❌ | ✅ | DataEnvGym | ❌ | ❌ |
| Stacey | ✅ | ✅ | CATCH | ❌ † | ✅ |
| TOP_ERL | ✅ | ✅ | NLPromptEval | ✅ | ✅ |

**Total ran:&nbsp; baseline 15/20 &nbsp;·&nbsp; causal-mem 16/20.**

Causal memory rescues repositories the baseline couldn't complete (e.g. GCTM, CATCH,
gated_attention) and, on the 13 repositories both configurations completed, cuts end-to-end
time by **~16%** (≈1,078 s saved) by avoiding rediscovery of known fixes. Counting rescued
repositories raises the effective speedup to ~42%; we report the conservative ~16% as the
headline.

---

## Repository layout

```
build_agent/
├── main.py                     # entry point; orchestrates the full pipeline
├── multi_main.py               # batch runner (parallel jobs, disk monitoring)
├── agents/
│   ├── planner.py              # Phase 1: static analysis + plan generation
│   ├── configuration.py        # Phase 2: guarded ReAct executor + hard guards
│   ├── paper_agent.py          # paper download + experiment shortlisting (Mode 2)
│   ├── paper_corpus.py         # PDF/HTML paper corpus for indexing
│   ├── researcher.py           # bounded multi-step web/lookup research
│   ├── cuda_kernel_agent.py    # CUDA→HIP kernel migration (experimental)
│   └── triton_kernel_agent.py  # Triton ROCm-compat checks (experimental)
├── observers/                  # async observer sidecar + reviewer + skills/*.md
├── knowledge/                  # ROCm image catalog, CUDA→ROCm maps, dynamic lookups
├── images/rocm_ranker.py       # ROCm base-image ranking (Jaccard + arch + KB filters)
├── learning/                   # memory provider, causal seed, distiller, graphify, mempalace
├── storage/                    # KBStore, TrajectoryStore, fingerprint, SuccessReport, models
├── errors/                     # error classifier + seed patterns
├── rules/engine.py             # deterministic / advisory rule matching
├── kernel_migration/           # dry-run hipify pipeline scaffold (experimental)
├── tools/                      # verify_paper_result, external_lookups, web_search,
│                               #   code_edit, generate_diff, pip/apt download, runtest, …
└── utils/                      # sandbox, llm, tools_config, waiting/conflict lists, parsers
benchmark/                      # AMD-60 dataset + harness (enrich / run / report)
```

---

## Design decisions

- **Two phases (plan then execute).** A single agent wastes turns on reconnaissance and
  loses early context before configuration begins. A cheap static planner runs once and
  injects a complete plan, so the executor acts from turn 1.
- **Evidence-gated execution.** A language model can produce plausible explanations for both
  success and failure, so free-form claims are untrustworthy. Gating irreversible actions on
  runtime evidence — and delegating verdicts to a deterministic verifier — keeps outcomes
  evidence-based, auditable, and resistant to gaming.
- **`docker commit` checkpoints, not snapshots.** Commit is fast and keeps the shell alive;
  full export/import would stall every turn. The cost is that process state isn't captured,
  which is fine for single-shot install/patch/test actions.
- **Waiting list + conflict list.** Repos ship multiple, sometimes-conflicting requirements
  files. Collecting all dependencies first, surfacing conflicts explicitly, then batch-
  installing reduces pip churn and catches conflicts early.
- **Injected ROCm knowledge over open-ended search.** Recurring ROCm failures have known
  fixes; injecting curated knowledge is faster and more consistent than noisy web search,
  with live PyPI/Docker Hub lookups available when needed.
- **Single action per turn.** Each rollback then undoes exactly one operation, keeping the
  sandbox state predictable.

---

## Limitations

- The curated ROCm compatibility knowledge base can go stale as packages, images, and
  hardware evolve; live lookups help but periodic updates are needed.
- Mode 2 result reproduction targets clearly defined, automatically verifiable metrics.
  Multi-metric, subjective, or large-scale-training evaluations are out of scope, and metric
  extraction assumes recognizable reporting patterns.
- Reproduction *failures* are documented but not always given definitive causal attribution;
  e.g. the dKV-Cache discrepancy is *hypothesized* to be bandwidth-related and would need
  controlled profiling to confirm.
- The CUDA→HIP kernel path prioritizes functional correctness over runtime optimization, and
  the kernel agents / `--optimize-kernels` phase are not yet auto-wired into the main loop.
- Running third-party research code carries inherent risk; runs are containerized and
  GPU-pinned, and generated patches should be reviewed before use.

---

## License

This project is based on [Repo2Run](https://github.com/bytedance/Repo2Run),
licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
