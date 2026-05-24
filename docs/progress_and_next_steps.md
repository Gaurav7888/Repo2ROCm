# Repo2ROCm — Progress and Next Steps

## Goal

Take the two planned components called out in `docs/next_research_plan.md`
on the `r2r++` branch — **(1) Causal Migration Memory** and **(2) a
correctness-only Kernel Converter Agent** — implement each in its own
git worktree, then run the AMD-60 benchmark in **Mode 1**
(`--mode env`, headline marker `ROCM_ENV_VERIFIED`) on a shared subset
and check whether either component drives the Mode 1 score up versus
the `r2r++` baseline.

Mode 1 scoring (see `benchmark/harness/scoring/rubric.py`):

- `score = 3` if `ROCM_ENV_VERIFIED` is present.
- `score = 2` if a `Generate success` Dockerfile is produced but env is
  not verified.
- `score = 1` if the loop ran to completion without success.
- `score = 0` on hard failure.
- `score = 4 / 5` are reserved for paper reproduction and are
  unreachable in Mode 1 by construction.

So the headline metric for this comparison is **`rocm_env_verified_rate`**
(equivalently, the rate of `score >= 3`).

## Worktrees and branches

| Worktree                              | Branch         | Parent     | Head     | Implements                                                          |
|---------------------------------------|----------------|------------|----------|---------------------------------------------------------------------|
| `/home/upandey/rocm`                  | `r2r++`        | —          | `ce383bb` | baseline + harness `--gpu-index` compat fix + this doc + helpers     |
| `/home/upandey/rocm-causal-mem`       | `causal-mem`   | `01bb613`  | `0175b95` | Track 1: causal migration memory (`55c37e2` + harness fix)           |
| `/home/upandey/rocm-kernel-agent`     | `kernel-agent` | `01bb613`  | `14bef24` | Track 2: correctness-only kernel converter agent (`b8733b6` + fix)   |

All three branches are pushed to `origin`
(`https://github.com/Gaurav7888/Repo2ROCm.git`):

- `r2r++` → `https://github.com/Gaurav7888/Repo2ROCm/tree/r2r++`
- `causal-mem` → `https://github.com/Gaurav7888/Repo2ROCm/tree/causal-mem`
- `kernel-agent` → `https://github.com/Gaurav7888/Repo2ROCm/tree/kernel-agent`

The harness compat commit is the new tip on each branch and is
content-identical across all three worktrees (verified by
`git diff` content equality on `build_agent/main.py` and
`build_agent/utils/sandbox.py`).

## Track 1 — Causal Migration Memory

Branch: `causal-mem`. Implementation commit: `55c37e2` ("Add causal
migration memory (state-action-outcome transitions)").

**Files changed** (`git show --stat 55c37e2`):

```
build_agent/agents/configuration.py      |  88 +
build_agent/learning/causal_seed.py      | 257 +
build_agent/learning/distiller.py        | 326 +
build_agent/learning/memory_provider.py  | 197 +
build_agent/main.py                      |  28 +/-
build_agent/storage/kb_store.py          | 181 +
build_agent/storage/models.py            | 163 +
tests/test_causal_memory.py              | 517 +
8 files changed, 1752 insertions(+), 5 deletions(-)
```

**Schema.** A new SQLite table `causal_transitions` is created by
`KBStore` with columns:

`id, transition_class, repo_fingerprint, image, gpu_arch, error_class,
error_signature, degradation_policy, degradation, action_type,
action_command, source, source_attempt, evidence_count, confidence,
created_at, last_seen, state_signature, data_json`.

Indexed on `error_class`, `image`, `gpu_arch`, `repo_fingerprint`,
`degradation_policy`, `state_signature`, and `transition_class`.

**Seeding.** `learning/causal_seed.py:seed_causal_transitions()` is
wired into `main.py` next to the existing `errors/seed_patterns`
seeder. It runs only when the table is empty (idempotent), and seeds
the five transition classes named in the plan:

- `cuda_only_wheel_to_rocm_source_build`
- `wrong_image_to_ranked_image_switch`
- `missing_gpu_runtime_to_rocm_base_image`
- `custom_cuda_compile_error_to_hipify_fix`
- `paper_metric_mismatch_to_not_reproduced`

**Retrieval.** `BuildMemoryProvider.provide_causal_memory()` formats
matching transitions via `format_causal_transition()` and emits
`[counterfactual: ...]` advisories via
`format_causal_counterfactuals()`. It is invoked from:

- the **BEGIN** phase prompt as the section `CAUSAL MIGRATION
  TRANSITIONS`, and
- **every turn** of the **IN** phase via
  `Configuration._provide_causal_memory_per_turn()` (records image,
  gpu_arch, error_class, return_code).

Per the plan, causal memory surfaces in **every mode including
`--mode env`** (Mode 1).

**Distillation.**
`TrajectoryDistiller.extract_causal_transitions(trajectory_records,
attempt, success_report)` is conservative by design: it only emits a
transition when

1. a failed turn is followed within ≤ 3 turns by a successful turn that
   resolves the same error class against the same package/file root,
   **and**
2. the run produced `ROCM_ENV_VERIFIED`.

Degradation is read from `success_report.degradation_flags` (mapped to
`D0..D3`). Persisted automatically as part of `distill_and_apply`.

**Tests.** `tests/test_causal_memory.py` covers serialisation
roundtrip, KB insert + state-similarity query, distiller extraction
from a synthetic `failure → success → ROCM_ENV_VERIFIED` trajectory
(and refusal without the marker), `[CAUSAL]` line formatting, BEGIN/IN
phase integration, and seed-on-empty idempotence. Suite status:
`tests/test_causal_memory.py` **11/11 passing**; full suite
**29/29 passing** (excluding the pre-existing broken
`tests/test_observer_bus.py`).

## Track 2 — Kernel Converter Agent

Branch: `kernel-agent`. Implementation commit: `b8733b6` ("Add
correctness-only kernel converter agent (hipify + granular fix +
verify)").

**Files changed** (`git show --stat b8733b6`):

```
build_agent/agents/configuration.py              | 200 +/-
build_agent/agents/kernel_converter_agent.py     | 655 +
build_agent/kernel_migration/__init__.py         |   8 +
build_agent/kernel_migration/executor_adapter.py | 153 +
build_agent/main.py                              |  13 +
build_agent/storage/models.py                    |  88 +
build_agent/storage/success_report.py            |  53 +
tests/test_kernel_converter_agent.py             | 379 +
8 files changed, 1548 insertions(+), 1 deletion(-)
```

**Data model.** New `KernelMigrationReport` dataclass in
`storage/models.py` with:

- `status` ∈ `no_kernels | hipify_planned | hipify_applied |
  compile_passed | manual_fixes_required | unsupported`
- `degradation` ∈ `D0 .. D5`
- plus per-file inventory, hipify command log, and structured
  manual-fix entries.

**Executor adapter.** `kernel_migration/executor_adapter.py` provides
two backends behind the same interface:

- `SandboxExecutor` — live, pexpect-backed, used in real runs.
- `DryRunExecutor` — used only in tests; deterministic replay.

**Agent.** `agents/kernel_converter_agent.py:KernelConverterAgent`
drives the phases:

1. **Inventory** — list `.cu` / `.cuh` / kernels in the repo.
2. **Examine** — `hipify-clang` first; fall back to `hipify-perl`.
3. **Apply** — non-in-place preferred (writes `*.hip.cpp`-style
   outputs).
4. **Granular fix** — `SEARCH/REPLACE` blocks for host-visible files;
   structured manual-fix entry with `requires_subagent=True` when the
   change exceeds safe-edit scope.
5. **Verify** — `hipcc` compile-check.
6. **Report** — writes `kernel_migration_report.json` and threads the
   status / degradation back into the success report.

**Trigger conditions** (`agents/configuration.py`):

- Enabled by default.
- Runs at most **once per attempt**.
- Active only when `--mode env` or `--mode full` (Mode 2 reproduce
  skips it).
- Fires when **any** of: the repo fingerprint flags custom CUDA, the
  repo contains `.cu`/`.cuh` files, or a turn observation matches
  `looks_like_cuda_compile_error`.

**CLI.** New flag `--no-kernel-converter` disables the agent. Passes
through `main.py`.

**Success report.** `success_report.build_success_report()` now
accepts `kernel_migration=` and promotes `kernel_migration_status` and
`kernel_migration_degradation` so the rubric and reporting layer can
read them without having to re-derive from the trajectory.

**Tests.** `tests/test_kernel_converter_agent.py` — 16/16 passing
(includes 5 sub-tests). Full suite **34/34 passing** (excluding the
pre-existing broken `tests/test_observer_bus.py`).

## Benchmark setup (ready to run)

- **Dataset.** `benchmark/harness/cache/tasks.json` — 60 tasks, 54
  runnable.
- **Mode 1 comparison subset.**
  `benchmark/harness/cache/tasks_kernel_subset.json` — 10 papers
  selected as:
  - 2 with explicit CUDA hints
    (`30-llm-reproducibility` → `nanomaoli/llm_reproducibility`,
    `183-...` → `1202kbs/GCTM`), and
  - 8 controls.
  The classification artifact is
  `benchmark/harness/cache/cuda_classification_v2.json`. The selection
  scripts (`find_cuda_repos.py`, `find_cuda_repos2.py`,
  `build_subset.py`) and the subset files
  (`tasks_kernel_subset.json`, `kernel_subset_paper_ids.txt`) are
  checked into `r2r++` under `benchmark/harness/cache/`.

  **Sync to the other worktrees before running.** Both `causal-mem`
  and `kernel-agent` need the subset files. Either merge `r2r++`
  forward:

  ```bash
  cd /home/upandey/rocm-causal-mem && git merge r2r++ --no-edit
  cd /home/upandey/rocm-kernel-agent && git merge r2r++ --no-edit
  ```

  …or `cp` the two `*kernel_subset*` files into each worktree's
  `benchmark/harness/cache/`.

- **Launch scripts.** Live in `benchmark/launch_logs/` on `r2r++`:
  `launch_baseline.sh`, `launch_causal.sh`, `launch_kernel.sh`. Each
  runs `python -m harness.main run --mode env --tasks-json
  harness/cache/tasks_kernel_subset.json --runs-dir runs_mode1 --db
  runs_mode1/progress.sqlite --approaches repo2rocm --gpus <split>
  --timeout 5400` against its own worktree.

  **The AMD LLM API key has been redacted** from each script and
  replaced with the guard:

  ```bash
  : "${AMD_LLM_API_KEY:?AMD_LLM_API_KEY must be exported before launch}"
  ```

  You must `export AMD_LLM_API_KEY=<key>` in the shell before
  launching.

- **GPU split.** baseline `0,1,2` · causal-mem `3,4,5` ·
  kernel-agent `6,7`. Per-task timeout 90 min. Each worktree writes
  to its own `benchmark/runs_mode1/` and `progress.sqlite`.

## Current status

- Track 1 (`causal-mem`) and Track 2 (`kernel-agent`) are
  implemented, committed, and pushed to origin.
- The harness `--gpu-index` compat fix is committed and pushed on all
  three branches.
- Helper scripts, subset selection, and launch scripts (API key
  redacted) are checked in on `r2r++`.
- Mode 1 benchmark has **not yet been run end-to-end** — see Blockers.

Branch heads on origin:

- `r2r++` @ `ce383bb` — `Fix harness compat: accept --gpu-index and
  forward to container env`
- `causal-mem` @ `0175b95` — same harness fix on top of `55c37e2`
- `kernel-agent` @ `14bef24` — same harness fix on top of `b8733b6`

## Blockers we hit

1. **Pre-existing harness bug** —
   `benchmark/harness/runners/runner_repo2rocm.py` passes
   `--gpu-index` to `build_agent/main.py`, but `main.py` did not
   declare the flag. Every harness invocation aborted with
   `unrecognized arguments: --gpu-index ...`. Fixed in the harness
   compat commit on all three branches.
2. **Missing Python deps on host.** `docker`, `bs4`, `pipreqs`,
   `pymupdf`, `mempalace`, `chromadb`, `ddgs`, `html2text`, `openpyxl`,
   `pandas`, `openai` were not installed on this host. Resolved by
   running `pip3 install -r requirements.txt` followed by
   `pip3 install pipreqs beautifulsoup4`.
3. **`build_agent/docker/` shadowed the `docker` PyPI package.** That
   directory contains only a `Dockerfile` and no `__init__.py`, so it
   acts as a PEP-420 namespace package when `build_agent/` is on
   `sys.path`. The real PyPI package wins once it is actually
   installed, so installing `docker` masks the symptom; the long-term
   fix is to rename `build_agent/docker/` to
   `build_agent/dockerfiles/`.
4. **Docker Hub rate limit (429 `toomanyrequests`).** The current
   host IP has exhausted the unauthenticated pull quota, so no ROCm
   image could be pulled. Either `docker login` (or a PAT) is needed
   to switch to account-based quota, or the rate-limit window must
   reset (~6 h).
5. **Shell backend instability.** Toward the end of the session the
   IDE shell tool repeatedly returned `Execution backend unavailable`,
   which made hands-on benchmarking impossible. All in-flight changes
   were saved to git and pushed to origin so work can resume from a
   fresh environment.

## Next steps to resume

1. Install host deps (from any worktree):

   ```bash
   pip3 install -r requirements.txt
   pip3 install pipreqs beautifulsoup4
   ```

2. Authenticate with Docker Hub to lift the pull rate limit:

   ```bash
   docker login
   ```

   (or a PAT). This raises the rate from 100 / 6 h IP-based to
   ≥ 200 / 6 h account-based.

3. Pre-pull the dominant ROCm image so all 8 workers reuse the cache:

   ```bash
   docker pull rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.7.1
   ```

4. Sync the subset files into the other two worktrees:

   ```bash
   cd /home/upandey/rocm-causal-mem && git merge r2r++ --no-edit
   cd /home/upandey/rocm-kernel-agent && git merge r2r++ --no-edit
   ```

   (Or `cp` the subset files manually if a merge is undesirable.)

5. Export the AMD LLM API key in the launching shell:

   ```bash
   export AMD_LLM_API_KEY=<key>
   ```

6. Launch the three runs concurrently:

   ```bash
   /home/upandey/rocm/benchmark/launch_logs/launch_baseline.sh > /tmp/baseline.log 2>&1 &
   /home/upandey/rocm/benchmark/launch_logs/launch_causal.sh   > /tmp/causal.log   2>&1 &
   /home/upandey/rocm/benchmark/launch_logs/launch_kernel.sh   > /tmp/kernel.log   2>&1 &
   ```

   Each script invokes
   `python -m harness.main run --mode env --tasks-json
   harness/cache/tasks_kernel_subset.json --runs-dir runs_mode1
   --db runs_mode1/progress.sqlite --approaches repo2rocm
   --gpus <split> --timeout 5400`.

7. Aggregate results once each branch finishes:

   ```bash
   cd <worktree>/benchmark
   python3 -m harness.main report \
       --runs-dir runs_mode1 \
       --reports-dir reports_mode1 \
       --tasks-json harness/cache/tasks_kernel_subset.json \
       --approaches repo2rocm
   ```

   Inspect `reports_mode1/summary.md` (per-approach scorecard,
   `ROCM_VERIFIED` rate) and `reports_mode1/results.csv`.

8. Compare `rocm_env_verified_rate` and `score >= 3` rate between
   `baseline`, `causal-mem`, and `kernel-agent`. With only 2 of 10
   papers exercising explicit CUDA, `kernel-agent`'s measurable signal
   is narrow on this subset; broaden the subset
   (e.g. `--max-paper-limit 30` against the full `tasks.json`) if more
   signal is required.

## Risks and known limitations

- **Narrow kernel-agent signal on this subset.** Only ≤ 2 of the 10
  selected papers exercise the hipify path
  (see `benchmark/harness/cache/cuda_classification_v2.json`). If the
  Mode 1 comparison is inconclusive, the next move is to widen the
  subset, not to amend the agent.
- **Causal memory only writes new transitions on
  `ROCM_ENV_VERIFIED` runs.** If the baseline never verifies, the
  seeded transitions are the only causal knowledge surfaced. The
  seeded set covers the five planned classes, so the BEGIN/IN-phase
  prompts still receive structured `[CAUSAL]` advisories on day one.
- **`build_agent/docker/` is a latent footgun.** On any host where
  the `docker` PyPI package is missing, that directory will silently
  hijack the `docker` namespace. The robust fix is to rename it to
  `build_agent/dockerfiles/`; this has been intentionally deferred to
  avoid touching unrelated paths in this work.
- **Pre-existing broken test.** `tests/test_observer_bus.py` is
  broken on the parent commit `01bb613`
  (`ImportError: HazardLedgerBuilder`). This pre-dates both Track 1
  and Track 2 and is unrelated to either implementation; the full
  test counts above explicitly exclude it.
