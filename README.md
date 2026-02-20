# Repo2ROCm

**Automated CUDA-to-ROCm environment migration for arbitrary GitHub repositories.**

Repo2ROCm is an LLM-powered agent that takes any GitHub repository, analyzes its
structure and dependencies, and produces a fully configured Docker environment
that builds and runs on AMD ROCm GPUs — without manual intervention.

It extends the [Repo2Run](https://github.com/bytedance/Repo2Run) framework
(Bytedance, 2025) with a complete ROCm migration pipeline: CUDA dependency
mapping, ROCm base image selection, dependency installation,
and automated verification.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
  - [System Overview](#system-overview)
  - [Component Breakdown](#component-breakdown)
  - [Agent Loop](#agent-loop)
  - [Sandbox & Rollback](#sandbox--rollback)
  - [ROCm Knowledge Base](#rocm-knowledge-base)
- [Design Decisions](#design-decisions)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Batch Processing](#batch-processing)
- [Future Improvements and Considerations](#future-improvements-and-considerations)
- [License](#license)

---

## Problem Statement

Migrating a CUDA-based ML repository to AMD ROCm is tedious and error-prone:

1. **Dependency resolution** — CUDA-only wheels (`nvidia-cuda-runtime-cu12`,
   `flash-attn` from PyPI) must be replaced with ROCm equivalents, but the
   mapping is non-obvious and constantly changing.
2. **Base image selection** — ROCm publishes ~10 specialized Docker images
   (vllm, sglang, jax, tensorflow, pytorch, megatron, onnxruntime, etc.). Picking
   the right one requires understanding what frameworks the repo actually uses.
3. **Build system diversity** — Repos use setup.py, pyproject.toml, poetry,
   conda, Makefiles, or nothing at all. Each has different failure modes.
4. **Code patches** — Some repos hardcode `nvidia-smi`, `cudnn` paths, or
   incompatible CUDA API calls that must be patched.
5. **Verification** — You need to confirm the environment actually works, not
   just that `pip install` succeeded.

Repo2ROCm automates all five steps with a two-phase agentic approach:
static planning followed by interactive trial-and-error inside a Docker sandbox.

---

## How It Works

```
GitHub Repo (URL + SHA)
        │
        ▼
┌───────────────────┐
│   1. Clone & Scan │  Clone repo, run pipreqs, collect config files
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│   2. Plan         │  Deep static analysis → strategic build plan
│                   │  Select ROCm base image, map CUDA deps,
│                   │  detect Python version, flag hazards
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│   3. Execute      │  LLM agent loop (up to 100 turns) inside
│                   │  Docker sandbox with commit/rollback.
│                   │  Installs deps, patches code, runs tests.
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│   4. Export        │  Extract executed commands → final Dockerfile
└───────────────────┘
```

**End result**: A reproducible `Dockerfile` that anyone can `docker build` to
get a working ROCm environment for the repository.

---

## Architecture

### System Overview

```
build_agent/
├── main.py                        # Entry point, orchestrates the full pipeline
├── multi_main.py                  # Batch runner (process pool, disk monitoring)
│
├── agents/
│   ├── agent.py                   # Base class (message history)
│   ├── planner.py                 # Phase 1: static repo analysis & plan generation
│   └── configuration.py           # Phase 2: interactive agent loop
│
├── knowledge/
│   └── rocm_knowledge.py          # ROCm image catalog, CUDA→ROCm mappings,
│                                  #   banned packages, code patterns, pre-installed lists
│
├── tools/                         # Executables copied into the Docker container
│   ├── pip_download.py            # pip install wrapper with error capture
│   ├── apt_download.py            # apt-get install wrapper
│   ├── runtest.py                 # pytest collection check (non-poetry)
│   ├── poetryruntest.py           # pytest via poetry run
│   ├── runpipreqs.py              # pipreqs import scanner
│   ├── code_edit.py               # SEARCH/REPLACE diff applier
│   └── generate_diff.py           # git diff exporter
│
└── utils/
    ├── llm.py                     # LLM client (AMD API Gateway / OpenAI)
    ├── sandbox.py                 # Docker container lifecycle + pexpect shell
    ├── waiting_list.py            # Dependency queue with conflict detection
    ├── conflict_list.py           # Version constraint conflict resolution
    ├── integrate_dockerfile.py    # Converts executed commands → Dockerfile
    ├── download.py                # Batch download pipeline
    ├── rich_logger.py             # Terminal UI (Rich panels, progress)
    ├── agent_util.py              # Message formatting, diff extraction
    ├── split_cmd.py               # Shell command statement splitter
    ├── easylist.py                # Generic indexed list base class
    ├── errorformat_list.py        # Malformed requirement tracker
    └── parser/
        ├── parse_command.py       # Regex matchers for tool commands
        ├── parse_requirements.py  # requirements.txt line parser
        └── parse_dialogue.py      # Thought/Action extraction from LLM output
```

### Component Breakdown

#### Phase 1: Planner (`agents/planner.py`)

The planner performs deep static analysis of the cloned repository **before**
any Docker commands run. It reads:

- **Config files**: `requirements*.txt`, `setup.py`, `pyproject.toml`, `Pipfile`,
  `environment.yml`, `.python-version`, GitHub Actions workflows
- **README**: Up to 6,000 characters for context
- **Source code**: Scans `.py` files for import statements
- **Framework detection**: Identifies PyTorch, TensorFlow, JAX, vLLM, SGLang,
  ONNX Runtime, Megatron usage patterns

From this analysis it produces:

| Output | Purpose |
|--------|---------|
| **Strategic plan** | Step-by-step build instructions injected into the agent's system prompt |
| **Recommended ROCm image** | Scored selection from the image catalog |
| **Filtered dependencies** | Removes pre-installed packages, maps CUDA deps, drops banned NVIDIA packages |
| **Compatibility fixes** | Python 3.12 breakages, stale version pins, code hazards |

The planner uses a scoring system to select the optimal ROCm base image.
Each image in the catalog is scored against the repo's imports, dependencies,
README patterns, directory structure, and source code — weighted by frequency.

#### Phase 2: Configuration Agent (`agents/configuration.py`)

The configuration agent is an LLM-driven control loop that runs inside the
Docker sandbox. On each turn it:

1. Receives the current environment state (command output, error messages)
2. Produces a **Thought** (reasoning) and an **Action** (one bash or diff block)
3. The action is executed in the container
4. The output is fed back as the next observation

**Key parameters:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max turns | 100 | Configurable via constructor |
| Token budget | ~150,000 | Older messages dropped FIFO when exceeded |
| Actions per turn | 1 | Only the first bash/diff block is executed |
| Timeout | 2 hours | Watchdog thread kills the process |

The agent has access to a tool suite defined in `utils/tools_config.py`:

| Tool | Command | What it does |
|------|---------|--------------|
| **waitinglist add** | `waitinglist add -p numpy -v ">=1.24" -t pip` | Queue a dependency for batch install |
| **waitinglist addfile** | `waitinglist addfile /repo/requirements.txt` | Queue all deps from a requirements file |
| **download** | `download` | Batch-install everything in the waiting list |
| **conflictlist solve** | `conflictlist solve -v "==2.0"` | Resolve a version conflict |
| **runtest** | `runtest` | Verify environment via pytest collection |
| **poetryruntest** | `poetryruntest` | Same, but through `poetry run` |
| **change_python_version** | `change_python_version 3.9` | Rebuild container with different Python |
| **change_base_image** | `change_base_image rocm/vllm:latest` | Switch to a different Docker image |
| **clear_configuration** | `clear_configuration` | Reset to initial state |
| **Code edits** | ` ```diff ``` ` blocks | SEARCH/REPLACE patches applied via `code_edit.py` |

In ROCm mode, `runtest` and `poetryruntest` are disabled — the agent must
instead run the repo's main script with scaled-down parameters or mock data,
and signal success with `ROCM_ENV_VERIFIED`.

#### LLM Integration (`utils/llm.py`)

The LLM client supports two backends:

| Backend | Endpoint | Auth | Used when |
|---------|----------|------|-----------|
| **AMD API Gateway** | `https://llm-api.amd.com/claude3/{model}/chat/completions` | `Ocp-Apim-Subscription-Key` header | Model name contains "claude" |
| **OpenAI** | Standard OpenAI API | `OPENAI_API_KEY` env var | All other models |


Both backends use retry with exponential backoff (5 attempts, 5–60s jitter).

### Agent Loop

```
                    ┌─────────────────────────┐
                    │     System Prompt        │
                    │  (tools, rules, plan,    │
                    │   ROCm knowledge)        │
                    └────────────┬────────────┘
                                 │
         ┌───────────────────────▼───────────────────────┐
         │                 LLM Call                       │
    ┌────│  messages = [system] + conversation history    │
    │    └───────────────────────┬───────────────────────┘
    │                            │
    │                   ┌────────▼────────┐
    │                   │  Parse Response │
    │                   │  Thought + Action│
    │                   └────────┬────────┘
    │                            │
    │              ┌─────────────▼─────────────┐
    │              │   Is it a tool command?    │
    │              │  (download, waitinglist,   │
    │              │   conflictlist, runtest)   │
    │              └──┬────────────────────┬───┘
    │                 │ YES                │ NO
    │         ┌───────▼───────┐    ┌──────▼──────┐
    │         │ Execute tool  │    │ Execute in  │
    │         │  in-process   │    │  container  │
    │         └───────┬───────┘    │  via shell  │
    │                 │            └──────┬──────┘
    │                 │                   │
    │                 │    ┌──────────────▼──────────────┐
    │                 │    │  Non-zero exit code?        │
    │                 │    │  AND not a safe/read-only   │
    │                 │    │  command?                   │
    │                 │    └──┬───────────────────┬─────┘
    │                 │       │ YES               │ NO
    │                 │  ┌────▼────┐              │
    │                 │  │ ROLLBACK│              │
    │                 │  │ to last │              │
    │                 │  │ commit  │              │
    │                 │  └────┬────┘              │
    │                 │       │                   │
    │         ┌───────▼───────▼───────────────────▼───┐
    │         │        Append observation to          │
    │         │        conversation history            │
    │         └───────────────────┬───────────────────┘
    │                             │
    │              ┌──────────────▼──────────────┐
    │         NO   │  Success signal detected?   │ YES ──► Done
    │◄─────────────│  (runtest pass / VERIFIED)  │
    │              └─────────────────────────────┘
    │                             │
    └──── next turn ◄─────── turn < 100?
```

### Sandbox & Rollback

The sandbox (`utils/sandbox.py`) manages the Docker container lifecycle through
`pexpect`, providing an interactive shell session that persists across turns.

**Commit/rollback mechanism:**

Before executing any non-safe command (i.e., not `ls`, `cat`, `grep`, etc.),
the sandbox commits the container as a Docker image (`docker commit`). If the
command fails with a non-zero exit code, the sandbox:

1. Stops and removes the current container
2. Starts a new container from the last committed image
3. Opens a fresh shell session

This gives the agent a transactional guarantee: failed `pip install` or
broken `apt-get` calls are automatically reverted, preventing cascading
failures from corrupting the environment.

**Exceptions to rollback** (commands that fail but don't revert):

- Read-only commands (`ls`, `cat`, `grep`, `find`, etc.)
- Heredoc file writes (`cat > /tmp/script.sh << 'EOF'`)
- Non-destructive Python commands (`python script.py`, `python -c "..."`)
- Timeout kills of non-destructive operations

### ROCm Knowledge Base

`knowledge/rocm_knowledge.py` is a structured data module (~1,600 lines) that
encodes AMD-specific migration knowledge. It is injected into the agent's system
prompt so the LLM can make informed decisions without external lookups.

**Image catalog** — 9 specialized ROCm Docker images with descriptions,
available tags, and selection criteria:

| Workload | Image | When to use |
|----------|-------|-------------|
| SGLang serving | `rocm/sgl-dev` | Repo is an SGLang project |
| vLLM development | `rocm/vllm-dev` | Repo is a vLLM fork |
| vLLM serving | `rocm/vllm` | Repo uses vLLM as a library |
| JAX | `rocm/jax` | Repo uses JAX/Flax/Optax |
| TensorFlow | `rocm/tensorflow` | Repo uses TensorFlow |
| ONNX Runtime | `rocm/onnxruntime` | Repo does ONNX inference |
| Distributed training | `rocm/pytorch-training` | DeepSpeed/FSDP/Megatron |
| Megatron-LM | `rocm/megatron-lm` | Megatron-based training |
| General PyTorch | `rocm/pytorch` | Default fallback |

**Pre-installed package lists** — Per-image lists of packages that are already
available in the container. The agent is instructed to skip these during
installation to avoid version conflicts with the ROCm runtime.

---

## Design Decisions

### Why two phases (plan then execute)?

Early versions used a single agent that both explored and configured.
This had two problems:

1. **Wasted turns** — The agent would spend 10–15 turns reading README files,
   listing directories, and understanding the repo structure, leaving fewer
   turns for actual configuration.
2. **Context loss** — By the time the agent started installing dependencies,
   the early reconnaissance had been pushed out of the token window.

The two-phase design solves both: the planner does deep static analysis once
(cheap, fast, no Docker), then injects a complete plan into the configuration
agent's system prompt. The agent starts executing immediately on turn 1.

### Why commit/rollback instead of container snapshots?

Docker `commit` is fast (typically <2s) and doesn't require stopping the
container. Full container snapshots (export/import) would take 30–60s and
kill the running shell. Since the agent averages ~50 turns per repo, the
cumulative overhead of snapshots would be prohibitive.

The tradeoff is that `commit` captures filesystem state but not running
process state. This is acceptable because the agent's commands are
single-shot (install, patch, test) rather than long-running daemons.

### Why a waiting list instead of direct pip install?

Repositories often have multiple requirements files (`requirements.txt`,
`requirements-dev.txt`, `requirements-test.txt`) with overlapping and
sometimes conflicting version constraints. Installing them one-by-one leads
to repeated dependency resolution and version churn.

The waiting list + conflict list pattern collects all dependencies first,
surfaces conflicts for explicit resolution, then installs everything in a
single batch. This reduces the number of pip invocations and catches
conflicts before they cause hard-to-debug failures.

### Why inject ROCm knowledge into the prompt?

An alternative would be giving the agent web search or documentation lookup
tools. We chose prompt injection because:

1. **Latency** — Web searches add 3–10s per turn; the knowledge base adds
   ~2,000 tokens to the prompt (cached after the first call).
2. **Reliability** — ROCm documentation is scattered across GitHub READMEs,
   Docker Hub descriptions, and forum posts. Search results are noisy.
3. **Specificity** — The knowledge base contains exact install commands,
   env vars, and gotchas.

The downside is that the knowledge base can go stale. It should be updated
when new ROCm images are released or package mappings change.

### Why single-action-per-turn?

Allowing multiple bash blocks per turn risks unrecoverable state: if the
second command fails after the first succeeded, the rollback reverts both.
Single-action-per-turn means each rollback undoes exactly one operation,
keeping the environment predictable.

---

## Quick Start

### Prerequisites

- Docker (with GPU passthrough configured for ROCm)
- Python 3.8+
- An AMD LLM API Gateway key (for Claude) or an OpenAI API key

### Installation

```bash
git clone <repo-url> Repo2Run
cd Repo2Run
pip install docker pexpect requests tenacity rich pipreqs
```

### Run on a single repository

```bash
# Get the latest commit SHA
SHA=$(git ls-remote https://github.com/user/repo HEAD | cut -f1)

# Run with ROCm mode
python build_agent/main.py \
  --full_name "user/repo" \
  --sha "$SHA" \
  --root_path . \
  --llm "claude-sonnet-4" \
  --rocm \
  --api-key "$AMD_LLM_API_KEY"
```

### Output

After a successful run, the output directory contains:

```
output/user/repo/
├── Dockerfile          # Reproducible build — docker build -t myenv .
├── plan.txt            # Strategic plan from the planner
├── track.json          # Full agent conversation history
├── inner_commands.json # All commands executed inside the container
├── outer_commands.json # LLM call timings and metadata
├── sha.txt             # Git SHA used
└── agent_debug_log.txt # Rich-formatted debug log
```

---

## CLI Reference

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--full_name` | string | *required* | GitHub repo (`owner/name`) |
| `--sha` | string | *required* | Git commit SHA to check out |
| `--root_path` | string | *required* | Working directory for clones and output |
| `--llm` | string | `gpt-4o-2024-05-13` | LLM model name (`claude-sonnet-4` for AMD gateway) |
| `--rocm` | flag | `false` | Enable ROCm migration mode |
| `--rocm-base-image` | string | auto-selected | Override the ROCm Docker base image |
| `--api-key` | string | env var | AMD LLM API Gateway key |
| `--verbose` | flag | `false` | Print full LLM prompts and responses |

---

## Batch Processing

Process multiple repositories in parallel:

```bash
# Create a script file with one command per line
cat > batch.sh << 'EOF'
python -u build_agent/main.py --full_name "org/repo1" --sha "abc123" --root_path . --llm "claude-sonnet-4" --rocm --api-key "$KEY"
python -u build_agent/main.py --full_name "org/repo2" --sha "def456" --root_path . --llm "claude-sonnet-4" --rocm --api-key "$KEY"
EOF

# Run with 3 concurrent processes
python build_agent/multi_main.py batch.sh
```

`multi_main.py` monitors disk usage on `/dev/vdb` and halts if it exceeds 90%.
It also cleans up containers between runs.

---

## Future Improvements and Considerations

- **Token Usage Optimization** — Efficient token utilization is essential to save costs when using LLM APIs. Incorporating strategies such as memory layers to cache relevant context or compress message history can be a valuable approach to minimize redundant tokens.
- **Cross-Repository Learning** — Capturing lessons learned and patterns from previous repository migrations can make the agent progressively smarter and more effective over time. On-going experience accumulation across runs should inform future plans and actions.
- **Integration with Additional Tools** — Including tools like AMD’s Hippify (hipify-perl, hipify-clang) would expand the automation and improve fidelity in CUDA-to-ROCm code conversion, further streamlining migration workflows.

---

## License

This project is based on [Repo2Run](https://github.com/bytedance/Repo2Run),
licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
