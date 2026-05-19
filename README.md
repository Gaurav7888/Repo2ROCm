# Repo2ROCm v2

**Production-grade multi-agent CUDA→ROCm migration system.**

A clean-room redesign of [Repo2ROCm](../Repo2ROCm) that incorporates the architectural
patterns from Anthropic's Claude Code — async-generator agent loop, self-describing tools,
streaming speculative execution, layered context compression, permission modes, frozen hook
snapshots, file-based memory with LLM recall, MCP, and end-to-end observability.

## Quick start

Requires **Python ≥ 3.10**. From a fresh clone:

```bash
git clone <repo-url> Repo2ROCm
cd Repo2ROCm

# (Recommended) isolate from system site-packages
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Runtime install
pip install .

# …or editable + dev extras for hacking on the code
pip install -e ".[dev]"
```

Then run:

```bash
repo2rocm doctor                                     # bootstrap diagnostics
repo2rocm migrate owner/repo --sha <sha> --mode env
repo2rocm reproduce owner/repo --sha <sha>           # env + paper reproduction
repo2rocm mcp serve docker-hub                       # spawn an MCP stdio server
```

### Troubleshooting install

If you cannot use a venv and `pip install` complains about a build-backend
dependency (e.g. an old `pathspec` / `hatchling` / `pip` on the host), force
setuptools' isolated build path and skip PEP 517 build isolation:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install --no-build-isolation -e ".[dev]"
```

On Ubuntu 20.04 / 22.04 where `pip` ships at 22.x, upgrading pip first
(`python -m pip install --upgrade pip`) is almost always sufficient.

## Architecture in 60 seconds

```
                     ┌───────────────┐
   user CLI ─────►   │  bootstrap()  │ (idempotent, 5 phases, checkpointed)
                     └──────┬────────┘
                            ▼
                     ┌───────────────┐
                     │  Coordinator  │ (3 tools: Agent / SendMessage / TaskStop)
                     └──────┬────────┘
        spawns       ┌──────┴───────┬──────────────┬────────────┐
                     ▼              ▼              ▼            ▼
                ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
                │ Explore │  │ Planner  │  │ Migrator │  │ Verifier │
                └─────────┘  └──────────┘  └──────────┘  └──────────┘
                   │   │         │  │           │  │           │  │
                   ▼   ▼         ▼  ▼           ▼  ▼           ▼  ▼
              Read,Grep,...   PyPI,Docker     all-tools   EnvVerify
                                Hub                       (adversarial)
```

Every agent runs the **same** async-generator agent loop (`core/query.py`). Agent type
is encoded in `AgentDefinition` (data), not control flow.

## Observability

* **OpenTelemetry spans** on every turn, tool call, sub-agent lifecycle, sandbox op.
  Set `OTEL_EXPORTER_OTLP_ENDPOINT=https://collector:4318` to ship.
* **Prometheus metrics** on `http://127.0.0.1:9464/metrics` (configurable):
  - `repo2rocm_tool_calls_total{tool,outcome}`
  - `repo2rocm_tool_latency_seconds{tool}`
  - `repo2rocm_turn_latency_seconds{agent_type}`
  - `repo2rocm_llm_tokens{model,kind}` (input/output/cache_read/cache_creation)
  - `repo2rocm_prompt_cache_hit_ratio{model}`
  - `repo2rocm_subagents_active{agent_type}`
  - `repo2rocm_permission_decisions_total{tool,mode,decision}`
  - `repo2rocm_hook_invocations_total{event,outcome}`
  - `repo2rocm_sandbox_ops_total{op,outcome}`
  - `repo2rocm_context_compactions_total{layer,outcome}`
  - `repo2rocm_migration_outcomes_total{mode,outcome}`
* **Structured logs** via `structlog`, JSON by default, Rich console when on a TTY.
* **JSONL transcripts** per (session, agent) in `output/<session>/<agent_id>.jsonl`,
  enabling auto-resume and replay.
* **Startup checkpoints** — 50+ named markers; dumped via `repo2rocm doctor`.

## Tests

```bash
pip install -e ".[dev]"
pytest -q tests/unit                    # fast, no docker/network
pytest -q tests/integration             # docker + httpx
```

## Layout

See `docs/architecture.md` for the full breakdown.
