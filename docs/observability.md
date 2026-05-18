# Observability

Repo2ROCm v2 instruments every boundary: turn, tool call, sub-agent, sandbox op,
context compaction, hook, permission decision, LLM call.

## OpenTelemetry traces

Configure via env:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://collector:4318
```

Spans emitted (non-exhaustive):

| Span | Attributes |
|---|---|
| `agent.turn`             | `turn`, `agent_type`, `agent_id` |
| `agent.lifecycle`        | `agent_id`, `agent_type`, `is_async` |
| `tool.invoke`            | `tool`, `agent_id` |
| `executor.run_tool`      | `tool`, `is_safe`, `tool_use_id` |
| `llm.stream`             | `model`, `provider` |
| `sandbox.exec`           | `command_preview`, `cwd`, `timeout_s` |
| `sandbox.commit`         | `label` |
| `sandbox.rollback`       | `target` |
| `context.collapse`       | — |
| `context.auto_compact`   | — |
| `hooks.execute`          | `event`, `count` |
| `memory.recall`          | `candidates` |
| `checkpoint`             | `name`, `delta_ms`, `cumulative_ms` |

Each span auto-records exceptions (`Status.ERROR` + `record_exception`).

## Prometheus metrics

`bootstrap()` starts a `/metrics` endpoint on port 9464 by default
(configurable via `REPO2ROCM_METRICS_PORT`).

| Metric | Labels | Notes |
|---|---|---|
| `repo2rocm_tool_calls_total`        | `tool, outcome`           | counter |
| `repo2rocm_tool_latency_seconds`    | `tool`                    | histogram |
| `repo2rocm_tool_result_bytes`       | `tool`                    | histogram |
| `repo2rocm_turn_latency_seconds`    | `agent_type`              | histogram |
| `repo2rocm_llm_tokens`              | `model, kind`             | histogram; kind ∈ {input, output, cache_read, cache_creation} |
| `repo2rocm_llm_latency_seconds`     | `model`                   | histogram |
| `repo2rocm_llm_errors_total`        | `model, error_class`      | counter |
| `repo2rocm_subagents_active`        | `agent_type`              | gauge |
| `repo2rocm_subagent_completions_total` | `agent_type, reason`   | counter |
| `repo2rocm_permission_decisions_total` | `tool, mode, decision` | counter |
| `repo2rocm_hook_invocations_total`  | `event, outcome`          | counter |
| `repo2rocm_sandbox_ops_total`       | `op, outcome`             | counter |
| `repo2rocm_migration_outcomes_total`| `mode, outcome`           | counter |
| `repo2rocm_context_compactions_total`| `layer, outcome`         | counter |
| `repo2rocm_prompt_cache_hit_ratio`  | `model`                   | gauge |

## Structured logs

`structlog` is configured during `bootstrap()`. The output renderer is auto-selected:

* TTY → Rich console with colors.
* Non-TTY (CI, batch) → JSON lines.

Every log line is bound with `trace_id` when an OTel span is active.

## JSONL transcripts

Each agent gets a JSONL file at
`output/<session_id>/<agent_id>.jsonl`. The session transcript lives at
`output/<session_id>/session.jsonl`. Each line carries:

```json
{
  "uuid":     "f3a1...",
  "parent_uuid": "e2b0...",
  "ts":       1747569420.123,
  "kind":     "tool_result",
  "tool":     "Read",
  "outcome":  "ok",
  "elapsed_s": 0.0123,
  "bytes":    1024,
  "input":    {"file_path": "src/main.py"}
}
```

These enable:

* **auto-resume** — `SendMessage` to a dead agent reconstructs its history from disk
* **replay** — re-run a sub-agent deterministically against a mock client
* **audit** — every tool call, permission decision, and terminal reason is on disk

## Startup checkpoints

`bootstrap()` emits 7 named markers; users can add their own with
`from repo2rocm.observability import checkpoint`. `repo2rocm doctor` prints them as a
table:

```
                 Bootstrap Checkpoints
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ name                        ┃   Δms ┃ cumulative_ms ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━┩
│ bootstrap.start             │ 36.53 │         36.53 │
│ bootstrap.config_load       │  0.61 │         37.14 │
│ bootstrap.observability     │  2.28 │         39.42 │
│ bootstrap.hooks_snapshot    │  0.11 │         39.53 │
│ bootstrap.skills_discovered │  0.61 │         40.14 │
│ bootstrap.tools_registered  │ 20.98 │         61.12 │
│ bootstrap.done              │  0.07 │         61.19 │
└─────────────────────────────┴───────┴───────────────┘
```
