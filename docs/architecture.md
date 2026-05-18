# Architecture

Repo2ROCm v2 is built on six core abstractions, mirroring the design that Claude Code
arrived at (see `@claude-code-from-source/book/ch01-architecture.md`):

1. **Query loop** (`core/query.py`) — the only place a model is called and tools execute.
2. **Tool system** (`tools/`) — self-describing `BaseTool` subclasses with Pydantic schemas.
3. **Tasks / sub-agents** (`agents/`) — recursive `run_agent` lifecycle; each agent type is *data*, not control flow.
4. **State** (`core/state.py`) — immutable `LoopState` + sticky `LatchedSet`.
5. **Memory** (`core/memory/`) — file-based; LLM-powered recall via cheap side-query.
6. **Hooks** (`core/hooks/`) — lifecycle interceptors frozen at startup.

## The golden path

```
keystroke → cli.migrate → bootstrap()
   → Coordinator agent  (PermissionMode.PLAN; tools={Agent, SendMessage, TaskStop})
      ├─→ Explore  (Haiku, read-only, omit_user_context=True)
      ├─→ Planner  (Sonnet, +Fetch/DockerHubTags/PyPIVersions, builtin skills preloaded)
      ├─→ Migrator (Sonnet, full tools, write-enabled)
      └─→ Verifier (Sonnet, background, adversarial, read-only)
            → Sandbox.exec(...) → docker exec → commit DAG
            → EnvVerify (typed verdict)
   → Dockerfile synthesis (commit-log trunk → Dockerfile)
   → Learning pipeline (KB facts + trajectory store)
```

## Module map

| Path | Responsibility |
|---|---|
| `core/messages.py`        | Pydantic message types (TextBlock, ToolUseBlock, …) |
| `core/terminal.py`        | 10 Terminal reasons + 7 Continue reasons |
| `core/state.py`           | `LoopState` (frozen) + `LatchedSet` |
| `core/api.py`             | `ModelClient` protocol; Anthropic/AMD/Mock implementations; streaming |
| `core/token_count.py`     | API-anchored counting + conservative estimation |
| `core/permissions.py`     | 6 modes + `resolve_permission()` chain |
| `core/hooks/`             | Frozen snapshot + runner + builtin gates |
| `core/memory/`            | File-based memory + LLM recall + staleness |
| `core/context_pipeline.py`| 4-layer compaction (tool_result_budget → snip → microcompact → collapse → autocompact) |
| `core/query.py`           | `QueryRun` async-generator loop |
| `tools/base.py`           | `BaseTool` + `ToolResult` + registry |
| `tools/executor/`         | Partition + StreamingToolExecutor |
| `tools/{repo,docker,packaging,external,verify,memory}/` | The 28 builtin tools |
| `tools/agent_tool.py`     | Agent / SendMessage / TaskStop |
| `agents/lifecycle.py`     | The 15-step `run_agent` |
| `agents/definition.py`    | `AgentDefinition` dataclass |
| `agents/builtin/`         | Coordinator, Explore, Planner, Migrator, Verifier, PaperReproducer, GeneralPurpose |
| `agents/registry.py`      | Task registry for SendMessage / TaskStop / auto-resume |
| `skills/`                 | Two-phase loader + 6 builtin ROCm skills |
| `mcp/`                    | MCP client wrapper + 2 reference servers (DockerHub, PyPI) |
| `sandbox/`                | `Sandbox` (docker-exec per command) + commit DAG |
| `learning/`               | KB store + trajectory store + distiller + error classifier + rule engine |
| `dockerfile/`             | Synthesizer + replay verifier |
| `observability/`          | OTel tracing + Prometheus metrics + JSONL transcripts + startup checkpoints |
| `cli.py`                  | Typer-based entry: migrate / batch / mcp serve / doctor |
| `bootstrap.py`            | 5-phase init pipeline |
| `config.py`               | `Settings` (pydantic-settings) |

## Design principles (what we keep returning to)

1. **Push complexity to the boundaries.** The interior — `query()`, `BaseTool`, `AgentDefinition` — stays small and pleasant. Streaming SSE parsing, prompt-cache discipline, hook precedence, OAuth flows — that's where the engineering lives.
2. **Fail-closed defaults.** A new tool that forgets `is_concurrency_safe` runs serial. A new tool that forgets `is_read_only` is treated as a write. Forgetting a permission check denies, not allows.
3. **Self-describing tools.** Adding tool N+1 requires zero changes to existing code: subclass `BaseTool`, register, done. No central dispatch table.
4. **Discriminated unions over flags.** `Terminal` has 10 reasons. `Continue` has 7. The type system enforces exhaustive handling — the call site can never silently miss a state.
5. **Permission modes, not permission checks.** Six modes; one resolution function; no scattered `if isAllowed`.
6. **Frozen at trust boundaries.** Hook config is snapshotted once. Subsequent disk changes to `.repo2rocm/settings.json` are ignored — defeats TOCTOU.
7. **File-based memory.** Markdown + YAML frontmatter. `vim` / `git` / `rm` are first-class tools for inspecting or correcting what the agent remembers.
