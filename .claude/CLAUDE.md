# Repo2ROCm — Self-Evolving Multi-Agent System

## Project Overview

Repo2ROCm is a multi-agent system that automatically configures GitHub repositories
to run on AMD ROCm GPUs. It clones a repo, analyzes its dependencies, creates a
strategic plan, spins up a Docker container, and drives an AI agent to install
everything and verify the environment works.

## Architecture

```
main.py → planner.py → configuration.py (agent loop) → Docker sandbox
                ↕                    ↕
        Knowledge Base       Intelligence Layer
       (kb_store.py)     (error classifier, rules, memory)
```

## Key Directories

- `build_agent/` — Main source code
- `build_agent/agents/` — Agent implementations (configuration, planner, CUDA/Triton kernel agents)
- `build_agent/utils/` — Utilities (LLM client, sandbox, logging)
- `build_agent/storage/` — Knowledge base, trajectory store, models
- `build_agent/errors/` — Error classifier and seed patterns
- `build_agent/rules/` — Rule engine
- `build_agent/learning/` — Memory provider, trajectory distiller

## Running

```bash
cd build_agent
python3 main.py --full_name owner/repo --sha <commit> --root_path /path --rocm --llm claude-sonnet-4-20250514

# With Claude Code mode:
python3 main.py --full_name owner/repo --sha <commit> --root_path /path --rocm --use-claude-code

# With full agentic mode:
python3 main.py --full_name owner/repo --sha <commit> --root_path /path --rocm --use-claude-code --claude-code-agentic
```

## ROCm Package Mappings

| CUDA Package | ROCm Replacement |
|---|---|
| torch (CUDA) | torch from ROCm wheel index |
| nvidia-* | Skip (not needed) |
| flash-attn | flash-attn from ROCm index |
| bitsandbytes | bitsandbytes-rocm |
| xformers | Skip or ROCm fork |
| triton | pytorch-triton-rocm |

## Conventions

- All Docker commands go through the `Sandbox` class
- The LLM interface is in `utils/llm.py` with `get_llm_response()`
- When `--use-claude-code` is set, LLM calls route through `utils/claude_code_client.py`
- Knowledge base is SQLite-backed (`storage/kb_store.py`)
- Error patterns and rules accumulate across builds (self-evolving)
