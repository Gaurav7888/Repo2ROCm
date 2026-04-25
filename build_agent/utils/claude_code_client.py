"""
Claude Code Agent SDK integration for Repo2ROCm.

Provides two modes of operation:
  1. LLM replacement mode: Uses the Agent SDK as a drop-in replacement
     for the AMD LLM API Gateway, returning text completions.
  2. Agentic mode: Leverages Claude Code's full agent capabilities
     (sub-agents, skills, memory, built-in tools) to autonomously
     drive environment configuration inside a Docker container.

Requires:
  pip install claude-agent-sdk

Authentication:
  Set ANTHROPIC_API_KEY env var, or use Bedrock/Vertex/Azure via
  CLAUDE_CODE_USE_BEDROCK=1 / CLAUDE_CODE_USE_VERTEX=1 / CLAUDE_CODE_USE_FOUNDRY=1
"""

import asyncio
import json
import os
import subprocess
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

_USE_CLAUDE_CODE = False
_CLAUDE_CODE_MODEL = None
_CLAUDE_CODE_AVAILABLE = None


def _estimate_usage(messages: List[Dict[str, str]],
                    content: str = "",
                    system_prompt: Optional[str] = None) -> Dict[str, int]:
    """Fallback estimate when Claude SDK/CLI does not return usage."""
    prompt_chars = len(system_prompt or "")
    for msg in messages or []:
        prompt_chars += len(str(msg.get("content", "")))
    prompt_tokens = max(1, prompt_chars // 4) if prompt_chars else 0
    completion_tokens = max(1, len(content or "") // 4) if content else 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "estimated": True,
    }


def set_claude_code_mode(enabled: bool, model: Optional[str] = None):
    """Enable/disable Claude Code mode globally."""
    global _USE_CLAUDE_CODE, _CLAUDE_CODE_MODEL
    _USE_CLAUDE_CODE = enabled
    _CLAUDE_CODE_MODEL = model


def is_claude_code_mode() -> bool:
    return _USE_CLAUDE_CODE


def _check_agent_sdk_available() -> bool:
    """Check if the claude-agent-sdk Python package is installed."""
    global _CLAUDE_CODE_AVAILABLE
    if _CLAUDE_CODE_AVAILABLE is not None:
        return _CLAUDE_CODE_AVAILABLE
    try:
        import claude_agent_sdk  # noqa: F401
        _CLAUDE_CODE_AVAILABLE = True
    except ImportError:
        _CLAUDE_CODE_AVAILABLE = False
    return _CLAUDE_CODE_AVAILABLE


def _check_claude_cli_available() -> bool:
    """Check if the `claude` CLI is installed and accessible."""
    return shutil.which("claude") is not None


# ── Mode 1: Agent SDK as LLM replacement ─────────────────────────────────────

async def _query_agent_sdk(
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> Tuple[str, Dict[str, int]]:
    """
    Use Claude Agent SDK's query() for a single-turn text completion.

    This strips out built-in tools so the SDK acts as a pure LLM,
    compatible with the existing get_llm_response() interface.
    """
    from claude_agent_sdk import query, ClaudeAgentOptions

    combined_prompt, extracted_system = _serialize_conversation(messages, system_prompt)

    options_kwargs: Dict[str, Any] = {
        "allowed_tools": [],
        "permission_mode": "dontAsk",
        "max_turns": 25,
    }
    if extracted_system:
        options_kwargs["system_prompt"] = extracted_system
    resolved = _resolve_claude_model(model)
    if resolved:
        options_kwargs["model"] = resolved

    options = ClaudeAgentOptions(**options_kwargs)

    result_text = ""
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    async for message in query(prompt=combined_prompt, options=options):
        if hasattr(message, "message") and hasattr(message.message, "content"):
            for block in message.message.content:
                if hasattr(block, "text"):
                    result_text += block.text
        elif hasattr(message, "result"):
            if not result_text and isinstance(message.result, str):
                result_text = message.result

        if hasattr(message, "message") and hasattr(message.message, "usage"):
            u = message.message.usage
            usage["prompt_tokens"] = getattr(u, "input_tokens", 0)
            usage["completion_tokens"] = getattr(u, "output_tokens", 0)
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

    if usage["total_tokens"] == 0:
        usage = _estimate_usage(messages, result_text, system_prompt)

    return result_text, usage


def get_claude_code_response(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    n: int = 1,
    max_tokens: int = 4096,
    system_prompt: Optional[str] = None,
) -> Tuple[Optional[List[str]], Optional[Dict[str, int]]]:
    """
    Drop-in replacement for get_llm_response() that uses Claude Code Agent SDK.

    Returns: ([content], usage_dict) matching get_llm_response() signature.
    """
    if not _check_agent_sdk_available():
        return _get_claude_cli_response(
            model, messages, temperature, max_tokens, system_prompt
        )

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                content, usage = pool.submit(
                    lambda: asyncio.run(_query_agent_sdk(
                        messages, system_prompt, model, temperature, max_tokens
                    ))
                ).result()
        else:
            content, usage = loop.run_until_complete(
                _query_agent_sdk(messages, system_prompt, model, temperature, max_tokens)
            )
        return [content], usage
    except RuntimeError:
        content, usage = asyncio.run(
            _query_agent_sdk(messages, system_prompt, model, temperature, max_tokens)
        )
        return [content], usage
    except Exception as e:
        print(f"Claude Code Agent SDK error: {e}")
        return _get_claude_cli_response(
            model, messages, temperature, max_tokens, system_prompt
        )


def _resolve_claude_model(model: Optional[str]) -> Optional[str]:
    """Map model names to Claude CLI --model values."""
    effective = model or _CLAUDE_CODE_MODEL
    if not effective:
        return None
    name = effective.lower().strip()
    if "opus" in name:
        return "opus"
    if "haiku" in name:
        return "haiku"
    # Default to sonnet for anything else (including "sonnet", "claude-sonnet-*", etc.)
    return "sonnet"


def _serialize_conversation(messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """
    Serialize a multi-turn conversation into a single prompt string.

    The Claude CLI only accepts a single prompt string, not a message array.
    We structure the conversation with clear role headers so the model
    understands the multi-turn context and doesn't repeat prior actions.
    """
    extracted_system = system_prompt
    parts = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if not content:
            continue

        if role == "system":
            if not extracted_system:
                extracted_system = content
            else:
                parts.append(f"[SYSTEM OBSERVATION]:\n{content}")
        elif role == "assistant":
            parts.append(f"[YOUR PREVIOUS RESPONSE]:\n{content}")
        elif role == "user":
            parts.append(f"[USER]:\n{content}")

    if len(parts) > 1:
        conversation = "\n\n---\n\n".join(parts)
        prompt = (
            "Below is the conversation history of your previous actions and "
            "their observations. Continue from where you left off. Do NOT repeat "
            "actions you already took — read the observations carefully and take "
            "the NEXT logical step.\n\n"
            + conversation
            + "\n\n---\n\n"
            "Now provide your next Thought and Action. Do NOT repeat any "
            "command you already executed above."
        )
    else:
        prompt = parts[0] if parts else ""

    return prompt, extracted_system


def _get_claude_cli_response(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 4096,
    system_prompt: Optional[str] = None,
) -> Tuple[Optional[List[str]], Optional[Dict[str, int]]]:
    """
    Fallback: use the `claude` CLI in print mode when Agent SDK is unavailable.

    Runs as a pure text completion — all tools are disabled so Claude
    cannot attempt Bash/Read/etc. and hit permission denials.
    """
    if not _check_claude_cli_available():
        raise RuntimeError(
            "Neither claude-agent-sdk nor claude CLI is available. "
            "Install via: pip install claude-agent-sdk  OR  "
            "curl -fsSL https://claude.ai/install.sh | bash"
        )

    prompt, extracted_system = _serialize_conversation(messages, system_prompt)

    cmd = [
        "claude", "-p",
        "--output-format", "json",
        "--bare",
        "--tools", "",
        "--dangerously-skip-permissions",
        "--max-turns", "25",
    ]

    resolved_model = _resolve_claude_model(model)
    if resolved_model:
        cmd.extend(["--model", resolved_model])

    if extracted_system:
        cmd.extend(["--system-prompt", extracted_system])

    cmd.append(prompt)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )

        output = result.stdout.strip()

        try:
            parsed = json.loads(output)
            content = parsed.get("result", output)
            usage_data = parsed.get("usage", {})
            usage = {
                "prompt_tokens": (
                    usage_data.get("input_tokens", 0)
                    + usage_data.get("cache_read_input_tokens", 0)
                    + usage_data.get("cache_creation_input_tokens", 0)
                ),
                "completion_tokens": usage_data.get("output_tokens", 0),
                "total_tokens": 0,
            }
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

            if parsed.get("is_error") and not content:
                error_reason = parsed.get("terminal_reason", "unknown")
                raise RuntimeError(
                    f"claude CLI returned error (reason={error_reason}): {output[:500]}"
                )
        except json.JSONDecodeError:
            if result.returncode != 0:
                error_msg = result.stderr.strip() or output
                raise RuntimeError(
                    f"claude CLI failed (rc={result.returncode}): {error_msg[:500]}"
                )
            content = output
            usage = _estimate_usage(messages, content, system_prompt)

        return [content], usage
    except subprocess.TimeoutExpired:
        raise RuntimeError("claude CLI timed out after 600s")


# ── Mode 2: Full Agentic Mode ────────────────────────────────────────────────

async def _run_agent_sdk_agentic(
    task_prompt: str,
    system_prompt: str,
    working_directory: str,
    model: Optional[str] = None,
    max_turns: int = 100,
    allowed_tools: Optional[List[str]] = None,
    agents: Optional[Dict[str, Any]] = None,
    docker_container_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run Claude Code Agent SDK in full agentic mode.

    Claude Code gets direct access to Bash, Read, Edit, etc. When a Docker
    container ID is provided, Bash commands are prefixed with docker exec
    to run inside the container.
    """
    from claude_agent_sdk import query, ClaudeAgentOptions

    if allowed_tools is None:
        allowed_tools = ["Read", "Edit", "Bash", "Glob", "Grep", "WebSearch", "WebFetch"]

    if docker_container_id:
        docker_preamble = (
            f"\n\nIMPORTANT: All shell commands must be executed inside Docker container "
            f"'{docker_container_id}'. Prefix every Bash command with:\n"
            f"  docker exec -w /repo {docker_container_id} bash -c '...'\n"
            f"For example:\n"
            f"  docker exec -w /repo {docker_container_id} bash -c 'pip install -q numpy'\n"
            f"The repository is mounted at /repo inside the container.\n"
        )
        system_prompt += docker_preamble

    options_kwargs: Dict[str, Any] = {
        "allowed_tools": allowed_tools,
        "permission_mode": "bypassPermissions",
        "max_turns": max_turns,
        "system_prompt": system_prompt,
        "cwd": working_directory,
    }
    resolved = _resolve_claude_model(model)
    if resolved:
        options_kwargs["model"] = resolved

    if agents:
        from claude_agent_sdk import AgentDefinition
        agent_defs = {}
        for name, cfg in agents.items():
            agent_defs[name] = AgentDefinition(
                description=cfg.get("description", ""),
                prompt=cfg.get("prompt", ""),
                tools=cfg.get("tools", ["Read", "Edit", "Bash", "Glob", "Grep"]),
            )
        options_kwargs["agents"] = agent_defs
        if "Agent" not in allowed_tools:
            allowed_tools.append("Agent")
            options_kwargs["allowed_tools"] = allowed_tools

    options = ClaudeAgentOptions(**options_kwargs)

    result = {
        "success": False,
        "messages": [],
        "final_text": "",
        "tool_calls": [],
        "total_turns": 0,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    async for message in query(prompt=task_prompt, options=options):
        msg_type = getattr(message, "type", "unknown")

        if msg_type == "assistant" and hasattr(message, "message"):
            content_blocks = getattr(message.message, "content", [])
            for block in content_blocks:
                if hasattr(block, "text"):
                    result["final_text"] += block.text
                    result["messages"].append({"type": "text", "content": block.text})
                elif hasattr(block, "name"):
                    tool_info = {
                        "tool": block.name,
                        "input": getattr(block, "input", {}),
                    }
                    result["tool_calls"].append(tool_info)
                    result["messages"].append({"type": "tool_call", **tool_info})

            usage = getattr(message.message, "usage", None)
            if usage:
                result["usage"]["prompt_tokens"] += getattr(usage, "input_tokens", 0)
                result["usage"]["completion_tokens"] += getattr(usage, "output_tokens", 0)
                result["usage"]["total_tokens"] = (
                    result["usage"]["prompt_tokens"] + result["usage"]["completion_tokens"]
                )
            result["total_turns"] += 1

        elif msg_type == "result":
            result["success"] = getattr(message, "subtype", "") == "success"
            if hasattr(message, "result") and isinstance(message.result, str):
                if not result["final_text"]:
                    result["final_text"] = message.result

    return result


def run_claude_code_agent(
    task_prompt: str,
    system_prompt: str,
    working_directory: str,
    model: Optional[str] = None,
    max_turns: int = 100,
    allowed_tools: Optional[List[str]] = None,
    agents: Optional[Dict[str, Any]] = None,
    docker_container_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Synchronous wrapper for the full agentic mode.

    Returns dict with keys: success, messages, final_text, tool_calls, total_turns, usage
    """
    if not _check_agent_sdk_available():
        return _run_claude_cli_agentic(
            task_prompt, system_prompt, working_directory,
            model, max_turns, docker_container_id,
        )

    try:
        return asyncio.run(_run_agent_sdk_agentic(
            task_prompt, system_prompt, working_directory,
            model, max_turns, allowed_tools, agents, docker_container_id,
        ))
    except Exception as e:
        print(f"Claude Code Agent SDK agentic mode error: {e}")
        return _run_claude_cli_agentic(
            task_prompt, system_prompt, working_directory,
            model, max_turns, docker_container_id,
        )


def _run_claude_cli_agentic(
    task_prompt: str,
    system_prompt: str,
    working_directory: str,
    model: Optional[str] = None,
    max_turns: int = 100,
    docker_container_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fallback agentic mode using `claude` CLI.
    """
    if not _check_claude_cli_available():
        raise RuntimeError(
            "Claude Code is not available. "
            "Install via: pip install claude-agent-sdk  OR  "
            "curl -fsSL https://claude.ai/install.sh | bash"
        )

    if docker_container_id:
        system_prompt += (
            f"\n\nAll Bash commands must run inside Docker container '{docker_container_id}'. "
            f"Use: docker exec -w /repo {docker_container_id} bash -c '...'\n"
        )

    cmd = [
        "claude", "-p",
        "--output-format", "json",
        "--dangerously-skip-permissions",
        "--system-prompt", system_prompt,
        "--max-turns", str(max_turns),
    ]

    resolved_model = _resolve_claude_model(model)
    if resolved_model:
        cmd.extend(["--model", resolved_model])

    cmd.append(task_prompt)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,
            cwd=working_directory,
        )

        try:
            parsed = json.loads(result.stdout)
            return {
                "success": result.returncode == 0,
                "messages": [],
                "final_text": parsed.get("result", result.stdout),
                "tool_calls": [],
                "total_turns": parsed.get("num_turns", 0),
                "usage": {
                    "prompt_tokens": parsed.get("input_tokens", 0),
                    "completion_tokens": parsed.get("output_tokens", 0),
                    "total_tokens": (
                        parsed.get("input_tokens", 0) + parsed.get("output_tokens", 0)
                    ),
                },
            }
        except json.JSONDecodeError:
            return {
                "success": result.returncode == 0,
                "messages": [],
                "final_text": result.stdout.strip(),
                "tool_calls": [],
                "total_turns": 0,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "messages": [],
            "final_text": "Claude Code CLI timed out after 2 hours",
            "tool_calls": [],
            "total_turns": 0,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }


# ── Sub-agent definitions for ROCm workflows ─────────────────────────────────

def get_rocm_subagents() -> Dict[str, Dict[str, Any]]:
    """Return sub-agent definitions for ROCm-specific specialized tasks."""
    return {
        "rocm-configurator": {
            "description": (
                "Expert ROCm environment configurator. Handles Docker container setup, "
                "base image selection, system-level dependencies (apt packages, ROCm libs), "
                "and environment variable configuration for AMD GPU workloads."
            ),
            "prompt": (
                "You are an expert ROCm environment configuration agent. Your job is to set up "
                "Docker containers for AMD GPU workloads.\n\n"
                "Key responsibilities:\n"
                "- Select and configure the correct ROCm Docker base image\n"
                "- Install system-level dependencies (apt packages)\n"
                "- Set environment variables (ROCM_HOME, HIP_VISIBLE_DEVICES, etc.)\n"
                "- Verify GPU visibility with rocm-smi and torch.cuda.is_available()\n"
                "- Handle ROCm-specific library paths and LD_LIBRARY_PATH\n\n"
                "Always verify GPU accessibility before declaring success."
            ),
            "tools": ["Bash", "Read", "Edit", "Glob", "Grep"],
        },
        "cuda-kernel-migrator": {
            "description": (
                "CUDA-to-HIP kernel migration specialist. Handles hipification of .cu/.cuh files, "
                "compilation fixes, and numerical equivalence testing for custom CUDA kernels."
            ),
            "prompt": (
                "You are a CUDA-to-HIP migration expert. Your job is to convert custom CUDA "
                "kernels to work on AMD GPUs via HIP.\n\n"
                "Workflow:\n"
                "1. Inventory all .cu/.cuh files and classify their purpose\n"
                "2. Run hipify-clang for automated conversion\n"
                "3. Fix compilation errors from the conversion\n"
                "4. Verify numerical equivalence between original and hipified kernels\n\n"
                "Key AMD differences to handle:\n"
                "- Warp size: 64 on AMD vs 32 on NVIDIA\n"
                "- Replace __shfl_* with __shfl_*_sync\n"
                "- Use hipBLAS instead of cuBLAS\n"
                "- AMD LDS vs NVIDIA shared memory patterns\n"
                "- hipcc compiler flags differ from nvcc\n"
            ),
            "tools": ["Bash", "Read", "Edit", "Glob", "Grep"],
        },
        "dependency-resolver": {
            "description": (
                "Python dependency resolution specialist. Handles pip/conda/poetry dependency "
                "conflicts, CUDA-to-ROCm package mappings, and version compatibility issues."
            ),
            "prompt": (
                "You are a Python dependency resolution expert specializing in CUDA-to-ROCm "
                "package migrations.\n\n"
                "Key mappings to know:\n"
                "- torch (CUDA) -> torch (ROCm wheels from repo.radeon.com)\n"
                "- nvidia-* packages -> skip (not needed on ROCm)\n"
                "- flash-attn -> flash-attn from ROCm index\n"
                "- bitsandbytes -> bitsandbytes-rocm\n"
                "- xformers -> skip or use ROCm fork\n"
                "- triton -> triton-rocm or pytorch-triton-rocm\n\n"
                "Always check pipdeptree for dependency conflicts after installs."
            ),
            "tools": ["Bash", "Read", "Edit", "Glob", "Grep"],
        },
        "triton-kernel-agent": {
            "description": (
                "Triton kernel compatibility specialist for AMD ROCm. Handles Triton autotuning "
                "configs, warp size fixes, and AMD-specific kernel compilation."
            ),
            "prompt": (
                "You are a Triton kernel expert specializing in AMD ROCm compatibility.\n\n"
                "Key issues to handle:\n"
                "- @triton.autotune configs may assume NVIDIA warp size (32)\n"
                "- tl.dot accumulator types behave differently on AMD\n"
                "- num_warps must be adjusted for AMD's wavefront64 architecture\n"
                "- Some tl.constexpr patterns compile but produce wrong results on gfx targets\n"
                "- Install triton-rocm or pytorch-triton-rocm instead of vanilla triton\n"
            ),
            "tools": ["Bash", "Read", "Edit", "Glob", "Grep"],
        },
        "paper-reproducer": {
            "description": (
                "Research-paper result reproduction judge. Reads /repo/paper.pdf, locates the "
                "numeric claims for a named experiment, parses the experiment's actual output "
                "log, and renders a strict-JSON verdict (reproduced / partial / not_reproduced) "
                "using a numeric tolerance check with LLM-judge fallback."
            ),
            "prompt": (
                "You are the paper-reproducer judge. Your ONLY job is to decide whether a "
                "just-run experiment reproduces a claim from the paper, and return a strict JSON "
                "verdict. You do NOT configure environments or install packages.\n\n"
                "Inputs you can rely on:\n"
                "- `/repo/paper.pdf` (use the `Read` tool; Claude Code reads PDFs natively).\n"
                "- `/repo/paper_experiment.log` with the chosen experiment's stdout.\n"
                "- The `PAPER REPRODUCTION TARGET` section of the strategic plan (the caller "
                "  pastes its relevant fields when delegating), which names the experiment, "
                "  the expected metric, and the tolerance rule.\n\n"
                "Workflow:\n"
                "1. Open /repo/paper.pdf with `Read`. Find the table/figure/section that "
                "   reports the target metric for the chosen experiment. Extract the exact "
                "   paper-reported value (and units).\n"
                "2. Read /repo/paper_experiment.log (and any artefact files it references). "
                "   Parse the actual numeric value of the same metric from the run.\n"
                "3. Compute `delta_pct = |actual - expected| / max(|expected|, 1e-8) * 100`.\n"
                "4. Tolerance defaults (unless overridden by the plan's tolerance rule):\n"
                "   - speedups / ratios:  <= 15% relative delta\n"
                "   - accuracy / F1 (%):  <= 3 absolute points (i.e. treat as absolute delta)\n"
                "   - perplexity / loss:  <= 5% relative delta\n"
                "   - throughput:         <= 15% relative delta\n"
                "5. Verdict:\n"
                "   - `reproduced`       if the delta is within tolerance.\n"
                "   - `partial`          if the direction/magnitude is plausible but outside "
                "                         tolerance (e.g. 20% off a 2x claim), or if only a "
                "                         related metric matched.\n"
                "   - `not_reproduced`   if the delta is gross, has the wrong sign, or the "
                "                         experiment failed to produce the metric.\n"
                "6. If the paper does NOT report a directly comparable number for this exact "
                "   experiment (or the units differ), DO NOT guess a number. Instead switch to "
                "   LLM-judge mode: summarise the paper's qualitative claim for this "
                "   experiment and the run's observed qualitative behaviour, and choose a "
                "   verdict with a short written justification.\n\n"
                "Return ONLY a single strict-JSON object, no prose, no markdown fences:\n"
                "{\n"
                "  \"verdict\": \"reproduced|partial|not_reproduced\",\n"
                "  \"metric\": {\n"
                "    \"name\": \"...\",\n"
                "    \"expected\": \"...\",\n"
                "    \"actual\": \"...\",\n"
                "    \"units\": \"...\"\n"
                "  },\n"
                "  \"delta_pct\": <number or null>,\n"
                "  \"tolerance_used\": \"...\",\n"
                "  \"mode\": \"numeric|llm_judge\",\n"
                "  \"justification\": \"<= 3 sentences\"\n"
                "}\n\n"
                "NEVER fabricate numbers. If parsing fails, return verdict=`not_reproduced` "
                "with a parsing note in `justification`."
            ),
            "tools": ["Bash", "Read", "Grep", "Glob"],
        },
    }


# ── CLAUDE.md generation ─────────────────────────────────────────────────────

def generate_claude_md(
    repo_path: str,
    plan: str = "",
    kb_context: str = "",
    rocm_mode: bool = True,
    paper_pdf_path: Optional[str] = None,
    reproduce_results: bool = False,
) -> str:
    """Generate CLAUDE.md content for Claude Code project memory."""
    sections = []
    sections.append("# Repo2ROCm Project Context\n")

    if rocm_mode:
        sections.append("## ROCm Migration Mode\n")
        sections.append(
            "This project is being configured for AMD ROCm GPU support. "
            "All CUDA-specific packages must be replaced with ROCm equivalents. "
            "NVIDIA-only packages (nvidia-*, cuda-*) must be skipped.\n"
        )

    sections.append("## Key Conventions\n")
    sections.append("- The repository source code is at `/repo` inside the Docker container")
    sections.append("- Use `pip install -q` for quiet installs to reduce output noise")
    sections.append("- Do NOT modify test files")
    sections.append("- Use `pipdeptree -p <pkg>` to inspect dependency chains")
    sections.append("- Verify GPU availability: `python -c \"import torch; print(torch.cuda.is_available())\"` ")
    sections.append("")

    sections.append("## Retrieval / Research Tools\n")
    sections.append(
        "If this project is running under the Repo2ROCm configuration loop, the parent "
        "agent exposes retrieval tools with these intents:\n"
        "- `paper_recall \"<question>\"` → retrieve paper-only context plus this-run state\n"
        "- `mem_recall \"<question>\"` → retrieve this-run context (decisions, failures, fixes)\n"
        "- `graphify_query \"<question>\"` → query the code graph for file/symbol locations before broad shell search\n"
        "- `pypi_versions <pkg>` / `dockerhub_tags <image>` → deterministic live version/tag lookups\n"
        "- `web_search \"<query>\"` / `visit_url <url>` → cached internet search + page read, especially for AMD/ROCm/HIP issues\n"
        "- `deep_research \"<question>\"` → bounded AMD-aware research helper that composes the above\n"
        "\n"
        "For paper reproduction, NEVER read `/repo/paper.pdf` directly as the first step; "
        "prefer `paper_recall`, then `graphify_query`, then `deep_research`, and only then "
        "raw PDF reads if needed.\n"
        "For AMD/ROCm-specific install/runtime issues, prefer live web evidence over static knowledge.\n"
    )
    sections.append("")

    if rocm_mode:
        sections.append("## ROCm Package Mappings\n")
        sections.append("| CUDA Package | ROCm Replacement | Install Command |")
        sections.append("|---|---|---|")
        sections.append("| torch (CUDA) | torch (ROCm) | `pip install torch --index-url https://download.pytorch.org/whl/rocm6.1` |")
        sections.append("| flash-attn | flash-attn | `pip install flash-attn --no-build-isolation` (from ROCm index) |")
        sections.append("| bitsandbytes | bitsandbytes-rocm | `pip install bitsandbytes-rocm` |")
        sections.append("| nvidia-* | (skip) | Not needed on ROCm |")
        sections.append("| xformers | (skip or ROCm fork) | Check compatibility first |")
        sections.append("| triton | pytorch-triton-rocm | `pip install pytorch-triton-rocm` |")
        sections.append("")

        sections.append("## Common ROCm Environment Variables\n")
        sections.append("```bash")
        sections.append("export ROCM_HOME=/opt/rocm")
        sections.append("export HIP_VISIBLE_DEVICES=0")
        sections.append("export HSA_OVERRIDE_GFX_VERSION=11.0.0  # if needed for gfx compatibility")
        sections.append("export PYTORCH_ROCM_ARCH=gfx90a  # or gfx942 for MI300X")
        sections.append("```\n")

    if kb_context:
        sections.append("## Knowledge Base Context\n")
        sections.append(kb_context)
        sections.append("")

    if reproduce_results:
        sections.append("## Paper Reproduction Mode\n")
        sections.append(
            "This run has `--reproduce-results` enabled. After the environment is "
            "verified (ROCM_ENV_VERIFIED), you MUST also run one paper experiment "
            "and verify its result against the paper's reported numbers.\n"
        )
        if paper_pdf_path:
            sections.append(
                f"The paper PDF has been placed at `/repo/paper.pdf` "
                f"(host path: `{paper_pdf_path}`). Use the `Read` tool on "
                "`/repo/paper.pdf` to open it directly; Claude Code natively "
                "handles PDFs.\n"
            )
        sections.append(
            "The strategic plan below contains a `PAPER REPRODUCTION TARGET` "
            "section that names the Chosen experiment, its paper-reported metric, "
            "the suggested command, a tolerance rule, and fallback experiments. "
            "Follow that section verbatim.\n"
        )
        sections.append(
            "Delegate result judgement to the `paper-reproducer` sub-agent "
            "(defined in `.claude/agents/paper-reproducer.md`). It will read "
            "`/repo/paper.pdf`, locate the relevant table/figure, parse the "
            "experiment's actual stdout, compute a numeric delta first, and "
            "fall back to an LLM-judge verdict when the metric is not directly "
            "comparable. Return its strict-JSON verdict verbatim to the main "
            "conversation, then echo exactly ONE of:\n"
            "```\n"
            "echo PAPER_RESULT_REPRODUCED metric=<name> actual=<v> expected=<v> delta_pct=<x>\n"
            "echo PAPER_RESULT_NOT_REPRODUCED <one-line reason>\n"
            "```\n"
        )

    if plan:
        sections.append("## Strategic Plan\n")
        sections.append(plan[:6000] if reproduce_results else plan[:3000])
        sections.append("")

    return "\n".join(sections)


def setup_claude_code_project(
    repo_root: str,
    plan: str = "",
    kb_context: str = "",
    learned_context: str = "",
    rocm_mode: bool = True,
    paper_pdf_path: Optional[str] = None,
    reproduce_results: bool = False,
):
    """
    Set up Claude Code project files (.claude/) for the repository.

    Creates CLAUDE.md, sub-agent definitions, and the core ROCm skill.

    When `reproduce_results` is True and `paper_pdf_path` points to a valid
    PDF, the file is (re)copied to `<repo_root>/paper.pdf` so Claude's Read
    tool can open it at `/repo/paper.pdf` inside the container.
    """
    claude_dir = os.path.join(repo_root, ".claude")
    agents_dir = os.path.join(claude_dir, "agents")
    skills_root = os.path.join(claude_dir, "skills")
    rocm_skills_dir = os.path.join(skills_root, "rocm-migration")
    amd_research_skills_dir = os.path.join(skills_root, "amd-live-research")
    os.makedirs(agents_dir, exist_ok=True)
    os.makedirs(rocm_skills_dir, exist_ok=True)
    os.makedirs(amd_research_skills_dir, exist_ok=True)

    if paper_pdf_path and os.path.isfile(paper_pdf_path):
        target = os.path.join(repo_root, "paper.pdf")
        try:
            if os.path.abspath(paper_pdf_path) != os.path.abspath(target):
                import shutil as _shutil
                _shutil.copyfile(paper_pdf_path, target)
        except Exception:
            pass

    claude_md = generate_claude_md(
        repo_root, plan, kb_context, rocm_mode,
        paper_pdf_path=paper_pdf_path,
        reproduce_results=reproduce_results,
    )
    with open(os.path.join(claude_dir, "CLAUDE.md"), "w") as f:
        f.write(claude_md)

    subagents = get_rocm_subagents()
    if not reproduce_results:
        subagents.pop("paper-reproducer", None)
    for name, cfg in subagents.items():
        agent_md = (
            f"---\n"
            f"name: {name}\n"
            f"description: {cfg['description']}\n"
            f"tools: {', '.join(cfg['tools'])}\n"
            f"model: inherit\n"
            f"memory: project\n"
            f"---\n\n"
            f"{cfg['prompt']}\n"
        )
        with open(os.path.join(agents_dir, f"{name}.md"), "w") as f:
            f.write(agent_md)

    skill_md = _generate_rocm_skill()
    with open(os.path.join(rocm_skills_dir, "SKILL.md"), "w") as f:
        f.write(skill_md)

    amd_skill_md = _generate_amd_live_research_skill()
    with open(os.path.join(amd_research_skills_dir, "SKILL.md"), "w") as f:
        f.write(amd_skill_md)

    return claude_dir


def _generate_rocm_skill() -> str:
    """Generate the ROCm migration skill SKILL.md content."""
    return """\
---
name: rocm-migration
description: Step-by-step guide for migrating CUDA-based repositories to AMD ROCm
---

# ROCm Migration Skill

When migrating a CUDA-based repository to ROCm, follow this structured workflow:

## Phase 1: Environment Setup

1. Start from a ROCm Docker image (e.g., `rocm/pytorch:latest`)
2. Verify GPU visibility: `rocm-smi` and `python -c "import torch; print(torch.cuda.is_available())"`
3. Set environment variables:
   ```bash
   export ROCM_HOME=/opt/rocm
   export HIP_VISIBLE_DEVICES=0
   ```

## Phase 2: Dependency Migration

1. Parse requirements.txt/setup.py for CUDA-specific packages
2. Apply package mappings:
   - `torch` -> install from ROCm wheel index
   - `nvidia-*` -> skip entirely
   - `flash-attn` -> install from ROCm-compatible source
   - `bitsandbytes` -> `bitsandbytes-rocm`
   - `triton` -> `pytorch-triton-rocm`
3. Install remaining non-CUDA dependencies normally
4. Run `pipdeptree` to verify no conflicts

## Phase 3: Code Patches

1. Replace `nvidia-smi` calls with `rocm-smi`
2. Guard `torch.backends.cudnn.*` with `if not getattr(torch.version, 'hip', None)`
3. Set `WANDB_MODE=offline` if wandb is used
4. Fix `torch.cuda.amp` deprecated calls -> `torch.amp.autocast('cuda')`

## Phase 4: Custom Kernel Migration (if applicable)

1. Inventory .cu/.cuh files
2. Run `hipify-clang` for automated conversion
3. Fix compilation errors (warp size 64 vs 32, API differences)
4. Verify numerical equivalence

## Phase 5: Verification

1. Run `python -c "import <main_package>"` to verify imports
2. Create minimal mock data if needed
3. Run the project's main script with mock/real data
4. Verify output shows CUDA device usage (not CPU)
5. Signal success: `echo ROCM_ENV_VERIFIED`
"""


def _generate_amd_live_research_skill() -> str:
    """Generate the AMD live research skill SKILL.md content."""
    return """\
---
name: amd-live-research
description: Uses live internet evidence and deterministic package or image lookups for AMD ROCm HIP-specific debugging. Use when a failure mentions ROCm, HIP, gfx, MIOpen, rocBLAS, libamdhip64, flash-attn, xformers, bitsandbytes, Triton, or fast-moving package and image compatibility.
---

# AMD Live Research

## Quick Start

1. Use `graphify_query` for repo structure before broad shell discovery.
2. Use `pypi_versions` or `dockerhub_tags` for package/image facts.
3. Use `web_search` with the exact error plus `AMD ROCm HIP`.
4. Use `visit_url` on one or two high-signal sources.
5. Use `deep_research` when the issue spans multiple versions, packages, or low-level runtime behavior.

## Rules

- Prefer live evidence over static prompt knowledge for AMD-specific facts.
- Prefer deterministic package/image lookups over guessing versions or tags.
- Quote exact error strings, versions, tags, and gfx architectures.
- Let current repo evidence win when it conflicts with old advice.
"""
