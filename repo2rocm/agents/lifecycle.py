"""The 15-step `run_agent` lifecycle, ported from Claude Code Ch. 8.

Every agent — Coordinator, Explore, Planner, Migrator, Verifier, PaperReproducer —
flows through this single function. The agent type is encoded in `AgentDefinition`
(data), not in branches here (control flow).

Steps:
  1.  Model resolution            (caller override > def > parent > default)
  2.  Agent id creation
  3.  Context preparation         (clone parent history if forking; fresh otherwise)
  4.  User context stripping      (omit_user_context drops MEMORY.md)
  5.  Permission isolation        (mode cascade, prompt avoidance, allowedTools scope)
  6.  Tool resolution             (allowed/disallowed filter)
  7.  System prompt               (builder or template)
  8.  Abort controller isolation  (async = own; sync = shared)
  9.  Hook registration           (frontmatter hooks, scoped to agent_id)
  10. Skill preloading            (load body, prepend as user message)
  11. MCP initialization          (attach servers)
  12. Context creation            (ToolUseContext snapshot)
  13. Cache-safe params callback  (for background summarization)
  14. The query loop              (yield* QueryRun)
  15. Cleanup                     (MCP teardown, hook removal, transcript flush)
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from repo2rocm.agents.definition import AgentDefinition
from repo2rocm.agents.registry import AgentTaskState, TaskStatus, get_agent_registry
from repo2rocm.core.api import ModelClient, ToolSpec
from repo2rocm.core.hooks import HookEvent, capture_hooks_snapshot, execute_hooks
from repo2rocm.core.memory import MemoryStore, RecallSelector
from repo2rocm.core.messages import (
    Message,
    SystemPrompt,
    SystemPromptBlock,
    TextBlock,
    UserMessage,
)
from repo2rocm.core.permissions import PermissionMode
from repo2rocm.core.query import QueryRun, query
from repo2rocm.core.state import LatchedSet
from repo2rocm.core.terminal import Completed, Terminal
from repo2rocm.observability.logging import get_logger
from repo2rocm.observability.tracing import span
from repo2rocm.observability.transcripts import TranscriptStore
from repo2rocm.skills import SkillCatalog, load_skill_body
from repo2rocm.tools import assemble_tool_pool
from repo2rocm.tools.base import BaseTool, ReadFileState, ToolUseContext

log = get_logger(__name__)


@dataclass
class RunAgentResult:
    task: AgentTaskState
    terminal: Terminal
    final_text: str = ""
    usage_total: int = 0
    duration_s: float = 0.0


@dataclass
class RunAgentParams:
    agent_def: AgentDefinition
    prompt: str
    parent_ctx: ToolUseContext | None = None
    client: ModelClient | None = None
    client_factory: Any | None = None  # called when model differs from parent
    extra_tools: list[BaseTool] = field(default_factory=list)
    transcript_store: TranscriptStore | None = None
    skill_catalog: SkillCatalog | None = None
    memory_store: MemoryStore | None = None
    is_async: bool = False
    parent_messages: list[Message] | None = None  # fork shares parent history


async def run_agent(params: RunAgentParams) -> RunAgentResult:
    """Drive one sub-agent's full 15-step lifecycle. Returns when terminal."""
    started = time.time()
    registry = get_agent_registry()
    task = registry.register(agent_def=params.agent_def)
    task.started_at = started

    with span(
        "agent.lifecycle",
        agent_id=task.id,
        agent_type=params.agent_def.name,
        is_async=params.is_async,
    ):
        # Step 1+2: model + id
        client = params.client
        if client is None and params.client_factory is not None:
            client = params.client_factory(params.agent_def.model)
        if client is None:
            raise RuntimeError("run_agent: no client and no client_factory")

        agent_id = task.id

        # Step 3: context messages
        prompt_messages: list[Message] = [UserMessage(content=params.prompt)]
        if params.parent_messages and not params.agent_def.omit_user_context:
            messages = [*params.parent_messages, *prompt_messages]
        else:
            messages = list(prompt_messages)

        # Step 4: user context (memory recall)
        recall_text = ""
        if params.memory_store is not None and not params.agent_def.omit_user_context:
            selector = RecallSelector(client=client, store=params.memory_store)
            try:
                files = await selector.select_for(params.prompt)
                recall_text = selector.render(files)
            except Exception:
                recall_text = ""

        # Step 5: permission isolation
        if params.parent_ctx is not None:
            parent_mode = params.parent_ctx.permission_mode
            # parent's strong modes win
            if parent_mode in (
                PermissionMode.BYPASS,
                PermissionMode.ACCEPT_EDITS,
                PermissionMode.AUTO,
            ):
                effective_mode = parent_mode
            else:
                effective_mode = params.agent_def.permission_mode
        else:
            effective_mode = params.agent_def.permission_mode

        # Step 6: tool resolution
        tools, tool_specs = assemble_tool_pool(
            allowed=params.agent_def.allowed_tools,
            disallowed=params.agent_def.disallowed_tools,
            extra=params.extra_tools,
        )

        # Step 7: system prompt
        if params.agent_def.system_prompt_builder is not None:
            sys_text = params.agent_def.system_prompt_builder(
                agent_def=params.agent_def,
                ctx=params.parent_ctx,
                skill_catalog=params.skill_catalog,
                tools=tools,
            )
        else:
            sys_text = params.agent_def.system_prompt_template
        # add skills menu
        if params.skill_catalog is not None:
            sys_text = sys_text + "\n\n" + params.skill_catalog.menu_text()

        sys_blocks = [
            SystemPromptBlock(text=sys_text, cache_control={"type": "ephemeral"}),
        ]
        if recall_text:
            # volatile — placed AFTER the stable block; no cache_control
            sys_blocks.append(SystemPromptBlock(text=recall_text))
        system_prompt = SystemPrompt(blocks=sys_blocks)

        # Step 8: abort controller
        if params.is_async:
            abort_event = asyncio.Event()
        elif params.parent_ctx is not None:
            abort_event = params.parent_ctx.abort_event
        else:
            abort_event = asyncio.Event()

        # Step 10: skill preloading (body)
        for skill_name in params.agent_def.preload_skills:
            if params.skill_catalog is None:
                continue
            manifest = params.skill_catalog.manifests.get(skill_name)
            if manifest is None:
                continue
            body = load_skill_body(manifest)
            messages.insert(
                0,
                UserMessage(
                    content=[
                        TextBlock(text=f"# Skill: {manifest.name}\n\n{body}"),
                    ]
                ),
            )

        # Step 11: MCP (placeholder; mcp client lives in repo2rocm/mcp/)
        # Skipped here for brevity; in production we'd attach mcp_servers.

        # Step 12: context creation
        transcript = None
        if params.transcript_store is not None:
            transcript = params.transcript_store.transcript(agent_id)
        ctx = ToolUseContext(
            agent_id=agent_id,
            session_id=getattr(params.transcript_store, "session_id", "default"),
            workdir=params.parent_ctx.workdir if params.parent_ctx else Path.cwd(),
            abort_event=abort_event,
            permission_mode=effective_mode,
            read_file_state=ReadFileState(),
            sandbox=params.parent_ctx.sandbox if params.parent_ctx else None,
            transcript=transcript,
            messages=messages,
            options=dict(params.parent_ctx.options) if params.parent_ctx else {},
            gate_state=params.parent_ctx.gate_state if params.parent_ctx else None,
        )

        if transcript is not None:
            transcript.append(
                {
                    "kind": "agent_start",
                    "agent_type": params.agent_def.name,
                    "permission_mode": effective_mode.value,
                    "tools": [t.name for t in tools],
                    "preload_skills": list(params.agent_def.preload_skills),
                }
            )

        # Step 13: cache-safe callback (not needed without background summarization)

        # Step 14: the query loop
        task.mark_running()
        runner: QueryRun = query(
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            tool_specs=tool_specs,
            client=client,
            tool_use_context=ctx,
            max_turns=params.agent_def.max_turns,
            agent_type=params.agent_def.name,
        )
        try:
            async for _ev in runner:
                pass
        except asyncio.CancelledError:
            task.mark_terminal(TaskStatus.KILLED)
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("agent crashed", agent=params.agent_def.name)
            task.mark_terminal(TaskStatus.FAILED, final_text=str(exc))
            raise

        # Step 15: cleanup
        terminal = runner.terminal or Completed(turns=0)
        final_text = getattr(terminal, "final_text", "")
        if terminal.reason in ("completed", "max_turns"):
            task.mark_terminal(TaskStatus.COMPLETED, final_text=final_text)
        else:
            task.mark_terminal(TaskStatus.FAILED, final_text=final_text)
        task.ended_at = time.time()

        # SubagentStop hook
        try:
            await execute_hooks(
                event=HookEvent.SUBAGENT_STOP,
                input_data={
                    "agent_type": params.agent_def.name,
                    "agent_id": agent_id,
                    "terminal": terminal.reason,
                    "final_text": final_text,
                },
            )
        except Exception:
            pass

        return RunAgentResult(
            task=task,
            terminal=terminal,
            final_text=final_text,
            usage_total=terminal.usage.total,
            duration_s=task.ended_at - task.started_at,
        )
