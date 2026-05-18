"""AgentTool — recursive sub-agent spawning. Plus SendMessage and TaskStop."""
from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from repo2rocm.tools.base import BaseTool, ToolResult, ToolUseContext, register_tool


class AgentInput(BaseModel):
    description: str = Field(..., description="3-5 word summary of the task.")
    prompt: str = Field(..., description="Full task description for the sub-agent.")
    subagent_type: str = Field(
        "general-purpose",
        description="Which agent definition to use (e.g. explore, planner, migrator, verifier).",
    )
    model: str | None = Field(None, description="Override model.")
    run_in_background: bool = False
    name: str | None = Field(None, description="If set, addressable via SendMessage.")


class AgentOutput(BaseModel):
    agent_id: str
    status: Literal["completed", "async_launched", "failed"]
    final_text: str = ""


class Agent(BaseTool[AgentInput, AgentOutput]):
    name: ClassVar[str] = "Agent"
    description: ClassVar[str] = (
        "Spawn a sub-agent to handle a focused task. Use Explore for read-only search, "
        "Planner for static analysis, Migrator for write-heavy migration of a disjoint "
        "file set, Verifier for adversarial env checks, PaperReproducer for paper metrics."
    )
    input_model: ClassVar[type[BaseModel]] = AgentInput
    max_result_size_chars: ClassVar[int] = 30_000

    def is_concurrency_safe(self, parsed: AgentInput) -> bool:
        # Multiple Agent calls in one turn run serially by default; coordinator can spawn
        # multiple via SendMessage if needed.
        return False

    def is_read_only(self, parsed: AgentInput) -> bool:
        # Spawning a sub-agent is pure delegation — not a mutation.
        # The sub-agent's own permission_mode (inherited from parent — see
        # agents/lifecycle.py) decides what IT can do.
        return True

    async def call(self, parsed: AgentInput, ctx: ToolUseContext) -> ToolResult[AgentOutput]:
        # Lazy import to avoid a cycle.
        from repo2rocm.agents.builtin import get_builtin_agents
        from repo2rocm.agents.lifecycle import RunAgentParams, run_agent

        agents = get_builtin_agents()
        agent_def = agents.get(parsed.subagent_type)
        if agent_def is None:
            return ToolResult(
                data=AgentOutput(agent_id="", status="failed", final_text=""),
                text=(
                    f"Unknown subagent_type: {parsed.subagent_type}. "
                    f"Known: {', '.join(sorted(agents))}"
                ),
                is_error=True,
            )

        # Use parent's client by default. The coordinator threads its client through
        # ctx.options["client_factory"].
        client = ctx.options.get("client")
        client_factory = ctx.options.get("client_factory")
        transcript_store = ctx.options.get("transcript_store")
        skill_catalog = ctx.options.get("skill_catalog")
        memory_store = ctx.options.get("memory_store")

        if parsed.model is not None:
            agent_def = agent_def.with_(model=parsed.model)

        result = await run_agent(
            RunAgentParams(
                agent_def=agent_def,
                prompt=parsed.prompt,
                parent_ctx=ctx,
                client=client,
                client_factory=client_factory,
                transcript_store=transcript_store,
                skill_catalog=skill_catalog,
                memory_store=memory_store,
                is_async=parsed.run_in_background,
            )
        )
        return ToolResult(
            data=AgentOutput(
                agent_id=result.task.id,
                status="completed",
                final_text=result.final_text,
            ),
            text=(
                f"[agent: {agent_def.name}, id={result.task.id}, "
                f"turns={result.terminal.turns}, terminal={result.terminal.reason}]\n\n"
                f"{result.final_text}"
            ),
        )


class SendMessageInput(BaseModel):
    to: str = Field(..., description="Agent name or id.")
    message: str
    summary: str | None = None


class SendMessageOutput(BaseModel):
    delivered: bool
    method: Literal["queued", "resumed", "not_found"]


class SendMessage(BaseTool[SendMessageInput, SendMessageOutput]):
    name: ClassVar[str] = "SendMessage"
    description: ClassVar[str] = (
        "Send a message to a named sub-agent. If running, queued for next tool-round; "
        "if completed, auto-resume from disk transcript."
    )
    input_model: ClassVar[type[BaseModel]] = SendMessageInput
    max_result_size_chars: ClassVar[int] = 2_000

    def is_concurrency_safe(self, parsed: SendMessageInput) -> bool:
        return False

    def is_read_only(self, parsed: SendMessageInput) -> bool:
        # Inter-agent messaging is orchestration, not mutation.
        return True

    async def call(
        self, parsed: SendMessageInput, ctx: ToolUseContext
    ) -> ToolResult[SendMessageOutput]:
        from repo2rocm.agents.registry import TaskStatus, get_agent_registry

        reg = get_agent_registry()
        ts = reg.resolve(parsed.to)
        if ts is None:
            return ToolResult(
                data=SendMessageOutput(delivered=False, method="not_found"),
                text=f"unknown recipient: {parsed.to}",
                is_error=True,
            )
        if ts.status == TaskStatus.RUNNING:
            ts.pending_messages.append(parsed.message)
            return ToolResult(
                data=SendMessageOutput(delivered=True, method="queued"),
                text=f"message queued for {parsed.to} ({ts.id})",
            )
        # auto-resume placeholder: full implementation requires transcript rehydration.
        return ToolResult(
            data=SendMessageOutput(delivered=False, method="not_found"),
            text=(
                f"{parsed.to} is in state {ts.status.value}; auto-resume from transcript "
                "is not yet implemented."
            ),
            is_error=True,
        )


class TaskStopInput(BaseModel):
    task_id: str


class TaskStopOutput(BaseModel):
    killed: bool


class TaskStop(BaseTool[TaskStopInput, TaskStopOutput]):
    name: ClassVar[str] = "TaskStop"
    description: ClassVar[str] = "Terminate a running sub-agent or background task."
    input_model: ClassVar[type[BaseModel]] = TaskStopInput
    max_result_size_chars: ClassVar[int] = 1_000

    def is_concurrency_safe(self, parsed: TaskStopInput) -> bool:
        return False

    def is_read_only(self, parsed: TaskStopInput) -> bool:
        # Killing a task is control-plane, not data mutation.
        return True

    async def call(
        self, parsed: TaskStopInput, ctx: ToolUseContext
    ) -> ToolResult[TaskStopOutput]:
        from repo2rocm.agents.registry import get_agent_registry

        ok = get_agent_registry().kill(parsed.task_id)
        return ToolResult(
            data=TaskStopOutput(killed=ok),
            text=("killed" if ok else "no such task") + f": {parsed.task_id}",
        )


def register_agent_tools() -> None:
    register_tool(Agent)
    register_tool(SendMessage)
    register_tool(TaskStop)
