"""DockerCommit / DockerRollback — explicit checkpointing tools."""
from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from repo2rocm.tools.base import BaseTool, ToolResult, ToolUseContext


class CommitInput(BaseModel):
    label: str = ""


class CommitOutput(BaseModel):
    commit_id: str


class DockerCommit(BaseTool[CommitInput, CommitOutput]):
    name: ClassVar[str] = "DockerCommit"
    description: ClassVar[str] = (
        "Snapshot the current container state as a new commit. Use after each "
        "successful install/edit so you can roll back on failure."
    )
    input_model: ClassVar[type[BaseModel]] = CommitInput
    max_result_size_chars: ClassVar[int] = 1_000

    def is_concurrency_safe(self, parsed: CommitInput) -> bool:
        return False

    def is_read_only(self, parsed: CommitInput) -> bool:
        return False

    async def call(self, parsed: CommitInput, ctx: ToolUseContext) -> ToolResult[CommitOutput]:
        if ctx.sandbox is None:
            return ToolResult(
                data=CommitOutput(commit_id=""),
                text="no sandbox attached",
                is_error=True,
            )
        commit_id = ctx.sandbox.commit(label=parsed.label)
        return ToolResult(
            data=CommitOutput(commit_id=commit_id),
            text=f"committed: {commit_id} (label={parsed.label!r})",
        )


class RollbackInput(BaseModel):
    commit_id: str | None = None


class RollbackOutput(BaseModel):
    head: str


class DockerRollback(BaseTool[RollbackInput, RollbackOutput]):
    name: ClassVar[str] = "DockerRollback"
    description: ClassVar[str] = (
        "Roll the container back to a prior commit. If commit_id omitted, rolls back one step."
    )
    input_model: ClassVar[type[BaseModel]] = RollbackInput
    max_result_size_chars: ClassVar[int] = 1_000

    def is_concurrency_safe(self, parsed: RollbackInput) -> bool:
        return False

    def is_read_only(self, parsed: RollbackInput) -> bool:
        return False

    async def call(
        self, parsed: RollbackInput, ctx: ToolUseContext
    ) -> ToolResult[RollbackOutput]:
        if ctx.sandbox is None:
            return ToolResult(
                data=RollbackOutput(head=""),
                text="no sandbox attached",
                is_error=True,
            )
        await ctx.sandbox.rollback(to_commit=parsed.commit_id)
        return ToolResult(
            data=RollbackOutput(head=ctx.sandbox.commit_log.head or ""),
            text=f"rolled back, head now {ctx.sandbox.commit_log.head}",
        )
