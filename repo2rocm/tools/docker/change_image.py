"""ChangeBaseImage / ChangePythonVersion — replace the container with a new base."""
from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from repo2rocm.tools.base import BaseTool, ToolResult, ToolUseContext


class ChangeBaseImageInput(BaseModel):
    base_image: str


class ChangeBaseImageOutput(BaseModel):
    new_image: str


class ChangeBaseImage(BaseTool[ChangeBaseImageInput, ChangeBaseImageOutput]):
    name: ClassVar[str] = "ChangeBaseImage"
    description: ClassVar[str] = (
        "Restart the sandbox on a different base image. Use only after `DockerHubTags` "
        "confirms the requested tag exists. Forgoes prior installations."
    )
    input_model: ClassVar[type[BaseModel]] = ChangeBaseImageInput
    max_result_size_chars: ClassVar[int] = 1_500

    def is_concurrency_safe(self, parsed: ChangeBaseImageInput) -> bool:
        return False

    def is_read_only(self, parsed: ChangeBaseImageInput) -> bool:
        return False

    async def call(
        self, parsed: ChangeBaseImageInput, ctx: ToolUseContext
    ) -> ToolResult[ChangeBaseImageOutput]:
        if ctx.sandbox is None:
            return ToolResult(
                data=ChangeBaseImageOutput(new_image=""),
                text="no sandbox attached",
                is_error=True,
            )
        await ctx.sandbox.stop()
        ctx.sandbox.current_image = parsed.base_image
        ctx.sandbox.cfg.base_image = parsed.base_image  # operator chose a new base; this IS the new portable base
        await ctx.sandbox.start()
        return ToolResult(
            data=ChangeBaseImageOutput(new_image=parsed.base_image),
            text=f"switched to {parsed.base_image}",
        )


class ChangePythonInput(BaseModel):
    python_version: str  # e.g. "3.10"


class ChangePythonOutput(BaseModel):
    python_version: str


class ChangePythonVersion(BaseTool[ChangePythonInput, ChangePythonOutput]):
    name: ClassVar[str] = "ChangePythonVersion"
    description: ClassVar[str] = (
        "Switch Python version inside the container by re-launching from python:<ver>. "
        "Forgoes prior pip/conda state."
    )
    input_model: ClassVar[type[BaseModel]] = ChangePythonInput
    max_result_size_chars: ClassVar[int] = 1_500

    def is_concurrency_safe(self, parsed: ChangePythonInput) -> bool:
        return False

    def is_read_only(self, parsed: ChangePythonInput) -> bool:
        return False

    async def call(
        self, parsed: ChangePythonInput, ctx: ToolUseContext
    ) -> ToolResult[ChangePythonOutput]:
        if ctx.sandbox is None:
            return ToolResult(
                data=ChangePythonOutput(python_version=""),
                text="no sandbox attached",
                is_error=True,
            )
        new_image = f"python:{parsed.python_version}"
        await ctx.sandbox.stop()
        ctx.sandbox.current_image = new_image
        ctx.sandbox.cfg.base_image = new_image  # operator-chosen base
        await ctx.sandbox.start()
        return ToolResult(
            data=ChangePythonOutput(python_version=parsed.python_version),
            text=f"python switched: {new_image}",
        )
