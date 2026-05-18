"""Download — batch-install everything in the waiting list via DockerExec.

This is the consolidated install step that prevents N separate `pip install` invocations
from spending time on dependency resolution.
"""
from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from repo2rocm.tools.base import BaseTool, ToolResult, ToolUseContext
from repo2rocm.tools.packaging.waiting_list import _get_or_create_wl


class DownloadInput(BaseModel):
    pip_extra_args: str = ""
    use_uv: bool = False


class DownloadOutput(BaseModel):
    pip_count: int
    apt_count: int
    success: bool
    exit_code: int


class Download(BaseTool[DownloadInput, DownloadOutput]):
    name: ClassVar[str] = "Download"
    description: ClassVar[str] = (
        "Batch-install everything in the waiting list (pip + apt) inside the container."
    )
    input_model: ClassVar[type[BaseModel]] = DownloadInput
    max_result_size_chars: ClassVar[int] = 80_000

    def is_concurrency_safe(self, parsed: DownloadInput) -> bool:
        return False

    def is_read_only(self, parsed: DownloadInput) -> bool:
        return False

    async def call(
        self, parsed: DownloadInput, ctx: ToolUseContext
    ) -> ToolResult[DownloadOutput]:
        wl = _get_or_create_wl(ctx)
        pip_items = [i for i in wl.items if i.tool == "pip"]
        apt_items = [i for i in wl.items if i.tool == "apt"]
        text_parts: list[str] = []
        exit_code = 0

        if apt_items:
            names = " ".join(i.name for i in apt_items)
            cmd = f"DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y --no-install-recommends {names}"
            res = await ctx.sandbox.exec(cmd) if ctx.sandbox else None
            if res:
                text_parts.append(f"$ {cmd}\nexit={res.exit_code}\n{res.stdout[-4000:]}")
                if res.exit_code != 0:
                    exit_code = res.exit_code

        if pip_items:
            installer = "uv pip" if parsed.use_uv else "pip"
            specs = " ".join(f'"{i.normalized()}"' for i in pip_items)
            cmd = f"{installer} install {parsed.pip_extra_args} {specs}".strip()
            res = await ctx.sandbox.exec(cmd) if ctx.sandbox else None
            if res:
                text_parts.append(f"$ {cmd}\nexit={res.exit_code}\n{res.stdout[-4000:]}")
                if res.exit_code != 0:
                    exit_code = res.exit_code

        if exit_code == 0:
            wl.clear()

        return ToolResult(
            data=DownloadOutput(
                pip_count=len(pip_items),
                apt_count=len(apt_items),
                success=exit_code == 0,
                exit_code=exit_code,
            ),
            text="\n\n".join(text_parts) or "(waiting list empty)",
            is_error=exit_code != 0,
        )
