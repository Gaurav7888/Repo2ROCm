"""DockerHubTags — list available tags for a Docker image."""
from __future__ import annotations

from typing import ClassVar

import httpx
from pydantic import BaseModel, Field

from repo2rocm.tools.base import BaseTool, ToolResult, ToolUseContext


class DHInput(BaseModel):
    image: str = Field(..., description="e.g. 'rocm/pytorch' or 'library/python'.")
    limit: int = Field(20, description="Max tags to return.")


class DHOutput(BaseModel):
    image: str
    tags: list[str]
    not_found: bool = False


class DockerHubTags(BaseTool[DHInput, DHOutput]):
    name: ClassVar[str] = "DockerHubTags"
    description: ClassVar[str] = (
        "List recent tags for a Docker image from Docker Hub. ALWAYS call before ChangeBaseImage."
    )
    input_model: ClassVar[type[BaseModel]] = DHInput
    max_result_size_chars: ClassVar[int] = 4_000
    interrupt_behavior: ClassVar[str] = "cancel"

    def is_concurrency_safe(self, parsed: DHInput) -> bool:
        return True

    def is_read_only(self, parsed: DHInput) -> bool:
        return True

    async def call(self, parsed: DHInput, ctx: ToolUseContext) -> ToolResult[DHOutput]:
        repo = parsed.image
        if "/" not in repo:
            repo = f"library/{repo}"
        url = f"https://hub.docker.com/v2/repositories/{repo}/tags?page_size={parsed.limit}"
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                r = await client.get(url)
                if r.status_code == 404:
                    return ToolResult(
                        data=DHOutput(image=repo, tags=[], not_found=True),
                        text=f"DockerHub: {repo!r} not found.",
                        is_error=True,
                    )
                r.raise_for_status()
                data = r.json()
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                data=DHOutput(image=repo, tags=[]),
                text=f"DockerHub lookup failed: {exc}",
                is_error=True,
            )
        tags = [t.get("name", "") for t in data.get("results", [])][: parsed.limit]
        ctx_gate = getattr(ctx, "gate_state", None)
        if ctx_gate is not None and hasattr(ctx_gate, "mark_dockerhub"):
            ctx_gate.mark_dockerhub(parsed.image)
        text = f"{repo} tags ({len(tags)}):\n  " + "\n  ".join(tags)
        return ToolResult(data=DHOutput(image=repo, tags=tags), text=text)
