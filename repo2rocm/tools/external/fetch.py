"""Fetch — HTTP GET an arbitrary URL and return the body (truncated)."""
from __future__ import annotations

from typing import ClassVar

import httpx
from pydantic import BaseModel, Field

from repo2rocm.tools.base import BaseTool, ToolResult, ToolUseContext


class FetchInput(BaseModel):
    url: str
    max_chars: int = Field(8_000, ge=512, le=200_000)


class FetchOutput(BaseModel):
    url: str
    status: int
    content_type: str
    body: str
    truncated: bool = False


class Fetch(BaseTool[FetchInput, FetchOutput]):
    name: ClassVar[str] = "Fetch"
    description: ClassVar[str] = "HTTP GET a URL and return its body (text only, truncated)."
    input_model: ClassVar[type[BaseModel]] = FetchInput
    max_result_size_chars: ClassVar[int] = 200_000
    interrupt_behavior: ClassVar[str] = "cancel"

    def is_concurrency_safe(self, parsed: FetchInput) -> bool:
        return True

    def is_read_only(self, parsed: FetchInput) -> bool:
        return True

    async def call(self, parsed: FetchInput, ctx: ToolUseContext) -> ToolResult[FetchOutput]:
        try:
            async with httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
                headers={"User-Agent": "repo2rocm/2.0"},
            ) as client:
                r = await client.get(parsed.url)
                ct = r.headers.get("content-type", "")
                body = r.text
                truncated = False
                if len(body) > parsed.max_chars:
                    body = body[: parsed.max_chars]
                    truncated = True
                out = FetchOutput(
                    url=parsed.url,
                    status=r.status_code,
                    content_type=ct,
                    body=body,
                    truncated=truncated,
                )
                return ToolResult(
                    data=out,
                    text=f"GET {parsed.url} -> {r.status_code} ({ct})\n\n{body}",
                )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                data=FetchOutput(
                    url=parsed.url, status=0, content_type="", body="", truncated=False
                ),
                text=f"fetch failed: {exc}",
                is_error=True,
            )
