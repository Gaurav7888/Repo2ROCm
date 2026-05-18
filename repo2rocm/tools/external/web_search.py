"""WebSearch — wraps a search backend. Stub-friendly: returns 'not configured' by default.

Production: plug in DuckDuckGo via httpx, Bing API, or Brave Search API via env vars.
"""
from __future__ import annotations

import os
from typing import ClassVar

import httpx
from pydantic import BaseModel, Field

from repo2rocm.tools.base import BaseTool, ToolResult, ToolUseContext


class WebSearchInput(BaseModel):
    query: str
    max_results: int = Field(5, ge=1, le=20)


class WebSearchHit(BaseModel):
    title: str
    url: str
    snippet: str = ""


class WebSearchOutput(BaseModel):
    query: str
    hits: list[WebSearchHit]
    backend: str = "duckduckgo"


class WebSearch(BaseTool[WebSearchInput, WebSearchOutput]):
    name: ClassVar[str] = "WebSearch"
    description: ClassVar[str] = (
        "Search the web. Use for fresh AMD/ROCm gotchas not covered by skills or KB."
    )
    input_model: ClassVar[type[BaseModel]] = WebSearchInput
    max_result_size_chars: ClassVar[int] = 6_000
    interrupt_behavior: ClassVar[str] = "cancel"

    def is_concurrency_safe(self, parsed: WebSearchInput) -> bool:
        return True

    def is_read_only(self, parsed: WebSearchInput) -> bool:
        return True

    async def call(
        self, parsed: WebSearchInput, ctx: ToolUseContext
    ) -> ToolResult[WebSearchOutput]:
        backend = os.environ.get("REPO2ROCM_SEARCH_BACKEND", "duckduckgo")
        if backend == "duckduckgo":
            hits = await self._ddg(parsed)
        else:
            hits = []
        body = (
            "\n".join(f"- {h.title}\n  {h.url}\n  {h.snippet}" for h in hits)
            or "(no results)"
        )
        return ToolResult(
            data=WebSearchOutput(query=parsed.query, hits=hits, backend=backend), text=body
        )

    async def _ddg(self, parsed: WebSearchInput) -> list[WebSearchHit]:
        url = "https://duckduckgo.com/html/"
        try:
            async with httpx.AsyncClient(
                timeout=15.0,
                follow_redirects=True,
                headers={"User-Agent": "repo2rocm/2.0"},
            ) as client:
                r = await client.post(url, data={"q": parsed.query})
                r.raise_for_status()
                text = r.text
        except Exception:
            return []
        # ultra-light HTML scrape; production users should plug in a real backend
        import re

        hits: list[WebSearchHit] = []
        for m in re.finditer(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', text
        ):
            url_ = m.group(1)
            title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
            hits.append(WebSearchHit(title=title, url=url_))
            if len(hits) >= parsed.max_results:
                break
        return hits
