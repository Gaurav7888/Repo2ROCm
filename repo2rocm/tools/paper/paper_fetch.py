"""PaperFetch — download a paper PDF into the corpus."""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from repo2rocm.core.permissions import PermissionDecision, allow
from repo2rocm.paper import arxiv_id_from_readme, fetch_paper
from repo2rocm.tools.base import BaseTool, ToolResult, ToolUseContext


class PaperFetchInput(BaseModel):
    source: Literal["url", "arxiv_id", "readme_arxiv_id"] = Field(
        ...,
        description=(
            "Where to find the paper: explicit URL, explicit arXiv id, or extract "
            "from the project's README."
        ),
    )
    value: str = Field("", description="URL or arXiv id; ignored for readme_arxiv_id.")
    readme_text: str = Field(
        "", description="README content; required when source='readme_arxiv_id'."
    )


class PaperFetchOutput(BaseModel):
    arxiv_id: str = ""
    pdf_path: str = ""
    bytes: int = 0
    error: str = ""


class PaperFetch(BaseTool[PaperFetchInput, PaperFetchOutput]):
    name: ClassVar[str] = "PaperFetch"
    description: ClassVar[str] = (
        "Download a paper PDF for later extraction. Sources: 'url', 'arxiv_id', or "
        "'readme_arxiv_id' (auto-detect from the project's README). The PDF is "
        "stored under work_dir/papers/."
    )
    input_model: ClassVar[type[BaseModel]] = PaperFetchInput
    max_result_size_chars: ClassVar[int] = 2_000
    interrupt_behavior: ClassVar[str] = "cancel"

    def is_concurrency_safe(self, parsed: PaperFetchInput) -> bool:
        return True

    def is_read_only(self, parsed: PaperFetchInput) -> bool:
        return False  # writes the corpus

    def check_permissions(
        self, parsed: PaperFetchInput, ctx: ToolUseContext
    ) -> PermissionDecision:
        # Only writes a PDF into ctx.workdir/papers/. Safe in any mode.
        return allow("PaperFetch only writes to the paper corpus under workdir")

    async def call(
        self, parsed: PaperFetchInput, ctx: ToolUseContext
    ) -> ToolResult[PaperFetchOutput]:
        dest = Path(ctx.workdir) / "papers"
        arxiv_id = ""
        url = ""
        if parsed.source == "arxiv_id":
            arxiv_id = parsed.value.strip()
            if not arxiv_id:
                return ToolResult(
                    data=PaperFetchOutput(error="empty arxiv_id"),
                    text="arxiv_id required",
                    is_error=True,
                )
        elif parsed.source == "readme_arxiv_id":
            arxiv_id = arxiv_id_from_readme(parsed.readme_text)
            if not arxiv_id:
                return ToolResult(
                    data=PaperFetchOutput(error="no arxiv id found in README"),
                    text="No arXiv id detected in README. Pass source='url' or source='arxiv_id'.",
                    is_error=True,
                )
        else:  # url
            url = parsed.value.strip()
            if not url:
                return ToolResult(
                    data=PaperFetchOutput(error="empty url"),
                    text="url required",
                    is_error=True,
                )
        try:
            path = await fetch_paper(url=url, arxiv_id=arxiv_id, dest_dir=dest)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                data=PaperFetchOutput(arxiv_id=arxiv_id, error=str(exc)),
                text=f"fetch failed: {exc}",
                is_error=True,
            )
        size = path.stat().st_size
        return ToolResult(
            data=PaperFetchOutput(
                arxiv_id=arxiv_id, pdf_path=str(path), bytes=size
            ),
            text=f"saved {size} bytes to {path}",
        )
