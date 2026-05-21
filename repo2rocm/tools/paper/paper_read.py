"""PaperRead \u2014 read paper text without arbitrary page caps.

Modes (mutually exclusive):

  * `section="4 Experiments"` \u2014 return the text from this section heading
    up to the next section heading. Heading match is case-insensitive on the
    first 60 chars; the agent should use the exact heading from `PaperOutline`.

  * `pages=[3, 4, 5]` \u2014 return concatenated text from these 1-indexed PDF
    pages (PDF-only; ignored for HTML sources).

  * `chunk=N, chars_per_chunk=12000` \u2014 page through the full text. Chunk 0
    returns chars [0:chars_per_chunk], chunk 1 returns [chars_per_chunk:2*],
    and so on. `total_chunks` and `has_next` are returned so the agent knows
    when to stop. Pages are not chunk-aligned in HTML mode.

  * (no args) \u2014 same as `chunk=0`.

Source preference: HTML (when a `.html` sibling exists) for table-heavy papers,
PDF otherwise. The chosen source is returned in `text_source` so the agent
can decide whether to switch.
"""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field, model_validator

from repo2rocm.core.permissions import PermissionDecision, allow
from repo2rocm.paper.extract import (
    extract_pdf_pages,
    extract_text_from_html_path,
    extract_text_with_fallbacks,
)
from repo2rocm.tools.base import BaseTool, ToolResult, ToolUseContext


class PaperReadInput(BaseModel):
    pdf_path: str = Field(..., description="Path to the fetched PDF (from PaperFetch).")
    section: str = Field(
        "",
        description=(
            "Exact section heading to slice on (e.g. '4 Experiments' or "
            "'4.2 Experimental Setup'). Mutually exclusive with `pages`/`chunk`."
        ),
    )
    pages: list[int] = Field(
        default_factory=list,
        description="1-indexed PDF pages to concatenate. PDF-only; mutually exclusive with `section`.",
    )
    chunk: int = Field(
        0,
        ge=0,
        description=(
            "Chunk index into the full text when no section/pages are given. "
            "Use with `chars_per_chunk` to page through long papers."
        ),
    )
    chars_per_chunk: int = Field(
        12_000,
        ge=1_000,
        le=60_000,
        description="Chunk size in characters when paging via `chunk`.",
    )
    source: str = Field(
        "auto",
        description="'auto' (default), 'pdf', or 'html'. 'auto' prefers HTML when available.",
    )

    @model_validator(mode="after")
    def _validate(self) -> "PaperReadInput":
        if self.section and self.pages:
            raise ValueError("Pass either `section` or `pages`, not both.")
        return self


class PaperReadOutput(BaseModel):
    pdf_path: str
    text_source: str = ""
    section: str = ""
    pages: list[int] = Field(default_factory=list)
    chunk: int = 0
    chars_per_chunk: int = 0
    total_chars: int = 0
    returned_chars: int = 0
    total_chunks: int = 0
    has_next: bool = False
    content: str = ""
    error: str = ""


class PaperRead(BaseTool[PaperReadInput, PaperReadOutput]):
    name: ClassVar[str] = "PaperRead"
    description: ClassVar[str] = (
        "Read paper text with NO page cap. Modes (pick one): "
        "(a) section='4 Experiments' \u2014 slice by section heading; "
        "(b) pages=[3,4,5] \u2014 specific PDF pages; "
        "(c) chunk=N, chars_per_chunk=12000 \u2014 page through the full text. "
        "Returns `has_next` so you know when to stop. Prefer HTML source for "
        "papers with many tables (set source='html')."
    )
    input_model: ClassVar[type[BaseModel]] = PaperReadInput
    max_result_size_chars: ClassVar[int] = 60_000

    def is_concurrency_safe(self, parsed: PaperReadInput) -> bool:
        return True

    def is_read_only(self, parsed: PaperReadInput) -> bool:
        return True

    def check_permissions(
        self, parsed: PaperReadInput, ctx: ToolUseContext
    ) -> PermissionDecision:
        return allow("PaperRead is read-only")

    async def call(
        self, parsed: PaperReadInput, ctx: ToolUseContext
    ) -> ToolResult[PaperReadOutput]:
        pdf = self._resolve(parsed.pdf_path, ctx)
        if pdf is None or not pdf.is_file():
            return ToolResult(
                data=PaperReadOutput(pdf_path=str(parsed.pdf_path), error="pdf not found"),
                text=f"PDF not found: {parsed.pdf_path}",
                is_error=True,
            )
        html_path = pdf.with_suffix(".html")

        # PDF page-range mode: PDF-only.
        if parsed.pages:
            content = extract_pdf_pages(pdf, pages=parsed.pages)
            if not content:
                return ToolResult(
                    data=PaperReadOutput(
                        pdf_path=str(pdf),
                        pages=parsed.pages,
                        text_source="pdf",
                        error="could not extract requested pages",
                    ),
                    text="No text extracted for those pages.",
                    is_error=True,
                )
            return ToolResult(
                data=PaperReadOutput(
                    pdf_path=str(pdf),
                    pages=parsed.pages,
                    text_source="pdf",
                    total_chars=len(content),
                    returned_chars=len(content),
                    content=content,
                ),
                text=content,
            )

        # Resolve source for section/chunk modes.
        src_pref = parsed.source.lower()
        text = ""
        text_source = ""
        if src_pref == "html" and html_path.is_file():
            text = extract_text_from_html_path(html_path)
            text_source = "html"
        elif src_pref == "pdf":
            text, _ = extract_text_with_fallbacks(pdf, html_path=None)
            text_source = "pdf"
        else:  # auto: HTML > PDF
            text, text_source = extract_text_with_fallbacks(pdf, html_path=html_path)

        if not text:
            return ToolResult(
                data=PaperReadOutput(
                    pdf_path=str(pdf),
                    text_source=text_source,
                    error="no extractable text",
                ),
                text="No text could be extracted from this paper.",
                is_error=True,
            )

        total = len(text)

        # Section mode.
        if parsed.section:
            content = _slice_section(text, parsed.section)
            if not content:
                return ToolResult(
                    data=PaperReadOutput(
                        pdf_path=str(pdf),
                        section=parsed.section,
                        text_source=text_source,
                        total_chars=total,
                        error="section not found",
                    ),
                    text=(
                        f"Section heading not found: {parsed.section!r}. Use the "
                        f"exact heading from PaperOutline."
                    ),
                    is_error=True,
                )
            return ToolResult(
                data=PaperReadOutput(
                    pdf_path=str(pdf),
                    section=parsed.section,
                    text_source=text_source,
                    total_chars=total,
                    returned_chars=len(content),
                    content=content,
                ),
                text=content,
            )

        # Chunk mode (default).
        chars_per_chunk = parsed.chars_per_chunk
        total_chunks = max(1, (total + chars_per_chunk - 1) // chars_per_chunk)
        start = parsed.chunk * chars_per_chunk
        end = min(total, start + chars_per_chunk)
        if start >= total:
            return ToolResult(
                data=PaperReadOutput(
                    pdf_path=str(pdf),
                    text_source=text_source,
                    chunk=parsed.chunk,
                    chars_per_chunk=chars_per_chunk,
                    total_chars=total,
                    total_chunks=total_chunks,
                    has_next=False,
                    error="chunk out of range",
                ),
                text=(
                    f"chunk={parsed.chunk} is past the end. "
                    f"total_chunks={total_chunks}."
                ),
                is_error=True,
            )
        content = text[start:end]
        has_next = end < total
        out = PaperReadOutput(
            pdf_path=str(pdf),
            text_source=text_source,
            chunk=parsed.chunk,
            chars_per_chunk=chars_per_chunk,
            total_chars=total,
            returned_chars=len(content),
            total_chunks=total_chunks,
            has_next=has_next,
            content=content,
        )
        header = (
            f"[chunk {parsed.chunk + 1}/{total_chunks}, "
            f"source={text_source}, total_chars={total}, has_next={has_next}]\n"
        )
        return ToolResult(data=out, text=header + content)

    @staticmethod
    def _resolve(user_path: str, ctx: ToolUseContext) -> Path | None:
        p = Path(user_path)
        if p.is_absolute():
            return p
        return Path(ctx.workdir) / p


def _slice_section(text: str, section: str) -> str:
    """Find a section heading and return text up to the next heading.

    Matching is case-insensitive and tolerates extra whitespace. The agent is
    expected to pass the heading exactly as PaperOutline reported it.
    """
    import re

    needle = re.sub(r"\s+", r"\\s+", re.escape(section.strip()))
    pattern = re.compile(rf"(?im)^\s*{needle}\s*$", re.MULTILINE)
    m = pattern.search(text)
    if not m:
        # Fall back: try matching just by the number prefix (e.g. "4.2") if the
        # heading text drifted between PDF extraction and what the agent saw.
        parts = section.strip().split(None, 1)
        if parts and re.match(r"^\d+(?:\.\d+){0,3}$", parts[0]):
            num_pat = re.compile(rf"(?m)^\s*{re.escape(parts[0])}\s+\S.*$")
            m = num_pat.search(text)
    if not m:
        return ""

    start = m.start()
    # Find the next section heading after this one.
    next_heading = re.compile(
        r"(?m)^\s*\d+(?:\.\d+){0,3}\s+[A-Z][A-Za-z0-9 ,\-:/&()]{3,120}\s*$"
    )
    nxt = next_heading.search(text, pos=m.end())
    end = nxt.start() if nxt else len(text)
    return text[start:end].strip()
