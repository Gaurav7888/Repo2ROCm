"""PaperOutline \u2014 cheap structural navigation aid for a paper PDF.

The LLM uses this *before* `PaperRead` to decide which sections and pages are
worth pulling into context. Returns:

  * page count (so the agent knows the budget)
  * section headings with the page they begin on
  * table captions with page numbers (the table's number + caption text is
    almost always enough to know whether to fetch the surrounding text)
  * figure captions
  * a one-paragraph hint about likely \u201cExperimental Setup\u201d / \u201cAppendix\u201d
    locations (cheap regex; the LLM uses skill `/paper_navigation` to decide
    what to do with them)

This is intentionally minimal heuristic. The agent is expected to follow up
with `PaperRead` to extract structured facts \u2014 the outline only points the
way.
"""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from repo2rocm.core.permissions import PermissionDecision, allow
from repo2rocm.paper.extract import (
    extract_pdf_page_count,
    extract_text_from_html_path,
    extract_text_with_fallbacks,
    scan_figure_captions,
    scan_section_headers,
    scan_table_captions,
)
from repo2rocm.tools.base import BaseTool, ToolResult, ToolUseContext


class PaperOutlineInput(BaseModel):
    pdf_path: str = Field(
        ..., description="Path to the fetched PDF (typically from PaperFetch)."
    )
    max_sections: int = Field(80, ge=4, le=400)
    max_tables: int = Field(40, ge=1, le=200)
    max_figures: int = Field(40, ge=1, le=200)


class SectionEntry(BaseModel):
    number: str
    title: str
    char_offset: int
    page_hint: int = 0


class CaptionEntry(BaseModel):
    number: str
    caption: str
    char_offset: int
    page_hint: int = 0


class PaperOutlineOutput(BaseModel):
    pdf_path: str
    text_source: str = ""
    page_count: int = 0
    text_chars: int = 0
    sections: list[SectionEntry] = Field(default_factory=list)
    tables: list[CaptionEntry] = Field(default_factory=list)
    figures: list[CaptionEntry] = Field(default_factory=list)
    setup_hint_offsets: list[int] = Field(
        default_factory=list,
        description=(
            "Char offsets where 'Setup', 'Implementation Details', or "
            "'Hyperparameters' first appear \u2014 likely places to find the "
            "config tuple. Use with PaperRead(section=...) or PaperRead(chunk=N)."
        ),
    )
    error: str = ""


_SETUP_HINTS = (
    "experimental setup",
    "implementation details",
    "hyperparameter",
    "training details",
    "setup",
)


class PaperOutline(BaseTool[PaperOutlineInput, PaperOutlineOutput]):
    name: ClassVar[str] = "PaperOutline"
    description: ClassVar[str] = (
        "Return a structural outline of a paper PDF (page count, section headings, "
        "table captions, figure captions, and 'Experimental Setup'-style hint "
        "offsets). Call this FIRST so you can navigate without reading the whole "
        "paper into context. Follow up with PaperRead to pull specific sections."
    )
    input_model: ClassVar[type[BaseModel]] = PaperOutlineInput
    max_result_size_chars: ClassVar[int] = 24_000

    def is_concurrency_safe(self, parsed: PaperOutlineInput) -> bool:
        return True

    def is_read_only(self, parsed: PaperOutlineInput) -> bool:
        return True

    def check_permissions(
        self, parsed: PaperOutlineInput, ctx: ToolUseContext
    ) -> PermissionDecision:
        return allow("PaperOutline is read-only")

    async def call(
        self, parsed: PaperOutlineInput, ctx: ToolUseContext
    ) -> ToolResult[PaperOutlineOutput]:
        pdf = self._resolve(parsed.pdf_path, ctx)
        if pdf is None or not pdf.is_file():
            return ToolResult(
                data=PaperOutlineOutput(
                    pdf_path=str(parsed.pdf_path), error="pdf not found"
                ),
                text=f"PDF not found: {parsed.pdf_path}",
                is_error=True,
            )

        # Prefer HTML for structure when available (arXiv rendered HTML preserves
        # section numbering more reliably than PDF text). Fall back to PDF.
        html_path = pdf.with_suffix(".html")
        text, source = extract_text_with_fallbacks(pdf, html_path=html_path)
        if not text and html_path.is_file():
            text = extract_text_from_html_path(html_path)
            source = "html"

        page_count = extract_pdf_page_count(pdf)
        chars_per_page = max(1, len(text) // max(1, page_count)) if text else 1

        def _to_page(offset: int) -> int:
            if not page_count or not text:
                return 0
            return min(page_count, max(1, offset // chars_per_page + 1))

        sections = [
            SectionEntry(
                number=num,
                title=title,
                char_offset=off,
                page_hint=_to_page(off),
            )
            for off, num, title in scan_section_headers(text)[: parsed.max_sections]
        ]
        tables = [
            CaptionEntry(
                number=num,
                caption=cap,
                char_offset=off,
                page_hint=_to_page(off),
            )
            for off, num, cap in scan_table_captions(text)[: parsed.max_tables]
        ]
        figures = [
            CaptionEntry(
                number=num,
                caption=cap,
                char_offset=off,
                page_hint=_to_page(off),
            )
            for off, num, cap in scan_figure_captions(text)[: parsed.max_figures]
        ]

        low = text.lower()
        setup_offsets: list[int] = []
        seen: set[int] = set()
        for hint in _SETUP_HINTS:
            idx = 0
            while True:
                idx = low.find(hint, idx)
                if idx < 0:
                    break
                if idx not in seen:
                    setup_offsets.append(idx)
                    seen.add(idx)
                idx += len(hint)
        setup_offsets.sort()

        out = PaperOutlineOutput(
            pdf_path=str(pdf),
            text_source=source,
            page_count=page_count,
            text_chars=len(text),
            sections=sections,
            tables=tables,
            figures=figures,
            setup_hint_offsets=setup_offsets[:20],
        )

        lines = [
            f"Outline of {pdf.name}  ({source or 'no-text'}, "
            f"{page_count} pages, {len(text)} chars)",
        ]
        if sections:
            lines.append("\nSections:")
            for s in sections[:40]:
                page = f" p.{s.page_hint}" if s.page_hint else ""
                lines.append(f"  \u00a7 {s.number} {s.title}{page}")
        if tables:
            lines.append("\nTables:")
            for t in tables[:30]:
                page = f" p.{t.page_hint}" if t.page_hint else ""
                lines.append(f"  Table {t.number}{page}: {t.caption[:140]}")
        if figures:
            lines.append("\nFigures:")
            for f in figures[:20]:
                page = f" p.{f.page_hint}" if f.page_hint else ""
                lines.append(f"  Figure {f.number}{page}: {f.caption[:120]}")
        if setup_offsets:
            lines.append(
                "\nSetup/Implementation-Details hint offsets: "
                + ", ".join(str(o) for o in setup_offsets[:10])
                + ". Use PaperRead(section=\u2026) or PaperRead(chunk=N) to pull the surrounding text."
            )
        return ToolResult(data=out, text="\n".join(lines))

    @staticmethod
    def _resolve(user_path: str, ctx: ToolUseContext) -> Path | None:
        p = Path(user_path)
        if p.is_absolute():
            return p
        return Path(ctx.workdir) / p
