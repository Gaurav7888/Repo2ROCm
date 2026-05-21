"""PDF/HTML \u2192 text extraction helpers.

This module is intentionally minimal: it gives the paper-research agent (and
its tools) a way to turn a PDF or rendered HTML into plain text. All
metric/section/table heuristics live in the agent + skills, not here.

`max_pages=None` (default) returns the entire paper. Pass an explicit int only
when you want to cap up-front \u2014 the LLM is responsible for paging through
large documents via `PaperRead(chunk=N)`, not for working around an extractor
cap.
"""
from __future__ import annotations

import html as html_lib
import re
import shutil
import subprocess
from pathlib import Path

try:
    import pypdf
except ImportError:  # pragma: no cover
    pypdf = None  # type: ignore[assignment]


# ── PDF/HTML text extraction ───────────────────────────────────────────────


def extract_text_from_pdf(pdf_path: Path, *, max_pages: int | None = None) -> str:
    """Best-effort PDF\u2192text. Returns the full paper unless `max_pages` is set.

    Extraction chain:
      1. `pypdf` if installed
      2. `pdftotext -layout` if the binary exists
    """
    text = _extract_text_with_pypdf(pdf_path, max_pages=max_pages)
    if _looks_like_text(text):
        return text
    text = _extract_text_with_pdftotext(pdf_path, max_pages=max_pages)
    return text


def extract_pdf_page_count(pdf_path: Path) -> int:
    """Return the number of pages in a PDF, or 0 if unreadable.

    Used by the `PaperOutline` tool so the LLM can budget chunked reads.
    """
    if pypdf is None:
        return 0
    try:
        reader = pypdf.PdfReader(str(pdf_path))
    except Exception:
        return 0
    try:
        return len(reader.pages)
    except Exception:
        return 0


def extract_pdf_pages(pdf_path: Path, *, pages: list[int]) -> str:
    """Return concatenated text from the requested 1-indexed page numbers."""
    if pypdf is None or not pages:
        return ""
    try:
        reader = pypdf.PdfReader(str(pdf_path))
    except Exception:
        return ""
    out: list[str] = []
    total = 0
    try:
        total = len(reader.pages)
    except Exception:
        total = 0
    for p in pages:
        if p < 1 or (total and p > total):
            continue
        try:
            text = reader.pages[p - 1].extract_text() or ""
        except Exception:
            text = ""
        out.append(f"<!-- page {p} -->\n{text}")
    return "\n\n".join(out)


def _extract_text_with_pypdf(pdf_path: Path, *, max_pages: int | None = None) -> str:
    if pypdf is None:
        return ""
    try:
        reader = pypdf.PdfReader(str(pdf_path))
    except Exception:
        return ""
    out: list[str] = []
    for i, page in enumerate(reader.pages):
        if max_pages is not None and i >= max_pages:
            break
        try:
            out.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(out)


def _extract_text_with_pdftotext(pdf_path: Path, *, max_pages: int | None = None) -> str:
    """Fallback extractor via the system `pdftotext` binary."""
    if shutil.which("pdftotext") is None:
        return ""
    cmd = ["pdftotext", "-layout"]
    if max_pages is not None:
        cmd.extend(["-f", "1", "-l", str(max_pages)])
    cmd.extend([str(pdf_path), "-"])
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except Exception:
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout or ""


def extract_text_from_html_path(html_path: Path) -> str:
    """Read and clean a rendered paper HTML file."""
    if not html_path.is_file():
        return ""
    try:
        raw = html_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    return extract_text_from_html(raw)


def extract_text_from_html(raw_html: str) -> str:
    """Small HTML\u2192text cleaner. Good enough for arXiv / ar5iv rendered papers."""
    if not raw_html:
        return ""
    text = raw_html
    text = re.sub(r"(?is)<(script|style|noscript|svg|math)[^>]*>.*?</\1>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(
        r"(?i)</(p|div|section|article|h1|h2|h3|h4|h5|h6|li|tr|td|th)>", "\n", text
    )
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html_lib.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n+ *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_with_fallbacks(
    pdf_path: Path,
    *,
    max_pages: int | None = None,
    html_path: Path | None = None,
) -> tuple[str, str]:
    """Try PDF extraction first, then sibling HTML.

    Returns `(text, source)` where `source` is `"pdf"`, `"html"`, or `""`.
    """
    pdf_text = extract_text_from_pdf(pdf_path, max_pages=max_pages)
    if _looks_like_text(pdf_text):
        return pdf_text, "pdf"
    if html_path is None:
        html_path = pdf_path.with_suffix(".html")
    html_text = extract_text_from_html_path(html_path)
    if _looks_like_text(html_text):
        return html_text, "html"
    if pdf_text.strip():
        return pdf_text, "pdf"
    if html_text.strip():
        return html_text, "html"
    return "", ""


def _looks_like_text(text: str) -> bool:
    """Heuristic: enough words to be useful downstream."""
    if not text:
        return False
    words = re.findall(r"\w+", text)
    return len(words) >= 40


# ── Lightweight outline helpers (used by PaperOutline) ─────────────────────


_SECTION_RE = re.compile(
    r"^\s*"
    r"(?P<num>\d+(?:\.\d+){0,3})\s+"
    r"(?P<title>[A-Z][A-Za-z0-9 ,\-:/&()]{3,120})\s*$",
    re.MULTILINE,
)

_TABLE_HEADER_RE = re.compile(
    r"^\s*Table\s+(?P<num>\d+)\s*[:.]?\s*(?P<caption>[^\n]{0,240})",
    re.MULTILINE,
)

_FIGURE_HEADER_RE = re.compile(
    r"^\s*Figure\s+(?P<num>\d+)\s*[:.]?\s*(?P<caption>[^\n]{0,240})",
    re.MULTILINE,
)


def scan_section_headers(text: str) -> list[tuple[int, str, str]]:
    """Return [(char_offset, number, title), ...]. Cheap, no LLM."""
    out: list[tuple[int, str, str]] = []
    for m in _SECTION_RE.finditer(text):
        out.append((m.start(), m.group("num"), m.group("title").strip(" .,:")))
    return out


def scan_table_captions(text: str) -> list[tuple[int, str, str]]:
    """Return [(char_offset, table_number, caption), ...]. Cheap, no LLM."""
    out: list[tuple[int, str, str]] = []
    for m in _TABLE_HEADER_RE.finditer(text):
        out.append((m.start(), m.group("num"), m.group("caption").strip()))
    return out


def scan_figure_captions(text: str) -> list[tuple[int, str, str]]:
    out: list[tuple[int, str, str]] = []
    for m in _FIGURE_HEADER_RE.finditer(text):
        out.append((m.start(), m.group("num"), m.group("caption").strip()))
    return out
