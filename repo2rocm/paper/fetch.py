"""Paper fetching primitives (arXiv only, plus generic URL).

For arXiv ids we fetch two artifacts into the paper corpus:

  * `<id>.pdf`  — the canonical PDF
  * `<id>.html` — best-effort rendered HTML (`/html/<id>` first, then ar5iv,
                  then the abstract page as the weakest fallback)

The HTML companion matters because PDF text extraction is brittle in minimal
environments. The paper pipeline can fall back to the rendered HTML when the
PDF parser is unavailable or returns no text.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

import httpx

_ARXIV_RE = re.compile(
    r"""(?:arxiv\.org/(?:abs|pdf)/|arXiv[:\s]+)([0-9]{4}\.[0-9]{4,5})(?:v\d+)?""",
    re.IGNORECASE,
)


def arxiv_id_from_readme(readme_text: str) -> str:
    if not readme_text:
        return ""
    m = _ARXIV_RE.search(readme_text)
    return m.group(1) if m else ""


async def fetch_arxiv_pdf(arxiv_id: str, dest_dir: Path) -> Path:
    """Download the PDF and rendered HTML for an arXiv ID into dest_dir/."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = dest_dir / f"{arxiv_id}.pdf"
    abs_path = dest_dir / f"{arxiv_id}.html"

    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    html_urls = [
        f"https://arxiv.org/html/{arxiv_id}",
        f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}",
        # weakest fallback: at least keep the abstract page if no rendered HTML
        f"https://arxiv.org/abs/{arxiv_id}",
    ]

    async with httpx.AsyncClient(
        timeout=60.0,
        follow_redirects=True,
        headers={"User-Agent": "repo2rocm/2.0"},
    ) as client:
        pdf_task = client.get(pdf_url)
        html_task = _fetch_first_success(client, html_urls)
        r_pdf, html_text = await asyncio.gather(pdf_task, html_task)
        pdf_path.write_bytes(r_pdf.content)
        if html_text:
            abs_path.write_text(html_text, encoding="utf-8", errors="ignore")
    return pdf_path


async def fetch_paper(*, url: str = "", arxiv_id: str = "", dest_dir: Path) -> Path:
    """Generic fetch — prefer arxiv_id; else download `url` as PDF.

    Returns the path to the downloaded PDF.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    if arxiv_id:
        return await fetch_arxiv_pdf(arxiv_id, dest_dir)
    if not url:
        raise ValueError("fetch_paper requires either arxiv_id or url")
    suffix = ".pdf" if ".pdf" in url.lower() else ".bin"
    out = dest_dir / f"paper{suffix}"
    async with httpx.AsyncClient(
        timeout=60.0,
        follow_redirects=True,
        headers={"User-Agent": "repo2rocm/2.0"},
    ) as client:
        r = await client.get(url)
        out.write_bytes(r.content)
    return out


async def _fetch_first_success(
    client: httpx.AsyncClient,
    urls: list[str],
) -> str:
    """Return the first successful HTML body from `urls`, else empty string."""
    for url in urls:
        try:
            r = await client.get(url)
        except Exception:  # noqa: BLE001
            continue
        if r.status_code >= 400:
            continue
        text = r.text or ""
        if text.strip():
            return text
    return ""
