"""
Paper corpus resolution utilities.

This module keeps paper ingestion source-aware while presenting a single
normalized corpus to the planner, Graphify, and paper-shortlisting logic.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse


ARXIV_HTML_BASES = (
    "https://arxiv.org/html",
    "https://ar5iv.labs.arxiv.org/html",
)


@dataclass
class PaperTextSource:
    source_kind: str
    source_label: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_index_payload(self) -> Dict[str, Any]:
        return {
            "source_kind": self.source_kind,
            "source_label": self.source_label,
            "source_file": self.metadata.get("source_file") or self.source_label,
            "text": self.text,
            "metadata": dict(self.metadata or {}),
        }


@dataclass
class PaperCorpus:
    pdf_path: Optional[str] = None
    arxiv_id: str = ""
    source_mode: str = "pdf"
    resolved_modes: List[str] = field(default_factory=list)
    sources: List[PaperTextSource] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def index_text(self) -> str:
        parts: List[str] = []
        for source in self.sources:
            text = (source.text or "").strip()
            if not text:
                continue
            parts.append(
                f"===== SOURCE {source.source_kind.upper()} :: {source.source_label} =====\n{text}"
            )
        return "\n\n".join(parts).strip()

    def has_text(self) -> bool:
        return any((source.text or "").strip() for source in self.sources)

    def source_payloads(self) -> List[Dict[str, Any]]:
        return [source.to_index_payload() for source in self.sources]


def extract_arxiv_id_from_url(url: str) -> Optional[str]:
    if not url:
        return None
    parsed = urlparse(url.strip())
    host = (parsed.netloc or "").lower()
    if "arxiv.org" not in host and "ar5iv.labs.arxiv.org" not in host:
        return None
    path = (parsed.path or "").strip("/")
    if not path:
        return None
    parts = [part for part in path.split("/") if part]
    if not parts:
        return None
    if parts[0] in {"abs", "pdf", "html"} and len(parts) >= 2:
        candidate = parts[1]
    else:
        candidate = parts[-1]
    if candidate.endswith(".pdf"):
        candidate = candidate[:-4]
    candidate = candidate.strip()
    if not candidate:
        return None
    if re.match(r"^\d{4}\.\d{4,5}(?:v\d+)?$", candidate, flags=re.IGNORECASE):
        return candidate
    return None


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_text_with_pymupdf(pdf_path: str,
                               max_chars: Optional[int] = None) -> str:
    try:
        import fitz  # type: ignore
    except Exception:
        return ""
    try:
        parts: List[str] = []
        total = 0
        with fitz.open(pdf_path) as doc:
            for page in doc:
                txt = page.get_text("text")
                if not txt:
                    continue
                parts.append(txt)
                total += len(txt)
                if max_chars is not None and total >= max_chars:
                    break
        out = "\n".join(parts)
        if max_chars is not None:
            out = out[:max_chars]
        return _normalize_text(out)
    except Exception:
        return ""


def _extract_text_with_pdftotext(pdf_path: str,
                                 max_chars: Optional[int] = None) -> str:
    if not shutil.which("pdftotext"):
        return ""
    try:
        res = subprocess.run(
            ["pdftotext", "-layout", pdf_path, "-"],
            capture_output=True,
            timeout=60,
        )
        if res.returncode == 0 and res.stdout:
            out = res.stdout.decode("utf-8", errors="ignore")
            if max_chars is not None:
                out = out[:max_chars]
            return _normalize_text(out)
    except Exception:
        pass
    return ""


def extract_pdf_text(pdf_path: str,
                     max_chars: Optional[int] = None) -> str:
    text = _extract_text_with_pymupdf(pdf_path, max_chars=max_chars)
    if text:
        return text
    return _extract_text_with_pdftotext(pdf_path, max_chars=max_chars)


def fetch_arxiv_html_text(arxiv_id: str,
                          max_chars: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
    if not arxiv_id:
        return "", {"error": "missing_arxiv_id"}
    try:
        import html2text
        import requests
    except Exception as exc:
        return "", {"error": f"missing_dependency:{exc}"}

    converter = html2text.HTML2Text()
    converter.body_width = 0
    converter.ignore_images = True
    converter.ignore_emphasis = False
    converter.ignore_links = False
    converter.ignore_tables = False

    for base in ARXIV_HTML_BASES:
        url = f"{base}/{arxiv_id}"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200 or not resp.text:
                continue
            markdown = converter.handle(resp.text or "")
            text = markdown
            if max_chars is not None:
                text = text[:max_chars]
            text = _normalize_text(text)
            if not text:
                continue
            return text, {"source_url": url, "status_code": resp.status_code}
        except Exception:
            continue
    return "", {"error": "html_fetch_failed"}


def _dedupe_sources(sources: List[PaperTextSource]) -> List[PaperTextSource]:
    deduped: List[PaperTextSource] = []
    seen = set()
    for source in sources:
        text = _normalize_text(source.text)
        if not text:
            continue
        key = re.sub(r"\s+", " ", text).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        source.text = text
        deduped.append(source)
    return deduped


def _normalized_mode(source_mode: str) -> str:
    mode = (source_mode or "pdf").strip().lower()
    if mode not in {"pdf", "html", "both"}:
        return "pdf"
    return mode


def build_paper_corpus(pdf_path: Optional[str],
                       arxiv_id: str = "",
                       source_mode: str = "pdf",
                       max_chars: Optional[int] = None) -> PaperCorpus:
    mode = _normalized_mode(source_mode)
    sources: List[PaperTextSource] = []
    resolved_modes: List[str] = []
    warnings: List[str] = []

    if pdf_path and os.path.isfile(pdf_path) and mode in {"pdf", "both"}:
        pdf_text = extract_pdf_text(pdf_path, max_chars=max_chars)
        if pdf_text:
            sources.append(
                PaperTextSource(
                    source_kind="pdf",
                    source_label=os.path.basename(pdf_path),
                    text=pdf_text,
                    metadata={"source_file": pdf_path, "chars": len(pdf_text)},
                )
            )
            resolved_modes.append("pdf")
        else:
            warnings.append("pdf_text_extraction_failed")

    if arxiv_id and mode in {"html", "both"}:
        html_text, html_meta = fetch_arxiv_html_text(arxiv_id, max_chars=max_chars)
        if html_text:
            sources.append(
                PaperTextSource(
                    source_kind="html",
                    source_label=f"arxiv:{arxiv_id}",
                    text=html_text,
                    metadata={
                        "source_file": html_meta.get("source_url", f"arxiv:{arxiv_id}"),
                        "chars": len(html_text),
                        **html_meta,
                    },
                )
            )
            resolved_modes.append("html")
        else:
            warnings.append(html_meta.get("error", "html_text_fetch_failed"))
    elif mode in {"html", "both"} and not arxiv_id:
        warnings.append("html_mode_requested_without_arxiv_id")

    if not sources and pdf_path and os.path.isfile(pdf_path):
        fallback_text = extract_pdf_text(pdf_path, max_chars=max_chars)
        if fallback_text:
            sources.append(
                PaperTextSource(
                    source_kind="pdf",
                    source_label=os.path.basename(pdf_path),
                    text=fallback_text,
                    metadata={"source_file": pdf_path, "chars": len(fallback_text)},
                )
            )
            if "pdf" not in resolved_modes:
                resolved_modes.append("pdf")
            warnings.append("fell_back_to_pdf_source")

    sources = _dedupe_sources(sources)
    provenance = {
        "requested_mode": mode,
        "resolved_modes": list(resolved_modes),
        "source_count": len(sources),
        "warnings": warnings,
        "arxiv_id": arxiv_id or "",
        "pdf_path": pdf_path or "",
        "source_chars": {
            source.source_kind: len(source.text or "")
            for source in sources
        },
    }
    return PaperCorpus(
        pdf_path=pdf_path,
        arxiv_id=arxiv_id or "",
        source_mode=mode,
        resolved_modes=resolved_modes,
        sources=sources,
        provenance=provenance,
    )
