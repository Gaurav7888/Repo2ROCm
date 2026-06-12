"""
Observation compactor — Stage 1 of the memory layer.

Goal: turn raw container stdout/stderr into a compact, LLM-friendly slice
without losing the signal. Full text is preserved on disk and routed to
mempalace; only the slice is injected into the next prompt.

Design (grounded in observed turn-12 data, where 94% of the prompt was
the appended history):

    short = head(N=10 lines) + tail(M=60 lines) + extracted_error_markers
    full  = unchanged raw text (for storage)
    parsed = {error_class, error_lines, exit_code, metrics}

Extraction is deterministic / regex-based; no LLM in the loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


HEAD_LINES = 10
TAIL_LINES = 60
MAX_TOTAL_CHARS = 6000  # hard cap on the compacted slice

# Error markers we want to surface even if they fall in the middle of long output.
_ERROR_PATTERNS = [
    re.compile(r"^.*?(Error|Exception|Traceback|FAILED|fatal|undefined symbol|ImportError|ModuleNotFoundError|RuntimeError|OSError|HIPError|CUDA error|out of memory|killed|Segmentation fault).*$",
               re.IGNORECASE | re.MULTILINE),
]
_ERROR_CLASS_RE = re.compile(
    r"\b("
    r"ModuleNotFoundError|ImportError|AttributeError|TypeError|ValueError|"
    r"RuntimeError|OSError|FileNotFoundError|PermissionError|HIPError|"
    r"OutOfMemoryError|AssertionError|KeyError|IndexError|"
    r"SyntaxError|IndentationError"
    r")\b"
)
_METRIC_RE = re.compile(
    r"\b(loss|acc|accuracy|f1|em|exact_match|perplexity|ppl|throughput|tokens?/s|samples?/s|epoch)\s*[:=]\s*([0-9]+\.?[0-9]*)",
    re.IGNORECASE,
)


@dataclass
class CompactedObservation:
    short: str          # what we inject into the next prompt
    full: str           # what we persist (mempalace + trajectory)
    error_class: Optional[str] = None
    error_lines: list = field(default_factory=list)
    metrics: list = field(default_factory=list)  # [(name, value), ...]
    truncated: bool = False
    orig_chars: int = 0
    compact_chars: int = 0


def _extract_error_lines(text: str, max_lines: int = 8) -> list:
    out = []
    seen = set()
    for pat in _ERROR_PATTERNS:
        for m in pat.finditer(text):
            line = m.group(0).strip()
            if line and line not in seen:
                seen.add(line)
                out.append(line)
                if len(out) >= max_lines:
                    return out
    return out


def _classify_error(text: str) -> Optional[str]:
    m = _ERROR_CLASS_RE.search(text)
    return m.group(1) if m else None


def _extract_metrics(text: str, max_metrics: int = 20) -> list:
    out = []
    for m in _METRIC_RE.finditer(text):
        try:
            out.append((m.group(1).lower(), float(m.group(2))))
        except ValueError:
            continue
        if len(out) >= max_metrics:
            break
    return out


def compact(text: str,
            action_content: str = "",
            head_lines: int = HEAD_LINES,
            tail_lines: int = TAIL_LINES,
            max_chars: int = MAX_TOTAL_CHARS) -> CompactedObservation:
    """
    Compact a raw observation string for the LLM.

    The returned `.short` is what goes into the prompt; `.full` is unchanged
    and should be stored verbatim in mempalace + trajectory.
    """
    if text is None:
        text = ""
    orig_len = len(text)
    lines = text.splitlines()

    error_class = _classify_error(text)
    error_lines = _extract_error_lines(text)
    metrics = _extract_metrics(text)

    truncated = False
    if len(lines) <= head_lines + tail_lines:
        body = "\n".join(lines)
    else:
        truncated = True
        head = lines[:head_lines]
        tail = lines[-tail_lines:]
        omitted = len(lines) - head_lines - tail_lines
        body = (
            "\n".join(head)
            + f"\n... [{omitted} lines omitted — full text in run memory] ...\n"
            + "\n".join(tail)
        )

    parts = [body]
    if error_lines:
        parts.append("\n[ERROR-LINES surfaced]:\n" + "\n".join(f"  {l}" for l in error_lines))
    if metrics:
        joined = ", ".join(f"{n}={v}" for n, v in metrics)
        parts.append(f"\n[METRICS detected]: {joined}")

    short = "\n".join(parts)
    if len(short) > max_chars:
        short = short[: max_chars - 80] + f"\n... [hard cap {max_chars} chars] ..."
        truncated = True

    return CompactedObservation(
        short=short,
        full=text,
        error_class=error_class,
        error_lines=error_lines,
        metrics=metrics,
        truncated=truncated,
        orig_chars=orig_len,
        compact_chars=len(short),
    )
