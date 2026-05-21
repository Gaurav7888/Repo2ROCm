"""README parsing — extract run commands, model references, expected outcomes."""
from __future__ import annotations

import re

_CMD_RE = re.compile(
    r"""(?:^|\n)\s*(?:\$\s*|>\s*|python(?:\d(?:\.\d+)?)?\s+|bash\s+|sh\s+|make\s+)"""
    r"""([^\n]+)""",
    re.MULTILINE,
)

_FENCED_RE = re.compile(
    r"```(?:bash|sh|shell|console)?\s*\n([^`]+?)\n```",
    re.MULTILINE,
)

_RUN_PREFIX_RE = re.compile(
    r"^(?:[A-Za-z_][A-Za-z0-9_]*=.*?\s+)*"
    r"(?:python(?:\d(?:\.\d+)?)?(?:\s+-m)?|torchrun|accelerate\s+launch|deepspeed|make|bash|sh|\./)\b"
)


def _iter_shell_commands(block: str) -> list[str]:
    """Return normalized shell commands from a fenced README block.

    Handles multi-line commands that use shell continuations (``\\``) so
    ``accelerate launch ... \\`` followed by ``--foo ...`` becomes a single
    command.
    """
    out: list[str] = []
    pending: list[str] = []

    def _flush() -> None:
        if not pending:
            return
        joined = " ".join(pending).strip()
        joined = re.sub(r"\s+", " ", joined)
        if joined:
            out.append(joined)
        pending.clear()

    for raw in block.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            _flush()
            continue
        if line.startswith("$ "):
            line = line[2:].strip()
        elif line.startswith("> "):
            line = line[2:].strip()

        continued = line.endswith("\\")
        if continued:
            line = line[:-1].rstrip()
        pending.append(line)
        if not continued:
            _flush()

    _flush()
    return out


def _looks_like_run_command(line: str) -> bool:
    return bool(_RUN_PREFIX_RE.search(line.strip()))


def extract_run_commands(readme_text: str, *, limit: int = 12) -> list[str]:
    """Find shell-looking lines that look like run commands."""
    if not readme_text:
        return []
    seen: set[str] = set()
    out: list[str] = []

    for block_match in _FENCED_RE.finditer(readme_text):
        block = block_match.group(1)
        for line in _iter_shell_commands(block):
            if not _looks_like_run_command(line):
                continue
            if line in seen:
                continue
            seen.add(line)
            out.append(line)
            if len(out) >= limit:
                return out

    for m in _CMD_RE.finditer(readme_text):
        line = m.group(1).strip()
        if not line or line in seen:
            continue
        if not _looks_like_run_command(line):
            continue
        seen.add(line)
        out.append(line)
        if len(out) >= limit:
            break
    return out


_MODEL_RE = re.compile(
    r"""(?:huggingface\.co/|hf\.co/|HuggingFace\s+model[:\s]+)"""
    r"""([a-zA-Z0-9_\-./]+)""",
)


def extract_model_references(readme_text: str, *, limit: int = 10) -> list[str]:
    if not readme_text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for m in _MODEL_RE.finditer(readme_text):
        ref = m.group(1).strip().rstrip(".,)]\"'")
        if "/" not in ref or ref in seen:
            continue
        seen.add(ref)
        out.append(ref)
        if len(out) >= limit:
            break
    return out


_OUTCOME_HINTS = (
    "accuracy",
    "perplexity",
    "F1",
    "BLEU",
    "speedup",
    "throughput",
    "samples/s",
    "tokens/s",
)


def extract_expected_outcomes(readme_text: str, *, limit: int = 6) -> list[str]:
    if not readme_text:
        return []
    out: list[str] = []
    for line in readme_text.splitlines():
        if any(tok.lower() in line.lower() for tok in _OUTCOME_HINTS):
            stripped = line.strip(" -*\t")
            if stripped and stripped not in out:
                out.append(stripped[:200])
                if len(out) >= limit:
                    break
    return out


def find_entry_scripts(repo_path: str, readme_text: str | None) -> list[str]:
    import os

    out: list[str] = []
    if readme_text:
        for line in extract_run_commands(readme_text, limit=20):
            m2 = re.match(r"(?:python(?:\d(?:\.\d+)?)?\s+)?(\S+\.py)\b", line)
            if m2:
                out.append(m2.group(1))
    try:
        for entry in os.listdir(repo_path):
            if entry in ("main.py", "train.py", "run.py", "eval.py", "demo.py"):
                out.append(entry)
    except OSError:
        pass
    seen: set[str] = set()
    dedup: list[str] = []
    for s in out:
        if s not in seen:
            seen.add(s)
            dedup.append(s)
    return dedup[:10]
