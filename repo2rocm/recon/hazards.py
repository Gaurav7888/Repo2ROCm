"""Hazard scans — Python-3.12 stdlib breakage, old version pins, code-level patterns."""
from __future__ import annotations

import os
import re

from repo2rocm.recon.files import read_file
from repo2rocm.recon.report import Hazard

# ── Python 3.12 removed/moved stdlib modules ────────────────────────────────

PY312_REMOVED_MODULES: dict[str, str] = {
    "imp": "importlib",
    "distutils": "setuptools (or vendored distutils)",
    "aifc": "(removed)",
    "audioop": "(removed)",
    "cgi": "(removed; use urllib)",
    "cgitb": "(removed)",
    "chunk": "(removed)",
    "crypt": "(removed)",
    "imghdr": "(use python-magic or filetype)",
    "mailcap": "(removed)",
    "nis": "(removed)",
    "nntplib": "(removed)",
    "ossaudiodev": "(removed)",
    "pipes": "(use subprocess)",
    "sndhdr": "(removed)",
    "spwd": "(removed)",
    "sunau": "(removed)",
    "telnetlib": "(use telnetlib3)",
    "uu": "(use base64)",
    "xdrlib": "(removed)",
}

PY310_COLLECTIONS_ABCS = (
    "Mapping", "MutableMapping", "Sequence", "MutableSequence",
    "Set", "MutableSet", "Iterable", "Iterator", "Callable",
    "ByteString", "Hashable", "Sized", "Container",
)


def detect_py312_issues(py_files: list[str], repo_path: str) -> list[Hazard]:
    removed_re = re.compile(
        r"^\s*(?:import|from)\s+("
        + "|".join(re.escape(m) for m in PY312_REMOVED_MODULES)
        + r")\b"
    )
    collections_re = re.compile(
        r"from\s+collections\s+import\s+.*\b("
        + "|".join(PY310_COLLECTIONS_ABCS)
        + r")\b"
    )
    out: list[Hazard] = []
    for fpath in py_files:
        content = read_file(fpath, max_chars=10000)
        if not content:
            continue
        rel = os.path.relpath(fpath, repo_path)
        for i, line in enumerate(content.splitlines(), 1):
            m = removed_re.match(line)
            if m:
                mod = m.group(1)
                out.append(
                    Hazard(
                        kind="py312_removed",
                        file=rel,
                        line=i,
                        description=f"`import {mod}` — removed in Python 3.12",
                        fix=f"Replace with {PY312_REMOVED_MODULES[mod]}",
                    )
                )
            m2 = collections_re.match(line)
            if m2:
                out.append(
                    Hazard(
                        kind="py312_collections_abc",
                        file=rel,
                        line=i,
                        description=f"`from collections import {m2.group(1)}` (use collections.abc)",
                        fix="Change `collections` to `collections.abc` in this import",
                    )
                )
    return out


# ── Old pin hazards ─────────────────────────────────────────────────────────

OLD_PIN_CUTOFFS: dict[str, str] = {
    "transformers": "4.36.0",
    "tokenizers": "0.15.0",
    "scipy": "1.11.0",
    "scikit-learn": "1.3.0",
    "pandas": "2.1.0",
    "numpy": "1.26.0",
    "pillow": "10.0.0",
    "grpcio": "1.58.0",
}


def _version_tuple(s: str) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in s.split("."))
    except ValueError:
        return (0,)


def detect_pin_hazards(config_contents: dict[str, str]) -> list[Hazard]:
    from repo2rocm.recon.configs import parse_requirements

    out: list[Hazard] = []
    for fname, content in config_contents.items():
        if not fname.startswith("requirements"):
            continue
        for pkg, spec in parse_requirements(content):
            key = pkg.lower().replace("_", "-")
            if key not in OLD_PIN_CUTOFFS:
                continue
            m = re.match(r"[=<>~!]*=\s*([0-9]+\.[0-9]+(?:\.[0-9]+)?)", spec)
            if not m:
                continue
            pinned = m.group(1)
            cutoff = OLD_PIN_CUTOFFS[key]
            if _version_tuple(pinned) < _version_tuple(cutoff):
                out.append(
                    Hazard(
                        kind="old_pin",
                        file=fname,
                        line=0,
                        description=f"{pkg}=={pinned} has no Python 3.12 wheel (cutoff {cutoff})",
                        fix=f"Drop the pin and use `pip install {pkg}` (latest)",
                    )
                )
    return out


# ── Code-level hazards ──────────────────────────────────────────────────────

CODE_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(r"wandb\.login\s*\("),
        "wandb_login",
        "wandb.login() fails without API key. Set WANDB_MODE=offline.",
    ),
    (
        re.compile(r"torch\.backends\.cudnn\.\w+\s*="),
        "cudnn_flags",
        "torch.backends.cudnn flags may error on ROCm. Guard with `if not torch.version.hip`.",
    ),
    (
        re.compile(r"nvidia-smi|nvidia_smi"),
        "nvidia_smi",
        "nvidia-smi not available on ROCm. Use rocm-smi.",
    ),
    (
        re.compile(r"torch\.cuda\.amp\."),
        "deprecated_amp",
        "torch.cuda.amp deprecated in PyTorch 2.x. Use torch.amp.autocast('cuda').",
    ),
    (
        re.compile(r"['\"]cuda:0['\"]"),
        "hardcoded_cuda_device",
        "Hardcoded 'cuda:0' — prefer device-agnostic selection.",
    ),
]


def detect_code_hazards(py_files: list[str], repo_path: str) -> list[Hazard]:
    out: list[Hazard] = []
    for fpath in py_files:
        content = read_file(fpath, max_chars=10000)
        if not content:
            continue
        rel = os.path.relpath(fpath, repo_path)
        for line_no, line in enumerate(content.splitlines(), 1):
            for pat, kind, desc in CODE_PATTERNS:
                if pat.search(line):
                    out.append(
                        Hazard(
                            kind=kind,
                            file=rel,
                            line=line_no,
                            description=desc,
                            fix=line.strip()[:140],
                        )
                    )
    return out


# ── Training param scan (used for scale-down hints) ─────────────────────────

_TRAINING_PARAMS = (
    "epochs", "num_epochs", "max_epochs",
    "max_steps", "num_steps", "training_steps",
    "num_iterations", "max_iter",
)


def detect_training_params(py_files: list[str], repo_path: str) -> list[Hazard]:
    pattern = re.compile(
        r"""(?:['"]?(?P<name>"""
        + "|".join(re.escape(p) for p in _TRAINING_PARAMS)
        + r""")['"]?\s*[=:]\s*)(?P<val>\d+)""",
        re.IGNORECASE,
    )
    out: list[Hazard] = []
    seen: set[tuple[str, int]] = set()
    for fpath in py_files:
        content = read_file(fpath, max_chars=8000)
        if not content:
            continue
        rel = os.path.relpath(fpath, repo_path)
        for i, line in enumerate(content.splitlines(), 1):
            m = pattern.search(line)
            if not m:
                continue
            try:
                val = int(m.group("val"))
            except ValueError:
                continue
            if val <= 5 or (rel, i) in seen:
                continue
            seen.add((rel, i))
            out.append(
                Hazard(
                    kind="training_param",
                    file=rel,
                    line=i,
                    description=f"{m.group('name')}={val}",
                    fix=f"Consider scaling down to {min(val, 2)} for smoke tests",
                )
            )
    return out
