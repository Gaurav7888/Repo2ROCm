"""Import scanning + framework detection."""
from __future__ import annotations

import re

from repo2rocm.recon.files import read_file

_IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_]*)")

_FRAMEWORK_PRIORITY: list[tuple[list[str], str]] = [
    (["vllm"], "vllm"),
    (["sglang"], "sglang"),
    (["jax", "flax"], "jax"),
    (["tensorflow", "keras"], "tensorflow"),
    (["onnxruntime", "onnx"], "onnxruntime"),
    (["torch", "pytorch"], "pytorch"),
]


def extract_imports(py_files: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for fpath in py_files:
        content = read_file(fpath, max_chars=8000)
        if not content:
            continue
        seen: set[str] = set()
        for line in content.splitlines():
            m = _IMPORT_RE.match(line)
            if not m:
                continue
            pkg = m.group(1)
            if pkg in seen:
                continue
            seen.add(pkg)
            counts[pkg] = counts.get(pkg, 0) + 1
    return counts


def detect_framework(import_counts: dict[str, int]) -> str:
    for pkgs, label in _FRAMEWORK_PRIORITY:
        for pkg in pkgs:
            if pkg in import_counts:
                return label
    return "unknown"
