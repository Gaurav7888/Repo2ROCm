"""ErrorClassifier — match runtime errors against seeded + learned patterns."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from repo2rocm.learning.kb_store import ErrorPattern, KBStore


@dataclass
class ClassifiedError:
    error_class: str
    confidence: float
    matched_pattern: str | None = None
    description: str = ""


# Seed patterns — written once, then evolved via the Distiller.
_SEEDS: list[ErrorPattern] = [
    ErrorPattern(
        signature="cuda_arch_mismatch",
        error_class="cuda_arch_mismatch",
        description="A CUDA-arch wheel was loaded on a non-NVIDIA device.",
        regex_pattern=r"(no kernel image|CUDA error.*invalid device function|sm_\d+ not supported)",
        confidence=0.95,
    ),
    ErrorPattern(
        signature="missing_torch_cuda",
        error_class="missing_torch_cuda",
        description="torch installed but cuda backend missing — wrong wheel.",
        regex_pattern=r"AssertionError.*Torch not compiled with CUDA enabled",
        confidence=0.95,
    ),
    ErrorPattern(
        signature="distutils_removed",
        error_class="py312_distutils",
        description="distutils removed in Python 3.12.",
        regex_pattern=r"ModuleNotFoundError.*distutils",
        confidence=0.99,
    ),
    ErrorPattern(
        signature="collections_abc",
        error_class="py310_collections_abc",
        description="collections.Mapping etc. removed in 3.10+.",
        regex_pattern=r"ImportError.*cannot import name '\w+' from 'collections'",
        confidence=0.99,
    ),
    ErrorPattern(
        signature="pip_resolution_conflict",
        error_class="pip_conflict",
        description="pip's dependency resolver could not satisfy constraints.",
        regex_pattern=r"ERROR: ResolutionImpossible|ERROR: Cannot install",
        confidence=0.9,
    ),
    ErrorPattern(
        signature="rocm_smi_not_found",
        error_class="missing_rocm",
        description="rocm-smi missing — container is not a ROCm image.",
        regex_pattern=r"rocm-smi: command not found",
        confidence=0.99,
    ),
    ErrorPattern(
        signature="flash_attn_cuda_only",
        error_class="cuda_only_wheel",
        description="flash-attn PyPI wheel hard-fails on AMD.",
        regex_pattern=r"flash_attn.*CUDA|flash_attn.*nvcc",
        confidence=0.85,
    ),
]


class ErrorClassifier:
    def __init__(self, kb: KBStore | None = None):
        self.kb = kb
        if kb is not None:
            for p in _SEEDS:
                kb.upsert_error_pattern(p)

    def classify(self, text: str) -> ClassifiedError:
        patterns: Iterable[ErrorPattern] = (
            self.kb.list_error_patterns() if self.kb else _SEEDS
        )
        for p in sorted(patterns, key=lambda x: -x.confidence):
            try:
                if re.search(p.regex_pattern, text, re.IGNORECASE):
                    return ClassifiedError(
                        error_class=p.error_class,
                        confidence=p.confidence,
                        matched_pattern=p.regex_pattern,
                        description=p.description,
                    )
            except re.error:
                continue
        return ClassifiedError(error_class="unknown", confidence=0.0)
