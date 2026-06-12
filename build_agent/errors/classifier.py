"""
Error Classification Engine — classifies terminal output into known
error patterns before the LLM sees it.

When a known error with a high-confidence fix is matched, the fix can be
applied deterministically (no LLM call needed).  Novel errors get sent to
the LLM with structured classification context.

The classifier is seeded from rocm_knowledge.py patterns and continuously
updated by the learning pipeline.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from storage.models import ErrorPattern, Fix, ErrorSeverity
from storage.kb_store import KBStore


@dataclass
class ClassifiedError:
    """Result of error classification."""
    error_class: str = ""
    confidence: float = 0.0
    pattern_id: Optional[str] = None
    known_fixes: List[Fix] = field(default_factory=list)
    is_novel: bool = True
    severity: str = ErrorSeverity.ERROR.value
    matched_text: str = ""
    description: str = ""
    deterministic_fix_available: bool = False
    best_fix: Optional[Fix] = None


class ErrorClassifier:
    """
    Classifies terminal output against known error patterns.

    Maintains an in-memory cache of compiled regex patterns for speed.
    Falls back to KB queries for patterns not in cache.
    """

    DETERMINISTIC_THRESHOLD = 0.85

    def __init__(self, kb: KBStore):
        self.kb = kb
        self._compiled_patterns: Dict[str, Tuple[re.Pattern, ErrorPattern]] = {}
        self._reload_patterns()

    def _reload_patterns(self):
        """Load all error patterns from KB and compile their regexes."""
        patterns = self.kb.get_all_error_patterns()
        self._compiled_patterns.clear()
        for p in patterns:
            if p.regex_pattern:
                try:
                    compiled = re.compile(p.regex_pattern, re.IGNORECASE | re.MULTILINE)
                    self._compiled_patterns[p.id] = (compiled, p)
                except re.error:
                    continue

    def classify(self, output: str, return_code: Optional[int] = None) -> ClassifiedError:
        """
        Classify terminal output into a known error pattern or mark as novel.

        Returns a ClassifiedError with:
          - Known error: error_class, confidence, known fixes, and potentially
            a deterministic fix ready for automatic application.
          - Novel error: is_novel=True with extracted error text for LLM.
        """
        if not output:
            return ClassifiedError()

        best_match: Optional[ClassifiedError] = None
        best_confidence = 0.0

        for pattern_id, (compiled, pattern) in self._compiled_patterns.items():
            match = compiled.search(output)
            if match:
                confidence = pattern.confidence
                if confidence > best_confidence:
                    fixes = self.kb.get_fixes_for_error(pattern_id)
                    high_confidence_fixes = [
                        f for f in fixes
                        if f.success_rate >= self.DETERMINISTIC_THRESHOLD
                        and f.evidence_count >= 3
                    ]
                    best_fix = high_confidence_fixes[0] if high_confidence_fixes else None
                    best_match = ClassifiedError(
                        error_class=pattern.error_class,
                        confidence=confidence,
                        pattern_id=pattern_id,
                        known_fixes=fixes,
                        is_novel=False,
                        severity=pattern.severity,
                        matched_text=match.group(0)[:500],
                        description=pattern.description,
                        deterministic_fix_available=bool(best_fix),
                        best_fix=best_fix,
                    )
                    best_confidence = confidence

        if best_match:
            self.kb.update_error_evidence(best_match.pattern_id)
            return best_match

        error_text = _extract_error_text(output)
        return ClassifiedError(
            error_class="unknown",
            confidence=0.0,
            is_novel=True,
            matched_text=error_text,
            severity=_infer_severity(output, return_code),
        )

    def classify_and_fix(self, output: str,
                         return_code: Optional[int] = None
                         ) -> Tuple[ClassifiedError, Optional[List[str]]]:
        """
        Classify and return (classification, fix_commands).

        If a deterministic fix is available, returns the fix commands directly.
        Otherwise returns (classification, None) and the caller should
        invoke the LLM.
        """
        classified = self.classify(output, return_code)
        if classified.deterministic_fix_available and classified.best_fix:
            return classified, classified.best_fix.commands
        return classified, None

    def add_pattern_from_output(self, output: str, error_class: str,
                                description: str, regex: str,
                                fix_commands: Optional[List[str]] = None,
                                source_attempt: str = "") -> str:
        """Register a new error pattern discovered during a build."""
        import hashlib
        sig = hashlib.sha256(regex.encode()).hexdigest()[:16]
        pattern = ErrorPattern(
            signature=sig,
            error_class=error_class,
            description=description,
            regex_pattern=regex,
            evidence_count=1,
            confidence=0.3,
        )
        pattern_id = self.kb.add_error_pattern(pattern, source_attempt)

        if fix_commands:
            fix = Fix(
                description=f"Fix for {error_class}",
                commands=fix_commands,
                success_rate=1.0,
                evidence_count=1,
            )
            fix_id = self.kb.add_fix(fix, source_attempt)
            self.kb.link_error_to_fix(pattern_id, fix_id)

        self._reload_patterns()
        return pattern_id

    def refresh(self):
        """Reload patterns from KB (call after learning pipeline updates)."""
        self._reload_patterns()


def _extract_error_text(output: str, max_lines: int = 30) -> str:
    """Extract the most relevant error portion from terminal output."""
    lines = output.splitlines()
    error_indicators = [
        "error:", "Error:", "ERROR:", "fatal:", "FATAL:",
        "Traceback", "ModuleNotFoundError", "ImportError",
        "FileNotFoundError", "RuntimeError", "OSError",
        "CalledProcessError", "CompileError", "LinkError",
        "hipErrorNotInitialized", "hipErrorNoBinaryForGpu",
        "undefined symbol", "cannot find", "No matching distribution",
        "Could not find a version", "conflict",
    ]

    error_start = -1
    for i, line in enumerate(lines):
        if any(indicator in line for indicator in error_indicators):
            error_start = max(0, i - 2)
            break

    if error_start == -1:
        return "\n".join(lines[-max_lines:]) if len(lines) > max_lines else output

    return "\n".join(lines[error_start:error_start + max_lines])


def _infer_severity(output: str, return_code: Optional[int]) -> str:
    if return_code is not None and return_code != 0:
        if "Traceback" in output or "FATAL" in output:
            return ErrorSeverity.FATAL.value
        return ErrorSeverity.ERROR.value
    if "warning" in output.lower():
        return ErrorSeverity.WARNING.value
    return ErrorSeverity.INFO.value
