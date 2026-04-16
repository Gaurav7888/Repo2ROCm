"""
Rule Engine — matches the current build context against executable rules
and returns applicable actions.

Rules are structured, versioned objects with condition/action/confidence.
The engine queries the KB for matching rules, ranks them by confidence,
and returns either deterministic commands (high confidence) or guidance
text (lower confidence) for the LLM.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

from storage.models import Rule, Fix
from storage.kb_store import KBStore


class RuleEngine:
    """
    Matches build context against KB rules and returns actions.

    The engine distinguishes between:
    - Deterministic rules (confidence >= threshold, evidence >= min):
      Applied automatically without LLM involvement.
    - Advisory rules (lower confidence): Injected as guidance into
      the LLM's context so it can decide whether to apply them.
    """

    DETERMINISTIC_THRESHOLD = 0.85
    ADVISORY_THRESHOLD = 0.4
    MIN_EVIDENCE_DETERMINISTIC = 3

    def __init__(self, kb: KBStore):
        self.kb = kb
        self._rules_cache: List[Rule] = []
        self._cache_time: float = 0
        self._cache_ttl: float = 60.0
        self.reload()

    def reload(self):
        """Refresh the in-memory rule cache from KB."""
        self._rules_cache = self.kb.get_active_rules()
        self._cache_time = time.time()

    def _ensure_cache(self):
        if time.time() - self._cache_time > self._cache_ttl:
            self.reload()

    def match(self, context: Dict[str, Any]) -> RuleMatchResult:
        """
        Match the current build context against all active rules.

        Args:
            context: Dict describing the current state. Keys include:
                - package_needed: str (package being installed)
                - package_matches: checked via regex
                - rocm_mode: bool
                - error_pattern: str (current error output)
                - code_contains: str (pattern found in source)
                - error_class: str (classified error)
                - gpu_arch: str

        Returns:
            RuleMatchResult with deterministic actions and advisory guidance.
        """
        self._ensure_cache()

        deterministic: List[Tuple[Rule, List[Dict[str, Any]]]] = []
        advisory: List[Tuple[Rule, List[Dict[str, Any]]]] = []

        for rule in self._rules_cache:
            if rule.deprecated:
                continue
            if self._rule_matches(rule, context):
                if (rule.confidence >= self.DETERMINISTIC_THRESHOLD
                        and rule.evidence_count >= self.MIN_EVIDENCE_DETERMINISTIC):
                    deterministic.append((rule, rule.action))
                elif rule.confidence >= self.ADVISORY_THRESHOLD:
                    advisory.append((rule, rule.action))

        deterministic.sort(key=lambda x: -x[0].confidence)
        advisory.sort(key=lambda x: -x[0].confidence)

        return RuleMatchResult(
            deterministic_rules=deterministic,
            advisory_rules=advisory,
        )

    def _rule_matches(self, rule: Rule, context: Dict[str, Any]) -> bool:
        """Check whether a rule's condition matches the context."""
        condition = rule.condition

        for key, expected in condition.items():
            actual = context.get(key)

            if key == "rocm_mode":
                if actual != expected:
                    return False

            elif key == "package_needed":
                if actual is None:
                    return False
                if isinstance(expected, str):
                    actual_norm = str(actual).lower().replace("-", "_")
                    expected_norm = expected.lower().replace("-", "_")
                    if actual_norm != expected_norm:
                        return False

            elif key == "package_matches":
                if actual is None:
                    return False
                try:
                    if not re.search(expected, str(actual), re.IGNORECASE):
                        return False
                except re.error:
                    return False

            elif key == "error_pattern":
                error_text = context.get("error_output", "") or context.get("error_pattern", "")
                if not error_text:
                    return False
                try:
                    if not re.search(expected, str(error_text), re.IGNORECASE):
                        return False
                except re.error:
                    return False

            elif key == "code_contains":
                code_text = context.get("code_content", "") or context.get("code_contains", "")
                if not code_text:
                    return False
                if expected not in code_text:
                    return False

            elif key == "error_class":
                if actual != expected:
                    return False

            elif key == "gpu_arch":
                if actual and expected:
                    if isinstance(expected, list):
                        if actual not in expected:
                            return False
                    elif actual != expected:
                        return False

        return True

    def record_outcome(self, rule_id: str, success: bool):
        """Record whether applying a rule led to success."""
        self.kb.update_rule_outcome(rule_id, success)

    def extract_commands(self, actions: List[Dict[str, Any]]) -> List[str]:
        """Extract bash commands from a rule's action list."""
        commands = []
        for action in actions:
            atype = action.get("type", "")
            if atype == "bash":
                cmd = action.get("command", "")
                if cmd:
                    commands.append(cmd)
            elif atype == "env":
                key = action.get("key", "")
                value = action.get("value", "")
                if key:
                    commands.append(f"export {key}={value}")
        return commands

    def extract_guidance(self, actions: List[Dict[str, Any]]) -> str:
        """Extract human-readable guidance from a rule's action list."""
        parts = []
        for action in actions:
            atype = action.get("type", "")
            if atype == "guidance":
                parts.append(action.get("text", ""))
            elif atype == "skip_pip":
                parts.append(f"Skip pip install of {action.get('package', '?')} — use alternative")
            elif atype == "skip_install":
                parts.append(f"Skip: {action.get('reason', '')}")
        return "\n".join(parts)

    def format_for_prompt(self, result: RuleMatchResult) -> str:
        """Format matching rules as guidance text for the LLM prompt."""
        if not result.advisory_rules and not result.deterministic_rules:
            return ""

        lines = []
        if result.deterministic_rules:
            lines.append("=== DETERMINISTIC ACTIONS (auto-applied) ===")
            for rule, actions in result.deterministic_rules:
                cmds = self.extract_commands(actions)
                guidance = self.extract_guidance(actions)
                lines.append(f"Rule: {rule.id} (confidence: {rule.confidence:.2f})")
                if cmds:
                    lines.append(f"  Commands: {' && '.join(cmds)}")
                if guidance:
                    lines.append(f"  Note: {guidance}")

        if result.advisory_rules:
            lines.append("\n=== ADVISORY (consider these) ===")
            for rule, actions in result.advisory_rules:
                guidance = self.extract_guidance(actions)
                cmds = self.extract_commands(actions)
                lines.append(f"Rule: {rule.id} (confidence: {rule.confidence:.2f})")
                if guidance:
                    lines.append(f"  Suggestion: {guidance}")
                if cmds:
                    lines.append(f"  Possible commands: {' && '.join(cmds)}")

        return "\n".join(lines)


class RuleMatchResult:
    """Container for rule matching results."""

    def __init__(
        self,
        deterministic_rules: Optional[List[Tuple[Rule, List[Dict[str, Any]]]]] = None,
        advisory_rules: Optional[List[Tuple[Rule, List[Dict[str, Any]]]]] = None,
    ):
        self.deterministic_rules = deterministic_rules or []
        self.advisory_rules = advisory_rules or []

    @property
    def has_deterministic(self) -> bool:
        return len(self.deterministic_rules) > 0

    @property
    def has_advisory(self) -> bool:
        return len(self.advisory_rules) > 0

    @property
    def all_deterministic_commands(self) -> List[str]:
        """Flatten all deterministic rule commands."""
        cmds = []
        for rule, actions in self.deterministic_rules:
            for action in actions:
                if action.get("type") == "bash":
                    cmd = action.get("command", "")
                    if cmd:
                        cmds.append(cmd)
                elif action.get("type") == "env":
                    key = action.get("key", "")
                    value = action.get("value", "")
                    if key:
                        cmds.append(f"export {key}={value}")
        return cmds

    @property
    def rule_ids(self) -> List[str]:
        return [r.id for r, _ in self.deterministic_rules + self.advisory_rules]
