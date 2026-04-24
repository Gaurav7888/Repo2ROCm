"""
Learning Pipeline — distils build trajectories into structured KB updates.

After every build attempt, the distiller should keep only structured knowledge
that is likely to transfer across repositories. Repo-specific prose, one-off
commands, and free-form rules learned from a single run are intentionally
excluded because they tend to overfit.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from storage.models import (
    TrajectoryRecord, BuildAttempt, BuildOutcome,
    ErrorPattern, Fix, Rule, RuleSource,
    KBUpdateProposal, KBUpdateType,
)
from storage.trajectory_store import TrajectoryStore
from storage.kb_store import KBStore
from utils.llm import get_llm_response


class TrajectoryDistiller:
    """
    Post-attempt analysis: extracts structured intelligence from
    build trajectories and proposes KB updates.
    """

    def __init__(self, kb: KBStore, trajectory_store: TrajectoryStore,
                 llm: Optional[str] = None):
        self.kb = kb
        self.trajectory_store = trajectory_store
        self.llm = llm

    def distill(self, attempt: BuildAttempt,
                trajectory: List[TrajectoryRecord]) -> List[KBUpdateProposal]:
        """
        Run full distillation on a completed build attempt.

        Returns a list of KB update proposals (not yet applied).
        """
        proposals: List[KBUpdateProposal] = []

        # Keep durable learning intentionally narrow:
        #   * package install paths / compatibility hints
        #   * confidence updates for existing structured rules
        # Avoid storing one-off regexes or free-form "generalized" rules mined
        # from a single repo trajectory.
        proposals.extend(self._extract_install_paths(attempt, trajectory))

        proposals.extend(self._update_rule_confidence(attempt, trajectory))

        return proposals

    def distill_and_apply(self, attempt: BuildAttempt,
                          trajectory: List[TrajectoryRecord]) -> int:
        """Distill and apply all non-conflicting proposals. Returns count applied."""
        proposals = self.distill(attempt, trajectory)
        applied = 0
        for proposal in proposals:
            if self.kb.apply_update(proposal):
                applied += 1
        return applied

    def _extract_novel_errors(self, attempt: BuildAttempt,
                              trajectory: List[TrajectoryRecord]
                              ) -> List[KBUpdateProposal]:
        """Find errors in the trajectory that were classified as novel."""
        proposals = []
        seen_classes = set()

        for record in trajectory:
            if not record.novel_situation or not record.error_class:
                continue
            if record.error_class in seen_classes or record.error_class == "unknown":
                continue
            seen_classes.add(record.error_class)

            error_text = record.observation_raw[:500] if record.observation_raw else ""
            regex = _generate_regex_from_error(error_text)
            if not regex:
                continue

            pattern = ErrorPattern(
                signature=record.error_class.lower(),
                error_class=record.error_class,
                description=f"Auto-discovered from {attempt.repo_id}",
                regex_pattern=regex,
                evidence_count=1,
                confidence=0.3,
            )
            proposals.append(KBUpdateProposal(
                update_type=KBUpdateType.ADD_ERROR_PATTERN.value,
                payload=pattern.to_dict(),
                source_attempt_id=attempt.id,
                confidence=0.3,
            ))

        return proposals

    def _extract_fix_sequences(self, attempt: BuildAttempt,
                               trajectory: List[TrajectoryRecord]
                               ) -> List[KBUpdateProposal]:
        """
        From successful builds, extract the minimal command sequence
        that was necessary (causal attribution).
        """
        proposals = []

        error_then_fix = []
        for i, record in enumerate(trajectory):
            if record.error_class and record.error_class != "unknown":
                for j in range(i + 1, min(i + 5, len(trajectory))):
                    if trajectory[j].return_code == 0 and trajectory[j].action_type == "bash":
                        error_then_fix.append({
                            "error_class": record.error_class,
                            "error_text": record.observation_raw[:200] if record.observation_raw else "",
                            "fix_command": trajectory[j].action_content.strip(),
                        })
                        break

        for etf in error_then_fix:
            existing = self.kb.find_error_by_class(etf["error_class"])
            if existing:
                fix = Fix(
                    description=f"Fix discovered from {attempt.repo_id}",
                    commands=[etf["fix_command"]],
                    success_rate=1.0,
                    evidence_count=1,
                )
                for pattern in existing:
                    proposals.append(KBUpdateProposal(
                        update_type=KBUpdateType.ADD_FACT.value,
                        payload={
                            "error_id": pattern.id,
                            "fix": fix.to_dict(),
                        },
                        source_attempt_id=attempt.id,
                        confidence=0.5,
                    ))

        return proposals

    def _extract_install_paths(self, attempt: BuildAttempt,
                               trajectory: List[TrajectoryRecord]
                               ) -> List[KBUpdateProposal]:
        """Extract package install methods that succeeded."""
        proposals = []

        for record in trajectory:
            if record.action_type != "bash" or record.return_code != 0:
                continue
            cmd = record.action_content.strip()
            pkg_info = _parse_install_command(cmd)
            if not pkg_info:
                continue

            proposals.append(KBUpdateProposal(
                update_type=KBUpdateType.ADD_INSTALL_PATH.value,
                payload={
                    "package": pkg_info["package"],
                    "rocm_version": attempt.rocm_version or "unknown",
                    "compatible": True,
                    "install_method": pkg_info["method"],
                    "install_commands": [cmd],
                    "notes": f"Discovered in {attempt.repo_id}",
                },
                source_attempt_id=attempt.id,
                confidence=0.6,
            ))

        return proposals

    def _update_rule_confidence(self, attempt: BuildAttempt,
                                trajectory: List[TrajectoryRecord]
                                ) -> List[KBUpdateProposal]:
        """Adjust confidence of rules that were applied during this attempt."""
        proposals = []
        success = attempt.outcome == BuildOutcome.SUCCESS.value

        applied_rule_ids = set()
        for record in trajectory:
            for rule_id in record.kb_rules_applied:
                applied_rule_ids.add(rule_id)

        for rule_id in applied_rule_ids:
            rule = self.kb.get_rule(rule_id)
            if not rule:
                continue
            rule.record_application(success)
            proposals.append(KBUpdateProposal(
                update_type=KBUpdateType.UPDATE_CONFIDENCE.value,
                target_id=rule_id,
                payload={"confidence": rule.confidence},
                source_attempt_id=attempt.id,
                confidence=rule.confidence,
            ))

        return proposals

    def _llm_generalise(self, attempt: BuildAttempt,
                        trajectory: List[TrajectoryRecord]
                        ) -> List[KBUpdateProposal]:
        """Use LLM to extract generalised patterns from the trajectory."""
        proposals = []

        summary_lines = []
        for r in trajectory[-30:]:
            status = "OK" if r.return_code == 0 else f"FAIL(rc={r.return_code})"
            error_info = f" [{r.error_class}]" if r.error_class else ""
            summary_lines.append(
                f"Turn {r.turn_number} [{r.action_type}] {status}{error_info}: "
                f"{r.action_content[:100]}"
            )

        trajectory_summary = "\n".join(summary_lines)
        outcome = attempt.outcome

        prompt = f"""\
Analyze this ROCm build trajectory and extract reusable patterns.

Repository: {attempt.repo_id}
Outcome: {outcome}
Docker image: {attempt.docker_image}

Trajectory (last 30 actions):
{trajectory_summary}

Extract up to 3 reusable patterns as a JSON array. Each pattern should be:
- Abstract (not specific to this repo — reusable across ANY repo with similar errors)
- Actionable (include condition that can be matched, and concrete fix commands)
- Generalised (use package categories, not specific repo names)

CRITICAL RULES for fix_commands:
- Commands MUST be real, executable bash commands. NO placeholders like <package> or <subdir>.
- Use actual command patterns: "pip install --no-build-isolation PACKAGE_NAME" is BAD (placeholder).
  Instead, describe the pattern in the condition and use concrete commands in the fix.
- If a command needs a variable, use shell variables: "$PACKAGE" not "<package>".
- Comments (lines starting with #) are NOT commands — do not include them.

Format:
[
  {{
    "pattern_name": "short_snake_case_name",
    "condition": {{"error_class": "...", "rocm_mode": true, "trigger": "when X happens"}},
    "fix_commands": ["concrete_command_1", "concrete_command_2"],
    "confidence": 0.5,
    "reasoning": "why this is reusable across repos"
  }}
]

Return ONLY the JSON array. Return [] if no reusable patterns found."""

        try:
            messages = [{"role": "user", "content": prompt}]
            response, usage = get_llm_response(
                self.llm, messages, temperature=0.2, max_tokens=1024
            )
            if response and response[0]:
                text = response[0].strip()
                if text.startswith("```"):
                    text = re.sub(r"^```\w*\n?", "", text)
                    text = re.sub(r"\n?```$", "", text)
                patterns = json.loads(text)
                if isinstance(patterns, list):
                    for p in patterns[:3]:
                        if not isinstance(p, dict):
                            continue
                        fix_cmds = p.get("fix_commands", [])
                        if not fix_cmds:
                            continue
                        if _has_template_placeholders(fix_cmds):
                            continue
                        rule = Rule(
                            id=f"rule_learned_{p.get('pattern_name', 'unknown')}_{int(time.time())}",
                            condition=p.get("condition", {}),
                            action=[{"type": "bash", "command": c}
                                    for c in fix_cmds
                                    if c.strip() and not c.strip().startswith("#")],
                            confidence=min(p.get("confidence", 0.3), 0.5),
                            source=RuleSource.LEARNED.value,
                            created_from_attempts=[attempt.id],
                            evidence_count=1,
                        )
                        if not rule.action:
                            continue
                        proposals.append(KBUpdateProposal(
                            update_type=KBUpdateType.SUPERSEDE_RULE.value,
                            payload={"new_rule": rule.to_dict()},
                            source_attempt_id=attempt.id,
                            confidence=rule.confidence,
                        ))
        except Exception:
            pass

        return proposals


# ── helpers ──────────────────────────────────────────────────────────────────

def _generate_regex_from_error(error_text: str) -> Optional[str]:
    """Generate a generalised regex from a specific error message."""
    if not error_text or len(error_text) < 10:
        return None

    for line in error_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if any(keyword in line.lower() for keyword in
               ("error", "fatal", "modulenotfounderror", "importerror",
                "runtimeerror", "no module named", "cannot find")):
            escaped = re.escape(line[:120])
            escaped = re.sub(r"(?:\\d)+", r"\\d+", escaped)
            escaped = re.sub(r"(?:\\/[\\w.]+)+", r"[\\w/.]+", escaped)
            escaped = re.sub(r"\\ +", r"\\s+", escaped)
            try:
                re.compile(escaped)
            except re.error:
                continue
            return escaped

    return None


def _is_install_command(cmd: str) -> bool:
    install_patterns = [
        r"pip\s+install", r"apt-get\s+install", r"conda\s+install",
        r"pip3\s+install", r"python\s+setup\.py\s+install",
        r"poetry\s+install", r"pip\s+install\s+-e",
    ]
    return any(re.search(p, cmd) for p in install_patterns)


def _parse_install_command(cmd: str) -> Optional[Dict[str, str]]:
    """Extract package name, version, and method from an install command."""
    pip_match = re.search(
        r"pip3?\s+install\s+(?:(?:-[a-zA-Z]+\s+)*)"
        r"([a-zA-Z0-9_\-\.]+(?:[=<>!]+[a-zA-Z0-9_\-\.]+)?)",
        cmd
    )
    if pip_match:
        pkg_spec = pip_match.group(1)
        # skip flags that look like packages (e.g., "-q")
        if pkg_spec.startswith("-"):
            return None
        name = re.split(r"[=<>!]", pkg_spec)[0]
        version = pkg_spec[len(name):].lstrip("=<>!") if len(pkg_spec) > len(name) else ""
        if not name or name in (".", "-e", "-r", "--"):
            return None
        return {"package": name, "version": version, "method": "pip"}

    apt_match = re.search(
        r"apt-get\s+install\s+(?:(?:-\S+\s+)*)([a-zA-Z0-9_\-\.]+)",
        cmd
    )
    if apt_match:
        name = apt_match.group(1)
        if name.startswith("-"):
            return None
        return {"package": name, "version": "", "method": "apt"}

    return None


def _has_template_placeholders(commands: List[str]) -> bool:
    """Reject commands that contain template placeholders like <package>, <subdir>, PACKAGE_NAME."""
    placeholder_patterns = [
        r"<[a-zA-Z_]+>",
        r"\b[A-Z_]{3,}_NAME\b",
        r"\bPACKAGE\b",
        r"/tmp/<",
        r"/repo/<",
    ]
    combined = " ".join(str(c) for c in commands)
    return any(re.search(p, combined) for p in placeholder_patterns)
