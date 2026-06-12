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
from typing import Any, Dict, List, Optional, Set

from storage.models import (
    TrajectoryRecord, BuildAttempt, BuildOutcome,
    ErrorPattern, Fix, Rule, RuleSource,
    KBUpdateProposal, KBUpdateType,
    CausalState, CausalAction, CausalOutcome, CausalTransition,
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
                          trajectory: List[TrajectoryRecord],
                          success_report: Optional[Dict[str, Any]] = None
                          ) -> int:
        """Distill and apply all non-conflicting proposals. Returns count applied.

        Also extracts and persists causal transitions when the run produced
        `ROCM_ENV_VERIFIED` (and contains at least one failure→success pair).
        Causal transitions live in their own table and are *not* funnelled
        through `KBUpdateProposal` because they have a different lifecycle
        (no consistency-check vs existing facts; we keep all evidence).
        """
        proposals = self.distill(attempt, trajectory)
        applied = 0
        for proposal in proposals:
            if self.kb.apply_update(proposal):
                applied += 1

        try:
            transitions = self.extract_causal_transitions(
                trajectory, attempt, success_report=success_report,
            )
            for t in transitions:
                self.kb.insert_transition(t, source_attempt=attempt.id)
                applied += 1
        except Exception:
            # Causal extraction must never break the existing learning
            # pipeline; failures are intentionally swallowed here.
            pass

        return applied

    # ── Causal Migration Memory ─────────────────────────────────────────────

    def extract_causal_transitions(
        self,
        trajectory_records: List[TrajectoryRecord],
        attempt: BuildAttempt,
        success_report: Optional[Dict[str, Any]] = None,
    ) -> List[CausalTransition]:
        """Conservative extractor for `state → action → outcome` transitions.

        Emits a transition only when ALL of the following hold:

        1. The run produced `ROCM_ENV_VERIFIED` (we look at the trajectory
           text for the marker, the attempt outcome, or the success_report
           if one was passed in).  Without that anchor we cannot claim the
           action *causally* unblocked the migration.
        2. There is a failed turn (non-zero return code OR a classified
           error) followed within ≤ 3 turns by a successful turn (return
           code 0) which "resolved" the same error class — meaning the
           error class disappears in the success window, or the success
           turn references the same package/file root.
        3. The successful turn has a concrete bash command we can bind to
           the action.

        Evidence is taken from `kb_rules_applied` plus parsed observations
        on the success turn.  Degradation is read from `success_report`
        flags when provided (`flash_attn_triton_amd_install`, `sdpa_fallback`,
        `base_image_changed`, `scale_down_engaged`, `loose_tolerance_pass`),
        otherwise defaults to `D0`.
        """
        if not trajectory_records:
            return []

        env_verified = _run_was_env_verified(
            trajectory_records, attempt, success_report,
        )
        if not env_verified:
            return []

        degradation = _degradation_from_success_report(success_report)
        repo_fingerprint = ""
        if attempt.fingerprint is not None:
            try:
                repo_fingerprint = attempt.fingerprint.signature()
            except Exception:
                repo_fingerprint = ""
        image = attempt.docker_image or ""
        gpu_arch = (attempt.gpu_arch or "").strip()
        degradation_policy = "strict"  # default for env-mode benchmark

        seen_classes: Set[str] = set()
        transitions: List[CausalTransition] = []

        for i, rec in enumerate(trajectory_records):
            if not _is_failure_turn(rec):
                continue
            err_class = (rec.error_class or "").strip()
            if not err_class or err_class == "unknown":
                continue
            if err_class in seen_classes:
                continue

            # Look up to 3 turns ahead for a success that resolved this
            # error class.
            success_idx = -1
            for j in range(i + 1, min(i + 4, len(trajectory_records))):
                cand = trajectory_records[j]
                if cand.return_code == 0 and cand.action_type == "bash" \
                        and (cand.action_content or "").strip() \
                        and _resolves_error(rec, cand):
                    success_idx = j
                    break
            if success_idx == -1:
                continue

            success = trajectory_records[success_idx]
            seen_classes.add(err_class)

            error_signature = _short_error_signature(rec)
            evidence: List[str] = []
            evidence.extend(
                f"kb_rule:{rid}" for rid in (success.kb_rules_applied or [])
            )
            obs_parsed = success.observation_parsed or {}
            if isinstance(obs_parsed, dict) and obs_parsed:
                for k in ("verification", "summary", "evidence"):
                    if k in obs_parsed and obs_parsed[k]:
                        evidence.append(f"{k}:{str(obs_parsed[k])[:160]}")

            transition_class = _infer_transition_class(rec, success)

            state = CausalState(
                repo_fingerprint=repo_fingerprint,
                image=image,
                gpu_arch=gpu_arch,
                error_class=err_class,
                error_signature=error_signature,
                degradation_policy=degradation_policy,
            )
            action = CausalAction(
                type=_infer_action_type(success),
                command=(success.action_content or "").strip()[:1024],
                evidence=evidence[:8],
            )
            outcome = CausalOutcome(
                return_code=int(success.return_code or 0),
                verification=_verification_from_success(success),
                degradation=degradation,
                confidence=0.6,
            )
            transitions.append(CausalTransition(
                transition_class=transition_class,
                state=state,
                action=action,
                outcome=outcome,
                counterfactuals=[],
                source_attempt_id=attempt.id,
                source="learned",
                evidence_count=1,
            ))

        return transitions

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


# ── Causal-extraction helpers ────────────────────────────────────────────────

# Map degradation flags surfaced by the success report into the typed
# `D0..D3` scale used in the causal record. The flags themselves come from
# `benchmark/harness/scoring/rubric.py:detect_flags`.
_DEGRADATION_FLAG_MAP = {
    "flash_attn_triton_amd_install": "D1",  # functionally equivalent backend
    "sdpa_fallback":                  "D2",  # different attention path
    "base_image_changed":             "D1",  # platform shift, same goal
    "scale_down_engaged":             "D2",  # smaller workload
    "loose_tolerance_pass":           "D3",  # accepted larger metric delta
}


def _degradation_from_success_report(success_report: Optional[Dict[str, Any]]) -> str:
    if not success_report or not isinstance(success_report, dict):
        return "D0"
    flags = success_report.get("degradation_flags") or success_report.get("flags") or []
    if not flags:
        return "D0"
    # Pick the worst (highest D-class) reported flag.
    worst = "D0"
    for f in flags:
        d = _DEGRADATION_FLAG_MAP.get(str(f), "D0")
        if d > worst:   # lexical compare works for D0..D3
            worst = d
    return worst


def _run_was_env_verified(
    trajectory_records: List[TrajectoryRecord],
    attempt: BuildAttempt,
    success_report: Optional[Dict[str, Any]],
) -> bool:
    """Best-effort check for `ROCM_ENV_VERIFIED` in the run."""
    if success_report and isinstance(success_report, dict):
        env = success_report.get("env") or {}
        if isinstance(env, dict) and env.get("stage1_marker_emitted"):
            return True
    if (attempt.outcome or "") == BuildOutcome.SUCCESS.value:
        # The post-run learning pipeline only marks attempts SUCCESS once
        # the dockerfile integration step succeeded — which on env-mode
        # implies ROCM_ENV_VERIFIED was emitted earlier in the loop.
        return True
    needle = "ROCM_ENV_VERIFIED"
    for rec in trajectory_records:
        if rec.action_content and needle in rec.action_content:
            return True
        if rec.observation_raw and needle in rec.observation_raw:
            return True
    return False


def _is_failure_turn(rec: TrajectoryRecord) -> bool:
    if rec.return_code is not None and rec.return_code != 0:
        return True
    if rec.error_class and rec.error_class != "unknown":
        return True
    return False


def _short_error_signature(rec: TrajectoryRecord) -> str:
    """Pick a short, deterministic signature line from the failure turn."""
    raw = (rec.observation_raw or "").strip()
    if not raw:
        return (rec.error_class or "")[:160]
    indicators = (
        "ModuleNotFoundError", "ImportError", "RuntimeError",
        "fatal error", "error:", "Error:", "FileNotFoundError",
        "AssertionError", "hipError", "undefined symbol",
        "No matching distribution",
    )
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        if any(ind in s for ind in indicators):
            return s[:160]
    # Fall back to last non-empty line — usually the most informative.
    for line in reversed(raw.splitlines()):
        s = line.strip()
        if s:
            return s[:160]
    return (rec.error_class or "")[:160]


def _resolves_error(failed: TrajectoryRecord, success: TrajectoryRecord) -> bool:
    """Heuristic: does `success` plausibly resolve `failed`'s error class?

    We accept the pair if either:
      * the success turn has no error class of its own (clean success), OR
      * the success turn's command/observation references the same package
        or file root that appears in the failed turn's error signature.
    """
    if success.return_code != 0:
        return False
    if not success.error_class or success.error_class == "unknown":
        return True

    sig = _short_error_signature(failed).lower()
    cmd = (success.action_content or "").lower()
    obs = (success.observation_raw or "").lower()

    # Pull out the most specific token from the failure (module name,
    # filename, or package).
    tokens = []
    for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_\-\.]{2,}", sig):
        if tok in ("error", "Error", "ERROR", "module", "named", "fatal"):
            continue
        tokens.append(tok.lower())
    if not tokens:
        return True

    return any(t in cmd or t in obs for t in tokens[:6])


def _verification_from_success(success: TrajectoryRecord) -> List[str]:
    out: List[str] = []
    cmd = (success.action_content or "").strip()
    if cmd:
        out.append(f"command: {cmd[:160]}")
    if success.return_code == 0:
        out.append("return_code: 0")
    obs = (success.observation_raw or "").strip().splitlines()
    if obs:
        out.append(f"observation_tail: {obs[-1][:160]}")
    return out


def _infer_action_type(success: TrajectoryRecord) -> str:
    """Best-effort categorisation of the action behind the successful turn."""
    cmd = (success.action_content or "").lower()
    if "change_base_image" in cmd:
        return "image_switch"
    if any(tok in cmd for tok in ("hipify-clang", "hipify-perl", "hipify")):
        return "kernel_fix"
    if "setup.py" in cmd or "build_ext" in cmd:
        return "package_strategy"
    if cmd.startswith(("pip ", "pip3 ")) or " pip install" in cmd:
        return "package_strategy"
    if "git clone" in cmd:
        return "package_strategy"
    if "rocm_env_verified" in cmd or "echo " in cmd:
        return "verdict_emit"
    return "command"


def _infer_transition_class(failed: TrajectoryRecord,
                            success: TrajectoryRecord) -> str:
    """Map an extracted (failure → success) pair to the named classes used
    by the seed transitions.  Falls back to a slug derived from the error
    class when no specific mapping applies."""
    err = (failed.error_class or "").upper()
    cmd = (success.action_content or "").lower()

    if err == "FLASH_ATTN_CUDA_WHEEL" or (
        "flash" in (failed.observation_raw or "").lower()
        and "triton" in cmd
    ):
        return "cuda_only_wheel_to_rocm_source_build"
    if err in ("TORCH_CUDA_NOT_AVAILABLE", "HIPBLAS_NOT_INITIALIZED") and \
            "change_base_image" in cmd:
        return "wrong_image_to_ranked_image_switch"
    if err == "HIPBLAS_NOT_INITIALIZED" and "rocm" in cmd:
        return "missing_gpu_runtime_to_rocm_base_image"
    if err == "SETUPTOOLS_BUILD_FAIL" and "hipify" in cmd:
        return "custom_cuda_compile_error_to_hipify_fix"
    if err == "PAPER_METRIC_MISMATCH":
        return "paper_metric_mismatch_to_not_reproduced"

    slug = re.sub(r"[^A-Za-z0-9]+", "_", err.lower()).strip("_")
    return f"{slug}_to_resolved" if slug else "generic_failure_to_resolved"
