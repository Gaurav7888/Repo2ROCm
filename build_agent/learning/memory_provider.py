"""
Memory Provider — BEGIN/IN phased retrieval adapted from MemEvolve.

Provides the configuration agent with contextual intelligence at two phases:

BEGIN phase (at planner time):
  - Queries KB for this repo's dependency fingerprint
  - Retrieves: known compatible images, install paths, predicted failures,
    estimated build time, and applicable rules
  - Returns structured context that replaces or augments rocm_knowledge.py

IN phase (at each configuration agent turn):
  - When the agent encounters an error, queries KB for similar errors
  - Returns deterministic fix commands (high confidence) or guidance text
  - Injects relevant rules into the agent's context

Key difference from MemEvolve: we return executable rules and fix commands,
not natural language memories.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from storage.models import (
    BuildFingerprint, MemoryRequest, MemoryResponse, MemoryItem, MemoryPhase,
    Rule, Fix,
    CausalState, CausalTransition,
)
from storage.kb_store import KBStore
from storage.trajectory_store import TrajectoryStore
from errors.classifier import ErrorClassifier, ClassifiedError
from rules.engine import RuleEngine


class BuildMemoryProvider:
    """
    Phased memory retrieval for build sessions.

    Maintains per-session state (which rules have been applied, which errors
    have been seen) and adapts retrieval accordingly.
    """

    def __init__(self, kb: KBStore, trajectory_store: TrajectoryStore,
                 error_classifier: ErrorClassifier, rule_engine: RuleEngine):
        self.kb = kb
        self.trajectory_store = trajectory_store
        self.error_classifier = error_classifier
        self.rule_engine = rule_engine

        self._session_rules_applied: List[str] = []
        self._session_errors_seen: List[str] = []
        self._session_fixes_applied: List[str] = []
        self._turn_count: int = 0

    def reset_session(self):
        self._session_rules_applied.clear()
        self._session_errors_seen.clear()
        self._session_fixes_applied.clear()
        self._turn_count = 0

    def provide_memory(self, request: MemoryRequest) -> MemoryResponse:
        """
        Main entry point for memory retrieval.

        Dispatches to BEGIN or IN phase handler based on request.phase.
        """
        if request.phase == MemoryPhase.BEGIN.value:
            return self._provide_begin(request)
        else:
            return self._provide_in(request)

    def _provide_begin(self, request: MemoryRequest) -> MemoryResponse:
        """
        BEGIN phase: comprehensive pre-build intelligence.

        Retrieves:
        - Similar successful builds for the fingerprint
        - Package compatibility data
        - Applicable rules for the repo's characteristics
        - Estimated difficulty and build time
        """
        items: List[MemoryItem] = []
        deterministic_fixes: List[Fix] = []

        fp = request.fingerprint

        # 1. Find similar successful builds
        if fp:
            similar = self.trajectory_store.get_successful_attempts_by_fingerprint(
                fp.signature(), limit=3
            )
            if similar:
                for attempt in similar:
                    items.append(MemoryItem(
                        id=f"similar_build_{attempt.id}",
                        content=(
                            f"Similar repo {attempt.repo_id} succeeded with "
                            f"image={attempt.docker_image}, "
                            f"turns={attempt.total_turns}, "
                            f"duration={attempt.duration_minutes:.1f}min"
                        ),
                        item_type="guidance",
                        confidence=0.7,
                    ))

        # 2. Query applicable rules for repo characteristics
        context = request.context or {}
        context["rocm_mode"] = True
        match_result = self.rule_engine.match(context)

        for rule, actions in match_result.deterministic_rules:
            cmds = self.rule_engine.extract_commands(actions)
            items.append(MemoryItem(
                id=f"rule_{rule.id}",
                content=self.rule_engine.extract_guidance(actions),
                item_type="rule",
                confidence=rule.confidence,
                source_rule_id=rule.id,
                executable=True,
                commands=cmds,
            ))

        for rule, actions in match_result.advisory_rules:
            items.append(MemoryItem(
                id=f"advisory_{rule.id}",
                content=self.rule_engine.extract_guidance(actions),
                item_type="guidance",
                confidence=rule.confidence,
                source_rule_id=rule.id,
            ))

        # 3. Package compatibility data
        if fp:
            for dep in fp.cuda_deps:
                compat = self.kb.get_compatibility(dep)
                if compat:
                    best = max(compat, key=lambda c: c["confidence"])
                    items.append(MemoryItem(
                        id=f"compat_{dep}",
                        content=(
                            f"{dep}: {'compatible' if best['compatible'] else 'INCOMPATIBLE'} "
                            f"with ROCm {best['rocm_version']} "
                            f"(method: {best['install_method']}, "
                            f"confidence: {best['confidence']:.2f})"
                        ),
                        item_type="pattern",
                        confidence=best["confidence"],
                        executable=bool(best.get("install_commands")),
                        commands=best.get("install_commands", []),
                    ))

        # 4. Estimate difficulty
        stats = self.trajectory_store.get_stats()
        guidance_parts = []
        if stats["total_attempts"] > 0:
            success_rate = stats["successful_attempts"] / stats["total_attempts"]
            guidance_parts.append(
                f"Global success rate: {success_rate:.0%} across "
                f"{stats['unique_repos']} repos"
            )

        guidance_text = "\n".join(guidance_parts)

        # 5. Causal migration memory (state→action→outcome priors).
        #    These run in every phase, including --mode env, so the agent
        #    sees the relevant transitions at the start of the loop even
        #    before a failure has occurred.
        try:
            causal_items = self._provide_causal(request)
            items.extend(causal_items)
        except Exception:
            pass

        overall_confidence = (
            sum(i.confidence for i in items) / len(items) if items else 0.0
        )

        return MemoryResponse(
            items=items,
            deterministic_fixes=deterministic_fixes,
            guidance_text=guidance_text,
            confidence=overall_confidence,
        )

    def _provide_in(self, request: MemoryRequest) -> MemoryResponse:
        """
        IN phase: per-turn contextual intelligence.

        When an error occurs, classifies it and returns fixes.
        Otherwise returns relevant advisory rules for the current action.
        """
        self._turn_count += 1
        items: List[MemoryItem] = []
        deterministic_fixes: List[Fix] = []

        # 1. If there's a current error, classify and retrieve fixes
        if request.current_error:
            classified = self.error_classifier.classify(
                request.current_error
            )

            if not classified.is_novel:
                self._session_errors_seen.append(classified.error_class)

                if classified.deterministic_fix_available and classified.best_fix:
                    deterministic_fixes.append(classified.best_fix)
                    items.append(MemoryItem(
                        id=f"fix_{classified.best_fix.id}",
                        content=(
                            f"DETERMINISTIC FIX for {classified.error_class}: "
                            f"{classified.best_fix.description}"
                        ),
                        item_type="fix",
                        confidence=classified.confidence,
                        source_fix_id=classified.best_fix.id,
                        executable=True,
                        commands=classified.best_fix.commands,
                    ))
                else:
                    for fix in classified.known_fixes[:3]:
                        items.append(MemoryItem(
                            id=f"fix_option_{fix.id}",
                            content=(
                                f"Possible fix for {classified.error_class} "
                                f"(success rate: {fix.success_rate:.0%}): "
                                f"{fix.description}"
                            ),
                            item_type="fix",
                            confidence=fix.success_rate,
                            source_fix_id=fix.id,
                            executable=True,
                            commands=fix.commands,
                        ))
            else:
                items.append(MemoryItem(
                    id="novel_error",
                    content=(
                        f"Novel error detected (no KB match). "
                        f"Extracted: {classified.matched_text[:200]}"
                    ),
                    item_type="guidance",
                    confidence=0.0,
                ))

        # 2. Context-based rule matching
        context = request.context or {}
        context["rocm_mode"] = True
        match_result = self.rule_engine.match(context)

        for rule, actions in match_result.advisory_rules:
            if rule.id not in self._session_rules_applied:
                guidance = self.rule_engine.extract_guidance(actions)
                if guidance:
                    items.append(MemoryItem(
                        id=f"in_advisory_{rule.id}",
                        content=guidance,
                        item_type="guidance",
                        confidence=rule.confidence,
                        source_rule_id=rule.id,
                    ))

        # 3. Causal migration memory: per-turn retrieval for the current
        #    error class / fingerprint / image. Always run, including in
        #    --mode env, so the agent gets `[CAUSAL]` guidance before
        #    deciding on the next command.
        try:
            causal_items = self._provide_causal(request)
            items.extend(causal_items)
        except Exception:
            pass

        guidance_text = ""
        if deterministic_fixes:
            guidance_text = (
                "Deterministic fix available — applying automatically "
                "without LLM consultation."
            )

        overall_confidence = (
            sum(i.confidence for i in items) / len(items) if items else 0.0
        )

        return MemoryResponse(
            items=items,
            deterministic_fixes=deterministic_fixes,
            guidance_text=guidance_text,
            confidence=overall_confidence,
        )

    # ── Causal Migration Memory ─────────────────────────────────────────────

    def provide_causal_memory(self, request: MemoryRequest,
                              top_k: int = 3) -> List[MemoryItem]:
        """Public entry point: causal transitions for a given request.

        Returns a list of `MemoryItem`s with `item_type="causal"` whose
        content is the structured `[CAUSAL] ...` line described in the
        research plan.  Empty list when the KB has no matching transitions
        — including when the table is empty, so this is safe to call from
        any phase / any run mode.
        """
        return self._provide_causal(request, top_k=top_k)

    def _provide_causal(self, request: MemoryRequest,
                        top_k: int = 3) -> List[MemoryItem]:
        ctx = request.context or {}
        fp_sig = ""
        fp = request.fingerprint
        if fp is not None:
            try:
                fp_sig = fp.signature()
            except Exception:
                fp_sig = ""

        error_class = (ctx.get("error_class") or "").strip()
        if not error_class and request.current_error:
            try:
                classified = self.error_classifier.classify(request.current_error)
                if classified and not classified.is_novel:
                    error_class = classified.error_class
            except Exception:
                error_class = error_class or ""

        error_signature = ""
        if request.current_error:
            for line in (request.current_error or "").splitlines():
                s = line.strip()
                if not s:
                    continue
                if any(tok in s for tok in (
                    "Error", "error", "fatal", "ModuleNotFoundError",
                    "ImportError", "RuntimeError", "AssertionError",
                    "FileNotFoundError",
                )):
                    error_signature = s[:160]
                    break

        state = CausalState(
            repo_fingerprint=fp_sig,
            image=str(ctx.get("image", "") or ""),
            gpu_arch=str(ctx.get("gpu_arch", "") or ""),
            error_class=error_class,
            error_signature=error_signature,
            degradation_policy=str(
                ctx.get("degradation_policy", "") or ""
            ),
        )

        try:
            transitions = self.kb.query_transitions(state, top_k=top_k)
        except Exception:
            transitions = []

        items: List[MemoryItem] = []
        for t in transitions:
            content = format_causal_transition(t)
            cf_lines = format_causal_counterfactuals(t)
            full_content = content
            if cf_lines:
                full_content = content + "\n" + "\n".join(cf_lines)
            items.append(MemoryItem(
                id=f"causal_{t.id}",
                content=full_content,
                item_type="causal",
                confidence=float(t.outcome.confidence or 0.6),
                executable=bool(t.action.command),
                commands=[t.action.command] if t.action.command else [],
            ))
        return items

    def format_causal_for_prompt(self, response: MemoryResponse) -> str:
        """Prepend `[CAUSAL]` lines for system-prompt / per-turn injection.

        Returns empty string when the response carries no causal items, so
        callers can safely concatenate this into existing prompts.
        """
        causal = [i for i in response.items if i.item_type == "causal"]
        if not causal:
            return ""
        sections = []
        sections.append("=" * 60)
        sections.append("CAUSAL MIGRATION MEMORY (typed state→action→outcome priors)")
        sections.append("=" * 60)
        for item in causal:
            sections.append(item.content)
        sections.append("=" * 60)
        return "\n".join(sections)

    def record_rule_applied(self, rule_id: str, success: bool):
        """Track that a rule was applied this session."""
        self._session_rules_applied.append(rule_id)
        self.rule_engine.record_outcome(rule_id, success)

    def record_fix_applied(self, fix_id: str, error_id: str, success: bool):
        """Track that a fix was applied this session."""
        self._session_fixes_applied.append(fix_id)
        self.kb.record_fix_outcome(fix_id, error_id, success)

    def format_begin_for_prompt(self, response: MemoryResponse) -> str:
        """Format BEGIN-phase memory as text for injection into the system prompt."""
        if not response.items:
            return ""

        sections = []
        sections.append("=" * 60)
        sections.append("KNOWLEDGE BASE INTELLIGENCE (from prior builds)")
        sections.append("=" * 60)

        rules = [i for i in response.items if i.item_type == "rule"]
        if rules:
            sections.append("\nAPPLICABLE RULES:")
            for item in rules:
                sections.append(f"  [{item.confidence:.0%}] {item.content}")
                if item.commands:
                    sections.append(f"    Commands: {' && '.join(item.commands)}")

        patterns = [i for i in response.items if i.item_type == "pattern"]
        if patterns:
            sections.append("\nPACKAGE COMPATIBILITY DATA:")
            for item in patterns:
                sections.append(f"  {item.content}")

        guidance = [i for i in response.items if i.item_type == "guidance"]
        if guidance:
            sections.append("\nPRIOR BUILD INSIGHTS:")
            for item in guidance:
                sections.append(f"  {item.content}")

        causal = [i for i in response.items if i.item_type == "causal"]
        if causal:
            sections.append("\nCAUSAL MIGRATION TRANSITIONS (state→action→outcome):")
            for item in causal:
                sections.append(item.content)

        if response.guidance_text:
            sections.append(f"\n{response.guidance_text}")

        sections.append("=" * 60)
        return "\n".join(sections)

    def format_begin_for_planner(self, response: MemoryResponse,
                                 max_items: int = 6) -> str:
        """
        Produce a short planner-facing brief from BEGIN memory.

        The planner needs only the highest-signal prior, not the full KB dump
        used in the configuration agent's system prompt.
        """
        if not response.items:
            return ""

        priority = {"rule": 0, "pattern": 1, "guidance": 2, "fix": 3}
        ranked = sorted(
            response.items,
            key=lambda item: (
                priority.get(item.item_type, 9),
                -float(item.confidence),
                item.id,
            ),
        )

        lines = []
        seen = set()
        for item in ranked:
            content = (item.content or "").strip()
            if not content:
                continue
            norm = " ".join(content.lower().split())
            if norm in seen:
                continue
            seen.add(norm)

            label = {
                "rule": "rule",
                "pattern": "compat",
                "guidance": "lesson",
                "fix": "fix",
            }.get(item.item_type, "note")
            line = f"  - [{label} {item.confidence:.0%}] {content}"
            if item.commands:
                line += f" | command hint: {item.commands[0]}"
            lines.append(line)
            if len(lines) >= max_items:
                break

        if not lines:
            return ""

        return (
            "\n========================================\n"
            "LEARNED PRIOR FOR PLANNING\n"
            "========================================\n"
            + "\n".join(lines)
            + "\n"
        )

    def format_in_for_observation(self, response: MemoryResponse) -> str:
        """Format IN-phase memory as text to append to the observation."""
        if not response.items:
            return ""

        parts = []
        for item in response.items:
            if item.item_type == "fix" and item.executable:
                parts.append(
                    f"\n** KB SUGGESTION [{item.confidence:.0%}]: {item.content} **"
                )
                if item.commands:
                    parts.append(f"   Try: {' && '.join(item.commands)}")
            elif item.item_type == "guidance":
                parts.append(f"\n[KB] {item.content}")
            elif item.item_type == "causal":
                # Structured `[CAUSAL]` line; already self-formatted.
                parts.append(f"\n{item.content}")

        return "\n".join(parts)


# ── Causal formatting helpers ────────────────────────────────────────────────

def format_causal_transition(t: CausalTransition) -> str:
    """Render a single transition as a structured `[CAUSAL]` guidance line.

    Example output:

        [CAUSAL] state{img=rocm/pytorch, arch=gfx942, err=cuda_only_wheel}
                 → action{FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE python setup.py install}
                 → outcome{ok, D1}
    """
    s = t.state
    a = t.action
    o = t.outcome

    state_bits = []
    if s.image:
        state_bits.append(f"img={s.image}")
    if s.gpu_arch:
        state_bits.append(f"arch={s.gpu_arch}")
    if s.error_class:
        state_bits.append(f"err={s.error_class}")
    if s.degradation_policy:
        state_bits.append(f"policy={s.degradation_policy}")
    if s.repo_fingerprint:
        state_bits.append(f"fp={s.repo_fingerprint[:10]}")
    state_str = ", ".join(state_bits) if state_bits else "(unspecified)"

    action_str = (a.command or a.type or "(no-op)").strip()
    if len(action_str) > 240:
        action_str = action_str[:237] + "..."

    rc = o.return_code
    rc_str = "ok" if rc == 0 else f"rc={rc}"
    outcome_bits = [rc_str]
    if o.degradation:
        outcome_bits.append(o.degradation)
    if o.confidence:
        outcome_bits.append(f"conf={o.confidence:.2f}")
    outcome_str = ", ".join(outcome_bits)

    return (
        f"[CAUSAL] state{{{state_str}}} "
        f"→ action{{{action_str}}} "
        f"→ outcome{{{outcome_str}}}"
    )


def format_causal_counterfactuals(t: CausalTransition) -> List[str]:
    """Render counterfactuals as `[counterfactual: ...]` advisory lines."""
    out: List[str] = []
    for cf in (t.counterfactuals or []):
        if not isinstance(cf, dict):
            continue
        action = str(cf.get("action", "")).strip()
        expected = str(cf.get("expected_outcome", "")).strip()
        reason = str(cf.get("reason", "")).strip()
        if not action and not reason:
            continue
        bits = [action]
        if expected:
            bits.append(f"→ expected {expected}")
        if reason:
            bits.append(f"({reason})")
        out.append("[counterfactual: " + " ".join(bits) + "]")
    return out
