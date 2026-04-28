"""
LLM-based Observer Reviewer.

The reviewer replaces the old hardcoded `StateInterpreter` /
`TrajectoryForecaster` / `PreparationPlanner` stack. Its goal is to read
recent turn snapshots the way a human reviewer would and decide:

    1. Is the run healthy or stuck?
    2. If stuck, which skill (from the on-disk catalog) applies?
    3. Should the researcher be invoked for web-grounded evidence?
    4. What is a useful piece of advice we can already articulate?

Pipeline per check-in:

    snapshots ── heuristic_gate() ──► should we even spend an LLM call?
                                       │
                                       ▼
                               llm_review()  (one structured JSON call)
                                       │
                                       ▼
                          materialize() ──► optional research() call
                                       │       (web search + synthesis)
                                       ▼
                                ObserverAdvice row

The reviewer never executes shell commands; it produces advice rows that
the executor reads at the next safe turn boundary.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from observers.skill_loader import (
    SkillCard,
    build_system_prompt,
    load_role_prompt,
    load_reviewer_instructions,
    load_skill_cards,
)
from observers.types import ObserverAdvice


# ── Heuristic gate signals (no LLM, pure text scan) ──────────────────────────
#
# These exist solely to avoid spending money on an LLM call when nothing
# interesting is happening. They are deliberately permissive: they err on
# the side of CALLING the reviewer, because the reviewer's first job is to
# return `intervene=false` cheaply if the run is fine.

_FAILURE_MARKERS = (
    "error:",
    "error :",
    "fatal",
    "traceback",
    "exception",
    "runtimeerror",
    "modulenotfounderror",
    "filenotfounderror",
    "importerror",
    "attributeerror",
    "keyerror",
    "valueerror",
    "typeerror",
    "indexerror",
    "no module named",
    "no such file",
    "undefined",
    "undeclared",
    "subprocess-exited-with-error",
    "failed-wheel-build",
    "could not find a version",
    "permission denied",
    "killed",
    "oom",
    "out of memory",
    "segmentation fault",
    "core dumped",
    "hip error",
    "rocm error",
    "miopen error",
    "cuda error",
    "assertion",
    "abort",
)

_LOOP_HINT_TOKENS = (
    "simple-knn",
    "diff-gaussian-rasterization",
    "flash-attn",
    "bitsandbytes",
    "xformers",
    "triton",
    "deepspeed",
    "verify_paper_result",
)


def _has_failure_marker(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    return any(m in low for m in _FAILURE_MARKERS)


def _extract_failure_lines(text: str, max_lines: int = 30) -> List[str]:
    """Pull out the lines that look like real error/diagnostic lines.

    De-duplicates lines that differ only by line/column number suffixes
    (e.g. `simple_knn.hip:89:15: error: use of undeclared identifier FLT_MAX`
    vs `simple_knn.hip:90:25: error: use of undeclared identifier FLT_MAX`)
    so the LLM sees one canonical exemplar per kind of error plus a count,
    instead of 18 near-duplicate lines.
    """
    if not text:
        return []
    seen: Dict[str, int] = {}
    order: List[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        low = line.lower()
        if not any(m in low for m in _FAILURE_MARKERS):
            continue
        # Canonicalize: strip leading whitespace, replace any `:NN:NN:`
        # line/col suffix with `:N:N:` so duplicates collapse.
        key = re.sub(r":\d+:\d+:", ":N:N:", line.strip())
        # Also collapse pure numeric line refs like `:123:` that appear
        # without a column.
        key = re.sub(r":\d+:", ":N:", key)
        if key in seen:
            seen[key] += 1
            continue
        seen[key] = 1
        order.append(line.strip())
        if len(order) >= max_lines:
            break
    out: List[str] = []
    for k in order:
        canon = re.sub(r":\d+:\d+:", ":N:N:", k)
        canon = re.sub(r":\d+:", ":N:", canon)
        count = seen.get(canon, 1)
        if count > 1:
            out.append(f"{k}    [×{count} similar lines]")
        else:
            out.append(k)
    return out


def _commands_text(snapshot: Dict[str, Any]) -> str:
    return " ".join(str(c) for c in (snapshot.get("commands") or []))


def _observation_text(snapshot: Dict[str, Any]) -> str:
    return str(snapshot.get("observation_excerpt") or "")


# ── Reviewer ─────────────────────────────────────────────────────────────────


@dataclass
class ReviewerConfig:
    llm: str
    max_history: int = 6
    cooldown_s: float = 6.0
    healthy_skip_streak: int = 3   # if N consecutive healthy turns, skip more aggressively
    forced_check_every: int = 5    # always review at least every K turns
    research_budget_s: float = 22.0
    # Per-turn observation budgets (chars). The latest turn gets the
    # largest budget because the freshest signal matters most; older
    # turns get progressively trimmed. Values are upper bounds — if the
    # snapshot already carries less, we use what we have.
    obs_budget_recent: int = 5000
    obs_budget_one_back: int = 3000
    obs_budget_two_back: int = 2000
    obs_budget_older: int = 1200
    rationale_budget_recent: int = 2000
    rationale_budget_older: int = 600
    plan_budget: int = 2400
    # How many lines containing a failure marker to surface at the top of
    # each turn block, before the (possibly long) observation body.
    error_line_focus: int = 30


@dataclass
class ReviewerDecision:
    intervene: bool = False
    skill: str = "progressOK"
    kind: str = "preventive"
    priority: str = "low"
    rationale: str = ""
    predicted_failure: str = ""
    applies_before: str = "next_turn"
    needs_web_search: bool = False
    research_question: str = ""
    fallback_advice: str = ""
    fallback_commands: List[str] = field(default_factory=list)
    severity_signal: str = "fine"

    @classmethod
    def progress_ok(cls) -> "ReviewerDecision":
        return cls(
            intervene=False, skill="progressOK", kind="preventive",
            priority="low", rationale="run progressing healthily",
            severity_signal="fine",
            fallback_advice="Run is progressing; no intervention needed.",
        )


class Reviewer:
    """LLM-driven decision maker for the observer sidecar."""

    def __init__(self, cfg: ReviewerConfig) -> None:
        self.cfg = cfg
        self._role = load_role_prompt()
        self._instructions = load_reviewer_instructions()
        self._skills: List[SkillCard] = load_skill_cards()
        self._system_prompt = build_system_prompt(
            role=self._role,
            instructions=self._instructions,
            skills=self._skills,
        )
        self._skill_names = {card.name for card in self._skills}
        self._last_review_ts: float = 0.0
        self._healthy_streak: int = 0
        self._last_decision_turn: int = -10
        self._last_skill: str = ""

    # ── Heuristic gate ──────────────────────────────────────────────────────

    def should_consider(self, snapshots: List[Dict[str, Any]]) -> bool:
        """Cheap pre-filter to decide whether to call the LLM at all.

        Strategy:
          - Always allow the first 2 turns (so we can catch early landmines).
          - If any of the last 3 turns has a failure marker in the
            observation text or a non-empty error_class, allow.
          - If the same artifact / submodule / package appears in the last
            3+ turns, allow (likely a loop).
          - Otherwise allow once every `forced_check_every` turns to spot
            slow drift.
          - Cooldown: never call the LLM more often than `cooldown_s`.
        """
        if not snapshots:
            return False
        now = time.time()
        if (now - self._last_review_ts) < self.cfg.cooldown_s:
            return False

        last = snapshots[-1]
        turn = int(last.get("turn") or 0)

        # Forced periodic check.
        if turn <= 2:
            return True
        if (turn - self._last_decision_turn) >= self.cfg.forced_check_every:
            return True

        recent = snapshots[-3:]

        # 1. Failure markers in observation text.
        for snap in recent:
            if _has_failure_marker(_observation_text(snap)):
                self._healthy_streak = 0
                return True
            if str(snap.get("error_class") or "").strip():
                self._healthy_streak = 0
                return True

        # 2. Repetition of the same hot token across recent turns.
        if len(recent) >= 3:
            joined = " ".join(_commands_text(s) for s in recent).lower()
            for tok in _LOOP_HINT_TOKENS:
                if joined.count(tok) >= 2:
                    self._healthy_streak = 0
                    return True

        # 3. Same primary command repeating across the last 3 turns.
        primary_cmds: List[str] = []
        for s in recent:
            cmds = s.get("commands") or []
            primary_cmds.append(str(cmds[0]) if cmds else "")
        if (
            len(primary_cmds) >= 3
            and primary_cmds[-1].strip()
            and primary_cmds.count(primary_cmds[-1]) >= 2
        ):
            self._healthy_streak = 0
            return True

        # 4. Long benchmark/verify silence in stage2 without paper retrieval.
        if last.get("stage") == "stage2" and not last.get("paper_retrieval_used"):
            return True

        # Nothing interesting → bump healthy streak, only check sparingly.
        self._healthy_streak += 1
        if self._healthy_streak <= self.cfg.healthy_skip_streak:
            return False
        # After enough healthy turns, do a low-frequency preventive check.
        return (turn % self.cfg.forced_check_every) == 0

    # ── LLM review ──────────────────────────────────────────────────────────

    def decide(self, snapshots: List[Dict[str, Any]],
               run_context: Dict[str, Any]) -> ReviewerDecision:
        if not snapshots:
            return ReviewerDecision.progress_ok()
        try:
            from utils.llm import get_llm_response
        except Exception as e:
            return self._fallback_when_llm_unavailable(snapshots, str(e))

        user_msg = self._build_user_message(snapshots, run_context)
        reply = ""
        llm_error: Optional[str] = None
        try:
            choices, _usage = get_llm_response(
                self.cfg.llm,
                [{"role": "user", "content": user_msg}],
                system_prompt=self._system_prompt,
                temperature=0.1, max_tokens=900,
            )
            # `get_llm_response` returns `(None, None)` when all internal
            # retries fail (auth error, network, etc) — it does NOT raise.
            # Treat that as an LLM-unavailable condition and fall back to
            # the heuristic emergency advice so we never silently swallow
            # the failure.
            if choices is None:
                llm_error = "get_llm_response returned None (provider failed)"
            else:
                reply = (choices[0] if choices else "") or ""
                if not reply.strip():
                    llm_error = "empty reply from LLM"
        except Exception as e:
            llm_error = str(e)
        finally:
            self._last_review_ts = time.time()

        if llm_error is not None:
            return self._fallback_when_llm_unavailable(snapshots, llm_error)

        decision = self._parse_reply(reply)
        if decision.skill not in self._skill_names:
            decision.skill = "progressOK" if not decision.intervene else "explorationStuck"
        return decision

    # ── Materialize advice (optional research call) ─────────────────────────

    def materialize(self, decision: ReviewerDecision,
                    snapshots: List[Dict[str, Any]],
                    run_context: Dict[str, Any]) -> Optional[ObserverAdvice]:
        if not decision.intervene:
            return None

        last = snapshots[-1]
        turn_seen = int(last.get("turn") or 0)

        answer_text = decision.fallback_advice.strip()
        suggestions: List[str] = list(decision.fallback_commands or [])
        evidence: List[str] = []
        confidence = 0.45  # default for prompt-only advice

        if decision.needs_web_search and decision.research_question:
            note = self._call_research(decision, snapshots, run_context)
            if note:
                research_answer = str(note.get("answer") or "").strip()
                if research_answer:
                    answer_text = research_answer
                # merge suggested commands (prepend research's; cap total)
                research_cmds = [str(c) for c in (note.get("suggested_commands") or [])]
                merged = list(research_cmds) + [c for c in suggestions if c not in research_cmds]
                suggestions = merged[:5]
                # citations → evidence
                for c in (note.get("citations") or [])[:4]:
                    if isinstance(c, dict):
                        title = str(c.get("title") or "").strip()
                        url = str(c.get("url") or "").strip()
                        if title or url:
                            evidence.append(f"{title[:100]} {url[:200]}".strip())
                confidence = max(confidence, float(note.get("confidence") or 0.0))

        if not answer_text and not suggestions:
            return None

        priority = decision.priority if decision.priority in ("high", "normal", "low") else "normal"
        kind = decision.kind if decision.kind in ("preventive", "reactive", "corrective") else "reactive"
        expires = turn_seen + (3 if kind == "preventive" else 2)

        self._last_decision_turn = turn_seen
        self._last_skill = decision.skill

        return ObserverAdvice.create(
            turn_seen=turn_seen,
            profile_used=decision.skill,
            diagnosis=(decision.rationale or decision.predicted_failure)[:600],
            recommended_strategy=answer_text[:1600],
            suggested_questions_or_tools=suggestions[:5],
            confidence=confidence,
            evidence=evidence,
            kind=kind,
            predicted_failure=decision.predicted_failure[:320],
            applies_before=(decision.applies_before or last.get("action_family") or "next_turn")[:120],
            expires_after_turn=expires,
            priority=priority,
        )

    # ── Internals ───────────────────────────────────────────────────────────

    def _call_research(self, decision: ReviewerDecision,
                       snapshots: List[Dict[str, Any]],
                       run_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            from agents.researcher import research
        except Exception:
            return None
        ctx = {
            "skill": decision.skill,
            "kind": decision.kind,
            "predicted_failure": decision.predicted_failure,
            "applies_before": decision.applies_before,
            "rationale": decision.rationale,
            "run_context": {
                "repo": run_context.get("repo"),
                "reproduce_results": run_context.get("reproduce_results"),
                "paper_title": run_context.get("paper_title"),
            },
            "recent_turn_digests": [
                self._digest(s) for s in snapshots[-self.cfg.max_history:]
            ],
        }
        try:
            return research(
                decision.research_question,
                llm=self.cfg.llm,
                use_cache=True,
                budget_s=self.cfg.research_budget_s,
                profile="observerCritic",
                context=ctx,
                extra_evidence=[json.dumps({
                    "predicted_failure": decision.predicted_failure,
                    "applies_before": decision.applies_before,
                    "skill": decision.skill,
                })],
                max_search_hits=4,
                max_visits=2,
            )
        except Exception:
            return None

    def _build_user_message(self, snapshots: List[Dict[str, Any]],
                            run_context: Dict[str, Any]) -> str:
        recent = snapshots[-self.cfg.max_history:]
        plan_excerpt = str(run_context.get("plan_excerpt") or "")[: self.cfg.plan_budget]
        repo = str(run_context.get("repo") or "")
        reproduce = bool(run_context.get("reproduce_results"))
        paper_title = str(run_context.get("paper_title") or "")

        parts: List[str] = []
        parts.append("# RUN CONTEXT\n")
        parts.append(f"- repo: {repo}")
        if paper_title:
            parts.append(f"- paper: {paper_title}")
        parts.append(f"- reproduce_results: {reproduce}")
        if plan_excerpt:
            parts.append("- plan excerpt:")
            parts.append("```")
            parts.append(plan_excerpt)
            parts.append("```")
        parts.append("")
        parts.append(
            "# RECENT TURNS (oldest → newest)\n"
            "Trust the **observation text** and the per-command returncodes "
            "in `recent_commands` more than the top-level `return_codes` "
            "field — pipelines like `... 2>&1 | grep ... | head` make the "
            "outer rc=0 even when the underlying build failed.\n"
        )

        # Allocate per-turn budgets: most recent turn gets the most chars,
        # older turns get progressively trimmed.
        budgets = self._per_turn_budgets(len(recent))
        for snap, budget in zip(recent, budgets):
            parts.append(self._format_snapshot_for_llm(
                snap,
                obs_budget=budget["obs"],
                rationale_budget=budget["rationale"],
            ))
            parts.append("")

        parts.append("# YOUR TASK\n")
        parts.append(
            "Read the recent turns the way a senior engineer reads a CI log. "
            "Look for repeated artifacts, cascading errors, and known AMD/HIP "
            "landmines. Emit ONE JSON object that follows the contract from "
            "the system prompt. Most turns should resolve to "
            "`intervene=false` with `skill=progressOK` — only intervene when "
            "you have something concretely useful to add."
        )
        return "\n".join(parts)

    def _per_turn_budgets(self, n: int) -> List[Dict[str, int]]:
        """Decide how many chars each turn block gets, newest turn first."""
        out: List[Dict[str, int]] = []
        for offset in range(n):
            from_end = n - 1 - offset  # 0 = oldest, n-1 = newest
            # i counts from newest backward: 0 = newest, 1 = one back, ...
            i = (n - 1) - from_end
            if i == 0:
                obs = self.cfg.obs_budget_recent
                rat = self.cfg.rationale_budget_recent
            elif i == 1:
                obs = self.cfg.obs_budget_one_back
                rat = self.cfg.rationale_budget_recent
            elif i == 2:
                obs = self.cfg.obs_budget_two_back
                rat = self.cfg.rationale_budget_older
            else:
                obs = self.cfg.obs_budget_older
                rat = self.cfg.rationale_budget_older
            out.append({"obs": obs, "rationale": rat})
        # We built the list newest-first; the caller iterates oldest-first,
        # so reverse to align indices.
        return list(reversed(out))

    def _format_snapshot_for_llm(self, snapshot: Dict[str, Any],
                                  obs_budget: int,
                                  rationale_budget: int) -> str:
        turn = snapshot.get("turn", "?")
        stage = snapshot.get("stage", "?")
        rcs = snapshot.get("return_codes") or []
        err_cls = snapshot.get("error_class") or ""
        duration = snapshot.get("duration_s") or 0
        commands = snapshot.get("commands") or []
        recent_commands = snapshot.get("recent_commands") or []
        observation = str(snapshot.get("observation_excerpt") or "")
        rationale = str(snapshot.get("assistant_response") or "")
        paper_used = bool(snapshot.get("paper_retrieval_used"))
        graphify_used = bool(snapshot.get("graphify_code_lookup_used"))

        # Pull a focused list of unique error/diagnostic lines so the LLM
        # sees the salient signal even before it eye-scans the body.
        failure_lines = _extract_failure_lines(observation, max_lines=self.cfg.error_line_focus)

        # Apply per-turn budgets to the long fields.
        if rationale_budget > 0 and len(rationale) > rationale_budget:
            rationale = rationale[:rationale_budget].rstrip() + "\n…[rationale truncated]"
        if obs_budget > 0 and len(observation) > obs_budget:
            head = observation[: int(obs_budget * 0.7)]
            tail = observation[-int(obs_budget * 0.3):]
            observation = (
                head.rstrip()
                + "\n…[middle of observation truncated]\n"
                + tail.lstrip()
            )

        lines: List[str] = []
        header = (
            f"## Turn {turn} "
            f"(stage={stage}, outer_return_codes={rcs}, "
            f"error_class={err_cls!r}, duration_s={duration})"
        )
        lines.append(header)
        flags: List[str] = []
        if paper_used:
            flags.append("paper_retrieval_used")
        if graphify_used:
            flags.append("graphify_code_lookup_used")
        if flags:
            lines.append(f"_flags_: {', '.join(flags)}")

        if commands:
            lines.append("**Commands this turn:**")
            for c in commands[:5]:
                lines.append(f"  - `{str(c)[:480]}`")

        if recent_commands:
            lines.append("**Recent sandbox command history (real per-command returncodes):**")
            for rc_entry in recent_commands[-8:]:
                if not isinstance(rc_entry, dict):
                    continue
                rc = rc_entry.get("returncode")
                cmd = str(rc_entry.get("command") or "")[:240]
                lines.append(f"  - rc={rc}  `{cmd}`")

        if failure_lines:
            lines.append("**Key error / diagnostic lines (deduped):**")
            for fline in failure_lines:
                lines.append(f"  - {fline[:400]}")

        if rationale.strip():
            lines.append("**Agent's own rationale:**")
            lines.append("```")
            lines.append(rationale)
            lines.append("```")

        lines.append("**Full observation excerpt:**")
        lines.append("```")
        lines.append(observation)
        lines.append("```")
        return "\n".join(lines)

    @staticmethod
    def _digest(snapshot: Dict[str, Any]) -> Dict[str, Any]:
        observation = str(snapshot.get("observation_excerpt") or "")
        # Give the researcher both a head excerpt and the deduped failure
        # lines, so the web evidence pass is grounded in the real symptoms.
        failure_lines = _extract_failure_lines(observation, max_lines=12)
        return {
            "turn": snapshot.get("turn"),
            "stage": snapshot.get("stage"),
            "commands": [str(c)[:240] for c in (snapshot.get("commands") or [])][:5],
            "return_codes": snapshot.get("return_codes") or [],
            "error_class": snapshot.get("error_class") or "",
            "observation_head": observation[:1600],
            "key_error_lines": failure_lines,
            "recent_commands": [
                {
                    "returncode": rc.get("returncode"),
                    "command": str(rc.get("command") or "")[:200],
                }
                for rc in (snapshot.get("recent_commands") or [])[-6:]
                if isinstance(rc, dict)
            ],
        }

    def _parse_reply(self, reply: str) -> ReviewerDecision:
        if not reply:
            return ReviewerDecision.progress_ok()
        text = reply.strip()
        # Strip a single layer of fences if present.
        if text.startswith("```"):
            nl = text.find("\n")
            if nl != -1:
                body = text[nl + 1:]
                if body.endswith("```"):
                    body = body[:-3]
                text = body.strip()
        # Try a strict parse; fall back to a regex-pulled JSON block.
        data: Optional[Dict[str, Any]] = None
        try:
            data = json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, re.S)
            if match:
                try:
                    data = json.loads(match.group(0))
                except Exception:
                    data = None
        if not isinstance(data, dict):
            return ReviewerDecision.progress_ok()

        decision = ReviewerDecision()
        decision.intervene = bool(data.get("intervene"))
        decision.skill = str(data.get("skill") or "progressOK").strip()
        decision.kind = str(data.get("kind") or "preventive").strip().lower()
        decision.priority = str(data.get("priority") or "normal").strip().lower()
        decision.rationale = str(data.get("rationale") or "").strip()
        decision.predicted_failure = str(data.get("predicted_failure") or "").strip()
        decision.applies_before = str(data.get("applies_before") or "next_turn").strip()
        decision.needs_web_search = bool(data.get("needs_web_search"))
        decision.research_question = str(data.get("research_question") or "").strip()
        decision.fallback_advice = str(data.get("fallback_advice") or "").strip()
        cmds = data.get("fallback_commands") or []
        if isinstance(cmds, list):
            decision.fallback_commands = [str(c).strip() for c in cmds if str(c).strip()][:4]
        decision.severity_signal = str(data.get("severity_signal") or "").strip()
        return decision

    def _fallback_when_llm_unavailable(self, snapshots: List[Dict[str, Any]],
                                       err: str) -> ReviewerDecision:
        """When the LLM is unreachable, derive the best advice we can from
        the snapshots alone.

        The strategy:
          1. Scan the last few turns for repeating artifacts (a submodule
             path, a wheel name, a header file) — that tells us the kind
             of loop the executor is in.
          2. Pick the matching skill from a small built-in lookup so the
             advice is at least topically correct (e.g. dependencyRepair
             for repeated submodule build failures).
          3. Emit concrete strategic guidance keyed off the most common
             AMD/ROCm landmines so the executor has something to act on.

        This path is deliberately conservative: we mark everything
        `priority="normal"` and `confidence` is left to the materializer.
        The richer reasoning is still the LLM's job; this is the
        graceful-degradation lane.
        """
        if not snapshots:
            return ReviewerDecision.progress_ok()
        last = snapshots[-1]
        recent = snapshots[-min(6, len(snapshots)):]

        # No errors anywhere in recent history → genuinely healthy.
        any_marker = any(
            _has_failure_marker(_observation_text(s)) or s.get("error_class")
            for s in recent
        )
        if not any_marker:
            return ReviewerDecision.progress_ok()

        # Identify recurring artifacts (submodule paths, package names,
        # header files) to pick the right skill.
        joined_obs = "\n".join(_observation_text(s) for s in recent).lower()
        joined_cmds = " ".join(_commands_text(s) for s in recent).lower()
        joined = joined_obs + "\n" + joined_cmds

        skill = "explorationStuck"
        predicted = "reviewer_llm_unavailable"
        advice = (
            "The observer LLM is currently unavailable. Recent observation "
            "text contains failure markers, so before retrying the same "
            "action consider gathering AMD ROCm evidence with `web_search` "
            "or `pypi_versions` first."
        )
        suggested_cmds: List[str] = []

        # Rule 1: cascading HIP/CUDA build failures in the same submodule.
        hip_loop_markers = (
            "subprocess-exited-with-error",
            ".hip:",
            "device_launch_parameters.h",
            "cooperative_groups/reduce.h",
            "use of undeclared identifier",
            "hipcc",
            "ninja",
        )
        same_submodule = re.findall(r"/repo/submodules/([\w\-\.]+)", joined)
        if (
            sum(1 for m in hip_loop_markers if m in joined) >= 2
            or (same_submodule and len(set(same_submodule)) == 1
                and joined.count(same_submodule[0]) >= 3)
        ):
            skill = "dependencyRepair"
            predicted = "hip_build_loop"
            target = same_submodule[0] if same_submodule else "the failing submodule"
            advice = (
                f"Repeated HIP build failures detected in `{target}` across "
                "recent turns. Stop hand-patching individual headers/symbols "
                "(`device_launch_parameters.h`, `cooperative_groups/reduce.h`, "
                "`__trap`, `FLT_MAX`) one at a time. For Gaussian-Splatting "
                "style submodules (simple-knn, diff-gaussian-rasterization, "
                "langsplat-rasterization), AMD publishes pre-ported wheels — "
                "try installing those first, or apply the full known set of "
                "HIP fixes in one pass and rebuild once."
            )
            suggested_cmds = [
                "pip install gsplat --extra-index-url=https://download.pytorch.org/whl/rocm6.2",
                "pip index versions amd-gsplat 2>&1 | head -5",
            ]

        # Rule 2: ImportError / ModuleNotFoundError after install.
        elif "modulenotfounderror" in joined or "no module named" in joined or "importerror" in joined:
            skill = "frameworkApiDrift"
            predicted = "import_failure_post_install"
            advice = (
                "Recent ImportError / ModuleNotFoundError after a successful "
                "install usually means the install went into a different "
                "site-packages than `python` resolves, or the extension "
                "module (`_C*.so`) is missing. Verify with "
                "`python -c \"import sys; print(sys.path)\"` and check "
                "`pip show <pkg>` to see the install location, then either "
                "reinstall in editable mode or set PYTHONPATH to the "
                "submodule directory."
            )
            suggested_cmds = [
                "python -c 'import sys; print(\"\\n\".join(sys.path))'",
            ]

        # Rule 3: paper reproduction / verifier loop (stage2).
        elif last.get("stage") == "stage2" and not last.get("paper_retrieval_used"):
            skill = "paperReproduction"
            predicted = "stage2_without_paper_retrieval"
            advice = (
                "Stage-2 verifier is running without paper retrieval being "
                "used. Pull the paper-side metric/script with "
                "`paper_retrieve` before claiming reproduction; otherwise "
                "the comparison is ungrounded."
            )

        return ReviewerDecision(
            intervene=True,
            skill=skill if skill in self._skill_names else "explorationStuck",
            kind="reactive",
            priority="normal",
            rationale=(
                f"LLM reviewer unavailable ({err[:160]}); using heuristic "
                f"fallback (skill={skill})."
            ),
            predicted_failure=predicted,
            applies_before="next_turn",
            needs_web_search=False,
            research_question="",
            fallback_advice=advice,
            fallback_commands=suggested_cmds,
            severity_signal="degraded",
        )
