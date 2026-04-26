"""
Async observer sidecar and file-bus client.

The sidecar watches structured turn snapshots and emits compact advisory notes
that the main configuration loop can choose to inject at safe turn boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_AGENT_ROOT = os.path.dirname(CURRENT_DIR)
if BUILD_AGENT_ROOT not in sys.path:
    sys.path.insert(0, BUILD_AGENT_ROOT)

from observers.types import (  # noqa: E402
    ObserverAdvice,
    ObserverEvent,
    append_jsonl,
    read_jsonl_from_offset,
)


@dataclass(frozen=True)
class ObserverSkill:
    name: str
    description: str


_SKILLS = [
    ObserverSkill(
        "progressOK",
        "Recent turns show healthy progress or useful evidence gathering. Usually no advice.",
    ),
    ObserverSkill(
        "repoExplorationStuck",
        "The run is circling without converging on a runnable target or verified repo surface.",
    ),
    ObserverSkill(
        "dependencyRepair",
        "Package installation, version pinning, or build dependency handling is wasting turns.",
    ),
    ObserverSkill(
        "rocmRuntime",
        "The run is blocked on AMD/ROCm runtime, device verification, HIP, or image assumptions.",
    ),
    ObserverSkill(
        "paperReproduction",
        "Stage 2 paper reproduction is drifting on metric choice, experiment mapping, or verifier discipline.",
    ),
]

def _skills_text() -> str:
    return "\n".join(f"- {skill.name}: {skill.description}" for skill in _SKILLS)


def _decision_fingerprint(decision: Dict[str, Any]) -> str:
    base = {
        "profile_used": decision.get("profile_used"),
        "diagnosis": decision.get("diagnosis"),
        "recommended_strategy": decision.get("recommended_strategy"),
        "suggested_questions_or_tools": decision.get("suggested_questions_or_tools"),
    }
    return hashlib.sha1(json.dumps(base, sort_keys=True).encode("utf-8")).hexdigest()[:20]


class ObserverSidecar:
    def __init__(self, events_path: str, advice_path: str, llm: str,
                 poll_interval_s: float = 1.0, max_history: int = 6) -> None:
        self.events_path = events_path
        self.advice_path = advice_path
        self.llm = llm
        self.poll_interval_s = max(0.2, poll_interval_s)
        self.max_history = max(2, max_history)
        self._events_offset = 0
        self._recent_turns: List[Dict[str, Any]] = []
        self._run_context: Dict[str, Any] = {}
        self._done = False
        self._last_advice_fp = ""
        self._last_advice_turn = -1

    def _build_prompt(self) -> str:
        payload = {
            "run_context": self._run_context,
            "recent_turns": self._recent_turns[-self.max_history:],
        }
        return (
            "Assess the current run state from the structured snapshots below.\n"
            "Emit advice only if there is a meaningful risk of wasted turns or an "
            "incorrect trajectory.\n\n"
            f"SNAPSHOTS:\n{json.dumps(payload, indent=2, default=str)}\n"
        )

    def _should_research(self) -> bool:
        recent = self._recent_turns[-min(len(self._recent_turns), 3):]
        if len(recent) < 2:
            return False
        issue_score = 0
        error_classes = [
            str(turn.get("error_class") or "").strip()
            for turn in recent
            if str(turn.get("error_class") or "").strip()
        ]
        if len(error_classes) >= 2 and len(set(error_classes[-2:])) == 1:
            issue_score += 2
        commands = [
            tuple(turn.get("commands") or [])
            for turn in recent
            if turn.get("commands")
        ]
        if len(commands) >= 2 and len(set(commands[-2:])) == 1:
            issue_score += 1
        for turn in recent:
            codes = [rc for rc in (turn.get("return_codes") or []) if rc is not None]
            if any(int(rc) != 0 for rc in codes):
                issue_score += 1
            if turn.get("action_type") == "none":
                issue_score += 1
        last = recent[-1]
        if last.get("stage") == "stage2" and not last.get("paper_retrieval_used"):
            issue_score += 1
        return issue_score >= 2

    def _derive_skill(self) -> str:
        recent = self._recent_turns[-min(len(self._recent_turns), 3):]
        if not recent:
            return "progressOK"
        last = recent[-1]
        if last.get("stage") == "stage2":
            return "paperReproduction"
        error_class = str(last.get("error_class") or "").lower()
        text = " ".join(
            [
                error_class,
                str(last.get("observation_excerpt") or "").lower(),
                " ".join(str(cmd).lower() for cmd in (last.get("commands") or [])),
            ]
        )
        if any(token in text for token in ("rocm", "hip", "rocblas", "miopen", "gfx", "kfd", "amd")):
            return "rocmRuntime"
        if any(token in text for token in ("pip", "install", "version", "wheel", "dependency")):
            return "dependencyRepair"
        if len(recent) >= 2:
            return "repoExplorationStuck"
        return "progressOK"

    def _build_research_question(self, skill_name: str) -> str:
        if skill_name == "paperReproduction":
            return (
                "Diagnose the current paper reproduction run. What external evidence or "
                "paper-specific clarification would most improve the next turn without "
                "changing the executor into a web-browsing agent?"
            )
        if skill_name == "rocmRuntime":
            return (
                "Diagnose the current AMD ROCm runtime issue and recommend the next "
                "high-level corrective strategy based on external evidence."
            )
        if skill_name == "dependencyRepair":
            return (
                "Diagnose the current dependency/version repair loop and recommend a "
                "better next strategy grounded in external package or compatibility evidence."
            )
        if skill_name == "repoExplorationStuck":
            return (
                "The run is not converging. What external evidence or reframing would "
                "most help the next turn focus on the right repo surface or execution path?"
            )
        return (
            "Assess whether the run needs any external evidence before the next turn, "
            "and if so, what strategic guidance should be injected?"
        )

    def _build_research_context(self) -> Dict[str, Any]:
        return {
            "run_context": self._run_context,
            "recent_turns": self._recent_turns[-self.max_history:],
            "skills": [skill.name for skill in _SKILLS],
        }

    @staticmethod
    def _note_to_advice(note: Dict[str, Any], turn_seen: int,
                        skill_name: str) -> Optional[ObserverAdvice]:
        answer = str(note.get("answer") or "").strip()
        followups = [str(item).strip() for item in (note.get("followups") or []) if str(item).strip()]
        if not answer and not followups:
            return None
        lowered = answer.lower()
        if "progress is healthy" in lowered or "continue" in lowered and float(note.get("confidence", 0.0) or 0.0) < 0.3:
            return None
        diagnosis = answer.split(".")[0].strip() if answer else f"Observer skill {skill_name} found a likely issue."
        suggestions = followups[:4]
        for command in (note.get("suggested_commands") or [])[:3]:
            cmd = str(command).strip()
            if cmd and cmd not in suggestions:
                suggestions.append(cmd)
        evidence = []
        for citation in (note.get("citations") or [])[:3]:
            if isinstance(citation, dict):
                title = str(citation.get("title") or "").strip()
                url = str(citation.get("url") or "").strip()
                if title or url:
                    evidence.append(f"{title[:100]} {url[:180]}".strip())
        return ObserverAdvice.create(
            turn_seen=turn_seen,
            profile_used=skill_name,
            diagnosis=diagnosis[:220],
            recommended_strategy=answer[:1200],
            suggested_questions_or_tools=suggestions,
            confidence=float(note.get("confidence", 0.0) or 0.0),
            evidence=evidence,
        )

    def _evaluate_recent_turns(self, turn_seen: int) -> Optional[ObserverAdvice]:
        if len(self._recent_turns) < 2 or not self.llm:
            return None
        if not self._should_research():
            return None
        skill_name = self._derive_skill()
        try:
            from agents.researcher import research
        except Exception:
            return None
        note = research(
            self._build_research_question(skill_name),
            llm=self.llm,
            use_cache=True,
            budget_s=25.0,
            profile="observerCritic",
            context=self._build_research_context(),
            extra_evidence=[self._build_prompt()],
            max_search_hits=4,
            max_visits=2,
        )
        advice = self._note_to_advice(note, turn_seen=turn_seen, skill_name=skill_name)
        if advice is None:
            return None
        decision = {
            "profile_used": advice.profile_used,
            "diagnosis": advice.diagnosis,
            "recommended_strategy": advice.recommended_strategy,
            "suggested_questions_or_tools": advice.suggested_questions_or_tools,
        }
        fingerprint = _decision_fingerprint(decision)
        if fingerprint == self._last_advice_fp and abs(turn_seen - self._last_advice_turn) <= 1:
            return None
        self._last_advice_fp = fingerprint
        self._last_advice_turn = turn_seen
        return advice

    def _handle_event(self, row: Dict[str, Any]) -> None:
        event_type = str(row.get("event_type") or "")
        payload = row.get("payload") or {}
        if event_type == "run_started":
            self._run_context = dict(payload or {})
            return
        if event_type == "turn_snapshot":
            self._recent_turns.append(dict(payload or {}))
            self._recent_turns = self._recent_turns[-self.max_history:]
            advice = self._evaluate_recent_turns(int(payload.get("turn", 0) or 0))
            if advice is not None:
                append_jsonl(self.advice_path, advice)
            return
        if event_type == "run_finished":
            self._done = True

    def run(self) -> None:
        while not self._done:
            rows, self._events_offset = read_jsonl_from_offset(
                self.events_path,
                self._events_offset,
            )
            if not rows:
                time.sleep(self.poll_interval_s)
                continue
            for row in rows:
                self._handle_event(row)
                if self._done:
                    break


class ObserverClient:
    def __init__(self, output_dir: str, llm: str,
                 api_key: str = "", enabled: bool = True) -> None:
        self.output_dir = output_dir
        self.llm = llm
        self.api_key = api_key or ""
        self.enabled = bool(enabled and llm)
        self.events_path = os.path.join(output_dir, "observer_events.jsonl")
        self.advice_path = os.path.join(output_dir, "observer_advice.jsonl")
        self.log_path = os.path.join(output_dir, "observer_sidecar.log")
        self._advice_offset = 0
        self._consumed_ids = set()
        self._proc: Optional[subprocess.Popen] = None
        self._log_handle = None

    def start(self) -> None:
        if not self.enabled:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        open(self.events_path, "w", encoding="utf-8").close()
        open(self.advice_path, "w", encoding="utf-8").close()
        self._log_handle = open(self.log_path, "a", encoding="utf-8")
        env = os.environ.copy()
        if self.api_key and not env.get("AMD_LLM_API_KEY"):
            env["AMD_LLM_API_KEY"] = self.api_key
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--events", self.events_path,
            "--advice", self.advice_path,
            "--llm", self.llm,
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=self._log_handle,
            stderr=self._log_handle,
            cwd=BUILD_AGENT_ROOT,
            env=env,
        )

    def emit_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        append_jsonl(self.events_path, ObserverEvent.create(event_type, payload))

    def consume_new_advice(self) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        rows, self._advice_offset = read_jsonl_from_offset(
            self.advice_path,
            self._advice_offset,
        )
        fresh: List[Dict[str, Any]] = []
        for row in rows:
            advice_id = str(row.get("advice_id") or "")
            if not advice_id or advice_id in self._consumed_ids:
                continue
            self._consumed_ids.add(advice_id)
            fresh.append(row)
        return fresh

    def shutdown(self) -> None:
        if self.enabled:
            self.emit_event("run_finished", {})
        if self._proc is not None:
            try:
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.terminate()
            self._proc = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repo2ROCm observer sidecar")
    parser.add_argument("--events", required=True)
    parser.add_argument("--advice", required=True)
    parser.add_argument("--llm", required=True)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--max-history", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    sidecar = ObserverSidecar(
        events_path=args.events,
        advice_path=args.advice,
        llm=args.llm,
        poll_interval_s=args.poll_interval,
        max_history=args.max_history,
    )
    sidecar.run()


if __name__ == "__main__":
    main()
