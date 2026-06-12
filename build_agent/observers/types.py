"""
Typed observer event, semantic state, hazard, forecast, and readiness records.

The observer sidecar is intentionally proactive:
  - it summarizes each turn into a `TurnState`,
  - keeps a `HazardLedger` of static risks discovered before execution starts,
  - emits `TrajectoryForecast` predictions about likely next failures, and
  - delivers `ReadinessPack` advice to the main loop before failures land.

Records are JSON-serialized through a single file bus so the main loop never
needs in-process coupling to the observer.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional, Tuple


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def append_jsonl(path: str, payload: Any) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(_to_jsonable(payload), default=str) + "\n")


def read_jsonl_from_offset(path: str, offset: int = 0) -> Tuple[List[Dict[str, Any]], int]:
    if not path or not os.path.exists(path):
        return [], offset
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        handle.seek(offset)
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        new_offset = handle.tell()
    return rows, new_offset


@dataclass
class ObserverEvent:
    event_id: str
    event_type: str
    timestamp: float
    payload: Dict[str, Any]

    @classmethod
    def create(cls, event_type: str, payload: Dict[str, Any]) -> "ObserverEvent":
        return cls(
            event_id=uuid.uuid4().hex[:16],
            event_type=event_type,
            timestamp=time.time(),
            payload=payload,
        )


@dataclass
class TurnState:
    """Semantic summary of one execution turn.

    Cheap to compute, derived from the structured `turn_snapshot` event so the
    observer reasons over typed fields rather than scraping observation text.
    """

    turn: int
    stage: str
    action_family: str
    action_target: str
    succeeded: bool
    return_codes: List[int] = field(default_factory=list)
    error_class: str = ""
    duration_s: float = 0.0
    repo_areas_touched: List[str] = field(default_factory=list)
    dependency_signals: List[str] = field(default_factory=list)
    paper_signals: List[str] = field(default_factory=list)
    runtime_signals: List[str] = field(default_factory=list)
    blocked_on: str = ""
    used_local_retrieval: bool = False
    paper_retrieval_used: bool = False
    notes: str = ""

    @classmethod
    def empty(cls, turn: int, stage: str = "stage1") -> "TurnState":
        return cls(turn=int(turn), stage=str(stage or "stage1"),
                   action_family="none", action_target="", succeeded=False)


@dataclass
class HazardSignal:
    """A single forward-looking risk detected from static run context."""

    hazard_id: str
    skill: str
    title: str
    description: str
    triggers_when: str
    suggested_prep: str
    confidence: float = 0.5
    evidence_refs: List[str] = field(default_factory=list)

    @classmethod
    def create(cls, *, skill: str, title: str, description: str,
               triggers_when: str, suggested_prep: str,
               confidence: float = 0.5,
               evidence_refs: Optional[List[str]] = None) -> "HazardSignal":
        return cls(
            hazard_id=uuid.uuid4().hex[:12],
            skill=str(skill or "observerCritic"),
            title=str(title or "")[:160],
            description=str(description or "")[:1200],
            triggers_when=str(triggers_when or "")[:320],
            suggested_prep=str(suggested_prep or "")[:1200],
            confidence=float(confidence or 0.0),
            evidence_refs=list(evidence_refs or []),
        )


@dataclass
class HazardLedger:
    """Collection of static hazards built once at run start."""

    created_at: float = field(default_factory=time.time)
    repo_id: str = ""
    plan_excerpt: str = ""
    hazards: List[HazardSignal] = field(default_factory=list)

    def add(self, hazard: HazardSignal) -> None:
        self.hazards.append(hazard)


@dataclass
class TrajectoryForecast:
    """Short-horizon prediction of what the main loop is about to do next."""

    forecast_id: str
    created_at: float
    turn_seen: int
    predicted_next_action_family: str
    predicted_next_target: str
    predicted_failures: List[str] = field(default_factory=list)
    failure_probability: float = 0.0
    failure_cost: str = "medium"
    horizon: str = "short"
    notes: str = ""

    @classmethod
    def create(cls, *, turn_seen: int, predicted_next_action_family: str,
               predicted_next_target: str,
               predicted_failures: Optional[List[str]] = None,
               failure_probability: float = 0.0,
               failure_cost: str = "medium",
               horizon: str = "short",
               notes: str = "") -> "TrajectoryForecast":
        return cls(
            forecast_id=uuid.uuid4().hex[:12],
            created_at=time.time(),
            turn_seen=int(turn_seen),
            predicted_next_action_family=str(predicted_next_action_family or "unknown"),
            predicted_next_target=str(predicted_next_target or "")[:240],
            predicted_failures=list(predicted_failures or []),
            failure_probability=float(failure_probability or 0.0),
            failure_cost=str(failure_cost or "medium"),
            horizon=str(horizon or "short"),
            notes=str(notes or "")[:600],
        )


@dataclass
class ObserverAdvice:
    """Backwards-compatible advice envelope.

    `kind` distinguishes preventive readiness packs from reactive fixes.
    `predicted_failure`, `applies_before`, `expires_after_turn`, and `priority`
    let the main loop schedule advice intelligently.
    """

    advice_id: str
    created_at: float
    turn_seen: int
    profile_used: str
    diagnosis: str
    recommended_strategy: str
    suggested_questions_or_tools: List[str] = field(default_factory=list)
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    kind: str = "reactive"  # "preventive" | "reactive" | "corrective"
    predicted_failure: str = ""
    applies_before: str = ""  # action family the advice prepares for
    expires_after_turn: int = -1
    priority: str = "normal"  # "high" | "normal" | "low"
    forecast_id: str = ""
    hazard_id: str = ""

    @classmethod
    def create(
        cls,
        turn_seen: int,
        profile_used: str,
        diagnosis: str,
        recommended_strategy: str,
        suggested_questions_or_tools: Optional[List[str]] = None,
        confidence: float = 0.0,
        evidence: Optional[List[str]] = None,
        kind: str = "reactive",
        predicted_failure: str = "",
        applies_before: str = "",
        expires_after_turn: int = -1,
        priority: str = "normal",
        forecast_id: str = "",
        hazard_id: str = "",
    ) -> "ObserverAdvice":
        return cls(
            advice_id=uuid.uuid4().hex[:16],
            created_at=time.time(),
            turn_seen=int(turn_seen),
            profile_used=str(profile_used or "observerCritic"),
            diagnosis=str(diagnosis or "")[:600],
            recommended_strategy=str(recommended_strategy or "")[:1600],
            suggested_questions_or_tools=list(suggested_questions_or_tools or []),
            confidence=float(confidence or 0.0),
            evidence=list(evidence or []),
            kind=str(kind or "reactive"),
            predicted_failure=str(predicted_failure or "")[:320],
            applies_before=str(applies_before or "")[:120],
            expires_after_turn=int(expires_after_turn) if expires_after_turn is not None else -1,
            priority=str(priority or "normal"),
            forecast_id=str(forecast_id or ""),
            hazard_id=str(hazard_id or ""),
        )
