"""
Typed observer event and advice records for the async sidecar.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, List, Tuple


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
class ObserverAdvice:
    advice_id: str
    created_at: float
    turn_seen: int
    profile_used: str
    diagnosis: str
    recommended_strategy: str
    suggested_questions_or_tools: List[str] = field(default_factory=list)
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        turn_seen: int,
        profile_used: str,
        diagnosis: str,
        recommended_strategy: str,
        suggested_questions_or_tools: List[str] | None = None,
        confidence: float = 0.0,
        evidence: List[str] | None = None,
    ) -> "ObserverAdvice":
        return cls(
            advice_id=uuid.uuid4().hex[:16],
            created_at=time.time(),
            turn_seen=int(turn_seen),
            profile_used=str(profile_used or "observerCritic"),
            diagnosis=str(diagnosis or ""),
            recommended_strategy=str(recommended_strategy or ""),
            suggested_questions_or_tools=list(suggested_questions_or_tools or []),
            confidence=float(confidence or 0.0),
            evidence=list(evidence or []),
        )
