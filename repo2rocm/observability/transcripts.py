"""Append-only JSONL transcripts per (session, agent).

Every message that flows through `core.query.query()` is recorded here so we can:
  * resume a killed sub-agent from disk (the auto-resume pattern)
  * replay a run for debugging
  * train future versions of the system
  * audit what an agent did
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any


def _atomic_append(path: Path, data: str) -> None:
    """O_APPEND on POSIX is atomic for writes <= PIPE_BUF; safe enough for JSONL."""
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    fd = os.open(str(path), flags, 0o644)
    try:
        os.write(fd, data.encode("utf-8"))
    finally:
        os.close(fd)


class Transcript:
    """One transcript file per (session_id, agent_id)."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last_uuid: str | None = None
        self._byte_offset = self.path.stat().st_size if self.path.exists() else 0
        self._record_count = 0

    def append(self, record: dict[str, Any]) -> str:
        """Append one record. Returns the record's UUID for chain continuity."""
        rec_uuid = str(uuid.uuid4())
        envelope = {
            "uuid": rec_uuid,
            "parent_uuid": self._last_uuid,
            "ts": time.time(),
            **record,
        }
        line = json.dumps(envelope, separators=(",", ":"), default=_json_default) + "\n"
        with self._lock:
            _atomic_append(self.path, line)
            self._last_uuid = rec_uuid
            self._byte_offset += len(line)
            self._record_count += 1
        return rec_uuid

    def read_all(self) -> list[dict[str, Any]]:
        """Read the full transcript. O(file size); used for resume."""
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    @property
    def record_count(self) -> int:
        return self._record_count


class TranscriptStore:
    """Owns the transcript directory layout for a session."""

    def __init__(self, root: Path, session_id: str | None = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.root = root / "sessions" / self.session_id
        self.root.mkdir(parents=True, exist_ok=True)
        self._transcripts: dict[str, Transcript] = {}
        self._lock = threading.Lock()

    def transcript(self, agent_id: str) -> Transcript:
        with self._lock:
            if agent_id not in self._transcripts:
                self._transcripts[agent_id] = Transcript(self.root / f"{agent_id}.jsonl")
            return self._transcripts[agent_id]

    def main(self) -> Transcript:
        """The top-level session transcript (used by the Coordinator)."""
        return self.transcript("session")


def _json_default(o: Any) -> Any:
    if hasattr(o, "model_dump"):
        return o.model_dump()
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)
