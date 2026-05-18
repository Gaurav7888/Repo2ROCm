"""Trajectory store — persists every build attempt + its trajectory file."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BuildAttempt:
    id: str = field(default_factory=lambda: f"build_{uuid.uuid4().hex[:12]}")
    repo_id: str = ""
    sha: str = ""
    docker_image: str = ""
    fingerprint: dict | None = None
    started_at: float = field(default_factory=time.time)
    outcome: str = "in_progress"  # success | failure | in_progress
    duration_s: float = 0.0
    total_turns: int = 0
    total_tokens: int = 0
    trajectory_file: str = ""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS attempts (
    id TEXT PRIMARY KEY,
    repo_id TEXT NOT NULL,
    sha TEXT,
    docker_image TEXT,
    fingerprint_json TEXT,
    started_at REAL,
    outcome TEXT,
    duration_s REAL,
    total_turns INTEGER,
    total_tokens INTEGER,
    trajectory_file TEXT
);
CREATE INDEX IF NOT EXISTS idx_repo ON attempts(repo_id);
CREATE INDEX IF NOT EXISTS idx_outcome ON attempts(outcome);
"""


class TrajectoryStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def start_attempt(self, a: BuildAttempt) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO attempts(id, repo_id, sha, docker_image, fingerprint_json,
                                     started_at, outcome, duration_s, total_turns, total_tokens, trajectory_file)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    a.id, a.repo_id, a.sha, a.docker_image,
                    json.dumps(a.fingerprint or {}),
                    a.started_at, a.outcome, a.duration_s,
                    a.total_turns, a.total_tokens, a.trajectory_file,
                ),
            )
            self._conn.commit()

    def complete_attempt(
        self, attempt_id: str, *, outcome: str, duration_s: float,
        total_turns: int, total_tokens: int,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE attempts SET outcome=?, duration_s=?, total_turns=?, total_tokens=?
                WHERE id=?
                """,
                (outcome, duration_s, total_turns, total_tokens, attempt_id),
            )
            self._conn.commit()

    def stats(self) -> dict[str, int]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT COUNT(*), SUM(CASE WHEN outcome='success' THEN 1 ELSE 0 END), COUNT(DISTINCT repo_id) FROM attempts"
            )
            total, success, unique = cur.fetchone()
            return {
                "total_attempts": int(total or 0),
                "successful_attempts": int(success or 0),
                "unique_repos": int(unique or 0),
            }
