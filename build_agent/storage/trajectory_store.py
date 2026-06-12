"""
Trajectory store — append-only JSONL recording of every agent action.

Write path: every action in the configuration loop emits a TrajectoryRecord.
Read path:  the learning pipeline queries trajectories by repo, attempt, or
            error class for post-hoc distillation.

Storage:
  - One JSONL file per build attempt in the output directory.
  - A lightweight SQLite index for cross-attempt queries (error patterns,
    timing, success rates).
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional

from storage.models import TrajectoryRecord, BuildAttempt, BuildOutcome


class TrajectoryStore:
    """Append-only trajectory writer + SQLite index for queries."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_db()

    # ── lifecycle ────────────────────────────────────────────────────────────

    def _ensure_db(self):
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS build_attempts (
                id TEXT PRIMARY KEY,
                repo_id TEXT NOT NULL,
                repo_url TEXT,
                sha TEXT,
                outcome TEXT,
                duration_minutes REAL,
                docker_image TEXT,
                total_turns INTEGER,
                total_tokens INTEGER,
                trajectory_file TEXT,
                started_at REAL,
                completed_at REAL,
                fingerprint_sig TEXT,
                data_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_attempts_repo ON build_attempts(repo_id);
            CREATE INDEX IF NOT EXISTS idx_attempts_outcome ON build_attempts(outcome);
            CREATE INDEX IF NOT EXISTS idx_attempts_fingerprint ON build_attempts(fingerprint_sig);

            CREATE TABLE IF NOT EXISTS trajectory_index (
                id TEXT PRIMARY KEY,
                attempt_id TEXT NOT NULL,
                repo_id TEXT NOT NULL,
                agent TEXT,
                action_type TEXT,
                outcome TEXT,
                return_code INTEGER,
                error_class TEXT,
                duration_seconds REAL,
                turn_number INTEGER,
                led_to_success INTEGER,
                novel_situation INTEGER,
                timestamp REAL,
                FOREIGN KEY (attempt_id) REFERENCES build_attempts(id)
            );
            CREATE INDEX IF NOT EXISTS idx_traj_attempt ON trajectory_index(attempt_id);
            CREATE INDEX IF NOT EXISTS idx_traj_error ON trajectory_index(error_class);
            CREATE INDEX IF NOT EXISTS idx_traj_repo ON trajectory_index(repo_id);
        """)
        self._conn.commit()

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── writing ──────────────────────────────────────────────────────────────

    def start_attempt(self, attempt: BuildAttempt) -> str:
        """Register a new build attempt and return its ID."""
        fp_sig = attempt.fingerprint.signature() if attempt.fingerprint else ""
        self._conn.execute(
            """INSERT OR REPLACE INTO build_attempts
               (id, repo_id, repo_url, sha, outcome, duration_minutes,
                docker_image, total_turns, total_tokens, trajectory_file,
                started_at, completed_at, fingerprint_sig, data_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (attempt.id, attempt.repo_id, attempt.repo_url, attempt.sha,
             attempt.outcome, attempt.duration_minutes, attempt.docker_image,
             attempt.total_turns, attempt.total_tokens, attempt.trajectory_file,
             attempt.started_at, attempt.completed_at, fp_sig,
             json.dumps(attempt.to_dict())),
        )
        self._conn.commit()
        return attempt.id

    def complete_attempt(self, attempt_id: str, outcome: str,
                         duration_minutes: float, total_turns: int,
                         total_tokens: int):
        """Mark a build attempt as completed."""
        self._conn.execute(
            """UPDATE build_attempts
               SET outcome=?, duration_minutes=?, total_turns=?,
                   total_tokens=?, completed_at=?
               WHERE id=?""",
            (outcome, duration_minutes, total_turns, total_tokens,
             time.time(), attempt_id),
        )
        self._conn.commit()

    def record_action(self, record: TrajectoryRecord, jsonl_path: str):
        """Write a trajectory record to JSONL and index in SQLite."""
        os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
        with open(jsonl_path, "a") as f:
            f.write(record.to_json() + "\n")

        self._conn.execute(
            """INSERT OR REPLACE INTO trajectory_index
               (id, attempt_id, repo_id, agent, action_type, outcome,
                return_code, error_class, duration_seconds, turn_number,
                led_to_success, novel_situation, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (record.id, record.attempt_id, record.repo_id, record.agent,
             record.action_type, record.outcome, record.return_code,
             record.error_class, record.duration_seconds, record.turn_number,
             1 if record.led_to_success else 0,
             1 if record.novel_situation else 0, record.timestamp),
        )
        self._conn.commit()

    def mark_success_retroactive(self, attempt_id: str):
        """After a successful build, mark all actions in the attempt as led_to_success."""
        self._conn.execute(
            "UPDATE trajectory_index SET led_to_success=1 WHERE attempt_id=?",
            (attempt_id,),
        )
        self._conn.commit()

    # ── querying ─────────────────────────────────────────────────────────────

    def get_attempt(self, attempt_id: str) -> Optional[BuildAttempt]:
        row = self._conn.execute(
            "SELECT data_json FROM build_attempts WHERE id=?", (attempt_id,)
        ).fetchone()
        if row:
            return BuildAttempt.from_dict(json.loads(row[0]))
        return None

    def get_attempts_for_repo(self, repo_id: str) -> List[BuildAttempt]:
        rows = self._conn.execute(
            "SELECT data_json FROM build_attempts WHERE repo_id=? ORDER BY started_at DESC",
            (repo_id,),
        ).fetchall()
        return [BuildAttempt.from_dict(json.loads(r[0])) for r in rows]

    def get_successful_attempts_by_fingerprint(self, fingerprint_sig: str,
                                                limit: int = 5) -> List[BuildAttempt]:
        """Find successful builds with a similar fingerprint."""
        rows = self._conn.execute(
            """SELECT data_json FROM build_attempts
               WHERE fingerprint_sig=? AND outcome=?
               ORDER BY completed_at DESC LIMIT ?""",
            (fingerprint_sig, BuildOutcome.SUCCESS.value, limit),
        ).fetchall()
        return [BuildAttempt.from_dict(json.loads(r[0])) for r in rows]

    def get_error_frequency(self, error_class: str) -> Dict[str, int]:
        """Count how often an error class appears across repos."""
        row = self._conn.execute(
            """SELECT COUNT(*), COUNT(DISTINCT repo_id)
               FROM trajectory_index WHERE error_class=?""",
            (error_class,),
        ).fetchone()
        return {"total_occurrences": row[0], "unique_repos": row[1]}

    def get_common_errors(self, limit: int = 20) -> List[Dict]:
        """Return the most common error classes across all attempts."""
        rows = self._conn.execute(
            """SELECT error_class, COUNT(*) as cnt, COUNT(DISTINCT repo_id) as repos
               FROM trajectory_index
               WHERE error_class IS NOT NULL AND error_class != ''
               GROUP BY error_class ORDER BY cnt DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [
            {"error_class": r[0], "count": r[1], "unique_repos": r[2]}
            for r in rows
        ]

    def load_trajectory(self, jsonl_path: str) -> List[TrajectoryRecord]:
        """Load a full trajectory from a JSONL file."""
        records = []
        if not os.path.exists(jsonl_path):
            return records
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(TrajectoryRecord.from_dict(json.loads(line)))
        return records

    def get_stats(self) -> Dict[str, Any]:
        """Global statistics for the trajectory store."""
        stats = {}
        row = self._conn.execute("SELECT COUNT(*) FROM build_attempts").fetchone()
        stats["total_attempts"] = row[0]
        row = self._conn.execute(
            "SELECT COUNT(*) FROM build_attempts WHERE outcome=?",
            (BuildOutcome.SUCCESS.value,)
        ).fetchone()
        stats["successful_attempts"] = row[0]
        row = self._conn.execute("SELECT COUNT(*) FROM trajectory_index").fetchone()
        stats["total_actions"] = row[0]
        row = self._conn.execute(
            "SELECT COUNT(DISTINCT repo_id) FROM build_attempts"
        ).fetchone()
        stats["unique_repos"] = row[0]
        return stats
