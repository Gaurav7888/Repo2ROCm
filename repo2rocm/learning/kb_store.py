"""KB store — SQLite-backed compatibility + rules + error patterns. Versioned."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CompatibilityRecord:
    package: str
    rocm_version: str
    compatible: bool
    install_method: str = "pip"
    install_commands: list[str] = field(default_factory=list)
    notes: str = ""
    confidence: float = 0.5
    evidence_count: int = 0


@dataclass
class ErrorPattern:
    signature: str
    error_class: str
    description: str
    regex_pattern: str
    severity: str = "error"
    evidence_count: int = 0
    confidence: float = 0.5


@dataclass
class Fix:
    id: str
    description: str
    commands: list[str]
    success_rate: float = 0.0
    evidence_count: int = 0


@dataclass
class Rule:
    name: str
    when: dict[str, Any]
    do: dict[str, Any]
    source: str = "seed"  # "seed" | "distilled" | "user"
    confidence: float = 0.5


_SCHEMA = """
CREATE TABLE IF NOT EXISTS packages (
    name TEXT PRIMARY KEY,
    data_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS compatibility (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    package TEXT NOT NULL,
    rocm_version TEXT NOT NULL,
    compatible INTEGER NOT NULL,
    install_method TEXT,
    install_commands TEXT,
    notes TEXT,
    confidence REAL DEFAULT 0.5,
    evidence_count INTEGER DEFAULT 0,
    updated_at REAL,
    UNIQUE(package, rocm_version, install_method)
);
CREATE INDEX IF NOT EXISTS idx_compat_pkg ON compatibility(package);
CREATE TABLE IF NOT EXISTS error_patterns (
    signature TEXT PRIMARY KEY,
    error_class TEXT NOT NULL,
    description TEXT,
    regex_pattern TEXT,
    severity TEXT DEFAULT 'error',
    evidence_count INTEGER DEFAULT 0,
    confidence REAL DEFAULT 0.5,
    updated_at REAL
);
CREATE INDEX IF NOT EXISTS idx_error_class ON error_patterns(error_class);
CREATE TABLE IF NOT EXISTS fixes (
    id TEXT PRIMARY KEY,
    description TEXT,
    commands TEXT,
    success_rate REAL DEFAULT 0.0,
    evidence_count INTEGER DEFAULT 0,
    updated_at REAL
);
CREATE TABLE IF NOT EXISTS rules (
    name TEXT PRIMARY KEY,
    when_json TEXT NOT NULL,
    do_json TEXT NOT NULL,
    source TEXT,
    confidence REAL,
    updated_at REAL
);
"""


class KBStore:
    DETERMINISTIC_CONFIDENCE_THRESHOLD = 0.85

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

    # ── compatibility ─────────────────────────────────────────────────────────

    def upsert_compatibility(self, rec: CompatibilityRecord) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO compatibility(package, rocm_version, compatible, install_method,
                                          install_commands, notes, confidence, evidence_count, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(package, rocm_version, install_method) DO UPDATE SET
                    compatible=excluded.compatible,
                    install_commands=excluded.install_commands,
                    notes=excluded.notes,
                    confidence=excluded.confidence,
                    evidence_count=evidence_count + 1,
                    updated_at=excluded.updated_at
                """,
                (
                    rec.package,
                    rec.rocm_version,
                    int(rec.compatible),
                    rec.install_method,
                    json.dumps(rec.install_commands),
                    rec.notes,
                    rec.confidence,
                    rec.evidence_count,
                    time.time(),
                ),
            )
            self._conn.commit()

    def get_compatibility(self, package: str) -> list[CompatibilityRecord]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT package, rocm_version, compatible, install_method, install_commands, notes, confidence, evidence_count FROM compatibility WHERE package = ?",
                (package,),
            )
            out = []
            for row in cur.fetchall():
                out.append(
                    CompatibilityRecord(
                        package=row[0],
                        rocm_version=row[1],
                        compatible=bool(row[2]),
                        install_method=row[3] or "pip",
                        install_commands=json.loads(row[4] or "[]"),
                        notes=row[5] or "",
                        confidence=row[6] or 0.5,
                        evidence_count=row[7] or 0,
                    )
                )
            return out

    # ── error patterns ─────────────────────────────────────────────────────────

    def upsert_error_pattern(self, ep: ErrorPattern) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO error_patterns(signature, error_class, description, regex_pattern, severity, evidence_count, confidence, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(signature) DO UPDATE SET
                    evidence_count=evidence_count + 1,
                    confidence=excluded.confidence,
                    updated_at=excluded.updated_at
                """,
                (
                    ep.signature,
                    ep.error_class,
                    ep.description,
                    ep.regex_pattern,
                    ep.severity,
                    ep.evidence_count,
                    ep.confidence,
                    time.time(),
                ),
            )
            self._conn.commit()

    def list_error_patterns(self) -> list[ErrorPattern]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT signature, error_class, description, regex_pattern, severity, evidence_count, confidence FROM error_patterns"
            )
            return [
                ErrorPattern(*row[:5], evidence_count=row[5] or 0, confidence=row[6] or 0.5)
                for row in cur.fetchall()
            ]

    # ── rules ─────────────────────────────────────────────────────────────────

    def upsert_rule(self, rule: Rule) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO rules(name, when_json, do_json, source, confidence, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    when_json=excluded.when_json,
                    do_json=excluded.do_json,
                    confidence=excluded.confidence,
                    updated_at=excluded.updated_at
                """,
                (
                    rule.name,
                    json.dumps(rule.when),
                    json.dumps(rule.do),
                    rule.source,
                    rule.confidence,
                    time.time(),
                ),
            )
            self._conn.commit()

    def list_rules(self, *, min_confidence: float = 0.0) -> list[Rule]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT name, when_json, do_json, source, confidence FROM rules WHERE confidence >= ?",
                (min_confidence,),
            )
            return [
                Rule(
                    name=row[0],
                    when=json.loads(row[1]),
                    do=json.loads(row[2]),
                    source=row[3] or "seed",
                    confidence=row[4] or 0.5,
                )
                for row in cur.fetchall()
            ]

    # ── stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, int]:
        with self._lock:
            def count(tbl: str) -> int:
                cur = self._conn.execute(f"SELECT COUNT(*) FROM {tbl}")
                return int(cur.fetchone()[0])

            return {
                "compatibility": count("compatibility"),
                "error_patterns": count("error_patterns"),
                "rules": count("rules"),
                "deterministic_rules": int(
                    self._conn.execute(
                        "SELECT COUNT(*) FROM rules WHERE confidence >= ?",
                        (self.DETERMINISTIC_CONFIDENCE_THRESHOLD,),
                    ).fetchone()[0]
                ),
            }
