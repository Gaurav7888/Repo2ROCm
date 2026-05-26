"""
Knowledge Graph Store — SQLite-backed versioned knowledge base.

Stores packages, compatibility relationships, error patterns, fixes,
and executable rules.  Every mutation is versioned so the KB can be
rolled back when an update makes things worse.

The graph topology (nodes + edges) is modeled as relational tables with
JSON payloads for flexibility.  This avoids a graph-DB dependency while
keeping queries fast at the expected scale (thousands of rules, tens of
thousands of facts).
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

from storage.models import (
    ErrorPattern, Fix, Rule, KBUpdateProposal, KBUpdateType, RuleSource,
)


class KBStore:
    """Versioned knowledge base backed by SQLite."""

    CONFIDENCE_THRESHOLD_DETERMINISTIC = 0.85
    DEPRECATION_THRESHOLD = 0.3
    MIN_EVIDENCE_FOR_DEPRECATION = 5

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_db()

    def _ensure_db(self):
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS packages (
                name TEXT PRIMARY KEY,
                data_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS compatibility (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                package TEXT NOT NULL,
                rocm_version TEXT NOT NULL,
                compatible INTEGER NOT NULL,  -- 1=yes, 0=no
                confidence REAL DEFAULT 0.5,
                install_method TEXT,
                install_commands TEXT,  -- JSON list
                notes TEXT,
                workaround TEXT,
                evidence_count INTEGER DEFAULT 0,
                source TEXT DEFAULT 'seed',
                created_at REAL,
                updated_at REAL,
                UNIQUE(package, rocm_version, install_method)
            );
            CREATE INDEX IF NOT EXISTS idx_compat_pkg ON compatibility(package);
            CREATE INDEX IF NOT EXISTS idx_compat_rocm ON compatibility(rocm_version);

            CREATE TABLE IF NOT EXISTS error_patterns (
                id TEXT PRIMARY KEY,
                signature TEXT UNIQUE,
                error_class TEXT NOT NULL,
                description TEXT,
                regex_pattern TEXT,
                severity TEXT DEFAULT 'error',
                rocm_version_range TEXT,
                evidence_count INTEGER DEFAULT 0,
                confidence REAL DEFAULT 0.5,
                created_at REAL,
                last_seen REAL,
                data_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_error_class ON error_patterns(error_class);

            CREATE TABLE IF NOT EXISTS fixes (
                id TEXT PRIMARY KEY,
                description TEXT,
                success_rate REAL DEFAULT 0.0,
                evidence_count INTEGER DEFAULT 0,
                valid_rocm_range TEXT,
                created_at REAL,
                data_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS error_fix_map (
                error_id TEXT NOT NULL,
                fix_id TEXT NOT NULL,
                success_rate REAL DEFAULT 0.0,
                application_count INTEGER DEFAULT 0,
                PRIMARY KEY (error_id, fix_id),
                FOREIGN KEY (error_id) REFERENCES error_patterns(id),
                FOREIGN KEY (fix_id) REFERENCES fixes(id)
            );

            CREATE TABLE IF NOT EXISTS rules (
                id TEXT PRIMARY KEY,
                version INTEGER DEFAULT 1,
                confidence REAL DEFAULT 0.5,
                source TEXT DEFAULT 'seed',
                evidence_count INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                deprecated INTEGER DEFAULT 0,
                valid_rocm_range TEXT,
                created_at REAL,
                last_applied REAL,
                data_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_rules_confidence ON rules(confidence);
            CREATE INDEX IF NOT EXISTS idx_rules_deprecated ON rules(deprecated);

            CREATE TABLE IF NOT EXISTS kb_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                update_type TEXT NOT NULL,
                target_table TEXT NOT NULL,
                target_id TEXT,
                old_data TEXT,
                new_data TEXT,
                source_attempt TEXT,
                timestamp REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS docker_images (
                tag TEXT PRIMARY KEY,
                rocm_version TEXT,
                pytorch_version TEXT,
                base_os TEXT,
                preinstalled_packages TEXT,  -- JSON list
                data_json TEXT NOT NULL
            );

            -- Per-host image-failure memory (Patch 2): lets the ranker
            -- hard-reject images that have killed the container at
            -- startup on this host in past tasks. `failure_count >= 2`
            -- is the strike threshold enforced by `is_image_known_bad`
            -- so a single transient crash does not permanently
            -- blacklist an image.
            CREATE TABLE IF NOT EXISTS host_image_failures (
              host_arch       TEXT NOT NULL,
              image           TEXT NOT NULL,
              failure_count   INTEGER DEFAULT 1,
              failure_kind    TEXT NOT NULL,
              last_seen       REAL,
              first_seen      REAL,
              PRIMARY KEY (host_arch, image)
            );
            CREATE INDEX IF NOT EXISTS idx_host_image_failures
                ON host_image_failures(host_arch, image);
        """)
        self._conn.commit()

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── version tracking ─────────────────────────────────────────────────────

    def _record_change(self, update_type: str, table: str,
                       target_id: str, old_data: Any, new_data: Any,
                       source_attempt: str = ""):
        self._conn.execute(
            """INSERT INTO kb_history
               (update_type, target_table, target_id, old_data, new_data,
                source_attempt, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (update_type, table, target_id,
             json.dumps(old_data) if old_data else None,
             json.dumps(new_data) if new_data else None,
             source_attempt, time.time()),
        )

    # ── error patterns ───────────────────────────────────────────────────────

    def add_error_pattern(self, pattern: ErrorPattern,
                          source_attempt: str = "") -> str:
        self._conn.execute(
            """INSERT OR REPLACE INTO error_patterns
               (id, signature, error_class, description, regex_pattern,
                severity, rocm_version_range, evidence_count, confidence,
                created_at, last_seen, data_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (pattern.id, pattern.signature, pattern.error_class,
             pattern.description, pattern.regex_pattern, pattern.severity,
             pattern.rocm_version_range, pattern.evidence_count,
             pattern.confidence, pattern.created_at, pattern.last_seen,
             json.dumps(pattern.to_dict())),
        )
        self._record_change("add_error_pattern", "error_patterns",
                            pattern.id, None, pattern.to_dict(), source_attempt)
        self._conn.commit()
        return pattern.id

    def get_error_pattern(self, pattern_id: str) -> Optional[ErrorPattern]:
        row = self._conn.execute(
            "SELECT data_json FROM error_patterns WHERE id=?", (pattern_id,)
        ).fetchone()
        return ErrorPattern.from_dict(json.loads(row[0])) if row else None

    def get_all_error_patterns(self) -> List[ErrorPattern]:
        rows = self._conn.execute(
            "SELECT data_json FROM error_patterns ORDER BY evidence_count DESC"
        ).fetchall()
        return [ErrorPattern.from_dict(json.loads(r[0])) for r in rows]

    def find_error_by_class(self, error_class: str) -> List[ErrorPattern]:
        rows = self._conn.execute(
            "SELECT data_json FROM error_patterns WHERE error_class=?",
            (error_class,),
        ).fetchall()
        return [ErrorPattern.from_dict(json.loads(r[0])) for r in rows]

    def update_error_evidence(self, pattern_id: str):
        """Bump evidence count and last_seen for a pattern."""
        self._conn.execute(
            """UPDATE error_patterns
               SET evidence_count = evidence_count + 1, last_seen = ?
               WHERE id = ?""",
            (time.time(), pattern_id),
        )
        self._conn.commit()

    # ── fixes ────────────────────────────────────────────────────────────────

    def add_fix(self, fix: Fix, source_attempt: str = "") -> str:
        self._conn.execute(
            """INSERT OR REPLACE INTO fixes
               (id, description, success_rate, evidence_count,
                valid_rocm_range, created_at, data_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (fix.id, fix.description, fix.success_rate, fix.evidence_count,
             fix.valid_rocm_range, fix.created_at,
             json.dumps(fix.to_dict())),
        )
        self._record_change("add_fix", "fixes", fix.id, None,
                            fix.to_dict(), source_attempt)
        self._conn.commit()
        return fix.id

    def get_fix(self, fix_id: str) -> Optional[Fix]:
        row = self._conn.execute(
            "SELECT data_json FROM fixes WHERE id=?", (fix_id,)
        ).fetchone()
        return Fix.from_dict(json.loads(row[0])) if row else None

    def link_error_to_fix(self, error_id: str, fix_id: str):
        self._conn.execute(
            """INSERT OR IGNORE INTO error_fix_map (error_id, fix_id)
               VALUES (?, ?)""",
            (error_id, fix_id),
        )
        self._conn.commit()

    def get_fixes_for_error(self, error_id: str) -> List[Fix]:
        rows = self._conn.execute(
            """SELECT f.data_json FROM fixes f
               JOIN error_fix_map m ON f.id = m.fix_id
               WHERE m.error_id = ?
               ORDER BY f.success_rate DESC""",
            (error_id,),
        ).fetchall()
        return [Fix.from_dict(json.loads(r[0])) for r in rows]

    def record_fix_outcome(self, fix_id: str, error_id: str, success: bool):
        """Update fix success rate after application."""
        fix = self.get_fix(fix_id)
        if not fix:
            return
        fix.evidence_count += 1
        if success:
            fix.success_rate = (
                (fix.success_rate * (fix.evidence_count - 1) + 1.0)
                / fix.evidence_count
            )
        else:
            fix.success_rate = (
                (fix.success_rate * (fix.evidence_count - 1))
                / fix.evidence_count
            )
        self._conn.execute(
            "UPDATE fixes SET success_rate=?, evidence_count=?, data_json=? WHERE id=?",
            (fix.success_rate, fix.evidence_count, json.dumps(fix.to_dict()), fix.id),
        )
        self._conn.execute(
            """UPDATE error_fix_map
               SET success_rate = ?, application_count = application_count + 1
               WHERE error_id = ? AND fix_id = ?""",
            (fix.success_rate, error_id, fix_id),
        )
        self._conn.commit()

    # ── rules ────────────────────────────────────────────────────────────────

    def add_rule(self, rule: Rule, source_attempt: str = "") -> str:
        self._conn.execute(
            """INSERT OR REPLACE INTO rules
               (id, version, confidence, source, evidence_count,
                success_rate, success_count, failure_count, deprecated,
                valid_rocm_range, created_at, last_applied, data_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (rule.id, rule.version, rule.confidence, rule.source,
             rule.evidence_count, rule.success_rate, rule.success_count,
             rule.failure_count, 1 if rule.deprecated else 0,
             rule.valid_rocm_range, rule.created_at, rule.last_applied,
             json.dumps(rule.to_dict())),
        )
        self._record_change("add_rule", "rules", rule.id, None,
                            rule.to_dict(), source_attempt)
        self._conn.commit()
        return rule.id

    def get_rule(self, rule_id: str) -> Optional[Rule]:
        row = self._conn.execute(
            "SELECT data_json FROM rules WHERE id=?", (rule_id,)
        ).fetchone()
        return Rule.from_dict(json.loads(row[0])) if row else None

    def get_active_rules(self) -> List[Rule]:
        """Return all non-deprecated rules sorted by confidence."""
        rows = self._conn.execute(
            """SELECT data_json FROM rules
               WHERE deprecated = 0
               ORDER BY confidence DESC"""
        ).fetchall()
        return [Rule.from_dict(json.loads(r[0])) for r in rows]

    def get_deterministic_rules(self) -> List[Rule]:
        """Rules with high enough confidence to apply without LLM."""
        rows = self._conn.execute(
            """SELECT data_json FROM rules
               WHERE deprecated = 0 AND confidence >= ? AND evidence_count >= 3
               ORDER BY confidence DESC""",
            (self.CONFIDENCE_THRESHOLD_DETERMINISTIC,),
        ).fetchall()
        return [Rule.from_dict(json.loads(r[0])) for r in rows]

    def update_rule_outcome(self, rule_id: str, success: bool):
        """Update rule statistics after application."""
        rule = self.get_rule(rule_id)
        if not rule:
            return
        rule.record_application(success)
        self._conn.execute(
            """UPDATE rules SET confidence=?, evidence_count=?,
               success_rate=?, success_count=?, failure_count=?,
               deprecated=?, last_applied=?, data_json=?
               WHERE id=?""",
            (rule.confidence, rule.evidence_count, rule.success_rate,
             rule.success_count, rule.failure_count,
             1 if rule.deprecated else 0, rule.last_applied,
             json.dumps(rule.to_dict()), rule.id),
        )
        self._conn.commit()

    def supersede_rule(self, old_rule_id: str, new_rule: Rule,
                       source_attempt: str = ""):
        """Replace an old rule with a new one."""
        old = self.get_rule(old_rule_id)
        if old:
            old.deprecated = True
            self._conn.execute(
                "UPDATE rules SET deprecated=1, data_json=? WHERE id=?",
                (json.dumps(old.to_dict()), old.id),
            )
        new_rule.supersedes.append(old_rule_id)
        self.add_rule(new_rule, source_attempt)

    # ── compatibility ────────────────────────────────────────────────────────

    def add_compatibility(self, package: str, rocm_version: str,
                          compatible: bool, install_method: str = "pip",
                          install_commands: Optional[List[str]] = None,
                          notes: str = "", confidence: float = 0.5,
                          source_attempt: str = ""):
        self._conn.execute(
            """INSERT OR REPLACE INTO compatibility
               (package, rocm_version, compatible, confidence,
                install_method, install_commands, notes, evidence_count,
                source, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 1, 'learned', ?, ?)""",
            (package, rocm_version, 1 if compatible else 0, confidence,
             install_method,
             json.dumps(install_commands) if install_commands else None,
             notes, time.time(), time.time()),
        )
        self._record_change("add_compatibility", "compatibility",
                            f"{package}@{rocm_version}", None,
                            {"package": package, "rocm_version": rocm_version,
                             "compatible": compatible},
                            source_attempt)
        self._conn.commit()

    def get_compatibility(self, package: str,
                          rocm_version: Optional[str] = None) -> List[Dict]:
        if rocm_version:
            rows = self._conn.execute(
                """SELECT package, rocm_version, compatible, confidence,
                          install_method, install_commands, notes
                   FROM compatibility
                   WHERE package=? AND rocm_version=?""",
                (package, rocm_version),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT package, rocm_version, compatible, confidence,
                          install_method, install_commands, notes
                   FROM compatibility WHERE package=?""",
                (package,),
            ).fetchall()
        return [
            {
                "package": r[0], "rocm_version": r[1],
                "compatible": bool(r[2]), "confidence": r[3],
                "install_method": r[4],
                "install_commands": json.loads(r[5]) if r[5] else [],
                "notes": r[6],
            }
            for r in rows
        ]

    # ── KB update proposals ──────────────────────────────────────────────────

    def apply_update(self, proposal: KBUpdateProposal) -> bool:
        """Apply a KB update proposal after consistency check."""
        if not self._consistency_check(proposal):
            return False

        ut = proposal.update_type
        if ut == KBUpdateType.ADD_FACT.value:
            p = proposal.payload
            error_id = p.get("error_id")
            fix_data = p.get("fix")
            if error_id and fix_data:
                fix = Fix.from_dict(fix_data)
                self.add_fix(fix, proposal.source_attempt_id)
                self.link_error_to_fix(error_id, fix.id)

        elif ut == KBUpdateType.ADD_ERROR_PATTERN.value:
            pattern = ErrorPattern.from_dict(proposal.payload)
            self.add_error_pattern(pattern, proposal.source_attempt_id)

        elif ut == KBUpdateType.ADD_INSTALL_PATH.value:
            p = proposal.payload
            self.add_compatibility(
                p["package"], p["rocm_version"], p.get("compatible", True),
                p.get("install_method", "pip"),
                p.get("install_commands"), p.get("notes", ""),
                proposal.confidence, proposal.source_attempt_id,
            )

        elif ut == KBUpdateType.SUPERSEDE_RULE.value:
            old_id = proposal.payload.get("old_rule_id")
            new_rule = Rule.from_dict(proposal.payload.get("new_rule", {}))
            if old_id:
                self.supersede_rule(old_id, new_rule, proposal.source_attempt_id)
            else:
                self.add_rule(new_rule, proposal.source_attempt_id)

        elif ut == KBUpdateType.UPDATE_CONFIDENCE.value:
            target = proposal.target_id
            new_conf = proposal.payload.get("confidence", 0.5)
            self._conn.execute(
                "UPDATE rules SET confidence=? WHERE id=?",
                (new_conf, target),
            )
            self._conn.commit()

        elif ut == KBUpdateType.DEPRECATE_FIX.value:
            fix_id = proposal.target_id
            if fix_id:
                fix = self.get_fix(fix_id)
                if fix:
                    fix.success_rate = 0.0
                    self._conn.execute(
                        "UPDATE fixes SET success_rate=0, data_json=? WHERE id=?",
                        (json.dumps(fix.to_dict()), fix.id),
                    )
                    self._conn.commit()

        return True

    def _consistency_check(self, proposal: KBUpdateProposal) -> bool:
        """Verify a proposed update doesn't contradict high-confidence existing knowledge."""
        if proposal.update_type == KBUpdateType.ADD_INSTALL_PATH.value:
            pkg = proposal.payload.get("package", "")
            rocm = proposal.payload.get("rocm_version", "")
            existing = self.get_compatibility(pkg, rocm)
            for entry in existing:
                if entry["confidence"] > 0.9 and entry["compatible"] != proposal.payload.get("compatible", True):
                    proposal.conflicts_with.append(
                        f"High-confidence existing record for {pkg}@{rocm} "
                        f"says compatible={entry['compatible']}"
                    )
                    return False
        return True

    # ── host image failures (Patch 2) ───────────────────────────────────────

    def record_image_failure(self, host_arch: str, image: str,
                             kind: str = "startup_crash") -> None:
        """UPSERT a per-(host, image) failure.

        Increments `failure_count` and refreshes `last_seen` on every call;
        `first_seen` is preserved across upserts so callers can age out
        stale entries later if they want.
        """
        now = time.time()
        self._conn.execute(
            """INSERT INTO host_image_failures
                 (host_arch, image, failure_count, failure_kind,
                  last_seen, first_seen)
               VALUES (?, ?, 1, ?, ?, ?)
               ON CONFLICT(host_arch, image) DO UPDATE SET
                 failure_count = failure_count + 1,
                 failure_kind = excluded.failure_kind,
                 last_seen = excluded.last_seen""",
            (host_arch, image, kind, now, now),
        )
        self._conn.commit()

    def is_image_known_bad(self, host_arch: str, image: str) -> bool:
        """True iff (host_arch, image) has failed at startup on at least
        two independent prior tasks."""
        row = self._conn.execute(
            """SELECT failure_count FROM host_image_failures
               WHERE host_arch = ? AND image = ?""",
            (host_arch, image),
        ).fetchone()
        return bool(row and row[0] >= 2)

    # ── stats ────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        stats = {}
        for table in ("error_patterns", "fixes", "rules", "compatibility"):
            row = self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            stats[f"{table}_count"] = row[0]
        row = self._conn.execute(
            "SELECT COUNT(*) FROM rules WHERE deprecated=0"
        ).fetchone()
        stats["active_rules"] = row[0]
        row = self._conn.execute(
            f"SELECT COUNT(*) FROM rules WHERE confidence >= {self.CONFIDENCE_THRESHOLD_DETERMINISTIC}"
        ).fetchone()
        stats["deterministic_rules"] = row[0]
        return stats
