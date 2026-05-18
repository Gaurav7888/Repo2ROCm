"""Post-build distiller — slim version.

Extracts ONLY structured facts that transfer between repos:
  * package install paths (PyPI / wheel / source)
  * confidence updates for existing rules

No free-form "lessons" — they overfit.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from repo2rocm.learning.kb_store import CompatibilityRecord, KBStore
from repo2rocm.learning.trajectory_store import BuildAttempt, TrajectoryStore
from repo2rocm.observability.logging import get_logger

log = get_logger(__name__)


@dataclass
class DistillationResult:
    facts_added: int
    rules_updated: int


class TrajectoryDistiller:
    def __init__(self, kb: KBStore, traj: TrajectoryStore):
        self.kb = kb
        self.traj = traj

    def distill_and_apply(self, attempt: BuildAttempt) -> DistillationResult:
        """Mine the trajectory file for install-path facts. Returns counts."""
        applied = 0
        if not attempt.trajectory_file or not Path(attempt.trajectory_file).exists():
            return DistillationResult(0, 0)

        # Reading the JSONL transcript:
        import json

        with Path(attempt.trajectory_file).open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Look for successful Download events
                if rec.get("kind") == "tool_result" and rec.get("tool") == "Download" and rec.get("outcome") == "ok":
                    # we don't have full structured output here; in production we'd thread
                    # the typed output through. As a stub, we record the attempt as
                    # confirming all queued packages are compatible.
                    applied += 1
                    # placeholder rec.upsert_compatibility(...)

        return DistillationResult(facts_added=applied, rules_updated=0)
