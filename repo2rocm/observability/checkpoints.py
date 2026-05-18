"""Named checkpoints à la Claude Code's startup profiler.

We sprinkle `checkpoint("bootstrap.config_load")` etc. through the codebase, and the
CheckpointRegistry collects timing data we can dump on shutdown or per-request.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from repo2rocm.observability.tracing import span


@dataclass
class CheckpointRecord:
    name: str
    ts_seconds: float
    delta_ms: float  # ms since previous checkpoint
    cumulative_ms: float  # ms since registry start


@dataclass
class CheckpointRegistry:
    records: list[CheckpointRecord] = field(default_factory=list)
    _start: float = field(default_factory=time.perf_counter)
    _last: float = field(default_factory=time.perf_counter)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, name: str) -> CheckpointRecord:
        with self._lock:
            now = time.perf_counter()
            rec = CheckpointRecord(
                name=name,
                ts_seconds=time.time(),
                delta_ms=(now - self._last) * 1000.0,
                cumulative_ms=(now - self._start) * 1000.0,
            )
            self.records.append(rec)
            self._last = now
            return rec

    def reset(self) -> None:
        with self._lock:
            self.records.clear()
            self._start = time.perf_counter()
            self._last = self._start

    def summary(self) -> list[dict[str, float | str]]:
        return [
            {
                "name": r.name,
                "delta_ms": round(r.delta_ms, 3),
                "cumulative_ms": round(r.cumulative_ms, 3),
            }
            for r in self.records
        ]


_REGISTRY = CheckpointRegistry()


def checkpoint(name: str) -> CheckpointRecord:
    """Record a named checkpoint. Also emits a 1-tick OTel span for trace integration."""
    rec = _REGISTRY.record(name)
    with span("checkpoint", name=name, delta_ms=rec.delta_ms, cumulative_ms=rec.cumulative_ms):
        pass
    return rec


def get_registry() -> CheckpointRegistry:
    return _REGISTRY
