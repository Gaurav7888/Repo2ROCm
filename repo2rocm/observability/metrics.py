"""Prometheus metrics for the agent kernel.

We track:
  * Per-tool call count + latency + bytes
  * Per-turn LLM latency + prompt/completion/cached tokens
  * Sub-agent fan-out (active gauge)
  * Permission decisions
  * Hook invocations
  * Sandbox commit/rollback counts
  * Migration outcomes

If `prometheus_client` is missing, all helpers become no-ops.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        start_http_server,
    )

    _PROM_AVAILABLE = True
except Exception:  # pragma: no cover
    _PROM_AVAILABLE = False
    CollectorRegistry = object  # type: ignore[assignment, misc]


_DEFAULT_LATENCY_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0,
    2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0,
)
_TOKEN_BUCKETS = (
    100, 500, 1_000, 2_000, 4_000, 8_000, 16_000, 32_000,
    65_000, 128_000, 200_000, 400_000, 800_000,
)


@dataclass
class _NoOpMetric:
    name: str = "noop"

    def labels(self, **kw: Any) -> _NoOpMetric:
        return self

    def inc(self, n: float = 1.0) -> None: ...
    def dec(self, n: float = 1.0) -> None: ...
    def observe(self, v: float) -> None: ...
    def set(self, v: float) -> None: ...


class Metrics:
    """All Prometheus metrics live on one object so we control registration."""

    def __init__(self) -> None:
        if _PROM_AVAILABLE:
            self.registry = CollectorRegistry()
            self.tool_calls = Counter(
                "repo2rocm_tool_calls_total",
                "Tool invocations by name and outcome",
                ["tool", "outcome"],
                registry=self.registry,
            )
            self.tool_latency = Histogram(
                "repo2rocm_tool_latency_seconds",
                "Tool call latency",
                ["tool"],
                buckets=_DEFAULT_LATENCY_BUCKETS,
                registry=self.registry,
            )
            self.tool_result_bytes = Histogram(
                "repo2rocm_tool_result_bytes",
                "Bytes returned by tool calls",
                ["tool"],
                buckets=(64, 256, 1_024, 4_096, 16_384, 65_536, 262_144, 1_048_576, 4_194_304),
                registry=self.registry,
            )
            self.turn_latency = Histogram(
                "repo2rocm_turn_latency_seconds",
                "Full agent turn latency",
                ["agent_type"],
                buckets=_DEFAULT_LATENCY_BUCKETS,
                registry=self.registry,
            )
            self.llm_tokens = Histogram(
                "repo2rocm_llm_tokens",
                "LLM token usage per request",
                ["model", "kind"],  # kind: input | output | cache_read | cache_creation
                buckets=_TOKEN_BUCKETS,
                registry=self.registry,
            )
            self.llm_latency = Histogram(
                "repo2rocm_llm_latency_seconds",
                "LLM request latency",
                ["model"],
                buckets=_DEFAULT_LATENCY_BUCKETS,
                registry=self.registry,
            )
            self.llm_errors = Counter(
                "repo2rocm_llm_errors_total",
                "LLM request errors",
                ["model", "error_class"],
                registry=self.registry,
            )
            self.subagent_active = Gauge(
                "repo2rocm_subagents_active",
                "Currently running sub-agents",
                ["agent_type"],
                registry=self.registry,
            )
            self.subagent_completions = Counter(
                "repo2rocm_subagent_completions_total",
                "Sub-agent completions by terminal reason",
                ["agent_type", "reason"],
                registry=self.registry,
            )
            self.permission_decisions = Counter(
                "repo2rocm_permission_decisions_total",
                "Permission resolution outcomes",
                ["tool", "mode", "decision"],
                registry=self.registry,
            )
            self.hook_invocations = Counter(
                "repo2rocm_hook_invocations_total",
                "Hook invocations",
                ["event", "outcome"],
                registry=self.registry,
            )
            self.sandbox_ops = Counter(
                "repo2rocm_sandbox_ops_total",
                "Sandbox container operations",
                ["op", "outcome"],
                registry=self.registry,
            )
            self.migration_outcomes = Counter(
                "repo2rocm_migration_outcomes_total",
                "End-to-end migration outcomes",
                ["mode", "outcome"],
                registry=self.registry,
            )
            self.context_compactions = Counter(
                "repo2rocm_context_compactions_total",
                "Context-pipeline compaction events",
                ["layer", "outcome"],
                registry=self.registry,
            )
            self.cache_hit_ratio = Gauge(
                "repo2rocm_prompt_cache_hit_ratio",
                "Rolling prompt-cache hit ratio (cache_read / (cache_read + cache_creation + input))",
                ["model"],
                registry=self.registry,
            )
        else:  # pragma: no cover
            noop = _NoOpMetric()
            self.registry = None  # type: ignore[assignment]
            self.tool_calls = noop  # type: ignore[assignment]
            self.tool_latency = noop  # type: ignore[assignment]
            self.tool_result_bytes = noop  # type: ignore[assignment]
            self.turn_latency = noop  # type: ignore[assignment]
            self.llm_tokens = noop  # type: ignore[assignment]
            self.llm_latency = noop  # type: ignore[assignment]
            self.llm_errors = noop  # type: ignore[assignment]
            self.subagent_active = noop  # type: ignore[assignment]
            self.subagent_completions = noop  # type: ignore[assignment]
            self.permission_decisions = noop  # type: ignore[assignment]
            self.hook_invocations = noop  # type: ignore[assignment]
            self.sandbox_ops = noop  # type: ignore[assignment]
            self.migration_outcomes = noop  # type: ignore[assignment]
            self.context_compactions = noop  # type: ignore[assignment]
            self.cache_hit_ratio = noop  # type: ignore[assignment]

        self._http_started = False
        self._http_lock = threading.Lock()

    def start_http_endpoint(self, port: int = 9464) -> None:
        """Start a `/metrics` HTTP endpoint on `port`. No-op if Prometheus missing."""
        if not _PROM_AVAILABLE:
            return
        with self._http_lock:
            if self._http_started:
                return
            try:
                start_http_server(port, registry=self.registry)
                self._http_started = True
            except OSError:
                # port already in use — fine for tests
                pass

    @contextmanager
    def time_tool(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.tool_latency.labels(tool=name).observe(time.perf_counter() - start)

    @contextmanager
    def time_turn(self, agent_type: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.turn_latency.labels(agent_type=agent_type).observe(time.perf_counter() - start)

    @contextmanager
    def time_llm(self, model: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.llm_latency.labels(model=model).observe(time.perf_counter() - start)


METRICS = Metrics()
