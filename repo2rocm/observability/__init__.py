"""Observability: OpenTelemetry traces + Prometheus metrics + JSONL transcripts + checkpoints.

Public surface:
  * `setup_observability(...)`   — initialize tracer + meter providers (idempotent).
  * `span(name, **attrs)`        — context manager wrapping `tracer.start_as_current_span`.
  * `checkpoint(name)`           — record a startup-style checkpoint (used by bootstrap).
  * `Metrics`                    — dataclass exposing counters/histograms used across the codebase.
  * `Transcript`                 — append-only JSONL writer per (session, agent).
  * `get_logger(name)`           — structlog logger bound with the current trace_id.

All public API is import-safe even if the OTel / Prometheus deps are missing — fall back to no-ops.
"""
from repo2rocm.observability.tracing import (
    setup_observability,
    span,
    get_tracer,
    current_trace_id,
)
from repo2rocm.observability.metrics import Metrics, METRICS
from repo2rocm.observability.transcripts import Transcript, TranscriptStore
from repo2rocm.observability.checkpoints import checkpoint, CheckpointRegistry
from repo2rocm.observability.logging import get_logger, configure_logging

__all__ = [
    "setup_observability",
    "span",
    "get_tracer",
    "current_trace_id",
    "Metrics",
    "METRICS",
    "Transcript",
    "TranscriptStore",
    "checkpoint",
    "CheckpointRegistry",
    "get_logger",
    "configure_logging",
]
