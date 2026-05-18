"""Repo2ROCm v2 — production-grade multi-agent CUDA->ROCm migration system.

Design principles (per `docs/architecture.md`):
  * One agent kernel — `core.query.query()` — shared by every agent type.
  * Self-describing tools (`BaseTool` subclass + Pydantic schema).
  * Permission modes, not scattered permission checks.
  * 4-layer context compression with circuit breakers.
  * File-based memory, LLM recall.
  * Observability at every boundary (OpenTelemetry + Prometheus + JSONL transcripts).
"""

__version__ = "2.0.0a1"
