"""OpenTelemetry-backed tracing.

We want trace coverage on every boundary that costs time or tokens:
  * agent loop turns
  * model API calls (with token usage as span attributes)
  * tool calls (with input/output sizes)
  * sub-agent lifecycles (15 steps)
  * sandbox container ops (docker exec, commit, rollback)
  * MCP roundtrips

OpenTelemetry deps are *optional* — if missing, all helpers degrade to no-ops so that
unit tests do not need the full OTel stack.
"""
from __future__ import annotations

import contextlib
import os
import threading
from collections.abc import Iterator
from typing import Any

try:  # pragma: no cover — exercised only when otel is installed
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )
    from opentelemetry.trace import Span, Status, StatusCode

    _OTEL_AVAILABLE = True
except Exception:  # pragma: no cover
    trace = None  # type: ignore[assignment]
    Span = object  # type: ignore[assignment, misc]
    _OTEL_AVAILABLE = False

_lock = threading.Lock()
_initialized = False
_tracer: Any = None


def setup_observability(
    *,
    service_name: str = "repo2rocm",
    otlp_endpoint: str | None = None,
    console: bool = False,
    sampling_ratio: float = 1.0,  # 100% by default; Repo2ROCm runs in trusted envs
) -> None:
    """Initialize OTel tracer provider. Idempotent."""
    global _initialized, _tracer

    with _lock:
        if _initialized:
            return
        if not _OTEL_AVAILABLE:
            _initialized = True
            return

        resource = Resource.create(
            {
                "service.name": service_name,
                "service.version": _read_version(),
            }
        )
        provider = TracerProvider(resource=resource)

        if console or os.environ.get("REPO2ROCM_OTEL_CONSOLE"):
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

        endpoint = otlp_endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        if endpoint:
            try:  # pragma: no cover — requires otlp dep
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter,
                )

                provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
            except Exception:
                pass

        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(service_name)
        _initialized = True

        # eagerly initialize metrics + logging so all subsystems are ready
        from repo2rocm.observability.metrics import METRICS  # noqa: F401
        from repo2rocm.observability.logging import configure_logging

        configure_logging()


def get_tracer() -> Any:
    """Return the active tracer (or a no-op if OTel disabled)."""
    if not _OTEL_AVAILABLE or _tracer is None:
        return _NoOpTracer()
    return _tracer


def current_trace_id() -> str | None:
    """Return the current trace id as hex, or None if unavailable."""
    if not _OTEL_AVAILABLE:
        return None
    try:  # pragma: no cover — best effort
        ctx = trace.get_current_span().get_span_context()
        if ctx.is_valid:
            return f"{ctx.trace_id:032x}"
    except Exception:
        pass
    return None


@contextlib.contextmanager
def span(name: str, /, **attributes: Any) -> Iterator[Any]:
    """Context manager wrapping `tracer.start_as_current_span`.

    Usage:
        with span("agent.turn", turn=3, agent_id="a7j3n9p2"):
            ...
    """
    tracer = get_tracer()
    if isinstance(tracer, _NoOpTracer):
        yield _NoOpSpan()
        return

    with tracer.start_as_current_span(name) as sp:  # type: ignore[union-attr]
        for k, v in attributes.items():
            try:
                sp.set_attribute(k, _coerce_attr(v))
            except Exception:
                pass
        try:
            yield sp
        except Exception as exc:
            try:
                sp.set_status(Status(StatusCode.ERROR, str(exc)))
                sp.record_exception(exc)
            except Exception:
                pass
            raise


def _coerce_attr(v: Any) -> Any:
    if isinstance(v, (str, bool, int, float)):
        return v
    if isinstance(v, (list, tuple)) and all(isinstance(x, (str, bool, int, float)) for x in v):
        return list(v)
    return str(v)[:1024]


def _read_version() -> str:
    try:
        from repo2rocm import __version__

        return __version__
    except Exception:
        return "0.0.0"


# ── No-op fallbacks ────────────────────────────────────────────────────────────


class _NoOpSpan:
    def set_attribute(self, *a: Any, **kw: Any) -> None: ...
    def set_status(self, *a: Any, **kw: Any) -> None: ...
    def record_exception(self, *a: Any, **kw: Any) -> None: ...
    def __enter__(self) -> _NoOpSpan:
        return self

    def __exit__(self, *a: Any) -> None: ...


class _NoOpTracer:
    @contextlib.contextmanager
    def start_as_current_span(self, name: str) -> Iterator[_NoOpSpan]:
        yield _NoOpSpan()
