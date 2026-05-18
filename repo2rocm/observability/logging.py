"""Structured logging with structlog.

Every log line is JSON-shaped with trace_id and agent_id bound where available.
A Rich-based console renderer is used in TTYs for human readability.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any

try:
    import structlog

    _STRUCTLOG = True
except Exception:  # pragma: no cover
    _STRUCTLOG = False

_configured = False


def configure_logging(*, level: str = "INFO", json_output: bool | None = None) -> None:
    """Idempotent logging configuration."""
    global _configured
    if _configured:
        return
    _configured = True

    level_num = getattr(logging, level.upper(), logging.INFO)

    # Default to JSON when not attached to a TTY (CI, batch runs).
    if json_output is None:
        json_output = not sys.stderr.isatty() or bool(os.environ.get("REPO2ROCM_LOG_JSON"))

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=level_num,
    )

    if not _STRUCTLOG:
        return

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        _trace_id_processor,
        structlog.processors.StackInfoRenderer(),
    ]
    if json_output:
        # JSONRenderer wants exc_info pre-formatted; ConsoleRenderer formats it itself.
        processors.append(structlog.processors.format_exc_info)
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level_num),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def _trace_id_processor(_logger: Any, _name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    from repo2rocm.observability.tracing import current_trace_id

    tid = current_trace_id()
    if tid:
        event_dict.setdefault("trace_id", tid)
    return event_dict


def get_logger(name: str) -> Any:
    """Return a structlog logger (or fall back to stdlib)."""
    if _STRUCTLOG:
        return structlog.get_logger(name)
    return logging.getLogger(name)
