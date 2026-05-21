from __future__ import annotations

import logging

from repo2rocm.observability import logging as r2r_logging


def test_configure_logging_quiets_http_transport_logs_by_default(monkeypatch):
    monkeypatch.delenv("REPO2ROCM_LOG_HTTP", raising=False)
    monkeypatch.setattr(r2r_logging, "_configured", False)

    logging.getLogger("httpx").setLevel(logging.NOTSET)
    logging.getLogger("httpcore").setLevel(logging.NOTSET)

    r2r_logging.configure_logging()

    assert logging.getLogger("httpx").getEffectiveLevel() == logging.WARNING
    assert logging.getLogger("httpcore").getEffectiveLevel() == logging.WARNING


def test_configure_logging_preserves_http_transport_logs_when_opted_in(monkeypatch):
    monkeypatch.setenv("REPO2ROCM_LOG_HTTP", "1")
    monkeypatch.setattr(r2r_logging, "_configured", False)

    logging.getLogger("httpx").setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.INFO)

    r2r_logging.configure_logging()

    assert logging.getLogger("httpx").getEffectiveLevel() == logging.INFO
    assert logging.getLogger("httpcore").getEffectiveLevel() == logging.INFO
