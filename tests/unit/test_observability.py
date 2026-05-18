"""Observability primitives are import-safe and produce records."""
from __future__ import annotations

from pathlib import Path

from repo2rocm.observability import (
    METRICS,
    Transcript,
    TranscriptStore,
    checkpoint,
    setup_observability,
    span,
)
from repo2rocm.observability.checkpoints import get_registry


def test_setup_is_idempotent():
    setup_observability(service_name="test")
    setup_observability(service_name="test")  # no error


def test_checkpoints_recorded():
    get_registry().reset()
    checkpoint("a")
    checkpoint("b")
    rs = get_registry().summary()
    names = [r["name"] for r in rs]
    assert "a" in names
    assert "b" in names


def test_span_is_no_op_safe():
    with span("test.span", x=1, y="z"):
        pass  # should not raise even without OTel


def test_transcript_writes_jsonl(tmp_path: Path):
    t = Transcript(tmp_path / "x.jsonl")
    uid = t.append({"kind": "test", "data": 1})
    uid2 = t.append({"kind": "test", "data": 2})
    assert uid and uid2 and uid != uid2
    records = t.read_all()
    assert len(records) == 2
    assert records[1]["parent_uuid"] == records[0]["uuid"]


def test_transcript_store_per_agent(tmp_path: Path):
    store = TranscriptStore(tmp_path, session_id="sess1")
    t1 = store.transcript("agent_a")
    t2 = store.transcript("agent_b")
    assert t1.path != t2.path
    t1.append({"kind": "x"})
    assert t1.record_count == 1


def test_metrics_labels_dont_throw():
    METRICS.tool_calls.labels(tool="t1", outcome="ok").inc()
    METRICS.tool_latency.labels(tool="t1").observe(0.05)
    METRICS.subagent_completions.labels(agent_type="x", reason="completed").inc()
