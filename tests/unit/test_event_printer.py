"""EventPrinter — converts the agent's LoopEvent stream into a pretty console feed."""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

from rich.console import Console

from repo2rocm.agents.lifecycle import _AgentStartEvent
from repo2rocm.core.api import ChunkText, ChunkToolUse, ChunkUsage, ChunkError
from repo2rocm.core.messages import TokenUsage, ToolUseBlock
from repo2rocm.core.query import LoopEvent
from repo2rocm.ui.event_printer import EventPrinter


def _capturing_console() -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return Console(file=buf, force_terminal=False, width=120), buf


@dataclass
class _FakeTool:
    name: str = "Read"


@dataclass
class _FakeResult:
    text: str
    is_error: bool = False


@dataclass
class _FakeTracked:
    tool: _FakeTool
    result: _FakeResult


def _stamp(ev: LoopEvent, agent_id: str = "a1", agent_type: str = "configuration") -> LoopEvent:
    # The lifecycle hook normally stamps these; do it here for the test.
    setattr(ev, "_agent_id", agent_id)
    setattr(ev, "_agent_type", agent_type)
    return ev


def test_agent_start_prints_header():
    console, buf = _capturing_console()
    p = EventPrinter(console)
    p(_AgentStartEvent(agent_id="a1", agent_type="configuration", permission_mode="bypassPermissions"))
    out = buf.getvalue()
    assert "agent: configuration" in out
    assert "a1" in out
    assert "mode=bypassPermissions" in out


def test_text_chunks_buffered_then_flushed_with_turn_header():
    console, buf = _capturing_console()
    p = EventPrinter(console)
    p(_AgentStartEvent(agent_id="a1", agent_type="configuration", permission_mode="bypassPermissions"))
    p(_stamp(LoopEvent("text", ChunkText(text="reading the ", block_index=0))))
    p(_stamp(LoopEvent("text", ChunkText(text="README first.", block_index=0))))
    # flush happens on tool_use OR usage:
    p(_stamp(LoopEvent("tool_use", ChunkToolUse(
        tool_use=ToolUseBlock(id="t1", name="Read", input={"file_path": "README.md"}),
        block_index=1,
    ))))
    out = buf.getvalue()
    # turn-header + concatenated text on one line
    assert "T 0" in out
    assert "reading the README first." in out
    # tool call on its own line
    assert "Read" in out
    assert "README.md" in out


def test_tool_result_ok_and_error():
    console, buf = _capturing_console()
    p = EventPrinter(console)
    p(_AgentStartEvent(agent_id="a1", agent_type="configuration", permission_mode="bypassPermissions"))
    p(_stamp(LoopEvent("tool_result", _FakeTracked(_FakeTool("Read"), _FakeResult("hello\n")))))
    p(_stamp(LoopEvent("tool_result", _FakeTracked(_FakeTool("DockerExec"), _FakeResult("boom", is_error=True)))))
    out = buf.getvalue()
    assert "Read" in out and "ok" in out
    assert "DockerExec" in out and "error" in out


def test_usage_prints_cache_hit_ratio():
    console, buf = _capturing_console()
    p = EventPrinter(console)
    p(_AgentStartEvent(agent_id="a1", agent_type="configuration", permission_mode="bypassPermissions"))
    p(_stamp(LoopEvent("usage", ChunkUsage(usage=TokenUsage(
        input_tokens=120,
        output_tokens=30,
        cache_read_input_tokens=400,
        cache_creation_input_tokens=80,
    )))))
    out = buf.getvalue()
    assert "in=" in out and "out=" in out and "cache_read=" in out
    # 400 / (120+80+400) = 67% cache hit
    assert "hit=67%" in out or "hit=66%" in out


def test_error_event_is_printed_with_class_and_message():
    console, buf = _capturing_console()
    p = EventPrinter(console)
    p(_AgentStartEvent(agent_id="a1", agent_type="configuration", permission_mode="bypassPermissions"))
    p(_stamp(LoopEvent("error", ChunkError(error_class="http_429", message="rate limited"))))
    out = buf.getvalue()
    assert "ERROR" in out
    assert "http_429" in out
    assert "rate limited" in out


def test_unknown_event_kind_is_ignored_silently():
    """Printer must never raise — bad events become no-ops."""
    console, buf = _capturing_console()
    p = EventPrinter(console)
    # event with no kind
    p(LoopEvent("totally_unknown", {"x": 1}))
    # event with the right kind but wrong payload shape — must not crash
    p(LoopEvent("text", None))
    p(LoopEvent("tool_use", None))
    p(LoopEvent("usage", None))
    p(LoopEvent("error", None))
    # printer survived; nothing meaningful printed (no AttributeError)
