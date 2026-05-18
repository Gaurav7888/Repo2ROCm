"""State + LatchedSet."""
from __future__ import annotations

from repo2rocm.core.messages import UserMessage
from repo2rocm.core.state import LatchedSet, LoopState


def test_latched_set_is_sticky():
    L = LatchedSet()
    assert L.get("foo") is False
    L.latch("foo")
    assert L.get("foo") is True
    # latch is idempotent
    L.latch("foo")
    assert L.snapshot() == {"foo": True}


def test_loop_state_init_is_frozen():
    s = LoopState.init([UserMessage(content="hi")], max_turns=42)
    assert s.turn_count == 0
    assert s.max_turns == 42
    # frozen dataclass: replacement is explicit
    s2 = s.with_(turn_count=1)
    assert s.turn_count == 0
    assert s2.turn_count == 1
