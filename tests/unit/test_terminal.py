"""Terminal discriminated unions."""
from __future__ import annotations

from pydantic import TypeAdapter

from repo2rocm.core.messages import TokenUsage
from repo2rocm.core.terminal import (
    Completed,
    MaxTurns,
    ModelError,
    Terminal,
)


def test_completed_has_reason_literal():
    c = Completed(turns=5, final_text="done", usage=TokenUsage(input_tokens=1, output_tokens=1))
    assert c.reason == "completed"
    assert c.turns == 5


def test_terminal_discriminator_round_trip():
    adapter = TypeAdapter(Terminal)
    raw = {"reason": "max_turns", "turns": 100, "usage": {"input_tokens": 0, "output_tokens": 0}}
    obj = adapter.validate_python(raw)
    assert isinstance(obj, MaxTurns)


def test_model_error_carries_class():
    e = ModelError(message="rate limited", error_class="http_429")
    assert e.error_class == "http_429"
