"""Loop state + latched fields.

The principle (per Claude Code Ch. 5): every `continue` site rebuilds the State object
in full. No `state.x = y` mutations. The verbosity is the feature — each transition
is self-documenting.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace
from typing import Any

from repo2rocm.core.messages import Message, TokenUsage


@dataclass
class LatchedSet:
    """Sticky-on bag of flags. Once a latch flips to True, it stays True for the session.

    Per Ch. 17 of the Claude Code book: this is how we keep the prompt cache stable
    even though the user might toggle features mid-session.
    """

    _flags: dict[str, bool] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def latch(self, name: str) -> None:
        with self._lock:
            self._flags[name] = True

    def get(self, name: str) -> bool:
        with self._lock:
            return self._flags.get(name, False)

    def snapshot(self) -> dict[str, bool]:
        with self._lock:
            return dict(self._flags)


@dataclass(frozen=True)
class LoopState:
    """The complete state of one running agent loop."""

    messages: list[Message]
    turn_count: int = 0
    max_turns: int = 100
    # output token reservation: 8K default → 64K on truncation
    max_output_tokens: int = 8192
    max_output_tokens_recovery_count: int = 0
    has_attempted_reactive_compact: bool = False
    consecutive_compact_failures: int = 0
    usage: TokenUsage = field(default_factory=TokenUsage)
    pending_tool_use_summary: Any | None = None
    stop_hook_active: bool = False
    transition_reason: str = "initial"
    latches: LatchedSet = field(default_factory=LatchedSet)

    @classmethod
    def init(cls, messages: list[Message], *, max_turns: int = 100) -> LoopState:
        return cls(messages=list(messages), max_turns=max_turns)

    def with_(
        self,
        *,
        messages: list[Message] | None = None,
        turn_count: int | None = None,
        max_output_tokens: int | None = None,
        max_output_tokens_recovery_count: int | None = None,
        has_attempted_reactive_compact: bool | None = None,
        consecutive_compact_failures: int | None = None,
        usage: TokenUsage | None = None,
        stop_hook_active: bool | None = None,
        transition_reason: str | None = None,
    ) -> LoopState:
        """Explicit `replace`-with-named-kwargs to keep every transition readable."""
        return replace(
            self,
            **{
                k: v
                for k, v in {
                    "messages": messages,
                    "turn_count": turn_count,
                    "max_output_tokens": max_output_tokens,
                    "max_output_tokens_recovery_count": max_output_tokens_recovery_count,
                    "has_attempted_reactive_compact": has_attempted_reactive_compact,
                    "consecutive_compact_failures": consecutive_compact_failures,
                    "usage": usage,
                    "stop_hook_active": stop_hook_active,
                    "transition_reason": transition_reason,
                }.items()
                if v is not None
            },
        )
