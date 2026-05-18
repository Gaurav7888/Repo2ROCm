"""Discriminated union of *every* possible way the agent loop can stop or continue.

Per Claude Code's design (`@claude-code-from-source/book/ch05-agent-loop.md`), making this
type explicit is what eliminates an entire class of "why did the agent stop?" bugs.
"""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from repo2rocm.core.messages import TokenUsage


class _TerminalBase(BaseModel):
    model_config = ConfigDict(extra="ignore")
    usage: TokenUsage = TokenUsage()
    turns: int = 0


# ── 10 terminal reasons ────────────────────────────────────────────────────────


class Completed(_TerminalBase):
    reason: Literal["completed"] = "completed"
    final_text: str = ""


class MaxTurns(_TerminalBase):
    reason: Literal["max_turns"] = "max_turns"


class AbortedStreaming(_TerminalBase):
    reason: Literal["aborted_streaming"] = "aborted_streaming"


class AbortedTools(_TerminalBase):
    reason: Literal["aborted_tools"] = "aborted_tools"


class BlockingLimit(_TerminalBase):
    reason: Literal["blocking_limit"] = "blocking_limit"


class PromptTooLong(_TerminalBase):
    reason: Literal["prompt_too_long"] = "prompt_too_long"
    message: str = ""


class ImageError(_TerminalBase):
    reason: Literal["image_error"] = "image_error"
    message: str = ""


class ModelError(_TerminalBase):
    reason: Literal["model_error"] = "model_error"
    message: str = ""
    error_class: str = "unknown"


class StopHookPrevented(_TerminalBase):
    reason: Literal["stop_hook_prevented"] = "stop_hook_prevented"
    hook_name: str = ""


class HookStopped(_TerminalBase):
    reason: Literal["hook_stopped"] = "hook_stopped"
    hook_name: str = ""


Terminal = Annotated[
    Union[
        Completed,
        MaxTurns,
        AbortedStreaming,
        AbortedTools,
        BlockingLimit,
        PromptTooLong,
        ImageError,
        ModelError,
        StopHookPrevented,
        HookStopped,
    ],
    Field(discriminator="reason"),
]


# ── 7 continuation reasons (recorded on State.transition) ──────────────────────


class _ContinueBase(BaseModel):
    model_config = ConfigDict(extra="ignore")


class ContinueNextTurn(_ContinueBase):
    reason: Literal["next_turn"] = "next_turn"


class ContinueCollapseDrainRetry(_ContinueBase):
    reason: Literal["collapse_drain_retry"] = "collapse_drain_retry"


class ContinueReactiveCompactRetry(_ContinueBase):
    reason: Literal["reactive_compact_retry"] = "reactive_compact_retry"


class ContinueMaxOutputTokensEscalate(_ContinueBase):
    reason: Literal["max_output_tokens_escalate"] = "max_output_tokens_escalate"


class ContinueMaxOutputTokensRecovery(_ContinueBase):
    reason: Literal["max_output_tokens_recovery"] = "max_output_tokens_recovery"
    attempt: int = 1


class ContinueStopHookBlocking(_ContinueBase):
    reason: Literal["stop_hook_blocking"] = "stop_hook_blocking"


class ContinueTokenBudgetContinuation(_ContinueBase):
    reason: Literal["token_budget_continuation"] = "token_budget_continuation"


Continue = Annotated[
    Union[
        ContinueNextTurn,
        ContinueCollapseDrainRetry,
        ContinueReactiveCompactRetry,
        ContinueMaxOutputTokensEscalate,
        ContinueMaxOutputTokensRecovery,
        ContinueStopHookBlocking,
        ContinueTokenBudgetContinuation,
    ],
    Field(discriminator="reason"),
]
