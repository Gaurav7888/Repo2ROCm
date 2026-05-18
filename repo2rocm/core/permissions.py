"""Permission system. Six modes, single resolution chain.

The Claude Code lesson (Ch. 6): every tool call goes through the same chain so the
system's security posture is reasoned about by knowing which mode is active.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from repo2rocm.observability.metrics import METRICS

if TYPE_CHECKING:
    from repo2rocm.tools.base import BaseTool, ToolUseContext


class PermissionMode(str, Enum):
    PLAN = "plan"
    DEFAULT = "default"
    ACCEPT_EDITS = "acceptEdits"
    AUTO = "auto"
    BYPASS = "bypassPermissions"
    BUBBLE = "bubble"


class PermissionDecisionKind(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"
    PASSTHROUGH = "passthrough"


@dataclass
class PermissionDecision:
    kind: PermissionDecisionKind
    reason: str = ""
    updated_input: dict[str, Any] | None = None


def allow(reason: str = "") -> PermissionDecision:
    return PermissionDecision(PermissionDecisionKind.ALLOW, reason)


def deny(reason: str) -> PermissionDecision:
    return PermissionDecision(PermissionDecisionKind.DENY, reason)


def passthrough() -> PermissionDecision:
    return PermissionDecision(PermissionDecisionKind.PASSTHROUGH)


# A rule pattern matcher: e.g. matches "Bash(git *)" against "git status"
@dataclass
class PermissionRule:
    tool: str
    behavior: PermissionDecisionKind
    pattern: str | None = None
    source: str = "session"

    def matches(self, tool_name: str, input_data: dict[str, Any]) -> bool:
        if tool_name != self.tool:
            return False
        if not self.pattern:
            return True
        # very simple glob: only support `*` suffix for now
        if self.pattern.endswith("*"):
            prefix = self.pattern[:-1]
            # try common input fields used as a "command-like" string
            for field_name in ("command", "cmd", "shell", "url"):
                v = input_data.get(field_name)
                if isinstance(v, str) and v.startswith(prefix):
                    return True
        return False


@dataclass
class PermissionRuleSet:
    always_allow: list[PermissionRule]
    always_deny: list[PermissionRule]
    always_ask: list[PermissionRule]

    @classmethod
    def empty(cls) -> PermissionRuleSet:
        return cls([], [], [])


CanUseToolFn = Callable[["BaseTool", dict[str, Any], "ToolUseContext"], PermissionDecision]


def resolve_permission(
    tool: BaseTool,
    parsed_input: dict[str, Any],
    ctx: ToolUseContext,
    *,
    rules: PermissionRuleSet,
    hook_decision: PermissionDecision | None = None,
) -> PermissionDecision:
    """The single resolution chain. Order matters."""
    mode = ctx.permission_mode

    # 1. Hook decision (already computed by hook runner) is final if it says allow/deny.
    if hook_decision is not None and hook_decision.kind in (
        PermissionDecisionKind.ALLOW,
        PermissionDecisionKind.DENY,
    ):
        _record(tool.name, mode, hook_decision.kind, "hook")
        return hook_decision

    # 2. Rule matching: deny > ask > allow.
    for r in rules.always_deny:
        if r.matches(tool.name, parsed_input):
            _record(tool.name, mode, PermissionDecisionKind.DENY, "rule")
            return deny(f"denied by rule from {r.source}: {r.pattern or '*'}")
    for r in rules.always_ask:
        if r.matches(tool.name, parsed_input):
            _record(tool.name, mode, PermissionDecisionKind.ASK, "rule")
            return PermissionDecision(PermissionDecisionKind.ASK, f"ask rule from {r.source}")
    for r in rules.always_allow:
        if r.matches(tool.name, parsed_input):
            _record(tool.name, mode, PermissionDecisionKind.ALLOW, "rule")
            return allow(f"allowed by rule from {r.source}")

    # 3. Tool-specific check.
    tool_check = tool.check_permissions(parsed_input, ctx)
    if tool_check.kind in (PermissionDecisionKind.ALLOW, PermissionDecisionKind.DENY):
        _record(tool.name, mode, tool_check.kind, "tool_specific")
        return tool_check

    # 4. Mode-based default.
    if mode == PermissionMode.BYPASS:
        _record(tool.name, mode, PermissionDecisionKind.ALLOW, "mode_bypass")
        return allow("bypass mode")
    if mode == PermissionMode.PLAN:
        if tool.is_read_only(parsed_input):
            _record(tool.name, mode, PermissionDecisionKind.ALLOW, "mode_plan_readonly")
            return allow("plan mode allows read-only")
        _record(tool.name, mode, PermissionDecisionKind.DENY, "mode_plan_mutation")
        return deny("plan mode denies mutations")
    if mode == PermissionMode.ACCEPT_EDITS:
        # auto-allow edits + reads, ask for everything else
        if tool.is_read_only(parsed_input) or tool.name in {"Edit", "Write", "ApplyDiff"}:
            _record(tool.name, mode, PermissionDecisionKind.ALLOW, "mode_accept_edits")
            return allow("acceptEdits mode")
        _record(tool.name, mode, PermissionDecisionKind.ASK, "mode_accept_edits_ask")
        return PermissionDecision(PermissionDecisionKind.ASK, "acceptEdits asks for non-edit ops")
    if mode == PermissionMode.AUTO:
        # placeholder: in production this would call a small classifier LLM
        _record(tool.name, mode, PermissionDecisionKind.ALLOW, "mode_auto_default_allow")
        return allow("auto mode (classifier not enabled)")
    if mode == PermissionMode.BUBBLE:
        _record(tool.name, mode, PermissionDecisionKind.ASK, "mode_bubble")
        return PermissionDecision(PermissionDecisionKind.ASK, "bubble to parent")

    # default: ask
    _record(tool.name, mode, PermissionDecisionKind.ASK, "mode_default")
    return PermissionDecision(PermissionDecisionKind.ASK, "default mode asks")


def _record(tool: str, mode: PermissionMode, decision: PermissionDecisionKind, where: str) -> None:
    METRICS.permission_decisions.labels(tool=tool, mode=mode.value, decision=decision.value).inc()
