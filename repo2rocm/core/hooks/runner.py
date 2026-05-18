"""Run hooks for a given event. Aggregate results with the deny>ask>allow precedence."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from repo2rocm.core.hooks.snapshot import CommandHookSpec, HooksSnapshot, get_snapshot
from repo2rocm.core.permissions import (
    PermissionDecision,
    PermissionDecisionKind,
    allow,
    deny,
)
from repo2rocm.observability.logging import get_logger
from repo2rocm.observability.metrics import METRICS
from repo2rocm.observability.tracing import span

log = get_logger(__name__)


class HookEvent(str, Enum):
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    STOP = "Stop"
    SUBAGENT_STOP = "SubagentStop"
    SESSION_START = "SessionStart"
    USER_PROMPT_SUBMIT = "UserPromptSubmit"


@dataclass
class HookOutcome:
    """Aggregated result of running all hooks for an event."""

    permission: PermissionDecision = field(default_factory=allow)
    updated_input: dict[str, Any] | None = None
    additional_context: str = ""
    blocked: bool = False
    block_reason: str = ""
    prevent_continuation: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


async def execute_hooks(
    *,
    event: HookEvent,
    input_data: dict[str, Any],
    snapshot: HooksSnapshot | None = None,
    workspace_trusted: bool = True,
) -> HookOutcome:
    """Run all matching hooks in parallel; aggregate per deny>ask>allow precedence."""
    if not workspace_trusted:
        # Hook runner gate (Ch. 12 trust check).
        return HookOutcome()

    snap = snapshot or get_snapshot()

    # Run command + callback hooks together
    cmd_hooks = snap.for_event(event.value)
    cb_hooks = snap.callbacks.get(event.value, [])

    if not cmd_hooks and not cb_hooks:
        return HookOutcome()

    with span("hooks.execute", event=event.value, count=len(cmd_hooks) + len(cb_hooks)):
        # Fast path: all internal callbacks → no subprocess overhead
        if not cmd_hooks:
            return _aggregate_callback_results(cb_hooks, event, input_data)

        # Filter command hooks by matcher
        matched_cmds = [h for h in cmd_hooks if _matches(h, event, input_data)]

        # Launch in parallel
        coros = [
            _run_command_hook(h, event, input_data) for h in matched_cmds
        ]
        for cb in cb_hooks:
            coros.append(_run_callback_hook(cb, event, input_data))

        results = await asyncio.gather(*coros, return_exceptions=True)
        once_to_remove = [h for h in matched_cmds if h.once]
        for h in once_to_remove:
            snap.remove_once_hook(event.value, h)

        return _aggregate(results)


# ── Matchers ──────────────────────────────────────────────────────────────────


def _matches(hook: CommandHookSpec, event: HookEvent, input_data: dict[str, Any]) -> bool:
    if hook.matcher_tool and event in (HookEvent.PRE_TOOL_USE, HookEvent.POST_TOOL_USE):
        if input_data.get("tool_name") != hook.matcher_tool:
            return False
    if hook.matcher_if:
        # very simple matcher: "Bash(git *)" → tool=Bash, prefix=git
        if "(" in hook.matcher_if and hook.matcher_if.endswith(")"):
            tool, _, rest = hook.matcher_if.partition("(")
            inner = rest[:-1]
            if input_data.get("tool_name") != tool:
                return False
            if inner.endswith("*"):
                prefix = inner[:-1]
                cmd_str = ""
                for k in ("command", "cmd"):
                    v = input_data.get("tool_input", {}).get(k)
                    if isinstance(v, str):
                        cmd_str = v
                        break
                if not cmd_str.startswith(prefix):
                    return False
    return True


# ── Runners ───────────────────────────────────────────────────────────────────


async def _run_command_hook(
    hook: CommandHookSpec,
    event: HookEvent,
    input_data: dict[str, Any],
) -> HookOutcome:
    payload = json.dumps({"event": event.value, **input_data}).encode()
    try:
        proc = await asyncio.create_subprocess_shell(
            hook.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(payload), timeout=hook.timeout_s
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            METRICS.hook_invocations.labels(event=event.value, outcome="timeout").inc()
            log.warning("hook timed out", command=hook.command, timeout_s=hook.timeout_s)
            return HookOutcome(extra={"timeout": True})
    except FileNotFoundError:
        METRICS.hook_invocations.labels(event=event.value, outcome="not_found").inc()
        return HookOutcome()

    rc = proc.returncode or 0
    out = stdout.decode("utf-8", errors="replace")
    err = stderr.decode("utf-8", errors="replace")

    outcome = HookOutcome()
    # Try parse stdout as JSON for structured signals (permissionBehavior, updatedInput…)
    try:
        body = json.loads(out)
    except (json.JSONDecodeError, ValueError):
        body = None

    if rc == 0:
        METRICS.hook_invocations.labels(event=event.value, outcome="pass").inc()
        if isinstance(body, dict):
            pb = body.get("permissionBehavior")
            if pb == "allow":
                outcome.permission = allow("hook allow")
            elif pb == "deny":
                outcome.permission = deny(body.get("reason", "hook deny"))
            if "updatedInput" in body:
                outcome.updated_input = body["updatedInput"]
            if "additionalContext" in body:
                outcome.additional_context = str(body["additionalContext"])
            if body.get("preventContinuation"):
                outcome.prevent_continuation = True
        return outcome

    if rc == 2:
        METRICS.hook_invocations.labels(event=event.value, outcome="block").inc()
        outcome.blocked = True
        outcome.block_reason = err.strip() or out.strip() or f"hook exit 2: {hook.command}"
        outcome.permission = deny(outcome.block_reason)
        return outcome

    # any other code: non-blocking warning
    METRICS.hook_invocations.labels(event=event.value, outcome="warn").inc()
    outcome.additional_context = err.strip() or out.strip()
    return outcome


async def _run_callback_hook(
    cb: Any, event: HookEvent, input_data: dict[str, Any]
) -> HookOutcome:
    try:
        if asyncio.iscoroutinefunction(cb):
            result = await cb(event.value, input_data)
        else:
            result = cb(event.value, input_data)
    except Exception as exc:  # noqa: BLE001
        METRICS.hook_invocations.labels(event=event.value, outcome="error").inc()
        log.warning("callback hook raised", error=str(exc))
        return HookOutcome()

    METRICS.hook_invocations.labels(event=event.value, outcome="pass").inc()
    if not isinstance(result, dict):
        return HookOutcome()

    outcome = HookOutcome()
    pb = result.get("permissionBehavior")
    if pb == "allow":
        outcome.permission = allow(result.get("reason", "cb allow"))
    elif pb == "deny":
        outcome.permission = deny(result.get("reason", "cb deny"))
    if "updatedInput" in result:
        outcome.updated_input = result["updatedInput"]
    if "additionalContext" in result:
        outcome.additional_context = str(result["additionalContext"])
    if result.get("preventContinuation"):
        outcome.prevent_continuation = True
    return outcome


def _aggregate_callback_results(
    cb_hooks: list[Any], event: HookEvent, input_data: dict[str, Any]
) -> HookOutcome:
    # synchronous fast-path; for the simple case where all hooks are sync callbacks
    outcomes: list[HookOutcome] = []
    for cb in cb_hooks:
        try:
            r = cb(event.value, input_data)
        except Exception:
            continue
        outcome = HookOutcome()
        if isinstance(r, dict):
            pb = r.get("permissionBehavior")
            if pb == "allow":
                outcome.permission = allow(r.get("reason", "cb allow"))
            elif pb == "deny":
                outcome.permission = deny(r.get("reason", "cb deny"))
        outcomes.append(outcome)
    return _aggregate(outcomes)


def _aggregate(results: list[Any]) -> HookOutcome:
    """deny > ask > allow precedence."""
    final = HookOutcome()
    contexts: list[str] = []
    updated_input: dict[str, Any] | None = None
    for r in results:
        if isinstance(r, BaseException):
            continue
        if not isinstance(r, HookOutcome):
            continue
        if r.blocked:
            final.blocked = True
            final.block_reason = r.block_reason
        if r.permission.kind == PermissionDecisionKind.DENY:
            final.permission = r.permission
        elif (
            r.permission.kind == PermissionDecisionKind.ASK
            and final.permission.kind != PermissionDecisionKind.DENY
        ):
            final.permission = r.permission
        if r.prevent_continuation:
            final.prevent_continuation = True
        if r.updated_input is not None:
            updated_input = r.updated_input
        if r.additional_context:
            contexts.append(r.additional_context)
    final.updated_input = updated_input
    final.additional_context = "\n".join(contexts)
    return final
