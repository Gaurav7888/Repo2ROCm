"""Hook system: lifecycle interceptors frozen at startup, runnable inline or via shell.

Per Claude Code Ch. 12:
  * Hook configuration is snapshotted once at startup; subsequent disk changes are ignored.
  * Six hook types: command, prompt, agent, http, callback (internal), function (internal).
  * Five most-used events: PreToolUse, PostToolUse, Stop, SessionStart, UserPromptSubmit.
  * Exit code 2 = blocking; 0 = pass; other = non-blocking warning.
"""
from repo2rocm.core.hooks.snapshot import HooksSnapshot, capture_hooks_snapshot
from repo2rocm.core.hooks.runner import HookEvent, HookOutcome, execute_hooks
from repo2rocm.core.hooks.builtin import register_builtin_hooks

__all__ = [
    "HooksSnapshot",
    "capture_hooks_snapshot",
    "HookEvent",
    "HookOutcome",
    "execute_hooks",
    "register_builtin_hooks",
]
