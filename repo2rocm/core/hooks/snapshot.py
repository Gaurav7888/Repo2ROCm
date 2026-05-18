"""Capture hooks once at startup; never re-read from disk.

This is the "frozen at trust boundaries" pattern from Ch. 12 of the Claude Code book.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# Internal callbacks: registered programmatically; do not require a subprocess.
CallbackHook = Callable[[str, dict[str, Any]], dict[str, Any] | None]


@dataclass
class CommandHookSpec:
    """A shell-command hook from a settings.json file."""

    command: str
    matcher_tool: str | None = None  # e.g. "Bash"; None matches any tool
    matcher_if: str | None = None  # e.g. "Bash(git commit*)"
    once: bool = False
    source: str = "userSettings"  # for trust precedence
    timeout_s: float = 30.0


@dataclass
class HooksSnapshot:
    """Frozen-at-startup hook config."""

    pre_tool_use: list[CommandHookSpec] = field(default_factory=list)
    post_tool_use: list[CommandHookSpec] = field(default_factory=list)
    stop: list[CommandHookSpec] = field(default_factory=list)
    user_prompt_submit: list[CommandHookSpec] = field(default_factory=list)
    session_start: list[CommandHookSpec] = field(default_factory=list)

    # Internal callbacks — keyed by event name
    callbacks: dict[str, list[CallbackHook]] = field(default_factory=dict)

    # Session-scoped hooks (from skills); cleared between sessions
    session_scoped: dict[str, list[CommandHookSpec]] = field(default_factory=dict)

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _captured: bool = False

    def register_callback(self, event: str, cb: CallbackHook) -> None:
        with self._lock:
            self.callbacks.setdefault(event, []).append(cb)

    def add_session_hook(self, event: str, spec: CommandHookSpec) -> None:
        with self._lock:
            self.session_scoped.setdefault(event, []).append(spec)

    def remove_once_hook(self, event: str, spec: CommandHookSpec) -> None:
        with self._lock:
            hooks = self.session_scoped.get(event, [])
            self.session_scoped[event] = [h for h in hooks if h is not spec]

    def for_event(self, event: str) -> list[CommandHookSpec]:
        attr = {
            "PreToolUse": "pre_tool_use",
            "PostToolUse": "post_tool_use",
            "Stop": "stop",
            "UserPromptSubmit": "user_prompt_submit",
            "SessionStart": "session_start",
        }.get(event)
        with self._lock:
            base = list(getattr(self, attr)) if attr else []
            base.extend(self.session_scoped.get(event, []))
            return base


_GLOBAL_SNAPSHOT: HooksSnapshot | None = None


def capture_hooks_snapshot(
    *,
    user_settings: Path | None = None,
    project_settings: Path | None = None,
    local_settings: Path | None = None,
    policy_settings: Path | None = None,
) -> HooksSnapshot:
    """Read hook configs once, return a frozen snapshot."""
    global _GLOBAL_SNAPSHOT

    snap = HooksSnapshot()
    for source_path, source_name in [
        (policy_settings, "policySettings"),
        (user_settings, "userSettings"),
        (project_settings, "projectSettings"),
        (local_settings, "localSettings"),
    ]:
        if source_path is None or not source_path.exists():
            continue
        try:
            cfg = json.loads(source_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for event_name, specs in (cfg.get("hooks") or {}).items():
            attr = {
                "PreToolUse": "pre_tool_use",
                "PostToolUse": "post_tool_use",
                "Stop": "stop",
                "UserPromptSubmit": "user_prompt_submit",
                "SessionStart": "session_start",
            }.get(event_name)
            if attr is None:
                continue
            target = getattr(snap, attr)
            for s in specs:
                for h in s.get("hooks", []):
                    target.append(
                        CommandHookSpec(
                            command=h.get("command", ""),
                            matcher_tool=s.get("matcher"),
                            matcher_if=h.get("if"),
                            once=bool(h.get("once", False)),
                            source=source_name,
                            timeout_s=float(h.get("timeout_s", 30.0)),
                        )
                    )

    snap._captured = True
    _GLOBAL_SNAPSHOT = snap
    return snap


def get_snapshot() -> HooksSnapshot:
    """Return the captured snapshot, or an empty one if `capture_hooks_snapshot` wasn't called."""
    if _GLOBAL_SNAPSHOT is None:
        return HooksSnapshot()
    return _GLOBAL_SNAPSHOT
