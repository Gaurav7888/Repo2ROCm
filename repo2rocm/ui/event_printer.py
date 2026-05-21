"""Live console renderer for agent events.

Wires into `RunAgentParams.on_event` and prints every text/tool-use/tool-result
in a compact, color-coded format. Built on Rich so it plays nicely with the
existing CLI output.

The format is the same one we use for tailing JSONL transcripts:

  ── agent: configuration (a07ba1811) mode=bypassPermissions ──
  [T 0] reading the README...
        → Read(file_path="README.md")
        ← ✓ Read              ok    1742b  0.001s
  [T 1] now the requirements file.
        → Read(file_path="requirements.txt")
        ← ✓ Read              ok     638b  0.001s
        usage: in=1240 out=42 cache_hit=0%

  ── final ──
  [completed] turns=14 tokens=in=87k out=2.3k cache_hit=78%
"""
from __future__ import annotations

import json
from typing import Any

from rich.console import Console


class EventPrinter:
    """Stateful Rich-based renderer for the LoopEvent stream.

    Designed to be passed as `RunAgentParams.on_event=printer`. Buffers text
    deltas so we print one coherent line per turn instead of token-by-token noise.
    """

    def __init__(self, console: Console | None = None, *, show_thinking: bool = False):
        self.console = console or Console()
        self.show_thinking = show_thinking
        # Per-agent state — supports nested sub-agents (Coordinator → Explore etc)
        self._turn_by_agent: dict[str, int] = {}
        self._text_buffer: dict[str, str] = {}
        self._thinking_buffer: dict[str, str] = {}
        self._first_text_of_turn: dict[str, bool] = {}
        self._current_agent: str = ""

    # ── Public callable interface used by lifecycle.on_event ───────────────

    def __call__(self, event: Any) -> None:
        kind = getattr(event, "kind", None)
        if kind == "agent_start":
            self._on_agent_start(event)
            return
        # All other events come from query.py as LoopEvent(kind=..., payload=...).
        agent_id = getattr(event, "_agent_id", "?")
        agent_type = getattr(event, "_agent_type", "?")
        if agent_id and agent_id != self._current_agent:
            # Switch focus when control crosses agent boundaries (e.g., a sub-agent
            # spawned by AgentTool starts streaming).
            self._current_agent = agent_id

        payload = getattr(event, "payload", None)
        if kind == "text":
            self._on_text(agent_id, payload)
        elif kind == "thinking":
            self._on_thinking(agent_id, payload)
        elif kind == "tool_use":
            self._on_tool_use(agent_id, payload)
        elif kind == "tool_result":
            self._on_tool_result(agent_id, payload)
        elif kind == "usage":
            self._on_usage(agent_id, payload)
        elif kind == "error":
            self._on_error(agent_id, payload)

    # ── Handlers ──────────────────────────────────────────────────────────

    def _on_agent_start(self, ev: Any) -> None:
        self._flush_text(self._current_agent)
        agent_id = getattr(ev, "agent_id", "?")
        agent_type = getattr(ev, "agent_type", "?")
        mode = getattr(ev, "permission_mode", "?")
        self._print()
        self._rule(
            f"[bold cyan]agent: {agent_type}[/] [dim]({agent_id})[/]  [dim]mode={mode}[/]",
            align="left",
        )
        self._turn_by_agent[agent_id] = 0
        self._first_text_of_turn[agent_id] = True
        self._current_agent = agent_id

    def _on_text(self, agent_id: str, chunk: Any) -> None:
        text = getattr(chunk, "text", "") or ""
        if not text:
            return
        # Buffer text deltas until we see a tool_use or usage event.
        self._text_buffer[agent_id] = self._text_buffer.get(agent_id, "") + text

    def _on_thinking(self, agent_id: str, chunk: Any) -> None:
        if not self.show_thinking:
            return
        text = getattr(chunk, "thinking", "") or ""
        if text:
            self._thinking_buffer[agent_id] = self._thinking_buffer.get(agent_id, "") + text

    def _on_tool_use(self, agent_id: str, chunk: Any) -> None:
        self._flush_text(agent_id)
        self._flush_thinking(agent_id)
        tu = getattr(chunk, "tool_use", None)
        if tu is None:
            return
        name = getattr(tu, "name", "?")
        try:
            args = json.dumps(getattr(tu, "input", {}) or {}, separators=(",", " "))
        except Exception:
            args = str(getattr(tu, "input", ""))
        args_short = args if len(args) <= 140 else args[:137] + "..."
        self._print(f"        [cyan]→ {name}[/]([dim]{_escape(args_short)}[/])")

    def _on_tool_result(self, agent_id: str, tracked: Any) -> None:
        # tracked is a TrackedTool — has .tool.name, .result.text, .result.is_error
        tool = getattr(tracked, "tool", None)
        result = getattr(tracked, "result", None)
        if tool is None or result is None:
            return
        name = getattr(tool, "name", "?")
        is_err = getattr(result, "is_error", False)
        text = getattr(result, "text", "") or ""
        nbytes = len(text)
        marker = "[bold red]✗" if is_err else "[bold green]✓"
        color = "red" if is_err else "green"
        # extract a short stdout/stderr preview
        preview = text.replace("\n", " ").strip()
        if len(preview) > 100:
            preview = preview[:97] + "..."
        outcome = "error" if is_err else "ok"
        self._print(
            f"        {marker}[/] [{color}]{name:<14}[/] "
            f"[dim]{outcome:>5s}  {nbytes:>6d}b[/]  [dim]{_escape(preview)}[/]"
        )

    def _on_usage(self, agent_id: str, chunk: Any) -> None:
        # usage marks the end of one assistant turn → bump turn counter
        self._flush_text(agent_id)
        self._flush_thinking(agent_id)
        usage = getattr(chunk, "usage", None)
        if usage is None:
            return
        t = self._turn_by_agent.get(agent_id, 0)
        in_tok = getattr(usage, "input_tokens", 0)
        out_tok = getattr(usage, "output_tokens", 0)
        cache_read = getattr(usage, "cache_read_input_tokens", 0)
        cache_create = getattr(usage, "cache_creation_input_tokens", 0)
        ratio_pct = 0
        denom = in_tok + cache_read + cache_create
        if denom:
            ratio_pct = int(round(100 * cache_read / denom))
        self._print(
            f"        [dim]usage: in={in_tok:>5d} out={out_tok:>4d} "
            f"cache_read={cache_read:>5d} hit={ratio_pct:>2d}%[/]"
        )
        self._turn_by_agent[agent_id] = t + 1
        self._first_text_of_turn[agent_id] = True

    def _on_error(self, agent_id: str, chunk: Any) -> None:
        self._flush_text(agent_id)
        err_class = getattr(chunk, "error_class", "?")
        msg = (getattr(chunk, "message", "") or "")[:300]
        self._print(f"        [bold red]✗ ERROR[/] [red]{err_class}[/]: [dim]{_escape(msg)}[/]")

    # ── Internal helpers ──────────────────────────────────────────────────

    def _flush_text(self, agent_id: str) -> None:
        text = self._text_buffer.pop(agent_id, "").strip()
        if not text:
            return
        # Print the turn-header line once per turn
        turn = self._turn_by_agent.get(agent_id, 0)
        if self._first_text_of_turn.get(agent_id, True):
            # condense whitespace for the in-line preview
            line = " ".join(text.split())
            if len(line) > 240:
                line = line[:237] + "..."
            self._print(f"  [bold]T{turn:>2d}[/]  {_escape(line)}")
            self._first_text_of_turn[agent_id] = False
        else:
            # mid-turn continuation (rare; only after a tool result)
            line = " ".join(text.split())
            if len(line) > 240:
                line = line[:237] + "..."
            self._print(f"        [dim]…[/]  {_escape(line)}")

    def _flush_thinking(self, agent_id: str) -> None:
        text = self._thinking_buffer.pop(agent_id, "").strip()
        if not text or not self.show_thinking:
            return
        line = " ".join(text.split())
        if len(line) > 240:
            line = line[:237] + "..."
        self._print(f"        [italic dim]💭 {_escape(line)}[/]")

    def _print(self, *args: Any, **kwargs: Any) -> None:
        self.console.print(*args, **kwargs)
        self._flush_console()

    def _rule(self, *args: Any, **kwargs: Any) -> None:
        self.console.rule(*args, **kwargs)
        self._flush_console()

    def _flush_console(self) -> None:
        file_obj = getattr(self.console, "file", None)
        flush = getattr(file_obj, "flush", None)
        if callable(flush):
            try:
                flush()
            except Exception:
                pass


def _escape(s: str) -> str:
    """Escape Rich markup characters so they don't get parsed as tags."""
    return s.replace("[", "\\[").replace("]", "\\]")
