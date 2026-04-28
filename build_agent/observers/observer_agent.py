"""
Async observer sidecar — LLM- and skill-file-driven.

Pipeline (per turn snapshot):
  1. Ingest the snapshot from the file bus.
  2. Heuristic gate (cheap, no LLM): is anything interesting happening?
  3. LLM Reviewer (one structured JSON call): should we intervene? which
     skill applies? do we need web evidence?
  4. Optional researcher call (web search + synthesis) to ground the
     advice with citations.
  5. Emit an `ObserverAdvice` row to the advice file bus, deduplicating
     against recent emissions.

The main loop never imports the sidecar directly. Communication is
one-way:

  - main loop appends events to `observer_events.jsonl`
  - sidecar appends advice rows to `observer_advice.jsonl`
  - main loop reads only fresh advice at safe turn boundaries

Skills, the role description, and the JSON contract live in Markdown
files under `observers/prompts/` and `observers/skills/`. Edits to those
take effect on the next sidecar restart with no code change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Set

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_AGENT_ROOT = os.path.dirname(CURRENT_DIR)
if BUILD_AGENT_ROOT not in sys.path:
    sys.path.insert(0, BUILD_AGENT_ROOT)

from observers.types import (  # noqa: E402
    ObserverAdvice,
    ObserverEvent,
    append_jsonl,
    read_jsonl_from_offset,
)
from observers.reviewer import Reviewer, ReviewerConfig  # noqa: E402


# ── Helpers ──────────────────────────────────────────────────────────────────


def _advice_fingerprint(advice: ObserverAdvice) -> str:
    """Stable key for deduplicating near-identical advice rows."""
    base = {
        "skill": advice.profile_used,
        "predicted_failure": advice.predicted_failure[:160],
        "diagnosis_head": advice.diagnosis[:160],
        "applies_before": advice.applies_before,
    }
    return hashlib.sha1(json.dumps(base, sort_keys=True).encode()).hexdigest()[:20]


# ── Sidecar ──────────────────────────────────────────────────────────────────


class ObserverSidecar:
    """Reads events, calls the Reviewer, writes advice."""

    def __init__(self, events_path: str, advice_path: str, llm: str,
                 poll_interval_s: float = 1.0, max_history: int = 10) -> None:
        self.events_path = events_path
        self.advice_path = advice_path
        self.llm = llm
        self.poll_interval_s = max(0.2, poll_interval_s)
        self.max_history = max(2, max_history)
        self._events_offset = 0
        self._run_context: Dict[str, Any] = {}
        self._snapshots: List[Dict[str, Any]] = []
        self._done = False

        # Dedup memory: last turn at which we emitted each fingerprint, and
        # last turn at which we emitted advice for each skill.
        self._emitted_fingerprints: Dict[str, int] = {}
        self._emitted_skill_at: Dict[str, int] = {}
        # Cooldowns expressed in turn-distance.
        self._fingerprint_cooldown_turns = 3
        self._skill_cooldown_turns = 4

        self.reviewer = Reviewer(ReviewerConfig(
            llm=self.llm,
            max_history=self.max_history,
        ))

    # ── Event handlers ─────────────────────────────────────────────────────

    def _on_run_started(self, payload: Dict[str, Any]) -> None:
        self._run_context = dict(payload or {})

    def _on_turn_snapshot(self, payload: Dict[str, Any]) -> None:
        snapshot = dict(payload or {})
        self._snapshots.append(snapshot)
        self._snapshots = self._snapshots[-self.max_history:]

        if not self.reviewer.should_consider(self._snapshots):
            return

        try:
            decision = self.reviewer.decide(self._snapshots, self._run_context)
        except Exception as e:
            self._log(f"reviewer.decide failed: {e}")
            return

        if not decision.intervene or decision.skill == "progressOK":
            return

        try:
            advice = self.reviewer.materialize(
                decision, self._snapshots, self._run_context,
            )
        except Exception as e:
            self._log(f"reviewer.materialize failed: {e}")
            return
        if advice is None:
            return

        if not self._should_emit(advice):
            return

        append_jsonl(self.advice_path, advice)
        fp = _advice_fingerprint(advice)
        self._emitted_fingerprints[fp] = advice.turn_seen
        self._emitted_skill_at[advice.profile_used] = advice.turn_seen

    def _should_emit(self, advice: ObserverAdvice) -> bool:
        if not advice.recommended_strategy and not advice.suggested_questions_or_tools:
            return False
        fp = _advice_fingerprint(advice)
        last_fp_turn = self._emitted_fingerprints.get(fp, -100)
        if (advice.turn_seen - last_fp_turn) <= self._fingerprint_cooldown_turns:
            return False
        last_skill_turn = self._emitted_skill_at.get(advice.profile_used, -100)
        if (advice.turn_seen - last_skill_turn) <= self._skill_cooldown_turns:
            # Allow re-emit if the advice is high priority and the skill
            # hasn't fired in 2+ turns — gives the observer a path to
            # escalate if the agent ignored an earlier nudge.
            if advice.priority == "high" and (advice.turn_seen - last_skill_turn) >= 2:
                return True
            return False
        return True

    # ── Logging ────────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        # Sidecar logs go to stderr by default; ObserverClient redirects
        # them to observer_sidecar.log so they survive the run.
        try:
            sys.stderr.write(f"[observer-sidecar] {msg}\n")
            sys.stderr.flush()
        except Exception:
            pass

    # ── Loop ───────────────────────────────────────────────────────────────

    def _handle_event(self, row: Dict[str, Any]) -> None:
        event_type = str(row.get("event_type") or "")
        payload = row.get("payload") or {}
        if event_type == "run_started":
            self._on_run_started(payload)
            return
        if event_type == "turn_snapshot":
            self._on_turn_snapshot(payload)
            return
        if event_type == "run_finished":
            self._done = True

    def run(self) -> None:
        self._log(f"started llm={self.llm} skills={len(self.reviewer._skills)}")
        while not self._done:
            rows, self._events_offset = read_jsonl_from_offset(
                self.events_path,
                self._events_offset,
            )
            if not rows:
                time.sleep(self.poll_interval_s)
                continue
            for row in rows:
                try:
                    self._handle_event(row)
                except Exception as e:
                    self._log(f"event handler error: {e}")
                if self._done:
                    break


# ── Client (used by main loop) ───────────────────────────────────────────────


class ObserverClient:
    """Thin client used by the main loop to start/feed/consume the sidecar."""

    def __init__(self, output_dir: str, llm: str,
                 api_key: str = "", enabled: bool = True,
                 use_claude_code: bool = False,
                 claude_code_model: Optional[str] = None) -> None:
        self.output_dir = output_dir
        self.llm = llm
        self.api_key = api_key or ""
        self.enabled = bool(enabled and llm)
        self.use_claude_code = bool(use_claude_code)
        self.claude_code_model = claude_code_model or ""
        self.events_path = os.path.join(output_dir, "observer_events.jsonl")
        self.advice_path = os.path.join(output_dir, "observer_advice.jsonl")
        self.log_path = os.path.join(output_dir, "observer_sidecar.log")
        self._advice_offset = 0
        self._consumed_ids: Set[str] = set()
        self._proc: Optional[subprocess.Popen] = None
        self._log_handle = None

    def start(self) -> None:
        if not self.enabled:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        # Truncate any prior bus / log contents from a previous run.
        # The log is opened with mode=w (not append) so old RetryError noise
        # from a prior failed run does not mask the current run's state.
        open(self.events_path, "w", encoding="utf-8").close()
        open(self.advice_path, "w", encoding="utf-8").close()
        self._log_handle = open(self.log_path, "w", encoding="utf-8")
        env = os.environ.copy()
        if self.api_key and not env.get("AMD_LLM_API_KEY"):
            env["AMD_LLM_API_KEY"] = self.api_key
        # Propagate Claude Code mode to the subprocess. The
        # `_USE_CLAUDE_CODE` flag in `utils/claude_code_client.py` is a
        # module global that the parent set at startup; the subprocess
        # imports the module fresh, so without this hand-off the sidecar
        # would silently fall through to the AMD gateway path and fail
        # when no AMD API key is available.
        if self.use_claude_code:
            env["REPO2ROCM_OBSERVER_USE_CLAUDE_CODE"] = "1"
            if self.claude_code_model:
                env["REPO2ROCM_OBSERVER_CLAUDE_CODE_MODEL"] = self.claude_code_model
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--events", self.events_path,
            "--advice", self.advice_path,
            "--llm", self.llm,
        ]
        if self.use_claude_code:
            cmd.append("--use-claude-code")
            if self.claude_code_model:
                cmd.extend(["--claude-code-model", self.claude_code_model])
        self._proc = subprocess.Popen(
            cmd,
            stdout=self._log_handle,
            stderr=self._log_handle,
            cwd=BUILD_AGENT_ROOT,
            env=env,
        )

    def emit_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        append_jsonl(self.events_path, ObserverEvent.create(event_type, payload))

    def consume_new_advice(self, current_turn: Optional[int] = None,
                           applies_to_action: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        rows, self._advice_offset = read_jsonl_from_offset(
            self.advice_path,
            self._advice_offset,
        )
        fresh: List[Dict[str, Any]] = []
        for row in rows:
            advice_id = str(row.get("advice_id") or "")
            if not advice_id or advice_id in self._consumed_ids:
                continue
            expires = row.get("expires_after_turn")
            if (
                current_turn is not None
                and isinstance(expires, (int, float))
                and int(expires) >= 0
                and int(expires) < int(current_turn)
            ):
                self._consumed_ids.add(advice_id)
                continue
            self._consumed_ids.add(advice_id)
            fresh.append(row)
        if applies_to_action and fresh:
            preferred = [r for r in fresh
                         if not r.get("applies_before")
                         or str(r.get("applies_before")).lower()
                            == applies_to_action.lower()
                         or str(r.get("applies_before")).lower() == "next_turn"]
            if preferred:
                fresh = preferred
        priority_rank = {"high": 0, "normal": 1, "low": 2}
        fresh.sort(key=lambda r: (
            priority_rank.get(str(r.get("priority") or "normal"), 1),
            -int(r.get("turn_seen") or 0),
        ))
        return fresh

    def shutdown(self) -> None:
        if self.enabled:
            try:
                self.emit_event("run_finished", {})
            except Exception:
                pass
        if self._proc is not None:
            try:
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.terminate()
                except Exception:
                    pass
            self._proc = None
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except Exception:
                pass
            self._log_handle = None


# ── CLI entrypoint (used when started by ObserverClient.start) ───────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repo2ROCm observer sidecar")
    parser.add_argument("--events", required=True)
    parser.add_argument("--advice", required=True)
    parser.add_argument("--llm", required=True)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--max-history", type=int, default=10)
    parser.add_argument("--use-claude-code", action="store_true",
                        help="Route LLM calls through the Claude Code SDK"
                             " (matches the parent agent's provider).")
    parser.add_argument("--claude-code-model", default="",
                        help="Model name for Claude Code mode (e.g. 'sonnet').")
    return parser.parse_args()


def _activate_llm_provider(args: argparse.Namespace) -> str:
    """Match the parent's LLM provider before any reviewer LLM call.

    Returns a short tag describing the provider that was activated, so the
    sidecar can log it on startup. The reviewer never directly references
    `utils.claude_code_client`; it goes through `utils.llm.get_llm_response`
    which dispatches based on the global flag we set here.
    """
    use_cc = bool(args.use_claude_code) or bool(
        os.environ.get("REPO2ROCM_OBSERVER_USE_CLAUDE_CODE")
    )
    if not use_cc:
        return "amd_gateway"
    cc_model = args.claude_code_model or os.environ.get(
        "REPO2ROCM_OBSERVER_CLAUDE_CODE_MODEL", ""
    )
    try:
        from utils.claude_code_client import set_claude_code_mode
        set_claude_code_mode(enabled=True, model=cc_model or None)
        return f"claude_code({cc_model or 'default'})"
    except Exception as e:
        sys.stderr.write(
            f"[observer-sidecar] failed to activate Claude Code mode "
            f"({e}); falling back to AMD gateway\n"
        )
        sys.stderr.flush()
        return "amd_gateway_fallback"


def main() -> None:
    args = _parse_args()
    provider = _activate_llm_provider(args)
    sys.stderr.write(f"[observer-sidecar] llm_provider={provider} model={args.llm}\n")
    sys.stderr.flush()
    sidecar = ObserverSidecar(
        events_path=args.events,
        advice_path=args.advice,
        llm=args.llm,
        poll_interval_s=args.poll_interval,
        max_history=args.max_history,
    )
    sidecar.run()


if __name__ == "__main__":
    main()
