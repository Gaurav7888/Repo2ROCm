"""
Mempalace provider — Stage 2 of the memory layer.

Two responsibilities:

1. **Per-run memory** (one palace per repo+sha). Persists every meaningful
   event of a build attempt as a verbatim drawer with rich metadata so that
   later turns / runs can do selective retrieval instead of "append everything".

2. **Cumulative knowledge base** (a single shared palace across all runs).
   At the end of every run we distill the trajectory into "DO" / "DON'T" /
   "PATTERN" / "COMPATIBILITY" lessons and write them to the global wing.
   Future runs can query this wing as a growing experience base.

This file is **write-only** for Stage 2. Retrieval (Stage 3) will plug into
`Configuration.run`'s per-turn loop later.

API (kept minimal so callers don't have to know about chromadb/mempalace):

    mem = RunMemory.create(full_name, sha, root_path)
    mem.write_plan(plan_text)
    mem.write_paper_extracts(paper_text, paper_experiments=[...])
    mem.write_readme_run_cmds([...])
    mem.write_decision("base_image", "rocm/pytorch:latest", reason="...")
    mem.write_turn(traj_record_dict, full_observation_text,
                   action_content, compacted_observation)
    mem.distill_and_write_lessons(trajectory_iter)   # called at end of run
    mem.close()
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Optional


_DEFAULT_PALACE_BASE = os.path.expanduser("~/.mempalace/palaces")
_GLOBAL_WING = "rocm_global_lessons"  # cumulative knowledge base wing


def _safe(name: str) -> str:
    """Make name safe for mempalace's `sanitize_name` (alnum + _ - .)."""
    s = re.sub(r"[^A-Za-z0-9_.\-]", "_", name)
    return s.strip("_") or "anon"


def _to_dict(rec: Any) -> Dict[str, Any]:
    if isinstance(rec, dict):
        return rec
    if is_dataclass(rec):
        return asdict(rec)
    if hasattr(rec, "__dict__"):
        return dict(rec.__dict__)
    return {}


class _NoopMemory:
    """Drop-in fallback when mempalace is unavailable. All ops become no-ops."""
    enabled = False
    palace_path = ""
    wing = ""

    def __getattr__(self, name):
        def _noop(*a, **kw):
            return None
        return _noop


class RunMemory:
    """Per-run mempalace store. Wing = repo+sha. Rooms = memory categories."""

    enabled = True

    # Room schema, derived from observed Repo2ROCm artifacts.
    ROOMS = {
        "plan": "Strategic plan sections (one drawer per ## header).",
        "paper_extracts": "Verbatim paper text chunks tagged with section markers.",
        "paper_experiments": "Shortlisted experiments as JSON.",
        "readme_run_cmds": "Run commands extracted verbatim from the README.",
        "configs": "Verbatim config files (yaml/toml/json).",
        "decisions": "Top-level decisions (base image, python version, env vars).",
        "commands_success": "Successful actions + compacted observation.",
        "commands_failed": "Failed actions + compacted observation + error class.",
        "fixes": "Failure → resolution pairs (failure_action, fix_action, error_class).",
        "patches": "Diff/code-edit blocks the agent applied.",
        "rocm_env": "Detected ROCm/HIP versions, env vars, devices.",
        "metrics": "Numeric metrics extracted from stdout (loss/acc/etc.).",
    }

    def __init__(self, full_name: str, sha: str, palace_base: str = _DEFAULT_PALACE_BASE):
        self.full_name = full_name
        self.sha = sha
        repo_short = _safe(full_name.split("/")[-1])
        sha_short = _safe(sha[:7])
        self.wing = f"{repo_short}_{sha_short}"
        self.palace_path = os.path.join(palace_base, self.wing)
        os.makedirs(self.palace_path, exist_ok=True)

        from mempalace.palace import get_collection
        self._col = get_collection(self.palace_path, create=True)

        self._chunk_counter: Dict[str, int] = {r: 0 for r in self.ROOMS}
        self._t0 = time.time()
        self._writes = 0

    @classmethod
    def create(cls, full_name: str, sha: str,
               palace_base: str = _DEFAULT_PALACE_BASE) -> "RunMemory":
        try:
            return cls(full_name, sha, palace_base)
        except Exception as e:
            print(f"[mempalace] disabled (init failed: {e})")
            return _NoopMemory()  # type: ignore[return-value]

    # ── Low-level write ──────────────────────────────────────────────────────

    def _add(self, room: str, content: str, source_file: str = "agent",
             tags: Optional[Dict[str, Any]] = None) -> None:
        if not content or not content.strip():
            return
        from mempalace.miner import add_drawer
        idx = self._chunk_counter.get(room, 0)
        self._chunk_counter[room] = idx + 1
        # Attach tags as a JSON sidecar appended to the drawer text so they
        # survive Chroma's metadata constraints (which require flat scalars).
        if tags:
            try:
                tagstr = json.dumps(tags, default=str, sort_keys=True)
                content = content.rstrip() + f"\n\n[META] {tagstr}"
            except Exception:
                pass
        try:
            add_drawer(self._col, self.wing, room, content,
                       source_file=source_file, chunk_index=idx,
                       agent="repo2rocm")
            self._writes += 1
        except Exception as e:
            print(f"[mempalace] add_drawer({room}) failed: {e}")

    # ── Pre-loop writes (called from main.py / planner / paper agent) ────────

    def write_plan(self, plan_text: str) -> None:
        """Split plan.txt by '## ' headers; one drawer per section."""
        if not plan_text:
            return
        sections = re.split(r"(?m)^\s*##\s+", plan_text)
        if sections and not sections[0].lstrip().startswith("#"):
            preamble = sections[0].strip()
            if preamble:
                self._add("plan", preamble, source_file="plan.txt:preamble")
            sections = sections[1:]
        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue
            header = sec.splitlines()[0][:80] if sec.splitlines() else "section"
            self._add("plan", "## " + sec,
                      source_file=f"plan.txt:{_safe(header)}")

    def write_paper_extracts(self, paper_text: str,
                              paper_experiments: Optional[list] = None,
                              chunk_chars: int = 4000) -> None:
        """Chunk the paper PDF text and write to paper_extracts."""
        if paper_text:
            n = len(paper_text)
            i = 0
            chunk_id = 0
            while i < n:
                seg = paper_text[i:i + chunk_chars]
                # try to break on a newline near the end
                if i + chunk_chars < n:
                    nl = seg.rfind("\n", chunk_chars // 2)
                    if nl > 0:
                        seg = seg[:nl]
                self._add("paper_extracts", seg,
                          source_file=f"paper.pdf:chunk_{chunk_id}",
                          tags={"chunk_id": chunk_id, "char_offset": i})
                i += len(seg) if seg else chunk_chars
                chunk_id += 1
        if paper_experiments:
            try:
                content = json.dumps(paper_experiments, indent=2, default=str)
                self._add("paper_experiments", content,
                          source_file="paper_experiments.json")
            except Exception:
                pass

    def write_readme_run_cmds(self, cmds: list) -> None:
        for i, cmd in enumerate(cmds or []):
            text = cmd if isinstance(cmd, str) else json.dumps(cmd, default=str)
            self._add("readme_run_cmds", text,
                      source_file=f"README.md:cmd_{i}")

    def write_configs(self, configs: Dict[str, str]) -> None:
        for path, body in (configs or {}).items():
            self._add("configs",
                      f"# ---- {path} ----\n{body}",
                      source_file=path)

    def write_decision(self, kind: str, value: str, reason: str = "") -> None:
        text = f"DECISION[{kind}] = {value}"
        if reason:
            text += f"\nReason: {reason}"
        self._add("decisions", text, source_file=f"decisions:{_safe(kind)}",
                  tags={"kind": kind, "value": str(value)[:200]})

    def write_rocm_env(self, info: Dict[str, Any]) -> None:
        try:
            self._add("rocm_env", json.dumps(info, indent=2, default=str),
                      source_file="rocm_env.json")
        except Exception:
            pass

    # ── Per-turn writes (called from configuration.py) ───────────────────────

    def write_turn(self, record: Any, full_observation: str,
                   compact_obj: Any = None) -> None:
        """
        Persist one turn. `record` is a TrajectoryRecord-like object
        (dict or dataclass). `compact_obj` is optional CompactedObservation.
        """
        d = _to_dict(record)
        turn = d.get("turn_number", -1)
        action_type = d.get("action_type", "")
        action = d.get("action_content", "") or ""
        outcome = d.get("outcome", "")
        return_code = d.get("return_code", "")
        error_class = d.get("error_class") or ""
        duration = d.get("duration_seconds", 0.0)

        room = "commands_success" if outcome == "success" else "commands_failed"
        if action_type == "diff":
            room = "patches"

        # Build the drawer text: action + compacted observation
        short_obs = ""
        metrics = []
        if compact_obj is not None:
            short_obs = getattr(compact_obj, "short", "") or ""
            metrics = list(getattr(compact_obj, "metrics", []) or [])
        else:
            short_obs = (full_observation or "")[:2000]

        text = (
            f"TURN {turn}  type={action_type}  outcome={outcome}  "
            f"rc={return_code}  duration={duration:.1f}s\n"
            f"--- ACTION ---\n{action}\n"
            f"--- OBSERVATION (compacted) ---\n{short_obs}\n"
        )
        if error_class:
            text += f"--- ERROR_CLASS ---\n{error_class}\n"

        self._add(
            room, text,
            source_file=f"trajectory:turn_{turn}",
            tags={
                "turn": turn,
                "action_type": action_type,
                "outcome": outcome,
                "return_code": return_code,
                "error_class": error_class,
                "duration_s": round(float(duration), 2),
            },
        )

        # Also persist metrics individually for easy retrieval later.
        for name, value in metrics:
            self._add("metrics",
                      f"turn={turn} {name}={value}",
                      source_file=f"metrics:turn_{turn}",
                      tags={"turn": turn, "name": name, "value": value})

        # Keep a full-fidelity copy too (chunked) — verbatim is the mempalace ethos.
        if full_observation and len(full_observation) > 2500:
            for j in range(0, len(full_observation), 4000):
                seg = full_observation[j:j + 4000]
                self._add(room + "_full", seg,
                          source_file=f"trajectory:turn_{turn}_obs_{j // 4000}",
                          tags={"turn": turn, "offset": j})

    # ── Stage 3: per-turn retrieval ──────────────────────────────────────────

    @staticmethod
    def _approx_tokens(s: str) -> int:
        return max(1, len(s) // 4)

    def _search(self, query: str, palace_path: str, wing: str,
                room: str, n_results: int) -> list:
        from mempalace.searcher import search_memories
        try:
            r = search_memories(query, palace_path, wing=wing, room=room,
                                n_results=n_results, max_distance=0.0)
            return r.get("results") or []
        except Exception as e:
            print(f"[mempalace] search({wing}/{room}) failed: {e}")
            return []

    def recall_pack(self, query: str,
                     rooms: tuple = ("commands_success", "commands_failed",
                                     "fixes", "decisions", "patches"),
                     n_per_room: int = 4,
                     token_budget: int = 1500,
                     header: str = "RELEVANT PRIOR CONTEXT (this run)") -> str:
        """Per-run recall pack. Returns a single string ready for prompt injection."""
        if not query:
            return ""
        chunks: List[str] = []
        for room in rooms:
            hits = self._search(query, self.palace_path, self.wing, room, n_per_room)
            for h in hits:
                t = (h.get("text") or "").strip()
                if not t:
                    continue
                # de-noise the [META] line — keep meta sidecar but compact it
                snippet = t.split("\n[META]")[0]
                chunks.append(f"  [{room}] {snippet}")
        return self._budget(header, chunks, token_budget)

    def recall_paper(self, queries: tuple = (
            "main results table headline accuracy metric",
            "hyperparameters learning rate batch size epochs seed lora rank",
            "experimental setup datasets benchmarks model sizes",
            "method algorithm initialization theorem",
        ), n_per_query: int = 3, token_budget: int = 8000,
        per_chunk_max_chars: int = 1500,
    ) -> str:
        """Multi-aspect recall over the `paper_extracts` room. Replaces the
        ~155 KB paper-text dump in paper_agent.shortlist_experiments."""
        if not queries:
            return ""
        sections: List[str] = []
        for q in queries:
            hits = self._search(q, self.palace_path, self.wing,
                                "paper_extracts", n_per_query)
            for h in hits:
                t = (h.get("text") or "").strip()
                if not t:
                    continue
                snippet = t.split("\n[META]")[0]
                sections.append(f"  [paper:{q[:60]}] {snippet}")
        if not sections:
            return ""
        return self._budget("PAPER EXCERPTS (retrieved by topic)",
                            sections, token_budget,
                            per_chunk_max_chars=per_chunk_max_chars)

    def recall_global_lessons(self, query: str,
                               rooms: tuple = ("dont", "do", "pattern"),
                               n_per_room: int = 3,
                               token_budget: int = 1000,
                               palace_base: str = _DEFAULT_PALACE_BASE,
                               header: str = "CROSS-RUN LESSONS (global KB)") -> str:
        """Recall from the cumulative knowledge base (shared across all repos)."""
        if not query:
            return ""
        from mempalace.palace import get_collection  # ensure global palace exists
        global_palace = os.path.join(palace_base, "_global")
        if not os.path.isdir(global_palace):
            return ""
        try:
            get_collection(global_palace, create=False)
        except Exception:
            return ""
        chunks: List[str] = []
        for room in rooms:
            hits = self._search(query, global_palace, _GLOBAL_WING, room, n_per_room)
            for h in hits:
                t = (h.get("text") or "").strip()
                if not t:
                    continue
                snippet = t.split("\n[META]")[0]
                chunks.append(f"  [{room}] {snippet}")
        return self._budget(header, chunks, token_budget)

    def _budget(self, header: str, chunks: List[str], token_budget: int,
                per_chunk_max_chars: int = 1200) -> str:
        """
        Greedy fill with per-chunk truncation. Each chunk is hard-capped to
        `per_chunk_max_chars` so a single giant `pip install ...` line cannot
        eat the whole budget. Skip (don't bail on) chunks that would overflow
        what's left so we maximize coverage across rooms.
        """
        if not chunks:
            return ""
        out: List[str] = []
        used = 0
        max_chars = max(token_budget * 4, per_chunk_max_chars)
        for c in chunks:
            piece = c if len(c) <= per_chunk_max_chars else (
                c[:per_chunk_max_chars] + " …[trunc]"
            )
            remaining = max_chars - used
            if remaining <= 80:
                break
            if len(piece) > remaining:
                piece = piece[: remaining - 16] + " …[trunc]"
            out.append(piece)
            used += len(piece) + 2
        if not out:
            return ""
        body = "\n".join(out)
        return (
            f"\n========================================\n"
            f"{header}\n"
            f"========================================\n"
            f"{body}\n"
        )

    # ── End-of-run lesson distillation → cumulative KB ───────────────────────

    def distill_and_write_lessons(self, trajectory_path: Optional[str] = None,
                                   palace_base: str = _DEFAULT_PALACE_BASE,
                                   final_status: str = "unknown") -> Dict[str, int]:
        """
        Read the per-run trajectory and emit DO/DONT/PATTERN/COMPAT lessons
        into the GLOBAL wing so future runs can recall them.

        Returns counts of lessons written.
        """
        from mempalace.palace import get_collection
        from mempalace.miner import add_drawer

        global_palace = os.path.join(palace_base, "_global")
        os.makedirs(global_palace, exist_ok=True)
        gcol = get_collection(global_palace, create=True)

        rows: List[Dict[str, Any]] = []
        if trajectory_path and os.path.exists(trajectory_path):
            with open(trajectory_path) as f:
                for line in f:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue

        # Soft-failure detection: many trajectories report rc=0 because the
        # action was piped (`cmd | tee log | tail`) which masks the upstream
        # exit code. Re-classify as "soft_failure" if the observation body
        # contains canonical failure markers.
        _soft_fail_re = re.compile(
            r"(Traceback \(most recent call last\)"
            r"|\bError:|\bException:|\bFAILED\b|\bfatal:"
            r"|\bModuleNotFoundError\b|\bImportError\b|\bRuntimeError\b"
            r"|\bundefined symbol\b|\bout of memory\b|\bHIPError\b|\bCUDA error\b"
            r"|\bKilled\b|\bSegmentation fault\b)"
        )
        _soft_class_re = re.compile(
            r"\b(ModuleNotFoundError|ImportError|AttributeError|TypeError|"
            r"ValueError|RuntimeError|OSError|FileNotFoundError|"
            r"PermissionError|HIPError|OutOfMemoryError|AssertionError|"
            r"KeyError|IndexError|SyntaxError|IndentationError)\b"
        )
        for r in rows:
            if r.get("outcome") == "success":
                obs = r.get("observation_raw") or ""
                if _soft_fail_re.search(obs):
                    r["outcome"] = "soft_failure"
                    if not r.get("error_class"):
                        m = _soft_class_re.search(obs)
                        r["error_class"] = m.group(1) if m else "SoftFailure"

        counts = {"do": 0, "dont": 0, "pattern": 0, "compatibility": 0}

        # Group consecutive (failure → success) on the same logical action prefix
        # to extract DO and DONT lessons.
        last_fail: Optional[Dict[str, Any]] = None
        for r in rows:
            outcome = r.get("outcome")
            action = (r.get("action_content") or "").strip()
            ec = r.get("error_class") or ""
            if outcome in ("failure", "soft_failure"):
                # DON'T lesson — record the action that failed with this error class.
                soft_tag = " (soft-failure inside rc=0 output)" if outcome == "soft_failure" else ""
                lesson = (
                    f"DON'T do this on ROCm{soft_tag} — failed with {ec or 'non-zero exit'}.\n"
                    f"Repo: {self.full_name}@{self.sha[:7]}\n"
                    f"Action:\n{action[:1200]}\n"
                )
                self._write_lesson(gcol, "dont", lesson, tags={
                    "repo": self.full_name, "sha": self.sha[:7],
                    "error_class": ec, "kind": "dont",
                    "outcome": outcome,
                })
                counts["dont"] += 1
                last_fail = r
            elif outcome == "success" and last_fail is not None:
                # DO / FIX lesson — this success followed a failure, likely fixed it.
                fail_action = (last_fail.get("action_content") or "").strip()
                fail_ec = last_fail.get("error_class") or ""
                lesson = (
                    f"DO this on ROCm to recover from {fail_ec or 'a failure'}.\n"
                    f"Repo: {self.full_name}@{self.sha[:7]}\n"
                    f"Failed action:\n{fail_action[:600]}\n"
                    f"Recovery action (worked):\n{action[:1200]}\n"
                )
                self._write_lesson(gcol, "do", lesson, tags={
                    "repo": self.full_name, "sha": self.sha[:7],
                    "error_class": fail_ec, "kind": "do",
                })
                counts["do"] += 1
                # Pattern: error_class → resolution
                pat = (
                    f"PATTERN: when you see `{fail_ec or 'failure'}` after running "
                    f"`{(fail_action.splitlines() or [''])[0][:120]}`, try:\n"
                    f"  {(action.splitlines() or [''])[0][:200]}"
                )
                self._write_lesson(gcol, "pattern", pat, tags={
                    "repo": self.full_name, "sha": self.sha[:7],
                    "error_class": fail_ec, "kind": "pattern",
                })
                counts["pattern"] += 1
                last_fail = None

        # Final-status summary lesson (always written)
        summary = (
            f"RUN_SUMMARY {self.full_name}@{self.sha[:7]} status={final_status}\n"
            f"turns={len(rows)} writes={self._writes} "
            f"duration={time.time() - self._t0:.0f}s"
        )
        self._write_lesson(gcol, "summary", summary, tags={
            "repo": self.full_name, "sha": self.sha[:7],
            "status": final_status, "kind": "summary",
        })

        return counts

    @staticmethod
    def _write_lesson(global_col, room: str, content: str,
                      tags: Optional[Dict[str, Any]] = None) -> None:
        from mempalace.miner import add_drawer
        if tags:
            try:
                content = content.rstrip() + "\n\n[META] " + json.dumps(
                    tags, default=str, sort_keys=True)
            except Exception:
                pass
        try:
            add_drawer(global_col, _GLOBAL_WING, room, content,
                       source_file=f"lessons:{room}", chunk_index=0,
                       agent="repo2rocm-distiller")
        except Exception as e:
            print(f"[mempalace-global] add_drawer({room}) failed: {e}")

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        return {
            "wing": self.wing,
            "palace_path": self.palace_path,
            "writes": self._writes,
            "by_room": dict(self._chunk_counter),
            "elapsed_s": round(time.time() - self._t0, 1),
        }

    def close(self) -> None:
        # Chroma persists automatically; nothing to flush explicitly.
        pass
