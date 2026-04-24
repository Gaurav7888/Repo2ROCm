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
    mem.write_paper_experiments([...])
    mem.write_readme_run_cmds([...])
    mem.write_decision("base_image", "rocm/pytorch:latest", reason="...")
    mem.write_turn(traj_record_dict, full_observation_text,
                   action_content, compacted_observation)
    mem.distill_and_write_lessons(trajectory_iter)   # called at end of run
    mem.close()
"""

from __future__ import annotations

import hashlib
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
    # NOTE: the legacy `paper_extracts` room has been removed; raw paper text
    # belongs to graphify's static corpus (`graphify-out/paper_chunks.jsonl`).
    ROOMS = {
        "plan": "Strategic plan sections (one drawer per ## header).",
        "paper_experiments": "Shortlisted experiments as JSON.",
        "experiment_state": "Chosen experiment / target metric / tolerance / stage markers.",
        "context_refs": "References to graphify paper/code nodes used during the run.",
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

    def write_paper_experiments(self, paper_experiments: Optional[list]) -> None:
        """Persist the shortlisted experiments as a single JSON drawer.

        This replaces the legacy `write_paper_extracts(paper_text, ...)` shim.
        Raw paper text intentionally lives in the graphify static corpus, not
        in mempalace; only run-state references and shortlists belong here.
        """
        if not paper_experiments:
            return
        try:
            content = json.dumps(paper_experiments, indent=2, default=str)
            self._add("paper_experiments", content,
                      source_file="paper_experiments.json")
        except Exception:
            pass

    def write_paper_extracts(self, paper_text: str,
                              paper_experiments: Optional[list] = None,
                              chunk_chars: int = 4000) -> None:
        """Deprecated shim that forwards to `write_paper_experiments`.

        Older callers may still pass `paper_text`; we drop it on the floor
        because raw paper bytes belong in graphify, not mempalace.
        """
        if paper_experiments:
            self.write_paper_experiments(paper_experiments)

    def write_experiment_state(self, name: str, payload: Dict[str, Any],
                               source_file: str = "experiment_state.json") -> None:
        """Store compact paper/run state, not the raw paper body."""
        if not name:
            return
        try:
            body = {
                "name": name,
                "payload": payload or {},
            }
            self._add(
                "experiment_state",
                json.dumps(body, indent=2, default=str),
                source_file=f"{source_file}:{_safe(name)}",
                tags={"name": name},
            )
        except Exception:
            pass

    def write_context_ref(self, kind: str, ref_id: str, source: str,
                          why_relevant: str = "", extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Store a reference to static corpus content (graphify paper/code node,
        paper chunk id, file path, etc.), rather than duplicating the content.
        """
        if not kind or not ref_id:
            return
        body = {
            "kind": kind,
            "ref_id": ref_id,
            "source": source,
            "why_relevant": why_relevant,
            "extra": extra or {},
        }
        self._add(
            "context_refs",
            json.dumps(body, indent=2, default=str),
            source_file=f"context_ref:{_safe(kind)}:{_safe(ref_id)[:80]}",
            tags={"kind": kind, "ref_id": ref_id, "source": source},
        )

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
        """
        Paper-related *run state* recall.

        This no longer returns raw paper chunks. Instead it surfaces:
          - shortlisted experiments
          - experiment_state
          - context_refs
          - plan / decisions

        The static paper corpus itself is queried via `GraphifyProvider.query_paper`.
        """
        if not queries:
            return ""
        sections: List[str] = []
        for q in queries:
            for room in ("paper_experiments", "experiment_state", "context_refs", "plan", "decisions"):
                hits = self._search(q, self.palace_path, self.wing, room, n_per_query)
                for h in hits:
                    t = (h.get("text") or "").strip()
                    if not t:
                        continue
                    snippet = t.split("\n[META]")[0]
                    sections.append(f"  [{room}:{q[:60]}] {snippet}")
        if not sections:
            return ""
        return self._budget("PAPER RUN STATE (references, choices, decisions)",
                            sections, token_budget,
                            per_chunk_max_chars=per_chunk_max_chars)

    def recall_global_lessons(self, query: str,
                               rooms: tuple = ("dont", "do", "pattern"),
                               n_per_room: int = 3,
                               token_budget: int = 1000,
                               min_confidence: float = 0.4,
                               prefer_source: str = "llm_synthesis",
                               palace_base: str = _DEFAULT_PALACE_BASE,
                               header: str = "CROSS-RUN LESSONS (global KB)") -> str:
        """Recall from the cumulative knowledge base (shared across all repos).

        Filters:
          * `min_confidence` drops low-confidence heuristic lessons (the new
            distiller stamps a confidence; legacy lessons without one are
            treated as 0.3 so they sink unless the caller relaxes the bar).
          * `prefer_source` boosts lessons emitted by the LLM synthesiser over
            the heuristic fallback (used for ordering, not hard filtering).
        """
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

        scored: List[tuple] = []
        for room in rooms:
            hits = self._search(query, global_palace, _GLOBAL_WING, room, n_per_room)
            for h in hits:
                t = (h.get("text") or "").strip()
                if not t:
                    continue
                meta = ""
                if "\n[META]" in t:
                    snippet, _, meta = t.partition("\n[META]")
                else:
                    snippet = t
                meta_dict = {}
                try:
                    meta_dict = json.loads(meta.strip()) if meta.strip() else {}
                except Exception:
                    meta_dict = {}
                conf = float(meta_dict.get("confidence", 0.3))
                source = meta_dict.get("source", "")
                if conf + (0.0 if source == prefer_source else -0.05) < min_confidence:
                    continue
                rank = (
                    -conf,
                    0 if source == prefer_source else 1,
                    -float(h.get("score") or 0.0),
                )
                scored.append((rank, f"  [{room} c={conf:.2f}] {snippet}"))
        scored.sort(key=lambda kv: kv[0])
        chunks = [s for _, s in scored]
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
                                   final_status: str = "unknown",
                                   llm: Optional[str] = None) -> Dict[str, int]:
        """
        Read the per-run trajectory and emit DO/DONT/PATTERN lessons into the
        GLOBAL wing so future runs can recall them.

        New strategy (much less noisy than the old per-failure dump):

        1. Walk the trajectory and identify *causal* failure → recovery pairs:
           a `failure` (or soft-failure) observation immediately followed by a
           clearly related successful action on the same logical target.
           Standalone failures are NOT written, because they are usually noise
           that the agent recovered from in a way the regex heuristic can't see.
        2. For each pair, ask the LLM (single, low-temperature call) to summarise
           "what to do" and "what NOT to do" and emit *one* DO and *one* DON'T
           lesson, plus an optional PATTERN.
        3. If no LLM is available, emit at most one DO/DON'T per pair using a
           strict heuristic, never per-failure.

        Returns counts of lessons written.
        """
        from mempalace.palace import get_collection

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

        # Re-classify rc=0 turns whose observation contains a canonical failure
        # marker as "soft_failure" so they participate in pair detection.
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

        counts = {"do": 0, "dont": 0, "pattern": 0, "compatibility": 0,
                  "pairs_seen": 0, "pairs_kept": 0}

        # ── Step 1: build causal pairs ─────────────────────────────────────
        pairs = self._extract_recovery_pairs(rows, max_pairs=8)
        counts["pairs_seen"] = len(pairs)

        # ── Step 2: LLM-driven synthesis (preferred). ──────────────────────
        wrote_via_llm = False
        if pairs and llm:
            try:
                lessons = self._llm_summarise_pairs(pairs, llm=llm,
                                                    final_status=final_status)
            except Exception as _llm_e:
                print(f"[mempalace-distill] LLM lesson synthesis failed: "
                      f"{_llm_e}; falling back to heuristic")
                lessons = []
            for lesson in lessons or []:
                kind = (lesson.get("kind") or "").lower()
                body = (lesson.get("body") or "").strip()
                if kind not in ("do", "dont", "pattern") or not body:
                    continue
                tags = {
                    "repo": self.full_name, "sha": self.sha[:7],
                    "kind": kind,
                    "error_class": lesson.get("error_class") or "",
                    "confidence": float(lesson.get("confidence") or 0.5),
                    "source": "llm_synthesis",
                }
                self._write_lesson(gcol, kind, body, tags=tags)
                counts[kind] = counts.get(kind, 0) + 1
                counts["pairs_kept"] += 1
                wrote_via_llm = True

        # ── Step 3: heuristic fallback (one DO + one DON'T per kept pair). ─
        if pairs and not wrote_via_llm:
            for p in pairs:
                fail = p["fail"]
                fix = p["fix"]
                fail_action = (fail.get("action_content") or "").strip()
                fix_action = (fix.get("action_content") or "").strip()
                ec = fail.get("error_class") or ""
                dont_body = (
                    f"DON'T: on ROCm, running `{(fail_action.splitlines() or [''])[0][:200]}` "
                    f"failed with {ec or 'non-zero exit'}."
                )
                do_body = (
                    f"DO: when you see `{ec or 'this failure mode'}`, try "
                    f"`{(fix_action.splitlines() or [''])[0][:240]}` instead."
                )
                self._write_lesson(gcol, "dont", dont_body, tags={
                    "repo": self.full_name, "sha": self.sha[:7],
                    "kind": "dont", "error_class": ec, "source": "heuristic",
                })
                self._write_lesson(gcol, "do", do_body, tags={
                    "repo": self.full_name, "sha": self.sha[:7],
                    "kind": "do", "error_class": ec, "source": "heuristic",
                })
                counts["dont"] += 1
                counts["do"] += 1
                counts["pairs_kept"] += 1

        # ── Step 4: a single, compact summary lesson per run. ──────────────
        summary = (
            f"RUN_SUMMARY {self.full_name}@{self.sha[:7]} status={final_status}\n"
            f"turns={len(rows)} writes={self._writes} "
            f"pairs={counts['pairs_seen']} kept={counts['pairs_kept']} "
            f"duration={time.time() - self._t0:.0f}s"
        )
        self._write_lesson(gcol, "summary", summary, tags={
            "repo": self.full_name, "sha": self.sha[:7],
            "status": final_status, "kind": "summary",
        })

        return counts

    # ── Lesson-mining helpers ────────────────────────────────────────────────

    @staticmethod
    def _action_target(action: str) -> str:
        """Return a coarse 'logical target' for an action so we can decide if
        a later success belongs to the same recovery as an earlier failure.

        Examples:
            'pip install torch==2.3.0'   -> 'pip:torch'
            'apt-get install -y libfoo'  -> 'apt:libfoo'
            'python train.py --epoch 1'  -> 'py:train.py'
            'change_base_image rocm/x:1' -> 'change_base_image:rocm/x'
        """
        if not action:
            return "?"
        a = action.strip()
        first = a.split(None, 1)[0].lower()
        m = re.search(r"\bpip\d?\s+install\s+(?:-[a-zA-Z]+\s+)*([A-Za-z0-9_.\-]+)", a)
        if m:
            return f"pip:{m.group(1).lower().split('[')[0]}"
        m = re.search(r"\bapt(?:-get)?\s+install\s+(?:-[a-zA-Z]+\s+)*([A-Za-z0-9_.\-]+)", a)
        if m:
            return f"apt:{m.group(1).lower()}"
        m = re.search(r"\bpython3?\s+(\S+\.py)\b", a)
        if m:
            return f"py:{os.path.basename(m.group(1))}"
        if first in ("change_base_image", "change_python_version"):
            target = a.split(None, 1)[1].strip().split(":", 1)[0] if " " in a else ""
            return f"{first}:{target}"
        return f"{first}:{a[:40].lower()}"

    def _extract_recovery_pairs(self, rows: List[Dict[str, Any]],
                                 max_pairs: int = 8) -> List[Dict[str, Any]]:
        """Return up to `max_pairs` (failure, fix) pairs that share a target.

        We require the fix to occur within `MAX_GAP` later turns AND share the
        same coarse target (`_action_target`). This drops spurious pairs where
        an unrelated success happens to follow an unrelated failure.
        """
        MAX_GAP = 12
        pairs: List[Dict[str, Any]] = []
        seen_keys: set = set()
        for i, r in enumerate(rows):
            if r.get("outcome") not in ("failure", "soft_failure"):
                continue
            fail_target = self._action_target(r.get("action_content") or "")
            ec = r.get("error_class") or ""
            for j in range(i + 1, min(i + 1 + MAX_GAP, len(rows))):
                cand = rows[j]
                if cand.get("outcome") != "success":
                    continue
                fix_target = self._action_target(cand.get("action_content") or "")
                # Same target OR same error class continuation -> probable recovery.
                if fix_target == fail_target or (ec and ec == cand.get("recovered_error_class")):
                    key = (fail_target, ec)
                    if key in seen_keys:
                        break
                    seen_keys.add(key)
                    pairs.append({
                        "fail": r, "fix": cand,
                        "target": fail_target, "error_class": ec,
                        "gap_turns": j - i,
                    })
                    break
            if len(pairs) >= max_pairs:
                break
        return pairs

    def _llm_summarise_pairs(self, pairs: List[Dict[str, Any]], llm: str,
                              final_status: str) -> List[Dict[str, Any]]:
        """One LLM call -> list of {kind, body, error_class, confidence} dicts.

        We deliberately give the model the failure/fix pairs ONLY (not the full
        trajectory) so it can't over-generalise from incidental noise. We ask
        for at most 5 lessons total.
        """
        if not pairs:
            return []
        from utils.llm import get_llm_response

        sketches: List[str] = []
        for k, p in enumerate(pairs):
            fail = p["fail"]
            fix = p["fix"]
            fail_obs = (fail.get("observation_raw") or "")[:600]
            fail_act = (fail.get("action_content") or "")[:300]
            fix_act = (fix.get("action_content") or "")[:300]
            sketches.append(
                f"PAIR {k+1}: target={p['target']} error_class={p['error_class']!r} "
                f"gap={p['gap_turns']} turns\n"
                f"  failed_action: {fail_act}\n"
                f"  failure_excerpt: {fail_obs}\n"
                f"  recovery_action: {fix_act}\n"
            )
        sketches_text = "\n".join(sketches)

        prompt = (
            "You are mining ONE Repo2ROCm build trajectory for cross-run "
            "lessons. Each PAIR below is a moment where the agent did "
            "something wrong on ROCm and then did the right thing.\n\n"
            "Your job: for the WHOLE batch, propose at most 5 lessons that\n"
            " - are GENERAL (apply to any repo with the same failure mode),\n"
            " - are ACTIONABLE (a future agent can copy/paste the recovery),\n"
            " - DROP repo-specific names, file paths, and version numbers,\n"
            " - merge near-duplicates into a single lesson.\n\n"
            "Return STRICT JSON of the form:\n"
            '[{"kind":"do|dont|pattern", "error_class":"...", '
            '"confidence":0.0-1.0, "body":"<<one paragraph>>"}]\n\n'
            "Use kind=do for 'when you see X, do Y'.\n"
            "Use kind=dont for 'do not do Z on ROCm because ...'.\n"
            "Use kind=pattern for 'when error matches /regex/, the fix is ...'\n"
            "Each body MUST be self-contained (a future agent will see it "
            "without the rest of the trajectory).\n\n"
            f"Final run status: {final_status}\n"
            f"Repo: {self.full_name}@{self.sha[:7]}\n\n"
            f"PAIRS:\n{sketches_text}\n"
        )

        messages = [{"role": "user", "content": prompt}]
        try:
            response, _usage = get_llm_response(
                llm, messages, temperature=0.1, max_tokens=900,
            )
        except Exception as _e:
            print(f"[mempalace-distill] llm call failed: {_e}")
            return []
        if not response or not response[0]:
            return []
        text = response[0].strip()
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        try:
            data = json.loads(text)
        except Exception:
            m = re.search(r"\[.*\]", text, re.DOTALL)
            if not m:
                return []
            try:
                data = json.loads(m.group(0))
            except Exception:
                return []
        if not isinstance(data, list):
            return []
        cleaned: List[Dict[str, Any]] = []
        for item in data[:5]:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "").strip().lower()
            body = str(item.get("body") or "").strip()
            if kind not in ("do", "dont", "pattern") or not body:
                continue
            try:
                conf = float(item.get("confidence", 0.5))
            except (TypeError, ValueError):
                conf = 0.5
            conf = max(0.0, min(1.0, conf))
            cleaned.append({
                "kind": kind,
                "body": body[:1500],
                "error_class": str(item.get("error_class") or "")[:80],
                "confidence": conf,
            })
        return cleaned

    @staticmethod
    def _write_lesson(global_col, room: str, content: str,
                      tags: Optional[Dict[str, Any]] = None) -> None:
        """
        Append a lesson drawer to the global wing.

        IMPORTANT: every drawer needs a unique (source_file, chunk_index) pair
        because mempalace/Chroma dedupes by a hash derived from those. We
        derive a deterministic unique id from the content so identical lessons
        from re-runs collapse, but distinct lessons across runs accumulate.
        """
        from mempalace.miner import add_drawer
        if tags:
            try:
                content = content.rstrip() + "\n\n[META] " + json.dumps(
                    tags, default=str, sort_keys=True)
            except Exception:
                pass
        try:
            cid = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:16]
            repo_part = ""
            if isinstance(tags, dict):
                repo_part = "_" + str(tags.get("repo", "")).replace("/", "_")[:40]
            source_file = f"lessons:{room}{repo_part}:{cid}"
            add_drawer(global_col, _GLOBAL_WING, room, content,
                       source_file=source_file, chunk_index=0,
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
