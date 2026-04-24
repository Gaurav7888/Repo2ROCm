"""
Researcher sub-agent — PR-C (deep web search loop).

A bounded LLM loop that turns ONE big question from the parent configuration
agent into ONE compact ResearchNote, by iteratively choosing among a small
palette of free, cached tools:

    search   (DDG)              -> top-k hits
    visit    (urllib + html2text) -> readable page text
    pypi     (PyPI JSON)         -> package versions
    docker   (Docker Hub)        -> image tags
    recall   (mempalace --global) -> prior cross-run lessons
    finish   (JSON ResearchNote)  -> terminate

The point: the parent agent spends ZERO turns reading search snippets. It just
asks `deep_research "..."` once and gets back a structured answer with
citations and (when applicable) verified install commands.

Cost / safety:
- max_turns:    6 (default)  — hard cap on internal LLM rounds
- budget_s:    90 (default)  — wall-clock cap; aborts gracefully
- max_calls:   12 (default)  — hard cap on tool invocations across turns
- cache_ttl:   14 days       — mempalace `room="research_notes"` global wing
- soft-fail:   any tool error becomes a `[error] ...` observation; the loop
               continues. No exceptions ever escape research().

Result is a typed dict (kept dict, not dataclass, so json round-trips trivially):

    {
      "question": str,
      "answer": str,                    # 1-3 paragraph distilled answer
      "suggested_commands": [str, ...], # bash/pip lines we believe will work
      "citations": [{"title": str, "url": str}, ...],
      "confidence": float,              # 0..1, self-reported
      "turns_used": int,
      "tool_calls": int,
      "wall_time_s": float,
      "stopped_reason": str,            # "finish" | "max_turns" | "budget_s" | ...
      "error": str | "",                # non-empty if everything failed
    }
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple


# ── Defaults ─────────────────────────────────────────────────────────────────

_DEFAULT_MAX_TURNS = 6
_DEFAULT_BUDGET_S = 90.0
_DEFAULT_MAX_CALLS = 12
_DEFAULT_CACHE_TTL_S = 14 * 24 * 3600
_DEFAULT_LLM = "claude-sonnet-4"

_GLOBAL_WING = "rocm_global_lessons"
_CACHE_ROOM = "research_notes"


# ── Cache (re-uses the same drawer convention as web_search/external_lookups) ─

def _palace_global_path() -> str:
    return os.path.expanduser("~/.mempalace/palaces/_global")


def _cache_id(question: str) -> str:
    return hashlib.sha256(("deep_research::" + question.strip().lower()).encode()).hexdigest()[:24]


def _cache_get(question: str, max_age_s: int) -> Optional[Dict[str, Any]]:
    try:
        from mempalace.searcher import search_memories
    except Exception:
        return None
    cid = _cache_id(question)
    try:
        r = search_memories(
            f"deep_research {question}", _palace_global_path(),
            wing=_GLOBAL_WING, room=_CACHE_ROOM,
            n_results=8, max_distance=0.0,
        )
    except Exception:
        return None
    now = time.time()
    for hit in (r.get("results") or []):
        text = hit.get("text") or ""
        if cid not in text:
            continue
        meta_idx = text.rfind("\n[META]")
        if meta_idx < 0:
            continue
        try:
            meta = json.loads(text[meta_idx + len("\n[META]"):].strip())
        except Exception:
            continue
        if now - float(meta.get("ts", 0)) > int(meta.get("ttl_s", _DEFAULT_CACHE_TTL_S)):
            continue
        body = text[:meta_idx]
        marker = f"[CACHE_ID {cid}]\n"
        if body.startswith(marker):
            body = body[len(marker):]
        # Try parse JSON note
        try:
            note = json.loads(body)
            note["_cache_hit"] = True
            return note
        except Exception:
            return {"question": question, "answer": body, "_cache_hit": True,
                    "citations": [], "suggested_commands": [], "confidence": 0.5,
                    "turns_used": 0, "tool_calls": 0, "wall_time_s": 0.0,
                    "stopped_reason": "cache", "error": ""}
    return None


def _cache_put(question: str, note: Dict[str, Any], ttl_s: int = _DEFAULT_CACHE_TTL_S) -> None:
    try:
        from mempalace.miner import add_drawer
        from mempalace.palace import get_collection
    except Exception:
        return
    try:
        os.makedirs(_palace_global_path(), exist_ok=True)
        col = get_collection(_palace_global_path(), create=True)
    except Exception:
        return
    cid = _cache_id(question)
    meta = {"kind": "deep_research", "key": question[:200],
            "ttl_s": int(ttl_s), "ts": time.time(), "cache_id": cid,
            "confidence": float(note.get("confidence", 0.0))}
    try:
        body = json.dumps(note, default=str, indent=2)
    except Exception:
        body = str(note)
    try:
        content = (
            f"[CACHE_ID {cid}]\n{body.rstrip()}\n\n[META] "
            + json.dumps(meta, default=str, sort_keys=True)
        )
        add_drawer(col, _GLOBAL_WING, _CACHE_ROOM, content,
                   source_file=f"deep_research:{cid}", chunk_index=0,
                   agent="researcher")
    except Exception as e:
        print(f"[researcher] cache put failed: {e}")


# ── Tool palette (thin wrappers — soft-fail, return short strings) ────────────

def _tool_search(query: str) -> Tuple[str, str]:
    try:
        from tools.web_search import web_search
        body, rc = web_search(query, max_results=5)
        return body if rc == 0 else f"[error] {body.strip()}", "search"
    except Exception as e:
        return f"[error] search failed: {e}", "search"


def _tool_visit(url: str) -> Tuple[str, str]:
    try:
        from tools.web_search import visit_url
        body, rc = visit_url(url, max_chars=4000)
        return body if rc == 0 else f"[error] {body.strip()}", "visit"
    except Exception as e:
        return f"[error] visit failed: {e}", "visit"


def _tool_pypi(pkg: str) -> Tuple[str, str]:
    try:
        from tools.external_lookups import pypi_versions
        body, rc = pypi_versions(pkg, limit=10)
        return body if rc == 0 else f"[error] {body.strip()}", "pypi"
    except Exception as e:
        return f"[error] pypi failed: {e}", "pypi"


def _tool_docker(image: str) -> Tuple[str, str]:
    try:
        from tools.external_lookups import dockerhub_tags
        body, rc = dockerhub_tags(image, limit=10)
        return body if rc == 0 else f"[error] {body.strip()}", "docker"
    except Exception as e:
        return f"[error] docker failed: {e}", "docker"


def _tool_recall(question: str) -> Tuple[str, str]:
    try:
        from learning.mempalace_provider import RunMemory
        # Use a transient run-memory instance scoped to a "researcher" pseudo-wing
        # so we can hit the global lessons. We don't want to write to a real wing.
        try:
            mem = RunMemory.create("researcher/anon", "0000000")
            pack = mem.recall_global_lessons(
                question, n_per_room=3, token_budget=800,
            ) or ""
            if not pack.strip():
                return f"[recall] no global lessons for {question!r}", "recall"
            return pack, "recall"
        except Exception as e:
            return f"[error] recall failed: {e}", "recall"
    except Exception as e:
        return f"[error] recall import failed: {e}", "recall"


# ── LLM-driven loop ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are the Researcher sub-agent for Repo2ROCm. Your sole job is to answer ONE
question from the parent configuration agent, using the tools listed below.

Tools (each call = ONE tool, exactly one line, no quotes around the tool name):

    search <query>          DuckDuckGo. Snippets only. ~5 hits.
    visit <url>             Fetch a URL, returns extracted text (max 4000 chars).
    pypi <pkg>              List recent PyPI versions + dates.
    docker <image>          List recent Docker Hub tags + dates.
    recall <question>       Query the cumulative ROCm knowledge base for prior do/dont/pattern lessons.
    finish <json>           Emit the final ResearchNote and terminate. JSON shape:
        {"answer": "...",
         "suggested_commands": ["cmd1", "cmd2"],
         "citations": [{"title":"...","url":"..."}, ...],
         "confidence": 0.0..1.0}

Strategy:
1. ALWAYS start with `recall <question>`. Most ROCm problems have already been
   solved in a prior run; if a confident answer comes back, you can finish in
   2-3 turns.
2. If recall is empty/weak, use `search` with terms that include both the
   error class and the ROCm/AMD context.
3. Pick ONE high-signal hit (prefer github.com/pytorch, github.com/ROCm,
   rocm.docs.amd.com, huggingface.co/transformers issues). `visit` it.
4. If you need a second source (dependency version, image tag), use `pypi` or
   `docker`.
5. Synthesize. `finish` with a 1-3 paragraph answer + the actual commands the
   parent should try, plus citations. Be HONEST about confidence.

Constraints:
- Output exactly ONE tool call per turn, no other text.
- Tool calls are stateless: the user message will show the tool result.
- Do NOT search more than twice without a `visit` in between.
- Do NOT visit non-essential pages (e.g. docs.python.org for stdlib).
- If you cannot find a confident answer in the budget, `finish` anyway with
  whatever partial information you have and a low `confidence`.
"""


def _parse_tool(line: str) -> Optional[Tuple[str, str]]:
    """Parse one of: search/visit/pypi/docker/recall/finish followed by an arg."""
    s = line.strip()
    # Strip ```bash fences if the LLM wraps the call in one
    s = re.sub(r"^```(?:bash|json)?\n?", "", s)
    s = re.sub(r"\n?```$", "", s)
    s = s.strip()
    m = re.match(r"^(search|visit|pypi|docker|recall|finish)\s+(.+)$",
                 s, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    tool = m.group(1).lower()
    arg = m.group(2).strip()
    # Strip surrounding quotes for the simple tools
    if tool in ("search", "visit", "pypi", "docker", "recall"):
        if (arg.startswith('"') and arg.endswith('"')) or (arg.startswith("'") and arg.endswith("'")):
            arg = arg[1:-1]
    return tool, arg


def _format_observation(tool_name: str, output: str, max_chars: int = 3500) -> str:
    o = output.strip()
    if len(o) > max_chars:
        o = o[: max_chars - 32] + "\n…[truncated]"
    return f"[{tool_name} result]\n{o}"


def _safe_finish(arg: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Parse the LLM's finish JSON; on failure, fall back to a partial note."""
    try:
        # Strip leading "finish" residue if any & accept stray text around JSON
        m = re.search(r"\{.*\}", arg, flags=re.DOTALL)
        if not m:
            raise ValueError("no JSON object found in finish arg")
        note = json.loads(m.group(0))
    except Exception as e:
        note = {
            "answer": "Researcher could not produce a structured answer "
                      f"(parse error: {e}). Raw arg: {arg[:600]}",
            "suggested_commands": [],
            "citations": [],
            "confidence": 0.0,
            "_parse_error": str(e),
        }
    # Normalize types
    note.setdefault("answer", "")
    note.setdefault("suggested_commands", [])
    note.setdefault("citations", [])
    note["confidence"] = float(note.get("confidence", 0.0) or 0.0)
    if not isinstance(note["suggested_commands"], list):
        note["suggested_commands"] = [str(note["suggested_commands"])]
    if not isinstance(note["citations"], list):
        note["citations"] = []
    return note


_SYNTH_SYSTEM_PROMPT = """\
You are the Synthesizer for Repo2ROCm's deep_research tool. The orchestrator
has already gathered raw evidence (cumulative-KB recall + web search hits +
fetched page extracts) for one ROCm/CUDA migration question. Your ONLY job is
to read that evidence and produce ONE compact JSON ResearchNote.

Output rules (STRICT — your reply will be JSON-parsed):
- Reply with EXACTLY one JSON object, no prose before or after, no Markdown
  fence, no comments. The object MUST contain these keys:

    "answer":              str   1-3 short paragraphs, plain prose. Quote
                                  exact error strings and version numbers.
                                  Tie the answer to the user question.
    "suggested_commands":  list  Bash/pip lines you believe will fix the
                                  problem on AMD ROCm. Empty list if you can
                                  only recommend reading. Do NOT invent
                                  commands; cite-or-omit.
    "citations":           list  [{"title": str, "url": str}, ...]. Pull
                                  these directly from the evidence; do NOT
                                  invent URLs. Up to 6.
    "confidence":          float 0.0..1.0. Be honest. < 0.3 = "I'm
                                  guessing"; 0.3-0.6 = "plausible but worth
                                  trying"; > 0.6 = "evidence directly
                                  supports this".

- If the evidence is contradictory or incomplete, say so in `answer` and
  drop `confidence`.
- Do NOT recommend installing the CUDA-only PyPI flash-attn wheel on AMD;
  always prefer build-from-source with FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
  or PyTorch SDPA fallback.
- Prefer ungated HuggingFace models when the gated original is referenced.
"""


def _gather_evidence(question: str, max_search_hits: int = 5,
                     max_visits: int = 2,
                     verbose: bool = False) -> Tuple[str, List[Dict[str, str]], int]:
    """
    Deterministic phase: recall + search + visit top-K. Returns
        (evidence_text, citation_list, tool_call_count)
    """
    parts: List[str] = []
    cites: List[Dict[str, str]] = []
    calls = 0

    # 1. Cumulative KB recall (free)
    rc_text, _ = _tool_recall(question)
    calls += 1
    parts.append("=== CUMULATIVE-KB RECALL ===")
    parts.append(rc_text)
    if verbose:
        print(f"[researcher] recall: {len(rc_text)} chars")

    # 2. Web search
    sr_text, _ = _tool_search(question)
    calls += 1
    parts.append("\n=== WEB SEARCH ===")
    parts.append(sr_text)
    if verbose:
        print(f"[researcher] search: {len(sr_text)} chars")

    # Extract URLs + titles from the search output for visit candidates.
    candidates: List[Dict[str, str]] = []
    # The web_search format we control:
    # 1. <title>
    #    <url>
    #    <snippet>
    lines = sr_text.splitlines()
    i = 0
    while i < len(lines) and len(candidates) < max_search_hits:
        m = re.match(r"^\s*\d+\.\s+(.+?)\s*$", lines[i])
        if m:
            title = m.group(1).strip()
            url = ""
            if i + 1 < len(lines):
                u = lines[i + 1].strip()
                if u.startswith("http"):
                    url = u
            if url:
                candidates.append({"title": title, "url": url})
        i += 1

    # 3. Visit the top-N hits, prioritizing high-signal sources.
    def _score(c: Dict[str, str]) -> int:
        u = c["url"].lower()
        s = 0
        if "github.com/pytorch" in u: s += 4
        if "github.com/rocm" in u: s += 4
        if "rocm.docs.amd.com" in u: s += 4
        if "github.com/dao-ailab/flash-attention" in u: s += 3
        if "github.com/huggingface" in u: s += 3
        if "github.com/" in u: s += 2
        if "stackoverflow.com" in u: s += 1
        if "huggingface.co" in u: s += 1
        return s

    candidates.sort(key=_score, reverse=True)
    visited = 0
    for c in candidates:
        if visited >= max_visits:
            break
        v_text, _ = _tool_visit(c["url"])
        calls += 1
        visited += 1
        parts.append(f"\n=== VISIT: {c['title'][:80]} | {c['url']} ===")
        parts.append(v_text)
        cites.append({"title": c["title"], "url": c["url"]})
        if verbose:
            print(f"[researcher] visit: {c['url']} → {len(v_text)} chars")

    # Also keep all not-yet-visited candidates as citation candidates.
    for c in candidates:
        if not any(x.get("url") == c["url"] for x in cites):
            cites.append({"title": c["title"], "url": c["url"]})

    return "\n".join(parts), cites[:6], calls


def research(question: str,
             llm: str = _DEFAULT_LLM,
             max_turns: int = _DEFAULT_MAX_TURNS,        # kept for API compat
             budget_s: float = _DEFAULT_BUDGET_S,
             max_calls: int = _DEFAULT_MAX_CALLS,        # kept for API compat
             use_cache: bool = True,
             cache_ttl_s: int = _DEFAULT_CACHE_TTL_S,
             verbose: bool = False) -> Dict[str, Any]:
    """
    Run the deep research sub-agent. Returns a ResearchNote dict (always).

    Two-phase implementation:
      Phase 1: deterministic evidence gathering (recall + search + top-K visits).
               No LLM, fully reliable, ~3-8s end-to-end.
      Phase 2: ONE LLM synthesis call that emits a structured JSON ResearchNote
               from the gathered evidence. ~5-10s.

    `max_turns` and `max_calls` are kept for API compatibility but not strictly
    enforced — the deterministic pipeline always uses ~3 calls (recall + search +
    1-2 visits) plus 1 LLM call for synthesis. `budget_s` IS enforced.
    """
    q = (question or "").strip()
    if not q:
        return _make_result(q, "deep_research: empty question.", [], [], 0.0,
                            0, 0, 0.0, "empty_question", "empty question")
    if use_cache:
        hit = _cache_get(q, cache_ttl_s)
        if hit:
            hit["_cache_hit"] = True
            return hit

    t0 = time.time()
    stopped = "finish"
    note: Optional[Dict[str, Any]] = None

    # ── Phase 1: gather evidence (no LLM) ──
    try:
        evidence, citations, tool_calls = _gather_evidence(
            q, max_search_hits=5, max_visits=2, verbose=verbose,
        )
    except Exception as e:
        return _make_result(q,
            f"deep_research: evidence gathering failed: {e}",
            [], [], 0.0, 0, 0, time.time() - t0, "gather_error", str(e))

    if time.time() - t0 > budget_s:
        return _make_result(q,
            f"deep_research: evidence gathering exhausted budget ({budget_s}s).",
            [], citations, 0.0, 0, tool_calls, time.time() - t0, "budget_s", "")

    # Hard-cap evidence size before sending to LLM.
    MAX_EVIDENCE_CHARS = 16000
    if len(evidence) > MAX_EVIDENCE_CHARS:
        evidence = evidence[: MAX_EVIDENCE_CHARS - 32] + "\n…[evidence truncated]\n"

    # ── Phase 2: single LLM synthesis call ──
    try:
        from utils.llm import get_llm_response
    except Exception as e:
        return _make_result(q,
            f"deep_research: LLM client unavailable ({e}); returning raw evidence only.",
            [], citations, 0.0, 0, tool_calls, time.time() - t0, "no_llm", str(e))

    user_msg = (
        f"QUESTION:\n{q}\n\n"
        f"EVIDENCE GATHERED (cumulative-KB recall + web search + top page extracts):\n"
        f"{evidence}\n\n"
        f"Now emit the JSON ResearchNote per the system rules. JSON ONLY."
    )

    try:
        choices, _usage = get_llm_response(
            llm,
            [{"role": "user", "content": user_msg}],
            system_prompt=_SYNTH_SYSTEM_PROMPT,
            temperature=0.1, max_tokens=1200,
        )
        reply = (choices or [""])[0] or ""
    except Exception as e:
        return _make_result(q,
            f"deep_research: synthesis LLM call failed: {e}; returning raw evidence.",
            [], citations, 0.0, 0, tool_calls, time.time() - t0, "llm_error", str(e))

    note = _safe_finish(reply, {})
    # Backfill citations from deterministic gathering if the LLM didn't supply any.
    if not note.get("citations"):
        note["citations"] = citations
    elapsed = time.time() - t0
    turns = 1  # one synthesis turn

    result = _make_result(
        q, note.get("answer", ""), note.get("suggested_commands", []),
        note.get("citations", []), float(note.get("confidence", 0.0) or 0.0),
        turns, tool_calls, elapsed, stopped, note.get("_parse_error", "") or "",
    )

    if use_cache:
        try:
            _cache_put(q, result, cache_ttl_s)
        except Exception:
            pass
    return result


def _make_result(question: str, answer: str, cmds: list, citations: list,
                 confidence: float, turns: int, tool_calls: int,
                 wall_time_s: float, stopped: str, error: str) -> Dict[str, Any]:
    return {
        "question": question,
        "answer": answer,
        "suggested_commands": list(cmds or []),
        "citations": list(citations or []),
        "confidence": float(confidence),
        "turns_used": int(turns),
        "tool_calls": int(tool_calls),
        "wall_time_s": round(float(wall_time_s), 2),
        "stopped_reason": stopped,
        "error": error,
    }


def format_for_observation(note: Dict[str, Any], max_chars: int = 1800) -> str:
    """Render a ResearchNote as a compact prompt-friendly string."""
    parts: List[str] = []
    cache_marker = " [cache hit]" if note.get("_cache_hit") else ""
    parts.append(
        f"deep_research{cache_marker}: confidence={note.get('confidence', 0):.2f} "
        f"turns={note.get('turns_used', 0)} calls={note.get('tool_calls', 0)} "
        f"time={note.get('wall_time_s', 0)}s stopped={note.get('stopped_reason', '?')}"
    )
    ans = (note.get("answer") or "").strip()
    if ans:
        parts.append("\n--- ANSWER ---\n" + ans)
    cmds = note.get("suggested_commands") or []
    if cmds:
        parts.append("\n--- SUGGESTED COMMANDS ---")
        for c in cmds[:6]:
            parts.append("  $ " + str(c).strip()[:240])
    cites = note.get("citations") or []
    if cites:
        parts.append("\n--- CITATIONS ---")
        for c in cites[:6]:
            t = (c.get("title") if isinstance(c, dict) else "") or ""
            u = (c.get("url") if isinstance(c, dict) else str(c)) or ""
            parts.append(f"  - {t[:80]} {u[:160]}")
    if note.get("error"):
        parts.append(f"\n[note] {note['error']}")
    out = "\n".join(parts).strip() + "\n"
    if len(out) > max_chars:
        out = out[: max_chars - 32] + "\n…[truncated]\n"
    return out
