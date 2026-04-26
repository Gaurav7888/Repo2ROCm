"""
Profile-based research worker for Repo2ROCm.

The same bounded helper is reused across:
  - runtime `deep_research` during configuration,
  - planner-side repository/image/dependency investigation,
  - paper-side metric/setup/reproducibility clarification,
  - observer-side diagnosis (via a separate profile).

Each profile keeps its own synthesis guidance and cache namespace while sharing
the same deterministic evidence-gathering substrate:

    recall  -> policy / compatibility note
    pypi    -> package versions
    docker  -> Docker Hub tags
    search  -> DDG web results
    visit   -> extracted page text

The worker spends zero parent turns reading snippets. Callers ask one focused
question and receive one compact ResearchNote with citations and, when
applicable, suggested commands or follow-up questions.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from utils.json_utils import load_json_loose


# ── Defaults ─────────────────────────────────────────────────────────────────

_DEFAULT_MAX_TURNS = 6
_DEFAULT_BUDGET_S = 90.0
_DEFAULT_MAX_CALLS = 12
_DEFAULT_CACHE_TTL_S = 14 * 24 * 3600
_DEFAULT_LLM = "claude-sonnet-4"

_GLOBAL_WING = "rocm_global_lessons"
_CACHE_ROOM = "research_notes"

_AMD_TERMS = ("amd", "rocm", "hip", "gfx", "miopen", "rocblas", "rccl", "amdgpu")
_PKG_HINTS = (
    "flash-attn", "flash_attn", "bitsandbytes", "xformers", "triton",
    "deepspeed", "transformers", "torch", "vllm", "sglang", "accelerate",
    "peft", "datasets",
)
_IMAGE_HINTS = (
    "rocm/pytorch", "rocm/vllm", "rocm/sgl-dev", "rocm/tensorflow",
    "rocm/jax", "rocm/megatron-lm", "rocm/onnxruntime",
)


# ── Research profiles ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ResearchProfile:
    name: str
    cache_namespace: str
    search_hint: str
    synth_guidance: str
    augment_rocm_terms: bool = True
    include_lookup_targets: bool = True
    include_recall_note: bool = True
    max_search_hits: int = 5
    max_visits: int = 2


_RUNTIME_REPAIR_GUIDANCE = """\
You are acting as a runtime repair specialist for AMD ROCm / HIP migrations.
- Prioritize concrete fixes for the exact failure or version conflict at hand.
- Prefer commands only when the evidence directly supports them.
- Do NOT recommend CUDA-only wheels on AMD unless the evidence explicitly says
  they work on ROCm.
- Prefer ROCm-native or PyTorch-native fallbacks over guessy package pinning.
"""

_REPO_RESEARCH_GUIDANCE = """\
You are assisting the planner before execution starts.
- Focus on the safest ROCm image choice, dependency strategy, and known caveats.
- Keep the answer planner-friendly: concise, comparative, and grounded.
- Suggested commands are optional; use them only when they are directly useful
  to the planner (for example a known source-build invocation or verified image
  tag).
"""

_PAPER_RESEARCH_GUIDANCE = """\
You are clarifying research-paper details for reproducibility.
- Focus on the metric semantics, setup details, and what is portable across GPU
  vendors.
- Prefer exact wording for experiment names, tables, figures, datasets,
  hyperparameters, and evaluation caveats.
- If ambiguity remains, use `followups` to tell the parent what exact question
  should be asked next or what evidence is still missing.
- Do not invent shell commands unless the evidence directly ties them to the
  question.
"""

_OBSERVER_GUIDANCE = """\
You are an observer-side critic for Repo2ROCm.
- Diagnose progress quality from recent turns without executing anything.
- Proactively use web-grounded evidence when the recent turns suggest the run is
  stalled on a compatibility, runtime, or paper-fidelity issue.
- If the run is genuinely making progress, say so and keep `suggested_commands`
  empty.
- If the run is stuck, propose a higher-level corrective strategy rather than a
  regex-like rule. Examples: switch from trial-installing to verified version
  lookup, revisit the planner assumptions, or verify the paper metric path
  before rerunning.
- Use `followups` for high-value questions or tools the parent should invoke
  next.
"""

_PROFILES = {
    "runtimeRepair": ResearchProfile(
        name="runtimeRepair",
        cache_namespace="runtime_repair",
        search_hint="AMD ROCm HIP",
        synth_guidance=_RUNTIME_REPAIR_GUIDANCE,
        augment_rocm_terms=True,
        include_lookup_targets=True,
        include_recall_note=True,
        max_search_hits=5,
        max_visits=2,
    ),
    "repoResearch": ResearchProfile(
        name="repoResearch",
        cache_namespace="repo_research",
        search_hint="AMD ROCm image compatibility dependency strategy",
        synth_guidance=_REPO_RESEARCH_GUIDANCE,
        augment_rocm_terms=True,
        include_lookup_targets=True,
        include_recall_note=True,
        max_search_hits=5,
        max_visits=2,
    ),
    "paperResearch": ResearchProfile(
        name="paperResearch",
        cache_namespace="paper_research",
        search_hint="paper metric setup appendix benchmark reproducibility",
        synth_guidance=_PAPER_RESEARCH_GUIDANCE,
        augment_rocm_terms=False,
        include_lookup_targets=True,
        include_recall_note=True,
        max_search_hits=5,
        max_visits=2,
    ),
    "observerCritic": ResearchProfile(
        name="observerCritic",
        cache_namespace="observer_critic",
        search_hint="AMD ROCm diagnosis strategy",
        synth_guidance=_OBSERVER_GUIDANCE,
        augment_rocm_terms=True,
        include_lookup_targets=True,
        include_recall_note=False,
        max_search_hits=4,
        max_visits=2,
    ),
}


def _resolve_profile(profile: str | None) -> ResearchProfile:
    key = (profile or "runtimeRepair").strip()
    if key in _PROFILES:
        return _PROFILES[key]
    lowered = key.lower()
    for profile_obj in _PROFILES.values():
        if profile_obj.name.lower() == lowered:
            return profile_obj
    return _PROFILES["runtimeRepair"]


def _context_to_text(context: Any) -> str:
    if context is None:
        return ""
    if isinstance(context, str):
        return context.strip()
    try:
        return json.dumps(context, indent=2, default=str)[:12000]
    except Exception:
        return str(context)[:12000]


# ── Cache (re-uses the same drawer convention as web_search/external_lookups) ─

def _palace_global_path() -> str:
    return os.path.expanduser("~/.mempalace/palaces/_global")


def _cache_id(question: str, profile: str, context_text: str = "") -> str:
    key = (
        "research_worker::"
        + profile.strip().lower()
        + "::"
        + question.strip().lower()
        + "::"
        + context_text.strip().lower()
    )
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def _cache_get(question: str, profile: str, max_age_s: int,
               context_text: str = "") -> Optional[Dict[str, Any]]:
    try:
        from mempalace.searcher import search_memories
    except Exception:
        return None
    cid = _cache_id(question, profile, context_text=context_text)
    try:
        r = search_memories(
            f"research_worker {profile} {question}", _palace_global_path(),
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
            return {
                "question": question,
                "answer": body,
                "_cache_hit": True,
                "citations": [],
                "suggested_commands": [],
                "confidence": 0.5,
                "turns_used": 0,
                "tool_calls": 0,
                "wall_time_s": 0.0,
                "stopped_reason": "cache",
                "error": "",
                "profile_used": profile,
                "followups": [],
            }
    return None


def _cache_put(question: str, profile: str, note: Dict[str, Any],
               ttl_s: int = _DEFAULT_CACHE_TTL_S,
               context_text: str = "") -> None:
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
    cid = _cache_id(question, profile, context_text=context_text)
    meta = {"kind": "research_worker", "profile": profile, "key": question[:200],
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
                   source_file=f"research_worker:{profile}:{cid}", chunk_index=0,
                   agent="researcher")
    except Exception as e:
        print(f"[researcher] cache put failed: {e}")


def _augment_rocm_query(question: str, suffix: str = "") -> str:
    q = (question or "").strip()
    if not q:
        return ""
    lowered = q.lower()
    if any(term in lowered for term in _AMD_TERMS):
        return " ".join(part for part in (q, suffix.strip()) if part).strip()
    return " ".join(part for part in (q, "AMD ROCm HIP", suffix.strip()) if part).strip()


def _infer_lookup_targets(question: str, context_text: str = "") -> Tuple[List[str], List[str]]:
    lowered = f"{question or ''}\n{context_text or ''}".lower()
    pkgs = []
    for pkg in _PKG_HINTS:
        if pkg in lowered and pkg not in pkgs:
            pkgs.append(pkg)

    images = []
    for image in _IMAGE_HINTS:
        if image in lowered and image not in images:
            images.append(image)

    if not images:
        if "rocm" in lowered and "pytorch" in lowered:
            images.append("rocm/pytorch")
        elif "rocm" in lowered and "vllm" in lowered:
            images.append("rocm/vllm")
        elif "rocm" in lowered and "sglang" in lowered:
            images.append("rocm/sgl-dev")

    return pkgs[:2], images[:1]


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
    return (
        "[recall] Cross-run natural-language lesson recall is disabled. "
        "Use live repo evidence, deterministic lookups, and current web results "
        "instead of relying on free-form lessons from prior repos.",
        "recall",
    )


def _safe_finish(arg: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Parse the LLM's finish JSON; on failure, fall back to a partial note."""
    try:
        note = load_json_loose(arg, expected="object")
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
    note.setdefault("followups", [])
    note["confidence"] = float(note.get("confidence", 0.0) or 0.0)
    if not isinstance(note["suggested_commands"], list):
        note["suggested_commands"] = [str(note["suggested_commands"])]
    if not isinstance(note["citations"], list):
        note["citations"] = []
    if not isinstance(note["followups"], list):
        note["followups"] = [str(note["followups"])]
    return note


_BASE_SYNTH_SYSTEM_PROMPT = """\
You are the Synthesizer for Repo2ROCm's research worker. The orchestrator has
already gathered raw evidence (policy note + deterministic lookups + web search
hits + fetched page extracts) for one focused question. Your ONLY job is to
read that evidence and produce ONE compact JSON ResearchNote.

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
    "followups":           list  Zero or more focused next questions the
                                  parent should ask when ambiguity remains.

- If the evidence is contradictory or incomplete, say so in `answer` and
  drop `confidence`.
"""


def _build_synth_system_prompt(profile_obj: ResearchProfile) -> str:
    return _BASE_SYNTH_SYSTEM_PROMPT + "\n\nPROFILE GUIDANCE:\n" + profile_obj.synth_guidance.strip() + "\n"


def _score_visit_candidate(url: str) -> int:
    u = (url or "").lower()
    score = 0
    if "github.com/pytorch" in u:
        score += 4
    if "github.com/rocm" in u:
        score += 4
    if "rocm.docs.amd.com" in u:
        score += 4
    if "github.com/dao-ailab/flash-attention" in u:
        score += 3
    if "github.com/huggingface" in u:
        score += 3
    if "github.com/" in u:
        score += 2
    if "huggingface.co" in u:
        score += 2
    if "arxiv.org" in u or "ar5iv" in u:
        score += 2
    if "stackoverflow.com" in u:
        score += 1
    return score


def _extract_search_candidates(search_text: str,
                               max_search_hits: int) -> List[Dict[str, str]]:
    candidates: List[Dict[str, str]] = []
    lines = (search_text or "").splitlines()
    i = 0
    while i < len(lines) and len(candidates) < max_search_hits:
        match = re.match(r"^\s*\d+\.\s+(.+?)\s*$", lines[i])
        if match:
            title = match.group(1).strip()
            url = ""
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith("http"):
                    url = next_line
            if url:
                candidates.append({"title": title, "url": url})
        i += 1
    candidates.sort(key=lambda item: _score_visit_candidate(item["url"]), reverse=True)
    return candidates


def _build_search_query(question: str,
                        profile_obj: ResearchProfile,
                        context_text: str) -> str:
    q = (question or "").strip()
    if not q:
        return ""
    context_bits = []
    if context_text:
        for line in context_text.splitlines():
            cleaned = line.strip()
            if cleaned:
                context_bits.append(cleaned[:120])
            if len(context_bits) >= 2:
                break
    suffix = profile_obj.search_hint
    if context_bits:
        suffix = f"{suffix} {' '.join(context_bits)}".strip()
    if profile_obj.augment_rocm_terms:
        return _augment_rocm_query(q, suffix=suffix)
    return " ".join(part for part in (q, suffix) if part).strip()


def _gather_evidence(question: str,
                     profile_obj: ResearchProfile,
                     context_text: str = "",
                     extra_evidence: Optional[List[str]] = None,
                     max_search_hits: Optional[int] = None,
                     max_visits: Optional[int] = None,
                     verbose: bool = False) -> Tuple[str, List[Dict[str, str]], int]:
    """
    Deterministic phase: recall + search + visit top-K. Returns
        (evidence_text, citation_list, tool_call_count)
    """
    parts: List[str] = []
    cites: List[Dict[str, str]] = []
    calls = 0

    search_hits = max_search_hits or profile_obj.max_search_hits
    max_visit_count = max_visits or profile_obj.max_visits

    if profile_obj.include_recall_note:
        recall_text, _ = _tool_recall(question)
        calls += 1
        parts.append("=== RECALL / POLICY NOTE ===")
        parts.append(recall_text)
        if verbose:
            print(f"[researcher] recall: {len(recall_text)} chars")

    if context_text:
        parts.append("\n=== PARENT CONTEXT ===")
        parts.append(context_text[:6000])

    if extra_evidence:
        rendered = [str(item).strip() for item in extra_evidence if str(item).strip()]
        if rendered:
            parts.append("\n=== CALLER-SUPPLIED EVIDENCE ===")
            parts.extend(rendered[:12])

    if profile_obj.include_lookup_targets:
        pkg_targets, image_targets = _infer_lookup_targets(question, context_text=context_text)
        for pkg in pkg_targets:
            p_text, _ = _tool_pypi(pkg)
            calls += 1
            parts.append(f"\n=== PYPI LOOKUP: {pkg} ===")
            parts.append(p_text)
            if verbose:
                print(f"[researcher] pypi {pkg}: {len(p_text)} chars")

        for image in image_targets:
            d_text, _ = _tool_docker(image)
            calls += 1
            parts.append(f"\n=== DOCKER LOOKUP: {image} ===")
            parts.append(d_text)
            if verbose:
                print(f"[researcher] docker {image}: {len(d_text)} chars")

    search_query = _build_search_query(question, profile_obj, context_text)
    sr_text, _ = _tool_search(search_query)
    calls += 1
    parts.append(f"\n=== WEB SEARCH ({search_query}) ===")
    parts.append(sr_text)
    if verbose:
        print(f"[researcher] search: {len(sr_text)} chars")

    candidates = _extract_search_candidates(sr_text, max_search_hits=search_hits)
    visited = 0
    for c in candidates:
        if visited >= max_visit_count:
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
             verbose: bool = False,
             profile: str = "runtimeRepair",
             context: Any = None,
             extra_evidence: Optional[List[str]] = None,
             max_search_hits: Optional[int] = None,
             max_visits: Optional[int] = None) -> Dict[str, Any]:
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
    profile_obj = _resolve_profile(profile)
    context_text = _context_to_text(context)
    if not q:
        return _make_result(
            q, "research_worker: empty question.", [], [], 0.0,
            0, 0, 0.0, "empty_question", "empty question",
            profile_used=profile_obj.name, followups=[],
        )
    if use_cache:
        hit = _cache_get(q, profile_obj.cache_namespace, cache_ttl_s, context_text=context_text)
        if hit:
            hit["_cache_hit"] = True
            return hit

    t0 = time.time()
    stopped = "finish"
    note: Optional[Dict[str, Any]] = None

    # ── Phase 1: gather evidence (no LLM) ──
    try:
        evidence, citations, tool_calls = _gather_evidence(
            q,
            profile_obj=profile_obj,
            context_text=context_text,
            extra_evidence=extra_evidence,
            max_search_hits=max_search_hits,
            max_visits=max_visits,
            verbose=verbose,
        )
    except Exception as e:
        return _make_result(
            q,
            f"research_worker: evidence gathering failed: {e}",
            [], [], 0.0, 0, 0, time.time() - t0, "gather_error", str(e),
            profile_used=profile_obj.name, followups=[],
        )

    if time.time() - t0 > budget_s:
        return _make_result(
            q,
            f"research_worker: evidence gathering exhausted budget ({budget_s}s).",
            [], citations, 0.0, 0, tool_calls, time.time() - t0, "budget_s", "",
            profile_used=profile_obj.name, followups=[],
        )

    # Hard-cap evidence size before sending to LLM.
    MAX_EVIDENCE_CHARS = 16000
    if len(evidence) > MAX_EVIDENCE_CHARS:
        evidence = evidence[: MAX_EVIDENCE_CHARS - 32] + "\n…[evidence truncated]\n"

    # ── Phase 2: single LLM synthesis call ──
    try:
        from utils.llm import get_llm_response
    except Exception as e:
        return _make_result(
            q,
            f"research_worker: LLM client unavailable ({e}); returning raw evidence only.",
            [], citations, 0.0, 0, tool_calls, time.time() - t0, "no_llm", str(e),
            profile_used=profile_obj.name, followups=[],
        )

    context_block = f"PARENT CONTEXT:\n{context_text}\n\n" if context_text else ""
    user_msg = (
        f"PROFILE:\n{profile_obj.name}\n\n"
        f"QUESTION:\n{q}\n\n"
        f"{context_block}"
        f"EVIDENCE GATHERED (cumulative-KB recall + web search + top page extracts):\n"
        f"{evidence}\n\n"
        f"Now emit the JSON ResearchNote per the system rules. JSON ONLY."
    )

    try:
        choices, _usage = get_llm_response(
            llm,
            [{"role": "user", "content": user_msg}],
            system_prompt=_build_synth_system_prompt(profile_obj),
            temperature=0.1, max_tokens=1200,
        )
        reply = (choices or [""])[0] or ""
    except Exception as e:
        return _make_result(
            q,
            f"research_worker: synthesis LLM call failed: {e}; returning raw evidence.",
            [], citations, 0.0, 0, tool_calls, time.time() - t0, "llm_error", str(e),
            profile_used=profile_obj.name, followups=[],
        )

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
        profile_used=profile_obj.name,
        followups=note.get("followups") or [],
    )

    if use_cache:
        try:
            _cache_put(
                q,
                profile_obj.cache_namespace,
                result,
                cache_ttl_s,
                context_text=context_text,
            )
        except Exception:
            pass
    return result


def _make_result(question: str, answer: str, cmds: list, citations: list,
                 confidence: float, turns: int, tool_calls: int,
                 wall_time_s: float, stopped: str, error: str,
                 profile_used: str,
                 followups: Optional[List[str]] = None) -> Dict[str, Any]:
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
        "profile_used": profile_used,
        "followups": list(followups or []),
    }


def format_for_observation(note: Dict[str, Any], max_chars: int = 1800) -> str:
    """Render a ResearchNote as a compact prompt-friendly string."""
    parts: List[str] = []
    cache_marker = " [cache hit]" if note.get("_cache_hit") else ""
    profile_label = str(note.get("profile_used") or "runtimeRepair")
    parts.append(
        f"research_worker[{profile_label}]{cache_marker}: confidence={note.get('confidence', 0):.2f} "
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
    followups = note.get("followups") or []
    if followups:
        parts.append("\n--- FOLLOWUPS ---")
        for item in followups[:4]:
            parts.append("  - " + str(item).strip()[:220])
    if note.get("error"):
        parts.append(f"\n[note] {note['error']}")
    out = "\n".join(parts).strip() + "\n"
    if len(out) > max_chars:
        out = out[: max_chars - 32] + "\n…[truncated]\n"
    return out
