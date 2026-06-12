"""
Web search + URL fetcher — PR-B.

Two cached, soft-failing tools the configuration agent can call when it gets
stuck on a problem that the in-repo / paper / cumulative-KB context can't
answer (typical case: a deep transformers/torch/ROCm interaction it has never
seen):

- `web_search "<query>" [--max-results N]`   → DDG (no API key) → top hits
- `visit_url <url> [--max-chars N]`          → fetch URL → readable markdown

Design (mirrors `external_lookups.py` so the dispatcher pattern is uniform):
  * Each call is cached for `_TTL` seconds in mempalace `room="research_notes"`
    on the GLOBAL wing. Cache hits cost zero network and zero LLM tokens.
  * Both fail soft: rc=1 with a single-line error, never an exception.
  * Output is hard-capped to ~2KB (web_search) / configurable (visit_url) so
    the agent observation stays short.
  * Live web is gated by `--reproduce-results` mode in main.py if the operator
    wants determinism (planned, not yet enforced).

The mental model is the same as `pypi_versions` / `dockerhub_tags`: a focused,
cheap, deterministic external lookup whose result lives in the global KB so
future runs (and parallel runs) reuse the same answer.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import List, Optional, Tuple


_TTL = 7 * 24 * 3600  # 7 days
_HTTP_TIMEOUT_S = 10.0
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) Repo2ROCm/web-search "
    "(non-commercial research)"
)

_DEFAULT_MAX_RESULTS = 5
_MAX_SEARCH_OBS_CHARS = 2200       # cap on search-result observation
_DEFAULT_VISIT_MAX_CHARS = 8000    # cap on a single page extract

_GLOBAL_WING = "rocm_global_lessons"
_CACHE_ROOM = "research_notes"


# ── Mempalace cache (shared semantics with external_lookups) ────────────────

def _palace_global_path() -> str:
    return os.path.expanduser("~/.mempalace/palaces/_global")


def _cache_id(kind: str, key: str) -> str:
    return hashlib.sha256(f"{kind}::{key}".encode()).hexdigest()[:24]


def _cache_get(kind: str, key: str, max_age_s: int) -> Optional[str]:
    try:
        from mempalace.searcher import search_memories
    except Exception:
        return None
    cid = _cache_id(kind, key)
    try:
        r = search_memories(
            f"{kind} {key}", _palace_global_path(),
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
        ttl = int(meta.get("ttl_s", _TTL))
        ts = float(meta.get("ts", 0))
        if now - ts > ttl:
            continue
        body = text[:meta_idx]
        marker = f"[CACHE_ID {cid}]\n"
        if body.startswith(marker):
            body = body[len(marker):]
        return body.rstrip()
    return None


def _cache_put(kind: str, key: str, body: str, ttl_s: int = _TTL) -> None:
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
    cid = _cache_id(kind, key)
    meta = {"kind": kind, "key": key, "ttl_s": int(ttl_s),
            "ts": time.time(), "cache_id": cid}
    try:
        content = (
            f"[CACHE_ID {cid}]\n{body.rstrip()}\n\n[META] "
            + json.dumps(meta, default=str, sort_keys=True)
        )
        add_drawer(col, _GLOBAL_WING, _CACHE_ROOM, content,
                   source_file=f"web:{kind}:{cid}", chunk_index=0,
                   agent="web-search")
    except Exception as e:
        print(f"[web_search] cache put failed: {e}")


# ── Tool 1: DDG web search ───────────────────────────────────────────────────

def _trim_snippet(text: str, n: int = 220) -> str:
    if not text:
        return ""
    s = re.sub(r"\s+", " ", text).strip()
    return s if len(s) <= n else (s[: n - 1] + "…")


def web_search(query: str, max_results: int = _DEFAULT_MAX_RESULTS,
               use_cache: bool = True) -> Tuple[str, int]:
    """
    DuckDuckGo text search. No API key. Returns (observation, return_code).

    Output format (one block per result):
        N. <title>
           <url>
           <snippet>
    """
    q = (query or "").strip()
    if not q or len(q) < 3:
        return "web_search: query too short.\n", 1
    cache_key = f"{q.lower()}::{int(max_results)}"
    if use_cache:
        cached = _cache_get("web_search", cache_key, _TTL)
        if cached:
            return f"[cache hit] {cached}", 0

    try:
        from ddgs import DDGS  # noqa: WPS433  (lazy import; optional dep)
    except Exception:
        try:
            from duckduckgo_search import DDGS  # type: ignore[no-redef]
        except Exception as e:
            return (
                "web_search: no DDG client available. "
                f"Install `ddgs` (`pip install --user ddgs`). Error: {e}\n"
            ), 1

    try:
        with DDGS() as d:
            raw = list(
                d.text(q, max_results=int(max_results), region="wt-wt")
            )
    except Exception as e:
        return f"web_search: DDG query failed: {type(e).__name__}: {e}\n", 1
    if not raw:
        return f"web_search: no results for {q!r}.\n", 1

    lines = [f"web_search {q!r}: top {len(raw)} results"]
    for i, r in enumerate(raw, 1):
        title = (r.get("title") or "?").strip()
        url = (r.get("href") or r.get("url") or "?").strip()
        snip = _trim_snippet(r.get("body") or r.get("snippet") or "", 220)
        lines.append(f"{i}. {title[:140]}")
        lines.append(f"   {url[:200]}")
        if snip:
            lines.append(f"   {snip}")

    body = "\n".join(lines).strip() + "\n"
    if len(body) > _MAX_SEARCH_OBS_CHARS:
        body = body[: _MAX_SEARCH_OBS_CHARS - 32] + "\n…[truncated]\n"

    if use_cache:
        _cache_put("web_search", cache_key, body)
    return body, 0


# ── Tool 2: URL fetch + readable extract ─────────────────────────────────────

_HTML_STRIP_TAGS_RE = re.compile(
    r"<(?:script|style|noscript|svg|nav|footer|header|form|aside|iframe)[^>]*>.*?</(?:script|style|noscript|svg|nav|footer|header|form|aside|iframe)>",
    re.IGNORECASE | re.DOTALL,
)


def _http_get_text(url: str, timeout: float = _HTTP_TIMEOUT_S,
                   max_bytes: int = 4_000_000) -> Tuple[Optional[str], Optional[str]]:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        return None, f"unsupported scheme {parsed.scheme!r}"
    if not parsed.netloc:
        return None, "missing host"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml,text/plain;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ctype = (resp.headers.get("Content-Type") or "").lower()
            data = resp.read(max_bytes + 1)
            if len(data) > max_bytes:
                data = data[:max_bytes]
            try:
                charset = resp.headers.get_content_charset() or "utf-8"
            except Exception:
                charset = "utf-8"
            return data.decode(charset, errors="replace"), ctype
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return None, f"URL error: {e.reason}"
    except Exception as e:  # timeouts, DNS, etc.
        return None, f"{type(e).__name__}: {e}"


def _html_to_text(html: str) -> str:
    """Best-effort HTML → readable plain text. Prefers html2text when present."""
    if not html:
        return ""
    # Strip noisy blocks first (html2text keeps a lot of <nav>/<footer> noise).
    cleaned = _HTML_STRIP_TAGS_RE.sub(" ", html)
    try:
        import html2text  # type: ignore[import-not-found]
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.ignore_emphasis = True
        h.body_width = 0  # no hard wrapping
        return h.handle(cleaned)
    except Exception:
        # Fallback: brutal tag strip. Good enough for "give me the prose".
        text = re.sub(r"<[^>]+>", " ", cleaned)
        text = re.sub(r"&nbsp;", " ", text)
        text = re.sub(r"&amp;", "&", text)
        text = re.sub(r"&lt;", "<", text)
        text = re.sub(r"&gt;", ">", text)
        text = re.sub(r"&quot;", '"', text)
        text = re.sub(r"\s+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def visit_url(url: str, max_chars: int = _DEFAULT_VISIT_MAX_CHARS,
              use_cache: bool = True) -> Tuple[str, int]:
    """Fetch a URL and return readable text. Returns (observation, return_code)."""
    u = (url or "").strip()
    if not u:
        return "visit_url: empty url.\n", 1
    cache_key = f"{u}::{int(max_chars)}"
    if use_cache:
        cached = _cache_get("visit_url", cache_key, _TTL)
        if cached:
            return f"[cache hit] {cached}", 0

    raw, err_or_ctype = _http_get_text(u)
    if raw is None:
        return f"visit_url: fetch failed: {err_or_ctype}\n", 1
    ctype = err_or_ctype or ""
    if "html" in ctype or "<html" in raw[:2000].lower():
        text = _html_to_text(raw)
    else:
        text = raw  # plain text / json / etc.
    text = text.strip()
    if not text:
        return f"visit_url: page {u!r} produced empty text after extraction.\n", 1

    head = f"visit_url {u}\n"
    if ctype:
        head += f"  content-type: {ctype}\n"
    head += f"  raw_size_chars: {len(raw)}\n"
    head += f"  extracted_chars: {len(text)}\n"
    head += "----- BEGIN PAGE -----\n"
    body = head + text
    if len(body) > max_chars:
        body = body[: max_chars - 64] + "\n…[truncated to {} chars]\n".format(max_chars)
    body += "\n----- END PAGE -----\n"

    if use_cache:
        _cache_put("visit_url", cache_key, body)
    return body, 0
