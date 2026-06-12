"""
External-lookup tools — PR-A.

Deterministic, no-auth-required HTTP lookups the configuration agent can call
in-loop instead of guessing-and-rolling-back inside the sandbox:

- `pypi_versions <pkg>`     -> ranked list of recent PyPI releases + dates
- `dockerhub_tags <image>`  -> recent tags on a Docker Hub repository

Both:
  * cache results to mempalace `room="research_notes"` on the GLOBAL wing
    (TTL via [META] tag), so repeated queries across runs cost zero network;
  * return short, prompt-friendly observations (~500-1500 chars);
  * fail soft (return non-zero rc with a single-line error, no exceptions).

These compose cleanly with the in-loop tool dispatch in
`agents/configuration.py:_maybe_run_retrieval_tool`.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

# Cap how big a single observation can be (chars). Keeps the LLM context tight.
_DEFAULT_LIMIT = 12
_MAX_OBS_CHARS = 1500
_DEFAULT_TTL_SECONDS = 7 * 24 * 3600  # 7 days
_HTTP_TIMEOUT_S = 8.0
_USER_AGENT = "Repo2ROCm/external-lookups"


# ── HTTP helper ──────────────────────────────────────────────────────────────

def _http_get_json(url: str, timeout: float = _HTTP_TIMEOUT_S) -> Tuple[Optional[dict], Optional[str]]:
    """GET a URL and parse as JSON. Returns (data, error_msg). Never raises."""
    req = urllib.request.Request(
        url, headers={"User-Agent": _USER_AGENT, "Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return json.loads(body), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return None, f"URL error: {e.reason}"
    except (json.JSONDecodeError, ValueError) as e:
        return None, f"JSON decode failed: {e}"
    except Exception as e:  # network timeouts, DNS, etc.
        return None, f"{type(e).__name__}: {e}"


# ── Mempalace cache (optional) ───────────────────────────────────────────────

_GLOBAL_WING = "rocm_global_lessons"
_CACHE_ROOM = "research_notes"


def _palace_global_path() -> str:
    return os.path.expanduser("~/.mempalace/palaces/_global")


def _cache_id(kind: str, key: str) -> str:
    return hashlib.sha256(f"{kind}::{key}".encode()).hexdigest()[:24]


def _cache_get(kind: str, key: str, max_age_s: int) -> Optional[str]:
    """Look up a previous lookup result. Returns drawer text or None."""
    try:
        from mempalace.searcher import search_memories
    except Exception:
        return None
    cid = _cache_id(kind, key)
    try:
        r = search_memories(
            f"{kind} {key}", _palace_global_path(),
            wing=_GLOBAL_WING, room=_CACHE_ROOM, n_results=8, max_distance=0.0,
        )
    except Exception:
        return None
    now = time.time()
    for hit in (r.get("results") or []):
        text = hit.get("text") or ""
        if cid not in text:
            continue
        # Parse [META] sidecar
        meta_idx = text.rfind("\n[META]")
        if meta_idx < 0:
            continue
        try:
            meta = json.loads(text[meta_idx + len("\n[META]"):].strip())
        except Exception:
            continue
        ttl = int(meta.get("ttl_s", _DEFAULT_TTL_SECONDS))
        ts = float(meta.get("ts", 0))
        if now - ts > ttl:
            continue
        # Strip the meta + cid header before returning to caller.
        body = text[:meta_idx]
        marker = f"[CACHE_ID {cid}]\n"
        if body.startswith(marker):
            body = body[len(marker):]
        return body.rstrip()
    return None


def _cache_put(kind: str, key: str, body: str,
               ttl_s: int = _DEFAULT_TTL_SECONDS) -> None:
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
    meta = {
        "kind": kind, "key": key, "ttl_s": int(ttl_s),
        "ts": time.time(), "cache_id": cid,
    }
    try:
        content = f"[CACHE_ID {cid}]\n{body.rstrip()}\n\n[META] " + json.dumps(meta, default=str, sort_keys=True)
        add_drawer(col, _GLOBAL_WING, _CACHE_ROOM, content,
                   source_file=f"external:{kind}:{key}", chunk_index=0,
                   agent="external-lookups")
    except Exception as e:
        print(f"[external-lookups] cache put failed: {e}")


# ── Tool 1: PyPI versions ────────────────────────────────────────────────────

def pypi_versions(package: str, limit: int = _DEFAULT_LIMIT,
                  use_cache: bool = True) -> Tuple[str, int]:
    """
    Fetch recent versions of a PyPI package and their release dates.

    Returns (observation, return_code). rc=0 on success, non-zero on failure.
    The observation is short (≤_MAX_OBS_CHARS) and prompt-friendly.
    """
    pkg = (package or "").strip()
    if not pkg:
        return "pypi_versions: empty package name.\n", 1
    if not _safe_name(pkg):
        return f"pypi_versions: invalid package name {pkg!r}.\n", 1

    if use_cache:
        cached = _cache_get("pypi_versions", pkg.lower(), _DEFAULT_TTL_SECONDS)
        if cached:
            return f"[cache hit] {cached}", 0

    url = f"https://pypi.org/pypi/{urllib.parse.quote(pkg)}/json"
    data, err = _http_get_json(url)
    if err or not data:
        return f"pypi_versions: lookup for {pkg!r} failed: {err or 'no data'}\n", 1

    info = data.get("info") or {}
    releases = data.get("releases") or {}
    rows: List[Tuple[str, str]] = []
    for ver, files in releases.items():
        if not files:
            continue
        # Earliest upload-time among the files for this version.
        upload_time = ""
        for f in files:
            ut = f.get("upload_time") or f.get("upload_time_iso_8601") or ""
            if ut and (not upload_time or ut < upload_time):
                upload_time = ut
        rows.append((ver, upload_time[:10] if upload_time else ""))

    # Sort by upload date desc, fall back to version string.
    rows.sort(key=lambda r: (r[1] or "0000-00-00"), reverse=True)
    top = rows[: max(1, int(limit))]

    lines = [f"pypi_versions {pkg}: latest={info.get('version','?')}, {len(rows)} total versions"]
    summary = (info.get("summary") or "").strip().splitlines()[0] if info.get("summary") else ""
    if summary:
        lines.append(f"  summary: {summary[:140]}")
    home = (info.get("home_page") or info.get("project_url") or "").strip()
    if home:
        lines.append(f"  home: {home[:100]}")
    lines.append(f"Recent {len(top)} releases (newest first):")
    for ver, date in top:
        lines.append(f"  {ver:20s}  {date or '????-??-??'}")
    body = "\n".join(lines).strip() + "\n"
    if len(body) > _MAX_OBS_CHARS:
        body = body[: _MAX_OBS_CHARS - 32] + "\n…[truncated]\n"

    if use_cache:
        _cache_put("pypi_versions", pkg.lower(), body)
    return body, 0


# ── Tool 2: Docker Hub tags ──────────────────────────────────────────────────

def dockerhub_tags(image: str, limit: int = _DEFAULT_LIMIT,
                   use_cache: bool = True) -> Tuple[str, int]:
    """
    Fetch recent tags from a Docker Hub repository.

    `image` may be `repo/name` (e.g. `rocm/pytorch`) or just `name` (treated
    as `library/name`). Returns (observation, return_code).
    """
    img = (image or "").strip()
    if "/" not in img:
        img = f"library/{img}"
    if not img or "/" not in img or not all(_safe_name(p) for p in img.split("/")):
        return f"dockerhub_tags: invalid image {image!r}.\n", 1
    repo_url = f"https://registry.hub.docker.com/v2/repositories/{urllib.parse.quote(img, safe='/')}"

    if use_cache:
        cached = _cache_get("dockerhub_tags", img.lower(), _DEFAULT_TTL_SECONDS)
        if cached:
            return f"[cache hit] {cached}", 0

    page_size = max(1, min(100, int(limit) * 4))
    url = f"{repo_url}/tags/?page_size={page_size}&ordering=last_updated"
    data, err = _http_get_json(url)
    if err or not data:
        return f"dockerhub_tags: lookup for {image!r} failed: {err or 'no data'}\n", 1

    results = data.get("results") or []
    if not results:
        return f"dockerhub_tags: no tags found for {image!r}.\n", 1
    rows = []
    for t in results[: int(limit)]:
        name = t.get("name") or "?"
        last = (t.get("last_updated") or "")[:10]
        size = t.get("full_size") or 0
        size_mb = round(size / (1024 * 1024)) if size else "?"
        rows.append((name, last, size_mb))

    lines = [f"dockerhub_tags {image}: showing {len(rows)} most recently updated tags (of {data.get('count', '?')})"]
    for name, last, size_mb in rows:
        lines.append(f"  {name:32s}  updated={last}  size~{size_mb}MB")
    body = "\n".join(lines).strip() + "\n"
    if len(body) > _MAX_OBS_CHARS:
        body = body[: _MAX_OBS_CHARS - 32] + "\n…[truncated]\n"

    if use_cache:
        _cache_put("dockerhub_tags", img.lower(), body)
    return body, 0


def dockerhub_tags_structured(image: str, limit: int = 25) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Fetch Docker Hub tags as structured records for planner-side scoring.

    This is intentionally separate from the prompt-facing `dockerhub_tags`
    observation above: the planner needs raw fields such as `name`,
    `last_updated`, and `full_size` to refresh stale static tags and rank
    candidate containers.
    """
    img = (image or "").strip()
    if "/" not in img:
        img = f"library/{img}"
    if not img or "/" not in img or not all(_safe_name(p) for p in img.split("/")):
        return [], f"invalid image {image!r}"

    repo_url = f"https://registry.hub.docker.com/v2/repositories/{urllib.parse.quote(img, safe='/')}"
    page_size = max(1, min(100, int(limit)))
    url = f"{repo_url}/tags/?page_size={page_size}&ordering=last_updated"
    data, err = _http_get_json(url)
    if err or not data:
        return [], err or "no data"

    rows: List[Dict[str, Any]] = []
    for tag in data.get("results") or []:
        rows.append({
            "name": tag.get("name") or "",
            "last_updated": tag.get("last_updated") or "",
            "full_size": tag.get("full_size") or 0,
            "digest": tag.get("digest") or "",
        })
    if not rows:
        return [], "no tags found"
    return rows, None


# ── Validation ────────────────────────────────────────────────────────────────

import re as _re

_NAME_RE = _re.compile(r"^[A-Za-z0-9._\-]{1,128}$")


def _safe_name(name: str) -> bool:
    return bool(name) and bool(_NAME_RE.match(name))
