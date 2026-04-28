"""
Enrichment pass for the AMD-60 benchmark.

Takes the raw `AMD_Agnostic_Accuracy_Benchmark_60.csv` (which only has the
paper title + GitHub repo link) and resolves the two pieces of information
the runners need but the CSV doesn't carry:

  * `paper_url`   - direct arXiv PDF URL (preferred), discovered by querying
                    the arXiv search API with the paper title.
  * `repo_sha`    - the HEAD commit SHA, discovered via `git ls-remote`.

Both lookups are cached on disk so the harness is idempotent and
re-running the benchmark does not re-hit external services.

Outputs:
  * `benchmark/harness/cache/paper_urls.json`
  * `benchmark/harness/cache/repo_shas.json`
  * `benchmark/harness/cache/tasks.json`   (consolidated task list)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "cache")
DEFAULT_CSV = os.path.normpath(
    os.path.join(HERE, "..", "AMD_Agnostic_Accuracy_Benchmark_60.csv")
)


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass
class Task:
    """One row in the enriched task list."""

    paper_id: str            # stable slug; used for output dir naming
    paper_title: str
    conference: str
    tags: str
    repo_full_name: str      # owner/name
    repo_url: str            # full https URL
    repo_sha: Optional[str]  # resolved HEAD SHA, or None on failure
    paper_url: Optional[str] # arXiv PDF URL, or None if not found
    arxiv_id: Optional[str]  # bare arXiv ID (e.g. 2501.12345) when available
    notes: List[str] = field(default_factory=list)

    @property
    def is_runnable(self) -> bool:
        return bool(self.repo_full_name and self.repo_sha)


# --------------------------------------------------------------------------- #
# CSV parsing
# --------------------------------------------------------------------------- #

_GH_RE = re.compile(r"github\.com/([^/\s]+)/([^/\s#?]+)", re.IGNORECASE)


def _slugify(text: str, max_len: int = 60) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return s[:max_len] or "paper"


def _parse_repo(repo_link: str) -> Optional[str]:
    if not repo_link:
        return None
    m = _GH_RE.search(repo_link.strip())
    if not m:
        return None
    owner, name = m.group(1), m.group(2)
    name = name.rstrip(".git").rstrip("/")
    return f"{owner}/{name}"


def load_csv_rows(csv_path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            if cleaned.get("Paper Title"):
                rows.append(cleaned)
    return rows


# --------------------------------------------------------------------------- #
# arXiv lookup
# --------------------------------------------------------------------------- #

_ARXIV_API = "https://export.arxiv.org/api/query"
_ATOM_NS = "{http://www.w3.org/2005/Atom}"


def _arxiv_search(title: str, max_results: int = 5, timeout: int = 30,
                  strict: bool = True) -> List[Dict[str, str]]:
    """Query the arXiv API by title; returns up to max_results candidates.

    Each candidate has 'id', 'title', 'pdf_url', 'arxiv_id'.
    `strict=True` uses `ti:"<title>"` (exact-phrase title match).
    `strict=False` uses `all:<title>` (general fielded search) - more permissive,
    used as a fallback when the strict query returns nothing useful.
    """
    if strict:
        query = f'ti:"{title}"'
    else:
        # arXiv treats unquoted tokens as AND-of-tokens; strip punctuation.
        cleaned = re.sub(r"[^a-zA-Z0-9 ]+", " ", title).strip()
        query = f"all:{cleaned}" if cleaned else f"all:{title}"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
    }
    url = f"{_ARXIV_API}?{urllib.parse.urlencode(params)}"

    req = urllib.request.Request(url, headers={"User-Agent": "Repo2ROCm-Benchmark/1.0"})
    body = b""
    last_err: Optional[Exception] = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
            last_err = None
            break
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 429:
                # Exponential backoff with jitter. arXiv asks for ~3s minimum
                # between requests; on 429 we back off significantly more.
                delay = 5.0 * (2 ** attempt)
                time.sleep(delay)
                continue
            raise
        except Exception as e:
            last_err = e
            time.sleep(2.0 * (attempt + 1))
    if body == b"" and last_err is not None:
        raise last_err

    root = ET.fromstring(body)
    results: List[Dict[str, str]] = []
    for entry in root.findall(f"{_ATOM_NS}entry"):
        entry_id = (entry.findtext(f"{_ATOM_NS}id") or "").strip()
        entry_title = (entry.findtext(f"{_ATOM_NS}title") or "").strip()
        entry_title = re.sub(r"\s+", " ", entry_title)
        # entry_id is usually "http://arxiv.org/abs/2501.12345v2"
        m = re.search(r"arxiv\.org/abs/([^/\s]+?)(v\d+)?$", entry_id)
        arxiv_id = m.group(1) if m else ""
        pdf_url = ""
        for link in entry.findall(f"{_ATOM_NS}link"):
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href", "")
                break
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        results.append({
            "id": entry_id,
            "title": entry_title,
            "pdf_url": pdf_url,
            "arxiv_id": arxiv_id,
        })
    return results


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _title_match_score(query: str, candidate: str) -> float:
    """Cheap token-overlap score in [0, 1]."""
    q = set(_normalize(query).split())
    c = set(_normalize(candidate).split())
    if not q or not c:
        return 0.0
    return len(q & c) / max(len(q | c), 1)


def resolve_paper_url(title: str, cache: Dict[str, Any], min_score: float = 0.5) -> Dict[str, Optional[str]]:
    """Look up an arXiv PDF URL for the given paper title.

    Cached across runs by exact title. Returns dict with keys:
       'paper_url', 'arxiv_id', 'matched_title', 'match_score'.
    """
    if title in cache:
        return cache[title]

    result: Dict[str, Optional[str]] = {
        "paper_url": None,
        "arxiv_id": None,
        "matched_title": None,
        "match_score": 0.0,
    }

    def _best_among(candidates: List[Dict[str, str]]) -> tuple:
        best: Optional[Dict[str, str]] = None
        best_score = 0.0
        for cand in candidates:
            score = _title_match_score(title, cand["title"])
            if score > best_score:
                best_score = score
                best = cand
        return best, best_score

    try:
        # Pass 1: strict title match.
        candidates = _arxiv_search(title, max_results=5, strict=True)
        best, best_score = _best_among(candidates)
        # Pass 2: relaxed all-fields search if strict didn't clear the bar.
        if best_score < min_score:
            time.sleep(3.0)  # be polite between requests
            candidates_relaxed = _arxiv_search(title, max_results=10, strict=False)
            best_r, score_r = _best_among(candidates_relaxed)
            if score_r > best_score:
                best, best_score = best_r, score_r
    except Exception as e:
        # Transient errors (HTTP 429 / network) are NOT cached so the next
        # invocation gets a fresh chance.
        result["error"] = f"arxiv_query_failed: {e}"
        return result

    if best and best_score >= min_score and best.get("pdf_url"):
        result["paper_url"] = best["pdf_url"]
        result["arxiv_id"] = best.get("arxiv_id") or None
        result["matched_title"] = best["title"]
        result["match_score"] = round(best_score, 3)

    cache[title] = result
    return result


# --------------------------------------------------------------------------- #
# Repo SHA lookup
# --------------------------------------------------------------------------- #

def resolve_repo_sha(full_name: str, cache: Dict[str, str], timeout: int = 60) -> Optional[str]:
    """Resolve `full_name` (e.g. owner/repo) to a HEAD commit SHA.

    Cached. Tries the default branch (HEAD) first; if `git ls-remote` fails
    we return None so the caller can flag the row.
    """
    if full_name in cache:
        return cache[full_name] or None

    url = f"https://github.com/{full_name}.git"
    try:
        out = subprocess.check_output(
            ["git", "ls-remote", url, "HEAD"],
            timeout=timeout,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", errors="replace")
    except Exception:
        cache[full_name] = ""
        return None

    sha = ""
    for line in out.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and len(parts[0]) == 40:
            sha = parts[0]
            break
    cache[full_name] = sha
    return sha or None


# --------------------------------------------------------------------------- #
# Cache helpers
# --------------------------------------------------------------------------- #

def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Top-level enrichment
# --------------------------------------------------------------------------- #

def enrich(csv_path: str, limit: Optional[int] = None,
           sleep_between: float = 1.0,
           cache_dir: str = CACHE_DIR) -> List[Task]:
    paper_cache_path = os.path.join(cache_dir, "paper_urls.json")
    sha_cache_path = os.path.join(cache_dir, "repo_shas.json")
    tasks_path = os.path.join(cache_dir, "tasks.json")

    paper_cache = _load_json(paper_cache_path)
    sha_cache = _load_json(sha_cache_path)

    rows = load_csv_rows(csv_path)
    if limit:
        rows = rows[:limit]

    tasks: List[Task] = []
    for row in rows:
        title = row.get("Paper Title", "").strip()
        repo_link = row.get("Open-Source Repository / Code Link", "").strip()
        full_name = _parse_repo(repo_link)
        notes: List[str] = []

        if not title:
            notes.append("missing_title")
        if not full_name:
            notes.append("missing_or_unparseable_repo")

        paper_lookup = resolve_paper_url(title, paper_cache) if title else {}
        if title and not paper_lookup.get("paper_url"):
            notes.append(f"no_arxiv_match (best={paper_lookup.get('match_score')})")

        sha = resolve_repo_sha(full_name, sha_cache) if full_name else None
        if full_name and not sha:
            notes.append("git_ls_remote_failed")

        # Persist incrementally so partial runs are not wasted
        _save_json(paper_cache_path, paper_cache)
        _save_json(sha_cache_path, sha_cache)

        # Stable, filesystem-safe paper_id: <num>-<title-slug>
        num = (row.get("#") or "").strip() or "x"
        slug = _slugify(title or full_name or f"row{len(tasks)}")
        paper_id = f"{num}-{slug}"

        tasks.append(Task(
            paper_id=paper_id,
            paper_title=title,
            conference=row.get("Conference", "").strip(),
            tags=row.get("Tags", "").strip(),
            repo_full_name=full_name or "",
            repo_url=repo_link,
            repo_sha=sha,
            paper_url=paper_lookup.get("paper_url"),
            arxiv_id=paper_lookup.get("arxiv_id"),
            notes=notes,
        ))

        if sleep_between > 0:
            time.sleep(sleep_between)

    _save_json(tasks_path, [asdict(t) for t in tasks])
    return tasks


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser(description="Enrich the AMD-60 CSV with arXiv URLs and repo SHAs.")
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N rows (smoke test).")
    p.add_argument("--sleep", type=float, default=1.0,
                   help="Delay between arXiv API calls (seconds).")
    p.add_argument("--print-summary", action="store_true",
                   help="Print a per-task summary table after enrichment.")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        return 2

    tasks = enrich(args.csv, limit=args.limit, sleep_between=args.sleep)

    runnable = sum(1 for t in tasks if t.is_runnable)
    with_paper = sum(1 for t in tasks if t.paper_url)
    print(f"\nEnriched {len(tasks)} rows: {runnable} runnable, {with_paper} with paper_url")
    print(f"Cache dir: {CACHE_DIR}")

    if args.print_summary:
        print()
        print(f"{'paper_id':<60} {'sha':<10} {'arxiv':<14} notes")
        print("-" * 110)
        for t in tasks:
            sha_short = (t.repo_sha or "")[:10]
            arxiv = t.arxiv_id or "-"
            notes = ",".join(t.notes) if t.notes else "ok"
            print(f"{t.paper_id[:58]:<60} {sha_short:<10} {arxiv:<14} {notes}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
