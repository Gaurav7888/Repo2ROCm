"""
Aggregate per-task records into reports/results.csv and reports/summary.md.

Walks `runs/<paper_id>/<approach>/` looking for the artifacts produced by
the runners, scores each task with `scoring.rubric`, then emits:

  * reports/results.csv  - one row per (paper, approach)
  * reports/summary.md   - per-approach success-tier counts, time/token
                           aggregates, head-to-head, per-tag breakdown.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from scoring.extract import extract  # noqa: E402
from scoring.rubric import score_record_full  # noqa: E402


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def _load_tasks_index(tasks_json: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(tasks_json):
        return {}
    with open(tasks_json, "r", encoding="utf-8") as f:
        records = json.load(f)
    return {r["paper_id"]: r for r in records if r.get("paper_id")}


def _walk_results(runs_dir: str, approaches: List[str], tasks_idx: Dict[str, Dict[str, Any]]
                  ) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.isdir(runs_dir):
        return rows
    for paper_id in sorted(os.listdir(runs_dir)):
        paper_dir = os.path.join(runs_dir, paper_id)
        if not os.path.isdir(paper_dir):
            continue
        meta = tasks_idx.get(paper_id, {})
        for approach in approaches:
            task_dir = os.path.join(paper_dir, approach)
            if not os.path.isdir(task_dir):
                continue
            extracted = extract(task_dir, approach)
            if not extracted:
                continue
            scored = score_record_full(extracted)
            row = {
                "paper_id": paper_id,
                "approach": approach,
                "paper_title": meta.get("paper_title", ""),
                "conference": meta.get("conference", ""),
                "tags": meta.get("tags", ""),
                "repo_full_name": meta.get("repo_full_name", ""),
                "repo_sha": meta.get("repo_sha", ""),
                "paper_url": meta.get("paper_url", ""),
                **scored,
            }
            row["degradation_flags"] = ",".join(scored["degradation_flags"])
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #

CSV_FIELDS = [
    "paper_id", "approach", "paper_title", "conference", "tags",
    "repo_full_name", "repo_sha", "paper_url",
    "score_0_5", "verdict_marker", "degradation_flags",
    "rocm_env_verified", "paper_reproduced", "paper_not_reproduced",
    "build_success", "dockerfile_present", "timed_out", "exit_code",
    "elapsed_s", "prompt_tokens", "completion_tokens", "total_tokens",
    "n_llm_calls", "chosen_base_image", "success_report_overall",
]


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #

def _safe_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    sv = sorted(values)
    p95_idx = max(0, int(round(0.95 * (len(sv) - 1))))
    return {
        "mean": round(statistics.fmean(sv), 2),
        "median": round(statistics.median(sv), 2),
        "p95": round(sv[p95_idx], 2),
        "max": round(sv[-1], 2),
    }


def _per_approach(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_ap: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_ap[r["approach"]].append(r)
    out: Dict[str, Dict[str, Any]] = {}
    for ap, recs in by_ap.items():
        score_counts = Counter(r["score_0_5"] for r in recs)
        flag_counts = Counter()
        for r in recs:
            for f in (r.get("degradation_flags") or "").split(","):
                f = f.strip()
                if f:
                    flag_counts[f] += 1
        out[ap] = {
            "n": len(recs),
            "score_counts": {str(k): score_counts.get(k, 0) for k in (5, 4, 3, 2, 1, 0)},
            "elapsed_s": _safe_stats([r["elapsed_s"] for r in recs]),
            "total_tokens": _safe_stats([r["total_tokens"] for r in recs]),
            "n_llm_calls": _safe_stats([r["n_llm_calls"] for r in recs]),
            "rocm_env_verified_rate": round(
                sum(1 for r in recs if r["rocm_env_verified"]) / max(len(recs), 1), 3),
            "paper_reproduced_rate": round(
                sum(1 for r in recs if r["paper_reproduced"]) / max(len(recs), 1), 3),
            "score_ge4_rate": round(
                sum(1 for r in recs if r["score_0_5"] >= 4) / max(len(recs), 1), 3),
            "flags": dict(flag_counts),
        }
    return out


def _head_to_head(rows: List[Dict[str, Any]],
                  baseline: str = "claude_cli",
                  challenger: str = "repo2rocm") -> Dict[str, Any]:
    by_paper: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        by_paper[r["paper_id"]][r["approach"]] = r

    wins_challenger = 0
    wins_baseline = 0
    ties = 0
    paired: List[Dict[str, Any]] = []
    for paper_id, ap_map in by_paper.items():
        if challenger not in ap_map or baseline not in ap_map:
            continue
        c = ap_map[challenger]["score_0_5"]
        b = ap_map[baseline]["score_0_5"]
        if c > b:
            wins_challenger += 1
        elif b > c:
            wins_baseline += 1
        else:
            ties += 1
        paired.append({
            "paper_id": paper_id,
            f"{challenger}_score": c,
            f"{baseline}_score": b,
            f"{challenger}_tokens": ap_map[challenger]["total_tokens"],
            f"{baseline}_tokens": ap_map[baseline]["total_tokens"],
            f"{challenger}_elapsed_s": ap_map[challenger]["elapsed_s"],
            f"{baseline}_elapsed_s": ap_map[baseline]["elapsed_s"],
        })
    return {
        f"{challenger}_wins": wins_challenger,
        f"{baseline}_wins": wins_baseline,
        "ties": ties,
        "n_paired": len(paired),
        "paired": paired,
    }


def _by_tag(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Mean score per (tag, approach). 'tags' is a free-text comma-list in the CSV."""
    out: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        tags = r.get("tags") or ""
        for raw_t in tags.split(","):
            t = raw_t.strip()
            if not t:
                continue
            out[t][r["approach"]].append(r["score_0_5"])
    final: Dict[str, Dict[str, Dict[str, float]]] = {}
    for tag, by_ap in out.items():
        final[tag] = {}
        for ap, scores in by_ap.items():
            final[tag][ap] = {
                "n": len(scores),
                "mean_score": round(statistics.fmean(scores), 2) if scores else 0.0,
            }
    return final


def write_summary(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    per_ap = _per_approach(rows)
    h2h = _head_to_head(rows)
    by_tag = _by_tag(rows)

    lines: List[str] = []
    lines.append("# AMD-60 Benchmark Summary\n")
    lines.append(f"Total task rows: **{len(rows)}**, "
                 f"approaches: **{', '.join(sorted(set(r['approach'] for r in rows)))}**, "
                 f"papers: **{len(set(r['paper_id'] for r in rows))}**\n")

    lines.append("\n## Per-approach scorecard\n")
    lines.append("| Approach | N | %=5 | %=4 | %=3 | %=2 | %=1 | %=0 | "
                 "ROCM_VERIFIED rate | PAPER_REPRO rate | score>=4 rate |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for ap, d in sorted(per_ap.items()):
        n = max(d["n"], 1)
        pct = lambda k: f"{100.0 * d['score_counts'].get(str(k), 0) / n:.1f}%"
        lines.append(
            f"| {ap} | {d['n']} | {pct(5)} | {pct(4)} | {pct(3)} | "
            f"{pct(2)} | {pct(1)} | {pct(0)} | "
            f"{100.0 * d['rocm_env_verified_rate']:.1f}% | "
            f"{100.0 * d['paper_reproduced_rate']:.1f}% | "
            f"{100.0 * d['score_ge4_rate']:.1f}% |"
        )

    lines.append("\n## Time and tokens\n")
    lines.append("| Approach | mean_s | median_s | p95_s | mean_tok | median_tok | p95_tok | mean_calls |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for ap, d in sorted(per_ap.items()):
        e = d["elapsed_s"]; t = d["total_tokens"]; c = d["n_llm_calls"]
        lines.append(
            f"| {ap} | {e['mean']:.1f} | {e['median']:.1f} | {e['p95']:.1f} | "
            f"{int(t['mean'])} | {int(t['median'])} | {int(t['p95'])} | {c['mean']:.1f} |"
        )

    lines.append("\n## Degradation flags\n")
    lines.append("| Approach | flash_attn_triton_amd_install | sdpa_fallback | "
                 "base_image_changed | scale_down_engaged | loose_tolerance_pass |")
    lines.append("|---|--:|--:|--:|--:|--:|")
    for ap, d in sorted(per_ap.items()):
        f = d["flags"]
        lines.append(
            f"| {ap} | {f.get('flash_attn_triton_amd_install', 0)} | "
            f"{f.get('sdpa_fallback', 0)} | {f.get('base_image_changed', 0)} | "
            f"{f.get('scale_down_engaged', 0)} | {f.get('loose_tolerance_pass', 0)} |"
        )

    if h2h["n_paired"]:
        lines.append("\n## Head-to-head: repo2rocm vs claude_cli\n")
        lines.append(f"- Paired comparisons: **{h2h['n_paired']}**")
        lines.append(f"- repo2rocm wins: **{h2h['repo2rocm_wins']}**")
        lines.append(f"- claude_cli wins: **{h2h['claude_cli_wins']}**")
        lines.append(f"- ties: **{h2h['ties']}**\n")

    if by_tag:
        lines.append("\n## Per-tag mean score\n")
        approaches = sorted(set(r["approach"] for r in rows))
        header = "| Tag | " + " | ".join(f"{ap} (n=mean)" for ap in approaches) + " |"
        sep = "|---|" + "--:|" * len(approaches)
        lines.append(header)
        lines.append(sep)
        for tag in sorted(by_tag):
            cells = []
            for ap in approaches:
                d = by_tag[tag].get(ap)
                cells.append(f"{d['n']}={d['mean_score']:.2f}" if d else "-")
            lines.append(f"| {tag} | " + " | ".join(cells) + " |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser(description="Aggregate AMD-60 benchmark results.")
    p.add_argument("--runs-dir", default=os.path.normpath(os.path.join(HERE, "..", "runs")))
    p.add_argument("--reports-dir", default=os.path.normpath(os.path.join(HERE, "..", "reports")))
    p.add_argument("--tasks-json", default=os.path.join(HERE, "cache", "tasks.json"))
    p.add_argument("--approaches", default="repo2rocm,claude_cli")
    args = p.parse_args()

    approaches = [a.strip() for a in args.approaches.split(",") if a.strip()]
    tasks_idx = _load_tasks_index(args.tasks_json)
    rows = _walk_results(args.runs_dir, approaches, tasks_idx)
    if not rows:
        print(f"No task artifacts found under {args.runs_dir}", file=sys.stderr)
        return 1

    csv_path = os.path.join(args.reports_dir, "results.csv")
    md_path = os.path.join(args.reports_dir, "summary.md")
    write_csv(rows, csv_path)
    write_summary(rows, md_path)
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
