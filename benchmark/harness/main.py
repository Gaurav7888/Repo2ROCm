"""
Top-level CLI for the AMD-60 benchmark harness.

Usage:

    # Enrich CSV (resolve arXiv URLs and repo SHAs); cache to harness/cache/.
    python -m harness.main enrich --csv ../AMD_Agnostic_Accuracy_Benchmark_60.csv

    # Run both approaches across 8 GPUs (default mode = full).
    AMD_LLM_API_KEY=... python -m harness.main run \
        --gpus 0,1,2,3,4,5,6,7 --approaches repo2rocm,claude_cli

    # Mode 1 only (functional correctness — ROCM_ENV_VERIFIED).
    AMD_LLM_API_KEY=... python -m harness.main run \
        --gpus 0,1,2,3,4,5,6,7 --approaches repo2rocm --mode env

    # Report (after some tasks have completed).
    python -m harness.main report

    # End-to-end (enrich -> run -> report).
    AMD_LLM_API_KEY=... python -m harness.main all --gpus 0,1,2,3,4,5,6,7
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from enrich import enrich, DEFAULT_CSV  # noqa: E402
from scheduler import (  # noqa: E402
    db_stats, reset_failed, run_pool, seed_tasks,
)
from report import (  # noqa: E402
    _walk_results, _load_tasks_index, write_csv, write_summary,
)


DEFAULT_DB = os.path.normpath(os.path.join(HERE, "..", "runs", "progress.sqlite"))
DEFAULT_RUNS = os.path.normpath(os.path.join(HERE, "..", "runs"))
DEFAULT_REPORTS = os.path.normpath(os.path.join(HERE, "..", "reports"))
DEFAULT_TASKS_JSON = os.path.join(HERE, "cache", "tasks.json")


# --------------------------------------------------------------------------- #
# Sub-commands
# --------------------------------------------------------------------------- #

def cmd_enrich(args: argparse.Namespace) -> int:
    tasks = enrich(args.csv, limit=args.limit, sleep_between=args.sleep)
    runnable = sum(1 for t in tasks if t.is_runnable)
    with_paper = sum(1 for t in tasks if t.paper_url)
    print(f"Enriched {len(tasks)} rows; {runnable} runnable; {with_paper} with paper_url")
    print(f"Tasks JSON: {DEFAULT_TASKS_JSON}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    if not os.path.exists(args.tasks_json):
        print(f"tasks.json missing; run `enrich` first.", file=sys.stderr)
        return 2

    import json
    with open(args.tasks_json, "r", encoding="utf-8") as f:
        records = json.load(f)
    paper_ids = [r["paper_id"] for r in records
                 if r.get("repo_full_name") and r.get("repo_sha")]
    if args.max_paper_limit:
        paper_ids = paper_ids[:args.max_paper_limit]

    approaches = [a.strip() for a in args.approaches.split(",") if a.strip()]
    inserted = seed_tasks(args.db, paper_ids, approaches)
    print(f"Seeded {inserted} new rows ({len(paper_ids)} papers x {len(approaches)} approaches)")

    if args.retry_failed:
        n = reset_failed(args.db)
        print(f"Reset {n} failed/timeout/running rows -> pending")

    print("Status (before):", db_stats(args.db))

    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    run_pool(
        db_path=args.db,
        runs_dir=args.runs_dir,
        gpus=gpus,
        timeout_s=args.timeout,
        max_disk_percent=args.max_disk_percent,
        extra_kwargs={
            "tasks_json": os.path.abspath(args.tasks_json),
            "mode": args.mode,
        },
    )
    print("Status (after):", db_stats(args.db))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
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


def cmd_all(args: argparse.Namespace) -> int:
    rc = cmd_enrich(args)
    if rc != 0:
        return rc
    rc = cmd_run(args)
    if rc != 0:
        return rc
    return cmd_report(args)


def cmd_status(args: argparse.Namespace) -> int:
    import json
    print(json.dumps(db_stats(args.db), indent=2))
    return 0


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #

def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--tasks-json", default=DEFAULT_TASKS_JSON)
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--runs-dir", default=DEFAULT_RUNS)
    p.add_argument("--reports-dir", default=DEFAULT_REPORTS)
    p.add_argument("--approaches", default="repo2rocm,claude_cli")
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--timeout", type=int, default=90 * 60)
    p.add_argument("--max-disk-percent", type=float, default=90.0)
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--mode", default="full", choices=["env", "reproduce", "full"],
                   help="Agent run-mode passed through to the runner. "
                        "env=Mode 1 ROCM_ENV_VERIFIED only (functional correctness), "
                        "reproduce=Mode 2 paper reproduction, "
                        "full=Mode 3 env+paper (default; legacy behavior).")
    p.add_argument("--limit", type=int, default=None,
                   help="(enrich) cap rows to process")
    p.add_argument("--sleep", type=float, default=1.0,
                   help="(enrich) seconds between arXiv calls")
    p.add_argument("--max-paper-limit", type=int, default=None,
                   help="(run) cap papers seeded into the DB")


def main(argv: list = None) -> int:
    p = argparse.ArgumentParser(prog="amd60-benchmark",
                                description="AMD-60 benchmark harness CLI.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_e = sub.add_parser("enrich", help="Resolve arXiv URLs + repo SHAs and write tasks.json")
    _add_common_args(p_e)
    p_e.set_defaults(func=cmd_enrich)

    p_r = sub.add_parser("run", help="Run the GPU-pinned benchmark scheduler")
    _add_common_args(p_r)
    p_r.set_defaults(func=cmd_run)

    p_p = sub.add_parser("report", help="Aggregate runs/ into reports/")
    _add_common_args(p_p)
    p_p.set_defaults(func=cmd_report)

    p_a = sub.add_parser("all", help="enrich -> run -> report")
    _add_common_args(p_a)
    p_a.set_defaults(func=cmd_all)

    p_s = sub.add_parser("status", help="Print progress.sqlite status counts")
    _add_common_args(p_s)
    p_s.set_defaults(func=cmd_status)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
