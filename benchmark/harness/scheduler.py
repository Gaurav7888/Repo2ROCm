"""
Resumable, GPU-pinned task scheduler for the AMD-60 benchmark.

The scheduler launches one worker process per GPU slot. Each worker:
  * pulls (paper_id, approach) tasks from `progress.sqlite`,
  * pins HIP_VISIBLE_DEVICES = <slot> for the duration of the task,
  * dispatches to the appropriate runner (repo2rocm or claude_cli),
  * marks the task done/failed/timeout,
  * cleans up dangling docker containers/images that match its slot prefix.

Resumability: re-running with the same `runs_dir` skips tasks already in the
`done` state. `--retry-failed` re-queues `failed` and `timeout` rows.

Cleanup: between tasks, each worker prunes containers it owns (named with
its slot prefix) and `docker images --filter dangling=true`.

Disk guard: a worker aborts (returns its slot to the pool but stops accepting
new work) if the host disk is over `--max-disk-percent` (default 90%).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import traceback
from contextlib import closing
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# Progress DB
# --------------------------------------------------------------------------- #

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
  paper_id     TEXT NOT NULL,
  approach     TEXT NOT NULL,
  status       TEXT NOT NULL DEFAULT 'pending',  -- pending|running|done|failed|timeout
  gpu_index    INTEGER,
  worker_pid   INTEGER,
  started_at   REAL,
  finished_at  REAL,
  exit_code    INTEGER,
  notes        TEXT,
  PRIMARY KEY (paper_id, approach)
);
CREATE INDEX IF NOT EXISTS idx_status ON tasks(status);
"""


def open_db(path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)  # autocommit
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    return conn


def seed_tasks(db_path: str, paper_ids: List[str], approaches: List[str]) -> int:
    """Insert (paper_id, approach) rows that don't already exist. Returns # inserted."""
    inserted = 0
    with closing(open_db(db_path)) as conn:
        for pid in paper_ids:
            for ap in approaches:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO tasks (paper_id, approach, status) VALUES (?, ?, 'pending')",
                    (pid, ap),
                )
                inserted += cur.rowcount
    return inserted


def reset_failed(db_path: str) -> int:
    with closing(open_db(db_path)) as conn:
        cur = conn.execute(
            "UPDATE tasks SET status='pending', worker_pid=NULL, started_at=NULL, "
            "finished_at=NULL, exit_code=NULL, notes=NULL "
            "WHERE status IN ('failed','timeout','running')"
        )
        return cur.rowcount


def claim_next_task(conn: sqlite3.Connection, gpu_index: int, worker_pid: int) -> Optional[Tuple[str, str]]:
    """Atomically pick one pending task and mark it running.

    Uses an immediate transaction + UPDATE...WHERE rowid IN (SELECT...LIMIT 1)
    to avoid two workers grabbing the same row.
    """
    while True:
        try:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute(
                "SELECT paper_id, approach FROM tasks "
                "WHERE status='pending' "
                "ORDER BY paper_id, approach LIMIT 1"
            )
            row = cur.fetchone()
            if row is None:
                conn.execute("COMMIT")
                return None
            paper_id, approach = row
            conn.execute(
                "UPDATE tasks SET status='running', gpu_index=?, worker_pid=?, started_at=? "
                "WHERE paper_id=? AND approach=?",
                (gpu_index, worker_pid, time.time(), paper_id, approach),
            )
            conn.execute("COMMIT")
            return paper_id, approach
        except sqlite3.OperationalError:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            time.sleep(0.1)


def finish_task(conn: sqlite3.Connection, paper_id: str, approach: str,
                status: str, exit_code: Optional[int], notes: str) -> None:
    conn.execute(
        "UPDATE tasks SET status=?, finished_at=?, exit_code=?, notes=? "
        "WHERE paper_id=? AND approach=?",
        (status, time.time(), exit_code, notes[:2000] if notes else None, paper_id, approach),
    )


def db_stats(db_path: str) -> Dict[str, int]:
    with closing(open_db(db_path)) as conn:
        cur = conn.execute("SELECT status, COUNT(*) FROM tasks GROUP BY status")
        return {row[0]: row[1] for row in cur.fetchall()}


# --------------------------------------------------------------------------- #
# Disk guard + cleanup helpers
# --------------------------------------------------------------------------- #

def disk_percent_used(path: str) -> float:
    try:
        usage = shutil.disk_usage(path)
        return 100.0 * usage.used / usage.total
    except Exception:
        return 0.0


def cleanup_worker_containers(slot_prefix: str) -> None:
    """Remove docker containers whose name starts with the slot prefix and
    prune dangling images. Best-effort: failures are swallowed."""
    try:
        out = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=20,
        )
        names = [n.strip() for n in out.stdout.splitlines() if n.strip().startswith(slot_prefix)]
        if names:
            subprocess.run(["docker", "rm", "-f", *names],
                           capture_output=True, timeout=120)
    except Exception:
        pass
    try:
        subprocess.run(
            "docker images --filter 'dangling=true' -q | xargs -r docker rmi",
            shell=True, capture_output=True, timeout=120,
        )
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #

@dataclass
class WorkerConfig:
    gpu_index: int
    db_path: str
    runs_dir: str
    timeout_s: int
    max_disk_percent: float
    runner_dispatch: Dict[str, str]  # approach -> module path (e.g. "harness.runner_repo2rocm")
    extra_kwargs: Dict[str, Any]


def _import_runner(module_path: str) -> Callable[..., Dict[str, Any]]:
    """Load `<module_path>.run_task` callable."""
    import importlib
    mod = importlib.import_module(module_path)
    fn = getattr(mod, "run_task", None)
    if not callable(fn):
        raise RuntimeError(f"Runner {module_path!r} has no callable `run_task`")
    return fn


def worker_loop(cfg: WorkerConfig) -> None:
    pid = os.getpid()
    slot_prefix = f"amd60-gpu{cfg.gpu_index}-"
    runners: Dict[str, Callable[..., Dict[str, Any]]] = {}

    print(f"[gpu{cfg.gpu_index} pid={pid}] worker started", flush=True)
    conn = open_db(cfg.db_path)
    try:
        while True:
            used = disk_percent_used(cfg.runs_dir)
            if used >= cfg.max_disk_percent:
                print(f"[gpu{cfg.gpu_index}] disk {used:.1f}% >= {cfg.max_disk_percent}%; stopping",
                      flush=True)
                return

            task = claim_next_task(conn, cfg.gpu_index, pid)
            if task is None:
                print(f"[gpu{cfg.gpu_index}] no more pending tasks; exiting", flush=True)
                return
            paper_id, approach = task
            module_path = cfg.runner_dispatch.get(approach)
            if not module_path:
                finish_task(conn, paper_id, approach, "failed", None,
                            f"unknown_approach: {approach}")
                continue

            if approach not in runners:
                try:
                    runners[approach] = _import_runner(module_path)
                except Exception as e:
                    finish_task(conn, paper_id, approach, "failed", None,
                                f"runner_import_failed: {e}")
                    continue

            run_task = runners[approach]
            task_dir = os.path.join(cfg.runs_dir, paper_id, approach)
            os.makedirs(task_dir, exist_ok=True)

            print(f"[gpu{cfg.gpu_index}] >>> {paper_id}::{approach}", flush=True)
            t0 = time.time()
            status = "failed"
            exit_code: Optional[int] = None
            notes = ""
            try:
                result = run_task(
                    paper_id=paper_id,
                    task_dir=task_dir,
                    gpu_index=cfg.gpu_index,
                    container_name_prefix=slot_prefix + paper_id,
                    timeout_s=cfg.timeout_s,
                    **cfg.extra_kwargs,
                )
                exit_code = int(result.get("exit_code", 0)) if isinstance(result, dict) else 0
                if isinstance(result, dict) and result.get("timed_out"):
                    status = "timeout"
                    notes = str(result.get("notes") or "timeout")
                elif exit_code == 0:
                    status = "done"
                    notes = str(result.get("notes") or "ok") if isinstance(result, dict) else "ok"
                else:
                    status = "failed"
                    notes = str(result.get("notes") or f"exit {exit_code}") \
                        if isinstance(result, dict) else f"exit {exit_code}"
            except Exception as e:
                status = "failed"
                notes = f"runner_exception: {e}\n{traceback.format_exc()[:1500]}"

            elapsed = time.time() - t0
            print(f"[gpu{cfg.gpu_index}] <<< {paper_id}::{approach} "
                  f"status={status} exit={exit_code} elapsed={elapsed:.1f}s",
                  flush=True)

            finish_task(conn, paper_id, approach, status, exit_code, notes)
            cleanup_worker_containers(slot_prefix)
    finally:
        try:
            conn.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def _dispatch_default() -> Dict[str, str]:
    return {
        "repo2rocm": "harness.runner_repo2rocm",
        "claude_cli": "harness.runner_claude_cli",
    }


def run_pool(*, db_path: str, runs_dir: str, gpus: List[int],
             timeout_s: int, max_disk_percent: float,
             runner_dispatch: Optional[Dict[str, str]] = None,
             extra_kwargs: Optional[Dict[str, Any]] = None) -> None:
    """Spawn one worker per GPU slot and wait for all to finish."""
    runner_dispatch = runner_dispatch or _dispatch_default()
    extra_kwargs = extra_kwargs or {}

    ctx = mp.get_context("spawn")
    procs = []
    for g in gpus:
        cfg = WorkerConfig(
            gpu_index=g,
            db_path=db_path,
            runs_dir=runs_dir,
            timeout_s=timeout_s,
            max_disk_percent=max_disk_percent,
            runner_dispatch=runner_dispatch,
            extra_kwargs=extra_kwargs,
        )
        p = ctx.Process(target=worker_loop, args=(cfg,), name=f"amd60-gpu{g}")
        p.daemon = False
        p.start()
        procs.append(p)

    def _signal_handler(signum, frame):
        print(f"\nReceived signal {signum}; terminating workers...", flush=True)
        for p in procs:
            if p.is_alive():
                p.terminate()
        sys.exit(130)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    for p in procs:
        p.join()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _load_paper_ids(tasks_json: str) -> List[str]:
    with open(tasks_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [t["paper_id"] for t in data if t.get("repo_full_name") and t.get("repo_sha")]


def main() -> int:
    p = argparse.ArgumentParser(description="Schedule the AMD-60 benchmark across GPUs.")
    p.add_argument("--tasks-json", default=os.path.join(HERE, "cache", "tasks.json"),
                   help="Output of enrich.py")
    p.add_argument("--db", default=os.path.normpath(os.path.join(HERE, "..", "runs", "progress.sqlite")))
    p.add_argument("--runs-dir", default=os.path.normpath(os.path.join(HERE, "..", "runs")))
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7",
                   help="Comma-separated GPU indices.")
    p.add_argument("--approaches", default="repo2rocm,claude_cli",
                   help="Comma-separated approaches to seed.")
    p.add_argument("--timeout", type=int, default=90 * 60,
                   help="Per-task timeout in seconds.")
    p.add_argument("--max-disk-percent", type=float, default=90.0)
    p.add_argument("--retry-failed", action="store_true",
                   help="Re-queue failed/timeout/running rows before scheduling.")
    p.add_argument("--seed-only", action="store_true",
                   help="Insert pending rows but don't start workers.")
    p.add_argument("--show", action="store_true",
                   help="Print status counts and exit.")
    p.add_argument("--max-paper-limit", type=int, default=None,
                   help="Cap the number of paper_ids seeded (smoke testing).")
    p.add_argument("--mode", default="full", choices=["env", "reproduce", "full"],
                   help="Agent run-mode passed to runners. env=Mode 1 "
                        "ROCM_ENV_VERIFIED only, reproduce=Mode 2 paper "
                        "reproduction, full=Mode 3 both (default; legacy behavior).")
    args = p.parse_args()

    if args.show:
        print(json.dumps(db_stats(args.db), indent=2))
        return 0

    if not os.path.exists(args.tasks_json):
        print(f"tasks.json not found: {args.tasks_json}\n"
              f"Run `python -m harness.enrich` first.", file=sys.stderr)
        return 2

    paper_ids = _load_paper_ids(args.tasks_json)
    if args.max_paper_limit:
        paper_ids = paper_ids[:args.max_paper_limit]
    approaches = [a.strip() for a in args.approaches.split(",") if a.strip()]
    inserted = seed_tasks(args.db, paper_ids, approaches)
    print(f"Seeded {inserted} new (paper, approach) rows "
          f"(papers={len(paper_ids)} approaches={approaches})")

    if args.retry_failed:
        n = reset_failed(args.db)
        print(f"Reset {n} failed/timeout/running rows -> pending")

    print("Status:", json.dumps(db_stats(args.db)))

    if args.seed_only:
        return 0

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
    print("Final status:", json.dumps(db_stats(args.db)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
