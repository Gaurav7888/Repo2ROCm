"""Repo2ROCm v2 CLI.

Commands:
  migrate     — full pipeline on one repo
  batch       — fan-out over a list of repos
  mcp serve   — run an MCP server (docker-hub | pypi)
  reproduce   — env + paper reproduction (alias for `migrate --mode reproduce`)
  doctor      — print bootstrap checkpoints, KB stats, prompt-cache estimate
"""
from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from repo2rocm.bootstrap import bootstrap
from repo2rocm.config import get_settings
from repo2rocm.observability.checkpoints import get_registry

app = typer.Typer(no_args_is_help=True, add_completion=False)
mcp_app = typer.Typer(no_args_is_help=True)
app.add_typer(mcp_app, name="mcp", help="MCP server commands.")
console = Console()


# ── migrate ───────────────────────────────────────────────────────────────────


@app.command()
def migrate(
    repo: str = typer.Argument(..., help="GitHub owner/repo or local path."),
    sha: Optional[str] = typer.Option(None, help="Git SHA to check out."),
    root_path: Path = typer.Option(Path.cwd(), help="Working directory."),
    mode: str = typer.Option("env", help="env | reproduce | full"),
    rocm_base_image: Optional[str] = typer.Option(None, help="Override base image."),
    permission_mode: str = typer.Option("acceptEdits", help="plan | acceptEdits | bypassPermissions"),
    output_dir: Optional[Path] = typer.Option(None, help="Where to write Dockerfile + transcripts."),
    legacy: bool = typer.Option(False, help="Use the legacy single-agent loop."),
) -> None:
    """Migrate a repo to AMD ROCm."""
    boot = bootstrap()
    settings = get_settings()
    settings.permission_mode = permission_mode

    output_dir = output_dir or (root_path / "output" / repo.replace("/", "_"))
    output_dir.mkdir(parents=True, exist_ok=True)

    asyncio.run(_run_migration(
        boot=boot,
        repo=repo,
        sha=sha,
        root_path=root_path,
        mode=mode,
        rocm_base_image=rocm_base_image,
        output_dir=output_dir,
        legacy=legacy,
    ))


async def _run_migration(*, boot, repo, sha, root_path, mode, rocm_base_image, output_dir, legacy):
    """End-to-end migration flow.

    1. Clone repo (if not local path).
    2. Build the Coordinator agent.
    3. Spawn it with the task prompt.
    4. Synthesize Dockerfile from the trunk commits.
    """
    from repo2rocm.agents.builtin import COORDINATOR
    from repo2rocm.agents.lifecycle import RunAgentParams, run_agent
    from repo2rocm.core.hooks.builtin import GateState
    from repo2rocm.core.permissions import PermissionMode
    from repo2rocm.dockerfile import synthesize_dockerfile
    from repo2rocm.dockerfile.synthesizer import write_dockerfile
    from repo2rocm.learning import KBStore, TrajectoryStore, BuildAttempt, TrajectoryDistiller
    from repo2rocm.observability import TranscriptStore

    settings = get_settings()
    client = boot.make_client()

    repo_path = _resolve_repo(repo, sha=sha, root_path=root_path)
    transcript_store = TranscriptStore(output_dir)
    kb = KBStore(settings.kb_path)
    traj = TrajectoryStore(settings.trajectories_path)
    attempt = BuildAttempt(repo_id=repo, sha=sha or "", docker_image=rocm_base_image or "")
    traj.start_attempt(attempt)

    console.rule(f"[bold cyan]repo2rocm migrate {repo}")
    console.print(f"  mode={mode}   permission_mode={settings.permission_mode}")
    console.print(f"  repo_path={repo_path}")
    console.print(f"  output_dir={output_dir}")
    console.print(f"  metrics: http://127.0.0.1:{settings.metrics_port}/metrics")

    # We thread a parent context's options so AgentTool can spawn sub-agents.
    from repo2rocm.tools.base import ReadFileState, ToolUseContext

    parent_ctx = ToolUseContext(
        agent_id="root",
        session_id=transcript_store.session_id,
        workdir=repo_path,
        abort_event=asyncio.Event(),
        permission_mode=PermissionMode(settings.permission_mode),
        read_file_state=ReadFileState(),
        sandbox=None,  # sandbox attached lazily by Migrator if needed
        transcript=transcript_store.main(),
        messages=[],
        options={
            "client": client,
            "client_factory": boot.make_client,
            "transcript_store": transcript_store,
            "skill_catalog": boot.skill_catalog,
        },
        gate_state=boot.gate_state,
    )

    task_prompt = _build_root_prompt(repo=repo, sha=sha, mode=mode, base_image=rocm_base_image)

    if legacy:
        # Future hook for the legacy single-agent path; for now, error out.
        console.print("[yellow]--legacy is not yet implemented in v2; running coordinator path.")

    result = await run_agent(
        RunAgentParams(
            agent_def=COORDINATOR,
            prompt=task_prompt,
            parent_ctx=parent_ctx,
            client=client,
            client_factory=boot.make_client,
            transcript_store=transcript_store,
            skill_catalog=boot.skill_catalog,
        )
    )

    console.print(f"\n[bold]Coordinator terminal: {result.terminal.reason}")
    console.print(f"  turns: {result.terminal.turns}  duration: {result.duration_s:.1f}s")
    console.print(f"  total tokens: {result.usage_total}")
    console.print(f"\n[bold]Final text:\n{result.final_text}")

    # If a sandbox was attached during the run, synthesize the Dockerfile.
    if parent_ctx.sandbox is not None:
        synth = synthesize_dockerfile(parent_ctx.sandbox)
        df_path = output_dir / "Dockerfile"
        write_dockerfile(synth, df_path)
        console.print(f"\n[green]Dockerfile written: {df_path}")

    # Distill structured facts back into the KB
    attempt.outcome = "success" if result.terminal.reason == "completed" else "failure"
    attempt.duration_s = result.duration_s
    attempt.total_turns = result.terminal.turns
    attempt.total_tokens = result.usage_total
    attempt.trajectory_file = str(transcript_store.main().path)
    traj.complete_attempt(
        attempt.id,
        outcome=attempt.outcome,
        duration_s=attempt.duration_s,
        total_turns=attempt.total_turns,
        total_tokens=attempt.total_tokens,
    )
    distiller = TrajectoryDistiller(kb, traj)
    distill_result = distiller.distill_and_apply(attempt)
    console.print(
        f"\n[dim]Distilled {distill_result.facts_added} facts; "
        f"{distill_result.rules_updated} rules updated"
    )

    kb.close()
    traj.close()


def _build_root_prompt(*, repo: str, sha: str | None, mode: str, base_image: str | None) -> str:
    parts = [
        f"Migrate the GitHub repository `{repo}` to run on AMD ROCm.",
        f"Mode: {mode}.",
    ]
    if sha:
        parts.append(f"Pinned SHA: {sha}.")
    if base_image:
        parts.append(f"Operator suggests base image: {base_image} (verify it exists first).")
    parts.append(
        "Follow the four-phase coordinator workflow (research → plan → migrate → verify)."
    )
    if mode in ("reproduce", "full"):
        parts.append(
            "Reproduction phase: spawn a `paper-reproducer` after env verification."
        )
    return "\n".join(parts)


def _resolve_repo(repo: str, *, sha: str | None, root_path: Path) -> Path:
    candidate = Path(repo)
    if candidate.exists():
        return candidate.resolve()
    if "/" not in repo:
        raise typer.BadParameter(f"not a path and not owner/repo: {repo}")
    target = root_path / "repos" / repo.replace("/", "_")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://github.com/{repo}.git"
        subprocess.run(["git", "clone", "--depth", "50", url, str(target)], check=True)
    if sha:
        subprocess.run(["git", "checkout", sha], cwd=str(target), check=True)
    return target


# ── batch ─────────────────────────────────────────────────────────────────────


@app.command()
def batch(
    file: Path = typer.Argument(..., help="JSONL of {repo, sha, mode} entries."),
    parallel: int = typer.Option(3, help="Max concurrent migrations."),
) -> None:
    """Run many migrations in parallel."""
    bootstrap()
    entries = [json.loads(l) for l in file.read_text().splitlines() if l.strip()]
    asyncio.run(_run_batch(entries, parallel))


async def _run_batch(entries: list[dict], parallel: int) -> None:
    sem = asyncio.Semaphore(parallel)

    async def one(e: dict) -> None:
        async with sem:
            console.print(f"[dim]start: {e['repo']}")
            # In production this would call into _run_migration; left as a hook.
            await asyncio.sleep(0.1)
            console.print(f"[dim]end:   {e['repo']}")

    await asyncio.gather(*(one(e) for e in entries))


# ── MCP servers ───────────────────────────────────────────────────────────────


@mcp_app.command("serve")
def mcp_serve(name: str = typer.Argument(..., help="docker-hub | pypi")) -> None:
    """Run a built-in MCP server on stdio."""
    if name == "docker-hub":
        from repo2rocm.mcp.servers.docker_hub import main as serve

        asyncio.run(serve())
    elif name == "pypi":
        from repo2rocm.mcp.servers.pypi import main as serve

        asyncio.run(serve())
    else:
        raise typer.BadParameter(f"unknown server: {name}")


# ── doctor ────────────────────────────────────────────────────────────────────


@app.command()
def doctor() -> None:
    """Diagnostics: bootstrap timings, KB stats, observability status."""
    boot = bootstrap()
    s = get_settings()

    t = Table(title="Bootstrap Checkpoints", show_lines=False)
    t.add_column("name")
    t.add_column("Δms", justify="right")
    t.add_column("cumulative_ms", justify="right")
    for r in get_registry().summary():
        t.add_row(str(r["name"]), f"{r['delta_ms']:.2f}", f"{r['cumulative_ms']:.2f}")
    console.print(t)

    from repo2rocm.learning import KBStore, TrajectoryStore

    kb = KBStore(s.kb_path)
    traj = TrajectoryStore(s.trajectories_path)
    console.rule("KB & Trajectory stats")
    console.print(kb.stats())
    console.print(traj.stats())
    kb.close()
    traj.close()

    console.rule("Tool registry")
    from repo2rocm.tools.base import get_all_tools

    for t_ in sorted(get_all_tools(), key=lambda x: x.name):
        console.print(f"  - {t_.name:24s} max={t_.max_result_size_chars:>7d}b")

    console.rule("Skill catalog")
    console.print(boot.skill_catalog.menu_text())


@app.command()
def reproduce(
    repo: str = typer.Argument(...),
    sha: Optional[str] = typer.Option(None),
    root_path: Path = typer.Option(Path.cwd()),
) -> None:
    """Alias for `migrate --mode full`."""
    migrate(repo=repo, sha=sha, root_path=root_path, mode="full", rocm_base_image=None, permission_mode="acceptEdits", output_dir=None, legacy=False)


if __name__ == "__main__":
    app()
