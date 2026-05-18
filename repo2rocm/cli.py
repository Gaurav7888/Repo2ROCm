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
# soft_wrap stops long tool_use input lines from disappearing in narrow terminals;
# stderr=False keeps our pretty output on stdout, separate from httpx INFO noise.
console = Console(soft_wrap=True)


# ── migrate ───────────────────────────────────────────────────────────────────


@app.command()
def migrate(
    repo: str = typer.Argument(..., help="GitHub owner/repo or local path."),
    sha: Optional[str] = typer.Option(None, help="Git SHA to check out."),
    root_path: Path = typer.Option(Path.cwd(), help="Working directory."),
    mode: str = typer.Option("env", help="env | reproduce | full"),
    rocm_base_image: str = typer.Option(
        "rocm/pytorch:latest",
        help="ROCm Docker base image to start the sandbox from.",
    ),
    agent_mode: str = typer.Option(
        "single",
        help="single | coordinator. Default 'single' = one configuration agent "
             "drives the sandbox end-to-end (original Repo2ROCm flow). "
             "'coordinator' = multi-agent (Explore/Planner/Migrator/Verifier).",
    ),
    output_dir: Optional[Path] = typer.Option(None, help="Where to write Dockerfile + transcripts."),
    max_turns: int = typer.Option(100, help="Hard cap on agent turns."),
    no_sandbox: bool = typer.Option(
        False,
        help="DEV: skip starting the Docker container. Agent runs without DockerExec; useful for testing.",
    ),
    quiet: bool = typer.Option(
        False,
        help="Suppress live per-turn event output (default OFF: stream every turn / tool / result).",
    ),
    show_thinking: bool = typer.Option(
        False,
        help="Also print the model's extended-thinking blocks (verbose).",
    ),
) -> None:
    """Migrate a repo to AMD ROCm.

    The default flow:
      1. Clone the repo (if not already cloned).
      2. Start a ROCm Docker container with /repo mounted from the host clone.
      3. Run the configuration agent — it inspects, edits, installs, runs the
         README's verification commands, and echoes ROCM_ENV_VERIFIED.
      4. Synthesize a reproducible Dockerfile from the recorded successful commands
         + any code patches the agent applied (extracted via `git diff` of the
         host clone).
      5. Stop the container.
    """
    boot = bootstrap()
    output_dir = output_dir or (root_path / "output" / repo.replace("/", "_"))
    output_dir.mkdir(parents=True, exist_ok=True)

    asyncio.run(_run_migration(
        boot=boot,
        repo=repo,
        sha=sha,
        root_path=root_path,
        mode=mode,
        rocm_base_image=rocm_base_image,
        agent_mode=agent_mode,
        output_dir=output_dir,
        max_turns=max_turns,
        no_sandbox=no_sandbox,
        quiet=quiet,
        show_thinking=show_thinking,
    ))


async def _run_migration(
    *, boot, repo, sha, root_path, mode, rocm_base_image, agent_mode,
    output_dir, max_turns, no_sandbox, quiet=False, show_thinking=False,
):
    """End-to-end migration flow — mirrors the original Repo2ROCm workflow."""
    from repo2rocm.agents.builtin import CONFIGURATION, COORDINATOR
    from repo2rocm.agents.lifecycle import RunAgentParams, run_agent
    from repo2rocm.core.permissions import PermissionMode
    from repo2rocm.dockerfile import synthesize_dockerfile
    from repo2rocm.dockerfile.synthesizer import write_dockerfile
    from repo2rocm.learning import KBStore, TrajectoryStore, BuildAttempt, TrajectoryDistiller
    from repo2rocm.observability import TranscriptStore
    from repo2rocm.sandbox import Sandbox, SandboxConfig

    settings = get_settings()
    client = boot.make_client()

    # 1. Clone the repo if needed
    repo_path = _resolve_repo(repo, sha=sha, root_path=root_path)
    transcript_store = TranscriptStore(output_dir)
    kb = KBStore(settings.kb_path)
    traj = TrajectoryStore(settings.trajectories_path)
    attempt = BuildAttempt(repo_id=repo, sha=sha or "", docker_image=rocm_base_image or "")
    traj.start_attempt(attempt)

    console.rule(f"[bold cyan]repo2rocm migrate {repo}")
    console.print(f"  agent_mode={agent_mode}  mode={mode}")
    console.print(f"  repo_path={repo_path}")
    console.print(f"  rocm_base_image={rocm_base_image}")
    console.print(f"  output_dir={output_dir}")
    console.print(f"  metrics: http://127.0.0.1:{settings.metrics_port}/metrics")

    # 2. Start the Docker sandbox (unless dev-mode disabled)
    sandbox = None
    if not no_sandbox:
        try:
            sandbox = Sandbox(SandboxConfig(
                base_image=rocm_base_image,
                repo_host_path=repo_path,
                repo_container_path="/repo",
                rocm_mode=True,
                pull_image=True,
            ))
            # Live pull progress so the user isn't staring at a black screen
            # while a 20 GB ROCm image downloads.
            console.print(f"[cyan]· Starting sandbox from [/]{rocm_base_image}[cyan] ...[/]")
            from rich.status import Status

            with Status(
                f"Pulling [bold]{rocm_base_image}[/] (skipped if already local)...",
                console=console,
                spinner="dots",
            ) as status:
                def _on_pull(stage: str, detail: str) -> None:
                    status.update(f"[cyan]{stage}[/]: {detail}")
                await sandbox.start(on_pull_progress=_on_pull)
            console.print(
                f"[green]✓ Sandbox started:[/] {sandbox.container.name} "
                f"[dim]({sandbox.container.short_id})[/]"
            )
        except Exception as exc:
            console.print(f"[red]✗ Sandbox failed to start: {exc}")
            console.print("[yellow]Continuing without a sandbox (DockerExec calls will error).")
            sandbox = None
    else:
        console.print("[yellow]Sandbox disabled (--no-sandbox)")

    # 3. Build the parent ToolUseContext — full permissions inside the sandbox
    from repo2rocm.tools.base import ReadFileState, ToolUseContext
    parent_ctx = ToolUseContext(
        agent_id="root",
        session_id=transcript_store.session_id,
        workdir=repo_path,
        abort_event=asyncio.Event(),
        permission_mode=PermissionMode.BYPASS,  # the container IS the safety boundary
        read_file_state=ReadFileState(),
        sandbox=sandbox,
        transcript=transcript_store.main(),
        messages=[],
        options={
            "client": client,
            "client_factory": boot.make_client,
            "transcript_store": transcript_store,
            "skill_catalog": boot.skill_catalog,
            "rocm_base_image": rocm_base_image,
            "run_mode": mode,
            "repo_full_name": repo,
            "sha": sha or "",
        },
        gate_state=boot.gate_state,
    )

    # 4. Spawn the chosen agent
    agent_def = COORDINATOR if agent_mode == "coordinator" else CONFIGURATION
    if max_turns:
        agent_def = agent_def.with_(max_turns=max_turns)

    task_prompt = _build_root_prompt(
        repo=repo, sha=sha, mode=mode, base_image=rocm_base_image, agent_mode=agent_mode,
    )

    # Live event printer — streams every turn / text / tool_use / tool_result
    # to the console in real time. Disabled with --quiet.
    printer = None
    if not quiet:
        from repo2rocm.ui.event_printer import EventPrinter
        printer = EventPrinter(console=console, show_thinking=show_thinking)

    console.print(
        f"[cyan]· Spawning [bold]{agent_def.name}[/] agent[/]  "
        f"[dim](max_turns={agent_def.max_turns})[/]"
    )
    console.print(f"[cyan]· LLM:[/] [bold]{client.model}[/]  [dim](provider={client.name})[/]\n")

    result = None
    try:
        result = await run_agent(
            RunAgentParams(
                agent_def=agent_def,
                prompt=task_prompt,
                parent_ctx=parent_ctx,
                client=client,
                client_factory=boot.make_client,
                transcript_store=transcript_store,
                skill_catalog=boot.skill_catalog,
                on_event=printer,
            )
        )

        console.print(f"\n[bold]Agent ({agent_def.name}) terminal: {result.terminal.reason}")
        console.print(f"  turns: {result.terminal.turns}  duration: {result.duration_s:.1f}s")
        console.print(f"  total tokens: {result.usage_total}")
        err_msg = getattr(result.terminal, "message", "")
        err_class = getattr(result.terminal, "error_class", "")
        if err_msg or err_class:
            console.print(f"  [red]error_class={err_class}")
            console.print(f"  [red]message={err_msg[:1500]}")
        console.print(f"\n[bold]Final text:\n{result.final_text}\n")
        console.print(f"[dim]Transcript: {transcript_store.transcript(result.task.id).path}")

        # 5. Synthesize the reproducible Dockerfile (if sandbox actually ran)
        if sandbox is not None and sandbox.commands:
            patches_dir = output_dir / "patches"
            synth = synthesize_dockerfile(
                sandbox,
                repo_full_name=repo,
                sha=sha or "",
                repo_host_path=repo_path,
                patches_dir=patches_dir,
            )
            df_path = output_dir / "Dockerfile"
            write_dockerfile(synth, df_path)
            console.print(f"\n[green]✓ Dockerfile written: {df_path}")
            console.print(f"  base_image: {synth.base_image}")
            console.print(f"  successful commands: {len(synth.successful_commands)}")
            if synth.patches:
                console.print(f"  code patches captured: {len(synth.patches)}")

            # Also dump inner_commands.json (matches original Repo2ROCm artifact)
            inner = [{
                "command": c.command,
                "returncode": c.exit_code,
                "cwd": c.cwd,
                "time": round(c.elapsed_s, 4),
            } for c in sandbox.commands]
            (output_dir / "inner_commands.json").write_text(json.dumps(inner, indent=2))
            console.print(f"  inner_commands.json: {len(inner)} entries")

        # 6. Update KB + trajectory
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

    finally:
        # 7. Always tear down the sandbox
        if sandbox is not None:
            try:
                await sandbox.stop()
                console.print("[dim]✓ Sandbox stopped + removed")
            except Exception as exc:
                console.print(f"[red]sandbox stop failed: {exc}")
        kb.close()
        traj.close()


def _build_root_prompt(
    *, repo: str, sha: str | None, mode: str, base_image: str | None, agent_mode: str
) -> str:
    parts = [
        f"Migrate the GitHub repository `{repo}` to run on AMD ROCm.",
        f"Run mode: {mode}.",
    ]
    if sha:
        parts.append(f"Pinned commit SHA: {sha}.")
    if base_image:
        parts.append(f"The Docker sandbox is already running from base image: {base_image}")
        parts.append("The repository is available inside the container at /repo (read-write).")
    if agent_mode == "single":
        parts.extend([
            "",
            "Workflow:",
            "  1. Briefly inspect the repo (README, requirements, main entry).",
            "  2. Install deps via DockerExec / Download (handle CUDA-only wheels per /cuda_to_rocm_mapping).",
            "  3. Apply any necessary code patches via Edit / ApplyDiff.",
            "  4. Run the README's actual verification command on the GPU (no fake test).",
            "  5. When torch.cuda.is_available() returns True and the project runs end-to-end,",
            "     `DockerExec('echo ROCM_ENV_VERIFIED')` to signal completion.",
        ])
    else:
        parts.append(
            "Follow the four-phase coordinator workflow (research → plan → migrate → verify)."
        )
    if mode in ("reproduce", "full"):
        parts.append(
            "After env is verified, run the paper experiment per the README "
            "and use PaperVerify for a typed metric verdict."
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
    rocm_base_image: str = typer.Option("rocm/pytorch:latest"),
) -> None:
    """Alias for `migrate --mode full` — env setup + paper reproduction."""
    migrate(
        repo=repo,
        sha=sha,
        root_path=root_path,
        mode="full",
        rocm_base_image=rocm_base_image,
        agent_mode="single",
        output_dir=None,
        max_turns=100,
        no_sandbox=False,
    )


if __name__ == "__main__":
    app()
