"""Repo2ROCm v2 CLI.

Commands:
  migrate     — full pipeline on one repo
  reproduce   — alias for `migrate --mode reproduce`
  batch       — fan-out over a list of repos
  mcp serve   — run an MCP server (docker-hub | pypi)
  doctor      — print bootstrap checkpoints, KB stats, tools, skills

Modes (only two):
  functional  — make the repo build and run on AMD ROCm; emit ROCM_ENV_VERIFIED
  reproduce   — functional + reproduce the paper's chosen experiment with PaperVerify
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
console = Console(soft_wrap=True)


VALID_MODES = ("functional", "reproduce")


def _normalize_mode(mode: str) -> str:
    """Accept old `env`/`full` names with a deprecation hint."""
    m = mode.lower().strip()
    if m == "env":
        console.print("[yellow]Note: mode 'env' is deprecated; using 'functional'.")
        return "functional"
    if m == "full":
        console.print("[yellow]Note: mode 'full' is deprecated; using 'reproduce'.")
        return "reproduce"
    if m not in VALID_MODES:
        raise typer.BadParameter(f"--mode must be one of {VALID_MODES}, got {mode!r}")
    return m


# ── migrate ───────────────────────────────────────────────────────────────────


@app.command()
def migrate(
    repo: str = typer.Argument(..., help="GitHub owner/repo or local path."),
    sha: Optional[str] = typer.Option(None, help="Git SHA to check out."),
    root_path: Path = typer.Option(Path.cwd(), help="Working directory."),
    mode: str = typer.Option("functional", help="functional | reproduce"),
    rocm_base_image: str = typer.Option(
        "",
        help=(
            "Override the recon's recommended ROCm Docker base image. Leave empty "
            "to use the deterministic preflight pick."
        ),
    ),
    agent_mode: str = typer.Option(
        "single",
        help=(
            "single | coordinator. Default 'single' = one configuration agent "
            "executes the MigrationPlan end-to-end. 'coordinator' = multi-agent."
        ),
    ),
    paper_url: Optional[str] = typer.Option(
        None, help="In reproduce mode: direct paper PDF URL."
    ),
    paper_arxiv_id: Optional[str] = typer.Option(
        None, help="In reproduce mode: explicit arXiv id."
    ),
    output_dir: Optional[Path] = typer.Option(None, help="Where to write artifacts."),
    max_turns: int = typer.Option(300, help="Hard cap on configuration agent turns."),
    no_sandbox: bool = typer.Option(False, help="DEV: skip starting the Docker container."),
    quiet: bool = typer.Option(False, help="Suppress live event stream."),
    show_thinking: bool = typer.Option(False, help="Also print extended-thinking blocks."),
) -> None:
    """Migrate a repo to AMD ROCm.

    Pipeline:
      0. Clone the repo (if not already cloned).
      1. Recon (deterministic): scan imports/configs/README; pick base image;
         partition requirements; collect hazards.
      2. (reproduce only) PaperResearch agent: navigate the paper, explore the
         repo, bind-check, and persist a typed PaperContext.
      3. Planner agent: emit a typed MigrationPlan via the EmitPlan tool.
      4. Start the Docker sandbox.
      5. Run Configuration (single-agent) or Coordinator (multi-agent) to
         execute the plan.
      6. Synthesize a reproducible Dockerfile from the recorded commands.
    """
    mode = _normalize_mode(mode)
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
        paper_url=paper_url,
        paper_arxiv_id=paper_arxiv_id,
        output_dir=output_dir,
        max_turns=max_turns,
        no_sandbox=no_sandbox,
        quiet=quiet,
        show_thinking=show_thinking,
    ))


async def _run_migration(
    *, boot, repo, sha, root_path, mode, rocm_base_image, agent_mode,
    paper_url, paper_arxiv_id, output_dir, max_turns, no_sandbox,
    quiet=False, show_thinking=False,
):
    from repo2rocm.agents.builtin import (
        CONFIGURATION,
        COORDINATOR,
        PAPER_RESEARCH,
        PLANNER,
    )
    from repo2rocm.agents.lifecycle import RunAgentParams, run_agent
    from repo2rocm.core.permissions import PermissionMode
    from repo2rocm.dockerfile import synthesize_dockerfile
    from repo2rocm.dockerfile.synthesizer import write_dockerfile
    from repo2rocm.learning import KBStore, TrajectoryStore, BuildAttempt, TrajectoryDistiller
    from repo2rocm.observability import TranscriptStore
    from repo2rocm.recon import run_recon
    from repo2rocm.sandbox import Sandbox, SandboxConfig

    settings = get_settings()
    client = boot.make_client()

    repo_path = _resolve_repo(repo, sha=sha, root_path=root_path)
    transcript_store = TranscriptStore(output_dir)
    kb = KBStore(settings.kb_path)
    traj = TrajectoryStore(settings.trajectories_path)
    attempt = BuildAttempt(repo_id=repo, sha=sha or "", docker_image=rocm_base_image or "")
    traj.start_attempt(attempt)

    console.rule(f"[bold cyan]repo2rocm migrate {repo}")
    console.print(f"  mode={mode}   agent_mode={agent_mode}")
    console.print(f"  repo_path={repo_path}")
    console.print(f"  output_dir={output_dir}")
    console.print(f"  metrics: http://127.0.0.1:{settings.metrics_port}/metrics")

    # Live event printer for every agent we spawn (paper-research, planner,
    # configuration). Without this, sub-agent activity is silent and the user
    # only sees the deterministic phases + a single summary line per agent.
    printer = None
    if not quiet:
        from repo2rocm.ui.event_printer import EventPrinter

        printer = EventPrinter(console=console, show_thinking=show_thinking)

    # ── Phase 1: deterministic recon ────────────────────────────────────────
    console.rule("[cyan]Recon (deterministic preflight)")
    recon = run_recon(
        repo_path=repo_path,
        repo_full_name=repo,
        mode=mode,
        sha=sha or "",
        rocm_base_image_override=rocm_base_image,
    )
    recon_path = output_dir / "recon_report.json"
    recon_path.write_text(recon.model_dump_json(indent=2), encoding="utf-8")
    console.print(recon.render_for_planner())
    console.print(f"\n[dim]Recon report: {recon_path}")

    base_image = rocm_base_image or (
        f"{recon.image_selection.image}:{recon.image_selection.tag}"
        if recon.image_selection
        else "rocm/pytorch:latest"
    )

    # Parent context that flows into every agent we spawn pre-sandbox.
    from repo2rocm.tools.base import ReadFileState, ToolUseContext

    def _make_parent_ctx(sandbox=None, agent_id="root") -> ToolUseContext:
        return ToolUseContext(
            agent_id=agent_id,
            session_id=transcript_store.session_id,
            workdir=output_dir,
            abort_event=asyncio.Event(),
            permission_mode=PermissionMode.BYPASS,
            read_file_state=ReadFileState(),
            sandbox=sandbox,
            transcript=transcript_store.main(),
            messages=[],
            options={
                "client": client,
                "client_factory": boot.make_client,
                "transcript_store": transcript_store,
                "skill_catalog": boot.skill_catalog,
                "rocm_base_image": base_image,
                "run_mode": mode,
                "repo_full_name": repo,
                "sha": sha or "",
                "recon_report": recon,
                "paper_hint": _paper_hint(paper_arxiv_id, paper_url),
                "repo_path": str(repo_path),
                "repo_container_path": "/repo",
            },
            gate_state=boot.gate_state,
        )

    # ── Phase 2 (reproduce only): PaperResearch agent ───────────────────────
    paper_ctx = None
    if mode == "reproduce":
        console.rule("[cyan]PaperResearch (LLM-driven, skill-taught)")

        pr_ctx = _make_parent_ctx(agent_id="paper-research-root")
        pr_prompt = _build_paper_research_prompt(
            recon=recon, paper_arxiv_id=paper_arxiv_id, paper_url=paper_url
        )
        pr_result = await run_agent(
            RunAgentParams(
                agent_def=PAPER_RESEARCH,
                prompt=pr_prompt,
                parent_ctx=pr_ctx,
                client=client,
                client_factory=boot.make_client,
                transcript_store=transcript_store,
                skill_catalog=boot.skill_catalog,
                on_event=printer,
            )
        )
        # Sub-agents get a COPY of ctx.options, so EmitPaperContext mutations
        # don't bubble back. The persisted JSON in output_dir/papers/ is the
        # source of truth across the agent boundary.
        paper_ctx = _load_latest_paper_context(output_dir)
        console.print(
            f"[dim]paper-research turns={pr_result.terminal.turns} "
            f"reason={pr_result.terminal.reason}"
        )

        if paper_ctx is not None:
            recon.paper_arxiv_id = paper_ctx.metadata.arxiv_id
            recon.paper_title = paper_ctx.metadata.title
            console.print(f"  paper: {paper_ctx.metadata.title or '(unknown)'}")
            console.print(
                f"  chosen experiment: {paper_ctx.chosen_experiment_id or '(none)'}"
            )
            chosen = paper_ctx.chosen()
            if chosen is not None:
                chosen.ensure_back_compat()
                bits = []
                if chosen.suggested_script:
                    bits.append(chosen.suggested_script)
                if chosen.dataset:
                    bits.append(f"dataset={chosen.dataset}")
                if chosen.model_checkpoint:
                    bits.append(f"model={chosen.model_checkpoint}")
                if chosen.estimated_runtime_min:
                    bits.append(f"runtime~{chosen.estimated_runtime_min}m")
                console.print("  target: " + ("  ".join(bits) or "(no script)"))
                if chosen.metric is not None:
                    console.print(f"  metric: {chosen.metric.display()}")
                if chosen.hyperparameters:
                    console.print(
                        f"  hyperparams: {len(chosen.hyperparameters)} "
                        f"(bound {len(chosen.repo_bindings)}, "
                        f"unbound {len(chosen.unbound_hyperparameters)})"
                    )
                if chosen.suggested_command:
                    cmd = " ".join(chosen.suggested_command.split())
                    if len(cmd) > 140:
                        cmd = cmd[:137] + "..."
                    console.print(f"  command: {cmd}")
        else:
            console.print("[yellow]No PaperContext produced.")

        if paper_ctx is None or not paper_ctx.chosen_experiment_id:
            console.print(
                "[red]No runnable paper experiment could be selected. "
                "Aborting reproduce mode before planning."
            )
            return

    # ── Phase 3: Planner emits MigrationPlan ────────────────────────────────
    console.rule("[cyan]Planner (emit typed MigrationPlan)")
    planner_ctx = _make_parent_ctx(agent_id="planner-root")
    planner_ctx.options["paper_context"] = paper_ctx
    planner_prompt = _build_planner_prompt(repo=repo, mode=mode, base_image=base_image)

    planner_result = await run_agent(
        RunAgentParams(
            agent_def=PLANNER,
            prompt=planner_prompt,
            parent_ctx=planner_ctx,
            client=client,
            client_factory=boot.make_client,
            transcript_store=transcript_store,
            skill_catalog=boot.skill_catalog,
            on_event=printer,
        )
    )
    # Same context-isolation issue — read the plan back from disk.
    plan, plan_path = _load_migration_plan(output_dir)
    if plan is None:
        console.print(
            "[red]Planner did not emit a MigrationPlan. Aborting before sandbox."
        )
        console.print(
            f"[dim]planner turns={planner_result.terminal.turns} "
            f"reason={planner_result.terminal.reason}"
        )
        return
    console.print(f"  steps={len(plan.steps)}  base_image={plan.base_image}")
    console.print(f"[dim]MigrationPlan: {plan_path}")
    console.print(
        f"[dim]planner turns={planner_result.terminal.turns} "
        f"reason={planner_result.terminal.reason}"
    )

    # ── Phase 4: start the sandbox ──────────────────────────────────────────
    sandbox = None
    if not no_sandbox:
        try:
            sandbox = Sandbox(SandboxConfig(
                base_image=plan.base_image,
                repo_host_path=repo_path,
                repo_container_path="/repo",
                rocm_mode=True,
                pull_image=True,
            ))
            console.print(f"[cyan]· Starting sandbox from [/]{plan.base_image} ...")
            from rich.status import Status

            with Status(
                f"Pulling [bold]{plan.base_image}[/] (skipped if already local)...",
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
            sandbox = None
    else:
        console.print("[yellow]Sandbox disabled (--no-sandbox)")

    # ── Phase 5: execute the plan ───────────────────────────────────────────
    parent_ctx = _make_parent_ctx(sandbox=sandbox, agent_id="root")
    parent_ctx.options["paper_context"] = paper_ctx
    parent_ctx.options["migration_plan"] = plan

    agent_def = COORDINATOR if agent_mode == "coordinator" else CONFIGURATION
    if max_turns:
        agent_def = agent_def.with_(max_turns=max_turns)

    task_prompt = _build_root_prompt(
        repo=repo, sha=sha, mode=mode, base_image=plan.base_image, agent_mode=agent_mode,
    )

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

        # ── Phase 6: synthesize a Dockerfile ────────────────────────────────
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

            inner = [{
                "command": c.command,
                "returncode": c.exit_code,
                "cwd": c.cwd,
                "time": round(c.elapsed_s, 4),
            } for c in sandbox.commands]
            (output_dir / "inner_commands.json").write_text(json.dumps(inner, indent=2))
            console.print(f"  inner_commands.json: {len(inner)} entries")

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
        if sandbox is not None:
            try:
                await sandbox.stop()
                console.print("[dim]✓ Sandbox stopped + removed")
            except Exception as exc:
                console.print(f"[red]sandbox stop failed: {exc}")
        kb.close()
        traj.close()


def _paper_hint(arxiv_id: str | None, url: str | None) -> str:
    if arxiv_id:
        return f"arxiv_id={arxiv_id}"
    if url:
        return f"url={url}"
    return ""


def _build_paper_research_prompt(*, recon, paper_arxiv_id, paper_url) -> str:
    """Short, action-oriented user message.

    The full methodology lives in the agent's system prompt + skills. We only
    pass the entry-point hint (arxiv id / url / README excerpt) plus the repo
    name. The agent's `paper_reproduction_recipes` skill takes it from there.
    """
    lines = [
        f"Repository: {recon.repo}",
        (
            "Goal: produce a fully-bound PaperContext for the project's main "
            "paper via EmitPaperContext, then end. Drive the flow yourself; "
            "consult /paper_reproduction_recipes for the suggested order."
        ),
    ]
    if paper_arxiv_id:
        lines.append(
            f"Paper hint: call PaperFetch with source='arxiv_id', value='{paper_arxiv_id}'."
        )
    elif paper_url:
        lines.append(
            f"Paper hint: call PaperFetch with source='url', value='{paper_url}'."
        )
    else:
        readme = (recon.readme_excerpt or "")[:800]
        lines.append(
            "No paper hint. Call PaperFetch with source='readme_arxiv_id' and the "
            "README excerpt below as `value`."
        )
        if readme:
            lines.append("\nREADME excerpt:")
            lines.append(readme)
    return "\n".join(lines)


def _load_latest_paper_context(output_dir: Path):
    """Read the newest PaperContext from output_dir/papers/*.json."""
    from repo2rocm.paper.types import PaperContext

    papers_dir = output_dir / "papers"
    if not papers_dir.is_dir():
        return None
    candidates = sorted(papers_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            return PaperContext.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
    return None


def _load_migration_plan(output_dir: Path):
    """Read the MigrationPlan that EmitPlan persisted under output_dir/plans/."""
    from repo2rocm.planning import MigrationPlan

    plan_path = output_dir / "plans" / "migration_plan.json"
    if not plan_path.is_file():
        return None, ""
    try:
        plan = MigrationPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
        return plan, str(plan_path)
    except Exception:  # noqa: BLE001
        return None, str(plan_path)


def _build_planner_prompt(*, repo: str, mode: str, base_image: str) -> str:
    return (
        f"Produce the MigrationPlan for `{repo}` in `{mode}` mode.\n"
        f"Default base_image: {base_image}\n"
        "The Recon Report is already in your system prompt. "
        "Emit the plan via the `EmitPlan` tool and end your turn."
    )


def _build_root_prompt(
    *, repo: str, sha: str | None, mode: str, base_image: str, agent_mode: str
) -> str:
    parts = [
        f"Execute the MigrationPlan for repository `{repo}` ({mode} mode).",
        f"The Docker sandbox is running from base image: {base_image}",
        "The repository is mounted inside the container at /repo (read-write).",
    ]
    if sha:
        parts.append(f"Pinned commit SHA: {sha}.")
    if agent_mode == "single":
        parts.append(
            "You are the Configuration agent. Walk the plan steps in `depends_on` "
            "order. Use the success_marker on each step to decide when to move on."
        )
    else:
        parts.append(
            "You are the Coordinator. Dispatch one worker per plan step "
            "(parallel where `parallel_group` is set)."
        )
    parts.append(
        "When env is verified, the verifier step emits ROCM_ENV_VERIFIED. "
        "In reproduce mode the paper-reproducer step then runs the chosen experiment."
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
    rocm_base_image: str = typer.Option(""),
    paper_url: Optional[str] = typer.Option(None),
    paper_arxiv_id: Optional[str] = typer.Option(None),
    max_turns: int = typer.Option(300, help="Hard cap on configuration agent turns."),
    no_sandbox: bool = typer.Option(False, help="DEV: skip the Docker sandbox + configuration agent."),
    quiet: bool = typer.Option(False, help="Suppress live agent event stream."),
    show_thinking: bool = typer.Option(False, help="Also print extended-thinking blocks."),
    output_dir: Optional[Path] = typer.Option(None, help="Where to write artifacts."),
) -> None:
    """Alias for `migrate --mode reproduce` \u2014 env + paper reproduction."""
    migrate(
        repo=repo,
        sha=sha,
        root_path=root_path,
        mode="reproduce",
        rocm_base_image=rocm_base_image,
        agent_mode="single",
        paper_url=paper_url,
        paper_arxiv_id=paper_arxiv_id,
        output_dir=output_dir,
        max_turns=max_turns,
        no_sandbox=no_sandbox,
        quiet=quiet,
        show_thinking=show_thinking,
    )


if __name__ == "__main__":
    app()
