"""Deterministic repo preflight.

Replaces v1's monolithic `generate_plan()` with a small, typed pipeline that runs
BEFORE any agent spawns or Docker container starts. Produces a `ReconReport` that
flows into the planner agent's prompt builder and into the migration plan.

Public API:
    run_recon(repo_path, repo_full_name, mode, llm_client=None) -> ReconReport
"""
from repo2rocm.recon.report import ReconReport, FilteredRequirements
from repo2rocm.recon.runner import run_recon

__all__ = ["ReconReport", "FilteredRequirements", "run_recon"]
