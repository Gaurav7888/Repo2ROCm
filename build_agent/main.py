# Copyright (2025) Bytedance Ltd. and/or its affiliates 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 


import argparse
import json
import multiprocessing
import time
import os
import sys
from datetime import datetime, timedelta
from typing import Optional
from utils.sandbox import Sandbox
from agents.configuration import Configuration
from utils.llm import set_api_key
import subprocess
from utils.waiting_list import WaitingList
from utils.conflict_list import ConflictList
from utils.integrate_dockerfile import integrate_dockerfile
from utils.rich_logger import log_header, log_phase, log_success, log_error, log_info, log_container_op, console, init_file_log, close_file_log
from agents.planner import generate_plan, print_plan
import ast
import shutil

from storage.models import BuildAttempt, BuildOutcome, TrajectoryRecord
from storage.trajectory_store import TrajectoryStore
from storage.kb_store import KBStore
from storage.build_fingerprint import compute_fingerprint
from errors.classifier import ErrorClassifier
from errors.seed_patterns import seed_if_empty
from rules.engine import RuleEngine
from learning.memory_provider import BuildMemoryProvider
from learning.distiller import TrajectoryDistiller
try:
    from learning.mempalace_provider import RunMemory  # Stage 2 memory layer
    _MEMPALACE_AVAILABLE = True
except Exception as _mp_e:
    print(f"[mempalace] not available ({_mp_e}); RunMemory will be disabled")
    _MEMPALACE_AVAILABLE = False
    class RunMemory:  # type: ignore[no-redef]
        enabled = False
        @classmethod
        def create(cls, *a, **kw): return cls()
        def __getattr__(self, n):
            def _noop(*a, **kw): return None
            return _noop

def move_files_to_repo(source_folder):
    # Temporary staging directory to flatten the repo's top-level folder
    target_folder = os.path.join(source_folder, 'repo_inner_directory_long_long_name_to_avoid_duplicate')
    
    # Create the staging directory if it doesn't exist
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
    
    # Move all items into the staging directory
    for item in os.listdir(source_folder):
        item_path = os.path.join(source_folder, item)
        
        # Skip the staging directory itself
        if item == 'repo_inner_directory_long_long_name_to_avoid_duplicate':
            continue
        
        # Move file/directory into the staging directory
        shutil.move(item_path, os.path.join(target_folder, item))

    os.rename(target_folder, os.path.join(source_folder, 'repo'))

# Clone the repo into utils/repo/, flatten the top-level directory, and remove any existing Dockerfile.
def download_repo(root_path, full_name, sha):
    if len(full_name.split('/')) != 2:
        raise Exception("full_name Wrong!!!")
    author_name = full_name.split('/')[0]
    repo_name = full_name.split('/')[1]
    if not os.path.exists(f'{root_path}/utils/repo/{author_name}/{repo_name}'):
        os.system(f'mkdir -p {root_path}/utils/repo/{author_name}/{repo_name}')
    download_cmd = f"git clone https://github.com/{full_name}.git"
    subprocess.run(download_cmd, cwd=f'{root_path}/utils/repo/{author_name}', check=True, shell=True)
    move_files_to_repo(f'{root_path}/utils/repo/{author_name}/{repo_name}')
    if os.path.exists(f"{root_path}/utils/repo/{author_name}/{repo_name}/repo/Dockerfile") and not os.path.isdir(f"{root_path}/utils/repo/{author_name}/{repo_name}/repo/Dockerfile"):
        rm_dockerfile_cmd = f"rm -rf {root_path}/utils/repo/{author_name}/{repo_name}/repo/Dockerfile"
        subprocess.run(rm_dockerfile_cmd, check=True, shell=True)
    pipreqs_cmd = "pipreqs --savepath=.pipreqs/requirements_pipreqs.txt --force"
    os.system(f'mkdir {root_path}/utils/repo/{author_name}/{repo_name}/repo/.pipreqs')
    try:
        pipreqs_warnings = subprocess.run(pipreqs_cmd, cwd=f"{root_path}/utils/repo/{author_name}/{repo_name}/repo", check=True, shell=True, capture_output=True)
        with open(f'{root_path}/utils/repo/{author_name}/{repo_name}/repo/.pipreqs/pipreqs_output.txt', 'w') as w1:
            w1.write(pipreqs_warnings.stdout.decode('utf-8'))
        with open(f'{root_path}/utils/repo/{author_name}/{repo_name}/repo/.pipreqs/pipreqs_error.txt', 'w') as w2:
            w2.write(pipreqs_warnings.stderr.decode('utf-8'))
    except:
        pass

    checkout_cmd = f"git checkout {sha}"
    subprocess.run(checkout_cmd, cwd=f'{root_path}/utils/repo/{author_name}/{repo_name}/repo', capture_output=True, shell=True)

    # x = subprocess.run('git log -1 --format="%H"', cwd=f'{root_path}/utils/repo/{author_name}/{repo_name}/repo', capture_output=True, shell=True)
    with open(f'{root_path}/output/{author_name}/{repo_name}/sha.txt', 'w') as w1:
        w1.write(sha)

def main():
    # subprocess.run('docker rm -f $(docker ps -aq)', shell=True)
    parser = argparse.ArgumentParser(description='Run script with repository full name as an argument.')
    parser.add_argument('--full_name', type=str, help='The full name of the repository (e.g., user/repo).')
    parser.add_argument('--sha', type=str, help='sha')
    parser.add_argument('--root_path', type=str, help='root path')
    parser.add_argument('--llm', type=str, default='gpt-4o-2024-05-13', help='base LLM name')
    parser.add_argument('--rocm', action='store_true', default=False, help='Enable AMD ROCm mode for GPU workloads')
    parser.add_argument('--rocm-base-image', type=str, default=None, help='Override the default ROCm base image (e.g., rocm/pytorch:latest)')
    parser.add_argument('--api-key', type=str, default=None, help='AMD LLM API Gateway key (or set AMD_LLM_API_KEY env var)')
    parser.add_argument('--verbose', action='store_true', default=False, help='Show full LLM messages, responses, and context tables in terminal')
    parser.add_argument('--no-scale-down', action='store_true', default=False,
                        help='Run the repo exactly as the README describes — no scale-down of training params, no mock data. Uses real data/commands from the README as-is.')
    parser.add_argument('--optimize-kernels', action='store_true', default=False,
                        help='Enable optional kernel performance optimization phase (Phase 2) for CUDA/Triton kernels. '
                             'Without this flag, only correctness hipification runs.')
    parser.add_argument('--kb-path', type=str, default=None,
                        help='Path to the knowledge base SQLite file. Defaults to <root_path>/kb/repo2rocm.db')
    parser.add_argument('--use-claude-code', action='store_true', default=False,
                        help='Use Claude Code Agent SDK instead of AMD LLM API Gateway. '
                             'Requires ANTHROPIC_API_KEY env var or claude CLI installed. '
                             'Enables sub-agents, skills, persistent memory, and built-in tools.')
    parser.add_argument('--claude-code-model', type=str, default=None,
                        help='Model to use with Claude Code (e.g., sonnet, opus, haiku). '
                             'Only used when --use-claude-code is set.')
    parser.add_argument('--claude-code-agentic', action='store_true', default=False,
                        help='Run Claude Code in full agentic mode where it drives the entire '
                             'configuration process autonomously with its built-in tools. '
                             'Only used when --use-claude-code is set.')
    parser.add_argument('--mode', type=str, default='env',
                        choices=['env', 'reproduce', 'full'],
                        help=(
                            'Run mode:\n'
                            '  env       — (default) Set up the repo to run on AMD GPU and verify '
                            'with ROCM_ENV_VERIFIED. Quick smoke-test with scaled-down params.\n'
                            '  reproduce — Download required datasets/checkpoints, run the paper '
                            'experiment with the EXACT paper config (no scale-down), and compare '
                            'results against the paper. Env setup happens as a prerequisite but '
                            'PAPER_RESULT_REPRODUCED/NOT_REPRODUCED is the primary goal.\n'
                            '  full      — Mode 1 first (ROCM_ENV_VERIFIED), then Mode 2 '
                            '(paper experiment + result comparison). Both outputs produced.'
                        ))
    parser.add_argument('--reproduce-results', action='store_true', default=False,
                        help='Alias for --mode full. Kept for backward compatibility.')
    parser.add_argument('--paper-url', type=str, default=None,
                        help='Direct URL to the research paper PDF (e.g. an arXiv pdf link). '
                             'Used by --reproduce-results. If omitted, auto-discovered from README.')
    parser.add_argument('--paper-pdf', type=str, default=None,
                        help='Local path to the research paper PDF. Overrides --paper-url.')
    parser.add_argument('--paper-source-mode', type=str, default='both',
                        choices=['pdf', 'html', 'both'],
                        help='How to build the local paper corpus for planning/retrieval. '
                             'Always preserves /repo/paper.pdf when available; this flag controls '
                             'whether Graphify indexes PDF text, arXiv HTML text, or both.')

    args = parser.parse_args()

    if args.verbose:
        from utils.rich_logger import set_verbose
        set_verbose(True)

    # ── Claude Code mode ──
    # IMPORTANT:
    #   - When `--use-claude-code` is set, normal planner / configuration LLM
    #     calls should route through Claude Code (Agent SDK / CLI).
    #   - We still set the AMD gateway key when present so the system can fall
    #     back cleanly if Claude Code is unavailable.
    #   - Agentic mode remains an extra switch on top of Claude Code mode.
    use_claude_code = args.use_claude_code
    claude_code_agentic = args.claude_code_agentic

    # Always set the AMD gateway key if present so fallback remains available.
    if args.api_key:
        set_api_key(args.api_key)
    elif os.environ.get("AMD_LLM_API_KEY"):
        set_api_key(os.environ["AMD_LLM_API_KEY"])

    from utils.claude_code_client import set_claude_code_mode
    if use_claude_code:
        set_claude_code_mode(enabled=True, model=args.claude_code_model)
        log_info("Claude Code mode ENABLED — using Agent SDK / CLI instead of AMD LLM API Gateway")
        if args.claude_code_model:
            log_info(f"Claude Code model: {args.claude_code_model}")
        if claude_code_agentic:
            log_info("Claude Code agentic mode: ON (autonomous execution with built-in tools)")
    else:
        set_claude_code_mode(enabled=False, model=args.claude_code_model)

    waiting_list = WaitingList()
    conflict_list = ConflictList()

    root_path = args.root_path

    if not os.path.isabs(root_path):
        root_path = os.path.abspath(root_path)

    full_name = args.full_name
    sha = args.sha
    llm = args.llm
    rocm_mode = args.rocm
    rocm_base_image = args.rocm_base_image
    no_scale_down = args.no_scale_down
    optimize_kernels = args.optimize_kernels
    paper_url = args.paper_url
    paper_pdf_arg = args.paper_pdf
    paper_source_mode = args.paper_source_mode

    # ── Resolve run mode ────────────────────────────────────────────────────
    # --reproduce-results is the legacy alias for --mode full.
    run_mode = args.mode
    if args.reproduce_results and run_mode == 'env':
        run_mode = 'full'

    # Mode 2 (reproduce) and Mode 3 (full) both need paper reproduction logic.
    reproduce_results = run_mode in ('reproduce', 'full')

    # Mode 2 always uses the exact paper config — no scale-down of params.
    if run_mode == 'reproduce' and not no_scale_down:
        no_scale_down = True

    if reproduce_results and not paper_url and not paper_pdf_arg:
        log_info(f"--mode {run_mode}: no --paper-url or --paper-pdf given; "
                 "will auto-discover from README.")

    _MODE_DESC = {
        'env':       'Mode 1 — ROCm Env Only  (goal: ROCM_ENV_VERIFIED)',
        'reproduce': 'Mode 2 — Paper Reproduce (goal: PAPER_RESULT_REPRODUCED/NOT_REPRODUCED)',
        'full':      'Mode 3 — Full            (ROCM_ENV_VERIFIED → PAPER_RESULT_REPRODUCED/NOT_REPRODUCED)',
    }
    log_header("REPO2ROCM", f"Self-Evolving Multi-Agent System")
    log_phase("CONFIGURATION")
    log_info(f"Run mode:   {_MODE_DESC[run_mode]}")
    log_info(f"Repository: {full_name}")
    log_info(f"SHA: {sha}")
    log_info(f"LLM: {llm}")
    log_info(f"ROCm Mode: {rocm_mode}")
    if no_scale_down:
        log_info(f"No-Scale-Down: ON (will follow README as-is, no mock data)")
    if optimize_kernels:
        log_info(f"Kernel Optimization: ON (Phase 2 performance tuning enabled)")
    if reproduce_results:
        log_info(f"Paper reproduction: ON")
        if paper_pdf_arg:
            log_info(f"  Paper PDF (local): {paper_pdf_arg}")
        elif paper_url:
            log_info(f"  Paper URL: {paper_url}")
        else:
            log_info(f"  Paper source: auto-discover from README")
    log_info(f"Root Path: {root_path}")

    # ── Initialize Intelligence Layer ──
    kb_path = args.kb_path or os.path.join(root_path, "kb", "repo2rocm.db")
    traj_db_path = os.path.join(root_path, "kb", "trajectories.db")
    os.makedirs(os.path.dirname(kb_path), exist_ok=True)

    kb_store = KBStore(kb_path)
    seed_if_empty(kb_store)
    trajectory_store = TrajectoryStore(traj_db_path)
    error_classifier = ErrorClassifier(kb_store)
    rule_engine = RuleEngine(kb_store)
    memory_provider = BuildMemoryProvider(
        kb_store, trajectory_store, error_classifier, rule_engine
    )
    distiller = TrajectoryDistiller(kb_store, trajectory_store, llm)

    kb_stats = kb_store.get_stats()
    traj_stats = trajectory_store.get_stats()
    log_info(f"Knowledge Base: {kb_stats.get('active_rules', 0)} rules, "
             f"{kb_stats.get('error_patterns_count', 0)} error patterns, "
             f"{kb_stats.get('deterministic_rules', 0)} deterministic")
    log_info(f"Trajectory Store: {traj_stats.get('total_attempts', 0)} past attempts, "
             f"{traj_stats.get('successful_attempts', 0)} successful, "
             f"{traj_stats.get('unique_repos', 0)} unique repos")
    if os.path.exists(f'{root_path}/output/{full_name}/patch'):
        rm_cmd = f"rm -rf {root_path}/output/{full_name}/patch"
        subprocess.run(rm_cmd, shell=True, check=True)
    output_dir = f'{root_path}/output/{full_name.split("/")[0]}/{full_name.split("/")[1]}'
    if not os.path.exists(output_dir):
        subprocess.run(f'mkdir -p {output_dir}', shell=True)

    # Start file logging into the output directory
    init_file_log(output_dir)

    if os.path.exists(f'{root_path}/utils/repo/{full_name}'):
        init_cmd = f"rm -rf {root_path}/utils/repo/{full_name} && mkdir -p {root_path}/utils/repo/{full_name}"
    else:
        init_cmd = f"mkdir -p {root_path}/utils/repo/{full_name}"
    subprocess.run(init_cmd, check=True, shell=True)
    # Watchdog timeout removed: paper-reproduction experiments (full benchmark
    # harnesses such as MMLU/HumanEval/GSM8K) can legitimately take longer
    # than 2 hours. Rely on the per-turn LLM cap and Docker timeouts instead.

    log_phase("CLONING REPOSITORY", f"{full_name} @ {sha[:12]}")
    download_repo(root_path, full_name, sha)
    log_success(f"Repository cloned: {full_name}")

    # ── Per-run mempalace memory (Stage 2) ──
    run_memory = RunMemory.create(full_name, sha)
    log_info(f"Run memory: wing={getattr(run_memory, 'wing', '?')} "
             f"palace={getattr(run_memory, 'palace_path', '?')}")

    # ── Per-run graphify code graph (Stage 4) ──
    repo_path = f"{root_path}/utils/repo/{full_name}/repo"
    try:
        from learning.graphify_provider import GraphifyProvider
        graphify_provider = GraphifyProvider.create(repo_path)
        graphify_provider.build_or_refresh()
        graphify_provider.index_repo_corpus()
        log_info(f"Graphify graph: {graphify_provider.stats()}")
    except Exception as _gp_e:
        log_info(f"Graphify provider disabled: {_gp_e}")
        class _NG:
            enabled = False
            graph_json = ""
            def __getattr__(self, n):
                def _noop(*a, **kw): return ""
                return _noop
        graphify_provider = _NG()

    # ── Upfront Planning Phase ──

    # Compute build fingerprint for KB queries
    log_phase("BUILD FINGERPRINT")
    fingerprint = compute_fingerprint(repo_path, repo_id=full_name)
    log_info(f"Fingerprint: frameworks={sorted(fingerprint.frameworks)}, "
             f"cuda_deps={sorted(fingerprint.cuda_deps)}, "
             f"build_system={fingerprint.build_system}, "
             f"workload={fingerprint.workload_type}")
    log_info(f"Custom CUDA kernels: {fingerprint.has_custom_cuda_kernels}, "
             f"Triton kernels: {fingerprint.has_triton_kernels}, "
             f"Distributed: {fingerprint.has_distributed}")

    # Query KB for prior intelligence (BEGIN phase)
    from storage.models import MemoryRequest, MemoryPhase
    begin_request = MemoryRequest(
        query=full_name,
        context={
            "frameworks": sorted(fingerprint.frameworks),
            "cuda_deps": sorted(fingerprint.cuda_deps),
            "workload_type": fingerprint.workload_type,
            "rocm_mode": rocm_mode,
        },
        phase=MemoryPhase.BEGIN.value,
        fingerprint=fingerprint,
    )
    begin_memory = memory_provider.provide_memory(begin_request)
    kb_context = memory_provider.format_begin_for_prompt(begin_memory)
    learned_context = memory_provider.format_begin_for_planner(begin_memory)
    if kb_context:
        log_info(f"KB returned {len(begin_memory.items)} memory items "
                 f"(confidence: {begin_memory.confidence:.2f})")

    # ── Resolve paper PDF (for --reproduce-results) ──
    paper_pdf_path: Optional[str] = None
    paper_arxiv_id: str = ""
    if reproduce_results:
        from agents.paper_agent import PaperAgent
        from agents.paper_corpus import extract_arxiv_id_from_url
        paper_dest = os.path.join(repo_path, "paper.pdf")
        paper_agent_for_dl = PaperAgent(llm=llm)
        try:
            if paper_pdf_arg and os.path.exists(paper_pdf_arg):
                shutil.copyfile(paper_pdf_arg, paper_dest)
                paper_pdf_path = paper_dest
                log_info(f"Paper PDF copied from {paper_pdf_arg} to {paper_dest}")
            elif paper_url:
                paper_arxiv_id = extract_arxiv_id_from_url(paper_url) or ""
                paper_pdf_path = paper_agent_for_dl.download_paper(paper_url, paper_dest)
                if paper_pdf_path:
                    log_info(f"Paper PDF downloaded to {paper_pdf_path}")
            else:
                readme_text = ""
                for rn in ("README.md", "readme.md", "README.rst", "README.txt", "README"):
                    rp = os.path.join(repo_path, rn)
                    if os.path.isfile(rp):
                        try:
                            with open(rp, "r", encoding="utf-8", errors="ignore") as rf:
                                readme_text = rf.read()
                            break
                        except Exception:
                            pass
                arxiv_id = paper_agent_for_dl.extract_paper_link(readme_text) if readme_text else None
                if arxiv_id:
                    paper_arxiv_id = arxiv_id
                    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                    paper_pdf_path = paper_agent_for_dl.download_paper(url, paper_dest)
                    if paper_pdf_path:
                        log_info(f"Paper PDF auto-discovered from README (arXiv:{arxiv_id}) and saved to {paper_pdf_path}")
                else:
                    log_info("No paper URL provided and none found in README; paper reproduction will be skipped.")
                    reproduce_results = False
        except Exception as e:
            log_info(f"Paper PDF resolution failed: {e}; paper reproduction will be skipped.")
            reproduce_results = False
            paper_pdf_path = None

    # Stage 4/6: Index the paper into graphify BEFORE generate_plan.
    # Static paper content belongs to graphify-out/, not mempalace.
    # Mempalace stores only references / experiment state.
    paper_corpus = None
    if reproduce_results and paper_pdf_path and os.path.exists(paper_pdf_path):
        try:
            from agents.paper_corpus import build_paper_corpus
            paper_corpus = build_paper_corpus(
                paper_pdf_path,
                arxiv_id=paper_arxiv_id,
                source_mode=paper_source_mode,
            )
            if paper_corpus.has_text():
                log_info(
                    "Indexing paper into graphify: "
                    f"mode={paper_corpus.source_mode}, "
                    f"resolved={paper_corpus.resolved_modes}, "
                    f"chars={len(paper_corpus.index_text):,}"
                )
                graphify_provider.index_paper_sources(paper_corpus.source_payloads())
                run_memory.write_experiment_state(
                    "paper_index",
                    {
                        "backend": "graphify",
                        "paper_pdf_path": paper_pdf_path,
                        "paper_source_mode": paper_source_mode,
                        "resolved_modes": paper_corpus.resolved_modes,
                        "paper_arxiv_id": paper_arxiv_id,
                        "indexed_chars": len(paper_corpus.index_text),
                    },
                    source_file="paper_index.json",
                )
                run_memory.write_context_ref(
                    kind="paper_index",
                    ref_id="graphify:paper_chunks",
                    source=graphify_provider.paper_chunks_jsonl,
                    why_relevant="static paper corpus for planner/stage2 retrieval",
                    extra={
                        "paper_pdf_path": paper_pdf_path,
                        "paper_source_mode": paper_source_mode,
                        "paper_arxiv_id": paper_arxiv_id,
                        "resolved_modes": paper_corpus.resolved_modes,
                        "indexed_chars": len(paper_corpus.index_text),
                    },
                )
            else:
                log_info(
                    "Paper corpus resolved but contained no indexable text; "
                    f"provenance={getattr(paper_corpus, 'provenance', {})}"
                )
        except Exception as _ext_e:
            log_info(f"Paper indexing into graphify failed: {_ext_e}")

    plan, recommended_image, paper_context = generate_plan(
        repo_path=repo_path,
        full_name=full_name,
        rocm_mode=rocm_mode,
        llm=llm,
        no_scale_down=no_scale_down,
        paper_pdf_path=paper_pdf_path,
        paper_corpus=paper_corpus,
        reproduce_results=reproduce_results,
        run_memory=run_memory,
        graphify_provider=graphify_provider,
        learned_context=learned_context,
        run_mode=run_mode,
    )
    print_plan(plan)
    paper_experiments = paper_context.get("experiments", []) if reproduce_results else []
    paper_title = paper_context.get("title", "") if reproduce_results else ""

    with open(f'{root_path}/output/{full_name}/plan.txt', 'w') as pf:
        pf.write(plan)
    with open(f'{root_path}/output/{full_name}/run_mode.txt', 'w') as mf:
        mf.write(f"{run_mode}\n{_MODE_DESC[run_mode]}\n")
    log_success(f"Plan saved to output/{full_name}/plan.txt  [mode={run_mode}]")

    # Persist plan + paper-experiment shortlist + base-image decision into mempalace.
    # NOTE: raw paper text is no longer stored in mempalace; only references/state.
    try:
        run_memory.write_plan(plan)
        if reproduce_results and paper_experiments:
            run_memory.write_paper_experiments(paper_experiments)
            run_memory.write_experiment_state(
                "chosen_experiment_shortlist",
                {
                    "title": paper_title,
                    "num_candidates": len(paper_experiments),
                    "top_candidate": paper_experiments[0] if paper_experiments else {},
                },
                source_file="paper_experiment_state.json",
            )
        if recommended_image:
            run_memory.write_decision("recommended_base_image", recommended_image,
                                      reason="planner")
    except Exception as _mp_e:
        log_info(f"Run memory write failed: {_mp_e}")

    # Register build attempt
    build_attempt = BuildAttempt(
        repo_id=full_name,
        repo_url=f"https://github.com/{full_name}",
        sha=sha,
        fingerprint=fingerprint,
        docker_image=recommended_image or "",
        plan_text=plan[:5000],
        trajectory_file=f"{root_path}/output/{full_name}/trajectory.jsonl",
    )
    trajectory_store.start_attempt(build_attempt)

    trajectory = []

    if rocm_mode:
        if rocm_base_image:
            base_image = rocm_base_image
            log_info(f"ROCm mode: using CLI-specified image: {base_image}")
            run_memory.write_decision("base_image", base_image, reason="cli-override")
        elif recommended_image:
            base_image = recommended_image
            log_info(f"ROCm mode: using planner-recommended image: {base_image}")
            run_memory.write_decision("base_image", base_image, reason="planner-recommended")
        else:
            base_image = 'rocm/pytorch:latest'
            log_info(f"ROCm mode: using default fallback image: {base_image}")
            run_memory.write_decision("base_image", base_image, reason="default-fallback")
    else:
        base_image = 'python:3.10'
        log_info(f"Using base image: {base_image}")
        run_memory.write_decision("base_image", base_image, reason="non-rocm")

    # ── Set up Claude Code project files (if enabled) ──
    if use_claude_code:
        from utils.claude_code_client import setup_claude_code_project
        claude_dir = setup_claude_code_project(
            repo_root=repo_path,
            plan=plan,
            kb_context=kb_context,
            learned_context=learned_context,
            rocm_mode=rocm_mode,
            paper_pdf_path=paper_pdf_path if reproduce_results else None,
            reproduce_results=reproduce_results,
        )
        log_info(f"Claude Code project files created at: {claude_dir}")

    log_phase("STARTING DOCKER CONTAINER")
    configuration_sandbox = Sandbox(base_image, full_name, root_path, rocm_mode=rocm_mode)
    configuration_sandbox.start_container()
    log_success("Container started and repo copied")

    # ── Choose execution mode ──
    if use_claude_code and claude_code_agentic:
        log_phase("RUNNING CLAUDE CODE AGENTIC MODE")
        from utils.claude_code_client import run_claude_code_agent, get_rocm_subagents

        container_id = configuration_sandbox.container.id

        agentic_system_prompt = Configuration._build_agentic_system_prompt_static(
            image_name=base_image,
            rocm_mode=rocm_mode,
            no_scale_down=no_scale_down,
            plan=plan,
            kb_context=kb_context,
            reproduce_results=reproduce_results,
        )

        if reproduce_results:
            agentic_task = (
                f"Configure the Docker container for repository '{full_name}' to run on AMD ROCm GPUs "
                f"and then reproduce one experiment from the paper.\n\n"
                f"All commands must be executed inside Docker container '{container_id}' using:\n"
                f"  docker exec -w /repo {container_id} bash -c '...'\n\n"
                f"STAGE 1 - Environment verification:\n"
                f"  Follow the strategic plan. When the environment is fully configured and the "
                f"project's main script produces real output on the GPU, execute:\n"
                f"    docker exec {container_id} bash -c 'echo ROCM_ENV_VERIFIED'\n\n"
                f"STAGE 2 - Paper result reproduction:\n"
                f"  After ROCM_ENV_VERIFIED, run the 'Chosen experiment' from the PAPER REPRODUCTION "
                f"TARGET section of the plan, with the EXACT paper/README config (no scale-down). "
                f"Capture its stdout to /repo/paper_experiment.log. Then invoke the 'paper-reproducer' "
                f"sub-agent to compare the produced metric(s) against the paper's reported value(s) "
                f"(paper.pdf is at /repo/paper.pdf inside the container — read it from the host copy "
                f"at {paper_pdf_path}). Based on its JSON verdict, execute exactly ONE of:\n"
                f"    echo PAPER_RESULT_REPRODUCED metric=<name> actual=<v> expected=<v> delta_pct=<x>\n"
                f"    echo PAPER_RESULT_NOT_REPRODUCED <one-line reason>\n"
                f"Do NOT fabricate numbers. If parsing fails, rerun once; if still unparseable, echo "
                f"PAPER_RESULT_NOT_REPRODUCED with a parsing note."
            )
        else:
            agentic_task = (
                f"Configure the Docker container for repository '{full_name}' to run on AMD ROCm GPUs. "
                f"Follow the strategic plan provided in the system prompt. "
                f"All commands must be executed inside Docker container '{container_id}' using: "
                f"docker exec -w /repo {container_id} bash -c '...'\n\n"
                f"When the environment is fully configured and the project runs successfully, "
                f"execute: docker exec {container_id} bash -c 'echo ROCM_ENV_VERIFIED'"
            )

        agent_result = run_claude_code_agent(
            task_prompt=agentic_task,
            system_prompt=agentic_system_prompt,
            working_directory=root_path,
            model=args.claude_code_model,
            max_turns=100,
            agents=get_rocm_subagents(),
            docker_container_id=container_id,
        )

        log_info(f"Claude Code agent completed: success={agent_result['success']}, "
                 f"turns={agent_result['total_turns']}, "
                 f"tokens={agent_result['usage']['total_tokens']}")

        final_text = agent_result.get("final_text", "") or ""
        rocm_verified = "ROCM_ENV_VERIFIED" in final_text
        paper_reproduced = "PAPER_RESULT_REPRODUCED" in final_text
        paper_not_reproduced = "PAPER_RESULT_NOT_REPRODUCED" in final_text

        if rocm_verified:
            test_lines = ['ROCM_ENV_VERIFIED']
            if reproduce_results:
                if paper_reproduced:
                    test_lines.append('PAPER_RESULT_REPRODUCED')
                elif paper_not_reproduced:
                    test_lines.append('PAPER_RESULT_NOT_REPRODUCED')
            with open(f'{root_path}/output/{full_name}/test.txt', 'w') as w3:
                w3.write('\n'.join(test_lines) + '\n')

        if reproduce_results:
            verdict = (
                "reproduced" if paper_reproduced else
                "not_reproduced" if paper_not_reproduced else
                "unknown"
            )

            def _extract_marker_line(text: str, marker: str) -> str:
                for line in text.splitlines():
                    if marker in line:
                        return line.strip()
                return ""

            repro_record = {
                "verdict": verdict,
                "rocm_env_verified": rocm_verified,
                "paper_pdf_path": paper_pdf_path,
                "paper_url": paper_url,
                "reproduced_line": _extract_marker_line(final_text, "PAPER_RESULT_REPRODUCED"),
                "not_reproduced_line": _extract_marker_line(final_text, "PAPER_RESULT_NOT_REPRODUCED"),
                "final_text_tail": final_text[-4000:],
            }
            try:
                with open(f'{root_path}/output/{full_name}/paper_reproduction.json', 'w') as w4:
                    w4.write(json.dumps(repro_record, indent=2))
                if verdict == "reproduced":
                    log_success(f"Paper result reproduced: {repro_record['reproduced_line']}")
                elif verdict == "not_reproduced":
                    log_error(f"Paper result NOT reproduced: {repro_record['not_reproduced_line']}")
                else:
                    log_info("Paper reproduction verdict: unknown (no marker in final output)")
            except Exception as e:
                log_info(f"Failed to write paper_reproduction.json: {e}")

        msg = trajectory
        outer_commands = agent_result.get("tool_calls", [])
    else:
        log_phase("RUNNING CONFIGURATION AGENT")
        observer_client = None
        try:
            from observers.observer_agent import ObserverClient
            observer_client = ObserverClient(
                output_dir=output_dir,
                llm=llm,
                api_key=args.api_key or os.environ.get("AMD_LLM_API_KEY", ""),
                enabled=bool(llm),
                use_claude_code=bool(use_claude_code),
                claude_code_model=getattr(args, "claude_code_model", None),
            )
            observer_client.start()
            log_info("Observer sidecar: ON")
        except Exception as _obs_e:
            observer_client = None
            log_info(f"Observer sidecar disabled: {_obs_e}")
        configuration_agent = Configuration(
            configuration_sandbox, base_image, full_name, root_path, llm, 100,
            rocm_mode=rocm_mode, plan=plan, no_scale_down=no_scale_down,
            error_classifier=error_classifier,
            rule_engine=rule_engine,
            memory_provider=memory_provider,
            trajectory_store=trajectory_store,
            build_attempt=build_attempt,
            kb_context=kb_context,
            optimize_kernels=optimize_kernels,
            use_claude_code=use_claude_code,
            reproduce_results=reproduce_results,
            paper_pdf_path=paper_pdf_path,
            paper_experiments=paper_experiments,
            paper_title=paper_title,
            run_memory=run_memory,
            graphify_provider=graphify_provider,
            observer_client=observer_client,
            run_mode=run_mode,
        )
        try:
            msg, outer_commands = configuration_agent.run('/tmp', trajectory, waiting_list, conflict_list)
        finally:
            if observer_client is not None:
                observer_client.shutdown()
        # Stage 1 / 3 compaction summary
        try:
            o = getattr(configuration_agent, "_compaction_orig_chars", 0)
            s = getattr(configuration_agent, "_compaction_short_chars", 0)
            if o:
                log_info(f"Stage 1 compaction: observation original={o:,} chars, "
                         f"compacted={s:,} chars ({(1 - s / o) * 100:.1f}% reduction)")
            ao = getattr(configuration_agent, "_appendix_old_chars", 0)
            an = getattr(configuration_agent, "_appendix_new_chars", 0)
            if ao:
                log_info(f"Stage 3 appendix: would-have-been={ao:,} chars, "
                         f"actual={an:,} chars ({(1 - an / max(ao, 1)) * 100:.1f}% reduction)")
        except Exception:
            pass
    with open(f'{root_path}/output/{full_name.split("/")[0]}/{full_name.split("/")[1]}/track.json', 'w') as w1:
        w1.write(json.dumps(msg, indent=4))
    commands = configuration_sandbox.stop_container()
    with open(f'{root_path}/output/{full_name.split("/")[0]}/{full_name.split("/")[1]}/inner_commands.json', 'w') as w2:
        w2.write(json.dumps(commands, indent=4))
    with open(f'{root_path}/output/{full_name.split("/")[0]}/{full_name.split("/")[1]}/outer_commands.json', 'w') as w3:
        w3.write(json.dumps(outer_commands, indent=4))

    run_success = False
    try:
        integrate_dockerfile(
            f'{root_path}/output/{full_name}',
            runtime_base_image=base_image,
        )
        msg_str = f'Generate success!'
        log_success(msg_str)
        with open(f'{root_path}/output/{full_name}/track.txt', 'a') as a1:
            a1.write(msg_str + '\n')
        run_success = True
    except Exception as e:
        msg_str = f'integrate_docker failed, reason:\n {e}'
        log_error(msg_str)
        with open(f'{root_path}/output/{full_name}/track.txt', 'a') as a1:
            a1.write(msg_str + '\n')

    # ── Post-Build Learning Pipeline ──
    log_phase("LEARNING PIPELINE")
    build_outcome = BuildOutcome.SUCCESS.value if run_success else BuildOutcome.FAILURE.value
    elapsed_minutes = (time.time() - build_attempt.started_at) / 60.0

    trajectory_store.complete_attempt(
        build_attempt.id, build_outcome, elapsed_minutes,
        total_turns=len(trajectory), total_tokens=0,
    )
    if run_success:
        trajectory_store.mark_success_retroactive(build_attempt.id)

    build_attempt.outcome = build_outcome
    build_attempt.duration_minutes = elapsed_minutes
    traj_records = trajectory_store.load_trajectory(build_attempt.trajectory_file)

    if run_success:
        for rec in traj_records:
            rec.led_to_success = True

    try:
        applied_count = distiller.distill_and_apply(build_attempt, traj_records)
        log_info(f"Learning pipeline: {applied_count} KB updates applied from this build")
    except Exception as e:
        log_info(f"Learning pipeline encountered an error: {e}")

    updated_stats = kb_store.get_stats()
    log_info(f"KB after learning: {updated_stats.get('active_rules', 0)} rules, "
             f"{updated_stats.get('error_patterns_count', 0)} patterns")

    kb_store.close()
    trajectory_store.close()
    memory_provider.reset_session()

    # ── Per-run memory is runtime-only: keep the trace, but do not turn it
    # into a second long-term natural-language KB. Structured KB/trajectory
    # learning above remains the single durable learning path.
    try:
        if getattr(run_memory, "enabled", False):
            log_info(f"Run memory stats: {run_memory.stats()}")
            run_memory.close()
    except Exception as _mp_e:
        log_info(f"Run memory shutdown failed: {_mp_e}")

    close_file_log()
    log_info(f"Debug log saved to: {output_dir}/agent_debug_log.txt")

if __name__ == '__main__':
    try:
        subprocess.run('docker rmi $(docker images --filter "dangling=true" -q) > /dev/null 2>&1', shell=True)
    except:
        pass
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    log_phase("COMPLETE", f"Total time: {elapsed_time:.1f}s")
    console.print(f"\n[bold]Total elapsed time: {elapsed_time:.1f}s[/bold]\n")