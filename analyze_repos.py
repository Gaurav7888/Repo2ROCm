#!/usr/bin/env python3
"""Static AMD/ROCm portability analyzer.

For each repo: clone it, run repo2rocm's deterministic recon (CUDA-dep / custom-kernel
/ hazard / requirements detection), then have a READ-ONLY LLM agent inspect the actual
code + README + requirements and judge:

    REQUIRES_REPO2ROCM  — needs AMD-specific work (CUDA-only deps, custom CUDA kernels,
                          hardcoded CUDA assumptions, cuda-pinned torch builds, bugs/patches)
    OUT_OF_BOX          — runs on AMD as-is on a stock ROCm PyTorch image (device-agnostic
                          torch code, all deps have ROCm/CPU wheels, no custom kernels)

No Docker, no GPU — pure static analysis, so it runs with high concurrency.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from rich.console import Console

from repo2rocm.agents.definition import AgentDefinition
from repo2rocm.agents.lifecycle import RunAgentParams, run_agent
from repo2rocm.bootstrap import bootstrap
from repo2rocm.core.permissions import PermissionMode
from repo2rocm.observability.transcripts import TranscriptStore
from repo2rocm.recon import run_recon
from repo2rocm.tools.base import ReadFileState, ToolUseContext

ROOT = Path("/home/gsarkar/Repo2ROCm")
LOG_DIR = ROOT / "analysis_logs"
RESULTS_PATH = ROOT / "analysis_results.jsonl"

KNOWN_SHAS = {
    "allenai/understanding_mcqa": "8b1aea4c1bce5e5295f93b79a899a79c2b1fc626",
    "Cohere-Labs-Community/m-rewardbench": "708d3bde7bbf110d5297bb0feb6729400ee7ade1",
    "sarahmart/HARDMath": "9ade062a80b7d475666b11ae13a2fb0d7b7f0341",
    "ZhiningLiu1998/SelfElicit": "2fa9d1d3ab5a14e583de16fa974dfcd617d861f8",
    "FYYFU/HeadKV": "0862a0955fe82e9ff611d59541918e02c5def625",
    "FFY0/AdaKV": "04497abac4c1a58426f3daf1014578990e225cc5",
    "thunlp/FR-Spec": "29d0136b43d372d7d48806db8702cc9c813fdccf",
    "ryienh/jl-metric": "ae0a1e4c4be07f8675234207cca5abafb6d1c65c",
    "shimo-lab/modelmap": "ce0826b8a494ee5cef04cde7f5eb4ebe51e4d307",
    "togethercomputer/moa": "1b5cab0f0905d9da821e37322ac6df96ba65e1a7",
    "SongW-SW/CEB": "436943b78fdda84e0fc411e994eecd5646187f17",
}

REPOS = [
    "qiuzh20/gated_attention", "facebookresearch/luckmatters", "backprop07/Self-Certainty",
    "THU-MIG/PrefixKV", "microsoft/SeqSNN", "showmeon/TimeEmb", "llmsresearch/paperbanana",
    "zhoujiahuan1991/NeurIPS2025-KFF", "nanomaoli/llm_reproducibility", "parameterlab/c-seo-bench",
    "alessiopittiglio/mm-argfallacy", "dayeonki/askqe", "sapienzanlp/bookcoref",
    "flairNLP/VerbalizED", "dxlong2000/NLPromptEval", "tanghme0w/ACL25-CoPE",
    "batmanlab/Ladder", "dmis-lab/TemporalHead", "UAlberta-NLP/SemEval2025-EAMT",
    "XiaoAI1989/ORMind", "SCUNLP/ELABORATION", "decisionintelligence/CATCH",
    "kaistAI/Knowledge-Entropy", "LehengTHU/AlphaRec", "OpenBMB/IoA", "Graph-COM/LayerDAG",
    "murtylab/topoloss", "neuro-galaxy/torch_brain", "shengliu66/VTI", "pouyahmdn/LCPO",
    "codezakh/DataEnvGym", "ModelTC/HarmoniCa", "xihongyang1999/AIRMVC", "DCDmllm/HealthGPT",
    "apple/ml-tarflow", "xinyuluo8561/Stacey", "GuanchengWan/EARTH",
    "Kambm/convolutional_diffusion", "mmoharami/Risk-Sensitive-CMDP",
    "YZY-stack/Effort-AIGI-Detection", "BruceGeLi/TOP_ERL_ICLR25", "liukidar/pcx",
    "yjucho1/CoMRes", "mint-vu/MCNC", "1202kbs/GCTM", "DeepakTatachar/CSL-Mem",
    "allenai/WildBench", "xlang-ai/BRIGHT", "THUKElab/CLEME", "FFY0/AdaKV", "FYYFU/HeadKV",
    "allenai/understanding_mcqa", "Cohere-Labs-Community/m-rewardbench", "sarahmart/HARDMath",
    "ZhiningLiu1998/SelfElicit", "thunlp/FR-Spec", "ryienh/jl-metric", "shimo-lab/modelmap",
    "togethercomputer/moa", "SongW-SW/CEB",
]


SYSTEM_PROMPT = """You are a senior AMD/ROCm porting engineer. Your job is to STATICALLY
judge whether a research repository (written and tested on NVIDIA CUDA GPUs) can run
on AMD ROCm GPUs OUT OF THE BOX, or whether it would REQUIRE porting work (the kind of
work the `repo2rocm` tool automates: installing AMD-specific libraries, swapping CUDA-only
wheels for ROCm builds, small code changes, and fixing bugs/incompatibilities).

ASSUME the target is a stock `rocm/pytorch:latest` image (ROCm PyTorch already installed
and working). You are NOT running anything — judge from the README, requirements/config,
and the actual source code. You have READ-ONLY tools: Read, Grep, Glob.

DECISION CRITERIA — mark **REQUIRES_REPO2ROCM** if ANY of these hold:
  * Custom CUDA / C++ extensions in the repo: `.cu`/`.cuh` files, `CUDAExtension`,
    `torch.utils.cpp_extension`, `nvcc`/`setup.py build_ext` compiling CUDA.
  * CUDA-only Python deps lacking a drop-in ROCm wheel (need AMD fork / special build /
    base-image swap): `flash-attn`/`flash_attn`, `bitsandbytes`, `xformers`, `apex`,
    `mamba-ssm`, `causal-conv1d`, `vllm` (needs rocm/vllm image), `tensorrt`, `faiss-gpu`,
    `cupy`, `pycuda`, `deepspeed` (often), `triton` custom CUDA kernels, `nvidia-*-cu1x`
    wheels, `torch-scatter/torch-sparse/torch-geometric` pinned to CUDA builds.
  * Requirements/README pin a CUDA torch build (e.g. `torch==x.y.z+cu121`, an
    `--index-url https://download.pytorch.org/whl/cu1xx`, or conda `pytorch-cuda=...`):
    installing that would clobber the ROCm torch.
  * Hardcoded CUDA-only assumptions that error on ROCm: e.g. `torch.backends.cudnn`
    flags without a HIP guard, hardcoded compute capability, `nccl`-specific code,
    `torch.cuda.amp` misuse, fp8 / specific kernels unsupported on the target.
  * Known bugs / patches needed to run.

ALSO mark **REQUIRES_REPO2ROCM** for NON-CUDA "silent" blockers that genuinely stop the
repo from running and need real fixing work (repo2rocm fixes code/dep bugs too):
  * SYNTAX / CODE BUGS: any `.py` fails to parse (syntax error, Python-2 print statements),
    or broken INTRA-REPO imports (the code imports a repo module/path that doesn't exist),
    or the repo ships only a README / no actual source to run. A syntax-check result is
    provided below.
  * REAL VERSION / API BREAKAGE: requirements pin versions that are yanked/unavailable or
    mutually conflicting; OR the code clearly calls an API that the (unpinned) current
    release removed/renamed (e.g. `huggingface_hub.cached_download`) so it WILL crash.
    Only count this when you can point to the actual broken call — not speculative drift.

DO **NOT** flip a repo to REQUIRES for these (they are normal setup, NOT porting work):
  * An ordinary third-party PyPI package imported but absent from requirements.txt
    (e.g. numpy, pandas, transformers, matplotlib, tqdm, click, einops, scikit-learn).
    A plain `pip install <pkg>` fixes it — that is OUT_OF_BOX. The "candidate undeclared
    imports" list below is only a HINT; treat a missing common package as a minor note,
    NOT a blocker. EXCEPTION: if the undeclared package is a CUDA-only lib (flash_attn,
    bitsandbytes, vllm, xformers, deepspeed, mamba-ssm, apex, etc.), that IS a blocker.
  * Needing to `pip install` documented deps (README `pip install ...`), or having no
    requirements file at all but only ordinary deps.

Mark **OUT_OF_BOX** if the repo is device-agnostic PyTorch with NO hard blocker above:
  * Uses `torch`, `transformers`, `numpy`, `datasets`, etc. with `.to('cuda')` /
    `device='cuda'` (HIP transparently maps `cuda` -> ROCm — this ALONE is NOT a blocker).
  * All deps have ROCm or pure-Python/CPU wheels; no custom kernels; no cuda-pinned torch.
  * Code parses cleanly; intra-repo imports resolve; only ordinary pip installs needed.
  * May need normal `pip install` of non-torch deps (that's fine and expected).

IMPORTANT NUANCES:
  * `device='cuda'` / `.cuda()` usage is FINE on ROCm — do not flag it by itself.
  * Needing to `pip install` ordinary packages is FINE (not a porting concern).
  * `triton` from pip generally works on ROCm for Triton-lang kernels; only flag it if
    the repo ships custom CUDA (.cu) Triton or pins a CUDA-specific build.
  * Be concrete: cite the exact file / requirement line that drives your verdict.

WORKFLOW: read the README, then requirements.txt / setup.py / pyproject.toml /
environment.yml, then grep the code for `.cu` files, `CUDAExtension`, `cpp_extension`,
`flash_attn`, `bitsandbytes`, `cudnn`, `+cu1`, `download.pytorch.org/whl/cu`. Inspect the
few files that matter. Keep it to ~15 tool calls.

FINAL MESSAGE — output EXACTLY these lines (and nothing after):
VERDICT: REQUIRES_REPO2ROCM | OUT_OF_BOX
CONFIDENCE: high | medium | low
BLOCKERS: <comma-separated concrete blockers, or "none">
REQUIRED_CHANGES: <one line: libs to install / code changes / image swap, or "none">
RATIONALE: <one or two sentences citing the specific files/lines>
"""


ANALYZER_AGENT = AgentDefinition(
    name="rocm_portability_analyzer",
    description="Read-only static judge: REQUIRES_REPO2ROCM vs OUT_OF_BOX.",
    allowed_tools=["Read", "Grep", "Glob"],
    permission_mode=PermissionMode.BYPASS,
    max_turns=45,
    max_tokens=4_096,
    system_prompt_template=SYSTEM_PROMPT,
    color="green",
)


_STDLIB = set(getattr(sys, "stdlib_module_names", set())) | {
    "__future__", "typing_extensions", "setuptools", "pkg_resources", "pip",
}
_IMPORT_ALIAS = {
    "cv2": "opencv-python", "PIL": "pillow", "sklearn": "scikit-learn", "yaml": "pyyaml",
    "skimage": "scikit-image", "bs4": "beautifulsoup4", "Crypto": "pycryptodome",
    "dateutil": "python-dateutil", "dotenv": "python-dotenv", "yaml": "pyyaml",
    "torch_geometric": "torch-geometric", "torch_scatter": "torch-scatter",
    "torch_sparse": "torch-sparse", "google": "protobuf", "OpenSSL": "pyopenssl",
    "attr": "attrs", "fitz": "pymupdf", "wandb": "wandb", "omegaconf": "omegaconf",
    "hydra": "hydra-core", "pytorch_lightning": "pytorch-lightning", "lightning": "lightning",
    "transformers": "transformers", "huggingface_hub": "huggingface-hub",
}
_SKIP_DIRS = {".git", "venv", ".venv", "env", "node_modules", "build", "dist",
              "__pycache__", ".tox", "site-packages", "third_party", "external"}


def _iter_py_files(repo_path: Path, limit: int = 1200):
    n = 0
    for p in repo_path.rglob("*.py"):
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        yield p
        n += 1
        if n >= limit:
            return


def syntax_errors(repo_path: Path) -> list[str]:
    errs: list[str] = []
    for p in _iter_py_files(repo_path):
        try:
            ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as e:
            errs.append(f"{p.relative_to(repo_path)}:{e.lineno}: {e.msg}")
        except Exception:
            pass
        if len(errs) >= 25:
            break
    return errs


def _top_imports(repo_path: Path) -> set[str]:
    names: set[str] = set()
    for p in _iter_py_files(repo_path):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    names.add(a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    names.add(node.module.split(".")[0])
    return names


def _local_modules(repo_path: Path) -> set[str]:
    """Collect local module/package names repo-wide so we don't flag intra-repo
    imports as undeclared (handles nested layouts like `pkg/agents/...`)."""
    local: set[str] = set()
    # every .py file stem is an importable local module name
    for p in _iter_py_files(repo_path):
        local.add(p.stem)
        # any directory that is a package (or just a code dir) is a local name
        for part in p.relative_to(repo_path).parts[:-1]:
            local.add(part)
    return local


def _declared_deps_blob(repo_path: Path) -> str:
    blob = []
    for name in ("requirements.txt", "requirements-dev.txt", "requirements_dev.txt",
                 "setup.py", "setup.cfg", "pyproject.toml", "environment.yml",
                 "environment.yaml", "Pipfile"):
        f = repo_path / name
        if f.is_file():
            try:
                blob.append(f.read_text(encoding="utf-8", errors="replace").lower())
            except Exception:
                pass
    # README-documented installs count as "available" (out-of-box = follow the README).
    # Include README text so `pip install transformers ...` lines mark those deps declared.
    for f in repo_path.glob("README*"):
        if f.is_file():
            try:
                blob.append(f.read_text(encoding="utf-8", errors="replace").lower())
            except Exception:
                pass
    # also any nested requirements*.txt / setup / pyproject (repo-wide, capped)
    nested = 0
    for f in repo_path.rglob("requirements*.txt"):
        if any(part in _SKIP_DIRS for part in f.parts):
            continue
        try:
            blob.append(f.read_text(encoding="utf-8", errors="replace").lower())
        except Exception:
            pass
        nested += 1
        if nested >= 40:
            break
    for name in ("setup.py", "pyproject.toml"):
        for f in repo_path.rglob(name):
            if any(part in _SKIP_DIRS for part in f.parts):
                continue
            try:
                blob.append(f.read_text(encoding="utf-8", errors="replace").lower())
            except Exception:
                pass
    return "\n".join(blob)


def undeclared_imports(repo_path: Path) -> list[str]:
    imports = _top_imports(repo_path)
    local = _local_modules(repo_path)
    declared = _declared_deps_blob(repo_path)
    has_any_reqs = bool(declared.strip())
    candidates: list[str] = []
    for name in sorted(imports):
        if name in _STDLIB or name in local or name.startswith("_"):
            continue
        norm = _IMPORT_ALIAS.get(name, name).lower().replace("_", "-")
        bare = name.lower().replace("_", "-")
        if declared and (norm in declared or bare in declared or name.lower() in declared):
            continue
        candidates.append(name)
    # cap noise
    out = candidates[:25]
    if not has_any_reqs and out:
        out = ["(NO requirements/setup file present) "] + out
    return out


def slug(full_name: str) -> str:
    return full_name.replace("/", "_")


def resolve_repo(full_name: str) -> Path:
    target = ROOT / "repos" / slug(full_name)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://github.com/{full_name}.git"
        subprocess.run(["git", "clone", "--depth", "50", url, str(target)],
                       check=True, capture_output=True, text=True)
    sha = KNOWN_SHAS.get(full_name)
    if sha:
        subprocess.run(["git", "checkout", sha], cwd=str(target),
                       check=False, capture_output=True, text=True)
    return target.resolve()


def parse_verdict(final_text: str, reason: str) -> dict:
    import re
    out = {"verdict": "INCONCLUSIVE", "confidence": "", "blockers": "",
           "required_changes": "", "rationale": f"(terminal={reason})"}

    def _field(label: str, text: str) -> str | None:
        # tolerate markdown bold/headers: **VERDICT:** x, ## VERDICT: x, - VERDICT: x
        m = re.search(rf"{label}\s*:?\s*\**\s*([^\n*]+)", text, flags=re.IGNORECASE)
        return m.group(1).strip(" *#-") if m else None

    txt = final_text or ""
    v = _field("VERDICT", txt)
    if v:
        vu = v.upper()
        if "REQUIRES_REPO2ROCM" in vu:
            out["verdict"] = "REQUIRES_REPO2ROCM"
        elif "OUT_OF_BOX" in vu:
            out["verdict"] = "OUT_OF_BOX"
    # fallback: token anywhere
    if out["verdict"] == "INCONCLUSIVE":
        if "REQUIRES_REPO2ROCM" in txt:
            out["verdict"] = "REQUIRES_REPO2ROCM"
        elif "OUT_OF_BOX" in txt:
            out["verdict"] = "OUT_OF_BOX"
    for key, label in [("confidence", "CONFIDENCE"), ("blockers", "BLOCKERS"),
                       ("required_changes", "REQUIRED_CHANGES"), ("rationale", "RATIONALE")]:
        val = _field(label, txt)
        if val:
            out[key] = val
    return out


async def analyze_one(full_name: str, boot, sem: asyncio.Semaphore, timeout_s: float) -> dict:
    async with sem:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"{slug(full_name)}.log"
        fh = open(log_path, "w", encoding="utf-8")
        console = Console(file=fh, soft_wrap=True, width=140)
        from repo2rocm.ui.event_printer import EventPrinter
        printer = EventPrinter(console=console, show_thinking=False)

        def on_event(ev):
            try:
                printer(ev)
            finally:
                fh.flush()

        started = time.time()
        rec = {"repo": full_name, "sha": KNOWN_SHAS.get(full_name, "HEAD")}
        console.rule(f"analyze {full_name}")
        try:
            repo_path = resolve_repo(full_name)
            console.print(f"repo_path={repo_path}")

            recon = run_recon(repo_path=repo_path, repo_full_name=full_name,
                              mode="functional", sha=KNOWN_SHAS.get(full_name, ""))
            recon_facts = recon.render_for_planner()
            rec["recon_cuda_deps"] = list(getattr(recon, "cuda_deps", []) or [])
            rec["recon_code_hazards"] = len(getattr(recon, "code_hazards", []) or [])

            syn = syntax_errors(repo_path)
            undecl = undeclared_imports(repo_path)
            rec["syntax_errors"] = syn[:10]
            rec["undeclared_candidates"] = undecl
            extra_facts = (
                "\n\n# Deterministic silent-bug scan\n"
                f"Syntax check: {('FAILED -> ' + '; '.join(syn[:10])) if syn else 'clean (all .py parsed)'}\n"
                f"Candidate undeclared imports (verify against requirements/config; "
                f"exclude local modules): {undecl if undecl else 'none detected'}"
            )

            client = boot.make_client()
            transcript_store = TranscriptStore(ROOT / "analysis_output" / slug(full_name))

            ctx = ToolUseContext(
                agent_id="analyzer-root",
                session_id=transcript_store.session_id,
                workdir=repo_path,
                abort_event=asyncio.Event(),
                permission_mode=PermissionMode.BYPASS,
                read_file_state=ReadFileState(),
                sandbox=None,
                transcript=transcript_store.main(),
                messages=[],
                options={
                    "client": client, "client_factory": boot.make_client,
                    "transcript_store": transcript_store,
                    "skill_catalog": boot.skill_catalog,
                    "run_mode": "functional", "repo_full_name": full_name,
                    "repo_path": str(repo_path), "repo_container_path": "/repo",
                    "recon_report": recon,
                },
                gate_state=boot.gate_state,
            )

            prompt = (
                f"Judge AMD/ROCm portability of `{full_name}`.\n\n"
                f"Deterministic recon facts (from a static scan):\n{recon_facts}\n"
                f"{extra_facts}\n\n"
                f"Now inspect the repo files (README, requirements/setup, source) with your "
                f"read-only tools. Confirm the silent-bug candidates above (undeclared imports, "
                f"version/API drift, syntax/code bugs) before deciding, then render the final "
                f"verdict block."
            )

            result = await asyncio.wait_for(
                run_agent(RunAgentParams(
                    agent_def=ANALYZER_AGENT, prompt=prompt, parent_ctx=ctx,
                    client=client, client_factory=boot.make_client,
                    transcript_store=transcript_store, skill_catalog=boot.skill_catalog,
                    on_event=on_event,
                )),
                timeout=timeout_s,
            )
            v = parse_verdict(result.final_text, result.terminal.reason)
            rec.update(v, turns=result.terminal.turns,
                       duration_s=round(result.duration_s, 1),
                       tokens=result.usage_total, terminal=result.terminal.reason)
        except asyncio.TimeoutError:
            rec.update(verdict="INCONCLUSIVE", rationale=f"timeout after {timeout_s}s",
                       duration_s=round(time.time() - started, 1))
        except Exception as exc:  # noqa: BLE001
            rec.update(verdict="ERROR", rationale=str(exc)[:300],
                       duration_s=round(time.time() - started, 1))
            console.print(f"[red]ERROR: {exc}")
        finally:
            console.print(f"VERDICT={rec.get('verdict')} :: {rec.get('rationale')}")
            fh.flush(); fh.close()

        with open(RESULTS_PATH, "a", encoding="utf-8") as rf:
            rf.write(json.dumps(rec) + "\n")
        print(f"[{rec.get('verdict')}] {full_name} :: {rec.get('blockers','')[:60]}", flush=True)
        return rec


async def main_async(repos: list[str], parallel: int, timeout_s: float) -> None:
    boot = bootstrap()
    sem = asyncio.Semaphore(parallel)
    results = await asyncio.gather(*(analyze_one(r, boot, sem, timeout_s) for r in repos))
    req = [r for r in results if r.get("verdict") == "REQUIRES_REPO2ROCM"]
    oob = [r for r in results if r.get("verdict") == "OUT_OF_BOX"]
    other = [r for r in results if r.get("verdict") not in ("REQUIRES_REPO2ROCM", "OUT_OF_BOX")]
    print("\n================ ANALYSIS SUMMARY ================", flush=True)
    print(f"REQUIRES_REPO2ROCM={len(req)}  OUT_OF_BOX={len(oob)}  OTHER={len(other)}", flush=True)
    for r in results:
        print(f"  {r.get('verdict'):20s} {r['repo']:45s} {r.get('blockers','')[:55]}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", type=str, default="")
    args = ap.parse_args()
    if args.only:
        repos = [x.strip() for x in args.only.split(",") if x.strip()]
    else:
        repos = REPOS[: args.limit] if args.limit else REPOS
    asyncio.run(main_async(repos, args.parallel, args.timeout))


if __name__ == "__main__":
    main()
