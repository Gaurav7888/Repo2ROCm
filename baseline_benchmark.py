#!/usr/bin/env python3
"""Out-of-box AMD/ROCm baseline benchmark.

For each repo: clone it, boot the STOCK `rocm/pytorch:latest` image, and run a
"no-fix" agent that follows the README verbatim (never installing torch, never
editing repo files) and emits OUT_OF_BOX_PASS / OUT_OF_BOX_FAIL.

This measures the baseline so we can compare against repo2rocm's migration.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
from pathlib import Path

from rich.console import Console

from repo2rocm.agents.definition import AgentDefinition
from repo2rocm.agents.lifecycle import RunAgentParams, run_agent
from repo2rocm.bootstrap import bootstrap
from repo2rocm.core.permissions import PermissionMode
from repo2rocm.core.terminal import Completed  # noqa: F401  (type hint clarity)
from repo2rocm.observability.transcripts import TranscriptStore
from repo2rocm.sandbox import Sandbox, SandboxConfig
from repo2rocm.tools.base import ReadFileState, ToolUseContext

ROOT = Path("/home/gsarkar/Repo2ROCm")
BASE_IMAGE = "rocm/pytorch:latest"
LOG_DIR = ROOT / "baseline_logs"
RESULTS_PATH = ROOT / "baseline_results.jsonl"

# Known SHAs (from batch_easylist_env.sh). Everything else => default-branch HEAD.
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

# The 60 verified repos.
REPOS = [
    "qiuzh20/gated_attention",
    "facebookresearch/luckmatters",
    "backprop07/Self-Certainty",
    "THU-MIG/PrefixKV",
    "microsoft/SeqSNN",
    "showmeon/TimeEmb",
    "llmsresearch/paperbanana",
    "zhoujiahuan1991/NeurIPS2025-KFF",
    "nanomaoli/llm_reproducibility",
    "parameterlab/c-seo-bench",
    "alessiopittiglio/mm-argfallacy",
    "dayeonki/askqe",
    "sapienzanlp/bookcoref",
    "flairNLP/VerbalizED",
    "dxlong2000/NLPromptEval",
    "tanghme0w/ACL25-CoPE",
    "batmanlab/Ladder",
    "dmis-lab/TemporalHead",
    "UAlberta-NLP/SemEval2025-EAMT",
    "XiaoAI1989/ORMind",
    "SCUNLP/ELABORATION",
    "decisionintelligence/CATCH",
    "kaistAI/Knowledge-Entropy",
    "LehengTHU/AlphaRec",
    "OpenBMB/IoA",
    "Graph-COM/LayerDAG",
    "murtylab/topoloss",
    "neuro-galaxy/torch_brain",
    "shengliu66/VTI",
    "pouyahmdn/LCPO",
    "codezakh/DataEnvGym",
    "ModelTC/HarmoniCa",
    "xihongyang1999/AIRMVC",
    "DCDmllm/HealthGPT",
    "apple/ml-tarflow",
    "xinyuluo8561/Stacey",
    "GuanchengWan/EARTH",
    "Kambm/convolutional_diffusion",
    "mmoharami/Risk-Sensitive-CMDP",
    "YZY-stack/Effort-AIGI-Detection",
    "BruceGeLi/TOP_ERL_ICLR25",
    "liukidar/pcx",
    "yjucho1/CoMRes",
    "mint-vu/MCNC",
    "1202kbs/GCTM",
    "DeepakTatachar/CSL-Mem",
    "allenai/WildBench",
    "xlang-ai/BRIGHT",
    "THUKElab/CLEME",
    "FFY0/AdaKV",
    "FYYFU/HeadKV",
    "allenai/understanding_mcqa",
    "Cohere-Labs-Community/m-rewardbench",
    "sarahmart/HARDMath",
    "ZhiningLiu1998/SelfElicit",
    "thunlp/FR-Spec",
    "ryienh/jl-metric",
    "shimo-lab/modelmap",
    "togethercomputer/moa",
    "SongW-SW/CEB",
]


SYSTEM_PROMPT = """You are an OUT-OF-THE-BOX AMD/ROCm compatibility tester.

You are inside a Docker container booted from the STOCK image `rocm/pytorch:latest`
(ROCm PyTorch is already installed and working). The target repository is mounted at
`/repo`. You are measuring a BASELINE: does this repo run on AMD GPUs by following its
README *as written*, with NO porting work? You must therefore NOT fix anything.

TOOL: You only have DockerExec (runs `bash -lc <command>` inside the container).

HARD RULES:
  1. NEVER (re)install torch / torchvision / torchaudio or any `nvidia-*-cu*` wheels.
     The container already has the correct ROCm torch; reinstalling would break it.
     If a requirements file or the README pins these, install everything ELSE but
     skip those packages, e.g.:
       grep -viE '^(torch|torchvision|torchaudio|nvidia-|cuda-)' requirements.txt > /tmp/req.txt
       pip install -r /tmp/req.txt
     Writing files under /tmp is allowed. Editing files under /repo is NOT.
  2. Do NOT edit / patch / `sed -i` / rewrite ANY file under /repo to make things work.
     No code changes, no config changes. (The program writing its own outputs, logs,
     or checkpoints during a normal run is fine.)
  3. Follow the README's documented install + run steps in order. Use the smallest /
     quickest runnable example the README provides (quickstart / demo / test / a single
     small command) to prove it runs on GPU. Cap any single run at timeout_s=600.
  4. If a step needs a dataset / model / API key and the README documents how to get it,
     do so. If it is gated or unavailable and the repo cannot run at all without it, and
     no smaller documented example runs, that is a FAIL.
  5. Confirm real GPU execution: `torch.cuda.is_available()` is True and the documented
     code actually runs without ROCm/CUDA errors.

VERDICT — your FINAL message must be exactly one of:
  OUT_OF_BOX_PASS: <one line — the README command that ran on AMD GPU>
  OUT_OF_BOX_FAIL: <one line — the first blocking error / why it needs changes>

DISCIPLINE: one command per turn, brief narration, stop as soon as the verdict is
justified. Do not loop or retry endlessly — if it clearly needs repo edits to proceed,
emit OUT_OF_BOX_FAIL.
"""


BASELINE_AGENT = AgentDefinition(
    name="baseline_tester",
    description="Out-of-box AMD/ROCm baseline tester (no edits, no torch install).",
    allowed_tools=["DockerExec"],
    permission_mode=PermissionMode.BYPASS,
    max_turns=40,
    max_tokens=8_192,
    system_prompt_template=SYSTEM_PROMPT,
    color="magenta",
)


def slug(full_name: str) -> str:
    return full_name.replace("/", "_")


def resolve_repo(full_name: str) -> Path:
    target = ROOT / "repos" / slug(full_name)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://github.com/{full_name}.git"
        subprocess.run(
            ["git", "clone", "--depth", "50", url, str(target)],
            check=True, capture_output=True, text=True,
        )
    sha = KNOWN_SHAS.get(full_name)
    if sha:
        subprocess.run(["git", "checkout", sha], cwd=str(target),
                       check=False, capture_output=True, text=True)
    return target.resolve()


def parse_verdict(final_text: str, reason: str) -> tuple[str, str]:
    txt = final_text or ""
    for line in txt.splitlines():
        s = line.strip()
        if s.startswith("OUT_OF_BOX_PASS"):
            return "PASS", s[len("OUT_OF_BOX_PASS"):].lstrip(": ").strip()
        if s.startswith("OUT_OF_BOX_FAIL"):
            return "FAIL", s[len("OUT_OF_BOX_FAIL"):].lstrip(": ").strip()
    if "OUT_OF_BOX_PASS" in txt:
        return "PASS", "(marker found in body)"
    if "OUT_OF_BOX_FAIL" in txt:
        return "FAIL", "(marker found in body)"
    return "INCONCLUSIVE", f"no verdict (terminal={reason})"


async def run_one(full_name: str, boot, sem: asyncio.Semaphore, timeout_s: float) -> dict:
    async with sem:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"{slug(full_name)}.log"
        fh = open(log_path, "w", encoding="utf-8")
        console = Console(file=fh, soft_wrap=True, width=140)
        from repo2rocm.ui.event_printer import EventPrinter
        printer = EventPrinter(console=console, show_thinking=False)

        started = time.time()
        rec = {"repo": full_name, "sha": KNOWN_SHAS.get(full_name, "HEAD")}
        console.rule(f"baseline {full_name}")
        sandbox = None
        try:
            repo_path = resolve_repo(full_name)
            console.print(f"repo_path={repo_path}")

            client = boot.make_client()
            transcript_store = TranscriptStore(ROOT / "baseline_output" / slug(full_name))

            sandbox = Sandbox(SandboxConfig(
                base_image=BASE_IMAGE,
                repo_host_path=repo_path,
                repo_container_path="/repo",
                rocm_mode=True,
                pull_image=True,
            ))
            await sandbox.start()
            console.print(f"sandbox up: {sandbox.container.name}")

            ctx = ToolUseContext(
                agent_id="baseline-root",
                session_id=transcript_store.session_id,
                workdir=repo_path,
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
                    "run_mode": "functional",
                    "repo_full_name": full_name,
                    "repo_path": str(repo_path),
                    "repo_container_path": "/repo",
                },
                gate_state=boot.gate_state,
            )

            prompt = (
                f"Test whether `{full_name}` runs out of the box on AMD ROCm GPUs by "
                f"following its README. The repo is mounted at /repo. Read the README "
                f"first (e.g. `cat /repo/README*`), then follow its install + run steps "
                f"under the HARD RULES. Emit OUT_OF_BOX_PASS or OUT_OF_BOX_FAIL when done."
            )

            result = await asyncio.wait_for(
                run_agent(RunAgentParams(
                    agent_def=BASELINE_AGENT,
                    prompt=prompt,
                    parent_ctx=ctx,
                    client=client,
                    client_factory=boot.make_client,
                    transcript_store=transcript_store,
                    skill_catalog=boot.skill_catalog,
                    on_event=printer,
                )),
                timeout=timeout_s,
            )
            verdict, reason = parse_verdict(result.final_text, result.terminal.reason)
            rec.update(
                verdict=verdict, reason=reason, turns=result.terminal.turns,
                duration_s=round(result.duration_s, 1), tokens=result.usage_total,
                terminal=result.terminal.reason, final_text=(result.final_text or "")[:800],
            )
        except asyncio.TimeoutError:
            rec.update(verdict="INCONCLUSIVE", reason=f"timeout after {timeout_s}s",
                       duration_s=round(time.time() - started, 1))
            console.print(f"[red]TIMEOUT after {timeout_s}s")
        except Exception as exc:  # noqa: BLE001
            rec.update(verdict="ERROR", reason=str(exc)[:300],
                       duration_s=round(time.time() - started, 1))
            console.print(f"[red]ERROR: {exc}")
        finally:
            if sandbox is not None:
                try:
                    await sandbox.stop()
                except Exception:
                    pass
            console.print(f"VERDICT={rec.get('verdict')} :: {rec.get('reason')}")
            fh.flush()
            fh.close()

        with open(RESULTS_PATH, "a", encoding="utf-8") as rf:
            rf.write(json.dumps(rec) + "\n")
        print(f"[{rec.get('verdict')}] {full_name} :: {rec.get('reason')}", flush=True)
        return rec


async def main_async(repos: list[str], parallel: int, timeout_s: float) -> None:
    boot = bootstrap()
    sem = asyncio.Semaphore(parallel)
    results = await asyncio.gather(*(run_one(r, boot, sem, timeout_s) for r in repos))
    passes = sum(1 for r in results if r.get("verdict") == "PASS")
    fails = sum(1 for r in results if r.get("verdict") == "FAIL")
    other = len(results) - passes - fails
    print("\n================ BASELINE SUMMARY ================", flush=True)
    print(f"PASS={passes}  FAIL={fails}  OTHER(inconclusive/error/timeout)={other}", flush=True)
    for r in results:
        print(f"  {r.get('verdict'):12s} {r['repo']:45s} {r.get('reason','')[:70]}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=1500.0, help="per-repo wallclock cap (s)")
    ap.add_argument("--limit", type=int, default=0, help="only first N repos (0=all)")
    ap.add_argument("--only", type=str, default="", help="comma-separated owner/repo to run")
    args = ap.parse_args()

    if args.only:
        repos = [x.strip() for x in args.only.split(",") if x.strip()]
    else:
        repos = REPOS[: args.limit] if args.limit else REPOS

    main_async_repos = repos
    asyncio.run(main_async(main_async_repos, args.parallel, args.timeout))


if __name__ == "__main__":
    main()
