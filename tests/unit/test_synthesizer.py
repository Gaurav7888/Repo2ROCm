"""Dockerfile synthesizer — produce a reproducible Dockerfile from sandbox.commands.

We can't run real Docker in unit tests; we fake the Sandbox + ExecResult shape.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from repo2rocm.dockerfile.synthesizer import (
    DockerfileSynthesis,
    synthesize_dockerfile,
    write_dockerfile,
)
from repo2rocm.sandbox.manager import ExecResult


# ── Minimal stand-in for the real Sandbox so we don't need docker-py ───────


@dataclass
class _FakeSandboxCfg:
    base_image: str = "rocm/pytorch:latest"
    repo_container_path: str = "/repo"


@dataclass
class _FakeSandbox:
    cfg: _FakeSandboxCfg = field(default_factory=_FakeSandboxCfg)
    commands: list[ExecResult] = field(default_factory=list)


def _exec(cmd: str, exit_code: int = 0, cwd: str = "/repo") -> ExecResult:
    return ExecResult(
        exit_code=exit_code,
        stdout="",
        stderr="",
        elapsed_s=0.01,
        command=cmd,
        cwd=cwd,
    )


# ── Tests ────────────────────────────────────────────────────────────────────


def test_synth_emits_from_and_pre_install_block(tmp_path: Path):
    sb = _FakeSandbox()
    sb.commands = [_exec("pip install numpy")]
    synth = synthesize_dockerfile(sb)
    text = synth.dockerfile_text
    assert text.startswith("FROM rocm/pytorch:latest")
    # pre-install block: apt + curl + git + poetry + pytest
    assert "apt-get update" in text
    assert "pip install pytest" in text
    assert "WORKDIR /" in text


def test_synth_includes_git_clone_and_checkout_when_repo_full_name_given():
    sb = _FakeSandbox()
    sb.commands = [_exec("pip install numpy")]
    synth = synthesize_dockerfile(
        sb,
        repo_full_name="tonbistudio/turboquant-pytorch",
        sha="abc123",
    )
    text = synth.dockerfile_text
    assert "git clone https://github.com/tonbistudio/turboquant-pytorch.git" in text
    assert "git checkout abc123" in text
    assert "cp -r /turboquant-pytorch/. /repo" in text


def test_synth_skips_inspection_and_marker_commands():
    """Read-only inspection commands + the ROCM_ENV_VERIFIED echo should NOT
    end up baked into the Dockerfile (they're noise)."""
    sb = _FakeSandbox()
    sb.commands = [
        _exec("ls -la"),
        _exec("cat README.md"),
        _exec("pip install torch"),
        _exec("python -c 'import torch; print(torch.cuda.is_available())'"),
        _exec("rocm-smi"),
        _exec("echo ROCM_ENV_VERIFIED"),
    ]
    synth = synthesize_dockerfile(sb)
    text = synth.dockerfile_text
    assert "RUN pip install torch" in text
    assert "RUN ls -la" not in text
    assert "RUN cat README.md" not in text
    assert "RUN echo ROCM_ENV_VERIFIED" not in text
    assert "rocm-smi" not in text or text.count("rocm-smi") <= 0
    # the torch.cuda check is also a probe → skipped
    assert "RUN python -c" not in text or "torch.cuda.is_available" not in text.split("RUN")[-1]


def test_synth_emits_cd_prefix_for_subdir_commands():
    sb = _FakeSandbox()
    sb.commands = [
        _exec("pip install -r requirements.txt"),               # cwd=/repo → no cd
        _exec("python setup.py install", cwd="/repo/sub_pkg"),  # different cwd → cd prefix
    ]
    text = synthesize_dockerfile(sb).dockerfile_text
    assert "RUN pip install -r requirements.txt" in text
    assert "RUN cd /repo/sub_pkg && python setup.py install" in text


def test_synth_drops_failed_commands():
    sb = _FakeSandbox()
    sb.commands = [
        _exec("pip install flash-attn", exit_code=1),  # CUDA-only wheel, fails
        _exec("pip install scipy"),                    # this succeeds
    ]
    text = synthesize_dockerfile(sb).dockerfile_text
    assert "RUN pip install scipy" in text
    assert "flash-attn" not in text


def test_synth_dedupes_normalized_run_lines():
    import re

    sb = _FakeSandbox()
    sb.commands = [
        _exec("pip install numpy"),
        _exec("pip install   numpy"),       # whitespace-different but same
        _exec("pip install scipy"),
    ]
    text = synthesize_dockerfile(sb).dockerfile_text
    # only ONE numpy install should survive (the last one wins); whitespace-normalize
    numpy_lines = [
        re.sub(r"\s+", " ", ln) for ln in text.splitlines() if "numpy" in ln and ln.startswith("RUN")
    ]
    assert numpy_lines == ["RUN pip install numpy"], numpy_lines
    assert "pip install scipy" in text


def test_synth_captures_git_diff_as_patch_file(tmp_path: Path):
    """If the host clone has uncommitted edits, extract them as a .diff and COPY
    them into the Dockerfile + git apply at build time."""
    import subprocess

    # Set up a tiny git repo with one modified file
    repo = tmp_path / "fake_repo"
    repo.mkdir()
    (repo / "main.py").write_text("device = 'cuda'\n")
    subprocess.run(["git", "init", "-q"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=str(repo), check=True)
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=str(repo), check=True)
    # Now edit the file (simulating what the agent did with Edit)
    (repo / "main.py").write_text("device = 'cuda' if torch.cuda.is_available() else 'cpu'\n")

    sb = _FakeSandbox()
    sb.commands = [_exec("pip install torch")]
    patches_dir = tmp_path / "out" / "patches"
    synth = synthesize_dockerfile(
        sb,
        repo_full_name="t/r",
        sha="HEAD",
        repo_host_path=repo,
        patches_dir=patches_dir,
    )
    assert len(synth.patches) == 1
    assert synth.patches[0].read_text().startswith("diff --git")
    assert "COPY agent_edits.diff /tmp/agent_edits.diff" in synth.dockerfile_text
    assert "git apply --reject /tmp/agent_edits.diff" in synth.dockerfile_text


def test_synth_skips_envverify_heredoc_probe():
    """Bug 5: the EnvVerify python heredoc is a probe — must NEVER be baked.
    Reproduces the exact heredoc shape we saw in the live run."""
    sb = _FakeSandbox()
    envverify_script = """
set -e
python - <<'PY'
import torch, sys, json
ok = torch.cuda.is_available()
info = {"torch_version": torch.__version__, "cuda_available": ok}
print("ENV_VERIFY_JSON:" + json.dumps(info))
sys.exit(0 if ok else 1)
PY
"""
    sb.commands = [
        _exec(envverify_script),
        _exec("pip install scipy"),
    ]
    text = synthesize_dockerfile(sb).dockerfile_text
    assert "ENV_VERIFY_JSON" not in text
    assert "torch.cuda.is_available" not in text
    assert "RUN pip install scipy" in text


def test_synth_skips_short_probe_heredocs_ending_in_sysexit():
    """Generic single-shot probe heredocs (short body + sys.exit) are diagnostics, not build steps."""
    sb = _FakeSandbox()
    probe = """
python - <<'PY'
import sys
print("alive")
sys.exit(0)
PY
"""
    sb.commands = [_exec(probe), _exec("pip install numpy")]
    text = synthesize_dockerfile(sb).dockerfile_text
    assert "alive" not in text
    assert "RUN pip install numpy" in text


def test_synth_keeps_real_multiline_install_scripts():
    """Conversely: a real multi-line install script (no sys.exit probe pattern)
    that uses a heredoc should still be baked in."""
    sb = _FakeSandbox()
    install_script = """
git clone https://github.com/Dao-AILab/flash-attention /tmp/fa
cd /tmp/fa
git checkout v2.6.3
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
python setup.py install
"""
    sb.commands = [_exec(install_script)]
    text = synthesize_dockerfile(sb).dockerfile_text
    assert "flash-attention" in text
    assert "FLASH_ATTENTION_TRITON_AMD_ENABLE" in text


def test_synth_skips_bare_set_e_lines():
    sb = _FakeSandbox()
    sb.commands = [_exec("set -e"), _exec("set -ex"), _exec("pip install torch")]
    text = synthesize_dockerfile(sb).dockerfile_text
    assert "RUN set -e" not in text
    assert "RUN set -ex" not in text
    assert "RUN pip install torch" in text


def test_write_dockerfile_also_copies_patches_next_to_dockerfile(tmp_path: Path):
    """`docker build .` needs the .diff file in its build context."""
    sb = _FakeSandbox()
    sb.commands = [_exec("pip install torch")]
    patches_dir = tmp_path / "scratch"
    patches_dir.mkdir()
    fake_patch = patches_dir / "agent_edits.diff"
    fake_patch.write_text("diff --git a/x b/x\n")
    synth = DockerfileSynthesis(
        dockerfile_text="FROM scratch\nCOPY agent_edits.diff /x\n",
        successful_commands=[],
        base_image="scratch",
        patches=[fake_patch],
    )
    out_dir = tmp_path / "out"
    write_dockerfile(synth, out_dir / "Dockerfile")
    assert (out_dir / "Dockerfile").exists()
    assert (out_dir / "agent_edits.diff").exists()
