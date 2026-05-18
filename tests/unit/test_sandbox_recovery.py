"""Bug-fix regression tests for sandbox lifecycle + DockerExec recovery."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from repo2rocm.core.permissions import PermissionMode
from repo2rocm.sandbox.commit_log import CommitLog
from repo2rocm.tools.base import ReadFileState, ToolUseContext
from repo2rocm.tools.docker.exec import (
    DockerExec,
    DockerExecInput,
    _diagnose_crash_exit,
    _maybe_restart_dead_container,
)


# ── A minimal fake Sandbox for testing without docker-py ───────────────────────


@dataclass
class _FakeExec:
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    elapsed_s: float = 0.0
    command: str = ""
    cwd: str = "/repo"
    env: dict | None = None
    truncated: bool = False


@dataclass
class _FakeCfg:
    base_image: str = "rocm/pytorch:latest"
    repo_container_path: str = "/repo"


@dataclass
class _FakeSandbox:
    cfg: _FakeCfg = field(default_factory=_FakeCfg)
    current_image: str = "rocm/pytorch:latest"
    container: object | None = "alive"  # truthy → exists
    commit_log: CommitLog = field(default_factory=CommitLog)
    commands: list = field(default_factory=list)

    # what the test wants us to return on next exec
    next_exit_code: int = 0
    next_stdout: str = ""
    next_stderr: str = ""

    # callbacks the tests inspect
    stop_calls: int = 0
    start_calls: int = 0
    rollback_calls: list = field(default_factory=list)

    async def exec(self, command, *, timeout_s=600, cwd=None, env=None):
        if self.container is None:
            raise RuntimeError("sandbox not started")
        res = _FakeExec(
            exit_code=self.next_exit_code,
            stdout=self.next_stdout,
            stderr=self.next_stderr,
            command=command,
            cwd=cwd or "/repo",
            env=env,
        )
        self.commands.append(res)
        return res

    async def stop(self):
        self.stop_calls += 1
        self.container = None

    async def start(self):
        self.start_calls += 1
        self.container = "alive"

    async def rollback(self, *, to_commit=None):
        self.rollback_calls.append(to_commit)
        await self.stop()
        # apply rollback: change current_image to the commit's image (NOT cfg.base_image)
        node = self.commit_log.nodes.get(to_commit)
        if node:
            self.current_image = node.image
        await self.start()
        self.commit_log.set_head(to_commit)

    def latest_commit(self):
        head = self.commit_log.head
        return head if head and head != "root" else None


def _ctx(sb: _FakeSandbox, tmp: Path) -> ToolUseContext:
    return ToolUseContext(
        agent_id="t",
        session_id="s",
        workdir=tmp,
        abort_event=asyncio.Event(),
        permission_mode=PermissionMode.BYPASS,
        read_file_state=ReadFileState(),
        sandbox=sb,
    )


# ── Bug 4: synthesizer reads sandbox.cfg.base_image — must stay immutable ────


def test_rollback_does_not_mutate_cfg_base_image():
    """The Dockerfile synthesizer bakes `FROM <cfg.base_image>`. Even after the
    agent does DockerCommit + DockerRollback many times, cfg.base_image MUST
    still be the operator's original rocm/pytorch:latest, not a sha256:... id."""
    from repo2rocm.sandbox.commit_log import CommitLog

    sb = _FakeSandbox()
    sb.commit_log.add(id="root", parent_id=None, image="rocm/pytorch:latest", label="initial")
    sb.commit_log.add(id="c1", parent_id="root", image="sha256:abc123", label="after-deps")
    sb.commit_log.add(id="c2", parent_id="c1", image="sha256:def456", label="after-edits")

    original = sb.cfg.base_image
    asyncio.run(sb.rollback(to_commit="c1"))

    # current_image moved
    assert sb.current_image == "sha256:abc123"
    # but cfg.base_image is STILL the portable original
    assert sb.cfg.base_image == original == "rocm/pytorch:latest"


# ── Bug 3: OOM (exit 137) diagnosis ──────────────────────────────────────────


def test_diagnose_crash_recognizes_oom_exit_137():
    hint = _diagnose_crash_exit(137, "Killed")
    assert "OOM" in hint or "out of memory" in hint.lower()
    assert "DockerRollback" in hint  # tells the model what to do next


def test_diagnose_crash_recognizes_segfault_exit_139():
    hint = _diagnose_crash_exit(139, "Segmentation fault")
    assert "SIGSEGV" in hint
    assert "bitsandbytes" in hint or "native" in hint.lower()


def test_diagnose_crash_recognizes_oom_in_stderr_text():
    hint = _diagnose_crash_exit(1, "Traceback ... torch.cuda.OutOfMemoryError: out of memory")
    assert hint  # some hint produced
    assert "memory" in hint.lower()


def test_diagnose_crash_silent_on_success():
    assert _diagnose_crash_exit(0, "") == ""


@pytest.mark.asyncio
async def test_dockerexec_appends_oom_hint_to_result_on_137(tmp_path):
    """When the container's process exits 137, the model-facing text MUST include
    the OOM diagnosis so the model doesn't blindly retry."""
    sb = _FakeSandbox()
    sb.next_exit_code = 137
    sb.next_stderr = "Killed"
    ctx = _ctx(sb, tmp_path)
    tool = DockerExec()
    result = await tool.call(DockerExecInput(command="python -m huge_test"), ctx)
    assert result.is_error
    assert "137" in result.text
    assert "OOM" in result.text or "out of memory" in result.text.lower()


# ── Bug 2: auto-restart when container is dead ───────────────────────────────


@pytest.mark.asyncio
async def test_auto_restart_rolls_back_to_latest_commit_when_container_is_none(tmp_path):
    """If the container died (sandbox.container is None) and there's a known-good
    commit, DockerExec must transparently roll back and surface the recovery."""
    sb = _FakeSandbox()
    sb.commit_log.add(id="root", parent_id=None, image="rocm/pytorch:latest", label="initial")
    sb.commit_log.add(id="c1", parent_id="root", image="sha256:abc", label="after-deps")
    sb.commit_log.set_head("c1")
    sb.container = None  # dead
    sb.next_exit_code = 0
    sb.next_stdout = "ok"

    note = await _maybe_restart_dead_container(sb)
    assert "restored to commit" in note
    assert "after-deps" in note or "c1" in note
    assert sb.container is not None
    assert sb.rollback_calls == ["c1"]


@pytest.mark.asyncio
async def test_auto_restart_cold_starts_when_no_commits_yet(tmp_path):
    """If there are no commits to roll back to, do a cold restart from the base image."""
    sb = _FakeSandbox()
    sb.commit_log.add(id="root", parent_id=None, image="rocm/pytorch:latest", label="initial")
    sb.container = None
    note = await _maybe_restart_dead_container(sb)
    assert "original base image" in note
    assert sb.container is not None
    assert sb.start_calls == 1


@pytest.mark.asyncio
async def test_dockerexec_surfaces_restart_note_in_result(tmp_path):
    """End-to-end: dead container → DockerExec call → result text contains the
    restart note AND the actual command output."""
    sb = _FakeSandbox()
    sb.commit_log.add(id="root", parent_id=None, image="rocm/pytorch:latest", label="initial")
    sb.commit_log.add(id="c1", parent_id="root", image="sha256:abc", label="after-deps")
    sb.commit_log.set_head("c1")
    sb.container = None
    sb.next_exit_code = 0
    sb.next_stdout = "Hello"
    ctx = _ctx(sb, tmp_path)
    tool = DockerExec()
    result = await tool.call(DockerExecInput(command="echo Hello"), ctx)
    assert not result.is_error
    assert "restored to commit" in result.text
    assert "Hello" in result.text
