"""Sandbox: thin wrapper around a docker SDK container.

Uses `docker exec` per command instead of a long-lived `pexpect` shell.

Why: pexpect is single-process; output framing breaks on color/cursor escapes;
and the commit/rollback stack prevents concurrent migrators. With docker exec,
N migrators can run in parallel against branched commits.

The full implementation requires `docker-py`. We gracefully degrade if docker is
not available so unit tests can still construct the object.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from repo2rocm.observability.logging import get_logger
from repo2rocm.observability.metrics import METRICS
from repo2rocm.observability.tracing import span
from repo2rocm.sandbox.commit_log import CommitLog

log = get_logger(__name__)


try:
    import docker as docker_sdk

    _DOCKER_AVAILABLE = True
except Exception:  # pragma: no cover
    _DOCKER_AVAILABLE = False
    docker_sdk = None  # type: ignore[assignment]


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    elapsed_s: float
    command: str
    truncated: bool = False


@dataclass
class SandboxConfig:
    base_image: str
    repo_host_path: Path
    repo_container_path: str = "/repo"
    rocm_mode: bool = False
    extra_run_args: list[str] = field(default_factory=list)
    name_prefix: str = "r2r2"
    network: str = "bridge"
    pull_image: bool = True


class Sandbox:
    """Docker sandbox with commit DAG.

    Lifecycle:
        sb = Sandbox(cfg)
        await sb.start()
        result = await sb.exec("pip install numpy")
        sb.commit("after-numpy-install")
        ...
        await sb.stop()
    """

    def __init__(self, cfg: SandboxConfig):
        self.cfg = cfg
        self.container: Any = None
        self.client: Any = None
        self.commit_log = CommitLog()
        self.commands: list[ExecResult] = []
        self._lock = asyncio.Lock()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if not _DOCKER_AVAILABLE:
            raise RuntimeError("docker-py not installed; install repo2rocm[full]")
        with span("sandbox.start", image=self.cfg.base_image):
            self.client = docker_sdk.from_env(timeout=300)
            if self.cfg.pull_image:
                try:
                    self.client.images.pull(self.cfg.base_image)
                except Exception as exc:
                    log.warning("pull failed; assuming local image", error=str(exc))

            name = f"{self.cfg.name_prefix}-{uuid.uuid4().hex[:8]}"
            volumes = {str(self.cfg.repo_host_path): {"bind": self.cfg.repo_container_path, "mode": "rw"}}
            device_requests = []
            if self.cfg.rocm_mode:
                # ROCm: pass /dev/kfd and /dev/dri
                device_requests = []
            self.container = self.client.containers.run(
                self.cfg.base_image,
                name=name,
                command="sleep infinity",
                detach=True,
                tty=False,
                volumes=volumes,
                network=self.cfg.network,
                devices=(["/dev/kfd", "/dev/dri"] if self.cfg.rocm_mode else None),
                group_add=(["video"] if self.cfg.rocm_mode else None),
                security_opt=(["seccomp=unconfined"] if self.cfg.rocm_mode else None),
            )
            METRICS.sandbox_ops.labels(op="start", outcome="ok").inc()
            self.commit_log.add(
                id="root",
                parent_id=None,
                image=self.cfg.base_image,
                label="initial",
            )

    async def stop(self) -> None:
        if self.container is None:
            return
        with span("sandbox.stop"):
            try:
                self.container.stop(timeout=5)
                self.container.remove(force=True)
                METRICS.sandbox_ops.labels(op="stop", outcome="ok").inc()
            except Exception as exc:
                log.warning("stop failed", error=str(exc))
                METRICS.sandbox_ops.labels(op="stop", outcome="error").inc()
            self.container = None

    # ── Exec ──────────────────────────────────────────────────────────────────

    async def exec(
        self,
        command: str,
        *,
        timeout_s: float = 1800.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        max_output_bytes: int = 2_000_000,
    ) -> ExecResult:
        if self.container is None:
            raise RuntimeError("sandbox not started")
        cwd = cwd or self.cfg.repo_container_path
        with span(
            "sandbox.exec", command_preview=command[:120], cwd=cwd, timeout_s=timeout_s
        ):
            start = time.perf_counter()
            try:
                exec_id = self.client.api.exec_create(
                    self.container.id,
                    cmd=["bash", "-lc", command],
                    workdir=cwd,
                    environment=env or {},
                    stdout=True,
                    stderr=True,
                    tty=False,
                )
                exec_id = exec_id.get("Id") if isinstance(exec_id, dict) else exec_id
                stream = self.client.api.exec_start(exec_id, stream=False, demux=True)
                out_bytes, err_bytes = stream
                stdout = (out_bytes or b"").decode("utf-8", errors="replace")
                stderr = (err_bytes or b"").decode("utf-8", errors="replace")
                truncated = False
                if len(stdout) > max_output_bytes:
                    stdout = stdout[:max_output_bytes] + "\n... [truncated]"
                    truncated = True
                inspect = self.client.api.exec_inspect(exec_id)
                exit_code = int(inspect.get("ExitCode") or 0)
            except Exception as exc:  # noqa: BLE001
                METRICS.sandbox_ops.labels(op="exec", outcome="error").inc()
                return ExecResult(
                    exit_code=-1,
                    stdout="",
                    stderr=f"docker exec failed: {exc}",
                    elapsed_s=time.perf_counter() - start,
                    command=command,
                )
            elapsed = time.perf_counter() - start
            outcome = "ok" if exit_code == 0 else "nonzero"
            METRICS.sandbox_ops.labels(op="exec", outcome=outcome).inc()
            res = ExecResult(
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                elapsed_s=elapsed,
                command=command,
                truncated=truncated,
            )
            self.commands.append(res)
            return res

    # ── Commit / rollback ─────────────────────────────────────────────────────

    def commit(self, label: str = "") -> str:
        if self.container is None:
            raise RuntimeError("sandbox not started")
        with span("sandbox.commit", label=label):
            img = self.container.commit(repository="repo2rocm-checkpoint")
            commit_id = uuid.uuid4().hex[:12]
            parent = self.commit_log.head
            self.commit_log.add(
                id=commit_id,
                parent_id=parent,
                image=img.id,
                label=label,
            )
            METRICS.sandbox_ops.labels(op="commit", outcome="ok").inc()
            return commit_id

    async def rollback(self, *, to_commit: str | None = None) -> None:
        """Stop current container and re-launch from a prior commit's image."""
        target = to_commit or (self.commit_log.head and self.commit_log.nodes[self.commit_log.head].parent_id)
        if target is None or target not in self.commit_log.nodes:
            raise RuntimeError(f"no rollback target available (asked for {to_commit})")
        with span("sandbox.rollback", target=target):
            node = self.commit_log.nodes[target]
            await self.stop()
            self.cfg.base_image = node.image
            await self.start()
            self.commit_log.set_head(target)
            METRICS.sandbox_ops.labels(op="rollback", outcome="ok").inc()
