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


def _image_exists_locally(client: Any, image: str) -> bool:
    """Return True if the image is already present in the local Docker daemon.

    Big win for ROCm images: rocm/pytorch:latest can be 20+ GB. If it's already
    pulled, we should never re-pull.
    """
    try:
        client.images.get(image)
        return True
    except Exception:
        return False


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    elapsed_s: float
    command: str
    cwd: str = "/repo"
    env: dict[str, str] | None = None
    truncated: bool = False


@dataclass
class SandboxConfig:
    base_image: str
    """The ORIGINAL, portable base image the operator chose (e.g. `rocm/pytorch:latest`).
    NEVER mutated — read by the Dockerfile synthesizer so the produced Dockerfile is
    `docker build`-portable on any machine."""

    repo_host_path: Path
    repo_container_path: str = "/repo"
    rocm_mode: bool = False
    extra_run_args: list[str] = field(default_factory=list)
    name_prefix: str = "r2r2"
    network: str = "bridge"
    pull_image: bool = True

    # Memory ceiling so the host OOM killer doesn't reap our container as PID 1
    # when a model load blows up. 30g matches the original Repo2ROCm setting.
    mem_limit: str = "30g"
    shm_size: str = "8g"


class Sandbox:
    """Docker sandbox with commit DAG.

    Lifecycle:
        sb = Sandbox(cfg)
        await sb.start()
        result = await sb.exec("pip install numpy")
        sb.commit("after-numpy-install")
        ...
        await sb.stop()

    Invariant: `cfg.base_image` is IMMUTABLE — it's what the synthesizer
    bakes into `FROM <base_image>`. The currently-running image (which mutates
    when the agent calls DockerCommit then DockerRollback) is `self.current_image`.
    """

    def __init__(self, cfg: SandboxConfig):
        self.cfg = cfg
        # current_image starts at the base; rollback() mutates THIS, not cfg.base_image
        self.current_image: str = cfg.base_image
        self.container: Any = None
        self.client: Any = None
        self.commit_log = CommitLog()
        self.commands: list[ExecResult] = []
        self._lock = asyncio.Lock()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self, *, on_pull_progress: Any | None = None) -> None:
        """Start the sandbox container. Pulls the base image if not local.

        Args:
            on_pull_progress: optional callback `(status: str, detail: str)` invoked
                during docker pull so the CLI can show progress instead of going silent
                for 5-20 minutes on a 20GB ROCm image pull.
        """
        if not _DOCKER_AVAILABLE:
            raise RuntimeError("docker-py not installed; install repo2rocm[full]")
        with span("sandbox.start", image=self.current_image):
            self.client = docker_sdk.from_env(timeout=300)

            # Only pull when:
            #   * pull_image is enabled
            #   * the image isn't a content-addressed sha256 we created locally
            #   * the image has a tag (otherwise treat as a local-only build)
            #   * the image isn't already present locally (huge time saver)
            should_pull = (
                self.cfg.pull_image
                and not self.current_image.startswith("sha256:")
                and ":" in self.current_image
                and not _image_exists_locally(self.client, self.current_image)
            )
            if should_pull:
                self._stream_pull(self.current_image, on_pull_progress)

            name = f"{self.cfg.name_prefix}-{uuid.uuid4().hex[:8]}"
            volumes = {str(self.cfg.repo_host_path): {"bind": self.cfg.repo_container_path, "mode": "rw"}}
            kwargs = dict(
                name=name,
                command="sleep infinity",
                detach=True,
                tty=False,
                volumes=volumes,
                network=self.cfg.network,
                mem_limit=self.cfg.mem_limit,
                shm_size=self.cfg.shm_size,
            )
            if self.cfg.rocm_mode:
                kwargs["devices"] = ["/dev/kfd", "/dev/dri"]
                kwargs["group_add"] = ["video"]
                kwargs["security_opt"] = ["seccomp=unconfined"]
            self.container = self.client.containers.run(self.current_image, **kwargs)
            METRICS.sandbox_ops.labels(op="start", outcome="ok").inc()
            if "root" not in self.commit_log.nodes:
                self.commit_log.add(
                    id="root",
                    parent_id=None,
                    image=self.current_image,
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
                    cwd=cwd,
                    env=env,
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
                cwd=cwd,
                env=env,
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
        """Stop current container and re-launch from a prior commit's image.

        Mutates `self.current_image` (which `start()` reads), NOT `cfg.base_image`.
        The synthesizer always emits `FROM <cfg.base_image>` so the produced
        Dockerfile is portable.
        """
        target = to_commit or (
            self.commit_log.head and self.commit_log.nodes[self.commit_log.head].parent_id
        )
        if target is None or target not in self.commit_log.nodes:
            raise RuntimeError(f"no rollback target available (asked for {to_commit})")
        with span("sandbox.rollback", target=target):
            node = self.commit_log.nodes[target]
            await self.stop()
            self.current_image = node.image   # ← mutate current_image, not cfg.base_image
            await self.start()
            self.commit_log.set_head(target)
            METRICS.sandbox_ops.labels(op="rollback", outcome="ok").inc()

    def _stream_pull(self, image: str, on_progress: Any | None = None) -> None:
        """Pull `image` while emitting layer-level progress to the caller.

        Uses the low-level `APIClient.pull(stream=True, decode=True)` so we get
        Docker's native progress events. Coalesces by `id` (=layer) and forwards
        a compact `(status, "layer abc123 [====>...] 312/2048 MB")` to `on_progress`.

        Falls back to a blocking pull on any error so we never break the migration.
        """
        try:
            api = self.client.api  # low-level APIClient
        except Exception:
            try:
                self.client.images.pull(image)
            except Exception as exc:
                log.warning("pull failed; assuming local image", error=str(exc))
            return

        per_layer: dict[str, dict[str, Any]] = {}
        last_summary = ""
        try:
            for event in api.pull(image, stream=True, decode=True):
                status = event.get("status", "")
                lid = event.get("id") or ""
                detail = event.get("progressDetail") or {}
                if lid:
                    per_layer[lid] = {
                        "status": status,
                        "current": detail.get("current", 0),
                        "total": detail.get("total", 0),
                    }
                else:
                    # No layer id — image-level status line (e.g. "Status: ...").
                    if on_progress is not None:
                        try:
                            on_progress(status, event.get("progress", ""))
                        except Exception:
                            pass

                # Summarize across layers: how many are done / extracting / total bytes
                downloading = sum(
                    1 for v in per_layer.values() if v["status"] in ("Downloading", "Extracting")
                )
                done = sum(
                    1 for v in per_layer.values() if v["status"] in ("Pull complete", "Already exists", "Download complete")
                )
                total_bytes = sum(v["total"] for v in per_layer.values())
                current_bytes = sum(v["current"] for v in per_layer.values())
                summary = (
                    f"{done}/{len(per_layer)} layers done, "
                    f"{downloading} active, "
                    f"{current_bytes / 1e9:.2f}/{total_bytes / 1e9:.2f} GB"
                )
                if on_progress is not None and summary != last_summary:
                    try:
                        on_progress("pulling", summary)
                    except Exception:
                        pass
                    last_summary = summary
        except Exception as exc:
            log.warning("streaming pull failed; assuming local image", error=str(exc))

    def latest_commit(self) -> str | None:
        """Return the most recent commit_id that's safe to roll back to (skip 'root')."""
        head = self.commit_log.head
        if head and head != "root":
            return head
        # walk children to find any user-labeled commit
        for nid, node in self.commit_log.nodes.items():
            if nid != "root":
                return nid
        return None
