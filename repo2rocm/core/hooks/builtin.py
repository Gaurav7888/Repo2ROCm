"""Built-in callback hooks that encode Repo2ROCm-specific quality gates.

These replace the ad-hoc booleans (`_dockerhub_tags_seen`, `_pypi_versions_seen`,
`_gpu_check_seen`) that were inlined in the old `configuration.py`.

Each gate is a `PreToolUse` callback that inspects the current per-session state and
returns `{"permissionBehavior": "deny", "reason": "..."}` when a precondition fails.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from repo2rocm.core.hooks.snapshot import HooksSnapshot


@dataclass
class GateState:
    """Per-session knowledge of what the agent has already done."""

    dockerhub_tags_seen: set[str] = field(default_factory=set)
    pypi_versions_seen: set[str] = field(default_factory=set)
    gpu_check_seen: bool = False
    env_verified: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def mark_dockerhub(self, repo: str) -> None:
        with self._lock:
            self.dockerhub_tags_seen.add(repo.lower().split(":", 1)[0])

    def mark_pypi(self, pkg: str) -> None:
        with self._lock:
            self.pypi_versions_seen.add(pkg.lower())

    def mark_gpu_check(self) -> None:
        with self._lock:
            self.gpu_check_seen = True

    def mark_env_verified(self) -> None:
        with self._lock:
            self.env_verified = True


def register_builtin_hooks(snapshot: HooksSnapshot, gate: GateState) -> None:
    """Install Repo2ROCm's quality gates as callback hooks."""

    def before_change_base_image(_event: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        if payload.get("tool_name") != "ChangeBaseImage":
            return None
        image = payload.get("tool_input", {}).get("base_image", "")
        repo = image.lower().split(":", 1)[0]
        if repo not in gate.dockerhub_tags_seen:
            return {
                "permissionBehavior": "deny",
                "reason": (
                    f"Refusing to change base image to '{image}' before checking "
                    f"available tags with `DockerHubTags(image='{repo}')`. "
                    "Call DockerHubTags first."
                ),
            }
        return None

    def before_pip_install_cuda_wheel(
        _event: str, payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        if payload.get("tool_name") != "DockerExec":
            return None
        cmd = payload.get("tool_input", {}).get("command", "")
        if "pip install" not in cmd:
            return None
        for pkg in ("nvidia-", "flash-attn", "bitsandbytes", "xformers"):
            if pkg in cmd:
                wheel_name = pkg.rstrip("-")
                if wheel_name not in gate.pypi_versions_seen:
                    return {
                        "permissionBehavior": "deny",
                        "reason": (
                            f"Refusing to `pip install {pkg}*` (CUDA-only wheel) without first "
                            f"calling `PyPIVersions(package='{wheel_name}')`. Many of these "
                            "packages need an AMD-specific source install (e.g. "
                            "FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE)."
                        ),
                    }
        return None

    def before_env_verified(_event: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Block emission of ROCM_ENV_VERIFIED before a GPU sanity check has been run."""
        if payload.get("tool_name") != "EnvVerify":
            return None
        if not gate.gpu_check_seen:
            return {
                "permissionBehavior": "deny",
                "reason": (
                    "EnvVerify requires a successful `torch.cuda.is_available()` or "
                    "`rocm-smi` check first. Use DockerExec to run one, then retry."
                ),
            }
        return None

    snapshot.register_callback("PreToolUse", before_change_base_image)
    snapshot.register_callback("PreToolUse", before_pip_install_cuda_wheel)
    snapshot.register_callback("PreToolUse", before_env_verified)
