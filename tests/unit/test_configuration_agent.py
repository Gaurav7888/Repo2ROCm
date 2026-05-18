"""ConfigurationAgent (the single-agent path) — sanity checks."""
from __future__ import annotations

from repo2rocm.agents.builtin import CONFIGURATION, get_builtin_agents
from repo2rocm.core.permissions import PermissionMode


def test_configuration_is_registered_as_default_lookup():
    agents = get_builtin_agents()
    assert "configuration" in agents
    assert agents["configuration"] is CONFIGURATION


def test_configuration_has_bypass_perms_so_container_is_the_boundary():
    """The container itself is the safety boundary — agent must have full perms
    so it can install deps, edit files, run README commands without permission
    gates fighting it."""
    assert CONFIGURATION.permission_mode == PermissionMode.BYPASS


def test_configuration_disallows_sub_agent_spawning():
    """Single-agent flow — the agent must NOT spawn sub-agents (defeats the design)."""
    for blocked in ("Agent", "SendMessage", "TaskStop"):
        assert blocked in CONFIGURATION.disallowed_tools, f"{blocked} should be disallowed"


def test_configuration_preloads_rocm_skills():
    """The agent should boot with the ROCm migration skills already in context."""
    expected = {
        "rocm_image_catalog",
        "cuda_to_rocm_mapping",
        "banned_nvidia_packages",
        "flash_attn_amd_install",
        "py312_compat",
    }
    assert expected.issubset(set(CONFIGURATION.preload_skills))


def test_configuration_uses_no_permission_mode_kwarg_in_cli():
    """Regression: CLI must no longer expose --permission-mode."""
    import inspect

    from repo2rocm.cli import migrate

    sig = inspect.signature(migrate)
    assert "permission_mode" not in sig.parameters
    # But it MUST expose --agent-mode and --rocm-base-image
    assert "agent_mode" in sig.parameters
    assert "rocm_base_image" in sig.parameters
