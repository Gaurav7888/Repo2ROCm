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
        "rocm_image_selection",
        "nvidia_alternatives",
        "banned_nvidia_packages",
        "pin_hazards",
        "amd_dependencies",
    }
    assert expected.issubset(set(CONFIGURATION.preload_skills))


def test_configuration_default_max_turns_is_300():
    assert CONFIGURATION.max_turns == 300


def test_configuration_uses_no_permission_mode_kwarg_in_cli():
    """Regression: CLI must no longer expose --permission-mode."""
    import inspect

    from repo2rocm.cli import migrate, reproduce

    migrate_sig = inspect.signature(migrate)
    reproduce_sig = inspect.signature(reproduce)
    assert "permission_mode" not in migrate_sig.parameters
    # But it MUST expose --agent-mode and --rocm-base-image
    assert "agent_mode" in migrate_sig.parameters
    assert "rocm_base_image" in migrate_sig.parameters
    assert migrate_sig.parameters["max_turns"].default.default == 300
    assert "max_turns" in reproduce_sig.parameters
    assert reproduce_sig.parameters["max_turns"].default.default == 300


def _build_prompt(mode: str) -> str:
    """Call the configuration agent's prompt-builder with a minimal ctx."""
    from dataclasses import dataclass

    @dataclass
    class _Ctx:
        options: dict

    ctx = _Ctx(options={"run_mode": mode})
    return CONFIGURATION.system_prompt_builder(
        agent_def=CONFIGURATION, ctx=ctx, skill_catalog=None, tools=[]
    )


def test_configuration_prompt_in_functional_mode_terminates_on_rocm_verified():
    """Functional mode: ROCM_ENV_VERIFIED IS the global terminal condition."""
    prompt = _build_prompt("functional")
    assert "ROCM_ENV_VERIFIED" in prompt
    assert "global stop condition" in prompt.lower()


def test_configuration_prompt_in_reproduce_mode_does_not_stop_at_env_verified():
    """Regression: previously the configuration agent in reproduce mode would
    emit ROCM_ENV_VERIFIED at S5 and stop, never running the paper experiment.
    The reproduce-mode prompt must explicitly state ROCM_ENV_VERIFIED is NOT
    terminal and require continuation to the paper-reproducer step."""
    prompt = _build_prompt("reproduce")
    assert "ROCM_ENV_VERIFIED" in prompt
    assert "NOT terminal" in prompt or "not terminal" in prompt.lower()
    assert "PAPER_REPRODUCED" in prompt
    assert "PAPER_RUN_FAILED" in prompt
    # The reproduce playbook must mention the steps the agent should execute.
    assert "PaperRecall" in prompt
    assert "PaperVerify" in prompt
    assert "synthetic placeholder" in prompt.lower()
