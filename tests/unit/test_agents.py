"""Built-in agent definitions sanity."""
from __future__ import annotations

from repo2rocm.agents.builtin import get_builtin_agents
from repo2rocm.core.permissions import PermissionMode


def test_coordinator_has_only_three_tools():
    agents = get_builtin_agents()
    coord = agents["coordinator"]
    assert set(coord.allowed_tools) == {"Agent", "SendMessage", "TaskStop"}
    assert coord.permission_mode == PermissionMode.PLAN


def test_explore_is_read_only():
    e = get_builtin_agents()["explore"]
    assert "Edit" not in (e.allowed_tools or [])
    assert "Write" not in (e.allowed_tools or [])
    assert e.permission_mode == PermissionMode.PLAN
    assert e.omit_user_context is True


def test_verifier_is_background():
    v = get_builtin_agents()["verifier"]
    assert v.background is True
    assert "EnvVerify" in (v.allowed_tools or [])


def test_migrator_has_no_agent_tool():
    m = get_builtin_agents()["migrator"]
    assert "Agent" in m.disallowed_tools
    assert m.permission_mode == PermissionMode.ACCEPT_EDITS
