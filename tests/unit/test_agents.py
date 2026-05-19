"""Built-in agent definitions sanity."""
from __future__ import annotations

from repo2rocm.agents.builtin import get_builtin_agents
from repo2rocm.agents.lifecycle import _STRICTNESS
from repo2rocm.core.permissions import PermissionMode
from repo2rocm.tools.agent_tool import Agent, SendMessage, TaskStop


def test_orchestration_tools_are_read_only():
    """Agent / SendMessage / TaskStop are delegation/control-plane, not mutation.

    Because they're read-only at the tool level, the coordinator's *own*
    permission_mode does not need to be PLAN to keep the coordinator safe —
    and in fact MUST NOT be PLAN (see test_coordinator_does_not_cascade_plan_to_children).
    """
    a, sm, ts = Agent(), SendMessage(), TaskStop()
    assert a.is_read_only(
        Agent.input_model(description="x", prompt="y", subagent_type="explore")
    )
    assert sm.is_read_only(SendMessage.input_model(to="x", message="y"))
    assert ts.is_read_only(TaskStop.input_model(task_id="t"))


def test_coordinator_has_only_three_tools():
    agents = get_builtin_agents()
    coord = agents["coordinator"]
    assert set(coord.allowed_tools) == {"Agent", "SendMessage", "TaskStop"}


def test_coordinator_does_not_cascade_plan_to_children():
    """Regression for the bug where coordinator-mode migrations did nothing.

    `agents/lifecycle.py` step 5 makes a child agent inherit the STRICTER of
    (parent_mode, child_def_mode). If the coordinator runs in PLAN (strictness
    5) every descendant — including migrators — is forced into PLAN, and every
    Edit/Write/DockerExec is denied with "plan mode denies mutations". The
    visible symptom is a "successful" coordinator run that produced zero file
    changes.

    The coordinator's own toolset is read-only at the tool level, so we don't
    need PLAN to protect the coordinator from itself; we just need a mode that
    lets each sub-agent use ITS own declared mode.
    """
    agents = get_builtin_agents()
    coord_mode = agents["coordinator"].permission_mode
    migrator_mode = agents["migrator"].permission_mode

    # The coordinator must not be the strictest mode, otherwise it caps every child.
    assert _STRICTNESS[coord_mode] <= _STRICTNESS[migrator_mode], (
        f"coordinator.permission_mode={coord_mode.value!r} is at least as "
        f"strict as migrator.permission_mode={migrator_mode.value!r}; this "
        "will trap migrators in a read-only mode via the cascade in "
        "agents/lifecycle.py step 5."
    )

    # Migrators MUST be able to mutate when spawned by the coordinator.
    # We replicate the cascade rule here so a future refactor of the rule
    # also has to revisit this invariant.
    effective_for_migrator = (
        coord_mode
        if _STRICTNESS[coord_mode] >= _STRICTNESS[migrator_mode]
        else migrator_mode
    )
    assert effective_for_migrator in {
        PermissionMode.ACCEPT_EDITS,
        PermissionMode.AUTO,
        PermissionMode.BYPASS,
    }, "Effective mode for a migrator under the coordinator must allow mutations."


def test_coordinator_mode_matches_single_agent_mode():
    """Coordinator + Migrator should have the same effective permissions as the
    single-agent `configuration` flow. Both flows do the same Docker-sandbox
    work (edit files, DockerExec installs, DockerCommit checkpoints); they only
    differ in whether the work is one long agent or split across sub-agents.
    The Docker container is the safety boundary in both cases — there should
    be no host-level permission asymmetry between them."""
    agents = get_builtin_agents()
    config_mode = agents["configuration"].permission_mode
    coord_mode = agents["coordinator"].permission_mode
    migrator_mode = agents["migrator"].permission_mode
    paper_mode = agents["paper-reproducer"].permission_mode

    assert config_mode == PermissionMode.BYPASS
    # Coordinator and its write-heavy children mirror configuration's BYPASS.
    assert coord_mode == PermissionMode.BYPASS
    assert migrator_mode == PermissionMode.BYPASS
    assert paper_mode == PermissionMode.BYPASS


def test_explore_is_read_only():
    e = get_builtin_agents()["explore"]
    assert "Edit" not in (e.allowed_tools or [])
    assert "Write" not in (e.allowed_tools or [])
    # Read-only sub-agents stay in PLAN as belt-and-suspenders enforcement of
    # their already-read-only allowlist. PLAN here is safe because they don't
    # have any descendants to cascade onto.
    assert e.permission_mode == PermissionMode.PLAN
    assert e.omit_user_context is True


def test_verifier_is_background():
    v = get_builtin_agents()["verifier"]
    assert v.background is True
    assert "EnvVerify" in (v.allowed_tools or [])
    # Verifier is intentionally adversarial / read-only.
    assert v.permission_mode == PermissionMode.PLAN


def test_migrator_has_no_agent_tool():
    m = get_builtin_agents()["migrator"]
    assert "Agent" in m.disallowed_tools
    assert m.permission_mode == PermissionMode.BYPASS
