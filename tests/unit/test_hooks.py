"""Hook snapshot + builtin gates."""
from __future__ import annotations

import pytest

from repo2rocm.core.hooks import HookEvent, HooksSnapshot, execute_hooks
from repo2rocm.core.hooks.builtin import GateState, register_builtin_hooks
from repo2rocm.core.permissions import PermissionDecisionKind


@pytest.mark.asyncio
async def test_change_base_image_blocked_without_dockerhub_check():
    snap = HooksSnapshot()
    gate = GateState()
    register_builtin_hooks(snap, gate)
    outcome = await execute_hooks(
        event=HookEvent.PRE_TOOL_USE,
        input_data={"tool_name": "ChangeBaseImage", "tool_input": {"base_image": "rocm/pytorch:latest"}},
        snapshot=snap,
    )
    assert outcome.permission.kind == PermissionDecisionKind.DENY
    assert "DockerHubTags" in outcome.permission.reason


@pytest.mark.asyncio
async def test_change_base_image_allowed_after_dockerhub_check():
    snap = HooksSnapshot()
    gate = GateState()
    register_builtin_hooks(snap, gate)
    gate.mark_dockerhub("rocm/pytorch")
    outcome = await execute_hooks(
        event=HookEvent.PRE_TOOL_USE,
        input_data={"tool_name": "ChangeBaseImage", "tool_input": {"base_image": "rocm/pytorch:latest"}},
        snapshot=snap,
    )
    assert outcome.permission.kind == PermissionDecisionKind.ALLOW


@pytest.mark.asyncio
async def test_pip_install_flash_attn_blocked_without_pypi_check():
    snap = HooksSnapshot()
    gate = GateState()
    register_builtin_hooks(snap, gate)
    outcome = await execute_hooks(
        event=HookEvent.PRE_TOOL_USE,
        input_data={
            "tool_name": "DockerExec",
            "tool_input": {"command": "pip install flash-attn==2.5"},
        },
        snapshot=snap,
    )
    assert outcome.permission.kind == PermissionDecisionKind.DENY
