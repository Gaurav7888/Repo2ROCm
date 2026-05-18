"""Permission resolution chain."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from repo2rocm.core.permissions import (
    PermissionDecisionKind,
    PermissionMode,
    PermissionRule,
    PermissionRuleSet,
    resolve_permission,
)
from repo2rocm.tools.base import ReadFileState, ToolUseContext
from repo2rocm.tools.repo.read import Read, ReadInput
from repo2rocm.tools.repo.write import Write


def _ctx(mode: PermissionMode, tmp_path: Path) -> ToolUseContext:
    return ToolUseContext(
        agent_id="t1",
        session_id="s1",
        workdir=tmp_path,
        abort_event=asyncio.Event(),
        permission_mode=mode,
        read_file_state=ReadFileState(),
    )


def test_plan_allows_read(tmp_path: Path):
    ctx = _ctx(PermissionMode.PLAN, tmp_path)
    decision = resolve_permission(Read(), {"file_path": "x"}, ctx, rules=PermissionRuleSet.empty())
    assert decision.kind == PermissionDecisionKind.ALLOW


def test_plan_denies_write(tmp_path: Path):
    ctx = _ctx(PermissionMode.PLAN, tmp_path)
    decision = resolve_permission(
        Write(), {"file_path": "x", "content": "y"}, ctx, rules=PermissionRuleSet.empty()
    )
    assert decision.kind == PermissionDecisionKind.DENY


def test_accept_edits_allows_edits(tmp_path: Path):
    ctx = _ctx(PermissionMode.ACCEPT_EDITS, tmp_path)
    decision = resolve_permission(
        Write(), {"file_path": "x", "content": "y"}, ctx, rules=PermissionRuleSet.empty()
    )
    assert decision.kind == PermissionDecisionKind.ALLOW


def test_deny_rule_beats_mode(tmp_path: Path):
    ctx = _ctx(PermissionMode.BYPASS, tmp_path)
    rules = PermissionRuleSet(
        always_allow=[],
        always_deny=[PermissionRule(tool="Read", behavior=PermissionDecisionKind.DENY)],
        always_ask=[],
    )
    decision = resolve_permission(Read(), {"file_path": "x"}, ctx, rules=rules)
    assert decision.kind == PermissionDecisionKind.DENY
