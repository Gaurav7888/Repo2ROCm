"""WaitingList + ConflictList semantics."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from repo2rocm.core.permissions import PermissionMode
from repo2rocm.tools.base import ReadFileState, ToolUseContext
from repo2rocm.tools.packaging.waiting_list import (
    PackageSpec,
    WaitingList,
    WaitingListAdd,
    WaitingListAddFile,
    WaitingListShow,
    WLAddInput,
    WLFileInput,
    WLShowInput,
)


@pytest.fixture
def ctx(tmp_workdir: Path) -> ToolUseContext:
    return ToolUseContext(
        agent_id="t",
        session_id="s",
        workdir=tmp_workdir,
        abort_event=asyncio.Event(),
        permission_mode=PermissionMode.ACCEPT_EDITS,
        read_file_state=ReadFileState(),
    )


@pytest.mark.asyncio
async def test_waiting_list_add_and_show(ctx):
    add = WaitingListAdd()
    await add.call(WLAddInput(name="numpy", version_constraint=">=1.24"), ctx)
    await add.call(WLAddInput(name="torch"), ctx)
    show = await WaitingListShow().call(WLShowInput(), ctx)
    assert "numpy>=1.24" in show.text
    assert "torch" in show.text


@pytest.mark.asyncio
async def test_waiting_list_addfile_parses_requirements(ctx):
    res = await WaitingListAddFile().call(WLFileInput(file_path="requirements.txt"), ctx)
    assert res.data.added_count >= 2  # torch, numpy, flash-attn ...


def test_in_process_waiting_list_detects_conflict():
    wl = WaitingList()
    ok, _ = wl.add(PackageSpec("numpy", ">=1.20"))
    assert ok
    ok2, msg = wl.add(PackageSpec("numpy", "==1.18"))
    assert not ok2 and "CONFLICT" in msg
