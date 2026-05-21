"""Repo tools: Read, Grep, Glob, Edit, Write, ApplyDiff."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from repo2rocm.core.permissions import PermissionMode
from repo2rocm.tools.base import ReadFileState, ToolUseContext
from repo2rocm.tools.repo import register_repo_tools
from repo2rocm.tools.repo.apply_diff import ApplyDiff, ApplyDiffInput
from repo2rocm.tools.repo.edit import Edit, EditInput
from repo2rocm.tools.repo.glob import Glob, GlobInput
from repo2rocm.tools.repo.grep import Grep, GrepInput
from repo2rocm.tools.repo.read import Read, ReadInput
from repo2rocm.tools.repo.write import Write, WriteInput


@pytest.fixture
def ctx(tmp_workdir: Path) -> ToolUseContext:
    return ToolUseContext(
        agent_id="t1",
        session_id="s1",
        workdir=tmp_workdir,
        abort_event=asyncio.Event(),
        permission_mode=PermissionMode.ACCEPT_EDITS,
        read_file_state=ReadFileState(),
    )


@pytest.mark.asyncio
async def test_read(ctx):
    r = await Read().call(ReadInput(file_path="src/main.py"), ctx)
    assert not r.is_error
    assert "hello, world" in r.text
    assert r.data.total_lines == 1


@pytest.mark.asyncio
async def test_read_missing_file(ctx):
    r = await Read().call(ReadInput(file_path="does/not/exist"), ctx)
    assert r.is_error


@pytest.mark.asyncio
async def test_grep_finds_matches(ctx):
    r = await Grep().call(GrepInput(pattern="hello", path="src"), ctx)
    assert not r.is_error
    assert any("main.py" in m.file for m in r.data.matches)


@pytest.mark.asyncio
async def test_grep_single_file_path_with_colons_in_match(ctx, tmp_workdir):
    # Regression: when `path` points at a single file, ripgrep would historically
    # omit the filename prefix, and a match body containing a ':' caused the
    # parser to crash with `int("def foo(d")`. Now we force -H/--with-filename.
    (tmp_workdir / "src" / "rotated.py").write_text(
        "import torch\n"
        "def generate_rotation_matrix(d: int, device: str = \"cpu\"):\n"
        "    return torch.eye(d, device=device)\n"
    )
    r = await Grep().call(
        GrepInput(pattern="device|cuda", path="src/rotated.py"),
        ctx,
    )
    assert not r.is_error
    assert any("rotated.py" in m.file for m in r.data.matches)
    assert any(m.line == 2 for m in r.data.matches)


@pytest.mark.asyncio
async def test_glob_lists_python_files(ctx):
    r = await Glob().call(GlobInput(pattern="**/*.py"), ctx)
    assert not r.is_error
    assert "src/main.py" in r.data.paths


@pytest.mark.asyncio
async def test_write_then_edit(ctx):
    await Write().call(WriteInput(file_path="new.txt", content="foo bar"), ctx)
    # warm file-state cache
    await Read().call(ReadInput(file_path="new.txt"), ctx)
    r = await Edit().call(
        EditInput(file_path="new.txt", old_string="foo", new_string="baz"),
        ctx,
    )
    assert not r.is_error
    assert r.data.replacements == 1
    assert (ctx.workdir / "new.txt").read_text() == "baz bar"


@pytest.mark.asyncio
async def test_edit_rejects_no_op(ctx):
    await Write().call(WriteInput(file_path="n.txt", content="abc"), ctx)
    e = Edit()
    sem = e.validate_semantic(EditInput(file_path="n.txt", old_string="abc", new_string="abc"), ctx)
    assert sem is not None


def test_write_rejects_synthetic_paper_log_in_reproduce_mode(ctx):
    ctx.options["run_mode"] = "reproduce"
    sem = Write().validate_semantic(
        WriteInput(
            file_path="/repo/paper_experiment_formatted.log",
            content="perplexity: 5.53\n",
        ),
        ctx,
    )
    assert sem is not None


def test_edit_rejects_paper_log_mutation_in_reproduce_mode(ctx):
    ctx.options["run_mode"] = "reproduce"
    sem = Edit().validate_semantic(
        EditInput(
            file_path="/repo/paper_experiment.log",
            old_string="old",
            new_string="new",
        ),
        ctx,
    )
    assert sem is not None


@pytest.mark.asyncio
async def test_apply_diff(ctx):
    await Write().call(WriteInput(file_path="d.txt", content="alpha\nbeta\ngamma\n"), ctx)
    await Read().call(ReadInput(file_path="d.txt"), ctx)
    diff = (
        "<<<<<<< SEARCH\nalpha\n=======\nALPHA\n>>>>>>> REPLACE\n"
        "<<<<<<< SEARCH\ngamma\n=======\nGAMMA\n>>>>>>> REPLACE\n"
    )
    r = await ApplyDiff().call(ApplyDiffInput(file_path="d.txt", diff=diff), ctx)
    assert not r.is_error
    assert r.data.hunks_applied == 2
    text = (ctx.workdir / "d.txt").read_text()
    assert "ALPHA" in text and "GAMMA" in text
