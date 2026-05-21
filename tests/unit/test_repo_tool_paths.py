from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from repo2rocm.core.permissions import PermissionMode
from repo2rocm.tools.base import ReadFileState, ToolUseContext
from repo2rocm.tools.docker.exec import DockerExec
from repo2rocm.tools.repo.glob import Glob
from repo2rocm.tools.repo.read import Read
from repo2rocm.tools.repo.write import Write


def _ctx(tmp_path: Path, repo_path: Path, *, run_mode: str = "functional") -> ToolUseContext:
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return ToolUseContext(
        agent_id="t",
        session_id="s",
        workdir=output_dir,
        abort_event=asyncio.Event(),
        permission_mode=PermissionMode.BYPASS,
        read_file_state=ReadFileState(),
        options={
            "repo_path": str(repo_path),
            "repo_container_path": "/repo",
            "run_mode": run_mode,
        },
    )


def _mkrepo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "requirements.txt").write_text("transformers==4.31.0\n", encoding="utf-8")
    (repo / "eval_ppl.py").write_text("print('ok')\n", encoding="utf-8")
    return repo


@pytest.mark.asyncio
async def test_read_uses_repo_root_not_output_dir(tmp_path: Path):
    repo = _mkrepo(tmp_path)
    ctx = _ctx(tmp_path, repo)
    (ctx.workdir / "requirements.txt").write_text("WRONG\n", encoding="utf-8")

    tool = Read()
    res_rel = await tool.invoke({"file_path": "requirements.txt"}, ctx)
    res_abs = await tool.invoke({"file_path": "/repo/requirements.txt"}, ctx)

    assert not res_rel.is_error
    assert not res_abs.is_error
    assert "transformers==4.31.0" in res_rel.text
    assert "transformers==4.31.0" in res_abs.text
    assert "WRONG" not in res_rel.text


@pytest.mark.asyncio
async def test_write_maps_repo_container_path_back_to_host_repo(tmp_path: Path):
    repo = _mkrepo(tmp_path)
    ctx = _ctx(tmp_path, repo)

    tool = Write()
    res = await tool.invoke(
        {"file_path": "/repo/data/real_input.json", "content": "{\"ok\": true}"},
        ctx,
    )

    assert not res.is_error, res.text
    assert (repo / "data" / "real_input.json").read_text(encoding="utf-8") == "{\"ok\": true}"


@pytest.mark.asyncio
async def test_glob_roots_patterns_in_repo_path(tmp_path: Path):
    repo = _mkrepo(tmp_path)
    ctx = _ctx(tmp_path, repo)
    (ctx.workdir / "other.py").write_text("print('wrong root')\n", encoding="utf-8")

    tool = Glob()
    res = await tool.invoke({"pattern": "/repo/*.py"}, ctx)

    assert not res.is_error
    assert "eval_ppl.py" in res.data.paths
    assert "other.py" not in res.data.paths


@pytest.mark.asyncio
async def test_write_rejects_synthetic_placeholder_inputs_in_reproduce_mode(tmp_path: Path):
    repo = _mkrepo(tmp_path)
    ctx = _ctx(tmp_path, repo, run_mode="reproduce")

    tool = Write()
    res = await tool.invoke(
        {
            "file_path": "/repo/data/test_data.json",
            "content": '[{"image":"test_image.jpg","question":"What is in this image?","answer":"A test image"}]',
        },
        ctx,
    )

    assert res.is_error
    assert "synthetic placeholder" in res.text
    assert "PAPER_RUN_FAILED" in res.text


@pytest.mark.asyncio
async def test_docker_exec_rejects_synthetic_placeholder_commands_in_reproduce_mode(tmp_path: Path):
    repo = _mkrepo(tmp_path)
    ctx = _ctx(tmp_path, repo, run_mode="reproduce")

    tool = DockerExec()
    res = await tool.invoke(
        {
            "command": "echo '[{\"image\":\"test_image.jpg\",\"question\":\"What is in this image?\"}]' > /repo/data/test_data.json"
        },
        ctx,
    )

    assert res.is_error
    assert "synthetic placeholder" in res.text
    assert "PAPER_RUN_FAILED" in res.text
