"""Shared pytest fixtures."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from repo2rocm.bootstrap import reset_for_tests
from repo2rocm.config import reset_settings_for_tests
from repo2rocm.tools.base import clear_registry


@pytest.fixture(autouse=True)
def _reset_globals():
    reset_for_tests()
    reset_settings_for_tests()
    clear_registry()
    yield
    reset_for_tests()
    reset_settings_for_tests()
    clear_registry()


@pytest.fixture
def tmp_workdir(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello, world')\n")
    (tmp_path / "requirements.txt").write_text(
        "torch>=2.0\nnumpy\nflash-attn==2.5.7\nnvidia-cuda-runtime-cu12\n"
    )
    (tmp_path / "README.md").write_text("# example repo\n\nThis is a test.\n")
    return tmp_path


@pytest.fixture
def abort_event() -> asyncio.Event:
    return asyncio.Event()
