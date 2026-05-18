"""bootstrap() is idempotent and registers all tools."""
from __future__ import annotations

from repo2rocm.bootstrap import bootstrap
from repo2rocm.tools.base import get_all_tools


def test_bootstrap_registers_expected_tools():
    bootstrap()
    names = {t.name for t in get_all_tools()}
    # repo
    for name in ("Read", "Grep", "Glob", "Edit", "Write", "ApplyDiff"):
        assert name in names, f"missing {name}"
    # docker
    for name in ("DockerExec", "DockerCommit", "DockerRollback", "ChangeBaseImage", "ChangePythonVersion"):
        assert name in names
    # packaging
    for name in ("WaitingListAdd", "Download", "ConflictListSolve"):
        assert name in names
    # external
    for name in ("PyPIVersions", "DockerHubTags", "WebSearch", "Fetch"):
        assert name in names
    # verify
    for name in ("EnvVerify", "PaperVerify"):
        assert name in names
    # agent
    for name in ("Agent", "SendMessage", "TaskStop"):
        assert name in names


def test_bootstrap_skill_catalog_has_builtins():
    boot = bootstrap()
    assert "rocm_image_catalog" in boot.skill_catalog.manifests


def test_bootstrap_is_idempotent():
    a = bootstrap()
    b = bootstrap()
    assert a is b
