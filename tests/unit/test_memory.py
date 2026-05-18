"""File-based memory store + staleness warnings."""
from __future__ import annotations

from pathlib import Path

from repo2rocm.core.memory import MemoryStore
from repo2rocm.core.memory.staleness import staleness_warning


def test_memory_store_round_trip(tmp_path: Path):
    store = MemoryStore(base_dir=tmp_path / "mem")
    store.base_dir.mkdir(parents=True, exist_ok=True)
    p = store.write_topic(
        slug="db_testing",
        type="feedback",
        name="Testing Policy",
        description="Integration tests must hit real DB.",
        body="Don't mock the database in integration tests.\n",
    )
    assert p.exists()
    files = store.list_files()
    assert len(files) == 1
    assert files[0].type == "feedback"
    assert files[0].name == "Testing Policy"
    body = store.load_body(files[0])
    assert "mock" in body


def test_index_manifest_string():
    store = MemoryStore(base_dir=Path("/nonexistent"))
    # gracefully empty
    assert store.manifest_for_recall() == "(no memories)"


def test_staleness_messages():
    assert staleness_warning(0.5) == ""
    assert "yesterday" in staleness_warning(1.5)
    assert "47" in staleness_warning(47)
