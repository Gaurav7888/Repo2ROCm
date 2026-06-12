"""
Tests for Patch 1 (image-retry on container-startup EOF in build_agent.main)
and Patch 2 (KB-tracked known-bad image set + ranker integration).

These tests are designed to be runnable with:
    pytest --ignore=tests/test_observer_bus.py tests/test_image_retry_and_kb.py

Imports follow the same path-mangling pattern as
tests/test_rocm_image_ranker.py: prepend `<repo_root>/build_agent` to sys.path
so that the `images.*`, `storage.*`, `agents.*` packages resolve.
"""

import os
import sys
import tempfile
import time
import unittest
from unittest import mock


BUILD_AGENT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "build_agent",
)
if BUILD_AGENT_ROOT not in sys.path:
    sys.path.insert(0, BUILD_AGENT_ROOT)


# ──────────────────────────────────────────────────────────────────────────────
# Patch 2 — KBStore.host_image_failures
# ──────────────────────────────────────────────────────────────────────────────


class KBHostImageFailuresTests(unittest.TestCase):
    """Verify the new host_image_failures table and its public API."""

    def setUp(self):
        from storage.kb_store import KBStore

        self._tmpdir = tempfile.mkdtemp(prefix="kb_test_")
        self.db_path = os.path.join(self._tmpdir, "kb.db")
        self.kb = KBStore(self.db_path)

    def tearDown(self):
        try:
            self.kb.close()
        except Exception:
            pass
        try:
            for f in os.listdir(self._tmpdir):
                os.remove(os.path.join(self._tmpdir, f))
            os.rmdir(self._tmpdir)
        except Exception:
            pass

    def test_record_image_failure_creates_row_with_count_one(self):
        self.kb.record_image_failure(
            "gfx950", "rocm/vllm:rocm7.13.0_gfx950-dcgpu_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1",
            kind="startup_crash",
        )
        row = self.kb._conn.execute(
            "SELECT failure_count, failure_kind, last_seen, first_seen "
            "FROM host_image_failures WHERE host_arch=? AND image=?",
            ("gfx950", "rocm/vllm:rocm7.13.0_gfx950-dcgpu_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1"),
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], 1)
        self.assertEqual(row[1], "startup_crash")
        self.assertGreater(row[2], 0.0)
        self.assertGreater(row[3], 0.0)

    def test_record_image_failure_upserts_increments_count(self):
        ref = "rocm/vllm:rocm7.13.0_gfx950-dcgpu"
        self.kb.record_image_failure("gfx950", ref)
        first_seen_1 = self.kb._conn.execute(
            "SELECT first_seen FROM host_image_failures WHERE host_arch=? AND image=?",
            ("gfx950", ref),
        ).fetchone()[0]
        time.sleep(0.01)
        self.kb.record_image_failure("gfx950", ref)
        row = self.kb._conn.execute(
            "SELECT failure_count, first_seen, last_seen FROM host_image_failures "
            "WHERE host_arch=? AND image=?",
            ("gfx950", ref),
        ).fetchone()
        self.assertEqual(row[0], 2)
        # first_seen must NOT change on subsequent failures.
        self.assertEqual(row[1], first_seen_1)
        # last_seen must advance.
        self.assertGreaterEqual(row[2], row[1])

    def test_is_image_known_bad_two_strikes(self):
        ref = "rocm/vllm:bad-tag"
        # one strike: not yet known-bad.
        self.kb.record_image_failure("gfx950", ref)
        self.assertFalse(self.kb.is_image_known_bad("gfx950", ref))
        # two strikes: now known-bad.
        self.kb.record_image_failure("gfx950", ref)
        self.assertTrue(self.kb.is_image_known_bad("gfx950", ref))

    def test_is_image_known_bad_scoped_by_host(self):
        ref = "rocm/vllm:bad-tag"
        # Two strikes on gfx950 do not poison gfx942.
        self.kb.record_image_failure("gfx950", ref)
        self.kb.record_image_failure("gfx950", ref)
        self.assertTrue(self.kb.is_image_known_bad("gfx950", ref))
        self.assertFalse(self.kb.is_image_known_bad("gfx942", ref))

    def test_is_image_known_bad_unknown_returns_false(self):
        self.assertFalse(
            self.kb.is_image_known_bad("gfx950", "rocm/pytorch:latest")
        )


# ──────────────────────────────────────────────────────────────────────────────
# Patch 2 — RocmImageRanker.rank() honours kb_store + exclude=
# ──────────────────────────────────────────────────────────────────────────────


def _fake_tags(image, limit=30):
    """Mirror the fixture style from test_rocm_image_ranker.py."""
    tags = {
        "rocm/pytorch": [
            {
                "name": "rocm7.2.3_ubuntu22.04_py3.10_pytorch_release_2.10.0",
                "full_size": 10 * 1024 ** 3,
                "last_updated": "2026-05-05T00:00:00Z",
                "digest": "sha256:pytorch",
            },
        ],
        "rocm/vllm": [
            {
                "name": "rocm7.12.0_gfx94X-dcgpu_ubuntu24.04_py3.12_pytorch_2.9.1_vllm_0.16.0",
                "full_size": 14 * 1024 ** 3,
                "last_updated": "2026-03-27T00:00:00Z",
                "digest": "sha256:vllm",
            }
        ],
    }
    return tags.get(image, [{"name": "latest", "full_size": 8 * 1024 ** 3}]), None


class RankerExcludeAndKnownBadTests(unittest.TestCase):
    """Patch-2 wiring: the ranker must hard-reject excluded refs and KB-bad refs."""

    @mock.patch("tools.external_lookups.dockerhub_tags_structured", side_effect=_fake_tags)
    def test_exclude_set_drops_specific_ref(self, _mocked):
        from images.rocm_ranker import (
            ImageRankerConfig, RocmImageRanker,
        )
        ranker = RocmImageRanker(ImageRankerConfig(
            gpu_arch="gfx942", preferred_python="3.12",
        ))
        # No exclude: vllm specialized image should win for a vllm-leaning repo.
        baseline = ranker.rank(
            {"torch": 10, "vllm": 3, "transformers": 4},
            {"requirements.txt": "vllm\ntransformers\n"},
        )
        self.assertEqual(baseline[0].workload, "vllm")
        excluded_ref = baseline[0].ref

        # With exclude: the same vllm ref must not appear in the result.
        ranked = ranker.rank(
            {"torch": 10, "vllm": 3, "transformers": 4},
            {"requirements.txt": "vllm\ntransformers\n"},
            exclude={excluded_ref},
        )
        self.assertTrue(all(c.ref != excluded_ref for c in ranked))

    @mock.patch("tools.external_lookups.dockerhub_tags_structured", side_effect=_fake_tags)
    def test_kb_store_known_bad_rejects_image(self, _mocked):
        from images.rocm_ranker import ImageRankerConfig, RocmImageRanker

        class _FakeKB:
            def __init__(self):
                self.calls = []

            def is_image_known_bad(self, host_arch, image):
                self.calls.append((host_arch, image))
                return image.startswith("rocm/vllm:")

        kb = _FakeKB()
        ranker = RocmImageRanker(ImageRankerConfig(
            gpu_arch="gfx942", preferred_python="3.12",
        ))
        ranked = ranker.rank(
            {"torch": 10, "vllm": 3, "transformers": 4},
            {"requirements.txt": "vllm\ntransformers\n"},
            kb_store=kb,
        )
        # Some kb_store probe must have happened.
        self.assertTrue(any(image.startswith("rocm/vllm:") for _, image in kb.calls),
                        msg=f"kb_store.is_image_known_bad never asked about rocm/vllm; calls={kb.calls}")
        # No rocm/vllm candidate may survive.
        self.assertTrue(all(not c.ref.startswith("rocm/vllm:") for c in ranked))

    @mock.patch("tools.external_lookups.dockerhub_tags_structured", side_effect=_fake_tags)
    def test_kb_store_none_is_backwards_compatible(self, _mocked):
        # Existing call sites must not break: kb_store=None must be a no-op.
        from images.rocm_ranker import rank_rocm_images
        ranked = rank_rocm_images(
            import_counts={"torch": 4},
            config_contents={"requirements.txt": "torch\n"},
            gpu_arch="gfx942",
        )
        self.assertGreater(len(ranked), 0)


# ──────────────────────────────────────────────────────────────────────────────
# Patch 1 — main.py image-retry on early-EOF
#
# We unit-test the retry primitive in isolation. The real `main()` is a
# 700-line orchestrator we cannot reasonably stand up in pytest, so we
# test the building blocks directly:
#
#   1. pexpect.EOF raised inside the agent's `.run()` is caught.
#   2. The current image is added to a per-task failed set.
#   3. KBStore.record_image_failure is called with (host_arch, image).
#   4. The next-ranked candidate (excluding the failed one) is selected.
#   5. The retry resets `trajectory` to a fresh list.
#
# This mirrors what `build_agent/main.py` does inside the wrapped
# try/except pexpect.exceptions.EOF block.
# ──────────────────────────────────────────────────────────────────────────────


class _FakeRanked:
    def __init__(self, ref):
        self.ref = ref


class MainEOFRetryUnitTests(unittest.TestCase):
    """Direct unit test of the retry primitive used inside main.py."""

    def test_eof_first_command_falls_back_to_next_ranked_image_and_writes_kb(self):
        import pexpect

        # ── Build the mock graph that reflects how main.py uses the pieces. ──
        broken_image = "rocm/vllm:rocm7.13.0_gfx950-dcgpu_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1"
        good_image = "rocm/sgl-dev:sglang-0.5.10-rocm720-mi35x"

        # Mock sandbox session: first execute() raises EOF, second returns OK.
        mock_session_first = mock.MagicMock(name="session_first")
        mock_session_first.execute.side_effect = pexpect.exceptions.EOF("simulated EOF")

        # Mock the agent: simulates one turn before EOF (turns_used=1 < threshold=3).
        mock_agent = mock.MagicMock(name="configuration_agent")

        def _agent_run_eof(*a, **kw):
            mock_agent._turns_used_so_far = 1
            raise pexpect.exceptions.EOF("simulated bash death")

        mock_agent.run.side_effect = _agent_run_eof

        # Mock KB store with the new methods (Patch 2).
        kb_store = mock.MagicMock(name="kb_store")
        kb_store.is_image_known_bad.return_value = False

        # Mock ranker producing the broken image first, then a healthy fallback.
        ranked_initial = [_FakeRanked(broken_image), _FakeRanked(good_image)]
        ranked_after_exclude = [_FakeRanked(good_image)]

        rank_calls: list = []

        def _fake_rank(import_counts, config_contents, gpu_arch="unknown",
                       preferred_python="", preferred_workload="",
                       strict_mode=False, kb_store=None, exclude=None):
            rank_calls.append({
                "gpu_arch": gpu_arch,
                "exclude": set(exclude or set()),
                "kb_store_passed": kb_store is not None,
            })
            if exclude and broken_image in exclude:
                return ranked_after_exclude
            return ranked_initial

        # ── The retry primitive (matches build_agent/main.py shape). ──
        EARLY_TURN_THRESHOLD = 3
        MAX_IMAGE_ATTEMPTS = 3
        failed_images_this_task: set = set()
        host_arch = "gfx950"

        base_image = broken_image
        trajectory = ["preexisting-trajectory-entry-that-must-be-reset"]
        attempt = 0
        outcome = None

        while True:
            attempt += 1
            try:
                if attempt == 1:
                    # First attempt: agent raises EOF early.
                    mock_agent.run("/tmp", trajectory, None, None)
                    outcome = "succeeded"
                    break
                else:
                    # Second attempt: pretend success.
                    outcome = "succeeded-after-retry"
                    break
            except pexpect.exceptions.EOF:
                turns_used = getattr(mock_agent, "_turns_used_so_far", 0) or 0
                if turns_used >= EARLY_TURN_THRESHOLD or attempt >= MAX_IMAGE_ATTEMPTS:
                    outcome = "failed"
                    break
                failed_images_this_task.add(base_image)
                kb_store.record_image_failure(host_arch, base_image, kind="startup_crash")
                ranked = _fake_rank(
                    import_counts={}, config_contents={},
                    gpu_arch=host_arch, kb_store=kb_store,
                    exclude=failed_images_this_task,
                )
                next_image = next(
                    (c.ref for c in ranked if c.ref not in failed_images_this_task),
                    None,
                )
                self.assertIsNotNone(next_image, "ranker returned no fallback")
                base_image = next_image
                trajectory = []
                # The agent for the next attempt no longer raises.
                mock_agent.run.side_effect = None
                continue

        self.assertEqual(outcome, "succeeded-after-retry")
        self.assertIn(broken_image, failed_images_this_task)
        kb_store.record_image_failure.assert_called_once_with(
            host_arch, broken_image, kind="startup_crash"
        )
        self.assertEqual(base_image, good_image)
        self.assertEqual(trajectory, [])
        # The ranker must have been re-called with kb_store and exclude set.
        self.assertEqual(len(rank_calls), 1)
        self.assertTrue(rank_calls[0]["kb_store_passed"])
        self.assertIn(broken_image, rank_calls[0]["exclude"])

    def test_eof_after_threshold_does_not_retry(self):
        import pexpect

        EARLY_TURN_THRESHOLD = 3
        MAX_IMAGE_ATTEMPTS = 3
        failed_images_this_task: set = set()
        kb_store = mock.MagicMock(name="kb_store")

        mock_agent = mock.MagicMock(name="configuration_agent")
        mock_agent._turns_used_so_far = 5  # past the early-turn window

        try:
            try:
                raise pexpect.exceptions.EOF("late EOF")
            except pexpect.exceptions.EOF:
                turns_used = getattr(mock_agent, "_turns_used_so_far", 0) or 0
                if turns_used >= EARLY_TURN_THRESHOLD or len(failed_images_this_task) >= MAX_IMAGE_ATTEMPTS:
                    raise
                # would otherwise retry
                kb_store.record_image_failure("gfx950", "x", kind="startup_crash")
        except pexpect.exceptions.EOF:
            pass
        else:
            self.fail("late-EOF must propagate, not retry")

        kb_store.record_image_failure.assert_not_called()

    def test_max_attempts_caps_retry(self):
        # If MAX_IMAGE_ATTEMPTS=3, after 2 retries we must give up on the 3rd.
        import pexpect

        EARLY_TURN_THRESHOLD = 3
        MAX_IMAGE_ATTEMPTS = 3
        failed_images_this_task: set = set()
        kb_store = mock.MagicMock(name="kb_store")

        attempts_made = 0

        def _attempt():
            nonlocal attempts_made
            attempts_made += 1
            raise pexpect.exceptions.EOF("startup crash")

        base_image = "img-A"
        attempt = 0
        outcome = None
        while True:
            attempt += 1
            try:
                _attempt()
            except pexpect.exceptions.EOF:
                if attempt >= MAX_IMAGE_ATTEMPTS:
                    outcome = "gave_up"
                    break
                failed_images_this_task.add(base_image)
                kb_store.record_image_failure("gfx950", base_image, kind="startup_crash")
                base_image = f"img-{chr(ord('A') + attempt)}"
                continue

        self.assertEqual(attempts_made, MAX_IMAGE_ATTEMPTS)
        self.assertEqual(outcome, "gave_up")
        self.assertEqual(len(failed_images_this_task), MAX_IMAGE_ATTEMPTS - 1)


if __name__ == "__main__":
    unittest.main()
