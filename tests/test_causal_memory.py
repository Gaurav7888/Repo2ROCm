"""Tests for Track 1: Causal Migration Memory.

Covers:
  * `CausalTransition` serialisation roundtrip + state similarity.
  * `KBStore.insert_transition` / `query_transitions` (state-similarity rank).
  * `TrajectoryDistiller.extract_causal_transitions` from a synthetic
    failure → success → ROCM_ENV_VERIFIED trajectory.
  * `BuildMemoryProvider.provide_causal_memory` formats transitions as the
    structured `[CAUSAL] state{...} → action{...} → outcome{...}` lines and
    surfaces them in BEGIN + IN phases.
  * `seed_causal_transitions` only seeds when the table is empty.
"""

import os
import sys
import tempfile
import unittest


BUILD_AGENT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "build_agent",
)
if BUILD_AGENT_ROOT not in sys.path:
    sys.path.insert(0, BUILD_AGENT_ROOT)

from storage.models import (  # noqa: E402
    BuildAttempt, BuildFingerprint, BuildOutcome,
    CausalAction, CausalOutcome, CausalState, CausalTransition,
    MemoryPhase, MemoryRequest, TrajectoryRecord,
)
from storage.kb_store import KBStore  # noqa: E402
from storage.trajectory_store import TrajectoryStore  # noqa: E402
from errors.classifier import ErrorClassifier  # noqa: E402
from errors.seed_patterns import seed_if_empty  # noqa: E402
from rules.engine import RuleEngine  # noqa: E402
from learning.memory_provider import (  # noqa: E402
    BuildMemoryProvider,
    format_causal_counterfactuals,
    format_causal_transition,
)
from learning.distiller import TrajectoryDistiller  # noqa: E402
from learning.causal_seed import seed_causal_transitions  # noqa: E402


def _make_kb_paths():
    tmp = tempfile.mkdtemp(prefix="causal_kb_")
    kb_path = os.path.join(tmp, "kb.db")
    traj_path = os.path.join(tmp, "trajectories.db")
    return tmp, kb_path, traj_path


class CausalSerializationTests(unittest.TestCase):
    def test_transition_roundtrip_preserves_all_fields(self):
        original = CausalTransition(
            id="t-roundtrip",
            transition_class="cuda_only_wheel_to_rocm_source_build",
            state=CausalState(
                repo_fingerprint="torch+flash_attn",
                image="rocm/pytorch:rocm7.2",
                gpu_arch="gfx942",
                error_class="FLASH_ATTN_CUDA_WHEEL",
                error_signature="No module named flash_attn_2_cuda",
                degradation_policy="strict",
            ),
            action=CausalAction(
                type="package_strategy",
                command="FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE python setup.py install",
                evidence=["pypi_versions", "rocm_package_guidance"],
            ),
            outcome=CausalOutcome(
                return_code=0,
                verification=["import flash_attn passed", "GPU smoke test passed"],
                degradation="D1",
                confidence=0.82,
            ),
            counterfactuals=[
                {
                    "action": "pip install flash-attn",
                    "expected_outcome": "fail",
                    "reason": "PyPI wheel is CUDA-only.",
                }
            ],
            source_attempt_id="attempt-123",
            source="learned",
            evidence_count=2,
        )
        recovered = CausalTransition.from_dict(original.to_dict())

        self.assertEqual(recovered.id, original.id)
        self.assertEqual(recovered.transition_class, original.transition_class)
        self.assertEqual(recovered.state.error_class, "FLASH_ATTN_CUDA_WHEEL")
        self.assertEqual(recovered.state.gpu_arch, "gfx942")
        self.assertEqual(recovered.action.command, original.action.command)
        self.assertEqual(recovered.outcome.degradation, "D1")
        self.assertAlmostEqual(recovered.outcome.confidence, 0.82, places=4)
        self.assertEqual(len(recovered.counterfactuals), 1)
        self.assertEqual(
            recovered.counterfactuals[0]["expected_outcome"], "fail"
        )
        self.assertEqual(recovered.source, "learned")
        self.assertEqual(recovered.evidence_count, 2)

    def test_state_signature_is_stable_and_distinguishing(self):
        s1 = CausalState(
            repo_fingerprint="fp-a", image="img-a", gpu_arch="gfx942",
            error_class="X", error_signature="sig", degradation_policy="strict",
        )
        s1_again = CausalState(
            repo_fingerprint="fp-a", image="img-a", gpu_arch="gfx942",
            error_class="X", error_signature="sig", degradation_policy="strict",
        )
        s2 = CausalState(
            repo_fingerprint="fp-a", image="img-b", gpu_arch="gfx942",
            error_class="X", error_signature="sig", degradation_policy="strict",
        )
        self.assertEqual(s1.signature(), s1_again.signature())
        self.assertNotEqual(s1.signature(), s2.signature())

    def test_state_similarity_weights_error_class_highest(self):
        t = CausalTransition(state=CausalState(
            error_class="FLASH_ATTN_CUDA_WHEEL",
            image="rocm/pytorch", gpu_arch="gfx942",
            error_signature="No module named flash_attn_2_cuda",
            repo_fingerprint="fp", degradation_policy="strict",
        ))
        same_err_diff_img = CausalState(
            error_class="FLASH_ATTN_CUDA_WHEEL", image="other",
            gpu_arch="gfx942", error_signature="diff",
            repo_fingerprint="fp", degradation_policy="strict",
        )
        diff_err_same_img = CausalState(
            error_class="OTHER", image="rocm/pytorch", gpu_arch="gfx942",
            error_signature="diff", repo_fingerprint="fp",
            degradation_policy="strict",
        )
        # Same error_class should outrank same image.
        self.assertGreater(
            t.similarity(same_err_diff_img),
            t.similarity(diff_err_same_img),
        )


class CausalKBStoreTests(unittest.TestCase):
    def test_insert_and_query_returns_best_match_first(self):
        tmp, kb_path, _ = _make_kb_paths()
        kb = KBStore(kb_path)
        try:
            t1 = CausalTransition(
                id="t1",
                transition_class="cuda_only_wheel_to_rocm_source_build",
                state=CausalState(
                    repo_fingerprint="fp1", image="rocm/pytorch",
                    gpu_arch="gfx942",
                    error_class="FLASH_ATTN_CUDA_WHEEL",
                    error_signature="No module named flash_attn_2_cuda",
                    degradation_policy="strict",
                ),
                action=CausalAction(
                    type="package_strategy",
                    command="FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE python setup.py install",
                ),
                outcome=CausalOutcome(return_code=0, degradation="D1",
                                      confidence=0.82),
            )
            t2 = CausalTransition(
                id="t2",
                transition_class="wrong_image_to_ranked_image_switch",
                state=CausalState(
                    repo_fingerprint="fp2", image="python:3.10",
                    gpu_arch="unknown",
                    error_class="TORCH_CUDA_NOT_AVAILABLE",
                    error_signature="torch.cuda.is_available() == False",
                    degradation_policy="strict",
                ),
                action=CausalAction(
                    type="image_switch",
                    command="change_base_image rocm/pytorch:latest",
                ),
                outcome=CausalOutcome(return_code=0, degradation="D0",
                                      confidence=0.9),
            )
            kb.insert_transition(t1)
            kb.insert_transition(t2)

            self.assertEqual(kb.count_transitions(), 2)

            query = CausalState(
                error_class="FLASH_ATTN_CUDA_WHEEL",
                image="rocm/pytorch", gpu_arch="gfx942",
                error_signature="No module named flash_attn_2_cuda",
                repo_fingerprint="fp1", degradation_policy="strict",
            )
            results = kb.query_transitions(query, top_k=5)
            # The strict filters (error_class, error_signature, image,
            # fingerprint) all match t1 only; t2 has no overlap on those
            # discriminating fields, so retrieval correctly excludes it.
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].id, "t1")

            # An empty query state forces the "no specific filter matched"
            # fallback path (most-recent records); both rows must surface
            # so callers always get *something* once any data exists.
            generic_results = kb.query_transitions(CausalState(), top_k=5)
            self.assertEqual(len(generic_results), 2)

            # Querying purely on image="rocm/pytorch" should still match
            # t1 (whose image is rocm/pytorch) but not t2.
            image_results = kb.query_transitions(
                CausalState(image="rocm/pytorch"), top_k=5,
            )
            self.assertGreaterEqual(len(image_results), 1)
            self.assertEqual(image_results[0].id, "t1")
        finally:
            kb.close()


class CausalRelevanceGateTests(unittest.TestCase):
    """PR #2 follow-up item 3: a `min_similarity` relevance gate so the
    per-turn `[CAUSAL]` injection only fires on transitions that actually
    match the current state (fixes the context-length blow-up on easy repos).
    """

    def _relevant_transition(self) -> CausalTransition:
        return CausalTransition(
            id="t-relevant",
            transition_class="cuda_only_wheel_to_rocm_source_build",
            state=CausalState(
                repo_fingerprint="fp1", image="rocm/pytorch",
                gpu_arch="gfx942",
                error_class="FLASH_ATTN_CUDA_WHEEL",
                error_signature="No module named flash_attn_2_cuda",
                degradation_policy="strict",
            ),
            action=CausalAction(
                type="package_strategy",
                command="FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE python setup.py install",
            ),
            outcome=CausalOutcome(return_code=0, degradation="D1",
                                  confidence=0.82),
        )

    def _unrelated_transition(self) -> CausalTransition:
        # Shares only `image` with the query so it is *absorbed* by the SQL
        # pre-filter (proving the gate, not the pre-filter, is what drops it),
        # but its weighted similarity to the query is just 0.15 (< 0.3).
        return CausalTransition(
            id="t-unrelated",
            transition_class="wrong_image_to_ranked_image_switch",
            state=CausalState(
                repo_fingerprint="fpX", image="rocm/pytorch",
                gpu_arch="",
                error_class="UNRELATED_LINKER_ERROR",
                error_signature="undefined reference to foo",
                degradation_policy="",
            ),
            action=CausalAction(
                type="image_switch",
                command="change_base_image other:latest",
            ),
            outcome=CausalOutcome(return_code=0, degradation="D0",
                                  confidence=0.9),
        )

    def _matching_query(self) -> CausalState:
        return CausalState(
            repo_fingerprint="fp1", image="rocm/pytorch", gpu_arch="gfx942",
            error_class="FLASH_ATTN_CUDA_WHEEL",
            error_signature="No module named flash_attn_2_cuda",
            degradation_policy="strict",
        )

    def test_query_transitions_gate_keeps_only_relevant(self):
        tmp, kb_path, _ = _make_kb_paths()
        kb = KBStore(kb_path)
        try:
            rel = self._relevant_transition()
            unrel = self._unrelated_transition()
            kb.insert_transition(rel)
            kb.insert_transition(unrel)

            query = self._matching_query()

            # Sanity: both share image="rocm/pytorch", so both are candidates.
            self.assertGreaterEqual(rel.similarity(query), 0.3)
            self.assertLess(unrel.similarity(query), 0.3)

            # Without a gate the un-gated path still surfaces the unrelated
            # candidate (this is the pre-fix behaviour we are guarding against).
            ungated = kb.query_transitions(query, top_k=5)
            self.assertEqual({t.id for t in ungated}, {"t-relevant", "t-unrelated"})

            # With the 0.3 gate only the relevant transition survives.
            gated = kb.query_transitions(query, top_k=5, min_similarity=0.3)
            self.assertEqual([t.id for t in gated], ["t-relevant"])
        finally:
            kb.close()

    def test_query_transitions_all_unrelated_returns_empty(self):
        tmp, kb_path, _ = _make_kb_paths()
        kb = KBStore(kb_path)
        try:
            kb.insert_transition(self._relevant_transition())
            kb.insert_transition(self._unrelated_transition())

            # A query unrelated to every stored transition: nothing matches
            # the SQL pre-filter, the fallback surfaces recent rows, and the
            # gate then drops them all → no injection.
            unrelated_query = CausalState(
                repo_fingerprint="zzz", image="alpine:3.18", gpu_arch="gfx906",
                error_class="GCC_INTERNAL_ERROR",
                error_signature="internal compiler error",
                degradation_policy="permissive",
            )
            gated = kb.query_transitions(unrelated_query, top_k=5,
                                         min_similarity=0.3)
            self.assertEqual(gated, [])

            # Without the gate the fallback still returns the recent rows.
            self.assertGreaterEqual(
                len(kb.query_transitions(unrelated_query, top_k=5)), 1,
            )
        finally:
            kb.close()

    def test_provider_per_turn_path_gates_by_relevance(self):
        tmp, kb_path, traj_path = _make_kb_paths()
        kb = KBStore(kb_path)
        traj = TrajectoryStore(traj_path)
        try:
            kb.insert_transition(self._relevant_transition())
            kb.insert_transition(self._unrelated_transition())
            provider = BuildMemoryProvider(
                kb, traj, ErrorClassifier(kb), RuleEngine(kb),
            )

            # Matching per-turn request → only the relevant transition is
            # surfaced even though the unrelated one shares the image filter.
            matching = MemoryRequest(
                query="pip install flash-attn",
                context={
                    "rocm_mode": True,
                    "image": "rocm/pytorch",
                    "gpu_arch": "gfx942",
                    "error_class": "FLASH_ATTN_CUDA_WHEEL",
                    "degradation_policy": "strict",
                },
                phase=MemoryPhase.IN.value,
                current_error="ImportError: No module named flash_attn_2_cuda",
                turn_number=1,
            )
            items = provider.provide_causal_memory(matching)  # default 0.3 gate
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].item_type, "causal")
            self.assertIn("err=FLASH_ATTN_CUDA_WHEEL", items[0].content)

            # A request that matches nothing relevant → empty list, so the
            # per-turn `[CAUSAL]` injection is suppressed entirely.
            irrelevant = MemoryRequest(
                query="ls -la",
                context={
                    "rocm_mode": True,
                    "image": "alpine:3.18",
                    "gpu_arch": "gfx906",
                    "error_class": "GCC_INTERNAL_ERROR",
                    "degradation_policy": "permissive",
                },
                phase=MemoryPhase.IN.value,
                current_error="internal compiler error",
                turn_number=2,
            )
            self.assertEqual(provider.provide_causal_memory(irrelevant), [])

            # The IN-phase response path applies the same gate.
            in_response = provider.provide_memory(irrelevant)
            self.assertEqual(
                [i for i in in_response.items if i.item_type == "causal"], [],
            )
        finally:
            kb.close()
            traj.close()


class CausalPerRunDedupTests(unittest.TestCase):
    """Per-run dedup of causal `[CAUSAL]` advisories: the same transition
    must not re-inject every turn. Complements the `min_similarity` relevance
    gate. Together they bound cumulative per-turn `[CAUSAL]` growth to
    O(unique matching transitions), not O(turns)."""

    def _make_configuration_stub(self, items_to_return):
        """Build a Configuration-shaped object with only the attributes
        `_provide_causal_memory_per_turn` reads, plus a memory_provider
        whose `provide_causal_memory` returns `items_to_return` on every
        call. Avoids the heavy real `Configuration.__init__` which would
        require a live Sandbox."""
        from unittest.mock import MagicMock
        from agents.configuration import Configuration  # noqa: E402

        cfg = object.__new__(Configuration)
        cfg._causal_transitions_surfaced_this_run = set()
        cfg.build_attempt = None
        cfg.image_name = "rocm/pytorch:tag"
        cfg.rocm_mode = True
        cfg.no_scale_down = True

        mp = MagicMock()
        mp.provide_causal_memory = MagicMock(return_value=items_to_return)
        cfg.memory_provider = mp
        return cfg

    def _make_item(self, item_id, content):
        from storage.models import MemoryItem  # noqa: E402
        return MemoryItem(
            id=item_id, content=content, item_type="causal", confidence=0.8,
        )

    def test_same_transition_only_surfaced_once_per_run(self):
        item = self._make_item(
            "causal_abc",
            "[CAUSAL] state{img=rocm/pytorch} → action{x} → outcome{ok}",
        )
        cfg = self._make_configuration_stub([item])

        out1 = cfg._provide_causal_memory_per_turn(
            command_text="some cmd", sandbox_res="", return_code=1,
            classified_error=None, turn=1,
        )
        self.assertIn("[CAUSAL]", out1)
        self.assertIn("causal_abc", cfg._causal_transitions_surfaced_this_run)

        # Provider returns the SAME item on the next turn → must be dedup'd.
        out2 = cfg._provide_causal_memory_per_turn(
            command_text="next cmd", sandbox_res="", return_code=1,
            classified_error=None, turn=2,
        )
        self.assertEqual(out2, "")

    def test_new_transition_after_dedup_still_surfaces(self):
        item_a = self._make_item("causal_a", "[CAUSAL] A line")
        item_b = self._make_item("causal_b", "[CAUSAL] B line")
        cfg = self._make_configuration_stub([item_a])

        out1 = cfg._provide_causal_memory_per_turn(
            command_text="cmd1", sandbox_res="", return_code=1,
            classified_error=None, turn=1,
        )
        self.assertIn("[CAUSAL] A line", out1)

        # Provider now returns BOTH items. A is dedup'd, B is new.
        cfg.memory_provider.provide_causal_memory.return_value = [item_a, item_b]
        out2 = cfg._provide_causal_memory_per_turn(
            command_text="cmd2", sandbox_res="", return_code=1,
            classified_error=None, turn=2,
        )
        self.assertNotIn("[CAUSAL] A line", out2)
        self.assertIn("[CAUSAL] B line", out2)
        self.assertEqual(
            cfg._causal_transitions_surfaced_this_run,
            {"causal_a", "causal_b"},
        )

    def test_empty_retrieval_does_not_pollute_dedup_set(self):
        cfg = self._make_configuration_stub([])

        out = cfg._provide_causal_memory_per_turn(
            command_text="x", sandbox_res="", return_code=0,
            classified_error=None, turn=1,
        )
        self.assertEqual(out, "")
        self.assertEqual(cfg._causal_transitions_surfaced_this_run, set())


class CausalDistillerTests(unittest.TestCase):
    def test_extracts_transition_from_failure_then_success_with_marker(self):
        tmp, kb_path, traj_path = _make_kb_paths()
        kb = KBStore(kb_path)
        traj = TrajectoryStore(traj_path)
        try:
            distiller = TrajectoryDistiller(kb, traj, llm=None)

            fp = BuildFingerprint(
                repo_id="user/repo",
                frameworks={"torch"},
                cuda_deps={"flash-attn"},
                build_system="setuptools",
            )
            attempt = BuildAttempt(
                id="att-1",
                repo_id="user/repo",
                fingerprint=fp,
                docker_image="rocm/pytorch:rocm7.2",
                gpu_arch="gfx942",
            )
            traj_records = [
                TrajectoryRecord(
                    repo_id="user/repo", attempt_id="att-1",
                    agent="configuration",
                    action_type="bash",
                    action_content="pip install flash-attn",
                    observation_raw=(
                        "ImportError: No module named flash_attn_2_cuda\n"
                        "ERROR: failed building wheel for flash-attn\n"
                    ),
                    outcome="failure",
                    return_code=1,
                    error_class="FLASH_ATTN_CUDA_WHEEL",
                    novel_situation=False,
                    turn_number=1,
                ),
                TrajectoryRecord(
                    repo_id="user/repo", attempt_id="att-1",
                    agent="configuration",
                    action_type="bash",
                    action_content=(
                        "git clone https://github.com/Dao-AILab/flash-attention.git "
                        "/tmp/flash-attention && cd /tmp/flash-attention && "
                        "FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE python setup.py install"
                    ),
                    observation_raw=(
                        "Successfully installed flash-attn-2.5.0\n"
                        "import flash_attn passed\n"
                    ),
                    outcome="success",
                    return_code=0,
                    error_class=None,
                    kb_rules_applied=["rule_flash_attn_triton_amd"],
                    turn_number=2,
                ),
                TrajectoryRecord(
                    repo_id="user/repo", attempt_id="att-1",
                    agent="configuration",
                    action_type="bash",
                    action_content="echo ROCM_ENV_VERIFIED",
                    observation_raw="ROCM_ENV_VERIFIED",
                    outcome="success",
                    return_code=0,
                    turn_number=3,
                ),
            ]

            transitions = distiller.extract_causal_transitions(
                traj_records, attempt,
            )
            self.assertEqual(len(transitions), 1)
            t = transitions[0]
            self.assertEqual(t.state.error_class, "FLASH_ATTN_CUDA_WHEEL")
            self.assertEqual(
                t.transition_class, "cuda_only_wheel_to_rocm_source_build",
            )
            self.assertEqual(t.action.type, "package_strategy")
            self.assertIn("FLASH_ATTENTION_TRITON_AMD_ENABLE", t.action.command)
            self.assertEqual(t.outcome.return_code, 0)
            self.assertEqual(t.outcome.degradation, "D0")  # no flags supplied
            self.assertIn(
                "kb_rule:rule_flash_attn_triton_amd",
                t.action.evidence,
            )
            self.assertEqual(t.state.image, "rocm/pytorch:rocm7.2")
            self.assertEqual(t.state.gpu_arch, "gfx942")

        finally:
            kb.close()
            traj.close()

    def test_no_transition_without_env_verified_marker(self):
        tmp, kb_path, traj_path = _make_kb_paths()
        kb = KBStore(kb_path)
        traj = TrajectoryStore(traj_path)
        try:
            distiller = TrajectoryDistiller(kb, traj, llm=None)
            attempt = BuildAttempt(id="att-2", repo_id="user/repo",
                                   docker_image="rocm/pytorch")
            records = [
                TrajectoryRecord(
                    attempt_id="att-2", action_type="bash",
                    action_content="pip install flash-attn",
                    observation_raw="ImportError: flash_attn_2_cuda",
                    return_code=1, error_class="FLASH_ATTN_CUDA_WHEEL",
                    turn_number=1,
                ),
                TrajectoryRecord(
                    attempt_id="att-2", action_type="bash",
                    action_content=(
                        "FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE python setup.py install"
                    ),
                    observation_raw="Successfully installed flash-attn",
                    return_code=0, turn_number=2,
                ),
            ]
            transitions = distiller.extract_causal_transitions(records, attempt)
            self.assertEqual(transitions, [])
        finally:
            kb.close()
            traj.close()


class CausalMemoryProviderTests(unittest.TestCase):
    def _make_provider(self, kb_path: str, traj_path: str):
        kb = KBStore(kb_path)
        seed_if_empty(kb)
        seed_causal_transitions(kb)
        traj = TrajectoryStore(traj_path)
        cls = ErrorClassifier(kb)
        rule_engine = RuleEngine(kb)
        provider = BuildMemoryProvider(kb, traj, cls, rule_engine)
        return kb, traj, provider

    def test_format_causal_transition_renders_structured_line(self):
        t = CausalTransition(
            transition_class="cuda_only_wheel_to_rocm_source_build",
            state=CausalState(
                image="rocm/pytorch", gpu_arch="gfx942",
                error_class="cuda_only_wheel",
                degradation_policy="strict",
            ),
            action=CausalAction(
                type="package_strategy",
                command="FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE python setup.py install",
            ),
            outcome=CausalOutcome(return_code=0, degradation="D1",
                                  confidence=0.82),
            counterfactuals=[
                {
                    "action": "pip install flash-attn",
                    "expected_outcome": "fail",
                    "reason": "CUDA-only wheel",
                }
            ],
        )
        line = format_causal_transition(t)
        self.assertTrue(line.startswith("[CAUSAL] "))
        self.assertIn("img=rocm/pytorch", line)
        self.assertIn("arch=gfx942", line)
        self.assertIn("err=cuda_only_wheel", line)
        self.assertIn("→ action{", line)
        self.assertIn("FLASH_ATTENTION_TRITON_AMD_ENABLE", line)
        self.assertIn("→ outcome{", line)
        self.assertIn("ok", line)
        self.assertIn("D1", line)

        cfs = format_causal_counterfactuals(t)
        self.assertTrue(any("[counterfactual:" in c for c in cfs))
        self.assertTrue(any("pip install flash-attn" in c for c in cfs))

    def test_provide_causal_memory_returns_items_for_known_error(self):
        tmp, kb_path, traj_path = _make_kb_paths()
        kb, traj, provider = self._make_provider(kb_path, traj_path)
        try:
            request = MemoryRequest(
                query="pip install flash-attn",
                context={
                    "rocm_mode": True,
                    "image": "rocm/pytorch",
                    "gpu_arch": "gfx942",
                    "error_class": "FLASH_ATTN_CUDA_WHEEL",
                    "degradation_policy": "strict",
                },
                phase=MemoryPhase.IN.value,
                current_error=(
                    "ImportError: No module named flash_attn_2_cuda\n"
                ),
                turn_number=1,
            )
            items = provider.provide_causal_memory(request, top_k=3)
            self.assertGreaterEqual(len(items), 1)
            top = items[0]
            self.assertEqual(top.item_type, "causal")
            self.assertTrue(top.content.startswith("[CAUSAL] "))
            self.assertIn("err=FLASH_ATTN_CUDA_WHEEL", top.content)
            self.assertTrue(top.executable)
            self.assertIn(
                "FLASH_ATTENTION_TRITON_AMD_ENABLE", top.commands[0]
            )
            self.assertIn("[counterfactual:", top.content)
        finally:
            kb.close()
            traj.close()

    def test_in_phase_response_includes_causal_items(self):
        tmp, kb_path, traj_path = _make_kb_paths()
        kb, traj, provider = self._make_provider(kb_path, traj_path)
        try:
            request = MemoryRequest(
                query="pip install flash-attn",
                context={
                    "rocm_mode": True,
                    "image": "rocm/pytorch",
                    "gpu_arch": "gfx942",
                    "error_class": "FLASH_ATTN_CUDA_WHEEL",
                },
                phase=MemoryPhase.IN.value,
                current_error="ImportError: No module named flash_attn_2_cuda",
                turn_number=2,
            )
            response = provider.provide_memory(request)
            causal_items = [
                i for i in response.items if i.item_type == "causal"
            ]
            self.assertGreaterEqual(len(causal_items), 1)
            obs_text = provider.format_in_for_observation(response)
            self.assertIn("[CAUSAL]", obs_text)

            begin_request = MemoryRequest(
                query="user/repo",
                context={
                    "rocm_mode": True,
                    "image": "rocm/pytorch",
                    "gpu_arch": "gfx942",
                },
                phase=MemoryPhase.BEGIN.value,
                fingerprint=BuildFingerprint(
                    repo_id="user/repo",
                    frameworks={"torch"},
                    cuda_deps={"flash-attn"},
                    build_system="setuptools",
                ),
            )
            begin_response = provider.provide_memory(begin_request)
            self.assertGreaterEqual(
                len([i for i in begin_response.items
                     if i.item_type == "causal"]),
                1,
            )
            begin_text = provider.format_begin_for_prompt(begin_response)
            self.assertIn("[CAUSAL]", begin_text)
        finally:
            kb.close()
            traj.close()


class CausalSeedingTests(unittest.TestCase):
    def test_seed_runs_only_when_table_empty(self):
        tmp, kb_path, _ = _make_kb_paths()
        kb = KBStore(kb_path)
        try:
            self.assertEqual(kb.count_transitions(), 0)
            inserted = seed_causal_transitions(kb)
            self.assertGreater(inserted, 0)
            seeded_count = kb.count_transitions()
            self.assertEqual(seeded_count, inserted)

            # Re-running must be a no-op.
            inserted_again = seed_causal_transitions(kb)
            self.assertEqual(inserted_again, 0)
            self.assertEqual(kb.count_transitions(), seeded_count)
        finally:
            kb.close()

    def test_seed_includes_required_transition_classes(self):
        tmp, kb_path, _ = _make_kb_paths()
        kb = KBStore(kb_path)
        try:
            seed_causal_transitions(kb)
            classes = set()
            # Pull all rows back through the public query path.
            results = kb.query_transitions(CausalState(), top_k=100)
            for t in results:
                classes.add(t.transition_class)
            for required in (
                "cuda_only_wheel_to_rocm_source_build",
                "wrong_image_to_ranked_image_switch",
                "missing_gpu_runtime_to_rocm_base_image",
                "custom_cuda_compile_error_to_hipify_fix",
                "paper_metric_mismatch_to_not_reproduced",
            ):
                self.assertIn(required, classes)
        finally:
            kb.close()


if __name__ == "__main__":
    unittest.main()
