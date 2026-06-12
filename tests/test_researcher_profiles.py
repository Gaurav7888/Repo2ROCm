import os
import sys
import unittest
from unittest import mock


BUILD_AGENT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "build_agent",
)
if BUILD_AGENT_ROOT not in sys.path:
    sys.path.insert(0, BUILD_AGENT_ROOT)

from agents import researcher  # noqa: E402


class ResearcherProseRecoveryTests(unittest.TestCase):
    def test_prose_recovery_extracts_answer_commands_followups_citations(self):
        prose = (
            "On AMD ROCm, the upstream PyPI flash-attn wheels are CUDA-only.\n"
            "You should build from source using the Triton AMD path.\n\n"
            "Suggested commands:\n"
            "$ FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE pip install flash-attn --no-build-isolation\n"
            "$ docker pull rocm/pytorch:latest\n\n"
            "Citations:\n"
            "- flash-attention AMD ROCm support https://github.com/Dao-AILab/flash-attention\n\n"
            "Followups:\n"
            "- Confirm ROCm version compatibility\n"
            "- Verify torch HIP build exists in the image\n"
        )
        recovered = researcher._scrape_prose_recovery(prose)
        self.assertTrue(recovered["answer"].startswith("On AMD ROCm"))
        self.assertGreaterEqual(len(recovered["suggested_commands"]), 2)
        self.assertGreaterEqual(len(recovered["citations"]), 1)
        self.assertGreaterEqual(len(recovered["followups"]), 2)

    def test_safe_finish_uses_prose_when_json_fails(self):
        prose = (
            "AMD Claude returned prose instead of JSON. The advice is to "
            "pre-download a Qwen2 0.5B Instruct checkpoint before launching "
            "the benchmark, since /models/* is empty in the container."
        )
        note = researcher._safe_finish(prose, {})
        self.assertIn("pre-download", note["answer"])
        self.assertGreater(note["confidence"], 0.0)

    def test_safe_finish_prefers_json_when_present(self):
        reply = (
            'Some preamble that should be ignored.\n'
            '{"answer": "use rocm/pytorch:latest", '
            '"suggested_commands": ["pip install -U torch"], '
            '"citations": [], "confidence": 0.7, "followups": []}\n'
            'and trailing prose.'
        )
        note = researcher._safe_finish(reply, {})
        self.assertEqual(note["answer"], "use rocm/pytorch:latest")
        self.assertIn("pip install -U torch", note["suggested_commands"])
        self.assertAlmostEqual(note["confidence"], 0.7, places=5)


class ResearcherProfileTests(unittest.TestCase):
    def test_profile_resolution_is_case_insensitive(self):
        profile = researcher._resolve_profile("paperresearch")
        self.assertEqual(profile.name, "paperResearch")

    def test_research_returns_profile_and_followups(self):
        with mock.patch.object(
            researcher,
            "_gather_evidence",
            return_value=("evidence block", [{"title": "Doc", "url": "https://example.com"}], 2),
        ), mock.patch.object(
            researcher,
            "_cache_get",
            return_value=None,
        ), mock.patch.object(
            researcher,
            "_cache_put",
            return_value=None,
        ), mock.patch(
            "utils.llm.get_llm_response",
            return_value=(
                [
                    '{"answer":"Use the repo-backed metric.","suggested_commands":[],"citations":[{"title":"Doc","url":"https://example.com"}],"confidence":0.8,"followups":["Check appendix tolerance"]}'
                ],
                {"total_tokens": 123},
            ),
        ):
            note = researcher.research(
                "Which metric should we trust?",
                llm="claude-sonnet-4",
                profile="paperResearch",
                use_cache=False,
            )

        self.assertEqual(note["profile_used"], "paperResearch")
        self.assertEqual(note["followups"], ["Check appendix tolerance"])
        self.assertEqual(note["citations"][0]["url"], "https://example.com")


if __name__ == "__main__":
    unittest.main()
