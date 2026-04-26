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
