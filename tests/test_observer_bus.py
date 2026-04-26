import os
import sys
import tempfile
import unittest
from unittest import mock


BUILD_AGENT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "build_agent",
)
if BUILD_AGENT_ROOT not in sys.path:
    sys.path.insert(0, BUILD_AGENT_ROOT)

from observers.observer_agent import ObserverClient, ObserverSidecar  # noqa: E402
from observers.types import ObserverAdvice, append_jsonl  # noqa: E402


class ObserverBusTests(unittest.TestCase):
    def test_consume_new_advice_is_one_time_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = ObserverClient(
                output_dir=tmpdir,
                llm="claude-sonnet-4",
                enabled=True,
            )
            open(client.events_path, "w", encoding="utf-8").close()
            open(client.advice_path, "w", encoding="utf-8").close()

            append_jsonl(
                client.advice_path,
                ObserverAdvice.create(
                    turn_seen=3,
                    profile_used="dependencyRepair",
                    diagnosis="Repeated package retries with no new evidence.",
                    recommended_strategy="Switch to verified version lookup before retrying.",
                    suggested_questions_or_tools=["pypi_versions flash-attn"],
                    confidence=0.77,
                    evidence=["turn 2 and turn 3 ended with the same failure"],
                ),
            )

            first = client.consume_new_advice()
            second = client.consume_new_advice()

        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]["profile_used"], "dependencyRepair")
        self.assertEqual(second, [])

    @mock.patch(
        "agents.researcher.research",
        return_value={
            "answer": "The run is stuck retrying dependency installs without new compatibility evidence.",
            "suggested_commands": [],
            "citations": [{"title": "PyPI", "url": "https://pypi.org/project/flash-attn/"}],
            "confidence": 0.74,
            "profile_used": "observerCritic",
            "followups": ["Use pypi_versions flash-attn before the next retry"],
        },
    )
    def test_sidecar_uses_research_worker_for_advice(self, mocked_research):
        sidecar = ObserverSidecar(
            events_path="/tmp/observer-events.jsonl",
            advice_path="/tmp/observer-advice.jsonl",
            llm="claude-sonnet-4",
        )
        sidecar._run_context = {"repo": "owner/repo"}
        sidecar._recent_turns = [
            {
                "turn": 1,
                "stage": "stage1",
                "commands": ["pip install flash-attn"],
                "return_codes": [1],
                "error_class": "InstallError",
                "action_type": "bash",
                "observation_excerpt": "build failed",
                "paper_retrieval_used": False,
            },
            {
                "turn": 2,
                "stage": "stage1",
                "commands": ["pip install flash-attn"],
                "return_codes": [1],
                "error_class": "InstallError",
                "action_type": "bash",
                "observation_excerpt": "same failure again",
                "paper_retrieval_used": False,
            },
        ]

        advice = sidecar._evaluate_recent_turns(turn_seen=2)

        self.assertIsNotNone(advice)
        self.assertEqual(advice.profile_used, "dependencyRepair")
        mocked_research.assert_called_once()


if __name__ == "__main__":
    unittest.main()
