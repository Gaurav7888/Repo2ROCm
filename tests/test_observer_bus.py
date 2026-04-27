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

from observers.observer_agent import (  # noqa: E402
    HazardLedgerBuilder,
    ObserverClient,
    ObserverSidecar,
    StateInterpreter,
    TrajectoryForecaster,
)
from observers.types import (  # noqa: E402
    HazardSignal,
    ObserverAdvice,
    TurnState,
    append_jsonl,
)


def _snapshot(turn, commands, return_codes=(0,), error_class="",
              stage="stage1", observation="", paper_retrieval_used=False):
    return {
        "turn": turn,
        "stage": stage,
        "commands": list(commands),
        "return_codes": list(return_codes),
        "error_class": error_class,
        "action_type": "bash" if commands else "none",
        "observation_excerpt": observation,
        "paper_retrieval_used": paper_retrieval_used,
        "duration_s": 1.0,
    }


class ObserverBusTests(unittest.TestCase):
    def test_consume_new_advice_filters_expired_packs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = ObserverClient(
                output_dir=tmpdir,
                llm="claude-sonnet-4",
                enabled=True,
            )
            open(client.events_path, "w", encoding="utf-8").close()
            open(client.advice_path, "w", encoding="utf-8").close()

            # Fresh, high-priority preventive pack — should be returned first.
            append_jsonl(client.advice_path, ObserverAdvice.create(
                turn_seen=3,
                profile_used="modelAssetReadiness",
                diagnosis="Benchmark expects /models/...",
                recommended_strategy="Pre-download a compatible Qwen2 checkpoint.",
                kind="preventive",
                predicted_failure="missing_model_asset_on_benchmark_launch",
                applies_before="benchmark_run",
                priority="high",
                expires_after_turn=99,
            ))
            # Stale pack — should be filtered out at turn 10.
            append_jsonl(client.advice_path, ObserverAdvice.create(
                turn_seen=1,
                profile_used="dependencyPreflight",
                diagnosis="flash-attn pin",
                recommended_strategy="Avoid the CUDA-only wheel.",
                kind="preventive",
                predicted_failure="cuda_only_wheel_on_amd",
                applies_before="dependency_install",
                priority="normal",
                expires_after_turn=2,
            ))

            fresh = client.consume_new_advice(current_turn=10)
            second = client.consume_new_advice(current_turn=10)

        self.assertEqual(len(fresh), 1, "Expected the expired pack to be filtered")
        self.assertEqual(fresh[0]["profile_used"], "modelAssetReadiness")
        self.assertEqual(second, [], "Already-consumed packs must not return again")


class StateInterpreterTests(unittest.TestCase):
    def test_classifies_dependency_install_failure(self):
        interpreter = StateInterpreter()
        snapshot = _snapshot(
            turn=4,
            commands=["pip install flash-attn"],
            return_codes=[1],
            error_class="InstallError",
            observation="ERROR: Could not find a version that satisfies flash-attn",
        )
        state = interpreter.interpret(snapshot)
        self.assertEqual(state.action_family, "dependency_install")
        self.assertFalse(state.succeeded)
        self.assertEqual(state.blocked_on, "pip_version_conflict")
        self.assertIn("install", state.dependency_signals)

    def test_classifies_benchmark_run(self):
        interpreter = StateInterpreter()
        snapshot = _snapshot(
            turn=8,
            commands=[
                "python benchmark/math_bench/pred.py --model qwen --model_path /models/x"
            ],
            return_codes=[0],
        )
        state = interpreter.interpret(snapshot)
        self.assertEqual(state.action_family, "benchmark_run")
        self.assertTrue(state.succeeded)


class HazardLedgerTests(unittest.TestCase):
    def test_builds_hazards_from_run_context(self):
        builder = HazardLedgerBuilder()
        ledger = builder.build({
            "repo": "owner/Sparse-vLLM",
            "reproduce_results": True,
            "plan_excerpt": (
                "Use rocm/pytorch:latest. Install transformers. "
                "Build flash-attn from source. Run benchmark with model_path /models/qwen."
            ),
        })
        skills = {hazard.skill for hazard in ledger.hazards}
        self.assertIn("dependencyPreflight", skills)
        self.assertIn("rocmRuntimeCompatibility", skills)
        self.assertIn("modelAssetReadiness", skills)
        self.assertIn("frameworkApiDrift", skills)
        self.assertIn("paperMetricPath", skills)


class TrajectoryForecasterTests(unittest.TestCase):
    def test_predicts_model_asset_failure_before_benchmark(self):
        forecaster = TrajectoryForecaster()
        history = [
            TurnState(turn=1, stage="stage1", action_family="dependency_install",
                      action_target="transformers", succeeded=True),
            TurnState(turn=2, stage="stage1", action_family="inspect",
                      action_target="ls", succeeded=True,
                      dependency_signals=["transformers"]),
        ]
        forecast = forecaster.forecast(history, ledger=None)
        self.assertIsNotNone(forecast)
        self.assertEqual(forecast.predicted_next_action_family, "benchmark_run")
        self.assertIn("missing_model_asset_on_benchmark_launch",
                      forecast.predicted_failures)


class SidecarPlanningTests(unittest.TestCase):
    @mock.patch(
        "agents.researcher.research",
        return_value={
            "answer": "Pre-download a compatible Qwen2 checkpoint matching the repo's expected family before launching the benchmark.",
            "suggested_commands": [],
            "citations": [{"title": "Qwen2.5 0.5B Instruct", "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct"}],
            "confidence": 0.78,
            "profile_used": "observerCritic",
            "followups": ["Confirm /models path or override --model_path before benchmark"],
        },
    )
    def test_sidecar_emits_preventive_pack_before_benchmark(self, _mocked):
        sidecar = ObserverSidecar(
            events_path="/tmp/__obs_events.jsonl",
            advice_path="/tmp/__obs_advice.jsonl",
            llm="claude-sonnet-4",
        )
        # Skip preventive throttle so the test is deterministic.
        sidecar._last_research_at = 0.0

        sidecar._on_run_started({
            "repo": "CURRENTF/Sparse-vLLM",
            "reproduce_results": True,
            "plan_excerpt": (
                "Run benchmark with --model_path /models/qwen. Install transformers. "
                "rocm/pytorch:latest. Build flash-attn from source."
            ),
        })

        # Two prior turns of inspection + dependency install set up the predicted
        # next benchmark launch.
        sidecar._on_turn_snapshot(_snapshot(
            turn=1, commands=["pip install transformers"], return_codes=[0],
        ))
        sidecar._on_turn_snapshot(_snapshot(
            turn=2,
            commands=["ls /repo"],
            return_codes=[0],
            observation="repo listed",
        ))

        # Read the advice file to confirm at least one preventive pack landed.
        if not os.path.exists(sidecar.advice_path):
            self.fail("Advice file was never written")
        with open(sidecar.advice_path, "r", encoding="utf-8") as handle:
            rows = [line for line in handle if line.strip()]
        self.assertGreaterEqual(len(rows), 1)
        # Cleanup the temp advice file.
        try:
            os.remove(sidecar.advice_path)
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main()
