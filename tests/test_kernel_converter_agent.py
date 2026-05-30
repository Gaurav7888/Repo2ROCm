"""Tests for the Track 2 kernel converter agent.

Covers (per docs/next_research_plan.md acceptance criteria):
    * Trigger conditions (has_custom_cuda_kernels / .cu detection / observation)
    * Sandbox executor adapter (CommandResult round-trip)
    * Report serialization (KernelMigrationReport.to_dict round-trip)
    * Sub-agent task packet generation (allowed_scope / forbidden)
    * End-to-end dry run on a synthetic repo (kernel_migration_report.json
      written; compile_passed False because hipcc is mocked).

No network calls. No real hipify/hipcc invocation.
"""

from __future__ import annotations

import json
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


from kernel_migration.executor_adapter import DryRunExecutor, SandboxExecutor  # noqa: E402
from kernel_migration.scaffold import (  # noqa: E402
    CommandResult,
    FixSuggestion,
    KernelCandidate,
    discover_cuda_sources,
)
from agents.kernel_converter_agent import (  # noqa: E402
    ALLOWED_SCOPE,
    FORBIDDEN_ACTIONS,
    KernelConverterAgent,
    build_subagent_task_packet,
    looks_like_cuda_compile_error,
    repo_has_cuda_sources,
)
from storage.models import KernelMigrationReport  # noqa: E402


CUDA_SAMPLE = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void fused_attention_kernel(float* x) {
  int lane = threadIdx.x % 32; // warp lane
  asm volatile("bar.sync 0;");
  if (__CUDA_ARCH__ >= 800) {
    x[threadIdx.x] = __shfl_sync(0xffffffff, x[threadIdx.x], lane);
  }
}
"""


class FakeSession:
    """Minimal stand-in for ``Sandbox.get_session()`` for the adapter test."""

    def __init__(self, scripted):
        self._scripted = list(scripted)
        self.executed = []
        self._last_rc = 0

    def execute_simple(self, command, timeout=86400):
        self.executed.append((command, timeout))
        if not self._scripted:
            self._last_rc = 0
            return True, ""
        ok, output, rc = self._scripted.pop(0)
        self._last_rc = int(rc)
        return bool(ok), output

    def get_returncode(self):
        return self._last_rc


# ── Trigger conditions ───────────────────────────────────────────────────────


class TriggerConditionsTests(unittest.TestCase):
    def test_repo_has_cuda_sources_true_for_cu_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            kernel_dir = os.path.join(tmp, "csrc")
            os.makedirs(kernel_dir, exist_ok=True)
            with open(os.path.join(kernel_dir, "kernel.cu"), "w") as f:
                f.write(CUDA_SAMPLE)
            self.assertTrue(repo_has_cuda_sources(tmp))

    def test_repo_has_cuda_sources_false_when_no_cu(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "model.py"), "w") as f:
                f.write("import torch\n")
            self.assertFalse(repo_has_cuda_sources(tmp))

    def test_repo_has_cuda_sources_skip_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            skip = os.path.join(tmp, "build")
            os.makedirs(skip, exist_ok=True)
            with open(os.path.join(skip, "kernel.cu"), "w") as f:
                f.write(CUDA_SAMPLE)
            self.assertFalse(repo_has_cuda_sources(tmp))

    def test_looks_like_cuda_compile_error_matches(self):
        observations = [
            "fatal error: cuda_runtime.h: No such file or directory",
            "undefined reference to `cudaMalloc'",
            "identifier 'cudaError_t' is undefined",
            "nvcc: command not found",
            "cublas_v2.h not found",
        ]
        for obs in observations:
            with self.subTest(obs=obs):
                self.assertTrue(looks_like_cuda_compile_error(obs))

    def test_looks_like_cuda_compile_error_negative(self):
        self.assertFalse(looks_like_cuda_compile_error(""))
        self.assertFalse(looks_like_cuda_compile_error(
            "ImportError: No module named 'numpy'"
        ))


# ── Sandbox executor adapter ────────────────────────────────────────────────


class ExecutorAdapterTests(unittest.TestCase):
    def test_dry_run_records_commands(self):
        ex = DryRunExecutor()
        r1 = ex("hipify-clang --examine /repo/kernel.cu")
        r2 = ex("hipcc -c /repo/kernel.hip.cpp -o /tmp/kernel.o")
        self.assertEqual(len(ex.commands), 2)
        self.assertEqual(r1.return_code, 0)
        self.assertEqual(r2.command, "hipcc -c /repo/kernel.hip.cpp -o /tmp/kernel.o")
        self.assertTrue(r1.ok)

    def test_dry_run_with_factory(self):
        ex = DryRunExecutor(stdout_factory=lambda cmd: "OK" if "examine" in cmd else "")
        r = ex("hipify-clang --examine x.cu")
        self.assertEqual(r.stdout, "OK")
        self.assertEqual(ex(""), ex.results[-1])

    def test_sandbox_executor_round_trip(self):
        session = FakeSession(scripted=[
            (True, "hipify ok", 0),
            (False, "compile failed", 1),
        ])
        adapter = SandboxExecutor(session, timeout=42)
        r1 = adapter("hipify-clang --examine x.cu")
        r2 = adapter("hipcc -c x.hip.cpp -o x.o")
        self.assertIsInstance(r1, CommandResult)
        self.assertEqual(r1.return_code, 0)
        self.assertEqual(r1.stdout, "hipify ok")
        self.assertEqual(r2.return_code, 1)
        self.assertEqual(session.executed[0][0], "hipify-clang --examine x.cu")
        self.assertEqual(session.executed[0][1], 42)

    def test_sandbox_executor_swallows_exceptions(self):
        class BrokenSession:
            def execute_simple(self, command, timeout=86400):
                raise RuntimeError("pipe closed")

            def get_returncode(self):
                raise RuntimeError("nope")

        adapter = SandboxExecutor(BrokenSession())
        result = adapter("anything")
        self.assertEqual(result.return_code, 124)
        self.assertIn("pipe closed", result.stderr)


# ── Report serialization ────────────────────────────────────────────────────


class ReportSerializationTests(unittest.TestCase):
    def test_round_trip(self):
        report = KernelMigrationReport(
            repo_id="user/repo",
            attempt_id="abc-123",
            n_kernels=2,
            kernels_examined=2,
            kernels_applied=1,
            compile_passed=0,
            compile_failed=1,
            manual_fix_count=2,
            degradation="D2",
            status="manual_fixes_required",
            risk_flags=["inline_ptx"],
            errors=["err1"],
            granular_fixes_applied=[
                {"issue": "cuda_runtime_header", "file": "k.cu", "status": "applied"}
            ],
            evidence=["inventory: 2 sources"],
        )
        d = report.to_dict()
        self.assertEqual(d["status"], "manual_fixes_required")
        self.assertEqual(d["degradation"], "D2")
        clone = KernelMigrationReport.from_dict(d)
        self.assertEqual(clone.to_dict(), d)

    def test_invalid_status_normalizes(self):
        report = KernelMigrationReport(status="garbage", degradation="D9")
        self.assertEqual(report.status, "no_kernels")
        self.assertEqual(report.degradation, "D0")


# ── Sub-agent task packet ───────────────────────────────────────────────────


class SubAgentTaskPacketTests(unittest.TestCase):
    def test_packet_shape_and_constants(self):
        candidate = KernelCandidate(
            path="src/kernels/attention.cu",
            risk_flags=["inline_ptx", "warp_size_assumption"],
        )
        suggestion = FixSuggestion(
            file_path="src/kernels/attention.cu",
            issue="inline_ptx",
            rationale="cannot transpile asm",
            patch_hint="replace asm block",
            requires_subagent=True,
        )
        packet = build_subagent_task_packet(
            candidate, suggestion,
            hipify_output="warning: unsupported asm",
            compile_error="error: cudaError_t undefined",
        )
        self.assertEqual(packet["candidate"], "src/kernels/attention.cu")
        self.assertEqual(packet["allowed_scope"], "correctness_only")
        self.assertEqual(packet["allowed_scope"], ALLOWED_SCOPE)
        self.assertEqual(packet["forbidden"], FORBIDDEN_ACTIONS)
        for forbidden in ("performance tuning", "large rewrites", "mock success"):
            self.assertIn(forbidden, packet["forbidden"])
        self.assertEqual(set(packet["risk_flags"]), {"inline_ptx", "warp_size_assumption"})
        self.assertIn("warning: unsupported asm", packet["hipify_output"])
        self.assertIn("cudaError_t", packet["compile_error"])
        # JSON-serializable.
        encoded = json.dumps(packet)
        self.assertIn("correctness_only", encoded)


# ── End-to-end dry run ──────────────────────────────────────────────────────


class EndToEndDryRunTests(unittest.TestCase):
    def test_synthetic_repo_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            kernel_dir = os.path.join(tmp, "csrc")
            os.makedirs(kernel_dir, exist_ok=True)
            kernel_path = os.path.join(kernel_dir, "attention.cu")
            with open(kernel_path, "w") as f:
                f.write(CUDA_SAMPLE)

            report_dir = os.path.join(tmp, "out")
            os.makedirs(report_dir, exist_ok=True)

            executor = DryRunExecutor()
            agent = KernelConverterAgent(
                executor=executor,
                repo_root=tmp,
                llm=None,
                dry_run=True,
                report_dir=report_dir,
                repo_id="user/repo",
                attempt_id="att-1",
            )
            report = agent.run()

            # Report fields
            self.assertEqual(report.n_kernels, 1)
            self.assertEqual(report.compile_passed, 0)  # hipcc never ran
            self.assertIn(report.status, ("hipify_planned", "no_kernels"))
            self.assertEqual(report.status, "hipify_planned")
            self.assertGreaterEqual(len(report.granular_fixes_applied), 1)
            issues = {f.get("issue") for f in report.granular_fixes_applied}
            self.assertIn("cuda_runtime_header", issues)

            # JSON artifact
            artifact = os.path.join(report_dir, "kernel_migration_report.json")
            self.assertTrue(os.path.exists(artifact))
            data = json.loads(open(artifact).read())
            self.assertEqual(data["repo_id"], "user/repo")
            self.assertEqual(data["attempt_id"], "att-1")
            self.assertEqual(data["status"], "hipify_planned")
            self.assertEqual(data["n_kernels"], 1)

            # The dry-run executor should have planned hipify AND hipcc
            # commands, but never executed real ones. The verification compile is
            # now a correctness-only syntax check (`hipcc -fsyntax-only`) carrying
            # torch/ATen includes + HIP defines, not a bare `hipcc -c`.
            joined = "\n".join(executor.commands)
            self.assertIn("hipify-clang --examine", joined)
            self.assertIn("hipcc -fsyntax-only", joined)

    def test_no_kernels_short_circuits(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "model.py"), "w") as f:
                f.write("import torch\n")
            report_dir = os.path.join(tmp, "out")
            executor = DryRunExecutor()
            agent = KernelConverterAgent(
                executor=executor,
                repo_root=tmp,
                dry_run=True,
                report_dir=report_dir,
                repo_id="user/repo",
                attempt_id="att-empty",
            )
            report = agent.run()
            self.assertEqual(report.status, "no_kernels")
            self.assertEqual(report.n_kernels, 0)
            self.assertEqual(executor.commands, [])
            artifact = os.path.join(report_dir, "kernel_migration_report.json")
            self.assertTrue(os.path.exists(artifact))

    def test_unsupported_when_toolchain_missing_in_real_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            kernel_dir = os.path.join(tmp, "csrc")
            os.makedirs(kernel_dir, exist_ok=True)
            with open(os.path.join(kernel_dir, "k.cu"), "w") as f:
                f.write(CUDA_SAMPLE)

            class MissingToolchainExecutor:
                def __init__(self):
                    self.commands = []

                def __call__(self, command):
                    self.commands.append(command)
                    return CommandResult(command=command, return_code=1)

            ex = MissingToolchainExecutor()
            agent = KernelConverterAgent(
                executor=ex,
                repo_root=tmp,
                dry_run=False,
                report_dir=tmp,
            )
            report = agent.run()
            self.assertEqual(report.status, "unsupported")
            self.assertEqual(report.degradation, "D4")


# ── success_report integration ──────────────────────────────────────────────


class SuccessReportIntegrationTests(unittest.TestCase):
    def test_kernel_migration_section_promoted(self):
        from storage.success_report import build_success_report

        km = KernelMigrationReport(
            n_kernels=1,
            kernels_examined=1,
            status="manual_fixes_required",
            degradation="D2",
            manual_fix_count=1,
        ).to_dict()

        sr = build_success_report(
            final_verdict="unknown",
            verifier_record=None,
            chosen_experiment=None,
            gpu_check_seen=True,
            stage1_marker_emitted=True,
            turns_used=12,
            tool_calls={},
            outer_commands=[],
            kernel_migration=km,
        )
        self.assertEqual(sr["kernel_migration_status"], "manual_fixes_required")
        self.assertEqual(sr["kernel_migration_degradation"], "D2")
        self.assertEqual(sr["kernel_migration"]["status"], "manual_fixes_required")
        self.assertEqual(sr["kernel_migration"]["raw"]["n_kernels"], 1)


if __name__ == "__main__":
    unittest.main()
