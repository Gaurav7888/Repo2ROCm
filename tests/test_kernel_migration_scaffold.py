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

from kernel_migration.scaffold import (  # noqa: E402
    GranularIssueFixer,
    KernelMigrationAgent,
    discover_cuda_sources,
)


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


class KernelMigrationScaffoldTests(unittest.TestCase):
    def test_discovers_cuda_candidate_and_risk_flags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kernel_dir = os.path.join(tmpdir, "kernels")
            os.makedirs(kernel_dir, exist_ok=True)
            with open(os.path.join(kernel_dir, "attention.cu"), "w", encoding="utf-8") as handle:
                handle.write(CUDA_SAMPLE)

            candidates = discover_cuda_sources(tmpdir)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].purpose, "attention")
        self.assertIn("cuda_runtime.h", candidates[0].includes)
        self.assertIn("inline_ptx", candidates[0].risk_flags)
        self.assertIn("cuda_arch_guard", candidates[0].risk_flags)

    def test_granular_fixer_surfaces_subagent_work_items(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "kernel.cu")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(CUDA_SAMPLE)
            candidate = discover_cuda_sources(tmpdir)[0]

            suggestions = GranularIssueFixer().suggest(
                candidate,
                CUDA_SAMPLE,
                "warning: unsupported inline asm needs review",
            )

        issues = {suggestion.issue for suggestion in suggestions}
        self.assertIn("cuda_runtime_header", issues)
        self.assertIn("cuda_fp16_header", issues)
        self.assertIn("inline_ptx", issues)
        self.assertIn("cuda_preprocessor_guards", issues)
        self.assertTrue(any(s.requires_subagent for s in suggestions))

    def test_agent_dry_run_produces_hipify_and_compile_plan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "kernel.cu"), "w", encoding="utf-8") as handle:
                handle.write(CUDA_SAMPLE)

            report = KernelMigrationAgent(tmpdir, dry_run=True).run()

        self.assertTrue(report.dry_run)
        self.assertEqual(len(report.candidates), 1)
        self.assertTrue(any("hipify-clang --examine" in cmd for cmd in report.commands_planned))
        self.assertTrue(any("hipify-perl" in cmd for cmd in report.commands_planned))
        self.assertTrue(any("hipcc -c" in cmd for cmd in report.compile_commands))
        self.assertGreaterEqual(len(report.fix_suggestions), 3)


if __name__ == "__main__":
    unittest.main()
