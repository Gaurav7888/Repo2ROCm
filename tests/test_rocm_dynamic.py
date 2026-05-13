import os
import sys
import unittest


BUILD_AGENT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "build_agent",
)
if BUILD_AGENT_ROOT not in sys.path:
    sys.path.insert(0, BUILD_AGENT_ROOT)

from knowledge.rocm_dynamic import (  # noqa: E402
    choose_live_tag,
    infer_degradation_policy,
    infer_model_stack,
    package_guidance_for,
)


class RocmDynamicTests(unittest.TestCase):
    def test_choose_live_tag_prefers_newest_rocm_pytorch_release(self):
        tag_info = choose_live_tag(
            "rocm/pytorch",
            static_default="latest",
            preferred_python="3.10",
            live_tags=[
                {"name": "latest", "last_updated": "2026-01-01T00:00:00Z"},
                {"name": "rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1", "last_updated": "2026-01-01T00:00:00Z"},
                {"name": "rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.1", "last_updated": "2026-01-01T00:00:00Z"},
                {"name": "rocm6.3_ubuntu22.04_py3.10_pytorch_release_2.4.0", "last_updated": "2025-01-01T00:00:00Z"},
            ],
        )

        self.assertEqual(
            tag_info["tag"],
            "rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.1",
        )
        self.assertEqual(tag_info["source"], "dockerhub_live")

    def test_model_stack_detects_llm_serving(self):
        stack = infer_model_stack(
            {"vllm": 3, "torch": 8, "transformers": 4},
            {"requirements.txt": "vllm\ntransformers\n"},
        )

        self.assertEqual(stack, "llm_serving")

    def test_flash_attention_guidance_branches_by_arch_and_policy(self):
        strict_notes = package_guidance_for(
            "flash-attn",
            model_stack="transformers_inference",
            gpu_arch="gfx1100",
            degradation_policy="strict",
        )
        permissive_notes = package_guidance_for(
            "flash-attn",
            model_stack="transformers_inference",
            gpu_arch="unknown",
            degradation_policy="permissive",
        )

        self.assertTrue(any("RDNA" in note for note in strict_notes))
        self.assertTrue(any("strict mode" in note for note in strict_notes))
        self.assertTrue(any("SDPA fallback is acceptable" in note for note in permissive_notes))

    def test_reproduce_mode_uses_strict_degradation_policy(self):
        self.assertEqual(
            infer_degradation_policy(reproduce_results=True, run_mode="reproduce"),
            "strict",
        )
        self.assertEqual(
            infer_degradation_policy(reproduce_results=False, run_mode="env"),
            "permissive",
        )


if __name__ == "__main__":
    unittest.main()
