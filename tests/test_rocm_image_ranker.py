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

from images.rocm_ranker import (  # noqa: E402
    ImageRankerConfig,
    RocmImageRanker,
    infer_preferred_workload,
)


def _fake_tags(image, limit=30):
    tags = {
        "rocm/pytorch": [
            {
                "name": "rocm7.2.3_ubuntu22.04_py3.10_pytorch_release_2.10.0",
                "full_size": 10 * 1024 ** 3,
                "last_updated": "2026-05-05T00:00:00Z",
                "digest": "sha256:pytorch",
            },
            {"name": "latest", "full_size": 10 * 1024 ** 3},
        ],
        "rocm/vllm": [
            {
                "name": "rocm7.12.0_gfx94X-dcgpu_ubuntu24.04_py3.12_pytorch_2.9.1_vllm_0.16.0",
                "full_size": 14 * 1024 ** 3,
                "last_updated": "2026-03-27T00:00:00Z",
                "digest": "sha256:vllm",
            }
        ],
        "rocm/sgl-dev": [
            {"name": "sglang-0.5.10-rocm720-mi35x", "full_size": 25 * 1024 ** 3}
        ],
    }
    return tags.get(image, [{"name": "latest", "full_size": 8 * 1024 ** 3}]), None


class RocmImageRankerTests(unittest.TestCase):
    def test_infers_preferred_workload_from_repo_signals(self):
        workload = infer_preferred_workload(
            {"torch": 10, "vllm": 2},
            {"requirements.txt": "vllm\ntransformers\n"},
        )

        self.assertEqual(workload, "vllm")

    @mock.patch("tools.external_lookups.dockerhub_tags_structured", side_effect=_fake_tags)
    def test_ranks_specialized_vllm_image_for_vllm_repo(self, _mocked):
        ranker = RocmImageRanker(ImageRankerConfig(
            gpu_arch="gfx942",
            preferred_python="3.12",
        ))

        ranked = ranker.rank(
            {"torch": 10, "vllm": 3, "transformers": 4},
            {"requirements.txt": "vllm\ntransformers\n"},
        )

        self.assertEqual(ranked[0].workload, "vllm")
        self.assertIn("gfx94X", ranked[0].tag)
        self.assertTrue(any("preferred workload" in reason for reason in ranked[0].reasons))

    @mock.patch("tools.external_lookups.dockerhub_tags_structured", side_effect=_fake_tags)
    def test_prefers_pytorch_release_tag_for_generic_torch_repo(self, _mocked):
        ranker = RocmImageRanker(ImageRankerConfig(
            preferred_python="3.10",
            strict_mode=True,
        ))

        ranked = ranker.rank(
            {"torch": 6, "numpy": 3},
            {"requirements.txt": "torch\nnumpy\n"},
        )

        self.assertEqual(ranked[0].workload, "pytorch")
        self.assertIn("py3.10", ranked[0].tag)
        self.assertNotEqual(ranked[0].tag, "latest")


if __name__ == "__main__":
    unittest.main()
