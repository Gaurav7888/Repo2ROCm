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

from learning.graphify_provider import GraphifyProvider  # noqa: E402


class RepoCorpusTests(unittest.TestCase):
    def test_repo_corpus_indexes_full_readme_and_code(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            readme_path = os.path.join(tmpdir, "README.md")
            code_path = os.path.join(tmpdir, "train.py")
            config_dir = os.path.join(tmpdir, "configs")
            os.makedirs(config_dir, exist_ok=True)
            config_path = os.path.join(config_dir, "train.yaml")

            with open(readme_path, "w", encoding="utf-8") as handle:
                handle.write("Install steps\nRun python train.py --config configs/train.yaml\nExpected accuracy: 91.2\n")
            with open(code_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "import argparse\n"
                    "def main():\n"
                    "    parser = argparse.ArgumentParser()\n"
                    "    parser.add_argument('--config')\n"
                    "    print('accuracy')\n"
                )
            with open(config_path, "w", encoding="utf-8") as handle:
                handle.write("batch_size: 8\nlearning_rate: 1e-4\nepochs: 3\n")

            provider = GraphifyProvider(tmpdir)
            self.assertTrue(provider.index_repo_corpus())

            readme_hits = provider.query_repo_corpus(
                "install run command expected accuracy",
                scope="readme",
            )
            code_hits = provider.query_repo_corpus(
                "argparse config accuracy print",
                scope="code",
            )
            config_hits = provider.query_repo_corpus(
                "batch size learning rate epochs",
                scope="config",
            )

        self.assertIn("README.md", readme_hits)
        self.assertIn("train.py", code_hits)
        self.assertIn("configs/train.yaml", config_hits)


if __name__ == "__main__":
    unittest.main()
