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

from agents.paper_corpus import build_paper_corpus  # noqa: E402


class PaperCorpusTests(unittest.TestCase):
    def test_builds_both_sources_and_preserves_provenance(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf") as handle:
            with mock.patch(
                "agents.paper_corpus.extract_pdf_text",
                return_value="PDF paper text",
            ), mock.patch(
                "agents.paper_corpus.fetch_arxiv_html_text",
                return_value=("HTML paper text", {"source_url": "https://arxiv.org/html/1234.5678"}),
            ):
                corpus = build_paper_corpus(
                    handle.name,
                    arxiv_id="1234.5678",
                    source_mode="both",
                )

        self.assertTrue(corpus.has_text())
        self.assertEqual(corpus.source_mode, "both")
        self.assertEqual(corpus.resolved_modes, ["pdf", "html"])
        self.assertEqual([source.source_kind for source in corpus.sources], ["pdf", "html"])
        self.assertIn("PDF paper text", corpus.index_text)
        self.assertIn("HTML paper text", corpus.index_text)
        self.assertEqual(corpus.provenance["source_count"], 2)

    def test_html_mode_falls_back_to_pdf_when_html_unavailable(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf") as handle:
            with mock.patch(
                "agents.paper_corpus.extract_pdf_text",
                return_value="Fallback PDF text",
            ), mock.patch(
                "agents.paper_corpus.fetch_arxiv_html_text",
                return_value=("", {"error": "html_fetch_failed"}),
            ):
                corpus = build_paper_corpus(
                    handle.name,
                    arxiv_id="1234.5678",
                    source_mode="html",
                )

        self.assertTrue(corpus.has_text())
        self.assertEqual(corpus.resolved_modes, ["pdf"])
        self.assertIn("fell_back_to_pdf_source", corpus.provenance["warnings"])


if __name__ == "__main__":
    unittest.main()
