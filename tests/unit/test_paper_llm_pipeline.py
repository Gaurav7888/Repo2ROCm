"""End-to-end smoke tests for the LLM-driven paper pipeline tools.

We don't spin up the LLM agent here \u2014 we directly invoke the deterministic
parts of each tool to make sure the contracts hold:

  * `PaperOutline` returns a structural map.
  * `PaperRead` reads sections / page ranges / chunks without an arbitrary cap.
  * `EmitPaperContext` validates the schema and rejects under-specified
    contexts. It accepts a fully-specified one and persists it to disk where
    `PaperRecall` can find it.

These tests use synthetic text inputs (no real PDF) by going through the
HTML fallback path \u2014 the helpers in `paper/extract.py` handle both.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from repo2rocm.core.permissions import PermissionMode
from repo2rocm.paper.types import (
    Experiment,
    Hyperparameter,
    MetricDefinition,
    PaperContext,
    PaperMetadata,
    RepoBinding,
)
from repo2rocm.tools.base import ReadFileState, ToolUseContext
from repo2rocm.tools.paper.emit_paper_context import EmitPaperContext
from repo2rocm.tools.paper.paper_outline import PaperOutline
from repo2rocm.tools.paper.paper_read import PaperRead
from repo2rocm.tools.paper.paper_recall import PaperRecall


def _ctx(tmp_path: Path) -> ToolUseContext:
    return ToolUseContext(
        agent_id="t",
        session_id="s",
        workdir=tmp_path,
        abort_event=asyncio.Event(),
        permission_mode=PermissionMode.BYPASS,
        read_file_state=ReadFileState(),
        options={},
    )


def _write_html_paper(tmp_path: Path, *, name: str = "paper") -> Path:
    """Write a synthetic paper as HTML (the extractor handles HTML directly).

    The .pdf sibling is empty \u2014 PaperOutline / PaperRead fall back to HTML
    when the PDF has no extractable text.
    """
    papers = tmp_path / "papers"
    papers.mkdir(parents=True, exist_ok=True)
    pdf = papers / f"{name}.pdf"
    html = papers / f"{name}.html"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")  # malformed; forces HTML fallback
    html.write_text(
        """
<html><body>
<h1>SnapKV: A Paper</h1>
<h2>1 Introduction</h2>
<p>We propose SnapKV.</p>
<h2>4 Experiments</h2>
<h3>4.1 Experimental Setup</h3>
<p>We evaluate Mistral-7B-Instruct-v0.2 on LongBench with
max_capacity_prompt=1024 and window_size=32.</p>
<h3>4.2 Results</h3>
<p>Table 1 reports F1 across LongBench tasks.</p>
<table>
  <caption>Table 1: Main results on LongBench.</caption>
  <tr><th>Method</th><th>qasper</th><th>multifieldqa_en</th></tr>
  <tr><td>FullKV</td><td>34.9</td><td>49.1</td></tr>
  <tr><td>SnapKV</td><td>35.6</td><td>49.0</td></tr>
</table>
<h2>Appendix B Hyperparameters</h2>
<p>Pooling is set to avgpool.</p>
</body></html>
""".strip(),
        encoding="utf-8",
    )
    return pdf


# ── PaperOutline ───────────────────────────────────────────────────────────


def test_outline_returns_sections_and_tables(tmp_path: Path):
    pdf = _write_html_paper(tmp_path)
    out = asyncio.run(
        PaperOutline().invoke({"pdf_path": str(pdf)}, _ctx(tmp_path))
    )
    assert not out.is_error
    titles = {s.title for s in out.data.sections}
    assert any("Experiments" in t for t in titles)
    assert any("Setup" in t or "Experimental" in t for t in titles)
    assert out.data.tables, "should see at least Table 1"
    assert any("LongBench" in t.caption for t in out.data.tables)
    assert out.data.setup_hint_offsets, "should detect 'Setup' anchor"


# ── PaperRead ──────────────────────────────────────────────────────────────


def test_read_chunk0_returns_content(tmp_path: Path):
    pdf = _write_html_paper(tmp_path)
    out = asyncio.run(
        PaperRead().invoke({"pdf_path": str(pdf), "chunk": 0}, _ctx(tmp_path))
    )
    assert not out.is_error
    assert out.data.total_chars > 0
    assert out.data.returned_chars > 0
    assert "SnapKV" in out.data.content


def test_read_section_slices_correctly(tmp_path: Path):
    pdf = _write_html_paper(tmp_path)
    out = asyncio.run(
        PaperRead().invoke(
            {"pdf_path": str(pdf), "section": "4.1 Experimental Setup"},
            _ctx(tmp_path),
        )
    )
    assert not out.is_error
    assert "Mistral" in out.data.content
    assert "max_capacity_prompt" in out.data.content
    # The next section's content should NOT bleed in.
    assert "Table 1" not in out.data.content


def test_read_chunks_walk_through_long_papers(tmp_path: Path):
    pdf = _write_html_paper(tmp_path, name="long")
    # Force tiny chunks so a real-sized paper would page.
    out0 = asyncio.run(
        PaperRead().invoke(
            {"pdf_path": str(pdf), "chunk": 0, "chars_per_chunk": 1000},
            _ctx(tmp_path),
        )
    )
    assert not out0.is_error
    if out0.data.has_next:
        out1 = asyncio.run(
            PaperRead().invoke(
                {"pdf_path": str(pdf), "chunk": 1, "chars_per_chunk": 1000},
                _ctx(tmp_path),
            )
        )
        assert not out1.is_error
        assert out0.data.content != out1.data.content


# ── EmitPaperContext ───────────────────────────────────────────────────────


def _valid_context_dict(arxiv_id: str = "2404.14469") -> dict:
    exp = Experiment(
        id="E1",
        title="qasper on Mistral with SnapKV",
        model_checkpoint="mistralai/Mistral-7B-Instruct-v0.2",
        dataset="LongBench/qasper",
        metric=MetricDefinition(
            name="qasper_f1",
            value=35.6,
            portability="accuracy",
            default_tolerance=0.03,
            paper_source="Table 1, row 'SnapKV', column 'qasper'",
        ),
        hyperparameters=[
            Hyperparameter(name="max_capacity_prompt", value="1024", paper_source="\u00a74.1"),
        ],
        repo_bindings=[
            RepoBinding(
                hyperparam_name="max_capacity_prompt",
                kind="cli_flag",
                location="experiments/LongBench/pred_snap.py --max_capacity_prompt",
                default="1024",
            ),
        ],
        suggested_command="python pred_snap.py --max_capacity_prompt 1024 > /repo/paper_experiment.log",
        runtime_class="medium",
    )
    ctx = PaperContext(
        metadata=PaperMetadata(arxiv_id=arxiv_id, title="SnapKV"),
        experiments=[exp],
        chosen_experiment_id="E1",
    )
    return ctx.model_dump()


def test_emit_persists_and_recall_loads(tmp_path: Path):
    payload = _valid_context_dict()
    ctx = _ctx(tmp_path)
    out = asyncio.run(EmitPaperContext().invoke({"context": payload}, ctx))
    assert not out.is_error, out.text
    assert out.data.ok
    assert Path(out.data.path).is_file()

    # PaperRecall finds it via the corpus.
    rec = asyncio.run(PaperRecall().invoke({}, _ctx(tmp_path)))
    assert not rec.is_error
    assert rec.data.found
    assert rec.data.context["chosen_experiment_id"] == "E1"


def test_emit_rejects_missing_metric(tmp_path: Path):
    payload = _valid_context_dict()
    payload["experiments"][0]["metric"] = None
    out = asyncio.run(
        EmitPaperContext().invoke({"context": payload}, _ctx(tmp_path))
    )
    assert out.is_error
    assert "metric" in out.text.lower()


def test_emit_rejects_missing_paper_source(tmp_path: Path):
    payload = _valid_context_dict()
    payload["experiments"][0]["metric"]["paper_source"] = ""
    out = asyncio.run(
        EmitPaperContext().invoke({"context": payload}, _ctx(tmp_path))
    )
    assert out.is_error
    assert "paper_source" in out.text.lower()


def test_emit_rejects_silently_omitted_binding(tmp_path: Path):
    payload = _valid_context_dict()
    # Add a hyperparameter without a corresponding binding or unbound entry.
    payload["experiments"][0]["hyperparameters"].append(
        {"name": "pooling", "value": "avgpool", "paper_source": "\u00a74.1"}
    )
    out = asyncio.run(
        EmitPaperContext().invoke({"context": payload}, _ctx(tmp_path))
    )
    assert out.is_error
    assert "pooling" in out.text
    assert "unbound" in out.text.lower() or "repo_bindings" in out.text.lower()


def test_emit_accepts_unbound_when_recorded(tmp_path: Path):
    payload = _valid_context_dict()
    payload["experiments"][0]["hyperparameters"].append(
        {"name": "pooling", "value": "avgpool", "paper_source": "\u00a74.1"}
    )
    payload["experiments"][0]["unbound_hyperparameters"].append("pooling")
    out = asyncio.run(
        EmitPaperContext().invoke({"context": payload}, _ctx(tmp_path))
    )
    assert not out.is_error, out.text
    assert "pooling" in out.text  # surfaced as an unbound warning
