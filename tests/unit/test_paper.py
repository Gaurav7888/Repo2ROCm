"""Paper module: typed contracts + corpus round-trip."""
from __future__ import annotations

from pathlib import Path

from repo2rocm.paper import PaperCorpus, arxiv_id_from_readme
from repo2rocm.paper.types import (
    Experiment,
    Hyperparameter,
    MetricDefinition,
    PaperContext,
    PaperMetadata,
    RepoBinding,
    runtime_class_to_min,
)


def test_arxiv_id_extraction():
    assert arxiv_id_from_readme("see arXiv:2401.01234 for details") == "2401.01234"
    assert arxiv_id_from_readme("https://arxiv.org/abs/2402.56789") == "2402.56789"
    assert arxiv_id_from_readme("https://arxiv.org/pdf/2403.99999v2.pdf") == "2403.99999"
    assert arxiv_id_from_readme("no paper here") == ""


def test_runtime_class_minutes():
    assert runtime_class_to_min("smoke") == 2
    assert runtime_class_to_min("short") == 15
    assert runtime_class_to_min("medium") == 90
    assert runtime_class_to_min("long") == 480
    assert runtime_class_to_min("unknown") == 60
    assert runtime_class_to_min("garbage") == 60


def _example_experiment() -> Experiment:
    return Experiment(
        id="E1",
        title="LongBench qasper on Mistral-7B with SnapKV",
        model_checkpoint="mistralai/Mistral-7B-Instruct-v0.2",
        dataset="LongBench/qasper",
        metric=MetricDefinition(
            name="qasper_f1",
            value=35.6,
            unit="",
            portability="accuracy",
            default_tolerance=0.03,
            paper_source="Table 1, row 'SnapKV', column 'qasper'",
            repo_eval_source="experiments/LongBench/eval.py:scorer_token_f1",
        ),
        hyperparameters=[
            Hyperparameter(
                name="max_capacity_prompt",
                value="1024",
                paper_source="\u00a74.1 Experimental Setup",
            ),
            Hyperparameter(
                name="window_size",
                value="32",
                paper_source="Appendix B",
            ),
        ],
        repo_bindings=[
            RepoBinding(
                hyperparam_name="max_capacity_prompt",
                kind="cli_flag",
                location="experiments/LongBench/pred_snap.py --max_capacity_prompt",
                default="1024",
            ),
            RepoBinding(
                hyperparam_name="window_size",
                kind="constant",
                location="snapkv/monkeypatch/snapkv_utils.py:88 window_size",
                default="32",
            ),
        ],
        suggested_script="experiments/LongBench/pred_snap.py",
        suggested_command=(
            "python experiments/LongBench/pred_snap.py --model mistral-7B "
            "--task qasper --max_capacity_prompt 1024 > /repo/paper_experiment.log"
        ),
        runtime_class="medium",
        rationale="accuracy metric ports cleanly across CUDA\u2192ROCm",
    )


def test_experiment_ensure_back_compat_populates_legacy_fields():
    exp = _example_experiment()
    assert exp.headline_metric is None
    assert exp.code_available is False
    assert exp.estimated_runtime_min == 0
    exp.ensure_back_compat()
    assert exp.headline_metric is not None
    assert exp.headline_metric.name == "qasper_f1"
    assert exp.headline_metric.value == 35.6
    assert exp.headline_metric.portability == "accuracy"
    assert exp.code_available is True
    assert exp.estimated_runtime_min == 90  # medium


def test_corpus_round_trip(tmp_path: Path):
    corpus = PaperCorpus(tmp_path / "papers")
    ctx = PaperContext(
        metadata=PaperMetadata(arxiv_id="2404.14469", title="SnapKV"),
        experiments=[_example_experiment()],
        chosen_experiment_id="E1",
    )
    path = corpus.save(ctx)
    assert path.is_file()
    loaded = corpus.load("2404.14469")
    assert loaded is not None
    assert loaded.chosen_experiment_id == "E1"
    assert loaded.metadata.title == "SnapKV"
    chosen = loaded.chosen()
    assert chosen is not None
    assert chosen.metric is not None
    assert chosen.metric.paper_source.startswith("Table 1")
    assert len(chosen.hyperparameters) == 2
    assert {b.hyperparam_name for b in chosen.repo_bindings} == {
        "max_capacity_prompt",
        "window_size",
    }


def test_paper_context_render_for_reproducer_includes_bindings():
    ctx = PaperContext(
        metadata=PaperMetadata(arxiv_id="2404.14469", title="SnapKV"),
        experiments=[_example_experiment()],
        chosen_experiment_id="E1",
    )
    txt = ctx.render_for_reproducer()
    assert "SnapKV" in txt
    assert "Chosen experiment" in txt
    assert "max_capacity_prompt = 1024" in txt
    assert "Repo bindings" in txt
    assert "Hyperparameters" in txt
    assert "qasper_f1" in txt
    assert "Suggested command" in txt
    assert "paper_experiment.log" in txt


def test_render_for_reproducer_lists_unbound_hyperparameters():
    exp = _example_experiment()
    exp.hyperparameters.append(
        Hyperparameter(name="pooling", value="avgpool", paper_source="\u00a74.1")
    )
    exp.unbound_hyperparameters.append("pooling")
    ctx = PaperContext(
        metadata=PaperMetadata(arxiv_id="2404.14469", title="SnapKV"),
        experiments=[exp],
        chosen_experiment_id="E1",
    )
    txt = ctx.render_for_reproducer()
    assert "Unbound hyperparameters" in txt
    assert "pooling" in txt
