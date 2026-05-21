"""Paper research module.

Owns the typed contract for paper metadata + experiments and a corpus store
under {work_dir}/papers/. The extraction/selection logic lives in the
paper-research agent + skills, not here \u2014 this package only exposes:

  * `PaperCorpus`           \u2014 read/write the persisted `PaperContext` JSON
  * `fetch_paper`           \u2014 download a PDF + companion HTML from arXiv/URL
  * `arxiv_id_from_readme`  \u2014 best-effort arXiv ID detector
  * typed records           \u2014 `PaperContext`, `Experiment`, `MetricDefinition`,
                              `Hyperparameter`, `RepoBinding`, ...
"""
from repo2rocm.paper.corpus import PaperCorpus
from repo2rocm.paper.fetch import (
    arxiv_id_from_readme,
    fetch_arxiv_pdf,
    fetch_paper,
)
from repo2rocm.paper.types import (
    CommandSpec,
    Experiment,
    Hyperparameter,
    MetricClass,
    MetricDefinition,
    MetricRow,
    PaperContext,
    PaperMetadata,
    RepoBinding,
    RuntimeClass,
    runtime_class_to_min,
)

__all__ = [
    "CommandSpec",
    "Experiment",
    "Hyperparameter",
    "MetricClass",
    "MetricDefinition",
    "MetricRow",
    "PaperContext",
    "PaperCorpus",
    "PaperMetadata",
    "RepoBinding",
    "RuntimeClass",
    "arxiv_id_from_readme",
    "fetch_arxiv_pdf",
    "fetch_paper",
    "runtime_class_to_min",
]
