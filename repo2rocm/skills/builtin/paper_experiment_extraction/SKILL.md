---
name: paper_experiment_extraction
description: How to extract a reproducible experiment spec (fields, provenance, headline vs. ablation)
when_to_use: After PaperRead, when constructing the Experiment object that will be passed to EmitPaperContext
allowed_tools: ["PaperRead", "PaperOutline"]
---

# Extracting a reproducible experiment spec

A reproducible experiment is not "a number". It is a fully-specified tuple
the reproducer can run end-to-end and verify against the paper's number. If
any field is missing or guessed, the reproduction is invalid — even if it
happens to match.

## The required fields (and where to find them)

| Field | Where it usually lives in the paper |
|---|---|
| `model_checkpoint` (e.g. `mistralai/Mistral-7B-Instruct-v0.2`) | §Setup or table caption ("results on Mistral-7B-Instruct-v0.2") |
| `dataset` (e.g. `LongBench/qasper`) | Table column header or row label |
| `metric.name` (e.g. `F1`, `accuracy`, `qasper_f1`, `speedup`) | Table column header / "we report X" sentence |
| `metric.value` (the target number) | Table cell — copy verbatim |
| `metric.unit` (`%`, `pp`, `x`, `ms`, …) | Table caption / column header |
| `metric.paper_source` (verbatim provenance) | Always required. "Table 1, row 'SnapKV', column 'qasper'" |
| `hyperparameters[]` (every knob the paper specifies) | §Setup + Appendix |
| `prompt_template` | §Setup or footnote; sometimes only in repo |

If the paper doesn't name a hyperparameter, **don't fill in a default** —
either find it in the repo (then record it with `paper_source=""` and a
RepoBinding) or list it in `unbound_hyperparameters`.

## Headline vs. ablation rows

Papers usually report:

- 1 **headline** row — the method's main claim, vs. a baseline column.
- N **ablation** rows — knob sweeps, intentionally suboptimal.

You want the **headline**, not whichever row has the best number. To spot
the headline:

- It's the row the abstract / introduction quotes verbatim.
- Its hyperparameters match the §Setup section (ablations override one knob).
- Its row label is the method's own name (e.g. "SnapKV"), not a variant
  ("SnapKV-pooling=max").

If the table reports the method vs. a baseline in the same column block,
extract **both** — put the headline row as `metric` and the baseline row in
`related_metrics`. PaperVerify can then sanity-check that the method beats
its own baseline by the claimed margin.

## Provenance discipline

Every numeric field must carry `paper_source`. Acceptable forms:

- `"Table 1, row 'SnapKV', column 'qasper'"`
- `"§4.2 paragraph 3"`
- `"Appendix B, Table 7"`
- `"caption of Figure 3"`

Forbidden:

- `""` (empty)
- `"the paper says so"`
- `"abstract"` (abstract numbers are summaries, not the target)

If you can't cite a specific location, you haven't actually read enough.
Go back to `PaperOutline` + `PaperRead`.

## Multiple experiments

Emit *several* candidates if you found multiple reproducible options.
`PaperContext.experiments` is a list; `chosen_experiment_id` picks one.
Order doesn't matter, but mark the headline as chosen.

## Common pitfalls

- **Decimal vs. percentage**: a table reporting "0.356" with caption "F1"
  means the same as "35.6". Pick one and stay consistent with the unit.
- **Average vs. per-task**: LongBench-style benchmarks often report both
  the "average" row and per-task rows. Use a single per-task row as the
  reproduction target; the average is too noisy to reproduce on a single
  task.
- **"Setup" claim doesn't match table column**: this means the table column
  is a different model size. Don't mix. Either pick a different table that
  uses the §Setup model, or pick a different model checkpoint to match the
  table.
- **Method = "Ours"**: read §3 (method section) to find the actual name.
  Never emit `model_checkpoint="Ours"`.

## What you should hand to EmitPaperContext

For each candidate experiment, every paper-side hyperparameter is in
`hyperparameters[]` with provenance. Every dataset / model is fully named.
The metric has both `paper_source` (where the target lives) and (after
`/repo_config_discovery`) `repo_eval_source` (where the reproduction
computes its value).
