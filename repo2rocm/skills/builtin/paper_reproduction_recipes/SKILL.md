---
name: paper_reproduction_recipes
description: End-to-end recipe for turning a paper + repo into ONE reproducible experiment on ROCm
when_to_use: At the very start of the paper-research and paper-reproducer agents
allowed_tools: ["PaperFetch", "PaperOutline", "PaperRead", "PaperRecall", "PaperVerify", "EmitPaperContext"]
---

# End-to-end paper reproduction recipe

The paper-research agent is exploratory, not pipelined. You pick which tools
to call based on what the paper and repo look like. The phases below are a
suggested order, not a fixed sequence — skip a phase if its outputs are
already in your context.

## Phase 1 — Locate and fetch the paper

- If the CLI provided `paper_arxiv_id` or `paper_url`, call `PaperFetch`
  with `source="arxiv_id"` or `source="url"` directly.
- Otherwise, call `PaperFetch(source="readme_arxiv_id", readme_text=...)`
  with the README excerpt — it will pull the arXiv id out.

The fetcher saves `papers/<id>.pdf` plus a companion `papers/<id>.html` when
available. The HTML is essential for table-heavy papers (PDF table
extraction is unreliable).

## Phase 2 — Navigate the paper (see `/paper_navigation`)

1. `PaperOutline(pdf_path=...)` once. Note section/table/figure lists,
   page count, and `setup_hint_offsets`.
2. Identify (a) the results table you'll target and (b) the §Setup section.
3. Read those sections via `PaperRead(section="...")`. For very long
   sections, page with `PaperRead(chunk=N)` until `has_next=False`.
4. Always also read the appendix if one exists — the §Setup section often
   defers config details there.

## Phase 3 — Extract candidate experiments (see `/paper_experiment_extraction`)

For each plausible reproducible row:
- Record the headline metric with `paper_source` (exact table cell / section
  paragraph).
- Record every hyperparameter the paper names (model checkpoint, dataset
  subset, all knobs, prompt template) with `paper_source`.
- Capture the baseline row from the same table as a `related_metric` when
  available.

Prefer accuracy/quality/ratio_speedup over absolute_perf (see
`/paper_metric_portability`).

## Phase 4 — Discover the repo's config surface (see `/repo_config_discovery`)

Use `Glob`, `Grep`, and `Read` (not deterministic scanners — read the actual
files). For each candidate experiment, find:
- The entry-point script (often under `experiments/`, `scripts/`, `eval/`).
- Its full CLI/argparse schema **with defaults**.
- Any JSON/YAML config files it loads.
- Hardcoded constants in monkey-patches / utils that the paper might rely on.
- The metric the eval script actually prints, so the reproducer can parse it.

## Phase 5 — Bind-check (see `/paper_repo_binding`)

For every paper hyperparameter:
- Build a `RepoBinding` if you found a matching CLI flag, JSON/YAML key, or
  constant — record `location` and `default`.
- Otherwise add the name to `unbound_hyperparameters`. Never silently omit.

Drop candidate experiments whose unbound count is unacceptably high; prefer
the experiment whose tuple is fully bound.

## Phase 6 — Pick ONE and emit

- `chosen_experiment_id = <the one>` — the headline claim with the best
  binding fit and the most portable metric.
- Build `suggested_command` fully bound (every flag has its paper value;
  redirect to `/repo/paper_experiment.log`).
- Call `EmitPaperContext(context=...)` exactly once. End your turn.

## Phase 7 — (later, in the paper-reproducer agent)

- `PaperRecall()` to load the persisted PaperContext.
- `DockerExec` to run `suggested_command` verbatim. Do not scale down.
- If `unbound_hyperparameters` is non-empty, apply the necessary code patch
  via `Edit` first, then run.
- `PaperVerify(log_path="/repo/paper_experiment.log", metrics=[...])`. Copy
  `expected_value` and `tolerance` straight from the chosen experiment's
  `metric.value` and `metric.default_tolerance`. Do not widen.
- Return exactly one of: `PAPER_REPRODUCED`, `PAPER_MISMATCH`,
  `PAPER_RUN_FAILED`.

## Anti-patterns to avoid

- **Reading the abstract and stopping**: the abstract has zero reproducible
  configs. Always read the §Setup section.
- **Skipping the appendix**: many "missing" hyperparameters are there.
- **Guessing a default**: if the paper doesn't name a knob and the repo
  doesn't expose it, it's unbound. Say so.
- **Picking the best row**: pick the headline row, not the ablation that
  happens to win.
- **Silently truncating sample counts to make it run**: that's no longer the
  experiment the paper described.
- **Fabricating metric values**: if `PaperVerify` returns `unknown` and a
  retry with extra logging still doesn't parse, return `PAPER_RUN_FAILED`.
