---
name: repo_config_discovery
description: How to discover a repo's actual config surface (CLI flags, JSON/YAML configs, hardcoded constants, monkeypatch points)
when_to_use: After paper extraction, when you need to bind paper hyperparameters to repo knobs
allowed_tools: ["Read", "Grep", "Glob"]
---

# Discovering a repo's config surface

Reproducibility = **paper-side config tuple ∩ repo-side config schema**. You
cannot do the intersection until you know what the repo actually exposes.

## Where to look (in order)

1. **README run snippets** — the canonical "how to run an experiment" example.
   Often shows the entry-point script and a representative flag set.
2. **Top-level `experiments/`, `scripts/`, `eval/`, `benchmark*/`, `examples/`
   directories** — the actual entry points.
3. **The argparse / click / typer / hydra block** of the entry-point script
   (usually at the bottom or inside `if __name__ == "__main__":`). Read it
   in **full**, not just the flag names — defaults and `choices=[...]` are
   load-bearing.
4. **`configs/`, `config/`, `conf/` directories with `*.json` / `*.yaml`** —
   in many ML repos this is the *real* source of truth for hyperparameters;
   the CLI just picks a config file by name.
5. **Library-internal constants** — for monkey-patch / plugin papers (SnapKV,
   H2O, StreamingLLM, …) some knobs live as defaults in a `*_utils.py` or
   `monkeypatch.py`. Find them with `Grep` on the paper-side name.
6. **`pyproject.toml` / `setup.py` / `requirements*.txt`** — for the version
   pin context (e.g. `transformers==4.37.0` means the monkey-patch is
   coupled to that exact API).

## Per-knob discovery recipe

For each paper hyperparameter `hp.name` (e.g. `max_capacity_prompt`):

```
1. Glob "**/*.py" + Grep for the exact name and obvious synonyms.
2. Grep "**/*.json" + "**/*.yaml" + "**/*.yml" for the same.
3. If a CLI flag matches:
   - Read the surrounding argparse block to confirm `default=` and `choices=`.
   - Record a RepoBinding(kind="cli_flag", location="<file> --<flag>", default=...).
4. If a JSON/YAML key matches:
   - Open the config file. Confirm the key is at the right nesting depth.
   - Record a RepoBinding(kind="json_key", location="<file>::<key.path>").
5. If only a hardcoded constant matches:
   - Decide: is it OK to patch this constant at run time, or does the paper
     need a different value than the hardcoded one?
   - If the hardcoded value matches the paper, record a RepoBinding(kind="constant").
   - If they differ, this is an *unbound* hyperparameter that needs a
     `kind="code_patch"` binding (or add to `unbound_hyperparameters`).
6. If nothing matches anywhere:
   - This hyperparameter is genuinely unbound. Add its name to
     `unbound_hyperparameters`.
```

## What the entry-point script tells you

For each candidate script:

- `arg_name → (default, choices)` — the full CLI schema, including defaults.
- `which model checkpoints are supported` — usually in a `MODEL_MAP` dict, a
  `model2path.json` config, or a series of `if model == "...":` branches.
- `which datasets / tasks are supported` — same pattern.
- `which metrics are computed` — find the `eval` / `evaluate` / `score`
  function and read it. The metric the reproducer can verify is the metric
  the eval script *prints* in a parseable form.

Read `eval.py` (or equivalent) in full. PaperVerify's regex needs the metric
to appear in the log as `name: value` or `name = value` — if the eval script
prints a JSON blob instead, you need to record that in `repo_eval_source`
so the reproducer knows to use the JSON parser path.

## Monkey-patch papers specifically

When the paper *patches* a library (e.g. `replace_mistral()`, `replace_llama()`),
some hyperparameters live as module-level constants in the patched code, not
as CLI flags. Examples:

- `snapkv/monkeypatch/snapkv_utils.py` — `window_size`, `kernel_size`,
  `pooling` are defaults; not all are exposed via CLI.
- `h2o/h2o_kv_cache.py` — `hh_size`, `recent_size` similar.

For each such constant: confirm the paper's value matches, or record a
`code_patch` binding so the reproducer applies the patch before running.

## Model-family / dataset support enumeration

Some entry-point scripts only support a subset of what the paper claims:

- SnapKV's `monkeypatch.py` only has `replace_llama`, `replace_mistral`,
  `replace_mixtral`. Anything else from the paper (e.g. LLaMA-3 if not yet
  patched) is not reproducible via this repo as-shipped.
- LongBench has 21 tasks but the repo's `config2maxlen.json` may only wire
  a subset. Anything outside that subset needs `unbound_hyperparameters`.

Always confirm the specific (model, dataset) pair is in the repo's
supported set before binding.

## What you should hand to `paper_repo_binding`

For each candidate experiment, a complete map:
`{hp.name → RepoBinding | "unbound"}`. No silent omissions. Either a binding
exists, or the name is in `unbound_hyperparameters`.
