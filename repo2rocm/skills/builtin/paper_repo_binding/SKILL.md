---
name: paper_repo_binding
description: Bind-check process — every paper hyperparameter must map to a repo knob or be marked unbound
when_to_use: After extracting paper hyperparameters and discovering the repo's config surface; before EmitPaperContext
allowed_tools: ["Read", "Grep", "PaperRead"]
---

# The bind-check: paper hyperparameters ↔ repo knobs

This is the load-bearing step that turns "a number from a paper" into
"a command we can run". An experiment fails the bind-check unless every
paper-side hyperparameter is accounted for.

## The rule

For each `hp` in `experiment.hyperparameters`, exactly one of the following
must hold:

- A `RepoBinding(hyperparam_name=hp.name)` exists with a verified
  `location`, OR
- `hp.name` is in `experiment.unbound_hyperparameters`.

Silently omitting a binding is forbidden. `EmitPaperContext` will reject the
context.

## Procedure

1. **Iterate** over `experiment.hyperparameters`.
2. **For each `hp`**, check the discovery output from
   `/repo_config_discovery`:
   - If a CLI flag, JSON/YAML key, or matching constant was found, build a
     `RepoBinding`. Set `default=` if you observed one.
   - If only a hardcoded constant was found whose value differs from the
     paper's, record a `RepoBinding(kind="code_patch", location=<file:line>)`.
     This tells the reproducer to apply a one-line patch before running.
   - If nothing was found, add `hp.name` to `unbound_hyperparameters`.
3. **Re-rank** candidate experiments: prefer experiments whose
   `unbound_hyperparameters` is empty. If every candidate has unbound
   hyperparameters, prefer the one with the *fewest* and the *cheapest* to
   patch (CLI > JSON > constant > code_patch).
4. **Sanity-check the chosen experiment**:
   - The chosen model checkpoint is supported by the entry-point script
     (e.g. has a `replace_*()` for monkey-patch papers).
   - The chosen dataset / task is in the entry-point's accepted list.
   - The metric the entry-point computes (per `repo_eval_source`) is the
     same metric the paper reports.

## Building `suggested_command`

Once bound, construct the *fully bound* command:

```
<launcher> <suggested_script> --flag1 <value1> --flag2 <value2> ... \
    --output_dir /repo/paper_experiment_output \
    > /repo/paper_experiment.log
```

Rules:

- Every CLI-bound hyperparameter appears as a flag, with its paper-specified
  value.
- For JSON-bound hyperparameters, the JSON file is created or edited by the
  reproducer; the command references that file (e.g.
  `--config /repo/paper_experiment_config.json`). Don't try to put a JSON
  blob inside `suggested_command`.
- Output redirection to `/repo/paper_experiment.log` is required — that's
  what PaperVerify reads.
- Do NOT scale down sample counts or sequence lengths "just to make it run".
  If a value is too big for the container, that's a real failure of the
  reproduction, not a knob the agent gets to silently tune.

## When a binding is wrong

Common failure modes you should detect before emitting:

- **Name match, wrong semantics**: paper's `max_length` is per-task; repo's
  `--max_length` is global. Read the script's usage of the flag before
  committing.
- **Range mismatch**: paper says `kv_budget=1024`; repo `--kv_budget` is
  validated `>= 2048`. This is an unbound case — record a code_patch.
- **Stale config**: a JSON file with the right key exists, but the entry
  point reads a different config path. Confirm the script actually loads
  *that* file.
- **Implicit dependency**: paper says `pooling="avgpool"`; repo's pooling
  defaults depend on the model family. Confirm by reading the patched code.

## Picking among candidates

If multiple experiments survive the bind-check, prefer (in order):

1. **Fewer unbound hyperparameters** — closer to a no-edit reproduction.
2. **More portable metric class** (see `/paper_metric_portability`).
3. **Non-baseline rows** — reproducing the baseline is uninteresting; the
   claim is the method vs. its baseline.
4. **Cheaper runtime class** — smoke / short over medium over long, only
   when (1)-(3) tie.

## What you should hand to EmitPaperContext

- `experiment.metric` is the headline metric, fully specified with provenance.
- `experiment.hyperparameters` lists every knob the paper named.
- `experiment.repo_bindings` lists exactly the bindings you found; one per
  hyperparameter where possible.
- `experiment.unbound_hyperparameters` lists exactly the remaining names —
  no fewer (silently filled in), no more (false positives).
- `experiment.suggested_command` is fully bound and ready to run as-is.
