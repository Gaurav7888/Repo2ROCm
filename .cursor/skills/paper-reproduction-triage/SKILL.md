---
name: paper-reproduction-triage
description: Investigates paper reproduction failures by reconciling paper claims, repo capabilities, and runtime evidence. Use when selecting a paper experiment, checking whether a paper target actually matches the codebase, debugging reproduction runs, or classifying failures as repo bug, missing artifact, paper-code mismatch, metric-definition mismatch, or environment issue.
---
# Paper Reproduction Triage

## Quick Start

Use this workflow whenever a paper result looks suspicious, a planned
experiment may not match the shipped repo, or a run failed and the reason is
unclear.

Work from three evidence surfaces in order:

1. **Paper facts**
   - What dataset, benchmark, horizon, metric, and hyperparameters does the
     paper actually claim?
   - Are the reported numbers numeric and directly comparable, or only
     qualitative / relative?
   - Read main body, tables, footnotes, and figure captions first.
   - Consult appendix / supplementary only if config, metric definition, or
     setup is still ambiguous after the first pass.
   - What caveats, disclaimers, or hidden conditions appear in tables,
     captions, appendix, or README text?

2. **Repo facts**
   - What entry scripts exist?
   - What CLI flags, config files, and defaults actually control the run?
   - What datasets / side inputs / checkpoints / helper scripts are actually
     shipped in the repo?
   - What metric names does the code log?

3. **Runtime facts**
   - What command actually ran?
   - What files were present at runtime?
   - What import, environment, dataset, or code errors occurred?
   - What metrics, if any, were actually printed?

Do not form a verdict until all three surfaces are checked.

## Core Rules

- Never compare a paper number to the wrong dataset, wrong horizon, wrong
  metric definition, or wrong aggregation.
- Never trust helper-script defaults if the paper row requires explicit
  overrides.
- If the repo does not ship a required dataset or side input, classify the
  run as **blocked by missing artifact**, not as a plain reproduction failure.
- If the code needs patches just to import or finish a run, separate
  **repo bug / incomplete repo** from **paper mismatch**.
- If the metric is not directly comparable, say so explicitly instead of
  fabricating a numeric conclusion.

## Failure Taxonomy

Classify every failed or inconclusive paper run into one primary bucket:

- `missing artifact`: required dataset, checkpoint, side input, or config is absent
- `unsupported config surface`: paper needs a flag/config path the repo does not expose
- `repo bug`: shipped code is broken or incomplete
- `paper-code mismatch`: repo defaults or implementation do not match the paper row
- `metric-definition mismatch`: logged metric is not the paper metric
- `environment issue`: install/runtime/GPU/container problem

Use the most specific label available.

## Output Template

Use this structure when reporting findings:

```markdown
## Paper Facts
- ...

## Repo Facts
- ...

## Runtime Facts
- ...

## Reconciliation
- ...

## Verdict
- status: reproduced | not reproduced | blocked | inconclusive
- reason: ...
```

## When Writing Code

If you want to automate part of this workflow:

- Keep **generic mechanisms** in production code:
  - metric verification
  - marker safety
  - generic file existence checks
  - generic CLI/config introspection
  - generic ranking by portability/runtime
- Keep **repo-specific reasoning** out of production code:
  - dataset aliases for one repo
  - helper-script variable names from one project
  - one-paper metric aliases
  - hand-written regexes that only make sense for a single failure

If a rule mentions a specific paper, dataset name, repo, or helper script, it
probably belongs in a plan, run note, or skill, not in core code.

## Additional Resource

For a detailed checklist and decision prompts, see [reference.md](reference.md).
