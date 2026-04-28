# Skill: paperReproduction

**Use when:** Stage 2 is active (paper reproduction), and either the
agent is calling `verify_paper_result` without prior `paper_recall` /
`graphify_query --scope paper`, or it has already run the verifier
with a path/metric mismatch.

**What to research:**
- The paper's exact reported metric name, table reference, expected
  value, and tolerance.
- The repo's mapping from training output → log path → metric extractor.

**What to recommend:**
- Call `paper_recall` for the metric/experiment first, then form the
  `verify_paper_result` invocation with `--metric <name>=<expected>`
  and the correct `--log_path`.
- Tee the run's stdout+stderr to `/repo/paper_experiment.log` explicitly
  so the verifier has stable input.
- If the repo writes results to CSV/TSV, point at that artifact instead
  of stdout.

**Tone:** procedural; Stage-2 discipline matters more than novelty.
