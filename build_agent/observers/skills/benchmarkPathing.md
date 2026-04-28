# Skill: benchmarkPathing

**Use when:** the agent is about to invoke a benchmark / eval script
and the entrypoint, working directory, or CLI flags look misaligned
with what the repo actually exposes (mismatched script name, missing
`--config`, wrong `cd`).

**What to research:** usually no web; this is a local-evidence skill.
Check the repo via `graphify_query --scope code` or `grep` to find the
real entrypoint and the documented flags.

**What to recommend:**
- The exact entrypoint path the repo ships.
- The correct `cd` before launch, especially when scripts use relative
  paths.
- Tee output to a stable log path (`/repo/paper_experiment.log`) so the
  paper verifier can read it.

**Tone:** quick, surgical. One line of the form
"the real entrypoint is `<path>`; cd to `<dir>` first; pass `<flag>`."
