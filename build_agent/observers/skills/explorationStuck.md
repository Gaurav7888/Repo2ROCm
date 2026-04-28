# Skill: explorationStuck

**Use when:** the agent is circling — running `ls`, `cat`, `grep`,
`find` on the same area for many turns without committing to an action;
or applying a series of small patches that each fix one error and
unmask the next, with no global plan.

**What to research:**
- A community ROCm port of this repo (search GitHub for
  `<repo-name> rocm` or `<repo-name> AMD`).
- AMD blog posts or ROCm docs that describe the canonical port path.

**What to recommend:**
- A clear unstick: "stop fixing errors one at a time, here is the
  validated AMD path / community fork / prebuilt wheel".
- Or, if no shortcut exists: "list the full set of HIP fixes you'll
  need before recompiling once" — turn N micro-builds into 1.
- Or, if the repo simply isn't worth porting from CUDA source: "use
  AMD's official `<package>` wheel from `pypi.amd.com`".

**Tone:** the senior engineer voice. Cut the loop, name the strategy.
