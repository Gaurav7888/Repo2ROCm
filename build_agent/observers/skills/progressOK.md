# Skill: progressOK

**Use when:** the agent is making real forward progress — new submodules
being installed, new files being inspected, errors changing shape and
resolving — and there is nothing useful to add.

**Behavior:** stay silent. Set `intervene=false`, `priority=low`, leave
`research_question` empty, and put a one-line `fallback_advice` like
"Run is progressing healthily; no intervention." This signals the
sidecar to suppress emission so the executor's prompt stays clean.

Choose this skill aggressively. Most turns should fall here.
