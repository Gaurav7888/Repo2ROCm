---
name: cross-run-learning
description: Distills completed Repo2ROCm runs into compact, reusable lessons and decides what should become structured rules versus skill-level workflow guidance. Use when reviewing a finished run, curating cross-run learning, reducing lesson noise, or deciding whether new intelligence belongs in code, KB rules, or a skill.
---

# Cross-Run Learning

## Quick Start

Use this workflow after a run completes or when the current lesson pool feels
too noisy.

1. Start from failure-to-recovery pairs, not isolated failures.
2. Write lessons as `do`, `dont`, or `pattern`.
3. Keep only lessons that transfer across repos.
4. Put deterministic, machine-checkable behavior into structured KB/rules.
5. Put fuzzy operating judgment into skills or curated planner context.

## Storage Decision

- Use the KB/rule layer for exact matches, executable commands, confidence, and
  outcomes that can be checked automatically.
- Use skills for workflow, decision discipline, and lessons that need judgment
  or evidence reconciliation.
- Use plans/run notes for repo-specific details that should not become global learning.

## Distillation Standard

A lesson is worth keeping only if it is:

- caused by a real failure and later recovery
- actionable for a future agent
- generic beyond the current repo
- short enough to survive prompt compression

## Output Template

```markdown
## Lesson
- kind: do | dont | pattern
- trigger: ...
- action: ...
- why: ...
- storage: kb-rule | skill | run-note
```

## Additional Resource

For a curation checklist and anti-noise rubric, see [reference.md](reference.md).
