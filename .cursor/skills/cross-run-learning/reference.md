# Cross-Run Learning Reference

## Keep These

- A wrong action followed by a clearly better recovery
- A reproducible ROCm/package/build/runtime failure mode
- A lesson that helps planning, image choice, experiment choice, or recovery
- Advice that can be stated without one repo's filenames or one paper's wording

## Drop These

- One-off repo trivia
- Lessons that only restate logs without an action
- Contradictory or duplicated lessons
- Advice that should really be a deterministic guard or verifier

## Promote To KB / Rules When

- The trigger is machine-checkable
- The fix can be executed exactly
- Confidence can be updated from outcomes
- The behavior should happen automatically or semi-automatically

## Keep As Skill Guidance When

- The lesson requires judgment
- The agent must reconcile repo facts, paper facts, and runtime facts
- Several tools or evidence sources must be consulted before acting
- Overfitting would be dangerous in production code

## Review Questions

1. What did the agent do wrong first?
2. What changed when it finally succeeded?
3. Is the winning behavior generic or repo-specific?
4. Would a deterministic matcher be reliable here?
5. If not, what compact workflow should a future agent follow?
