# Reference Checklist

## Paper Facts Checklist

- Did you read the main body, tables, footnotes, and figure captions first?
- Is appendix/supplementary actually needed, or is the main paper body already enough?
- What is the exact experiment row / table / figure being targeted?
- What is the exact dataset name in the paper?
- What is the exact horizon / split / benchmark / seed policy?
- What metric names are reported?
- Are the targets numeric, relative, or qualitative?
- Are there appendix-only caveats or missing-data conditions?

## Repo Facts Checklist

- What is the real entry script?
- What flags does it actually expose?
- What config files control the experiment?
- What data files and side inputs are present?
- What helper scripts or README examples exist?
- What are the repo defaults, and do they silently differ from the paper row?
- What metric names are printed in logs?

## Runtime Facts Checklist

- What command actually executed?
- Was the environment valid and stable?
- Were files present at runtime?
- Did the run hit import or code errors?
- Did the run complete far enough to emit real metrics?
- Were any stubs or disabling patches introduced?

## Reconciliation Questions

- Is the planned experiment really runnable from the shipped repo?
- Is the run using the same dataset/horizon/metric as the paper claim?
- Is the logged metric definition actually comparable?
- Did the agent test the method, or accidentally test a fallback/baseline/default?
- Is the real conclusion "not reproduced", or "blocked", or "wrong target chosen"?

## Automation Gate

Before adding new production logic, ask:

1. Is this rule generic across many repos?
2. Can it be expressed as introspection instead of pattern matching?
3. Does it depend on one repo's filenames, helper script, or paper wording?
4. Would this be better as a skill / workflow / plan note?

If answers 3 or 4 are "yes", prefer the skill/workflow path.
