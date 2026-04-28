# Skill: frameworkApiDrift

**Use when:** the agent's repo carries pinned old versions of
`transformers`, `accelerate`, `peft`, `diffusers`, `pytorch-lightning`,
or vendors its own model code that diverges from current upstream APIs.
Symptoms include `AttributeError`, `TypeError: __init__() got an
unexpected keyword`, `cannot import name`, or model loading exploding
on architecture init (e.g. `rope_theta` vs `rope_parameters`).

**What to research:**
- Which exact `transformers` / `accelerate` version aligns with the
  repo's vendored code or the model checkpoint family.
- Known compatibility shims or release notes for the drift point.

**What to recommend:**
- Pin the framework version that matches the repo's intent, OR apply a
  narrow `getattr(...)` shim so old and new APIs both work.
- Don't blindly upgrade the framework when the repo expects an older
  shape — that just trades one drift for another.

**Tone:** specific. Name the version range. Name the attribute.
