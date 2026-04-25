# Source Priority

Prefer sources in this order:

1. Current repo code and config
2. Deterministic lookups:
   - `pypi_versions`
   - `dockerhub_tags`
3. Official or primary sources:
   - `rocm.docs.amd.com`
   - `github.com/ROCm`
   - `github.com/pytorch`
   - package upstream repos
4. High-signal issue threads
5. General blog posts or low-signal pages

## Escalation

- Use `graphify_query` before broad shell discovery.
- Use `web_search` first for a single AMD-specific runtime error.
- Use `visit_url` only on the best 1-2 hits.
- Use `deep_research` when:
  - multiple packages or versions interact
  - the issue depends on ROCm image tags
  - the issue depends on gfx arch / low-level runtime behavior
  - the fix needs evidence from more than one source

## What Not To Promote

Do not turn these into durable KB facts:

- one repo's helper-script flags
- one paper's wording
- one issue thread's local workaround
- one specific container's temporary environment hack

Promote only facts that are generic and machine-checkable:

- installable package versions
- valid Docker image tags
- stable package replacement mappings
- deterministic verifier behavior
