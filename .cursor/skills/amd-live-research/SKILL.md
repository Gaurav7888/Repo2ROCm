---
name: amd-live-research
description: Uses live internet evidence and deterministic package or image lookups for AMD ROCm HIP-specific debugging. Use when the repo hits ROCm, HIP, gfx, MIOpen, rocBLAS, libamdhip64, flash-attn, xformers, bitsandbytes, Triton, or fast-moving package and image compatibility issues where static knowledge may be stale.
---
# AMD Live Research

## Quick Start

When a problem is AMD-specific or version-sensitive:

1. Use `graphify_query` first for repo structure, entrypoints, config loaders, and metric logging.
2. Use `pypi_versions` or `dockerhub_tags` for package/image facts.
3. Use `web_search` with the exact error string plus `AMD ROCm HIP`.
4. Use `visit_url` on one or two high-signal sources.
5. Escalate to `deep_research` when the issue spans multiple packages, versions, or low-level runtime behavior.

## Tool Choice

- `graphify_query "<question>" --scope code`:
  Use for "where is the entrypoint/config/metric logger?" before broad `find` or `grep -r`.
- `pypi_versions <pkg>`:
  Use for fast-moving Python package compatibility.
- `dockerhub_tags <image>`:
  Use for real ROCm image tags before `change_base_image`.
- `web_search "<query>"`:
  Use for one-shot AMD or ROCm runtime errors.
- `visit_url <url>`:
  Use to read the best GitHub issue, ROCm doc, or package README hit.
- `deep_research "<question>"`:
  Use when the failure likely depends on multiple moving parts and a single search result will not be enough.

## Query Templates

- Runtime error:
  `web_search "<exact error> AMD ROCm HIP"`
- Package compatibility:
  `web_search "<package> ROCm <torch version> AMD"`
- gfx-arch issue:
  `web_search "<exact error> gfx942 ROCm"`
- Paper-runtime mismatch:
  `deep_research "What AMD-specific caveats matter for reproducing this paper result on ROCm?"`

## Rules

- Prefer live evidence over static prompt knowledge for AMD-specific facts.
- Prefer deterministic package/image lookups over guessing versions or tags.
- Quote exact error strings, package versions, image tags, and gfx architectures.
- Do not store repo-specific or one-off web findings as global knowledge.
- Let current repo evidence and deterministic verifier outputs win when they conflict with web advice.

## Additional Resource

For source priority and escalation guidance, see [reference.md](reference.md).
