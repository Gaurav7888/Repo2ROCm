"""
Helpers for turning accumulated cross-run lessons into compact agent context.

This layer deliberately sits between raw storage and prompts/skills:
the KB and mempalace remain the sources of truth, while this module
curates only the highest-signal lessons for planners and agent skills.
"""

from __future__ import annotations

import re
from typing import Sequence


def build_lesson_query(
    full_name: str,
    frameworks: Sequence[str],
    cuda_deps: Sequence[str],
    workload_type: str = "",
    reproduce_results: bool = False,
) -> str:
    """Build a retrieval query that favors lessons relevant to this repo/run."""
    parts = [full_name, "ROCm migration build"]
    if frameworks:
        parts.append("frameworks " + " ".join(str(x) for x in frameworks[:4]))
    if cuda_deps:
        parts.append("cuda deps " + " ".join(str(x) for x in cuda_deps[:6]))
    if workload_type:
        parts.append(f"workload {workload_type}")
    if reproduce_results:
        parts.append("paper reproduction experiment selection metric verification")
    return " ".join(p.strip() for p in parts if p and p.strip())


def curate_global_lessons(
    raw_text: str,
    header: str = "CROSS-RUN LESSONS",
    max_lessons: int = 6,
    max_chars: int = 1500,
) -> str:
    """
    Convert raw mempalace lesson recall into a short, de-duplicated brief.

    The input usually contains headers and room/confidence prefixes like
    `[do c=0.72] ...`; this function strips that boilerplate so the planner
    and generated skills see only the actionable lesson text.
    """
    if not raw_text or not raw_text.strip():
        return ""

    lessons = []
    seen = set()
    used_chars = 0

    for line in raw_text.splitlines():
        cleaned = _clean_lesson_line(line)
        if not cleaned:
            continue
        norm = _normalize_for_dedupe(cleaned)
        if norm in seen:
            continue
        seen.add(norm)
        bullet = f"  - {cleaned}"
        if used_chars + len(bullet) > max_chars and lessons:
            break
        lessons.append(bullet)
        used_chars += len(bullet) + 1
        if len(lessons) >= max_lessons:
            break

    if not lessons:
        return ""

    return (
        "\n========================================\n"
        f"{header}\n"
        "========================================\n"
        + "\n".join(lessons)
        + "\n"
    )


def generate_cross_run_skill_md(learned_context: str = "") -> str:
    """Render a compact Claude skill from curated cross-run lessons."""
    learned_section = ""
    if learned_context and learned_context.strip():
        learned_section = (
            "## Retrieved Lessons For This Repo\n\n"
            "Apply the following lessons as a strong prior, but still verify them "
            "against the current repo, logs, and tool output before making a risky change.\n\n"
            f"{learned_context.strip()}\n\n"
        )

    return f"""\
---
name: cross-run-learning
description: Applies distilled cross-run ROCm lessons when planning, choosing a base image, selecting experiments, or recovering from repeated package/build/runtime failures. Use when prior runs likely contain relevant migration or reproduction lessons.
---

# Cross-Run Learning

## Quick Start

Use this skill to turn prior run experience into a compact prior for the
current repo.

1. Start from retrieved lessons, not from raw memory dumps.
2. Prefer lessons that match the current framework, package surface, or error.
3. Treat current repo evidence and tool output as ground truth when they
   contradict an older lesson.
4. Keep lessons generic and actionable; move repo-specific facts into the
   plan or run notes instead of storing them as global learning.

## Distillation Rules

- Keep `do` lessons in the form "when X happens, do Y".
- Keep `dont` lessons in the form "do not do Z because it leads to W".
- Keep `pattern` lessons only when the trigger is reusable across repos.
- Merge duplicates aggressively; one strong lesson is better than five noisy ones.
- Drop lessons that depend on one repo's filenames, one paper's wording, or a
  one-off local workaround.

## When To Trust A Lesson

- Trust it more when it matches the same framework, dependency family, or error class.
- Trust it less when it conflicts with the current README, code, or live tool output.
- Never let a lesson override verified metric checks or hard safety guards.

{learned_section}## Write New Lessons Carefully

After a run, only keep a lesson if it is:

- caused by a real failure-to-recovery transition
- reusable outside the current repo
- concrete enough that a future agent can act on it

For detailed curation criteria, see [reference.md](reference.md).
"""


def _clean_lesson_line(line: str) -> str:
    text = (line or "").strip()
    if not text:
        return ""
    if set(text) == {"="}:
        return ""
    upper = text.upper()
    if upper.startswith("CROSS-RUN LESSONS"):
        return ""
    text = re.sub(r"^\[(?:do|dont|pattern)[^\]]*\]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*[-*]\s*", "", text)
    return text.strip()


def _normalize_for_dedupe(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"\s+", " ", lowered)
    lowered = re.sub(r"[`'\".,;:()]+", "", lowered)
    return lowered
