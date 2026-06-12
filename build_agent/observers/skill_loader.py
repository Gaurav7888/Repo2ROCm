"""
Disk-backed loader for the Observer's role prompt, reviewer instructions,
and skill catalog.

Why this exists
---------------
The observer used to encode skills, triggers, and decision rules in Python
data classes and regex-based heuristics. That made the observer brittle and
hard to evolve — every new heuristic required a code change.

The redesign treats the observer as a small LLM agent driven by *content*
files. The Markdown files in `observers/prompts/` and `observers/skills/`
describe the role, the JSON contract, and the catalogue of skills the
reviewer can invoke. They are loaded once at module import and concatenated
into the system prompt sent to the reviewer LLM.

Edits to those Markdown files take effect on the next sidecar restart with
no Python code change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional


_HERE = os.path.dirname(os.path.abspath(__file__))
_PROMPTS_DIR = os.path.join(_HERE, "prompts")
_SKILLS_DIR = os.path.join(_HERE, "skills")


@dataclass(frozen=True)
class SkillCard:
    name: str
    summary: str           # one-line summary derived from the first non-empty line
    full_text: str         # full Markdown body (for the system prompt)


def _read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except Exception:
        return ""


def _summary_from_markdown(text: str) -> str:
    """Extract a one-line summary (the 'Use when:' / first prose line)."""
    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        # Prefer 'Use when:' or 'Behavior:' lines as the headline.
        if s.lower().startswith("**use when:**"):
            return s.replace("**Use when:**", "").strip()[:200]
        if s.lower().startswith("**behavior:**"):
            return s.replace("**Behavior:**", "").strip()[:200]
        return s[:200]
    return ""


def load_role_prompt() -> str:
    return _read_file(os.path.join(_PROMPTS_DIR, "observer_role.md")).strip()


def load_reviewer_instructions() -> str:
    return _read_file(os.path.join(_PROMPTS_DIR, "reviewer_instructions.md")).strip()


def load_skill_cards() -> List[SkillCard]:
    cards: List[SkillCard] = []
    if not os.path.isdir(_SKILLS_DIR):
        return cards
    for fname in sorted(os.listdir(_SKILLS_DIR)):
        if not fname.endswith(".md"):
            continue
        path = os.path.join(_SKILLS_DIR, fname)
        body = _read_file(path).strip()
        if not body:
            continue
        # Skill name = filename without extension. The body stays untouched
        # so the reviewer reads the full guidance, not just the headline.
        name = os.path.splitext(fname)[0]
        cards.append(SkillCard(
            name=name,
            summary=_summary_from_markdown(body),
            full_text=body,
        ))
    return cards


def build_system_prompt(role: Optional[str] = None,
                        instructions: Optional[str] = None,
                        skills: Optional[List[SkillCard]] = None) -> str:
    """Concatenate role + instructions + skill catalogue into one prompt."""
    role_text = role if role is not None else load_role_prompt()
    inst_text = instructions if instructions is not None else load_reviewer_instructions()
    skill_cards = skills if skills is not None else load_skill_cards()

    catalog_lines: List[str] = ["# Skill Catalog", ""]
    catalog_lines.append("Choose exactly one skill name from this list:\n")
    for card in skill_cards:
        catalog_lines.append(f"- **{card.name}** — {card.summary}")
    catalog_lines.append("")
    catalog_lines.append("Full skill cards (read these to decide):")
    catalog_lines.append("")
    for card in skill_cards:
        catalog_lines.append(card.full_text.strip())
        catalog_lines.append("")

    parts: List[str] = []
    if role_text:
        parts.append(role_text)
    parts.append("---")
    parts.append("\n".join(catalog_lines).strip())
    parts.append("---")
    if inst_text:
        parts.append(inst_text)
    return "\n\n".join(p for p in parts if p).strip()


def skill_names() -> List[str]:
    return [card.name for card in load_skill_cards()]
