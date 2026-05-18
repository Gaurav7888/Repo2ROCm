"""Two-phase skill loader.

Phase 1 (startup): read frontmatter only, build menu for system prompt.
Phase 2 (invocation): load full body when `/skill-name` is invoked.
"""
from repo2rocm.skills.loader import (
    SkillManifest,
    SkillCatalog,
    discover_skills,
    load_skill_body,
)

__all__ = [
    "SkillManifest",
    "SkillCatalog",
    "discover_skills",
    "load_skill_body",
]
