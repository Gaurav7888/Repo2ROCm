"""InvokeSkill — on-demand skill body loader.

Skills are surfaced to the model as a frontmatter-only menu. When the agent
decides a skill is relevant, it calls `InvokeSkill(name="<skill>")` to receive
the full markdown body in its tool-result message.
"""
from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from repo2rocm.skills import SkillCatalog, discover_skills, load_skill_body
from repo2rocm.tools.base import BaseTool, ToolResult, ToolUseContext


class InvokeSkillInput(BaseModel):
    name: str = Field(..., description="Skill name (no leading slash).")


class InvokeSkillOutput(BaseModel):
    name: str
    found: bool
    body: str = ""
    when_to_use: str = ""


_CACHE: SkillCatalog | None = None


def _catalog() -> SkillCatalog:
    global _CACHE
    if _CACHE is None:
        _CACHE = discover_skills()
    return _CACHE


class InvokeSkill(BaseTool[InvokeSkillInput, InvokeSkillOutput]):
    name: ClassVar[str] = "InvokeSkill"
    description: ClassVar[str] = (
        "Load the full body of a named skill on demand. Use this when you need "
        "the detailed reference content for a skill you saw in the startup menu. "
        "Example: InvokeSkill(name='nvidia_alternatives')."
    )
    input_model: ClassVar[type[BaseModel]] = InvokeSkillInput
    max_result_size_chars: ClassVar[int] = 30_000

    def is_concurrency_safe(self, parsed: InvokeSkillInput) -> bool:
        return True

    def is_read_only(self, parsed: InvokeSkillInput) -> bool:
        return True

    async def call(
        self, parsed: InvokeSkillInput, ctx: ToolUseContext
    ) -> ToolResult[InvokeSkillOutput]:
        cat = _catalog()
        key = parsed.name.lstrip("/").strip()
        manifest = cat.manifests.get(key)
        if manifest is None:
            return ToolResult(
                data=InvokeSkillOutput(name=key, found=False),
                text=(
                    f"Skill {key!r} not found. Available: "
                    + ", ".join(sorted(cat.manifests))
                ),
                is_error=True,
            )
        try:
            body = load_skill_body(manifest)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                data=InvokeSkillOutput(name=key, found=False),
                text=f"Failed to load skill body: {exc}",
                is_error=True,
            )
        return ToolResult(
            data=InvokeSkillOutput(
                name=key, found=True, body=body, when_to_use=manifest.when_to_use
            ),
            text=f"# Skill: {key}\n\n{body}",
        )
