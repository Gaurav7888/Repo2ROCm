"""Typed migration plan (the planner's output contract)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

AgentName = Literal[
    "migrator", "verifier", "paper-reproducer", "explore",
]


class PlanStep(BaseModel):
    """A single executable unit in the plan."""

    id: str = Field(..., description="Stable step id, e.g. 'S1', 'S2a'.")
    title: str = Field(..., description="Short human title.")
    agent: AgentName = Field(..., description="Agent responsible for executing this step.")
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Step-specific inputs the assigned agent should treat as authoritative. "
            "Examples: {file_path, edit_diff, packages, image, command, log_path, metrics}."
        ),
    )
    skills: list[str] = Field(
        default_factory=list,
        description="Skill names to preload before this step runs.",
    )
    tools_hint: list[str] = Field(
        default_factory=list,
        description="Suggested tools (not enforced; the agent's allowed_tools still wins).",
    )
    success_marker: str = Field(
        "",
        description="Free-form success criterion (e.g. 'ROCM_ENV_VERIFIED', 'all packages installed').",
    )
    depends_on: list[str] = Field(default_factory=list, description="Predecessor step ids.")
    parallel_group: str | None = Field(
        None,
        description="Steps sharing a group name can run in parallel.",
    )
    timeout_s: int = Field(1800, ge=60, le=14_400)
    notes: str = ""


class MigrationPlan(BaseModel):
    """The planner's typed output."""

    repo: str
    sha: str = ""
    mode: Literal["functional", "reproduce"]
    base_image: str
    base_image_reasoning: str = ""

    steps: list[PlanStep]
    risks: list[str] = Field(default_factory=list)
    rollback_points: list[str] = Field(
        default_factory=list,
        description="Step ids after which DockerCommit should be called.",
    )

    def step(self, step_id: str) -> PlanStep | None:
        for s in self.steps:
            if s.id == step_id:
                return s
        return None

    def render_for_executor(self, *, limit: int = 80) -> str:
        """Compact textual rendering injected into executor agent prompts."""
        lines = [f"# Migration Plan ({self.mode})  base_image={self.base_image}"]
        if self.base_image_reasoning:
            lines.append(f"  reason: {self.base_image_reasoning}")
        if self.risks:
            lines.append("Risks:")
            for r in self.risks[:5]:
                lines.append(f"  - {r}")
        lines.append("")
        lines.append("Steps:")
        shown = 0
        for s in self.steps:
            shown += 1
            if shown > limit:
                lines.append(f"  ... ({len(self.steps) - limit} more steps elided)")
                break
            deps = f"  deps={','.join(s.depends_on)}" if s.depends_on else ""
            par = f"  parallel={s.parallel_group}" if s.parallel_group else ""
            mark = f"  marker='{s.success_marker}'" if s.success_marker else ""
            lines.append(f"  [{s.id}] ({s.agent}) {s.title}{deps}{par}{mark}")
            if s.inputs:
                inputs_preview = ", ".join(f"{k}={_short(v)}" for k, v in list(s.inputs.items())[:6])
                lines.append(f"      inputs: {inputs_preview}")
            if s.skills:
                lines.append(f"      skills: {', '.join(s.skills)}")
            if s.notes:
                lines.append(f"      note: {s.notes[:200]}")
        return "\n".join(lines)


def _short(v: Any) -> str:
    s = str(v)
    return (s[:80] + "...") if len(s) > 80 else s
