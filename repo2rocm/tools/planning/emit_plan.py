"""EmitPlan — planner's terminal tool. Validates and persists a MigrationPlan."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from repo2rocm.core.permissions import PermissionDecision, allow
from repo2rocm.planning import MigrationPlan
from repo2rocm.tools.base import BaseTool, ToolResult, ToolUseContext


class EmitPlanInput(BaseModel):
    plan: dict[str, Any] = Field(
        ...,
        description=(
            "The full MigrationPlan as a JSON object: "
            "{repo, sha, mode, base_image, base_image_reasoning, steps[], risks[], rollback_points[]}."
        ),
    )


class EmitPlanOutput(BaseModel):
    ok: bool
    path: str = ""
    step_count: int = 0
    error: str = ""


class EmitPlan(BaseTool[EmitPlanInput, EmitPlanOutput]):
    name: ClassVar[str] = "EmitPlan"
    description: ClassVar[str] = (
        "Validate and persist the MigrationPlan. The planner agent calls this exactly "
        "ONCE at the end of its turn. Schema: mode∈{'functional','reproduce'}, "
        "steps[].id, steps[].title, steps[].agent∈{migrator,verifier,paper-reproducer,explore}, "
        "steps[].inputs(dict), steps[].skills(list[str]), steps[].depends_on(list[str]), "
        "steps[].success_marker(str). After EmitPlan returns ok=true, end your turn."
    )
    input_model: ClassVar[type[BaseModel]] = EmitPlanInput
    max_result_size_chars: ClassVar[int] = 8_000

    def is_concurrency_safe(self, parsed: EmitPlanInput) -> bool:
        return False

    def is_read_only(self, parsed: EmitPlanInput) -> bool:
        return False

    def check_permissions(
        self, parsed: EmitPlanInput, ctx: ToolUseContext
    ) -> PermissionDecision:
        # EmitPlan only writes a JSON plan into ctx.workdir/plans/. It cannot touch
        # the user's repo or system. Allow regardless of permission_mode so a PLAN-
        # mode planner is never trapped in a deny→retry loop.
        return allow("EmitPlan is an internal coordination tool (writes only to workdir)")

    async def call(
        self, parsed: EmitPlanInput, ctx: ToolUseContext
    ) -> ToolResult[EmitPlanOutput]:
        try:
            plan = MigrationPlan.model_validate(parsed.plan)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                data=EmitPlanOutput(ok=False, error=str(exc)),
                text=f"Plan validation failed: {exc}",
                is_error=True,
            )

        out_dir: Path = ctx.workdir / "plans"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "migration_plan.json"
        path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")

        ctx.options["migration_plan"] = plan
        ctx.options["migration_plan_path"] = str(path)

        return ToolResult(
            data=EmitPlanOutput(ok=True, path=str(path), step_count=len(plan.steps)),
            text=(
                f"MigrationPlan stored: {path} ({len(plan.steps)} steps, "
                f"base_image={plan.base_image}). End your turn now."
            ),
        )
