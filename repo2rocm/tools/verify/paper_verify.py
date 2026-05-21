"""PaperVerify — typed verdict for paper-result reproduction.

Replaces the textual PAPER_RESULT_REPRODUCED / NOT_REPRODUCED markers with a typed
verdict the Coordinator can branch on.
"""
from __future__ import annotations

import re
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from repo2rocm.paper.types import PaperContext
from repo2rocm.tools.base import BaseTool, ToolResult, ToolUseContext


class MetricSpec(BaseModel):
    name: str
    expected_value: float
    tolerance: float = 0.0


class PaperVerifyInput(BaseModel):
    log_path: str = Field(..., description="Path to the experiment stdout log (inside container).")
    metrics: list[MetricSpec]


class PaperVerifyOutput(BaseModel):
    verdict: Literal["reproduced", "not_reproduced", "unknown"]
    found: dict[str, float]
    deltas: dict[str, float]
    reason: str


_FLOAT_RE = re.compile(r"([-+]?\d*\.\d+|[-+]?\d+\.?\d*)")


def _coerce_paper_context(obj) -> PaperContext | None:
    if obj is None:
        return None
    if isinstance(obj, PaperContext):
        return obj
    try:
        return PaperContext.model_validate(obj)
    except Exception:
        return None


def _norm_metric(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


class PaperVerify(BaseTool[PaperVerifyInput, PaperVerifyOutput]):
    name: ClassVar[str] = "PaperVerify"
    description: ClassVar[str] = (
        "Parse an experiment log and compare measured metrics against expected paper values. "
        "Returns a typed verdict — never fabricate numbers; if parsing fails, returns 'unknown'."
    )
    input_model: ClassVar[type[BaseModel]] = PaperVerifyInput
    max_result_size_chars: ClassVar[int] = 4_000

    def is_concurrency_safe(self, parsed: PaperVerifyInput) -> bool:
        return True

    def is_read_only(self, parsed: PaperVerifyInput) -> bool:
        return True

    def validate_semantic(self, parsed: PaperVerifyInput, ctx: ToolUseContext) -> str | None:
        if str(ctx.options.get("run_mode") or "").lower() != "reproduce":
            return None
        low_log_path = parsed.log_path.lower()
        if "formatted" in low_log_path or "synthetic" in low_log_path:
            return (
                "Reproduce-mode guard: refusing to verify a synthetic/formatted log. "
                "Verify the real experiment log instead."
            )
        paper_ctx = _coerce_paper_context(ctx.options.get("paper_context"))
        chosen = paper_ctx.chosen() if paper_ctx is not None else None
        if chosen is None or chosen.headline_metric is None or chosen.headline_metric.value is None:
            return None
        allowed_metrics = [chosen.headline_metric, *chosen.related_metrics]
        headline_ok = False
        for metric in parsed.metrics:
            match = next(
                (
                    allowed
                    for allowed in allowed_metrics
                    if allowed.value is not None
                    and _norm_metric(allowed.name) == _norm_metric(metric.name)
                    and abs(float(allowed.value) - float(metric.expected_value)) <= 1e-6
                ),
                None,
            )
            if match is None:
                return (
                    "Reproduce-mode guard: PaperVerify metrics must come directly from the "
                    "chosen PaperContext experiment."
                )
            if metric.tolerance > float(match.default_tolerance) + 1e-9:
                return (
                    "Reproduce-mode guard: PaperVerify tolerance exceeds the chosen "
                    f"experiment's default tolerance for `{match.name}`."
                )
            if _norm_metric(metric.name) == _norm_metric(chosen.headline_metric.name):
                headline_ok = True
        if parsed.metrics and not headline_ok:
            return (
                "Reproduce-mode guard: PaperVerify must include the chosen experiment's "
                "headline metric."
            )
        return None

    async def call(
        self, parsed: PaperVerifyInput, ctx: ToolUseContext
    ) -> ToolResult[PaperVerifyOutput]:
        if ctx.sandbox is None:
            return ToolResult(
                data=PaperVerifyOutput(
                    verdict="unknown", found={}, deltas={}, reason="no sandbox"
                ),
                text="no sandbox attached",
                is_error=True,
            )
        cat = await ctx.sandbox.exec(f"cat {parsed.log_path}", timeout_s=30.0)
        if cat.exit_code != 0:
            return ToolResult(
                data=PaperVerifyOutput(
                    verdict="unknown",
                    found={},
                    deltas={},
                    reason=f"log read failed: {cat.stderr[:300]}",
                ),
                text=f"could not read {parsed.log_path}",
                is_error=True,
            )
        body = cat.stdout
        found: dict[str, float] = {}
        deltas: dict[str, float] = {}
        for m in parsed.metrics:
            # find lines mentioning the metric name (case-insensitive)
            pattern = re.compile(rf"{re.escape(m.name)}\s*[=:]\s*({_FLOAT_RE.pattern})", re.IGNORECASE)
            match = pattern.search(body)
            if not match:
                continue
            try:
                val = float(match.group(1))
            except ValueError:
                continue
            found[m.name] = val
            deltas[m.name] = val - m.expected_value
        if not found:
            return ToolResult(
                data=PaperVerifyOutput(
                    verdict="unknown",
                    found={},
                    deltas={},
                    reason="could not parse any metric from log",
                ),
                text=(
                    "Could not parse any metric value from the log. "
                    "Refusing to fabricate; verdict=unknown."
                ),
            )
        all_within = all(
            abs(found[m.name] - m.expected_value) <= m.tolerance
            for m in parsed.metrics
            if m.name in found
        )
        verdict: Literal["reproduced", "not_reproduced", "unknown"] = (
            "reproduced" if all_within else "not_reproduced"
        )
        reason_parts = [
            f"{name}: actual={val:.4f}, expected={next(m for m in parsed.metrics if m.name == name).expected_value:.4f}, delta={deltas[name]:+.4f}"
            for name, val in found.items()
        ]
        reason = "; ".join(reason_parts)
        return ToolResult(
            data=PaperVerifyOutput(
                verdict=verdict, found=found, deltas=deltas, reason=reason
            ),
            text=f"verdict={verdict}\n{reason}",
            is_error=False,
        )
