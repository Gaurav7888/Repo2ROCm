"""PyPIVersions — query PyPI for available versions + classifiers.

Always concurrency-safe (HTTP GET).
"""
from __future__ import annotations

from typing import ClassVar

import httpx
from pydantic import BaseModel, Field

from repo2rocm.tools.base import BaseTool, ToolResult, ToolUseContext


class PyPIInput(BaseModel):
    package: str
    limit: int = Field(12, description="Max number of recent versions to return.")


class PyPIOutput(BaseModel):
    package: str
    versions: list[str]
    requires_python: str | None = None
    yanked: list[str] = []
    classifiers: list[str] = []
    not_found: bool = False


class PyPIVersions(BaseTool[PyPIInput, PyPIOutput]):
    name: ClassVar[str] = "PyPIVersions"
    description: ClassVar[str] = (
        "Query PyPI for available versions, classifiers, and Python compatibility. "
        "ALWAYS call this before `pip install` of a CUDA-flavored wheel."
    )
    input_model: ClassVar[type[BaseModel]] = PyPIInput
    max_result_size_chars: ClassVar[int] = 4_000
    interrupt_behavior: ClassVar[str] = "cancel"

    def is_concurrency_safe(self, parsed: PyPIInput) -> bool:
        return True

    def is_read_only(self, parsed: PyPIInput) -> bool:
        return True

    async def call(self, parsed: PyPIInput, ctx: ToolUseContext) -> ToolResult[PyPIOutput]:
        url = f"https://pypi.org/pypi/{parsed.package}/json"
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.get(url)
                if r.status_code == 404:
                    out = PyPIOutput(package=parsed.package, versions=[], not_found=True)
                    return ToolResult(
                        data=out, text=f"PyPI: {parsed.package!r} not found.", is_error=True
                    )
                r.raise_for_status()
                data = r.json()
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                data=PyPIOutput(package=parsed.package, versions=[]),
                text=f"PyPI lookup failed: {exc}",
                is_error=True,
            )

        info = data.get("info", {})
        releases = data.get("releases", {}) or {}
        versions = sorted(releases.keys(), reverse=True)[: parsed.limit]
        yanked = [v for v in versions if all(r.get("yanked") for r in releases.get(v) or [])]
        out = PyPIOutput(
            package=parsed.package,
            versions=versions,
            requires_python=info.get("requires_python"),
            yanked=yanked,
            classifiers=(info.get("classifiers") or [])[:20],
        )
        ctx_gate = getattr(ctx, "gate_state", None)
        if ctx_gate is not None and hasattr(ctx_gate, "mark_pypi"):
            ctx_gate.mark_pypi(parsed.package)
        body = (
            f"PyPI {parsed.package}\n"
            f"  versions (newest first): {', '.join(versions)}\n"
            f"  requires_python: {out.requires_python}\n"
            f"  yanked: {', '.join(yanked) or '-'}\n"
            f"  classifiers: {len(out.classifiers)}"
        )
        return ToolResult(data=out, text=body)
