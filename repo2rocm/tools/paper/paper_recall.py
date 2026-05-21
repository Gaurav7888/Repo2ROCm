"""PaperRecall — load the PaperContext (saved by paper-research) inside the reproducer agent."""
from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from repo2rocm.paper import PaperCorpus
from repo2rocm.paper.types import PaperContext
from repo2rocm.tools.base import BaseTool, ToolResult, ToolUseContext


class PaperRecallInput(BaseModel):
    arxiv_id: str = Field("", description="If empty, return the most recent saved context.")


class PaperRecallOutput(BaseModel):
    found: bool
    context: dict[str, Any] = {}
    error: str = ""


class PaperRecall(BaseTool[PaperRecallInput, PaperRecallOutput]):
    name: ClassVar[str] = "PaperRecall"
    description: ClassVar[str] = (
        "Load a previously saved PaperContext from work_dir/papers/. The reproducer "
        "agent calls this once to fetch the chosen experiment and metric specs."
    )
    input_model: ClassVar[type[BaseModel]] = PaperRecallInput
    max_result_size_chars: ClassVar[int] = 16_000

    def is_concurrency_safe(self, parsed: PaperRecallInput) -> bool:
        return True

    def is_read_only(self, parsed: PaperRecallInput) -> bool:
        return True

    async def call(
        self, parsed: PaperRecallInput, ctx: ToolUseContext
    ) -> ToolResult[PaperRecallOutput]:
        opt_ctx = ctx.options.get("paper_context")
        if isinstance(opt_ctx, PaperContext):
            if not parsed.arxiv_id or opt_ctx.metadata.arxiv_id == parsed.arxiv_id:
                return ToolResult(
                    data=PaperRecallOutput(found=True, context=opt_ctx.model_dump()),
                    text=opt_ctx.render_for_reproducer(),
                )

        corpus = PaperCorpus(Path(ctx.workdir) / "papers")
        result = None
        if parsed.arxiv_id:
            result = corpus.load(parsed.arxiv_id)
        else:
            jsons = sorted(
                corpus.root.glob("*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for path in jsons:
                try:
                    result = PaperContext.model_validate_json(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if result is not None:
                    break
        if result is None:
            return ToolResult(
                data=PaperRecallOutput(found=False, error="no context saved"),
                text="No saved PaperContext found.",
                is_error=True,
            )
        return ToolResult(
            data=PaperRecallOutput(found=True, context=result.model_dump()),
            text=result.render_for_reproducer(),
        )
