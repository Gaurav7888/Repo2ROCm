from repo2rocm.tools.base import register_tool
from repo2rocm.tools.paper.emit_paper_context import EmitPaperContext
from repo2rocm.tools.paper.paper_fetch import PaperFetch
from repo2rocm.tools.paper.paper_outline import PaperOutline
from repo2rocm.tools.paper.paper_read import PaperRead
from repo2rocm.tools.paper.paper_recall import PaperRecall


def register_paper_tools() -> None:
    for cls in (
        PaperFetch,
        PaperOutline,
        PaperRead,
        PaperRecall,
        EmitPaperContext,
    ):
        register_tool(cls)


__all__ = [
    "EmitPaperContext",
    "PaperFetch",
    "PaperOutline",
    "PaperRead",
    "PaperRecall",
    "register_paper_tools",
]
