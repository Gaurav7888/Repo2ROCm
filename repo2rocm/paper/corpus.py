"""Persistent paper corpus under work_dir/papers/."""
from __future__ import annotations

from pathlib import Path

from repo2rocm.paper.types import PaperContext


class PaperCorpus:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def context_path(self, arxiv_id: str) -> Path:
        return self.root / f"{arxiv_id or 'paper'}.json"

    def save(self, ctx: PaperContext) -> Path:
        path = self.context_path(ctx.metadata.arxiv_id)
        path.write_text(ctx.model_dump_json(indent=2), encoding="utf-8")
        return path

    def load(self, arxiv_id: str = "") -> PaperContext | None:
        path = self.context_path(arxiv_id)
        if not path.is_file():
            return None
        try:
            return PaperContext.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            return None
