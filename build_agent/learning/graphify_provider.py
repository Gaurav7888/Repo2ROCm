"""
Graphify provider — static corpus layer.

Graphify owns *static* project artifacts that exist before the run starts:
  - codebase
  - config files
  - README/docs
  - research paper

Mempalace owns *dynamic* run state:
  - commands executed
  - failures / fixes
  - decisions
  - metrics
  - cross-run lessons

This provider therefore exposes two retrieval surfaces:

1. Code graph (tree-sitter / NetworkX)
   - `build_or_refresh()`
   - `query(question)`
   - `list_entry_scripts()`

2. Paper index (lightweight local sidecar under `graphify-out/`)
   - `index_paper_text(paper_text)`
   - `query_paper(question)`

The paper path is intentionally simple: chunk the PDF text into JSONL with
section hints and score chunks lexically. It is not as semantically rich as the
full `/graphify` skill pipeline, but it gives us a stable, local, *graphify-
owned* paper corpus so mempalace no longer needs to store the raw paper body.
"""

from __future__ import annotations

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class _NoopGraph:
    enabled = False
    graph_json: str = ""

    def __getattr__(self, n):
        def _noop(*a, **kw):
            return ""
        return _noop


class GraphifyProvider:
    enabled = True

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).resolve()
        self.out_dir = self.repo_path / "graphify-out"
        self.graph_json = str(self.out_dir / "graph.json")
        self._graph = None  # lazy NetworkX graph
        self.paper_chunks_jsonl = str(self.out_dir / "paper_chunks.jsonl")
        self.paper_index_meta = str(self.out_dir / "paper_index_meta.json")

    @classmethod
    def create(cls, repo_path: str) -> "GraphifyProvider":
        try:
            inst = cls(repo_path)
            return inst
        except Exception as e:
            print(f"[graphify] disabled (init failed: {e})")
            return _NoopGraph()  # type: ignore[return-value]

    # ── Build ────────────────────────────────────────────────────────────────

    def build_or_refresh(self, force: bool = False) -> bool:
        """Build graph.json if missing (or force=True). Returns True on success."""
        if not force and os.path.exists(self.graph_json):
            return True
        try:
            from graphify.detect import detect
            from graphify.extract import extract
            from graphify.build import build_from_json
            from graphify.cluster import cluster
            from graphify.export import to_json
        except Exception as e:
            print(f"[graphify] not installed: {e}")
            return False

        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            det = detect(self.repo_path)
            code_paths = [Path(p) for p in det.get("files", {}).get("code", [])]
            if not code_paths:
                print("[graphify] no code files detected")
                return False
            ast = extract(code_paths)
            G = build_from_json(ast)
            comms = cluster(G)
            to_json(G, comms, self.graph_json)
            print(f"[graphify] built graph: {len(G.nodes)} nodes, "
                  f"{len(G.edges)} edges → {self.graph_json}")
            return True
        except Exception as e:
            print(f"[graphify] build failed: {e}")
            return False

    # ── Query ────────────────────────────────────────────────────────────────

    def _load(self):
        if self._graph is not None:
            return self._graph
        try:
            from graphify.serve import _load_graph
            self._graph = _load_graph(Path(self.graph_json))
            return self._graph
        except Exception as e:
            print(f"[graphify] _load_graph failed: {e}")
            return None

    def query(self, question: str, token_budget: int = 1500,
              depth: int = 1, top_seeds: int = 6) -> str:
        """
        Return a compact text view of the subgraph most relevant to `question`.
        Uses graphify's keyword scoring + BFS expansion + subgraph-to-text.
        """
        if not question:
            return ""
        G = self._load()
        if G is None:
            return ""
        try:
            from graphify.serve import _score_nodes, _bfs, _subgraph_to_text
            terms = [t for t in question.split() if len(t) > 2][:8]
            scored = _score_nodes(G, terms)
            seeds = [nid for _, nid in scored[:top_seeds]]
            if not seeds:
                return ""
            nodes, edges = _bfs(G, seeds, depth=depth)
            return _subgraph_to_text(G, nodes, edges, token_budget=token_budget)
        except Exception as e:
            print(f"[graphify] query failed: {e}")
            return ""

    # ── Paper index ───────────────────────────────────────────────────────────

    @staticmethod
    def _paper_section_hint(text: str) -> str:
        """Best-effort section label from a paper chunk."""
        if not text:
            return ""
        lines = [ln.strip() for ln in text.splitlines()[:30] if ln.strip()]
        patterns = [
            r"^(Table\s+\d+[:.\s-].*)$",
            r"^(Figure\s+\d+[:.\s-].*)$",
            r"^(Appendix\s+[A-Z0-9]+[:.\s-].*)$",
            r"^(Section\s+\d+(\.\d+)*[:.\s-].*)$",
            r"^([A-Z]\.\d+[^a-z].*)$",
            r"^(\d+(\.\d+)+\s+.*)$",
            r"^(Abstract)$",
            r"^(Introduction)$",
            r"^(Methods?|Experiments?|Results?|Evaluation|Ablations?)$",
        ]
        for ln in lines:
            for pat in patterns:
                m = re.match(pat, ln, flags=re.IGNORECASE)
                if m:
                    return m.group(1)[:120]
        return lines[0][:120] if lines else ""

    @staticmethod
    def _paper_score(question: str, rec: Dict[str, str]) -> float:
        """Simple lexical scoring for paper chunks."""
        q_terms = {
            t.lower() for t in re.findall(r"[A-Za-z0-9_]+", question)
            if len(t) >= 3
        }
        if not q_terms:
            return 0.0
        text = " ".join([
            rec.get("section_hint", ""),
            rec.get("text", "")[:3000],
            rec.get("source_file", ""),
        ]).lower()
        text_terms = set(re.findall(r"[A-Za-z0-9_]+", text))
        overlap = len(q_terms & text_terms)
        if overlap == 0:
            return 0.0

        # Extra boosts for paper-structured hints
        boost = 0.0
        hint = (rec.get("section_hint") or "").lower()
        if "table" in hint and any(t in q_terms for t in {"table", "metric", "results", "speedup", "accuracy"}):
            boost += 2.0
        if "appendix" in hint and any(t in q_terms for t in {"hyperparameters", "appendix", "seed", "batch", "epochs"}):
            boost += 1.5
        if any(t in text for t in ["accuracy", "f1", "speedup", "latency", "throughput", "batch", "learning", "gamma"]):
            boost += 0.5
        return overlap + boost

    def index_paper_text(self, paper_text: str, source_file: str = "paper.pdf",
                         chunk_chars: int = 4000) -> bool:
        """
        Store a lightweight graphify-owned paper index under graphify-out/.

        The index is a JSONL sidecar, not part of graph.json:
            {"id", "source_file", "chunk_id", "char_offset", "section_hint", "text"}
        """
        if not paper_text or not paper_text.strip():
            return False
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            n = len(paper_text)
            i = 0
            chunk_id = 0
            lines = []
            while i < n:
                seg = paper_text[i:i + chunk_chars]
                if i + chunk_chars < n:
                    nl = seg.rfind("\n", chunk_chars // 2)
                    if nl > 0:
                        seg = seg[:nl]
                rec = {
                    "id": f"paper:{Path(source_file).name}:chunk_{chunk_id}",
                    "source_file": source_file,
                    "chunk_id": chunk_id,
                    "char_offset": i,
                    "section_hint": self._paper_section_hint(seg),
                    "text": seg,
                }
                lines.append(json.dumps(rec, ensure_ascii=False))
                i += len(seg) if seg else chunk_chars
                chunk_id += 1
            with open(self.paper_chunks_jsonl, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            meta = {
                "source_file": source_file,
                "chunk_count": chunk_id,
                "chars": n,
            }
            with open(self.paper_index_meta, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            print(f"[graphify] indexed paper: {chunk_id} chunks → {self.paper_chunks_jsonl}")
            return True
        except Exception as e:
            print(f"[graphify] paper indexing failed: {e}")
            return False

    def _load_paper_records(self) -> List[Dict[str, str]]:
        if not os.path.exists(self.paper_chunks_jsonl):
            return []
        out: List[Dict[str, str]] = []
        try:
            with open(self.paper_chunks_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        continue
        except Exception as e:
            print(f"[graphify] failed loading paper records: {e}")
        return out

    def query_paper(self, question: str, token_budget: int = 3000,
                    max_chunks: int = 6, per_chunk_max_chars: int = 1500) -> str:
        """Query the graphify-owned paper index."""
        if not question:
            return ""
        records = self._load_paper_records()
        if not records:
            return ""
        scored = [
            (self._paper_score(question, rec), rec)
            for rec in records
        ]
        scored = [x for x in scored if x[0] > 0]
        scored.sort(key=lambda t: t[0], reverse=True)
        if not scored:
            return ""

        used = 0
        max_chars = max(token_budget * 4, per_chunk_max_chars)
        parts = [
            "========================================",
            "PAPER CONTEXT (graphify index)",
            "========================================",
        ]
        for _score, rec in scored[:max_chunks]:
            text = (rec.get("text") or "").strip()
            if not text:
                continue
            if len(text) > per_chunk_max_chars:
                text = text[:per_chunk_max_chars] + " …[trunc]"
            block = (
                f"  [paper:{rec.get('section_hint') or rec.get('chunk_id')}] "
                f"{text}"
            )
            if used + len(block) > max_chars:
                remaining = max_chars - used
                if remaining <= 80:
                    break
                block = block[: remaining - 16] + " …[trunc]"
            parts.append(block)
            used += len(block) + 2
        return "\n".join(parts) + "\n"

    def list_entry_scripts(self, max_files: int = 12) -> List[str]:
        """
        Heuristic list of likely entry scripts — files containing `if __name__
        == "__main__"` or top-level `main()` / argparse, falling back to
        graphify god-nodes.
        """
        out: List[str] = []
        try:
            for fp in self.repo_path.glob("*.py"):
                try:
                    txt = fp.read_text(encoding="utf-8", errors="ignore")
                    if "__main__" in txt or "argparse" in txt or "@hydra.main" in txt:
                        out.append(fp.name)
                        if len(out) >= max_files:
                            return out
                except Exception:
                    continue
        except Exception:
            pass
        return out

    def stats(self) -> dict:
        try:
            G = self._load()
            return {
                "graph_json": self.graph_json,
                "nodes": len(G.nodes) if G is not None else 0,
                "edges": len(G.edges) if G is not None else 0,
                "paper_chunks_jsonl": self.paper_chunks_jsonl,
                "paper_chunks": len(self._load_paper_records()),
            }
        except Exception:
            return {
                "graph_json": self.graph_json,
                "nodes": 0,
                "edges": 0,
                "paper_chunks_jsonl": self.paper_chunks_jsonl,
                "paper_chunks": 0,
            }
