"""
Graphify provider — Stage 4 of the memory layer.

Builds a per-repo deterministic code graph (tree-sitter, no LLM) under
`<repo_path>/graphify-out/graph.json` and exposes a small query API:

    gp = GraphifyProvider.create(repo_path)
    gp.build_or_refresh()
    snippets = gp.query("entry script for training", token_budget=1500)

The graph captures files, classes, functions, imports, calls — exactly what we
need to answer questions like "find the entry point" or "find code that uses
flash-attn" without re-walking the repo each time.

This module deliberately does NOT touch paper text — paper recall is delegated
to mempalace (`paper_extracts` room) because graphify's semantic-paper pipeline
requires the `/graphify` skill (LLM-driven). Code-only is enough to replace the
file-listing + grep dumps in the planner/paper agent.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple


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
            }
        except Exception:
            return {"graph_json": self.graph_json, "nodes": 0, "edges": 0}
