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

This provider therefore exposes two retrieval surfaces:

1. Code graph (tree-sitter / NetworkX)
   - `build_or_refresh()`
   - `query(question)`
   - `list_entry_scripts()`

2. Paper index (lightweight local sidecar under `graphify-out/`)
   - `index_paper_text(paper_text)`
   - `query_paper(question)`

3. Repo text corpus (README / config / source text sidecar under `graphify-out/`)
   - `index_repo_corpus()`
   - `query_repo_corpus(question, scope=...)`

The paper path is intentionally simple: chunk the PDF text into JSONL with
section hints and score chunks lexically. It is not as semantically rich as the
full `/graphify` skill pipeline, but it gives us a stable, local, *graphify-
owned* paper corpus so mempalace no longer needs to store the raw paper body.
"""

from __future__ import annotations

import os
import json
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class _NoopGraph:
    enabled = False
    graph_json: str = ""

    def __getattr__(self, n):
        def _noop(*a, **kw):
            return ""
        return _noop


class GraphifyProvider:
    enabled = True

    _REPO_TEXT_EXTS = {
        ".py", ".sh", ".bash", ".zsh",
        ".md", ".rst", ".txt",
        ".yaml", ".yml", ".toml", ".cfg", ".ini", ".json", ".jsonnet",
        ".js", ".jsx", ".ts", ".tsx",
        ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".cu", ".cuh",
        ".go", ".rs", ".java", ".scala", ".swift",
        ".ipynb",
    }
    _REPO_SKIP_DIRS = {
        ".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build",
        "site-packages", "checkpoints", "wandb", "outputs", "graphify-out",
    }

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).resolve()
        self.out_dir = self.repo_path / "graphify-out"
        self.graph_json = str(self.out_dir / "graph.json")
        self._graph = None  # lazy NetworkX graph
        self.paper_chunks_jsonl = str(self.out_dir / "paper_chunks.jsonl")
        self.paper_index_meta = str(self.out_dir / "paper_index_meta.json")
        self.repo_chunks_jsonl = str(self.out_dir / "repo_chunks.jsonl")
        self.repo_index_meta = str(self.out_dir / "repo_index_meta.json")

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

    def index_paper_sources(self, sources: List[Dict[str, Any]],
                            chunk_chars: int = 4000) -> bool:
        """
        Store one or more normalized paper sources under graphify-out/.

        Each source entry is expected to contain:
          - source_kind
          - source_label
          - source_file
          - text
          - metadata
        """
        if not sources:
            return False
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            chunk_id = 0
            total_chars = 0
            meta_sources: List[Dict[str, Any]] = []
            lines = []
            seen_chunks = set()
            for source in sources:
                text = str(source.get("text") or "").strip()
                if not text:
                    continue
                source_kind = str(source.get("source_kind") or "paper")
                source_label = str(
                    source.get("source_label")
                    or source.get("source_file")
                    or f"{source_kind}_{chunk_id}"
                )
                source_file = str(source.get("source_file") or source_label)
                metadata = dict(source.get("metadata") or {})
                n = len(text)
                total_chars += n
                meta_sources.append({
                    "source_kind": source_kind,
                    "source_label": source_label,
                    "source_file": source_file,
                    "chars": n,
                    "metadata": metadata,
                })
                i = 0
                while i < n:
                    seg = text[i:i + chunk_chars]
                    if i + chunk_chars < n:
                        nl = seg.rfind("\n", chunk_chars // 2)
                        if nl > 0:
                            seg = seg[:nl]
                    seg = seg.strip()
                    if not seg:
                        i += chunk_chars
                        continue
                    dedupe_key = hashlib.sha1(
                        re.sub(r"\s+", " ", seg).strip().lower().encode("utf-8")
                    ).hexdigest()
                    if dedupe_key in seen_chunks:
                        i += len(seg) if seg else chunk_chars
                        continue
                    seen_chunks.add(dedupe_key)
                    rec = {
                        "id": f"paper:{source_kind}:{Path(source_label).name}:chunk_{chunk_id}",
                        "source_file": source_file,
                        "source_label": source_label,
                        "source_kind": source_kind,
                        "chunk_id": chunk_id,
                        "char_offset": i,
                        "section_hint": self._paper_section_hint(seg),
                        "text": seg,
                        "metadata": metadata,
                    }
                    lines.append(json.dumps(rec, ensure_ascii=False))
                    i += len(seg) if seg else chunk_chars
                    chunk_id += 1
            if not lines:
                return False
            with open(self.paper_chunks_jsonl, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            meta = {
                "sources": meta_sources,
                "chunk_count": chunk_id,
                "chars": total_chars,
            }
            with open(self.paper_index_meta, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            print(
                f"[graphify] indexed paper sources: {len(meta_sources)} sources, "
                f"{chunk_id} chunks → {self.paper_chunks_jsonl}"
            )
            return True
        except Exception as e:
            print(f"[graphify] paper indexing failed: {e}")
            return False

    def index_paper_text(self, paper_text: str, source_file: str = "paper.pdf",
                         chunk_chars: int = 4000) -> bool:
        return self.index_paper_sources(
            [{
                "source_kind": "paper",
                "source_label": source_file,
                "source_file": source_file,
                "text": paper_text,
                "metadata": {},
            }],
            chunk_chars=chunk_chars,
        )

    def _load_paper_records(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.paper_chunks_jsonl):
            return []
        out: List[Dict[str, Any]] = []
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

    def _iter_repo_text_files(self) -> List[Tuple[str, Path]]:
        files: List[Tuple[str, Path]] = []
        for root, dirs, filenames in os.walk(self.repo_path):
            dirs[:] = [
                d for d in dirs
                if not d.startswith(".") and d not in self._REPO_SKIP_DIRS
            ]
            for name in filenames:
                full = Path(root) / name
                rel = os.path.relpath(full, self.repo_path)
                suffix = full.suffix.lower()
                if suffix not in self._REPO_TEXT_EXTS:
                    continue
                files.append((rel, full))
        files.sort(key=lambda item: item[0])
        return files

    @staticmethod
    def _repo_source_kind(rel_path: str) -> str:
        rel = (rel_path or "").replace("\\", "/").lower()
        base = os.path.basename(rel)
        if base.startswith("readme"):
            return "readme"
        if rel.endswith((".md", ".rst", ".txt")) and "/docs/" in f"/{rel}/":
            return "docs"
        if rel.endswith((".yaml", ".yml", ".toml", ".cfg", ".ini", ".json", ".jsonnet")):
            return "config"
        if rel.endswith(".ipynb"):
            return "notebook"
        if rel.endswith((".sh", ".bash", ".zsh")):
            return "script"
        return "code"

    @staticmethod
    def _read_repo_text_file(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""

    def index_repo_corpus(self, chunk_chars: int = 4000) -> bool:
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            chunk_id = 0
            total_chars = 0
            meta_sources: List[Dict[str, Any]] = []
            lines = []
            for rel_path, full_path in self._iter_repo_text_files():
                text = self._read_repo_text_file(full_path).strip()
                if not text:
                    continue
                source_kind = self._repo_source_kind(rel_path)
                total_chars += len(text)
                meta_sources.append({
                    "source_file": rel_path,
                    "source_kind": source_kind,
                    "chars": len(text),
                })
                i = 0
                while i < len(text):
                    seg = text[i:i + chunk_chars]
                    if i + chunk_chars < len(text):
                        nl = seg.rfind("\n", chunk_chars // 2)
                        if nl > 0:
                            seg = seg[:nl]
                    seg = seg.strip()
                    if not seg:
                        i += chunk_chars
                        continue
                    rec = {
                        "id": f"repo:{source_kind}:{rel_path}:chunk_{chunk_id}",
                        "source_file": rel_path,
                        "source_kind": source_kind,
                        "chunk_id": chunk_id,
                        "char_offset": i,
                        "section_hint": rel_path,
                        "text": seg,
                    }
                    lines.append(json.dumps(rec, ensure_ascii=False))
                    i += len(seg) if seg else chunk_chars
                    chunk_id += 1
            if not lines:
                return False
            with open(self.repo_chunks_jsonl, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            with open(self.repo_index_meta, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "chunk_count": chunk_id,
                        "chars": total_chars,
                        "sources": meta_sources,
                    },
                    f,
                    indent=2,
                )
            print(
                f"[graphify] indexed repo corpus: {len(meta_sources)} files, "
                f"{chunk_id} chunks → {self.repo_chunks_jsonl}"
            )
            return True
        except Exception as e:
            print(f"[graphify] repo corpus indexing failed: {e}")
            return False

    def _load_repo_records(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.repo_chunks_jsonl):
            return []
        out: List[Dict[str, Any]] = []
        try:
            with open(self.repo_chunks_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        continue
        except Exception as e:
            print(f"[graphify] failed loading repo records: {e}")
        return out

    @staticmethod
    def _repo_score(question: str, rec: Dict[str, Any]) -> float:
        q_terms = {
            t.lower() for t in re.findall(r"[A-Za-z0-9_./-]+", question)
            if len(t) >= 3
        }
        if not q_terms:
            return 0.0
        text = " ".join([
            str(rec.get("source_kind", "")),
            str(rec.get("source_file", "")),
            str(rec.get("text", ""))[:5000],
        ]).lower()
        text_terms = set(re.findall(r"[A-Za-z0-9_./-]+", text))
        overlap = len(q_terms & text_terms)
        if overlap == 0:
            return 0.0
        boost = 0.0
        source_kind = str(rec.get("source_kind", "")).lower()
        if source_kind == "readme" and any(t in q_terms for t in {"readme", "install", "usage", "command", "run"}):
            boost += 2.0
        if source_kind == "config" and any(t in q_terms for t in {"config", "batch", "epochs", "learning", "optimizer", "yaml", "toml"}):
            boost += 1.5
        if source_kind in {"code", "script", "notebook"} and any(t in q_terms for t in {"main", "entry", "metric", "eval", "train", "argparse", "hydra"}):
            boost += 1.0
        return overlap + boost

    def query_repo_corpus(self, question: str, scope: str = "all",
                          token_budget: int = 3000, max_chunks: int = 8,
                          per_chunk_max_chars: int = 1500) -> str:
        if not question:
            return ""
        scope = (scope or "all").lower()
        allowed = {
            "all": None,
            "readme": {"readme"},
            "config": {"config"},
            "code": {"code", "script", "notebook"},
            "docs": {"docs", "readme"},
        }.get(scope)
        records = self._load_repo_records()
        if not records:
            return ""
        scored = []
        for rec in records:
            if allowed is not None and str(rec.get("source_kind", "")).lower() not in allowed:
                continue
            score = self._repo_score(question, rec)
            if score > 0:
                scored.append((score, rec))
        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored:
            return ""
        max_chars = max(token_budget * 4, per_chunk_max_chars)
        used = 0
        parts = [
            "========================================",
            f"REPO CONTEXT ({scope})",
            "========================================",
        ]
        for _, rec in scored[:max_chunks]:
            text = str(rec.get("text") or "").strip()
            if not text:
                continue
            if len(text) > per_chunk_max_chars:
                text = text[:per_chunk_max_chars] + " …[trunc]"
            block = (
                f"  [{rec.get('source_kind', 'repo')}:{rec.get('source_file', '')}] "
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
                f"  [paper:{rec.get('source_kind', 'paper')}:{rec.get('section_hint') or rec.get('chunk_id')}] "
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
                "repo_chunks_jsonl": self.repo_chunks_jsonl,
                "repo_chunks": len(self._load_repo_records()),
            }
        except Exception:
            return {
                "graph_json": self.graph_json,
                "nodes": 0,
                "edges": 0,
                "paper_chunks_jsonl": self.paper_chunks_jsonl,
                "paper_chunks": 0,
                "repo_chunks_jsonl": self.repo_chunks_jsonl,
                "repo_chunks": 0,
            }
