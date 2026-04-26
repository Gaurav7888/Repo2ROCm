"""
Paper Reproduction Agent — extracts metadata from research papers
and verifies reproduction on ROCm.

Workflow:
1. Extract arXiv/paper links from README
2. Fetch paper via arXiv API / Semantic Scholar
3. LLM-based structured extraction (hardware, libraries, benchmarks, commands)
4. Shortlist reproducible experiments against the shipped repo
5. Hand off execution and deterministic verification to the main agent loop
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from storage.models import ExperimentCandidate, PaperMetadata
from utils.json_utils import load_json_loose
from utils.llm import get_llm_response
from utils.rich_logger import log_info, log_warning


# ── Metric classification (generic, not paper-specific) ─────────────────────
#
# When the paper's headline experiment was run on different hardware than ours
# (e.g. A100/H100 vs AMD MI250X/MI300X), absolute throughput/latency numbers are
# not directly comparable. Ratios/speedups and accuracy-like metrics are much
# more hardware-portable. This classifier lets the planner/ranker prefer the
# portable metric when available.
#
# Classes (ordered from most-portable-across-hardware to least):
#   1. ratio_speedup  — unitless ratios (e.g. "2.5x", "+35%" speedup)
#   2. accuracy       — % accuracy, F1, BLEU, EM, Top-k — hardware-independent
#   3. quality        — PPL, NLL, loss, reward — almost hardware-independent
#   4. absolute_perf  — tokens/s, samples/s, QPS, latency (ms) — hardware-dependent!
#   5. other          — anything else
#
# Each class has a sensible default tolerance used when the LLM didn't supply one.

_RATIO_NAME_TOKENS = (
    "speedup", "speed-up", "speed up", "speedup ratio", " ratio",
    "improvement", "acceleration", "compression", "reduction",
    "efficiency gain", "relative",
)
_RATIO_UNIT_TOKENS = ("x", "×")
# `%` is intentionally NOT here — accuracy/F1/pass@k also report percentages.
# A bare `%` is ambiguous; we rely on the metric *name* to disambiguate.

_ACCURACY_NAME_TOKENS = (
    "accuracy", "acc@", "acc ", "acc)", "acc,", "top-",
    "f1", "em ", "em,", "exact match", "exact-match",
    "bleu", "rouge", "meteor", "chrf",
    "top1", "top5", "top_", "hit@",
    "pass@", "pass rate", "success rate", "win rate",
    "precision", "recall", " map ", "ndcg",
)

_QUALITY_NAME_TOKENS = (
    "perplexity", "ppl", "nll", "loss", "reward", "bpc", "bpb", "cross-entropy",
)

_ABSOLUTE_PERF_NAME_TOKENS = (
    "throughput", "latency", "qps", "rps",
    "tokens/s", "tok/s", "samples/s", "images/s", "frames/s", "sequences/s",
    "tokens per second", "samples per second", "frames per second",
    "time per", "wall time", "wall-clock", "runtime", "time to first token",
    "ttft", "tpot", "elapsed",
)
_ABSOLUTE_PERF_UNIT_TOKENS = (
    "tokens/s", "tok/s", "samples/s", "images/s", "frames/s",
    "ms", "milliseconds", "s", "sec", "seconds", "min", "minutes",
    "ns", "us", "µs",
)

_BASELINE_NAME_TOKENS = (
    "baseline", "no-cache", "no cache", "no_cache",
    "vanilla", "without ", "w/o ", "w\\o ",
    "origin", "--origin", "naive", "reference", "unmodified",
    "plain", "default", "standard", "fp16 baseline", "fp32 baseline",
    "untuned",
)

# Substrings that indicate the experiment IS a method measurement taken
# relative to a baseline — NOT a baseline itself. When present, baseline
# keywords in the same name should not flag is_baseline=True.
_BASELINE_NEGATORS = (
    "vs baseline", "vs. baseline", "versus baseline", "over baseline",
    "relative to baseline", "against baseline", "than baseline",
    "vs. the baseline", "over the baseline", "speedup over",
)


def _classify_metric(metric_name: str, metric_units: str) -> Tuple[str, str]:
    """Classify a metric into one of the portability classes and return a
    (class_name, default_tolerance_rule) pair.

    Priority order (specific names beat generic units):
      1. name matches ratio/speedup tokens, OR units are `x`/`×`
      2. name matches accuracy/F1/pass@k style tokens
      3. name matches perplexity/loss/NLL/reward tokens
      4. name matches absolute throughput/latency tokens, OR units are time/rate
      5. units are `%` / `pct` / `percent` (ambiguous; bucket as accuracy since
         that is the most common use of a bare % in ML papers)
      6. fall back to `other`
    """
    name = (metric_name or "").lower().strip()
    units = (metric_units or "").lower().strip()

    def _contains_any(s: str, tokens) -> bool:
        return any(tok in s for tok in tokens)

    if _contains_any(name, _RATIO_NAME_TOKENS) or units in _RATIO_UNIT_TOKENS:
        return "ratio_speedup", "<=15% relative delta for ratios/speedups/percentages"

    if _contains_any(name, _ACCURACY_NAME_TOKENS):
        return "accuracy", "<=3 absolute percentage points for accuracy-style metrics"

    if _contains_any(name, _QUALITY_NAME_TOKENS):
        return "quality", "<=5% relative delta for perplexity/loss/NLL"

    if _contains_any(name, _ABSOLUTE_PERF_NAME_TOKENS) or units in _ABSOLUTE_PERF_UNIT_TOKENS:
        return (
            "absolute_perf",
            "<=25% relative delta (absolute throughput/latency on different GPU is expected to differ)",
        )

    if units in ("%", "pct", "percent"):
        # Ambiguous bare percentage with an unknown metric name — assume it's
        # an accuracy-style score (most common case in ML papers).
        return "accuracy", "<=3 absolute percentage points for percentage-valued metrics"

    return "other", "<=15% relative delta (best-effort default)"


def _looks_like_baseline(name: str) -> bool:
    """Generic baseline detector by keyword match on the experiment name.

    Suppresses false positives for phrases like 'speedup vs baseline' where
    the experiment is a method measurement relative to a baseline (not a
    baseline itself).
    """
    if not name:
        return False
    n = name.lower()
    if any(neg in n for neg in _BASELINE_NEGATORS):
        return False
    return any(tok in n for tok in _BASELINE_NAME_TOKENS)


_METRIC_CLASS_RANK = {
    # Lower is better (more portable / more meaningful on different hardware).
    "ratio_speedup": 0,
    "accuracy": 1,
    "quality": 2,
    "other": 3,
    "absolute_perf": 4,
}


def _extract_text_with_pymupdf(pdf_path: str, max_chars: int = 400000) -> str:
    """Try extracting text via PyMuPDF (fitz). Returns empty string on failure."""
    try:
        import fitz  # type: ignore
    except Exception:
        return ""
    try:
        parts: List[str] = []
        total = 0
        with fitz.open(pdf_path) as doc:
            for page in doc:
                txt = page.get_text("text")
                if not txt:
                    continue
                parts.append(txt)
                total += len(txt)
                if total >= max_chars:
                    break
        return ("\n".join(parts))[:max_chars]
    except Exception:
        return ""


def _extract_text_with_pdftotext(pdf_path: str, max_chars: int = 400000) -> str:
    """Fallback: use the `pdftotext` CLI from poppler-utils if available."""
    if not shutil.which("pdftotext"):
        return ""
    try:
        res = subprocess.run(
            ["pdftotext", "-layout", pdf_path, "-"],
            capture_output=True, timeout=60,
        )
        if res.returncode == 0 and res.stdout:
            return res.stdout.decode("utf-8", errors="ignore")[:max_chars]
    except Exception:
        pass
    return ""


def extract_pdf_text(pdf_path: str, max_chars: int = 400000) -> str:
    """Best-effort PDF -> text extraction. Empty string if nothing works."""
    text = _extract_text_with_pymupdf(pdf_path, max_chars=max_chars)
    if text:
        return text
    text = _extract_text_with_pdftotext(pdf_path, max_chars=max_chars)
    return text


class PaperAgent:
    """Extracts paper metadata and drives reproduction verification."""

    def __init__(self, llm: str = ""):
        self.llm = llm

    def extract_paper_link(self, readme_content: str) -> Optional[str]:
        """Find arXiv or paper links in README."""
        arxiv_patterns = [
            r"arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)",
            r"arxiv\.org/pdf/(\d{4}\.\d{4,5}(?:v\d+)?)",
            r"\[(?:paper|arxiv)\]\s*\(https?://arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)\)",
        ]
        for pattern in arxiv_patterns:
            m = re.search(pattern, readme_content, re.IGNORECASE)
            if m:
                return m.group(1)
        return None

    def fetch_paper_metadata(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """Fetch paper abstract and metadata from arXiv API."""
        try:
            import urllib.request
            import xml.etree.ElementTree as ET

            url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = resp.read().decode()

            ns = {"atom": "http://www.w3.org/2005/Atom"}
            root = ET.fromstring(data)
            entry = root.find("atom:entry", ns)
            if entry is None:
                return None

            title = entry.findtext("atom:title", "", ns).strip()
            abstract = entry.findtext("atom:summary", "", ns).strip()
            authors = [
                a.findtext("atom:name", "", ns)
                for a in entry.findall("atom:author", ns)
            ]

            return {
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "authors": authors[:10],
            }
        except Exception:
            return None

    def extract_structured_metadata(self, paper_text: str,
                                    readme_content: str = "") -> PaperMetadata:
        """
        Use LLM to extract structured metadata from paper abstract + README.
        """
        if not self.llm:
            return PaperMetadata()

        prompt = f"""\
Extract structured information from this research paper context.

PAPER ABSTRACT / CONTENT:
{paper_text[:3000]}

README EXCERPT:
{readme_content[:2000]}

Return a JSON object with these fields:
{{
  "hardware_used": ["list of GPUs mentioned, e.g. A100 80GB"],
  "cuda_version_mentioned": "version string or empty",
  "key_libraries": ["list of key libraries/frameworks"],
  "custom_kernels_described": true/false,
  "kernel_purpose": ["list of kernel purposes if custom kernels exist"],
  "reproduction_commands": ["commands from paper/README for reproduction"],
  "benchmark_metrics": {{"metric_name": "value"}},
  "model_scale": "parameter count or model size",
  "training_compute": "compute budget if mentioned",
  "key_tricks": ["notable implementation tricks"]
}}

Return ONLY the JSON object, no markdown fences."""

        try:
            messages = [{"role": "user", "content": prompt}]
            response, _ = get_llm_response(self.llm, messages,
                                           temperature=0.1, max_tokens=1024)
            if response and response[0]:
                data = load_json_loose(response[0], expected="object")
                return PaperMetadata.from_dict(data)
        except Exception:
            pass

        return PaperMetadata()

    # ── Paper download + experiment shortlisting ─────────────────────────────

    @staticmethod
    def _normalize_arxiv_url(url: str) -> str:
        """Normalize arxiv abs/html URLs to a PDF URL. Leave other URLs alone."""
        m = re.match(r"https?://arxiv\.org/(?:abs|pdf|html)/([\w.\-/]+?)(?:\.pdf)?/?$",
                     url.strip(), re.IGNORECASE)
        if m:
            return f"https://arxiv.org/pdf/{m.group(1)}.pdf"
        return url.strip()

    def download_paper(self, url: str, dest: str) -> Optional[str]:
        """Download a paper PDF to `dest`. Returns the dest path on success."""
        if not url:
            return None
        url = self._normalize_arxiv_url(url)
        try:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Repo2ROCm-PaperAgent/1.0"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                with open(dest, "wb") as f:
                    shutil.copyfileobj(resp, f)
            if os.path.getsize(dest) < 1024:
                return None
            return dest
        except Exception:
            return None

    def _list_repo_code_files(self, repo_path: str, limit: int = 400) -> List[str]:
        """List candidate entry-script files in the repo for keyword matching."""
        exts = {".py", ".sh", ".ipynb", ".bash"}
        results: List[str] = []
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in {
                "node_modules", "__pycache__", "venv", ".venv", "build", "dist"
            }]
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext in exts:
                    rel = os.path.relpath(os.path.join(root, fn), repo_path)
                    results.append(rel)
                    if len(results) >= limit:
                        return results
        return results

    @staticmethod
    def _collect_repo_configs(
        repo_path: str,
        max_files: int = 60,
        max_per_file_chars: int = 6000,
        total_budget_chars: int = 80000,
    ) -> Dict[str, str]:
        """Return {relative_path: file_contents} for config-like files in the repo.

        Generic across frameworks: covers Hydra (`conf/`), pure YAML (`configs/`,
        `config/`), JSON config dumps, TOML, INI, and any top-level config files.
        Hyperparameters that are NOT in the README often live here, and the
        planner should reconcile them against the paper.
        """
        config_exts = {".yaml", ".yml", ".toml", ".cfg", ".ini", ".json", ".jsonnet"}
        # Files we want to skip even if extension matches:
        skip_filename_substrings = (
            "lock",  # poetry.lock, package-lock.json, yarn.lock
            "tsconfig",
            "package.json",  # JS deps, not ML config
            "package-lock",
            "schema",
            "tsconfig",
        )
        # Subdirectories whose contents are very likely config:
        config_dir_hints = (
            "conf", "configs", "config", "cfg", "args", "params",
            "hparams", "hyperparams", "experiments", "experiment",
            "recipes", "runs", "run_configs", "training",
        )
        results: Dict[str, str] = {}
        total = 0
        # Walk the repo and collect candidate config files.
        candidates: List[Tuple[str, int]] = []
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in {
                "node_modules", "__pycache__", "venv", ".venv", "build", "dist",
                "site-packages", "checkpoints", "wandb", "outputs",
                "graphify-out",  # our own per-repo code-graph cache
            }]
            rel_root = os.path.relpath(root, repo_path)
            in_config_dir = any(
                p in (rel_root.split(os.sep)) for p in config_dir_hints
            )
            for fn in files:
                low = fn.lower()
                if any(s in low for s in skip_filename_substrings):
                    continue
                ext = os.path.splitext(fn)[1].lower()
                if ext not in config_exts:
                    continue
                full = os.path.join(root, fn)
                try:
                    sz = os.path.getsize(full)
                except OSError:
                    continue
                if sz > 200_000:  # skip huge generated json blobs
                    continue
                # Heuristic priority: anything inside a config-y dir > files at
                # repo root > everything else.
                rel = os.path.relpath(full, repo_path)
                depth = rel.count(os.sep)
                if in_config_dir:
                    priority = 0
                elif depth == 0:
                    priority = 1
                else:
                    priority = 2 + depth
                candidates.append((rel, priority))
        # Read in priority order until budget exhausted.
        candidates.sort(key=lambda t: (t[1], t[0]))
        for rel, _prio in candidates:
            if len(results) >= max_files or total >= total_budget_chars:
                break
            full = os.path.join(repo_path, rel)
            try:
                with open(full, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read(max_per_file_chars + 1)
            except Exception:
                continue
            if not txt:
                continue
            if len(txt) > max_per_file_chars:
                txt = txt[:max_per_file_chars] + "\n# ... [truncated]"
            results[rel] = txt
            total += len(txt)
        return results

    @staticmethod
    def _bucket_runtime(minutes: float) -> str:
        if minutes <= 10:
            return "small"
        if minutes <= 60:
            return "medium"
        return "large"

    @staticmethod
    def _runtime_from_bucket(bucket: str) -> float:
        return {"small": 5.0, "medium": 30.0, "large": 240.0}.get(bucket.lower(), 60.0)

    def _match_code_for_experiment(self, suggested_command: str,
                                   repo_files: List[str]) -> List[str]:
        """Return repo-relative files that plausibly back the suggested command."""
        if not suggested_command:
            return []
        try:
            parts = shlex.split(suggested_command)
        except ValueError:
            parts = suggested_command.split()
        tokens = [
            os.path.basename(tok)
            for tok in parts
            if tok.endswith(".py") or tok.endswith(".sh")
        ]
        if not tokens:
            return []
        matches: List[str] = []
        lower_files = [(f, f.lower()) for f in repo_files]
        for tok in tokens:
            tlow = tok.lower()
            for orig, flow in lower_files:
                if flow.endswith("/" + tlow) or flow == tlow or flow.endswith(tlow):
                    if orig not in matches:
                        matches.append(orig)
        return matches

    def _normalize_repo_experiments(
        self,
        readme_run_commands: Optional[List[Dict[str, Any]]],
        entry_scripts: List[str],
        repo_files: List[str],
        max_candidates: int = 10,
    ) -> List[Dict[str, Any]]:
        """Build simple repo experiment surfaces from README commands first."""
        candidates: List[Dict[str, Any]] = []
        seen_commands = set()

        for idx, item in enumerate(readme_run_commands or [], 1):
            if not isinstance(item, dict):
                continue
            command = str(item.get("command") or "").strip()
            if not command or command in seen_commands:
                continue
            seen_commands.add(command)
            candidates.append({
                "id": f"repo_exp_{idx}",
                "source": "readme",
                "command": command,
                "context": str(item.get("context") or "").strip(),
                "matched_files": self._match_code_for_experiment(command, repo_files)[:8],
            })
            if len(candidates) >= max_candidates:
                return candidates

        if candidates:
            return candidates

        for idx, script in enumerate(entry_scripts[:max_candidates], 1):
            command = (
                f"python {script}" if script.endswith(".py")
                else f"bash {script}" if script.endswith(".sh")
                else script
            )
            candidates.append({
                "id": f"repo_exp_{idx}",
                "source": "entrypoint",
                "command": command,
                "context": "Inferred from repo entrypoints because no README command was available.",
                "matched_files": [script],
            })
        return candidates

    @staticmethod
    def _collect_repo_metric_surfaces(
        repo_path: str,
        repo_experiments: List[Dict[str, Any]],
        repo_files: List[str],
        max_files: int = 8,
        max_chars_per_file: int = 2200,
    ) -> str:
        """Collect small repo/eval snippets that hint at runtime metric names."""
        selected: List[str] = []
        seen = set()
        lower_repo_files = [(path, path.lower()) for path in repo_files]

        for exp in repo_experiments:
            for matched in exp.get("matched_files") or []:
                if matched not in seen:
                    selected.append(matched)
                    seen.add(matched)
                parent = matched.rsplit("/", 1)[0] if "/" in matched else ""
                for orig, low in lower_repo_files:
                    if len(selected) >= max_files:
                        break
                    if orig in seen:
                        continue
                    if parent and not low.startswith(parent.lower() + "/"):
                        continue
                    base = os.path.basename(low)
                    if any(key in base for key in ("eval", "metric", "score", "pred")):
                        selected.append(orig)
                        seen.add(orig)
                if len(selected) >= max_files:
                    break
            if len(selected) >= max_files:
                break

        blocks: List[str] = []
        for rel in selected[:max_files]:
            full = os.path.join(repo_path, rel)
            try:
                with open(full, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read(max_chars_per_file + 1)
            except Exception:
                continue
            if len(text) > max_chars_per_file:
                text = text[:max_chars_per_file] + "\n# ... [truncated]"
            blocks.append(f"# ---- {rel} ----\n{text}")

        return "\n\n".join(blocks)

    @staticmethod
    def _select_prompt_configs(
        repo_experiments: List[Dict[str, Any]],
        repo_configs: Dict[str, str],
        max_paths: int = 6,
    ) -> List[str]:
        """Pick a small config set near the chosen repo experiment surfaces."""
        if not repo_configs:
            return []

        selected: List[str] = []
        seen = set()
        for exp in repo_experiments:
            for matched in exp.get("matched_files") or []:
                parent = matched.rsplit("/", 1)[0] if "/" in matched else ""
                if not parent:
                    continue
                for path in repo_configs:
                    if path in seen or path.startswith("graphify-out/"):
                        continue
                    if path.startswith(parent + "/"):
                        selected.append(path)
                        seen.add(path)
                        if len(selected) >= max_paths:
                            return selected

        for path in repo_configs:
            if path in seen or path.startswith("graphify-out/"):
                continue
            selected.append(path)
            seen.add(path)
            if len(selected) >= max_paths:
                break
        return selected

    def shortlist_experiments(
        self,
        paper_pdf_path: Optional[str],
        repo_path: str,
        readme_content: str = "",
        readme_run_commands: Optional[List[Dict[str, Any]]] = None,
        readme_expected_outcomes: Optional[List[Dict[str, Any]]] = None,
        llm: Optional[str] = None,
        max_candidates: int = 8,
        run_memory: Optional[Any] = None,
        graphify_provider: Optional[Any] = None,
    ) -> Tuple[List[ExperimentCandidate], str]:
        """
        Enumerate runnable repo experiments, map them to paper rows, and return
        (ranked_candidates, paper_title).

        Ranking priorities (most-significant first):
          1. `code_available` — must be True; experiments with no code match are
             pushed to the bottom.
          2. README-backed repo commands are preferred over inferred entrypoints.
          3. `is_baseline` — pure baselines (no-cache / --origin / vanilla / naive)
             are pushed below method experiments, because reproducing only a
             baseline does NOT demonstrate the paper's contribution.
          4. `metric_class` portability — ratio/speedup/percentage > accuracy-style
             > perplexity/loss > absolute throughput/latency. Hardware-portable
             metrics are preferred because we typically run on a different GPU
             than the paper (e.g. AMD MI250X vs NVIDIA A100).
          5. Shorter estimated runtime.
          6. Shorter suggested command (tie-breaker).

        This ranking is generic — it starts from runnable repo surfaces, then
        asks which paper row those surfaces best correspond to.
        """
        effective_llm = llm or self.llm
        paper_text = ""
        if paper_pdf_path and os.path.isfile(paper_pdf_path):
            paper_text = extract_pdf_text(paper_pdf_path)

        repo_files = self._list_repo_code_files(repo_path)
        repo_configs = self._collect_repo_configs(repo_path)
        candidates: List[ExperimentCandidate] = []
        paper_title = ""

        if effective_llm and (paper_text or readme_content):
            readme_slice = readme_content[:50000] if readme_content else ""

            # ── Repo-first experiment discovery ───────────────────────────────
            entry_scripts: List[str] = []
            if graphify_provider is not None and getattr(graphify_provider, "enabled", False):
                try:
                    if not os.path.exists(getattr(graphify_provider, "graph_json", "")):
                        graphify_provider.build_or_refresh()
                    entry_scripts = graphify_provider.list_entry_scripts(max_files=12)
                except Exception as _e:
                    log_warning(
                        f"  Paper shortlist: graphify entry-script lookup failed: {_e}"
                    )
            files_for_prompt = entry_scripts if entry_scripts else repo_files[:200]
            repo_experiments = self._normalize_repo_experiments(
                readme_run_commands=readme_run_commands,
                entry_scripts=entry_scripts,
                repo_files=repo_files,
                max_candidates=max(max_candidates, 8),
            )
            repo_experiment_map = {
                str(item.get("id") or ""): item
                for item in repo_experiments
                if item.get("id")
            }
            repo_experiment_block = json.dumps(
                repo_experiments[: max(max_candidates, 8)],
                indent=2,
                default=str,
            )
            readme_outcomes_block = json.dumps(
                (readme_expected_outcomes or [])[:8],
                indent=2,
                default=str,
            ) if readme_expected_outcomes else "[]"
            repo_metric_block = self._collect_repo_metric_surfaces(
                repo_path=repo_path,
                repo_experiments=repo_experiments,
                repo_files=repo_files,
                max_files=8,
            ) or "(no repo/eval metric surfaces captured)"

            # ── Stage 4/6: paper text via graphify-owned paper index ──────────
            paper_block = ""
            paper_backend = "fallback"
            if graphify_provider is not None and getattr(graphify_provider, "enabled", False):
                try:
                    paper_block = graphify_provider.query_paper(
                        "main paper body experiments methods setup evaluation results "
                        "tables table footnotes figure captions headline metric "
                        "accuracy F1 EM perplexity speedup datasets benchmarks "
                        "model sizes evaluation protocol method algorithm",
                        token_budget=8000,
                        max_chunks=8,
                        per_chunk_max_chars=1500,
                    ) or ""
                    if paper_block:
                        paper_backend = "graphify-pass1"
                        if run_memory is not None and getattr(run_memory, "enabled", False):
                            run_memory.write_context_ref(
                                kind="paper_query",
                                ref_id="graphify:paper_chunks",
                                source=getattr(graphify_provider, "paper_chunks_jsonl", ""),
                                why_relevant="paper shortlist pass1 main-body evidence",
                                extra={"question": "paper shortlist repo-first mapping query"},
                            )
                except Exception as _e:
                    log_warning(
                        f"  Paper shortlist: graphify paper query failed: {_e}"
                    )
            if not paper_block and paper_text:
                # Fallback dump: paper text is secondary evidence used to map
                # repo-backed commands to paper rows and resolve ambiguity.
                paper_block = (
                    "PAPER TEXT DUMP (use to map repo-backed experiments to paper rows; appendix only if config/metric/setup remains ambiguous):\n"
                    + paper_text[:350000]
                )
                paper_backend = "fallback-dump"

            # ── Stage 5a: small repo-config surface for the chosen commands ─────
            configs_block = ""
            try:
                relevant_paths = self._select_prompt_configs(
                    repo_experiments,
                    repo_configs,
                    max_paths=6,
                )
            except Exception as _e:
                log_warning(
                    f"  Paper shortlist: _select_prompt_configs failed: {_e}"
                )
                relevant_paths = []
            if relevant_paths:
                chunks = [
                    f"# ---- {p} ----\n{repo_configs[p][:6000]}"
                    for p in relevant_paths if p in repo_configs
                ]
                if chunks:
                    configs_block = (
                        f"REPO CONFIG FILES (selected by relevance; full repo "
                        f"had {len(repo_configs)} config files):\n\n"
                        + "\n\n".join(chunks)
                    )
                    configs_method = f"repo-surface ({len(chunks)}/{len(repo_configs)} files)"
            if not configs_block:
                if repo_configs:
                    config_chunks = [
                        f"# ---- {path} ----\n{content[:8000]}"
                        for path, content in list(repo_configs.items())[:12]
                    ]
                    configs_block = "\n\n".join(config_chunks)
                else:
                    configs_block = "(no yaml/toml/json/cfg config files found in the repo)"

            # ── Researcher pattern phase 1: deterministic evidence ─────────
            # We pre-fetch a tiny amount of live evidence the synth call would
            # otherwise have to guess at, namely:
            #   * pypi_versions for common evaluation frameworks the LLM tends
            #     to invoke as flags (transformers, datasets, lm-eval).
            #   * a single deep_research snippet about the paper's headline
            #     metric, when an llm/budget is available.
            evidence_lines: List[str] = []
            try:
                from tools.external_lookups import pypi_versions as _pv
                for pkg in ("transformers", "datasets", "lm-eval"):
                    body, rc = _pv(pkg, limit=4)
                    if rc == 0 and body:
                        evidence_lines.append(f"pypi_versions {pkg}:")
                        for ln in body.splitlines()[:5]:
                            if ln.strip():
                                evidence_lines.append(f"  {ln.strip()}")
            except Exception:
                pass
            paper_hint = ""
            if effective_llm and os.environ.get("AMD_LLM_API_KEY"):
                try:
                    from agents.researcher import research
                    note = research(
                        "Which metrics in this paper are most reliably "
                        "reproducible across GPU vendors? Prefer accuracy / "
                        "F1 / ratio metrics over absolute throughput. Be "
                        "concise.",
                        llm=effective_llm, budget_s=20.0, use_cache=True,
                    )
                    paper_hint = (note.get("answer") or "")[:300]
                except Exception:
                    paper_hint = ""
            evidence_block = ""
            if evidence_lines or paper_hint:
                evidence_block = (
                    "\nLIVE EVIDENCE (deterministic tools, prefer over training "
                    "data):\n" + "\n".join(evidence_lines[:30])
                )
                if paper_hint:
                    evidence_block += f"\nResearcher hint: {paper_hint}\n"

            log_info(
                f"  Paper shortlist prompt sources: "
                f"repo_experiments={len(repo_experiments)}, "
                f"paper_block={len(paper_block):,} chars "
                f"(backend={paper_backend}), "
                f"readme={len(readme_slice):,}, "
                f"repo_metric_block={len(repo_metric_block):,}, "
                f"configs_block={len(configs_block):,} ({configs_method}), "
                f"entry_scripts={len(files_for_prompt)}, "
                f"evidence={len(evidence_block):,}"
            )

            def _run_shortlist_prompt(prompt_text: str, label: str) -> Tuple[Optional[Dict[str, Any]], str]:
                raw_text = ""
                parsed_obj: Optional[Dict[str, Any]] = None
                try:
                    messages = [{"role": "user", "content": prompt_text}]
                    response, _ = get_llm_response(
                        effective_llm, messages, temperature=0.1, max_tokens=16384
                    )
                    if response and response[0]:
                        raw_text = response[0].strip()
                        try:
                            parsed_obj = load_json_loose(raw_text, expected="object")
                        except ValueError:
                            log_warning(
                                f"  Paper shortlist {label}: JSON parse failed; "
                                f"raw response len={len(raw_text):,} chars; head="
                                f"{raw_text[:300]!r} ... tail={raw_text[-300:]!r}"
                            )
                    else:
                        log_warning(
                            f"  Paper shortlist {label}: LLM returned an empty response."
                        )
                except Exception as e:
                    log_warning(
                        f"  Paper shortlist {label}: prompt call failed: "
                        f"{type(e).__name__}: {e}"
                    )
                return parsed_obj, raw_text

            schema_prompt = f"""\
Return a STRICT JSON object with this shape (no markdown fences, no prose):
{{
  "title": "<paper title if you can tell, else empty string>",
  "experiments": [
    {{
      "name": "<short descriptive name; include the method name if applicable>",
      "section": "<paper section, table, or figure reference, e.g. 'Table 1, Sec 4.2'>",
      "expected_metric": {{
        "name": "<headline metric, e.g. speedup, accuracy, F1, PPL>",
        "value": "<numeric value as string, e.g. '2.5', '3x', '78.4'>",
        "units": "<e.g. x, %, tokens/s, ms, '' >"
      }},
      "repo_experiment_id": "<id from REPO EXPERIMENT SURFACES>",
      "repo_command_source": "<readme | entrypoint>",
      "repo_context": "<short note about where this command came from>",
      "runtime_metric_source": "<repo/eval file or README outcome that justifies the runtime metric name>",
      "primary_metrics": [
        {{
          "name": "<EXACT metric name as it would appear in the run log>",
          "expected_value": "<numeric value the verifier should compare against>",
          "tolerance": "<per-metric rule, e.g. '<=15%' or '<=3 abs pts'>",
          "direction": "<higher_is_better | lower_is_better | equal>"
        }}
      ],
      "is_baseline": <true only if this row is a NO-METHOD baseline (vanilla /
        no-cache / --origin / naive / reference). False if it exercises the
        paper's proposed method — even if the metric is reported *relative to*
        a baseline.>,
      "hardware": "<GPU(s) used in paper for THIS row, if mentioned>",
      "est_runtime_minutes": <number>,
      "paper_config": {{
        "model": "<exact model id>",
        "batch_size": "<value>",
        "seq_len": "<value>",
        "steps": "<value>",
        "block_length": "<value>",
        "temperature": "<value>",
        "cfg_scale": "<value>",
        "sampling_alg": "<value>",
        "cache_steps": "<value>",
        "window_size": "<value>",
        "precision": "<fp16|bf16|int8|...>",
        "dataset": "<name>",
        "num_samples": "<value>",
        "<any other paper-reported hyperparam>": "<value>"
      }},
      "config_source": "<verbatim phrase / table cell / yaml path where you found each value>",
      "codebase_config_files": [
        "<relative path of every yaml/toml/json/cfg file that governs this experiment's hyperparameters>"
      ],
      "suggested_command": "<start from the chosen repo experiment command verbatim when possible; only substitute local paths/model ids and add non-default flags the repo surface actually supports. Do NOT invent a fresh command from the paper alone>",
      "comparison_mode": "<single | vs_baseline>",
      "baseline_reference": {{
        "section": "<paper row for the baseline, when needed>",
        "repo_experiment_id": "<matching repo experiment id, if there is one>",
        "suggested_command": "<repo-backed command for the baseline if the claim is relative>",
        "expected_metric_name": "<baseline headline metric name>",
        "expected_metric_value": "<baseline expected value>",
        "notes": "<blank unless the main claim needs a baseline run>"
      }},
      "missing_flags": ["<flag the paper used but the script can't accept>"],
      "caveats": [
        "<paper/README disclaimer verbatim>",
        "<another caveat if relevant>"
      ],
      "tolerance_rule": "<e.g. '<=15% relative for speedup', '<=3 absolute points for accuracy'>",
      "notes": "<anything else, e.g. 'compute speedup = method_tok/s / baseline_tok/s locally'>"
    }}
  ],
  "unresolved_items": [
    "<config / metric / setup detail that remains ambiguous after the repo-first pass>"
  ],
  "followup_questions": [
    "<targeted question a second paper retrieval pass should answer; mention appendix/supplementary only if needed>"
  ]
}}

RULES for primary_metrics (the deterministic verifier reads this list):
- Always populate `primary_metrics` with EVERY headline metric the paper
  reports for the chosen experiment. If the paper claims "RMSE 0.123 / PCC
  0.987", you MUST list BOTH — otherwise the verifier cannot detect the
  classic "RMSE better but PCC much worse" failure mode.
- `name` MUST come from repo/runtime evidence first: README EXPECTED OUTCOMES,
  REPO METRIC SURFACES, eval scripts, or the repo command context. Only fall
  back to paper wording if the repo gives no signal.
- `direction` is mandatory: pick from higher_is_better, lower_is_better, or
  equal. RMSE/MSE/MAE/loss/PPL → lower_is_better. Accuracy/F1/PCC/AUC →
  higher_is_better. Speedups → higher_is_better.
- `tolerance` is per-metric and overrides the experiment's coarse
  `tolerance_rule` when the verifier is invoked with no explicit override.

RULES for choosing and ordering experiments:
- Start from the REPO EXPERIMENT SURFACES. Prefer mapping one of them to a
  paper row over inventing a brand-new command from the paper.
- Prefer the README-backed repo command when it already reaches a meaningful
  paper result.
- Prefer method rows over pure baselines.
- Only use `comparison_mode = "vs_baseline"` when the paper claim is inherently
  relative and needs a second row to interpret the result.
- Return AT LEAST 3 and AT MOST {max_candidates} experiments, best first.
- `repo_experiment_id` MUST point at one of the supplied repo experiment
  surfaces. If none of the supplied repo experiments can realize a paper row,
  say so in `missing_flags` / `caveats` instead of inventing a repo surface.
- `suggested_command` MUST use flags the entry script actually supports. If
  the paper uses flags the script can't accept, put them in `missing_flags`
  rather than inventing them.
- `est_runtime_minutes` is your estimate for a single run on one modern GPU.
"""

            phase1_prompt = f"""\
You are mapping runnable REPO experiments to exact PAPER results.

Your goal: choose ONE repo-backed experiment we can actually run on AMD ROCm,
then identify exactly which paper row it should reproduce and which runtime
metric(s) must match.

{paper_block}

README (full):
{readme_slice}

REPO EXPERIMENT SURFACES (PRIMARY INPUT):
{repo_experiment_block}

README EXPECTED OUTCOMES:
{readme_outcomes_block}

REPO METRIC SURFACES:
{repo_metric_block}

REPO CONFIG FILES (yaml/toml/json/cfg/ini — these are the codebase's actual
default hyperparameters; treat them as authoritative when the paper is silent
or specifies a sweep without picking a value):
{configs_block}
{evidence_block}

PHASE 1 WORKFLOW:
1. REPO FIRST: start from REPO EXPERIMENT SURFACES. These are the commands the
   repo most clearly exposes through the README or entrypoints.
2. For each repo experiment surface, ask which paper row it most plausibly
   corresponds to. Do NOT invent a fresh experiment if a repo surface already
   matches the paper's contribution.
3. Use README EXPECTED OUTCOMES and REPO METRIC SURFACES as the PRIMARY source
   for runtime metric naming. Your job is to bind the paper's metric to the
   metric name the repo/eval path will actually emit.
4. RECONCILE paper and codebase. The codebase config files are usually the
   actual values the repo ships with; the paper may quote them generically
   ("we tune the learning rate via grid search") while the yaml/toml file
   ships with the chosen default.
   - If the paper FIXES a value (e.g. "lr=2e-4") and the codebase ALSO has it,
     they should agree — use that value. If they disagree, prefer the PAPER's
     value and add a caveat citing the conflict.
   - If the paper is AMBIGUOUS (e.g. a sweep, or "see appendix") but the
     codebase has a hardcoded value, USE THE CODEBASE VALUE and cite the
     yaml/toml path in `config_source`.
   - If neither paper nor codebase specifies a value, fall back to the
     repo surface default and note that in `caveats`.
5. Use Appendix/Supplementary ONLY if a config / metric / setup detail remains
   ambiguous after the repo-first reconciliation.
6. If anything remains ambiguous after this first pass, keep the best current
   experiment list but record the gap in `unresolved_items` and write precise
   `followup_questions` for a second retrieval pass.
7. For EACH candidate experiment, extract the FULL configuration. Include:
   - model name (exact HuggingFace repo id when possible)
   - batch_size, sequence length, steps/epochs, learning_rate, optimizer,
     lr_scheduler, warmup_ratio, weight_decay, block_length, temperature,
     cfg_scale, sampling algorithm, cache_steps, window_size, precision
     (fp16/bf16/int8), decoding algorithm, peft/lora hyperparams
   - dataset / benchmark name (MMLU, GSM8K, HumanEval, MRPC, etc.)
   - any environment variables the paper/README calls out
8. `suggested_command` should start from the chosen repo command verbatim when
   possible. Only substitute local paths/model ids and add non-default flags
   the repo surface clearly supports. If the paper's config cannot be
   expressed through the script's CLI (e.g. the script hardcodes
   batch_size=1), list the unreachable flags in `missing_flags`.
9. If the paper claim is relative (speedup vs baseline, degradation vs full,
   compression vs full attention), use `comparison_mode = "vs_baseline"` and
   fill `baseline_reference`. Otherwise leave `comparison_mode = "single"` and
   keep `baseline_reference` empty.
10. Check the paper AND the README for DISCLAIMERS about the config, e.g.
   "speedup not significant at batch_size=1", "requires H100 for 10x claim",
   "accuracy measured on MMLU only". Put every disclaimer in `caveats`.

{schema_prompt}
"""

            phase1_parsed, phase1_text = _run_shortlist_prompt(phase1_prompt, "pass1")
            parsed: Optional[Dict[str, Any]] = phase1_parsed
            text = phase1_text

            phase1_unresolved: List[str] = []
            phase1_followups: List[str] = []
            if isinstance(phase1_parsed, dict):
                raw_unresolved = phase1_parsed.get("unresolved_items") or []
                if isinstance(raw_unresolved, str):
                    raw_unresolved = [raw_unresolved]
                phase1_unresolved = [str(x)[:220] for x in raw_unresolved if x][:6]

                raw_followups = phase1_parsed.get("followup_questions") or []
                if isinstance(raw_followups, str):
                    raw_followups = [raw_followups]
                phase1_followups = [str(x)[:260] for x in raw_followups if x][:4]

            if phase1_followups or phase1_unresolved:
                followup_block = ""
                followup_backend = ""
                followup_query = "Resolve remaining config / metric / setup ambiguity: "
                if phase1_followups:
                    followup_query += " ; ".join(phase1_followups)
                else:
                    followup_query += " ; ".join(phase1_unresolved)

                if graphify_provider is not None and getattr(graphify_provider, "enabled", False):
                    try:
                        followup_block = graphify_provider.query_paper(
                            followup_query,
                            token_budget=4500,
                            max_chunks=5,
                            per_chunk_max_chars=1200,
                        ) or ""
                        if followup_block:
                            followup_backend = "graphify-pass2"
                            if run_memory is not None and getattr(run_memory, "enabled", False):
                                run_memory.write_context_ref(
                                    kind="paper_query",
                                    ref_id="graphify:paper_chunks",
                                    source=getattr(graphify_provider, "paper_chunks_jsonl", ""),
                                    why_relevant="paper shortlist pass2 ambiguity follow-up",
                                    extra={"question": followup_query},
                                )
                    except Exception as _e:
                        log_warning(
                            f"  Paper shortlist: graphify pass2 follow-up failed: {_e}"
                        )
                elif paper_text:
                    followup_block = (
                        "FOLLOW-UP PAPER DUMP (use only to resolve remaining "
                        "config/metric/setup ambiguity; appendix/supplementary "
                        "may help here):\n" + paper_text[:250000]
                    )
                    followup_backend = "fallback-dump-pass2"

                if followup_block:
                    log_info(
                        "  Paper shortlist pass2 follow-up: "
                        f"unresolved={len(phase1_unresolved)}, "
                        f"questions={len(phase1_followups)}, "
                        f"backend={followup_backend}, "
                        f"evidence={len(followup_block):,}"
                    )
                    unresolved_block = (
                        "\n".join(f"- {x}" for x in phase1_unresolved)
                        if phase1_unresolved else "- (none listed)"
                    )
                    followup_q_block = (
                        "\n".join(f"- {x}" for x in phase1_followups)
                        if phase1_followups else "- (none listed)"
                    )
                    phase2_prompt = f"""\
You are in PASS 2 of experiment selection.

PASS 1 was intentionally REPO-FIRST: it started from runnable repo experiment
surfaces, then mapped them to the paper's rows using README, config files, and
paper evidence.

Only use the follow-up evidence below to resolve REMAINING ambiguity in
config / metric / setup details after the repo-first mapping. If the
follow-up evidence includes Appendix/Supplementary material, use it only to
fill missing or ambiguous fields. Do NOT let appendix details override a
clear repo surface or README command unless they explicitly resolve that
ambiguity.

PASS 1 OUTPUT (JSON):
{json.dumps(phase1_parsed or {}, indent=2, default=str)}

UNRESOLVED ITEMS:
{unresolved_block}

FOLLOW-UP QUESTIONS:
{followup_q_block}

FOLLOW-UP PAPER EVIDENCE:
{followup_block}

Return the SAME JSON SHAPE as PASS 1 (`title`, `experiments`,
`unresolved_items`, `followup_questions`). Keep already-well-specified
experiments unchanged. Update only fields clarified by the follow-up evidence.
"""
                    phase2_parsed, phase2_text = _run_shortlist_prompt(phase2_prompt, "pass2")
                    if phase2_parsed is not None:
                        parsed = phase2_parsed
                        text = phase2_text

            try:
                if parsed is not None:
                    paper_title = (parsed.get("title") or "").strip()
                    for exp in parsed.get("experiments", [])[:max_candidates]:
                        em = exp.get("expected_metric") or {}
                        runtime_min = exp.get("est_runtime_minutes")
                        try:
                            runtime_f = float(runtime_min)
                        except (TypeError, ValueError):
                            runtime_f = 0.0
                        raw_caveats = exp.get("caveats") or []
                        if isinstance(raw_caveats, str):
                            raw_caveats = [raw_caveats]
                        caveats_list = [
                            str(c)[:400] for c in raw_caveats if c
                        ][:6]
                        raw_missing = exp.get("missing_flags") or []
                        if isinstance(raw_missing, str):
                            raw_missing = [raw_missing]
                        missing_list = [
                            str(m)[:120] for m in raw_missing if m
                        ][:10]
                        raw_cfg_files = exp.get("codebase_config_files") or []
                        if isinstance(raw_cfg_files, str):
                            raw_cfg_files = [raw_cfg_files]
                        codebase_cfg_files = [
                            str(p)[:200] for p in raw_cfg_files if p
                        ][:15]
                        raw_primary = exp.get("primary_metrics") or []
                        if isinstance(raw_primary, dict):
                            raw_primary = [raw_primary]
                        primary_metrics_list: List[Dict[str, Any]] = []
                        for pm in raw_primary[:6]:
                            if not isinstance(pm, dict) or not pm.get("name"):
                                continue
                            primary_metrics_list.append({
                                "name": str(pm.get("name", ""))[:60],
                                "expected_value": str(
                                    pm.get("expected_value")
                                    if pm.get("expected_value") is not None
                                    else pm.get("value", "")
                                )[:60],
                                "tolerance": str(pm.get("tolerance", ""))[:160],
                                "direction": str(pm.get("direction", ""))[:30],
                            })
                        if not primary_metrics_list and em.get("name"):
                            primary_metrics_list.append({
                                "name": str(em.get("name", ""))[:60],
                                "expected_value": str(em.get("value", ""))[:60],
                                "tolerance": str(exp.get("tolerance_rule", ""))[:160],
                                "direction": "",
                            })
                        repo_experiment_id = str(exp.get("repo_experiment_id", "")).strip()
                        repo_surface = repo_experiment_map.get(repo_experiment_id, {})
                        repo_command_source = str(
                            exp.get("repo_command_source")
                            or repo_surface.get("source")
                            or ""
                        )[:40]
                        repo_context = str(
                            exp.get("repo_context")
                            or repo_surface.get("context")
                            or ""
                        )[:400]
                        runtime_metric_source = str(
                            exp.get("runtime_metric_source") or ""
                        )[:400]
                        comparison_mode = str(
                            exp.get("comparison_mode") or "single"
                        ).strip().lower()
                        if comparison_mode not in ("single", "vs_baseline"):
                            comparison_mode = "single"
                        raw_baseline = exp.get("baseline_reference") or {}
                        if not isinstance(raw_baseline, dict):
                            raw_baseline = {}
                        baseline_reference = {
                            "section": str(raw_baseline.get("section", ""))[:120],
                            "repo_experiment_id": str(raw_baseline.get("repo_experiment_id", ""))[:40],
                            "suggested_command": str(raw_baseline.get("suggested_command", ""))[:800],
                            "expected_metric_name": str(raw_baseline.get("expected_metric_name", ""))[:60],
                            "expected_metric_value": str(raw_baseline.get("expected_metric_value", ""))[:60],
                            "notes": str(raw_baseline.get("notes", ""))[:240],
                        }
                        if not any(v for v in baseline_reference.values()):
                            baseline_reference = {}
                        suggested_command = str(
                            exp.get("suggested_command")
                            or repo_surface.get("command")
                            or ""
                        )[:1000]
                        cand = ExperimentCandidate(
                            name=str(exp.get("name", ""))[:200],
                            section=str(exp.get("section", ""))[:120],
                            repo_experiment_id=repo_experiment_id[:40],
                            repo_command_source=repo_command_source,
                            repo_context=repo_context,
                            runtime_metric_source=runtime_metric_source,
                            expected_metric_name=str(em.get("name", ""))[:60],
                            expected_metric_value=str(em.get("value", ""))[:60],
                            expected_metric_units=str(em.get("units", ""))[:30],
                            hardware=str(exp.get("hardware", ""))[:80],
                            est_runtime_minutes=runtime_f,
                            runtime_bucket=self._bucket_runtime(runtime_f)
                                if runtime_f > 0 else "",
                            paper_config=exp.get("paper_config") or {},
                            suggested_command=suggested_command,
                            tolerance_rule=str(exp.get("tolerance_rule", ""))[:160],
                            notes=str(exp.get("notes", ""))[:400],
                            is_baseline=bool(exp.get("is_baseline", False)),
                            caveats=caveats_list,
                            missing_flags=missing_list,
                            config_source=str(exp.get("config_source", ""))[:400],
                            codebase_config_files=codebase_cfg_files,
                            comparison_mode=comparison_mode,
                            baseline_reference=baseline_reference,
                            primary_metrics=primary_metrics_list,
                        )
                        candidates.append(cand)
            except Exception as e:
                import traceback as _tb
                log_warning(
                    "  Paper shortlist: unexpected error while parsing "
                    f"experiments: {type(e).__name__}: {e}"
                )
                log_warning(
                    "  Paper shortlist traceback:\n" + _tb.format_exc()
                )
                if text:
                    log_warning(
                        f"  Paper shortlist response head ({len(text):,} "
                        f"chars): {text[:500]!r}"
                    )
                candidates = []

        # ── Post-process: classify metrics, match code, fill tolerance ──
        for cand in candidates:
            matches = self._match_code_for_experiment(cand.suggested_command, repo_files)
            cand.matched_files = matches
            cand.code_available = bool(matches)
            if cand.est_runtime_minutes <= 0 and cand.runtime_bucket:
                cand.est_runtime_minutes = self._runtime_from_bucket(cand.runtime_bucket)
            # Generic metric classification (wins over LLM's free-text guesses).
            metric_class, default_tol = _classify_metric(
                cand.expected_metric_name, cand.expected_metric_units
            )
            cand.metric_class = metric_class
            if not cand.tolerance_rule:
                cand.tolerance_rule = default_tol
            # If LLM didn't mark is_baseline, fall back to the keyword heuristic.
            if not cand.is_baseline:
                cand.is_baseline = _looks_like_baseline(cand.name) or _looks_like_baseline(
                    cand.suggested_command
                )

        def _score(c: ExperimentCandidate):
            # Tuple sort: smaller is better in every component.
            runtime = c.est_runtime_minutes if c.est_runtime_minutes > 0 else 999.0
            return (
                0 if c.code_available else 1,                    # code must exist
                0 if c.repo_command_source == "readme" else 1,   # README beats inference
                1 if c.is_baseline else 0,                       # method > baseline
                runtime,                                         # shorter runs first
                _METRIC_CLASS_RANK.get(c.metric_class, 3),       # portable metric as tie-breaker
                len(c.suggested_command),                        # simpler cmd first
            )

        candidates.sort(key=_score)
        for i, c in enumerate(candidates):
            c.rank_score = float(i)

        return candidates, paper_title

    def generate_delta_report(self, cuda_results: Dict[str, Any],
                              rocm_results: Dict[str, Any]) -> str:
        """Generate a ROCm vs CUDA comparison report."""
        lines = [
            "=" * 60,
            "ROCm vs CUDA Delta Report",
            "=" * 60,
        ]

        for metric in set(list(cuda_results.keys()) + list(rocm_results.keys())):
            cuda_val = cuda_results.get(metric, "N/A")
            rocm_val = rocm_results.get(metric, "N/A")

            try:
                cuda_num = float(cuda_val)
                rocm_num = float(rocm_val)
                diff_pct = ((rocm_num - cuda_num) / (abs(cuda_num) + 1e-8)) * 100
                lines.append(
                    f"  {metric}: CUDA={cuda_val}, ROCm={rocm_val} "
                    f"(delta: {diff_pct:+.1f}%)"
                )
            except (ValueError, TypeError):
                lines.append(
                    f"  {metric}: CUDA={cuda_val}, ROCm={rocm_val}"
                )

        lines.append("=" * 60)
        return "\n".join(lines)
