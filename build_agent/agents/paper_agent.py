"""
Paper Reproduction Agent — extracts metadata from research papers
and verifies reproduction on ROCm.

Workflow:
1. Extract arXiv/paper links from README
2. Fetch paper via arXiv API / Semantic Scholar
3. LLM-based structured extraction (hardware, libraries, benchmarks, commands)
4. Scale-adjusted reproduction plan
5. Run experiments, compare against paper metrics
6. ROCm vs CUDA delta report
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from storage.models import ExperimentCandidate, PaperMetadata, ReproductionResult
from utils.llm import get_llm_response


_DEFAULT_ENTRY_SCRIPT_PATTERNS = [
    r"(?<!\w)(\w+\.py)\b",
    r"(?<!\w)(\w+\.sh)\b",
]


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
                text = response[0].strip()
                if text.startswith("```"):
                    text = re.sub(r"^```\w*\n?", "", text)
                    text = re.sub(r"\n?```$", "", text)
                data = json.loads(text)
                return PaperMetadata.from_dict(data)
        except Exception:
            pass

        return PaperMetadata()

    def create_reproduction_plan(self, metadata: PaperMetadata,
                                 available_gpu_memory_gb: float = 24.0
                                 ) -> List[Dict[str, Any]]:
        """
        Create a scale-adjusted reproduction plan.

        Full reproduction at paper scale is often impossible. This creates
        a feasible plan that validates correctness:
        - Reduce batch/seq length/model size to fit available VRAM
        - Reduce training steps to just enough to confirm loss curves
        - For inference, run on smaller inputs but verify throughput scales
        """
        plan = []

        for cmd in metadata.reproduction_commands:
            step = {
                "original_command": cmd,
                "scaled_command": cmd,
                "scale_factor": 1.0,
                "notes": "",
            }

            batch_match = re.search(r"--batch[_-]size\s+(\d+)", cmd)
            if batch_match:
                original_bs = int(batch_match.group(1))
                scaled_bs = min(original_bs, max(1, int(available_gpu_memory_gb / 4)))
                if scaled_bs < original_bs:
                    step["scaled_command"] = re.sub(
                        r"--batch[_-]size\s+\d+",
                        f"--batch_size {scaled_bs}",
                        cmd
                    )
                    step["scale_factor"] = scaled_bs / original_bs
                    step["notes"] = f"Batch size reduced from {original_bs} to {scaled_bs}"

            steps_match = re.search(r"--(?:max_)?steps\s+(\d+)", cmd)
            if steps_match:
                original_steps = int(steps_match.group(1))
                scaled_steps = min(original_steps, 100)
                if scaled_steps < original_steps:
                    step["scaled_command"] = re.sub(
                        r"--(?:max_)?steps\s+\d+",
                        f"--max_steps {scaled_steps}",
                        step["scaled_command"]
                    )
                    step["notes"] += f" Steps reduced from {original_steps} to {scaled_steps}"

            epochs_match = re.search(r"--(?:num_)?epochs?\s+(\d+)", cmd)
            if epochs_match:
                original_epochs = int(epochs_match.group(1))
                scaled_epochs = min(original_epochs, 2)
                if scaled_epochs < original_epochs:
                    step["scaled_command"] = re.sub(
                        r"--(?:num_)?epochs?\s+\d+",
                        f"--num_epochs {scaled_epochs}",
                        step["scaled_command"]
                    )
                    step["notes"] += f" Epochs reduced from {original_epochs} to {scaled_epochs}"

            plan.append(step)

        return plan

    def compare_results(self, expected: Dict[str, str],
                        actual: Dict[str, str],
                        scale_factor: float = 1.0
                        ) -> ReproductionResult:
        """
        Compare reproduction results against paper-reported metrics.

        Returns match status: "match", "partial", or "mismatch".
        """
        result = ReproductionResult(
            expected_output=json.dumps(expected),
            actual_output=json.dumps(actual),
            scaled=scale_factor != 1.0,
            scale_factor=scale_factor,
        )

        if not expected or not actual:
            result.match_status = "partial"
            return result

        match_count = 0
        total = len(expected)

        for metric, expected_val in expected.items():
            actual_val = actual.get(metric)
            if actual_val is None:
                continue

            try:
                exp_num = float(re.search(r"[\d.]+", str(expected_val)).group())
                act_num = float(re.search(r"[\d.]+", str(actual_val)).group())

                tolerance = 0.1 if scale_factor == 1.0 else 0.3
                rel_diff = abs(act_num - exp_num) / (abs(exp_num) + 1e-8)
                result.metric_deltas[metric] = rel_diff

                if rel_diff <= tolerance:
                    match_count += 1
            except (ValueError, AttributeError):
                if str(expected_val).lower() == str(actual_val).lower():
                    match_count += 1

        if total == 0:
            result.match_status = "partial"
        elif match_count == total:
            result.match_status = "match"
        elif match_count > 0:
            result.match_status = "partial"
        else:
            result.match_status = "mismatch"

        return result

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
        tokens: List[str] = []
        for pat in _DEFAULT_ENTRY_SCRIPT_PATTERNS:
            tokens.extend(re.findall(pat, suggested_command))
        tokens = [t for t in tokens if t and len(t) > 3]
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

    def shortlist_experiments(
        self,
        paper_pdf_path: Optional[str],
        repo_path: str,
        readme_content: str = "",
        llm: Optional[str] = None,
        max_candidates: int = 8,
        run_memory: Optional[Any] = None,
        graphify_provider: Optional[Any] = None,
    ) -> Tuple[List[ExperimentCandidate], str]:
        """
        Enumerate experiments from the paper, cross-reference with code in the repo,
        and return (ranked_candidates, paper_title).

        Ranking priorities (most-significant first):
          1. `code_available` — must be True; experiments with no code match are
             pushed to the bottom.
          2. `is_baseline` — pure baselines (no-cache / --origin / vanilla / naive)
             are pushed below method experiments, because reproducing only a
             baseline does NOT demonstrate the paper's contribution.
          3. `metric_class` portability — ratio/speedup/percentage > accuracy-style
             > perplexity/loss > absolute throughput/latency. Hardware-portable
             metrics are preferred because we typically run on a different GPU
             than the paper (e.g. AMD MI250X vs NVIDIA A100).
          4. Shorter estimated runtime.
          5. Shorter suggested command (tie-breaker).

        This ranking is generic — it uses only metric names/units and a small
        set of keyword heuristics, no paper-specific logic.
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

            # ── Stage 4: paper text via mempalace retrieval (was 350K dump) ──
            paper_block = ""
            mempalace_used = False
            if run_memory is not None and getattr(run_memory, "enabled", False):
                try:
                    paper_block = run_memory.recall_paper(
                        queries=(
                            "main results table headline metric accuracy F1 EM "
                            "perplexity speedup",
                            "hyperparameters learning rate batch size epochs seed "
                            "lora rank alpha gamma weight decay warmup",
                            "experimental setup datasets benchmarks model sizes "
                            "evaluation protocol",
                            "method algorithm initialization theorem proposition "
                            "preconditioner gradient SVD",
                        ),
                        n_per_query=3,
                        token_budget=8000,
                        per_chunk_max_chars=1500,
                    ) or ""
                    if paper_block:
                        mempalace_used = True
                except Exception as _e:
                    print(f"[paper_agent] mempalace recall_paper failed: {_e}")
            if not paper_block and paper_text:
                # Legacy fallback: the original 350K dump.
                paper_block = (
                    "PAPER TEXT (full body, including tables and appendix):\n"
                    + paper_text[:350000]
                )

            # ── Stage 4: repo file list via graphify (deterministic, ranked) ──
            entry_scripts: List[str] = []
            if graphify_provider is not None and getattr(graphify_provider, "enabled", False):
                try:
                    if not os.path.exists(getattr(graphify_provider, "graph_json", "")):
                        graphify_provider.build_or_refresh()
                    entry_scripts = graphify_provider.list_entry_scripts(max_files=12)
                except Exception as _e:
                    print(f"[paper_agent] graphify lookup failed: {_e}")
            files_for_prompt = entry_scripts if entry_scripts else repo_files[:200]

            # ── Stage 5a: configs_block via graphify keyword query (was 65K dump) ──
            # Old behavior dumped EVERY yaml/toml/json file verbatim (~65K chars on
            # the LoRA-One repo). New behavior asks graphify for the configs whose
            # nodes match hyperparameter-related terms, then renders only those.
            configs_block = ""
            configs_method = "fallback-dump"
            if graphify_provider is not None and getattr(graphify_provider, "enabled", False):
                try:
                    cfg_snippet = graphify_provider.query(
                        "config files hyperparameters learning rate batch size "
                        "epochs lora rank alpha optimizer scheduler",
                        token_budget=2500, depth=1, top_seeds=8,
                    ) or ""
                    # Pair the graphify subgraph with the actual file contents
                    # of any yaml/toml/json node it surfaces.
                    relevant_paths = self._select_relevant_configs(
                        cfg_snippet, repo_configs, max_paths=4)
                    if relevant_paths:
                        chunks = [
                            f"# ---- {p} ----\n{repo_configs[p][:6000]}"
                            for p in relevant_paths if p in repo_configs
                        ]
                        if chunks:
                            configs_block = (
                                "REPO CONFIG FILES (graphify-selected, most "
                                "hyperparameter-relevant; full repo had "
                                f"{len(repo_configs)} config files):\n\n"
                                + "\n\n".join(chunks)
                            )
                            configs_method = f"graphify ({len(chunks)}/{len(repo_configs)} files)"
                except Exception as _e:
                    print(f"[paper_agent] graphify configs query failed: {_e}")
            if not configs_block:
                # Legacy fallback: dump everything (capped per-file).
                if repo_configs:
                    config_chunks = [
                        f"# ---- {path} ----\n{content[:8000]}"
                        for path, content in list(repo_configs.items())[:12]
                    ]
                    configs_block = "\n\n".join(config_chunks)
                else:
                    configs_block = "(no yaml/toml/json/cfg config files found in the repo)"

            print(
                f"[paper_agent] Stage 4+5a prompt sources: "
                f"paper_block={len(paper_block):,} chars "
                f"(mempalace={'yes' if mempalace_used else 'fallback'}), "
                f"readme={len(readme_slice):,}, "
                f"configs_block={len(configs_block):,} ({configs_method}), "
                f"entry_scripts={len(files_for_prompt)}"
            )

            prompt = f"""\
You are analyzing a research paper to pick ONE experiment we can reproduce on
different GPU hardware than the paper used (e.g. the paper used NVIDIA
A100/A6000/H100 but we will run on AMD MI250X/MI300X). Your goal: choose an
experiment that captures the paper's CORE CONTRIBUTION, whose metric is
MEANINGFUL across different GPUs, AND whose EXACT configuration (all
hyperparameters) is unambiguously determined by RECONCILING the paper with
the codebase's actual config files.

{paper_block}

README (full):
{readme_slice}

REPO FILES (candidate entry scripts):
{json.dumps(files_for_prompt, indent=2)}

REPO CONFIG FILES (yaml/toml/json/cfg/ini — these are the codebase's actual
default hyperparameters; treat them as authoritative when the paper is silent
or specifies a sweep without picking a value):
{configs_block}

INSTRUCTIONS:
1. READ THE FULL PAPER AND FULL README END-TO-END before answering. Also READ
   THE REPO CONFIG FILES end-to-end. In particular scan Experiments/Methods/
   Ablations sections, ALL Tables (including captions and footnotes), ALL
   Figure captions, the Setup subsection, and the Appendix/Supplementary —
   hyperparameters are often buried there.
2. RECONCILE paper and codebase. The codebase config files are usually the
   exact values the paper authors used; the paper may quote them generically
   ("we tune the learning rate via grid search") while the yaml/toml file
   ships with the actual chosen value. Reconciliation rules:
   - If the paper FIXES a value (e.g. "lr=2e-4") and the codebase ALSO has it,
     they should agree — use that value. If they disagree, prefer the PAPER's
     value (it is the authoritative reproduction target) and add a caveat
     citing the conflict.
   - If the paper is AMBIGUOUS (e.g. "grid search over {{1e-3, 5e-4, 2e-4,
     1e-4}}", "see appendix") but the codebase has a hardcoded value, USE
     THE CODEBASE VALUE — that is what the authors actually ran. Cite the
     yaml/toml path in `config_source`.
   - If neither paper nor codebase specifies a value, fall back to the
     entry-script default and note that in `caveats`.
3. For EACH candidate experiment, extract the FULL configuration. Include:
   - model name (exact HuggingFace repo id when possible)
   - batch_size, sequence length, steps/epochs, learning_rate, optimizer,
     lr_scheduler, warmup_ratio, weight_decay, block_length, temperature,
     cfg_scale, sampling algorithm, cache_steps, window_size, precision
     (fp16/bf16/int8), decoding algorithm, peft/lora hyperparams
   - dataset / benchmark name (MMLU, GSM8K, HumanEval, MRPC, etc.)
   - any environment variables the paper/README calls out
4. PRECISELY include every non-default flag in `suggested_command`. Do NOT
   rely on script defaults — spell them out, even if the value matches the
   yaml default (so the experiment is reproducible without depending on
   environment). If the paper's config cannot be expressed through the
   script's CLI (e.g. the script hardcodes batch_size=1), list the
   unreachable flags in `missing_flags`.
5. List every yaml/toml/json/cfg file that governs this experiment's
   hyperparameters in `codebase_config_files` (relative paths). The runtime
   agent will use these to override values rather than guessing.
6. Check the paper AND the README for DISCLAIMERS about the config, e.g.
   "speedup not significant at batch_size=1", "requires H100 for 10x claim",
   "accuracy measured on MMLU only". Put every disclaimer in `caveats`.

Return a STRICT JSON object with this shape (no markdown fences, no prose):
{{
  "title": "<paper title if you can tell, else empty string>",
  "experiments": [
    {{
      "name": "<short descriptive name; include the method name if applicable>",
      "section": "<paper section, table, or figure reference, e.g. 'Table 1, Sec 4.2'>",
      "expected_metric": {{
        "name": "<e.g. speedup, accuracy, F1, PPL, throughput, latency>",
        "value": "<numeric value as string, e.g. '2.5', '3x', '78.4'>",
        "units": "<e.g. x, %, tokens/s, ms, '' >"
      }},
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
      "config_source": "<verbatim phrase / table cell / yaml path where you
        found each value, e.g. 'paper Table 2 + conf/model/t5base.yaml lr=1e-4
        + Appendix G.4 stable_gamma=64'>",
      "codebase_config_files": [
        "<relative path of every yaml/toml/json/cfg file that governs this
         experiment's hyperparameters, e.g. 'conf/model/t5base.yaml'>"
      ],
      "suggested_command": "<shell command with EVERY non-default flag spelled
        out explicitly, taken from the README verbatim when possible, extended
        with flags from the paper's config and codebase yaml — do NOT invent
        flags the script doesn't support; put those in missing_flags instead.
        Use Hydra-style overrides like '++model.learning_rate=1e-4' if the
        codebase uses Hydra/OmegaConf>",
      "missing_flags": ["<flag the paper used but the script can't accept>"],
      "caveats": [
        "<paper/README disclaimer verbatim, e.g. 'README: speedup not
          significant at batch_size=1, can sometimes be slower'>",
        "<e.g. 'Paper: 10x claim requires prefill length >= 2048'>"
      ],
      "tolerance_rule": "<e.g. '<=15% relative for speedup', '<=3 absolute
        points for accuracy', '<=25% relative for absolute throughput on
        different GPU'>",
      "notes": "<anything else, e.g. 'compute speedup = method_tok/s /
        baseline_tok/s locally'>"
    }}
  ]
}}

RULES for choosing and ordering experiments:
- Reproducing the paper means demonstrating the paper's CONTRIBUTION, not just
  rerunning a baseline. PREFER experiments that exercise the paper's proposed
  method (`is_baseline: false`). Include baselines only as fallbacks or as the
  denominator of a speedup ratio.
- PREFER metrics that are hardware-portable across GPUs:
  * FIRST:  ratios / speedups / percentages / compression factors
  * SECOND: accuracy-style metrics (accuracy, F1, BLEU, EM, Top-k, pass@k)
  * THIRD:  quality metrics (PPL, loss, NLL, reward)
  * AVOID picking absolute throughput or latency as the headline unless
    nothing else is available.
- If the paper's headline is a speedup X vs. baseline Y, set the chosen
  experiment to run the METHOD config and put "compute speedup =
  method_tok/s / baseline_tok/s" in `notes`.
- PENALIZE experiments whose `caveats` make reproduction on our setup
  impossible (e.g. "requires 8xH100", "requires proprietary dataset"). Put
  them lower in the ranking. Prefer experiments whose config is fully
  reproducible from the repo and HuggingFace alone.
- Prefer inference/demo/benchmark experiments under ~30 min.
- Return AT LEAST 3 and AT MOST {max_candidates} experiments, best first.
  The first experiment MUST be: (method, not baseline) AND (portable metric)
  AND (config fully specified) AND (code in the repo) — in that priority order.
- `suggested_command` MUST use flags the entry script actually supports. If
  the paper uses flags the script can't accept, put them in `missing_flags`
  rather than inventing them.
- `est_runtime_minutes` is your estimate for a single run on one modern GPU.
"""
            try:
                messages = [{"role": "user", "content": prompt}]
                response, _ = get_llm_response(
                    effective_llm, messages, temperature=0.1, max_tokens=4096
                )
                if response and response[0]:
                    text = response[0].strip()
                    if text.startswith("```"):
                        text = re.sub(r"^```\w*\n?", "", text)
                        text = re.sub(r"\n?```$", "", text)
                    parsed = json.loads(text)
                    paper_title = (parsed.get("title") or "").strip()
                    for exp in parsed.get("experiments", [])[:max_candidates]:
                        em = exp.get("expected_metric") or {}
                        runtime_min = exp.get("est_runtime_minutes")
                        try:
                            runtime_f = float(runtime_min)
                        except (TypeError, ValueError):
                            runtime_f = 0.0
                        # Normalize optional list fields (LLM may omit or return non-lists)
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
                        cand = ExperimentCandidate(
                            name=str(exp.get("name", ""))[:200],
                            section=str(exp.get("section", ""))[:120],
                            expected_metric_name=str(em.get("name", ""))[:60],
                            expected_metric_value=str(em.get("value", ""))[:60],
                            expected_metric_units=str(em.get("units", ""))[:30],
                            hardware=str(exp.get("hardware", ""))[:80],
                            est_runtime_minutes=runtime_f,
                            runtime_bucket=self._bucket_runtime(runtime_f)
                                if runtime_f > 0 else "",
                            paper_config=exp.get("paper_config") or {},
                            suggested_command=str(exp.get("suggested_command", ""))[:1000],
                            tolerance_rule=str(exp.get("tolerance_rule", ""))[:160],
                            notes=str(exp.get("notes", ""))[:400],
                            is_baseline=bool(exp.get("is_baseline", False)),
                            caveats=caveats_list,
                            missing_flags=missing_list,
                            config_source=str(exp.get("config_source", ""))[:400],
                            codebase_config_files=codebase_cfg_files,
                        )
                        candidates.append(cand)
            except Exception:
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
                1 if c.is_baseline else 0,                       # method > baseline
                _METRIC_CLASS_RANK.get(c.metric_class, 3),       # portable metric first
                runtime,                                         # shorter runs first
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
