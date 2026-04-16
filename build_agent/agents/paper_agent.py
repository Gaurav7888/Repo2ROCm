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
from typing import Any, Dict, List, Optional, Tuple

from storage.models import PaperMetadata, ReproductionResult
from utils.llm import get_llm_response


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
