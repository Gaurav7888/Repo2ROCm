"""Deterministic ROCm image scoring.

Uses `IMAGE_SIGNALS` from `knowledge.rocm_data`. No LLM call here; the planner
agent may override based on richer context, but the recon report always carries
a deterministic recommendation that mode=functional can use directly.
"""
from __future__ import annotations

import re

from repo2rocm.knowledge import IMAGE_SIGNALS, ROCM_IMAGE_CATALOG
from repo2rocm.recon.report import ImageSelection

# weights for each evidence channel
_W_IMPORT = 3.0
_W_DEP = 2.0
_W_README = 0.5


def select_rocm_image(
    *,
    import_counts: dict[str, int],
    config_contents: dict[str, str],
    readme_text: str = "",
) -> ImageSelection:
    """Score each candidate image; return the best."""
    config_blob = "\n".join(config_contents.values())
    scores: dict[str, float] = {}
    reasoning: dict[str, list[str]] = {}

    imports_lower = {k.lower().replace("-", "_"): v for k, v in import_counts.items()}

    for image_key, sig in IMAGE_SIGNALS.items():
        score = 0.0
        why: list[str] = []

        for imp in sig.get("strong_imports", []):
            key = imp.lower().replace("-", "_")
            count = imports_lower.get(key, 0)
            if count > 0:
                contrib = _W_IMPORT * (1.0 + 0.1 * min(count, 10))
                score += contrib
                why.append(f"import `{imp}` × {count}")

        for dep in sig.get("strong_deps", []):
            if re.search(rf"\b{re.escape(dep)}\b", config_blob, re.IGNORECASE):
                score += _W_DEP
                why.append(f"dep `{dep}` in config")

        if readme_text:
            for pat in sig.get("readme_patterns", []):
                try:
                    if re.search(pat, readme_text):
                        score += _W_README
                        why.append(f"README mentions /{pat}/")
                        break  # don't double-count
                except re.error:
                    continue

        if score > 0:
            scores[image_key] = score
            reasoning[image_key] = why

    if not scores:
        # Fallback: PyTorch general image.
        catalog = ROCM_IMAGE_CATALOG["pytorch"]
        return ImageSelection(
            image=catalog["image"],
            tag=catalog["default_tag"],
            workload="pytorch",
            score=0.0,
            reasoning=["no strong signals; defaulting to general PyTorch ROCm image"],
        )

    # Prefer the most specialized image when scores are within 1 point of the top
    # (so vllm beats pytorch when both score, but PyTorch wins clear toss-ups).
    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_key, top_score = sorted_scores[0]

    # `accelerate` appears as a dep in many *inference-only* repos that just call
    # `Accelerator()` on a single GPU. That used to push us to the much-larger
    # `rocm/pytorch-training` image. Demote pytorch-training unless there's
    # actual multi-process / distributed-launcher evidence (torchrun / deepspeed /
    # `accelerate launch`, FSDP, multi-GPU training language).
    _LAUNCHER_RE = re.compile(
        r"\b(torchrun|deepspeed|accelerate\s+launch|mpirun|srun|FSDP|"
        r"DistributedDataParallel|--nproc[-_]per[-_]node|--num_processes)\b",
        re.IGNORECASE,
    )
    has_launcher = bool(_LAUNCHER_RE.search(readme_text or "")) or bool(
        _LAUNCHER_RE.search(config_blob)
    )
    if "pytorch-training" in scores and not has_launcher:
        scores["pytorch-training"] = scores["pytorch-training"] * 0.4
        reasoning.setdefault("pytorch-training", []).append(
            "demoted: no multi-process launcher (torchrun/deepspeed/accelerate launch) found"
        )
        sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_key, top_score = sorted_scores[0]

    specialization_rank = {
        "sglang": 0, "vllm-dev": 0, "vllm": 1,
        "jax": 2, "tensorflow": 2, "onnxruntime": 2,
        "megatron": 3, "pytorch-training": 4, "pytorch": 5,
    }
    candidates = [k for k, s in sorted_scores if top_score - s <= 1.0]
    candidates.sort(key=lambda k: specialization_rank.get(k, 99))
    final_key = candidates[0]
    final_score = scores[final_key]

    catalog = ROCM_IMAGE_CATALOG[final_key]
    return ImageSelection(
        image=catalog["image"],
        tag=catalog["default_tag"],
        workload=final_key,
        score=round(final_score, 2),
        reasoning=reasoning[final_key][:8],
    )
