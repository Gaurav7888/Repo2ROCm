"""
Dynamic ROCm planning helpers.

The static knowledge in `rocm_knowledge.py` is useful as a seed, but ROCm
Docker tags and acceleration-library support move quickly. This module turns
repo signals plus live registry evidence into:

1. A Jaccard-style container match score over desired packages/features.
2. A refreshed Docker tag for the selected image repo.
3. Package guidance that branches by GPU architecture, model stack, and how
   much performance degradation the run mode can tolerate.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from knowledge.rocm_knowledge import (
    CUDA_TO_ROCM_MAPPING,
    ROCM_IMAGE_CATALOG,
    ROCM_PREINSTALLED_PACKAGES,
)
from images.rocm_ranker import rank_rocm_images


_MODEL_STACK_KEYWORDS = {
    "llm_serving": {"vllm", "sglang", "fastapi", "uvicorn", "transformers"},
    "llm_training": {"deepspeed", "accelerate", "megatron", "megatron_core", "lightning", "pytorch_lightning"},
    "transformers_inference": {"transformers", "tokenizers", "safetensors", "torch"},
    "diffusion": {"diffusers", "xformers", "transformers", "torch"},
    "jax": {"jax", "flax", "optax"},
    "tensorflow": {"tensorflow", "keras"},
    "onnx": {"onnxruntime", "onnx"},
    "generic_torch": {"torch", "torchvision", "torchaudio"},
}


def normalize_package(name: str) -> str:
    return (name or "").strip().lower().replace("-", "_").replace(".", "_")


def _all_config_text(config_contents: Dict[str, str]) -> str:
    return "\n".join(config_contents.values() if config_contents else []).lower()


def infer_repo_requirements(import_counts: Dict[str, int],
                            config_contents: Dict[str, str]) -> Set[str]:
    """Return normalized feature/package tokens the repo appears to need."""
    reqs: Set[str] = {normalize_package(name) for name in (import_counts or {})}
    text = _all_config_text(config_contents)

    package_markers = {
        "torch", "torchvision", "torchaudio", "transformers", "tokenizers",
        "safetensors", "triton", "flash_attn", "flash_attn_2", "xformers",
        "bitsandbytes", "deepspeed", "accelerate", "megatron", "megatron_core",
        "vllm", "sglang", "jax", "flax", "optax", "tensorflow", "keras",
        "onnxruntime", "onnx", "diffusers", "ninja", "cmake",
    }
    for marker in package_markers:
        if marker.replace("_", "-") in text or marker in text:
            reqs.add(marker)
    return {r for r in reqs if r}


def infer_model_stack(import_counts: Dict[str, int],
                      config_contents: Dict[str, str]) -> str:
    """Classify the repo's dominant model/runtime stack."""
    reqs = infer_repo_requirements(import_counts, config_contents)
    scores: Dict[str, int] = {}
    for stack, markers in _MODEL_STACK_KEYWORDS.items():
        scores[stack] = len(reqs & {normalize_package(m) for m in markers})
    if not scores:
        return "unknown"
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"


def infer_degradation_policy(no_scale_down: bool = False,
                             reproduce_results: bool = False,
                             run_mode: str = "env") -> str:
    """
    Return how willing the plan should be to trade performance for portability.

    - strict: paper reproduction / no-scale-down runs should preserve the
      intended acceleration path whenever possible.
    - moderate: full mode can accept a documented fallback only after trying
      the intended acceleration path.
    - permissive: environment-only smoke tests may use SDPA/eager fallbacks if
      they are clearly marked as degradation.
    """
    if reproduce_results or no_scale_down or run_mode == "reproduce":
        return "strict"
    if run_mode == "full":
        return "moderate"
    return "permissive"


def detect_gpu_arch_hint() -> str:
    """
    Best-effort host/env GPU architecture hint.

    Sources, in order:
      1. Operator overrides via REPO2ROCM_GPU_ARCH / PYTORCH_ROCM_ARCH /
         AMDGPU_TARGETS / HSA_OVERRIDE_GFX_VERSION. These let benchmark
         harnesses force a specific arch (or one we cross-compile for).
      2. Live host probe via `images.rocm_ranker._detect_host_gpu_arch`,
         which shells out to `rocm-smi` / `nvidia-smi` once per process
         (memoized) and is the new default so the planner self-discovers
         the host arch without an explicit CLI flag.
      3. 'unknown' when nothing identifies a GPU.
    """
    for key in ("REPO2ROCM_GPU_ARCH", "PYTORCH_ROCM_ARCH", "AMDGPU_TARGETS", "HSA_OVERRIDE_GFX_VERSION"):
        val = os.environ.get(key, "").strip()
        if val:
            return val
    try:
        from images.rocm_ranker import _detect_host_gpu_arch
        detected = (_detect_host_gpu_arch() or "").strip()
    except Exception:
        detected = ""
    return detected or "unknown"


def _image_inventory_tokens(workload: str, entry: Dict[str, Any],
                            live_tags: Optional[List[Dict[str, Any]]] = None) -> Set[str]:
    repo = entry.get("image", "")
    tokens: Set[str] = {
        normalize_package(p)
        for p in ROCM_PREINSTALLED_PACKAGES.get(repo, [])
    }
    tokens.add(normalize_package(workload))
    tokens.update(normalize_package(t) for t in re.findall(r"[A-Za-z0-9_.+-]+", entry.get("description", "")))
    for tag in (live_tags or [])[:12]:
        tokens.update(normalize_package(t) for t in re.findall(r"[A-Za-z0-9_.+-]+", tag.get("name", "")))
    return {t for t in tokens if t}


def _jaccard(left: Set[str], right: Set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _fetch_live_tags(image_repo: str, limit: int = 25) -> Tuple[List[Dict[str, Any]], str]:
    try:
        from tools.external_lookups import dockerhub_tags_structured
        tags, err = dockerhub_tags_structured(image_repo, limit=limit)
        return tags, err or ""
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


def _tag_version_tuple(tag_name: str) -> Tuple[int, ...]:
    match = re.search(r"rocm(\d+(?:\.\d+)*)", tag_name or "", flags=re.IGNORECASE)
    if not match:
        return ()
    try:
        return tuple(int(part) for part in match.group(1).split("."))
    except ValueError:
        return ()


def _tag_matches_python(tag_name: str, preferred_python: str) -> bool:
    if not preferred_python:
        return True
    py = preferred_python.strip().lower()
    py_compact = py.replace(".", "")
    tag = (tag_name or "").lower()
    return f"py{py}" in tag or f"py{py_compact}" in tag


def choose_live_tag(image_repo: str,
                    static_default: str = "latest",
                    static_tags: Optional[Iterable[str]] = None,
                    preferred_python: str = "",
                    live_tags: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Pick a Docker tag from live DockerHub data, falling back to static defaults.

    For `rocm/pytorch`, prefer explicit release tags over floating `latest` so
    reproduced Dockerfiles remain stable. For specialized repos that only
    publish `main`/`latest`, keep those floating tags when they are live.
    """
    tags = list(live_tags or [])
    error = ""
    if not tags:
        tags, error = _fetch_live_tags(image_repo, limit=40)

    static_set = {str(t) for t in (static_tags or []) if t}
    if not tags:
        return {
            "tag": static_default or "latest",
            "source": "static_fallback",
            "reason": error or "live DockerHub lookup unavailable",
            "live_tags_checked": 0,
        }

    tag_names = [str(t.get("name") or "") for t in tags if t.get("name")]
    preferred: List[str] = []

    if image_repo == "rocm/pytorch":
        release_tags = [
            name for name in tag_names
            if "pytorch_release" in name and "rocm" in name
        ]
        if preferred_python:
            py_matches = [name for name in release_tags if _tag_matches_python(name, preferred_python)]
            if py_matches:
                release_tags = py_matches
        preferred = sorted(
            release_tags,
            key=lambda name: (_tag_version_tuple(name), name),
            reverse=True,
        )

    if not preferred:
        preferred = [
            name for name in tag_names
            if name in static_set or name in {"latest-release", "latest", "main"}
        ]

    selected = preferred[0] if preferred else tag_names[0]
    return {
        "tag": selected,
        "source": "dockerhub_live",
        "reason": f"selected from {len(tag_names)} live DockerHub tags",
        "live_tags_checked": len(tag_names),
        "top_live_tags": tag_names[:8],
    }


def select_image_with_jaccard(import_counts: Dict[str, int],
                              config_contents: Dict[str, str],
                              preferred_workload: str = "",
                              preferred_python: str = "") -> Dict[str, Any]:
    """
    Score ROCm images by overlap between repo-required packages/features and
    the image inventory inferred from preinstalled packages + live tag tokens.
    """
    ranked = rank_rocm_images(
        import_counts=import_counts,
        config_contents=config_contents,
        gpu_arch=detect_gpu_arch_hint(),
        preferred_python=preferred_python,
        preferred_workload=preferred_workload,
    )
    winner = ranked[0] if ranked else None
    if winner is None:
        entry = ROCM_IMAGE_CATALOG["pytorch"]
        return {
            "selected_workload": "pytorch",
            "selected_image": entry["image"],
            "selected_tag": entry.get("default_tag", "latest"),
            "selected_image_ref": f"{entry['image']}:{entry.get('default_tag', 'latest')}",
            "desired_tokens": sorted(infer_repo_requirements(import_counts, config_contents))[:40],
            "tag_info": {"tag": entry.get("default_tag", "latest"), "source": "static_fallback"},
            "scores": [],
        }
    tag_info = dict(winner.evidence or {})
    tag_info.update({
        "tag": winner.tag,
        "source": tag_info.get("source", "ranker"),
        "reason": "; ".join(winner.reasons[:2]),
        "live_tags_checked": tag_info.get("live_tags_checked", 0),
    })
    return {
        "selected_workload": winner.workload,
        "selected_image": winner.image,
        "selected_tag": winner.tag,
        "selected_image_ref": winner.ref,
        "desired_tokens": tag_info.get("desired_tokens", []),
        "tag_info": tag_info,
        "scores": [
            {
                "workload": c.workload,
                "image": c.image,
                "score": c.score,
                "confidence": c.confidence,
                "jaccard": c.jaccard,
                "overlap": c.overlap,
                "missing": c.missing,
                "live_tags": c.evidence.get("top_live_tags", []),
                "risks": c.risks,
                "reasons": c.reasons,
            }
            for c in ranked[:8]
        ],
    }


def _arch_family(gpu_arch: str) -> str:
    arch = (gpu_arch or "").lower()
    if any(x in arch for x in ("gfx90a", "gfx94", "mi200", "mi250", "mi300", "cdna")):
        return "cdna"
    if any(x in arch for x in ("gfx10", "gfx11", "gfx12", "rdna", "7900", "9070")):
        return "rdna"
    return "unknown"


def package_guidance_for(dep: str,
                         model_stack: str,
                         gpu_arch: str,
                         degradation_policy: str) -> List[str]:
    """Return context-aware guidance lines for one CUDA-ish dependency."""
    dep_norm = normalize_package(dep)
    arch = _arch_family(gpu_arch)
    strict = degradation_policy == "strict"
    lines: List[str] = []

    if dep_norm in {"flash_attn", "flash_attn_2", "flash_attention", "flash_attn-2"}:
        lines.append("preferred: use upstream Dao-AILab/flash-attention with AMD backend, not PyPI CUDA wheels")
        if arch == "rdna":
            lines.append("architecture: RDNA detected/hinted -> prefer Triton backend (`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`)")
        elif arch == "cdna":
            lines.append("architecture: CDNA/Instinct detected/hinted -> CK backend is viable; Triton backend remains a good fallback")
        else:
            lines.append("architecture: unknown -> start with Triton backend because it covers both CDNA and RDNA more broadly")
        lines.append("install: `git clone https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention && cd /tmp/flash-attention && FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE python setup.py install`")
        if strict:
            lines.append("degradation: strict mode -> do not silently patch to eager/SDPA unless install fails and verdict records the degradation")
        else:
            lines.append("degradation: permissive mode -> PyTorch SDPA fallback is acceptable for env verification if clearly marked")
        return lines

    if dep_norm == "xformers":
        lines.append("preferred: use ROCm/xformers source build when xFormers kernels are required")
        if arch == "cdna":
            lines.append("install: `git clone https://github.com/ROCm/xformers.git /tmp/xformers && cd /tmp/xformers && git submodule update --init --recursive && PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH:-gfx942} python setup.py install`")
        else:
            lines.append("install: if GPU arch is unknown/RDNA, try PyTorch SDPA first; source-build xFormers only when the repo requires xformers-specific APIs")
        if strict:
            lines.append("degradation: strict mode -> if replacing xFormers with SDPA, mark paper result NOT_REPRODUCED unless metric remains valid")
        return lines

    if dep_norm == "bitsandbytes":
        lines.append("preferred: use ROCm-aware bitsandbytes (`bitsandbytes-rocm` or ROCm/bitsandbytes source), never NVIDIA CUDA wheels")
        if model_stack in {"llm_serving", "transformers_inference", "llm_training"}:
            lines.append("model stack: LLM stack detected -> preserve quantization only if ROCm backend imports and runs on the target GPU")
        if strict:
            lines.append("degradation: strict mode -> do not silently switch 4-bit/8-bit experiments to fp16/bf16")
        else:
            lines.append("degradation: permissive mode -> fp16/bf16 fallback is acceptable for env smoke tests if memory fits")
        return lines

    if dep_norm == "deepspeed":
        lines.append("preferred: use `rocm/pytorch-training` or a ROCm PyTorch image with matching DeepSpeed support")
        lines.append("install: if source build is required, set `DS_BUILD_OPS=1 DS_BUILD_AIO=0` and verify `torch.version.hip` first")
        if strict:
            lines.append("degradation: strict mode -> disabling fused/deepspeed ops can invalidate training-performance claims")
        return lines

    if dep_norm == "triton":
        lines.append("preferred: use Triton from the selected ROCm PyTorch image; avoid unpinned reinstall unless `import triton` fails")
        lines.append("verification: run `python -c \"import triton, torch; print(triton.__version__, torch.version.hip)\"` before changing versions")
        return lines

    mapping = CUDA_TO_ROCM_MAPPING.get(dep) or CUDA_TO_ROCM_MAPPING.get(dep.replace("_", "-"))
    if mapping:
        lines.append(f"preferred: `{mapping.get('rocm_package')}`")
        if mapping.get("install_cmd"):
            lines.append(f"install: `{mapping['install_cmd']}`")
        if mapping.get("notes"):
            lines.append(f"notes: {mapping['notes']}")
    else:
        lines.append("no static ROCm mapping; use `pypi_versions`, project docs, and live web evidence before installing")
    return lines


def build_dynamic_package_guidance(cuda_deps: List[str],
                                   import_counts: Dict[str, int],
                                   config_contents: Dict[str, str],
                                   no_scale_down: bool = False,
                                   reproduce_results: bool = False,
                                   run_mode: str = "env",
                                   gpu_arch: str = "") -> List[str]:
    if not cuda_deps:
        return []
    model_stack = infer_model_stack(import_counts, config_contents)
    arch = gpu_arch or detect_gpu_arch_hint()
    policy = infer_degradation_policy(no_scale_down, reproduce_results, run_mode)
    lines = [
        "Context-aware CUDA-to-ROCm Package Guidance:",
        f"  Model stack: {model_stack}",
        f"  GPU architecture hint: {arch}",
        f"  Acceptable degradation policy: {policy}",
    ]
    for dep in cuda_deps:
        lines.append(f"  - {dep}:")
        for note in package_guidance_for(dep, model_stack, arch, policy):
            lines.append(f"      {note}")
    return lines
