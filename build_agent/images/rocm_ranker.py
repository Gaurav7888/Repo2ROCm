"""
ROCm Docker image ranker.

Goal: pick the best possible ROCm base image before the agent starts expensive
pip installs or kernel migration. The ranker is deliberately cost-aware:

* It uses no-pull metadata first: static catalog, DockerHub tags, tag-parsed
  versions, image size, env/label/build-history hints when available.
* It does not claim a full `pip list` unless a later expensive probe provides
  one. The output records which evidence was cheap/inferred.

The downstream planner can still pull/probe the winner in strict mode, but this
ranker should eliminate obviously wrong images without paying that cost.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from knowledge.rocm_knowledge import ROCM_IMAGE_CATALOG, ROCM_PREINSTALLED_PACKAGES


def _norm(token: str) -> str:
    return (token or "").strip().lower().replace("-", "_").replace(".", "_")


def _version_tuple(value: str) -> Tuple[int, ...]:
    try:
        return tuple(int(part) for part in value.split(".") if part != "")
    except ValueError:
        return ()


def _extract_first(pattern: str, text: str) -> str:
    match = re.search(pattern, text or "", flags=re.IGNORECASE)
    return match.group(1) if match else ""


def _tag_matches_python(tag: str, preferred_python: str) -> bool:
    if not preferred_python:
        return True
    py = preferred_python.strip().lower()
    compact = py.replace(".", "")
    low = (tag or "").lower()
    return f"py{py}" in low or f"py{compact}" in low


def _rocm_version_from_tag(tag: str) -> str:
    return _extract_first(r"rocm[_-]?(\d+(?:\.\d+)*)", tag)


def _python_version_from_tag(tag: str) -> str:
    raw = _extract_first(r"py(?:thon)?[_-]?(\d+(?:\.\d+)?)", tag)
    if raw and "." not in raw and len(raw) >= 2:
        return raw[0] + "." + raw[1:]
    return raw


def _component_version_from_tag(component_pattern: str, tag: str) -> str:
    return _extract_first(rf"{component_pattern}[_-]?(\d+(?:\.\d+)*)", tag)


def _gpu_arch_tokens(gpu_arch: str) -> Set[str]:
    arch = (gpu_arch or "").lower()
    tokens = {_norm(arch)} if arch and arch != "unknown" else set()
    if "gfx94" in arch or "mi300" in arch or "mi250" in arch or "gfx90a" in arch:
        tokens.update({"gfx94x", "gfx942", "cdna", "mi300", "mi250"})
    if "gfx95" in arch:
        tokens.update({"gfx950", "gfx95x", "cdna"})
    if "gfx11" in arch or "gfx12" in arch or "rdna" in arch:
        tokens.update({"rdna", "gfx11", "gfx12", "gfx120x"})
    return tokens


def infer_required_tokens(import_counts: Dict[str, int],
                          config_contents: Dict[str, str]) -> Set[str]:
    """Infer repo package/framework needs from imports and config strings."""
    tokens = {_norm(name) for name in (import_counts or {}) if name}
    text = "\n".join((config_contents or {}).values()).lower()
    markers = {
        "torch", "torchvision", "torchaudio", "pytorch", "transformers",
        "tokenizers", "safetensors", "triton", "flash_attn", "flash_attention",
        "xformers", "bitsandbytes", "deepspeed", "accelerate", "megatron",
        "megatron_core", "vllm", "sglang", "jax", "flax", "optax",
        "tensorflow", "keras", "onnxruntime", "onnx", "diffusers", "apex",
        "ninja", "cmake", "flashinfer", "aiter",
    }
    for marker in markers:
        if marker in text or marker.replace("_", "-") in text:
            tokens.add(marker)
    return {t for t in tokens if t}


def infer_preferred_workload(import_counts: Dict[str, int],
                             config_contents: Dict[str, str]) -> str:
    tokens = infer_required_tokens(import_counts, config_contents)
    if "sglang" in tokens:
        return "sglang"
    if "vllm" in tokens:
        return "vllm"
    if "megatron" in tokens or "megatron_core" in tokens:
        return "megatron"
    if tokens & {"deepspeed", "accelerate", "pytorch_lightning", "lightning"}:
        return "pytorch-training"
    if tokens & {"jax", "flax", "optax"} and "torch" not in tokens:
        return "jax"
    if tokens & {"tensorflow", "keras"}:
        return "tensorflow"
    if tokens & {"onnxruntime", "onnx"}:
        return "onnxruntime"
    return "pytorch"


def _jaccard(left: Set[str], right: Set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _tag_sort_key(tag: str) -> Tuple[Tuple[int, ...], Tuple[int, ...], str]:
    return (_version_tuple(_rocm_version_from_tag(tag)), _version_tuple(_python_version_from_tag(tag)), tag)


@dataclass
class ImageRankerConfig:
    gpu_arch: str = "unknown"
    preferred_python: str = ""
    preferred_workload: str = ""
    max_tags_per_repo: int = 30
    size_penalty_gb: float = 0.004
    strict_mode: bool = False


@dataclass
class ImageCandidate:
    workload: str
    image: str
    tag: str
    score: float
    confidence: float
    jaccard: float
    reasons: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    overlap: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def ref(self) -> str:
        return f"{self.image}:{self.tag}"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["ref"] = self.ref
        return data


class RocmImageRanker:
    """Rank ROCm image candidates from cheap metadata and repo signals."""

    def __init__(self, config: Optional[ImageRankerConfig] = None):
        self.config = config or ImageRankerConfig()

    def rank(self, import_counts: Dict[str, int],
             config_contents: Dict[str, str]) -> List[ImageCandidate]:
        desired = infer_required_tokens(import_counts, config_contents)
        preferred_workload = (
            self.config.preferred_workload
            or infer_preferred_workload(import_counts, config_contents)
        )
        arch_tokens = _gpu_arch_tokens(self.config.gpu_arch)
        candidates: List[ImageCandidate] = []

        for workload, entry in ROCM_IMAGE_CATALOG.items():
            image = entry["image"]
            live_tags, tag_error = self._fetch_tags(image)
            tag_records = live_tags or [
                {"name": tag, "full_size": 0, "last_updated": "", "digest": ""}
                for tag in entry.get("tags", []) or [entry.get("default_tag", "latest")]
            ]
            tag = self._choose_tag(image, tag_records, entry)
            selected_record = next((r for r in tag_records if r.get("name") == tag), tag_records[0])
            inventory = self._inventory_tokens(workload, entry, tag, selected_record)
            jaccard = _jaccard(desired, inventory)
            overlap = sorted(desired & inventory)
            missing = sorted(desired - inventory)

            score = jaccard
            reasons = []
            risks = []

            if workload == preferred_workload:
                score += 0.25
                reasons.append(f"matches preferred workload `{preferred_workload}`")
            if live_tags:
                score += 0.06
                reasons.append("tag exists in live DockerHub metadata")
            else:
                risks.append(f"live tag lookup unavailable: {tag_error or 'unknown'}")
            if self.config.preferred_python:
                if _tag_matches_python(tag, self.config.preferred_python):
                    score += 0.08
                    reasons.append(f"tag matches Python {self.config.preferred_python}")
                else:
                    score -= 0.08
                    risks.append(f"tag does not encode Python {self.config.preferred_python}")
            if arch_tokens:
                tag_tokens = {_norm(tok) for tok in re.findall(r"[A-Za-z0-9_.+-]+", tag)}
                if arch_tokens & tag_tokens:
                    score += 0.12
                    reasons.append(f"tag matches GPU arch hint `{self.config.gpu_arch}`")
                elif workload in {"vllm", "sglang"}:
                    score -= 0.10
                    risks.append(f"specialized serving tag may not match GPU arch `{self.config.gpu_arch}`")

            size_gb = self._size_gb(selected_record)
            if size_gb:
                penalty = min(0.15, size_gb * self.config.size_penalty_gb)
                score -= penalty
                if size_gb >= 15:
                    risks.append(f"large image: {size_gb:.1f}GB compressed")
            if tag in {"latest", "main"} and self.config.strict_mode:
                score -= 0.06
                risks.append("floating tag in strict mode")

            confidence = 0.35 + min(0.35, len(overlap) * 0.04)
            if live_tags:
                confidence += 0.15
            if reasons:
                confidence += 0.10
            confidence = max(0.0, min(1.0, confidence))

            candidates.append(ImageCandidate(
                workload=workload,
                image=image,
                tag=tag,
                score=round(score, 4),
                confidence=round(confidence, 4),
                jaccard=round(jaccard, 4),
                reasons=reasons or ["fallback catalog candidate"],
                risks=risks,
                overlap=overlap[:20],
                missing=missing[:20],
                evidence={
                    "source": "dockerhub_live" if live_tags else "static_catalog",
                    "live_tags_checked": len(live_tags),
                    "top_live_tags": [str(r.get("name") or "") for r in tag_records[:8]],
                    "digest": selected_record.get("digest", ""),
                    "last_updated": selected_record.get("last_updated", ""),
                    "compressed_size_gb": round(size_gb, 2) if size_gb else 0,
                    "parsed": self._parse_tag(tag),
                    "desired_tokens": sorted(desired)[:50],
                },
            ))

        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        return candidates

    def _fetch_tags(self, image: str) -> Tuple[List[Dict[str, Any]], str]:
        try:
            from tools.external_lookups import dockerhub_tags_structured
            tags, err = dockerhub_tags_structured(
                image, limit=self.config.max_tags_per_repo,
            )
            return tags, err or ""
        except Exception as exc:
            return [], f"{type(exc).__name__}: {exc}"

    def _choose_tag(self, image: str, tags: List[Dict[str, Any]],
                    entry: Dict[str, Any]) -> str:
        tag_names = [str(tag.get("name") or "") for tag in tags if tag.get("name")]
        if not tag_names:
            return entry.get("default_tag", "latest")
        if image == "rocm/pytorch":
            release_tags = [
                name for name in tag_names
                if "pytorch_release" in name and _rocm_version_from_tag(name)
            ]
            if self.config.preferred_python:
                py_matches = [name for name in release_tags if _tag_matches_python(name, self.config.preferred_python)]
                if py_matches:
                    release_tags = py_matches
            if release_tags:
                return sorted(release_tags, key=_tag_sort_key, reverse=True)[0]
        if self.config.gpu_arch and self.config.gpu_arch != "unknown":
            arch_tokens = _gpu_arch_tokens(self.config.gpu_arch)
            arch_matches = [
                name for name in tag_names
                if arch_tokens & {_norm(tok) for tok in re.findall(r"[A-Za-z0-9_.+-]+", name)}
            ]
            if arch_matches:
                return sorted(arch_matches, key=_tag_sort_key, reverse=True)[0]
        stable = [name for name in tag_names if name not in {"latest", "main"}]
        if stable:
            return sorted(stable, key=_tag_sort_key, reverse=True)[0]
        return tag_names[0]

    def _inventory_tokens(self, workload: str, entry: Dict[str, Any],
                          tag: str, tag_record: Dict[str, Any]) -> Set[str]:
        image = entry.get("image", "")
        tokens = {_norm(pkg) for pkg in ROCM_PREINSTALLED_PACKAGES.get(image, [])}
        tokens.add(_norm(workload))
        tokens.update(_norm(tok) for tok in re.findall(r"[A-Za-z0-9_.+-]+", entry.get("description", "")))
        tokens.update(_norm(tok) for tok in re.findall(r"[A-Za-z0-9_.+-]+", tag))
        parsed = self._parse_tag(tag)
        for key, value in parsed.items():
            if value:
                tokens.add(_norm(key))
                tokens.add(_norm(value))
        if workload == "vllm":
            tokens.update({"vllm", "torch", "pytorch", "transformers", "triton", "aiter"})
        if workload == "sglang":
            tokens.update({"sglang", "vllm", "torch", "pytorch", "triton", "flashinfer"})
        return {token for token in tokens if token}

    def _parse_tag(self, tag: str) -> Dict[str, str]:
        return {
            "rocm": _rocm_version_from_tag(tag),
            "python": _python_version_from_tag(tag),
            "pytorch": _component_version_from_tag("pytorch(?:_release)?", tag),
            "vllm": _component_version_from_tag("vllm", tag),
            "jax": _component_version_from_tag("jax", tag),
            "tensorflow": _component_version_from_tag("tf", tag),
        }

    @staticmethod
    def _size_gb(tag_record: Dict[str, Any]) -> float:
        try:
            size = float(tag_record.get("full_size") or 0)
        except (TypeError, ValueError):
            return 0.0
        return size / (1024 ** 3) if size > 0 else 0.0


def rank_rocm_images(import_counts: Dict[str, int],
                     config_contents: Dict[str, str],
                     gpu_arch: str = "unknown",
                     preferred_python: str = "",
                     preferred_workload: str = "",
                     strict_mode: bool = False) -> List[ImageCandidate]:
    config = ImageRankerConfig(
        gpu_arch=gpu_arch,
        preferred_python=preferred_python,
        preferred_workload=preferred_workload,
        strict_mode=strict_mode,
    )
    return RocmImageRanker(config).rank(import_counts, config_contents)
