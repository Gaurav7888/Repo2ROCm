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

import functools
import re
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

from knowledge.rocm_knowledge import ROCM_IMAGE_CATALOG, ROCM_PREINSTALLED_PACKAGES


# ── Architecture compatibility metadata ─────────────────────────────────────
#
# Hard-coded GPU arch facts the ranker depends on. These are purely additive:
# they only feed `arch_compatible()` (the hard pre-filter) and tag selection.
# Existing scoring weights are untouched.

ARCH_TOKEN_ALIASES: Dict[str, Set[str]] = {
    "gfx906": {"mi50", "vega20"},
    "gfx908": {"mi100"},
    "gfx90a": {"mi200", "mi210", "mi250"},
    "gfx940": {"mi300", "mi300a", "gfx94x", "gfx94"},
    "gfx941": {"mi300", "mi300x", "gfx94x", "gfx94"},
    "gfx942": {"mi300", "mi300x", "mi300a", "gfx94x", "gfx94"},
    "gfx950": {"mi35x", "mi355x", "mi350x", "mi350", "gfx95x", "gfx95"},
}

ARCH_RELEASE_DATE_MAP: Dict[str, str] = {
    # Conservative lower bound for the host arch's public availability. Images
    # pushed BEFORE this date are treated as known-incompatible when a date is
    # available. Missing push_date never rejects on date alone.
    "gfx950": "2026-01-01",  # MI350 / MI355 series (per planner spec)
}

# Workload widening: some host arches ship a working PyTorch + ROCm stack in
# *non-rocm/pytorch* repos before the generic image catalog catches up. We
# tag those repos as workload="pytorch" so they compete on the same Jaccard
# footing as the boring case; the `tag_filter` field constrains us to the
# arch-specific tags within those repos.
ARCH_BONUS_REPOS: Dict[str, List[Dict[str, Any]]] = {
    "gfx950": [
        {
            "workload": "pytorch",
            "image": "rocm/sgl-dev",
            "tag_filter": "mi35x",
            "default_tag": "main",
            "description": (
                "rocm/sgl-dev mi35x build -- ships a working PyTorch + ROCm "
                "stack for gfx950 (MI350/MI355). Suitable as a generic "
                "PyTorch base on gfx950 hosts before rocm/pytorch carries "
                "gfx950 tags."
            ),
        },
        {
            "workload": "pytorch",
            "image": "lmsysorg/sglang",
            "tag_filter": "mi35x",
            "default_tag": "latest",
            "description": (
                "LMSYS sglang mi35x build -- ships a working PyTorch + ROCm "
                "stack for gfx950. Same role as rocm/sgl-dev:*mi35x* for "
                "planning purposes."
            ),
        },
    ],
}


@functools.lru_cache(maxsize=1)
def _detect_host_gpu_arch() -> str:
    """Return the host's GPU arch token (e.g. 'gfx950') or '' if unknown.

    Best-effort: probes rocm-smi then nvidia-smi. Memoized for the lifetime
    of the process so the planner pays the subprocess cost at most once.
    Never raises.
    """
    for cmd in (
        ["rocm-smi", "--showproductname"],
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
    ):
        try:
            out = subprocess.check_output(
                cmd, stderr=subprocess.DEVNULL, timeout=5,
            ).decode(errors="replace")
        except Exception:
            continue
        m = re.search(r"\bgfx[0-9a-f]+\b", out, re.IGNORECASE)
        if m:
            return m.group(0).lower()
        if any(tok in out for tok in ("Tesla", "A100", "H100", "RTX")):
            return "cuda"
    return ""


@functools.lru_cache(maxsize=1)
def _local_image_set() -> FrozenSet[str]:
    """Return locally-cached `repo:tag` strings. Memoized; empty on failure."""
    try:
        out = subprocess.check_output(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            stderr=subprocess.DEVNULL, timeout=10,
        ).decode(errors="replace")
    except Exception:
        return frozenset()
    return frozenset(
        line.strip()
        for line in out.splitlines()
        if line.strip() and "<none>" not in line
    )


def _date_to_tuple(value: str) -> Optional[Tuple[int, int, int]]:
    if not value:
        return None
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", value)
    if not m:
        return None
    try:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _normalize_host_arch(arch: str) -> str:
    a = (arch or "").strip().lower()
    return "" if (not a or a == "unknown") else a


def _bounded_contains(haystack_low: str, token: str) -> bool:
    """Substring match with non-alphanumeric word boundaries."""
    if not token:
        return False
    return re.search(
        rf"(^|[^a-z0-9]){re.escape(token)}([^a-z0-9]|$)", haystack_low
    ) is not None


def arch_compatible(image_tag: str, host_arch: str, push_date: str = "") -> bool:
    """Return True if the image tag plausibly supports host_arch.

    Rules (in order, first match wins):
      1. If host_arch is empty/unknown -> allow all (preserve current behavior).
      2. If the tag explicitly contains host_arch (e.g. 'gfx950') -> True.
      3. If the tag contains any alias for host_arch (e.g. 'mi35x' for
         gfx950, 'gfx94x' for gfx942) -> True.
      4. If the image's push_date is BEFORE the host_arch release date,
         reject as known-incompatible.
      5. If the tag explicitly names a *different* arch (e.g. 'mi300x'
         or 'gfx94x' on a gfx950 host) and that arch is NOT a subset of
         host_arch's aliases -> reject.
      6. Otherwise allow (the generic `rocm/pytorch:rocm7.2_ubuntu...`
         tags don't carry an arch token and may still be safe).

    Missing push_date metadata never rejects on date alone.
    """
    host = _normalize_host_arch(host_arch)
    if not host:
        return True  # rule 1

    tag_low = (image_tag or "").lower()
    host_aliases = {a.lower() for a in ARCH_TOKEN_ALIASES.get(host, set())}

    # rule 2
    if host in tag_low:
        return True

    # rule 3
    for alias in host_aliases:
        if _bounded_contains(tag_low, alias):
            return True

    # rule 4: build-date heuristic
    release_date_s = ARCH_RELEASE_DATE_MAP.get(host)
    if release_date_s:
        release_t = _date_to_tuple(release_date_s)
        push_t = _date_to_tuple(push_date)
        if release_t and push_t and push_t < release_t:
            return False

    # rule 5a: any other gfx token in the tag is an explicitly different arch
    for tok in re.findall(r"gfx[0-9a-f]+x?", tag_low):
        if tok == host or tok in host_aliases:
            return True  # safety net for rules 2/3
        return False

    # rule 5b: an alias that uniquely identifies a different arch
    for other_arch, other_aliases in ARCH_TOKEN_ALIASES.items():
        if other_arch == host:
            continue
        unique_aliases = {a.lower() for a in other_aliases} - host_aliases
        for alias in unique_aliases:
            if _bounded_contains(tag_low, alias):
                return False

    # rule 6: generic tag with no arch token. For arches whose generic
    # catalog repos do not yet ship arch-compatible builds (i.e. arches
    # we track in ARCH_BONUS_REPOS), require an explicit arch tag and
    # reject the generic fallback. For arches without that signal we
    # preserve the historical allow-by-default behavior.
    if host in ARCH_BONUS_REPOS:
        return False
    return True


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
    tokens: Set[str] = set()
    if not arch or arch == "unknown":
        return tokens
    tokens.add(_norm(arch))
    # Canonical aliases for this exact arch
    for alias in ARCH_TOKEN_ALIASES.get(arch, set()):
        tokens.add(_norm(alias))
    # Legacy family-level hints (kept for backwards compat with existing scoring)
    if "gfx94" in arch or "mi300" in arch or "mi250" in arch or "gfx90a" in arch:
        tokens.update({"gfx94x", "gfx942", "cdna", "mi300", "mi250"})
    if "gfx95" in arch:
        tokens.update({"gfx950", "gfx95x", "cdna", "mi35x", "mi350x", "mi355x"})
    if "gfx11" in arch or "gfx12" in arch or "rdna" in arch:
        tokens.update({"rdna", "gfx11", "gfx12", "gfx120x"})
    return {t for t in tokens if t}


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
        effective_arch = _normalize_host_arch(self.config.gpu_arch)
        arch_tokens = _gpu_arch_tokens(self.config.gpu_arch)
        local_images = _local_image_set()
        candidates: List[ImageCandidate] = []

        # Iterate the base catalog plus any arch-specific bonus repos. Bonus
        # repos let us surface e.g. `rocm/sgl-dev:*mi35x*` as a pytorch base
        # on gfx950 hosts where the generic rocm/pytorch repo does not yet
        # carry a gfx950 tag.
        base_entries = list(ROCM_IMAGE_CATALOG.items())
        bonus_entries: List[Tuple[str, Dict[str, Any]]] = [
            (entry.get("workload", "pytorch"), entry)
            for entry in ARCH_BONUS_REPOS.get(effective_arch, [])
        ]

        for workload, entry in (base_entries + bonus_entries):
            image = entry["image"]
            tag_filter = entry.get("tag_filter")
            live_tags, tag_error = self._fetch_tags(image)
            tag_records = live_tags or [
                {"name": tag, "full_size": 0, "last_updated": "", "digest": ""}
                for tag in entry.get("tags", []) or [entry.get("default_tag", "latest")]
            ]
            # Inject locally-cached tags for this image so they survive even
            # when DockerHub's recency-ordered tag list has scrolled past them
            # (typical for daily-built dev images like rocm/sgl-dev). The +1.0
            # cache bonus in `rank()` is otherwise wasted because the chosen
            # tag would always be the latest uncached one. We only do this for
            # arch-bonus entries: the base catalog already relies on its live
            # tag list being canonical, and unconditionally injecting cached
            # tags there would surface random local dev tags as candidates.
            if tag_filter:
                for ref in local_images:
                    if not ref.startswith(f"{image}:"):
                        continue
                    local_tag = ref.split(":", 1)[1]
                    if not any(r.get("name") == local_tag for r in tag_records):
                        tag_records.append({
                            "name": local_tag, "full_size": 0,
                            "last_updated": "", "digest": "",
                        })

            if tag_filter:
                filtered = [
                    r for r in tag_records
                    if tag_filter.lower() in str(r.get("name") or "").lower()
                ]
                if not filtered:
                    # Bonus repo has no arch-matching tag right now -- skip.
                    continue
                tag_records = filtered

            tag = self._choose_tag(image, tag_records, entry)
            selected_record = next((r for r in tag_records if r.get("name") == tag), tag_records[0])
            push_date = str(selected_record.get("last_updated") or "")

            # ── HARD ARCH FILTER (Change 2): runs BEFORE scoring. ───────────
            # If host arch is empty/unknown this is a no-op (rule 1).
            if not arch_compatible(tag, effective_arch, push_date):
                continue

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

            # ── LOCAL CACHE PREFERENCE (Change 3): +1.0 bonus for cached
            # arch-matching images. arch_compatible has already gated us so
            # we never prefer a locally-cached but arch-incompatible image.
            ref = f"{image}:{tag}"
            locally_cached = ref in local_images
            if locally_cached:
                score += 1.0
                reasons.append("locally cached arch-matching image (no pull needed)")

            if tag_filter:
                reasons.append(
                    f"arch-specific repo for host `{effective_arch or 'unknown'}` "
                    f"(tag filter: `{tag_filter}`)"
                )

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
                    "host_arch": effective_arch,
                    "locally_cached": locally_cached,
                    "arch_filter": "hard" if effective_arch else "off",
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
        # Prefer locally-cached arch-matching tags: this lets the +1.0 cache
        # bonus in `rank()` actually fire for repos like rocm/sgl-dev where
        # DockerHub publishes many newer-but-uncached mi35x tags every day.
        # We use bounded substring matching here (not _norm-tokenization)
        # because the legacy `[A-Za-z0-9_.+-]+` regex collapses an entire
        # dashed/dotted tag into a single token and would never intersect
        # with short arch aliases like `mi35x`.
        if self.config.gpu_arch and self.config.gpu_arch != "unknown":
            local_cache = _local_image_set()
            cached_here = [n for n in tag_names if f"{image}:{n}" in local_cache]
            if cached_here:
                arch_tokens = _gpu_arch_tokens(self.config.gpu_arch)
                cached_arch = [
                    n for n in cached_here
                    if any(_bounded_contains(n.lower(), tok) for tok in arch_tokens)
                ]
                if cached_arch:
                    return sorted(cached_arch, key=_tag_sort_key, reverse=True)[0]
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
        # Arch-specific bonus repos (Change 4) are marketed as PyTorch-compatible
        # bases for the target arch -- inject the typical torch/pytorch stack so
        # they pick up Jaccard overlap with plain PyTorch repos.
        if entry.get("tag_filter") and workload == "pytorch":
            tokens.update({
                "torch", "pytorch", "torchvision", "torchaudio",
                "triton", "transformers", "numpy",
            })
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
    # Change 1: if the caller did not supply a host arch (or said
    # 'unknown'), auto-detect once per process. Operators can still force a
    # specific arch via env vars / explicit argument. Detection is memoized
    # and never raises.
    if not _normalize_host_arch(gpu_arch):
        detected = _detect_host_gpu_arch()
        if detected:
            gpu_arch = detected
    config = ImageRankerConfig(
        gpu_arch=gpu_arch,
        preferred_python=preferred_python,
        preferred_workload=preferred_workload,
        strict_mode=strict_mode,
    )
    return RocmImageRanker(config).rank(import_counts, config_contents)
