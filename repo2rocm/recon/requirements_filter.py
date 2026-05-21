"""Partition pip requirements into INSTALL / SKIP-preinstalled / SKIP-banned / SPECIAL."""
from __future__ import annotations

from repo2rocm.knowledge import (
    CUDA_TO_ROCM_MAPPING,
    get_preinstalled_packages,
    is_banned_package,
)
from repo2rocm.recon.configs import parse_requirements
from repo2rocm.recon.report import FilteredRequirements


def filter_requirements(
    *,
    config_contents: dict[str, str],
    base_image: str,
) -> FilteredRequirements:
    preinstalled = {p.lower() for p in get_preinstalled_packages(base_image)}
    fr = FilteredRequirements()

    seen: set[str] = set()
    for fname, content in config_contents.items():
        if not (fname.startswith("requirements") and fname.endswith(".txt")):
            continue
        for pkg, spec in parse_requirements(content):
            key = pkg.lower().replace("_", "-")
            if key in seen:
                continue
            seen.add(key)
            line = pkg + (f" {spec}" if spec else "")

            if is_banned_package(pkg):
                fr.skip_banned.append(line)
                continue
            if pkg.lower() in preinstalled or key in preinstalled:
                fr.skip_preinstalled.append(line)
                continue
            mapping = CUDA_TO_ROCM_MAPPING.get(pkg.lower()) or CUDA_TO_ROCM_MAPPING.get(key)
            if mapping is not None:
                fr.special_handling.append(f"{line}  →  {mapping.get('notes', '')[:120]}")
                continue
            fr.install.append(line)
    return fr
