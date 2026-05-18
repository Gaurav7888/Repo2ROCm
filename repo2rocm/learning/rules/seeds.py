"""Seed rules — the bootstrap KB shipped with the distribution."""
from __future__ import annotations

from repo2rocm.learning.kb_store import Rule

SEED_RULES: list[Rule] = [
    Rule(
        name="prefer_rocm_pytorch_for_torch",
        when={"frameworks": "pytorch", "rocm_mode": True},
        do={"recommend_base_image": "rocm/pytorch:latest"},
        source="seed",
        confidence=0.85,
    ),
    Rule(
        name="prefer_rocm_vllm_for_vllm_repo",
        when={"frameworks": "vllm", "rocm_mode": True},
        do={"recommend_base_image": "rocm/vllm:latest"},
        source="seed",
        confidence=0.9,
    ),
    Rule(
        name="strip_banned_nvidia_pkgs",
        when={"rocm_mode": True},
        do={"strip_packages_glob": "nvidia-*-cu1?"},
        source="seed",
        confidence=0.99,
    ),
    Rule(
        name="flash_attn_use_amd_triton",
        when={"requires_package": "flash-attn", "rocm_mode": True},
        do={
            "install_strategy": "git",
            "git_url": "https://github.com/Dao-AILab/flash-attention",
            "env": {"FLASH_ATTENTION_TRITON_AMD_ENABLE": "TRUE"},
            "build_command": "python setup.py install",
        },
        source="seed",
        confidence=0.95,
    ),
    Rule(
        name="python312_distutils_compat",
        when={"python_version": "3.12", "requires_module": "distutils"},
        do={"install": ["setuptools"], "patch": "from setuptools._distutils import ..."},
        source="seed",
        confidence=0.95,
    ),
]
