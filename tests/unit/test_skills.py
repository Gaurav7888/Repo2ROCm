"""Skill discovery: builtins must be present."""
from __future__ import annotations

from repo2rocm.skills import discover_skills, load_skill_body


def test_builtin_skills_discovered():
    cat = discover_skills()
    names = set(cat.manifests)
    assert "rocm_image_catalog" in names
    assert "cuda_to_rocm_mapping" in names
    assert "flash_attn_amd_install" in names
    assert "py312_compat" in names


def test_skill_body_loads():
    cat = discover_skills()
    mf = cat.manifests["rocm_image_catalog"]
    body = load_skill_body(mf)
    assert "rocm/pytorch" in body
    assert "ROCm Image Catalog" in body


def test_menu_text_lists_skills():
    cat = discover_skills()
    text = cat.menu_text()
    assert "/rocm_image_catalog" in text
    assert "/cuda_to_rocm_mapping" in text
