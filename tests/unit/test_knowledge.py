"""Knowledge tables: lookups and consistency."""
from repo2rocm.knowledge import (
    BANNED_NVIDIA_PACKAGES,
    CUDA_TO_ROCM_MAPPING,
    IMAGE_SIGNALS,
    ROCM_IMAGE_CATALOG,
    ROCM_PREINSTALLED_PACKAGES,
    get_preinstalled_packages,
    get_rocm_alternative,
    is_banned_package,
)


def test_image_catalog_has_required_keys():
    for key, entry in ROCM_IMAGE_CATALOG.items():
        assert {"image", "tags", "default_tag", "description"} <= entry.keys(), key
        assert entry["default_tag"] in entry["tags"], key
        assert entry["image"].startswith("rocm/"), key


def test_signals_keys_align_with_catalog():
    # every IMAGE_SIGNALS key must have a catalog entry
    for key in IMAGE_SIGNALS:
        assert key in ROCM_IMAGE_CATALOG, key


def test_preinstalled_lookup():
    assert "torch" in get_preinstalled_packages("rocm/pytorch")
    assert "torch" in get_preinstalled_packages("rocm/pytorch:rocm6.3_ubuntu22.04_py3.10_pytorch_release_2.4.0")
    assert get_preinstalled_packages("rocm/unknown") == []


def test_get_rocm_alternative_known():
    rec = get_rocm_alternative("flash-attn")
    assert rec is not None
    assert "FLASH_ATTENTION_TRITON_AMD_ENABLE" in rec["install_cmd"]
    rec2 = get_rocm_alternative("flash_attn")
    assert rec2 is not None
    rec3 = get_rocm_alternative("not-a-real-pkg")
    assert rec3 is None


def test_banned_detection():
    assert is_banned_package("nvidia-cublas-cu12")
    assert is_banned_package("nvidia-cublas-cu12==12.4")
    assert not is_banned_package("torch")


def test_preinstalled_keys_subset_of_image_repos():
    catalog_images = {v["image"] for v in ROCM_IMAGE_CATALOG.values()}
    for k in ROCM_PREINSTALLED_PACKAGES:
        assert k in catalog_images, k


def test_banned_list_nonempty():
    assert len(BANNED_NVIDIA_PACKAGES) >= 10
    assert any(b.endswith("cu12") for b in BANNED_NVIDIA_PACKAGES)


def test_cuda_to_rocm_mapping_has_install_or_alternative():
    for name, rec in CUDA_TO_ROCM_MAPPING.items():
        assert "notes" in rec, name
        # either has an install_cmd (positive recipe) or has a None alternative
        # with explanatory notes
        if rec["rocm_package"] is not None:
            assert rec["install_cmd"] is not None or rec["notes"]
