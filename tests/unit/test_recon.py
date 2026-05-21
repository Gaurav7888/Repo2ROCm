"""Recon pipeline: deterministic preflight on synthetic repos."""
from __future__ import annotations

from pathlib import Path

import pytest

from repo2rocm.recon import run_recon
from repo2rocm.recon.configs import (
    detect_install_mechanisms,
    detect_python_version,
    parse_requirements,
)
from repo2rocm.recon.hazards import (
    detect_pin_hazards,
    detect_py312_issues,
)
from repo2rocm.recon.image_select import select_rocm_image
from repo2rocm.recon.imports import detect_framework, extract_imports
from repo2rocm.recon.readme import (
    extract_expected_outcomes,
    extract_run_commands,
)
from repo2rocm.recon.requirements_filter import filter_requirements


def _mkrepo(tmp: Path, files: dict[str, str]) -> Path:
    repo = tmp / "repo"
    repo.mkdir()
    for name, content in files.items():
        p = repo / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return repo


def test_parse_requirements_basic():
    rows = parse_requirements("# comment\ntorch==2.4.0\nnumpy>=1.23\n-r other.txt\nflask\n")
    assert ("torch", "==2.4.0") in rows
    assert ("numpy", ">=1.23") in rows
    assert ("flask", "") in rows
    assert all(p != "-r" for p, _ in rows)


def test_detect_python_version_pyproject():
    v = detect_python_version({"pyproject.toml": "requires-python = '>=3.10,<3.13'\n"})
    assert ">=3.10" in v


def test_install_mechanisms_priority():
    mech = detect_install_mechanisms({"requirements.txt": "torch\n", "setup.py": ""})
    assert "pip install -r requirements.txt" in mech
    assert "pip install -e ." in mech


def test_extract_imports_and_framework(tmp_path: Path):
    repo = _mkrepo(tmp_path, {
        "a.py": "import torch\nimport numpy\nfrom torch import nn\n",
        "b.py": "import torch\nimport vllm\n",
    })
    from repo2rocm.recon.files import find_python_files

    counts = extract_imports(find_python_files(repo))
    assert counts["torch"] == 2
    assert counts["vllm"] == 1
    assert detect_framework(counts) == "vllm"


def test_select_image_jax_strong():
    sel = select_rocm_image(
        import_counts={"jax": 5, "flax": 3, "torch": 0},
        config_contents={"requirements.txt": "jax\nflax\n"},
        readme_text="A JAX project using jnp.",
    )
    assert sel.workload == "jax"
    assert sel.image == "rocm/jax"


def test_select_image_falls_back_to_pytorch_when_no_signals():
    sel = select_rocm_image(import_counts={"random": 1}, config_contents={}, readme_text="")
    assert sel.image == "rocm/pytorch"


def test_select_image_specialization_tiebreak():
    sel = select_rocm_image(
        import_counts={"torch": 5, "deepspeed": 3},
        config_contents={"requirements.txt": "torch\ndeepspeed\naccelerate\n"},
        readme_text="distributed training with DeepSpeed and FSDP",
    )
    assert sel.workload in {"pytorch-training", "pytorch"}


def test_select_image_demotes_pytorch_training_without_launcher():
    """Regression: inference repos that import `accelerate` for single-GPU use
    used to be pushed onto `rocm/pytorch-training`. With no torchrun /
    deepspeed / `accelerate launch` evidence the selector now stays on the
    smaller general `rocm/pytorch` image."""
    sel = select_rocm_image(
        import_counts={"torch": 8, "accelerate": 1, "transformers": 4},
        config_contents={"requirements.txt": "torch\ntransformers\naccelerate\n"},
        readme_text=(
            "PrefixKV: efficient KV-cache pruning.\n"
            "Run `python eval.py --model llama-3-8b` to reproduce.\n"
        ),
    )
    assert sel.workload == "pytorch", (
        f"expected pytorch (no launcher signal) but got {sel.workload!r}; "
        f"reasoning={sel.reasoning}"
    )


def test_select_image_keeps_pytorch_training_with_launcher():
    """Counterpart to the demotion test: when the README actually shows
    torchrun/deepspeed/`accelerate launch`, pytorch-training is correct."""
    sel = select_rocm_image(
        import_counts={"torch": 8, "deepspeed": 4, "accelerate": 2},
        config_contents={"requirements.txt": "torch\ndeepspeed\naccelerate\n"},
        readme_text=(
            "Multi-GPU training:\n"
            "```bash\naccelerate launch --num_processes 8 train.py\n```\n"
            "Or with DeepSpeed: `deepspeed --num_gpus 8 train.py`.\n"
        ),
    )
    assert sel.workload == "pytorch-training"


def test_requirements_filter_partitions():
    fr = filter_requirements(
        config_contents={
            "requirements.txt": (
                "torch==2.4.0\n"
                "numpy\n"
                "flash-attn==2.5.0\n"
                "nvidia-cublas-cu12==12.4\n"
                "tqdm\n"
            )
        },
        base_image="rocm/pytorch",
    )
    assert any("flash-attn" in s for s in fr.special_handling)
    assert any("nvidia-cublas" in s for s in fr.skip_banned)
    # torch + numpy are preinstalled on rocm/pytorch
    assert any("torch" in s for s in fr.skip_preinstalled)
    assert any("tqdm" in s for s in fr.install)


def test_detect_py312_issues(tmp_path: Path):
    repo = _mkrepo(tmp_path, {
        "x.py": "import imp\nfrom collections import Mapping\n",
    })
    from repo2rocm.recon.files import find_python_files

    hazards = detect_py312_issues(find_python_files(repo), str(repo))
    kinds = {h.kind for h in hazards}
    assert "py312_removed" in kinds
    assert "py312_collections_abc" in kinds


def test_detect_pin_hazards():
    hazards = detect_pin_hazards({"requirements.txt": "transformers==4.20.0\nscipy>=1.5\n"})
    assert any(h.description.startswith("transformers==4.20.0") for h in hazards)


def test_extract_run_commands_from_fenced_block():
    md = (
        "## Usage\n\n"
        "```bash\n"
        "$ python train.py --steps 1000\n"
        "python -m turboquant.eval\n"
        "```\n"
        "Then run\n```\nmake test\n```\n"
    )
    cmds = extract_run_commands(md)
    assert any("train.py" in c for c in cmds)
    assert any("turboquant" in c for c in cmds)


def test_expected_outcomes_picks_metrics():
    md = "We report perplexity 12.3 and 2.5x speedup on H100.\nUnrelated line."
    out = extract_expected_outcomes(md)
    assert any("perplexity" in o.lower() for o in out)
    assert any("speedup" in o.lower() for o in out)


def test_run_recon_end_to_end(tmp_path: Path):
    repo = _mkrepo(tmp_path, {
        "requirements.txt": "torch==2.4.0\nvllm\nflash-attn\n",
        "README.md": "# vLLM serve\n```bash\npython -m vllm.entrypoints.api_server\n```",
        "src/__init__.py": "",
        "src/main.py": "import torch\nimport vllm\nfrom vllm import LLM\n",
    })
    report = run_recon(
        repo_path=repo,
        repo_full_name="owner/repo",
        mode="functional",
    )
    assert report.mode == "functional"
    assert report.framework == "vllm"
    assert report.image_selection is not None
    assert report.image_selection.workload in {"vllm", "vllm-dev"}
    assert any("flash-attn" in s for s in report.filtered_requirements.special_handling)


def test_run_recon_rejects_bad_mode(tmp_path: Path):
    repo = _mkrepo(tmp_path, {"README.md": "x"})
    with pytest.raises(ValueError):
        run_recon(repo_path=repo, repo_full_name="r", mode="env")
