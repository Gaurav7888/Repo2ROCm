"""
Upfront Reconnaissance & Strategy Planner.

Performs deep static analysis of a cloned repository before any Docker
commands are executed.  Produces a comprehensive, actionable plan that is:
  1. Printed to the terminal for operator visibility.
  2. Injected into the Configuration agent's system prompt so the LLM
     has a strong strategic prior from turn 1.

Deep analysis includes:
  - Python 3.12 compatibility hazards (removed stdlib modules)
  - Version pin hazards (old pins that won't have wheels)
  - Code-level hazards (wandb, hardcoded paths, cudnn flags, large epochs)
  - Filtered requirements (pre-installed / banned packages removed)
  - CUDA-to-ROCm migration specifics
"""

import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple, Set

from knowledge.rocm_knowledge import (
    ROCM_IMAGE_CATALOG,
    ROCM_PREINSTALLED_PACKAGES,
    CUDA_TO_ROCM_MAPPING,
    BANNED_NVIDIA_PACKAGES,
)
from knowledge.amd_rocm_repos import get_relevant_amd_repos
from utils.json_utils import load_json_loose
from utils.llm import get_llm_response
from utils.rich_logger import log_phase, log_info, log_success, log_warning, console


# ── constants ────────────────────────────────────────────────────────────────

_CONFIG_FILES = [
    "requirements.txt", "requirements_dev.txt", "requirements-dev.txt",
    "requirements_test.txt", "requirements-test.txt",
    "setup.py", "setup.cfg", "pyproject.toml",
    "Pipfile", "Pipfile.lock",
    "environment.yml", "environment.yaml",
    "poetry.lock", "tox.ini", "Makefile",
    ".python-version",
]

_README_NAMES = ["README.md", "readme.md", "README.rst", "README.txt", "README"]

_MAX_FILE_CHARS = 4000
_MAX_SOURCE_SCAN_FILES = None

# stdlib modules removed or significantly changed in Python 3.12
_PY312_REMOVED_MODULES = {
    "imp": "importlib",
    "distutils": "setuptools",
    "aifc": "(removed, no replacement)",
    "audioop": "(removed, no replacement)",
    "cgi": "(removed, use urllib or frameworks)",
    "cgitb": "(removed, no replacement)",
    "chunk": "(removed, no replacement)",
    "crypt": "(removed, no replacement)",
    "imghdr": "(removed, use python-magic or filetype)",
    "mailcap": "(removed, no replacement)",
    "msilib": "(removed, no replacement)",
    "nis": "(removed, no replacement)",
    "nntplib": "(removed, no replacement)",
    "ossaudiodev": "(removed, no replacement)",
    "pipes": "(removed, use subprocess)",
    "sndhdr": "(removed, no replacement)",
    "spwd": "(removed, no replacement)",
    "sunau": "(removed, no replacement)",
    "telnetlib": "(removed, use telnetlib3)",
    "uu": "(removed, use base64)",
    "xdrlib": "(removed, no replacement)",
}

# collections ABCs moved to collections.abc in 3.10+
_PY310_COLLECTIONS_ABCS = [
    "Mapping", "MutableMapping", "Sequence", "MutableSequence",
    "Set", "MutableSet", "Iterable", "Iterator", "Callable",
    "ByteString", "Hashable", "Sized", "Container",
]

# Packages that are known to not have Python 3.12 wheels for old versions
_OLD_PIN_HAZARD_CUTOFFS = {
    "transformers": "4.36.0",
    "tokenizers": "0.15.0",
    "scipy": "1.11.0",
    "scikit-learn": "1.3.0",
    "pandas": "2.1.0",
    "numpy": "1.26.0",
    "pillow": "10.0.0",
    "grpcio": "1.58.0",
}


# ── file helpers ─────────────────────────────────────────────────────────────

def _read_file(path: str, max_chars: Optional[int] = _MAX_FILE_CHARS) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", errors="ignore") as f:
            return f.read() if max_chars is None else f.read(max_chars)
    except Exception:
        return None


def _find_python_files(repo_path: str, limit: Optional[int] = _MAX_SOURCE_SCAN_FILES) -> List[str]:
    py_files = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__" and d != "node_modules"]
        for f in files:
            if f.endswith(".py"):
                py_files.append(os.path.join(root, f))
                if limit is not None and len(py_files) >= limit:
                    return py_files
    return py_files


# ── import analysis ──────────────────────────────────────────────────────────

def _extract_imports(py_files: List[str]) -> Dict[str, int]:
    import_counts: Dict[str, int] = {}
    import_re = re.compile(r"^\s*(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_]*)")
    for fpath in py_files:
        content = _read_file(fpath, max_chars=8000)
        if not content:
            continue
        seen_in_file: Set[str] = set()
        for line in content.splitlines():
            m = import_re.match(line)
            if m:
                pkg = m.group(1)
                if pkg not in seen_in_file:
                    seen_in_file.add(pkg)
                    import_counts[pkg] = import_counts.get(pkg, 0) + 1
    return import_counts


def _detect_cuda_deps(import_counts: Dict[str, int], config_contents: Dict[str, str]) -> List[str]:
    cuda_indicators: Set[str] = set()
    cuda_keywords = {"cuda", "cudnn", "nccl", "nvidia", "nvml", "pynvml", "bitsandbytes",
                     "flash_attn", "flash_attn_2", "apex", "xformers", "triton"}
    for pkg in import_counts:
        if pkg.lower() in cuda_keywords or "cuda" in pkg.lower() or "nvidia" in pkg.lower():
            cuda_indicators.add(pkg)
    all_config = "\n".join(config_contents.values())
    for banned in BANNED_NVIDIA_PACKAGES:
        if banned in all_config:
            cuda_indicators.add(banned)
    for mapped_pkg in CUDA_TO_ROCM_MAPPING:
        if mapped_pkg in all_config or mapped_pkg in import_counts:
            cuda_indicators.add(mapped_pkg)
    return sorted(cuda_indicators)


def _detect_framework(import_counts: Dict[str, int]) -> str:
    """Quick framework label for display purposes (not used for image selection)."""
    priority = [
        (["vllm"], "vllm"),
        (["sglang"], "sglang"),
        (["jax", "flax"], "jax"),
        (["tensorflow", "keras"], "tensorflow"),
        (["onnxruntime", "onnx"], "onnxruntime"),
        (["torch", "pytorch"], "pytorch"),
    ]
    for pkgs, label in priority:
        for pkg in pkgs:
            if pkg in import_counts:
                return label
    return "unknown"


# ── deep analysis: Python 3.12 compatibility ─────────────────────────────────

def _detect_py312_compat_issues(py_files: List[str], repo_path: str) -> List[Dict]:
    """Scan source files for Python 3.12 incompatibilities."""
    issues = []
    removed_import_re = re.compile(
        r"^\s*(?:import|from)\s+(" + "|".join(re.escape(m) for m in _PY312_REMOVED_MODULES) + r")\b"
    )
    collections_abc_re = re.compile(
        r"from\s+collections\s+import\s+.*\b(" + "|".join(_PY310_COLLECTIONS_ABCS) + r")\b"
    )

    for fpath in py_files:
        content = _read_file(fpath, max_chars=10000)
        if not content:
            continue
        rel_path = os.path.relpath(fpath, repo_path)
        for i, line in enumerate(content.splitlines(), 1):
            m = removed_import_re.match(line)
            if m:
                mod = m.group(1)
                replacement = _PY312_REMOVED_MODULES[mod]
                issues.append({
                    "file": rel_path, "line": i, "module": mod,
                    "fix": f"Replace `import {mod}` with `import {replacement}`",
                    "sed": f"sed -i 's/import {mod}/import {replacement} as {mod}/' /repo/{rel_path}",
                })
            m2 = collections_abc_re.match(line)
            if m2:
                issues.append({
                    "file": rel_path, "line": i, "module": "collections",
                    "fix": f"Change `from collections import {m2.group(1)}` to `from collections.abc import {m2.group(1)}`",
                    "sed": f"sed -i 's/from collections import/from collections.abc import/' /repo/{rel_path}",
                })
    return issues


# ── deep analysis: version pin hazards ───────────────────────────────────────

def _parse_requirements_lines(content: str) -> List[Tuple[str, str]]:
    """Parse requirements.txt content into (package_name, version_spec) pairs."""
    results = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        m = re.match(r"^([a-zA-Z0-9_\-\.]+)\s*(.*)", line)
        if m:
            results.append((m.group(1).strip(), m.group(2).strip()))
    return results


def _version_tuple(ver_str: str) -> Tuple[int, ...]:
    try:
        return tuple(int(x) for x in ver_str.split("."))
    except ValueError:
        return (0,)


def _detect_version_pin_hazards(config_contents: Dict[str, str]) -> List[Dict]:
    """Flag version pins that are too old for Python 3.12 wheels."""
    hazards = []
    for fname, content in config_contents.items():
        if not fname.startswith("requirements"):
            continue
        for pkg_name, ver_spec in _parse_requirements_lines(content):
            pkg_lower = pkg_name.lower().replace("_", "-")
            if pkg_lower not in _OLD_PIN_HAZARD_CUTOFFS:
                continue
            pinned_match = re.match(r"[=<>~!]*=\s*([0-9]+\.[0-9]+(?:\.[0-9]+)?)", ver_spec)
            if not pinned_match:
                continue
            pinned_ver = pinned_match.group(1)
            cutoff = _OLD_PIN_HAZARD_CUTOFFS[pkg_lower]
            if _version_tuple(pinned_ver) < _version_tuple(cutoff):
                hazards.append({
                    "file": fname,
                    "package": pkg_name,
                    "pinned": pinned_ver,
                    "cutoff": cutoff,
                    "fix": (f"`{pkg_name}=={pinned_ver}` has no Python 3.12 wheels. "
                            f"Will require building from source (needs Rust/C compiler). "
                            f"Recommend: drop the pin and use `pip install {pkg_name}` (latest)."),
                })
    return hazards


# ── deep analysis: code-level hazards ────────────────────────────────────────

def _detect_code_hazards(py_files: List[str], repo_path: str) -> List[Dict]:
    """Scan source code for patterns that cause runtime problems."""
    hazards = []
    patterns = [
        (re.compile(r"wandb\.login\s*\("), "wandb_login",
         "wandb.login() will fail without API key. Set WANDB_MODE=offline."),
        (re.compile(r"torch\.backends\.cudnn\.\w+\s*="), "cudnn_flags",
         "cudnn flags may error on ROCm. Guard with `if not getattr(torch.version, 'hip', None)`."),
        (re.compile(r"nvidia-smi|nvidia_smi"), "nvidia_smi",
         "nvidia-smi not available on ROCm. Replace with rocm-smi."),
        (re.compile(r"torch\.cuda\.amp\."), "deprecated_amp",
         "torch.cuda.amp deprecated in PyTorch 2.x. Use torch.amp.autocast('cuda')."),
    ]

    for fpath in py_files:
        content = _read_file(fpath, max_chars=10000)
        if not content:
            continue
        rel_path = os.path.relpath(fpath, repo_path)
        for line_no, line in enumerate(content.splitlines(), 1):
            for pat, kind, desc in patterns:
                if pat.search(line):
                    hazards.append({
                        "file": rel_path, "line": line_no,
                        "kind": kind, "description": desc,
                        "code": line.strip()[:120],
                    })
    return hazards


# ── deep analysis: training parameter detection ─────────────────────────────

# Exhaustive list of parameter names that control training duration.
# Covers snake_case, camelCase, UPPER_CASE, hyphenated (YAML/CLI), and
# both singular and plural forms.
_TRAINING_PARAM_NAMES = [
    # epochs
    "epochs", "epoch", "num_epochs", "n_epochs", "max_epochs", "max_epoch",
    "total_epochs", "training_epochs", "train_epochs", "nb_epochs",
    "num_epoch", "n_epoch", "nepochs", "EPOCHS", "NUM_EPOCHS", "MAX_EPOCHS",
    # steps
    "max_steps", "num_steps", "total_steps", "n_steps", "training_steps",
    "train_steps", "max_train_steps", "num_training_steps", "nsteps",
    "max_step", "num_step", "total_step",
    "MAX_STEPS", "NUM_STEPS", "TOTAL_STEPS",
    # iterations
    "iterations", "num_iterations", "n_iterations", "max_iterations",
    "max_iter", "num_iter", "n_iter", "total_iterations", "total_iter",
    "iters", "max_iters", "num_iters", "niters", "niter",
    "ITERATIONS", "MAX_ITER", "NUM_ITERATIONS",
    # training samples / data sizes
    "num_train", "ntrain", "num_train_samples", "train_samples",
    "num_test", "ntest", "num_test_samples", "test_samples",
    "num_samples", "n_samples", "total_samples",
    "num_val", "nval", "num_eval", "eval_samples",
    # batches
    "num_batches", "n_batches", "max_batches",
    # rounds / cycles
    "num_rounds", "n_rounds", "max_rounds", "rounds",
    "num_cycles", "n_cycles", "max_cycles",
    # warmup (large warmup = long training)
    "warmup_steps", "warmup_epochs", "num_warmup_steps",
]

# Build regex alternation from the name list (case-insensitive matching
# is done at compile time; we also allow hyphenated forms for YAML/CLI).
_PARAM_NAME_ALT = "|".join(
    re.escape(n).replace("_", "[_\\-]") for n in sorted(set(_TRAINING_PARAM_NAMES), key=len, reverse=True)
)


def _detect_training_params(py_files: List[str], repo_path: str) -> List[Dict]:
    """
    Exhaustively find large training-duration parameters across:
      - Python source (assignments, dict literals, argparse defaults, dataclass fields)
      - YAML config files
      - JSON config files
      - TOML config files
      - .cfg / .ini config files
    """
    results = []
    seen: Set[Tuple[str, int]] = set()

    # ── Pattern 1: Python assignments & dict entries ─────────────────────
    # Matches: epochs = 200, "epochs": 200, 'epochs': 200, epochs=200
    py_assign_re = re.compile(
        rf"""(?:['"]?(?:{_PARAM_NAME_ALT})['"]?\s*[=:]\s*)(\d+)""",
        re.IGNORECASE,
    )

    # ── Pattern 2: argparse defaults ─────────────────────────────────────
    # Matches: add_argument('--epochs', ..., default=1000, ...)
    # Matches: add_argument('--epochs', ..., default = 1000, ...)
    argparse_flag_re = re.compile(
        rf"""add_argument\s*\(\s*['"]--?({_PARAM_NAME_ALT})['"]""",
        re.IGNORECASE,
    )
    argparse_default_re = re.compile(
        r"""default\s*=\s*(\d+)""",
        re.IGNORECASE,
    )

    # ── Pattern 3: dataclass / attrs fields ──────────────────────────────
    # Matches: epochs: int = 200
    dataclass_re = re.compile(
        rf"""({_PARAM_NAME_ALT})\s*:\s*\w+\s*=\s*(\d+)""",
        re.IGNORECASE,
    )

    def _add_result(rel_path: str, line_no: int, val: int, code: str, source: str):
        key = (rel_path, line_no)
        if key in seen or val <= 5:
            return
        seen.add(key)
        results.append({
            "file": rel_path, "line": line_no,
            "value": val, "code": code.strip()[:120],
            "source": source,
            "sed": f"sed -i '{line_no}s/{val}/2/' /repo/{rel_path}",
        })

    # ── Scan Python files ────────────────────────────────────────────────
    for fpath in py_files:
        content = _read_file(fpath, max_chars=20000)
        if not content:
            continue
        rel_path = os.path.relpath(fpath, repo_path)
        for line_no, line in enumerate(content.splitlines(), 1):
            # Pattern 1: assignments / dict entries
            for m in py_assign_re.finditer(line):
                _add_result(rel_path, line_no, int(m.group(1)), line, "python_assign")

            # Pattern 2: argparse defaults
            flag_m = argparse_flag_re.search(line)
            if flag_m:
                def_m = argparse_default_re.search(line)
                if def_m:
                    _add_result(rel_path, line_no, int(def_m.group(1)), line, "argparse_default")

            # Pattern 3: dataclass fields
            for m in dataclass_re.finditer(line):
                _add_result(rel_path, line_no, int(m.group(2)), line, "dataclass_field")

    # ── Scan YAML config files ───────────────────────────────────────────
    yaml_param_re = re.compile(
        rf"""^\s*({_PARAM_NAME_ALT})\s*:\s*(\d+)""",
        re.IGNORECASE,
    )
    yaml_exts = (".yml", ".yaml")
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__" and d != "node_modules"]
        rel_root = os.path.relpath(root, repo_path)
        if rel_root.count(os.sep) > 3:
            continue
        for f in files:
            if not any(f.endswith(ext) for ext in yaml_exts):
                continue
            fpath = os.path.join(root, f)
            content = _read_file(fpath, max_chars=10000)
            if not content:
                continue
            rel_path = os.path.relpath(fpath, repo_path)
            for line_no, line in enumerate(content.splitlines(), 1):
                m = yaml_param_re.match(line)
                if m:
                    _add_result(rel_path, line_no, int(m.group(2)), line, "yaml_config")

    # ── Scan JSON config files ───────────────────────────────────────────
    json_param_re = re.compile(
        rf"""['"]({_PARAM_NAME_ALT})['"]?\s*:\s*(\d+)""",
        re.IGNORECASE,
    )
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__" and d != "node_modules"]
        rel_root = os.path.relpath(root, repo_path)
        if rel_root.count(os.sep) > 3:
            continue
        for f in files:
            if not f.endswith(".json") or f in ("package.json", "package-lock.json", "tsconfig.json"):
                continue
            fpath = os.path.join(root, f)
            content = _read_file(fpath, max_chars=10000)
            if not content:
                continue
            rel_path = os.path.relpath(fpath, repo_path)
            for line_no, line in enumerate(content.splitlines(), 1):
                for m in json_param_re.finditer(line):
                    _add_result(rel_path, line_no, int(m.group(2)), line, "json_config")

    # ── Scan TOML config files ───────────────────────────────────────────
    toml_param_re = re.compile(
        rf"""({_PARAM_NAME_ALT})\s*=\s*(\d+)""",
        re.IGNORECASE,
    )
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__" and d != "node_modules"]
        rel_root = os.path.relpath(root, repo_path)
        if rel_root.count(os.sep) > 3:
            continue
        for f in files:
            if not f.endswith(".toml"):
                continue
            fpath = os.path.join(root, f)
            content = _read_file(fpath, max_chars=10000)
            if not content:
                continue
            rel_path = os.path.relpath(fpath, repo_path)
            for line_no, line in enumerate(content.splitlines(), 1):
                m = toml_param_re.search(line)
                if m:
                    _add_result(rel_path, line_no, int(m.group(2)), line, "toml_config")

    # ── Scan .cfg / .ini config files ────────────────────────────────────
    cfg_param_re = re.compile(
        rf"""({_PARAM_NAME_ALT})\s*[=:]\s*(\d+)""",
        re.IGNORECASE,
    )
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__" and d != "node_modules"]
        rel_root = os.path.relpath(root, repo_path)
        if rel_root.count(os.sep) > 3:
            continue
        for f in files:
            if not (f.endswith(".cfg") or f.endswith(".ini") or f.endswith(".conf")):
                continue
            fpath = os.path.join(root, f)
            content = _read_file(fpath, max_chars=10000)
            if not content:
                continue
            rel_path = os.path.relpath(fpath, repo_path)
            for line_no, line in enumerate(content.splitlines(), 1):
                if line.strip().startswith("#") or line.strip().startswith(";"):
                    continue
                m = cfg_param_re.search(line)
                if m:
                    _add_result(rel_path, line_no, int(m.group(2)), line, "cfg_config")

    # Sort by value descending so the most aggressive params appear first
    results.sort(key=lambda r: -r["value"])
    return results


# ── filtered requirements ────────────────────────────────────────────────────

def _produce_filtered_requirements(
    config_contents: Dict[str, str],
    preinstalled: List[str],
    rocm_mode: bool,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns:
        install_pkgs:  packages to install as-is
        skip_pkgs:     packages skipped (pre-installed or banned)
        flagged_pkgs:  packages that need special handling (CUDA mapping, risky pins)
    """
    install_pkgs = []
    skip_pkgs = []
    flagged_pkgs = []

    preinstalled_lower = {p.lower().replace("-", "_") for p in preinstalled}
    banned_lower = {p.lower().replace("-", "_") for p in BANNED_NVIDIA_PACKAGES} if rocm_mode else set()
    cuda_mapped_lower = {p.lower().replace("-", "_") for p in CUDA_TO_ROCM_MAPPING} if rocm_mode else set()

    for fname, content in config_contents.items():
        if not fname.startswith("requirements"):
            continue
        for pkg_name, ver_spec in _parse_requirements_lines(content):
            pkg_lower = pkg_name.lower().replace("-", "_")
            full_spec = f"{pkg_name}{ver_spec}" if ver_spec else pkg_name
            if pkg_lower in preinstalled_lower:
                skip_pkgs.append(f"{full_spec} (pre-installed)")
            elif pkg_lower in banned_lower:
                skip_pkgs.append(f"{full_spec} (BANNED nvidia package)")
            elif pkg_lower in cuda_mapped_lower:
                mapping = CUDA_TO_ROCM_MAPPING.get(pkg_name) or CUDA_TO_ROCM_MAPPING.get(
                    pkg_name.lower().replace("_", "-"))
                if mapping:
                    flagged_pkgs.append(f"{full_spec} -> ROCm: `{mapping['install_cmd']}`")
                else:
                    flagged_pkgs.append(f"{full_spec} (needs CUDA->ROCm mapping)")
            else:
                install_pkgs.append(full_spec)
    return install_pkgs, skip_pkgs, flagged_pkgs


# ── LLM-based image selection ─────────────────────────────────────────────

def _build_image_catalog_description() -> str:
    """Format the ROCM_IMAGE_CATALOG into a readable list for the LLM."""
    lines = []
    for workload, entry in ROCM_IMAGE_CATALOG.items():
        lines.append(
            f"- workload key: \"{workload}\" -> image: {entry['image']}:{entry['default_tag']}"
            f"\n  Description: {entry['description']}"
        )
    return "\n".join(lines)


def _build_import_summary(import_counts: Dict[str, int]) -> str:
    """Top imports sorted by frequency for LLM context."""
    sorted_imports = sorted(import_counts.items(), key=lambda x: -x[1])[:40]
    return "\n".join(f"  {pkg}: used in {count} files" for pkg, count in sorted_imports)


def _build_code_snippets_summary(py_file_contents: Dict[str, str], max_chars: int = 6000) -> str:
    """Extract import sections from source files to show the LLM what the code actually uses."""
    summaries = []
    total = 0
    for fpath, content in py_file_contents.items():
        import_lines = []
        for line in content.splitlines()[:60]:
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")) and not stripped.startswith("# "):
                import_lines.append(stripped)
        if import_lines:
            block = f"--- {fpath} ---\n" + "\n".join(import_lines)
            if total + len(block) > max_chars:
                break
            summaries.append(block)
            total += len(block)
    return "\n".join(summaries)


def _candidate_workloads(import_counts: Dict[str, int]) -> List[str]:
    """Pick 3-5 plausible workloads from the catalog based on imports.

    The synthesis prompt will see the full catalog, but the evidence-gathering
    phase only fetches `dockerhub_tags` for a small candidate set so we don't
    spam Docker Hub with one request per workload.
    """
    imp = {k.lower(): v for k, v in (import_counts or {}).items()}
    cands: List[str] = []
    if any(p in imp for p in ("sglang",)):
        cands.append("sglang")
    if any(p in imp for p in ("vllm",)):
        cands.append("vllm")
    if any(p in imp for p in ("deepspeed", "lightning", "pytorch_lightning",
                              "accelerate", "megatron", "megatron_core")):
        cands.append("pytorch-training")
    if any(p in imp for p in ("jax", "flax")):
        cands.append("jax")
    if any(p in imp for p in ("tensorflow", "keras")):
        cands.append("tensorflow")
    if "torch" in imp or "pytorch" in imp:
        if "pytorch" not in cands:
            cands.append("pytorch")
    if "pytorch" not in cands and not cands:
        cands.append("pytorch")
    seen: set = set()
    out: List[str] = []
    for c in cands:
        if c in ROCM_IMAGE_CATALOG and c not in seen:
            out.append(c)
            seen.add(c)
        if len(out) >= 5:
            break
    return out


def _gather_image_evidence(import_counts: Dict[str, int],
                            config_contents: Dict[str, str],
                            llm: Optional[str]) -> str:
    """Phase 1 of the researcher pattern: deterministic evidence gathering.

    Returns a compact text block listing, per candidate workload:
      - the live `dockerhub_tags` for its image repo
      - the live `pypi_versions` for the candidate's defining package
    Optionally appends a single deep_research note for niche frameworks.
    """
    lines: List[str] = ["## Live evidence (deterministic tools)"]
    workloads = _candidate_workloads(import_counts)
    if not workloads:
        return ""

    for wl in workloads:
        entry = ROCM_IMAGE_CATALOG.get(wl)
        if not entry:
            continue
        repo = entry["image"]
        lines.append(f"\n### Workload candidate: {wl}  (image repo: {repo})")
        try:
            from tools.external_lookups import dockerhub_tags
            body, rc = dockerhub_tags(repo, limit=6)
            if rc == 0 and body:
                for ln in body.splitlines()[:8]:
                    if ln.strip():
                        lines.append(f"  dockerhub_tags: {ln.strip()}")
            else:
                lines.append(f"  dockerhub_tags: lookup failed (rc={rc})")
        except Exception as e:
            lines.append(f"  dockerhub_tags: error {e}")

    primary_pkgs: List[str] = []
    imp = {k.lower(): v for k, v in (import_counts or {}).items()}
    for cand in ("torch", "jax", "tensorflow", "vllm", "sglang", "deepspeed",
                 "transformers"):
        if cand in imp:
            primary_pkgs.append(cand)
        if len(primary_pkgs) >= 3:
            break
    for pkg in primary_pkgs:
        try:
            from tools.external_lookups import pypi_versions
            body, rc = pypi_versions(pkg, limit=5)
            if rc == 0 and body:
                lines.append(f"\npypi_versions {pkg}:")
                for ln in body.splitlines()[:6]:
                    if ln.strip():
                        lines.append(f"  {ln.strip()}")
        except Exception:
            pass

    if llm and os.environ.get("AMD_LLM_API_KEY") and primary_pkgs:
        primary = primary_pkgs[0]
        try:
            from agents.researcher import research
            note = research(
                f"Best AMD ROCm Docker image for a repository whose primary "
                f"framework is `{primary}` in 2026. Mention concrete tags from "
                f"`rocm/{primary}` or `rocm/pytorch` and any known caveats.",
                llm=llm,
                budget_s=30.0,
                use_cache=True,
                profile="repoResearch",
                context={
                    "primary_framework": primary,
                    "candidate_workloads": workloads[:6],
                    "top_imports": sorted(import_counts.items(), key=lambda item: -item[1])[:12],
                    "config_excerpt": "\n".join(
                        f"# {name}\n{content[:1200]}"
                        for name, content in list((config_contents or {}).items())[:3]
                    ),
                },
            )
            ans = (note.get("answer") or "").strip()
            if ans:
                lines.append("\nResearcher note (one-shot):")
                lines.append(f"  {ans[:400]}")
        except Exception:
            pass

    bounded: List[str] = []
    used = 0
    for line in lines:
        if used + len(line) > 3500:
            break
        bounded.append(line)
        used += len(line) + 1
    return "\n".join(bounded)


def _llm_select_rocm_image(
    import_counts: Dict[str, int],
    config_contents: Dict[str, str],
    readme_content: Optional[str],
    py_file_contents: Dict[str, str],
    learned_context: str,
    llm: str,
) -> dict:
    """
    Two-phase researcher selection of the ROCm base image.

    Phase 1 (deterministic): `_gather_image_evidence` runs `dockerhub_tags` and
    `pypi_versions` for the most plausible candidate workloads/frameworks so
    the synthesiser never has to invent live registry state.

    Phase 2 (LLM synthesis): one round-trip that picks a workload from the
    catalog given the evidence + repo signals.

    Returns dict with: image, tag, workload, description, reasoning
    """
    catalog_desc = _build_image_catalog_description()
    import_summary = _build_import_summary(import_counts)
    code_summary = _build_code_snippets_summary(py_file_contents)
    evidence = _gather_image_evidence(import_counts, config_contents, llm)

    config_section = ""
    for fname, content in config_contents.items():
        config_section += f"\n--- {fname} ---\n{content[:2000]}\n"

    readme_snippet = ""
    if readme_content:
        readme_snippet = f"\n--- README (first 1500 chars) ---\n{readme_content[:1500]}\n"

    evidence_block = ""
    if evidence:
        evidence_block = (
            "\n## Live registry evidence (tool output, trust over training data)\n"
            f"{evidence}\n"
        )

    learned_block = ""
    if learned_context:
        learned_block = (
            "\n## Learned prior from previous runs\n"
            "Use this as a strong prior when it matches the current repo, but let "
            "live registry evidence and current repo files win if there is a conflict.\n"
            f"{learned_context}\n"
        )

    prompt = f"""\
You are an expert build engineer selecting the best ROCm Docker base image for a repository.
You receive (a) the catalog of available images, (b) the repo signals, and
(c) live registry evidence gathered by deterministic tools moments ago.
**Treat the live evidence as ground truth. Prefer it over any prior knowledge.**

## Available ROCm Docker Images

{catalog_desc}

## Repository Analysis

### Python imports (package: number of files importing it)
{import_summary}

### Dependency / config files
{config_section}
{readme_snippet}
### Import statements from source files
{code_summary}
{learned_block}
{evidence_block}
## Task

Analyze the repository's PRIMARY framework. Look at:
1. Which framework has the MOST imports across files (frequency matters most)
2. What the requirements.txt / setup.py actually lists as dependencies
3. What the README describes the project as
4. Whether a specialized image (sglang, vllm, megatron) is needed, or a general one
5. Which candidate's image actually has live tags on Docker Hub (live evidence above)

A repo might import multiple frameworks (e.g. both torch and jax) but typically one is the
PRIMARY framework used for the core logic, and others are secondary/utility imports.
Choose the image that matches the PRIMARY framework AND has live tags.

IMPORTANT rules:
- If the repo primarily uses PyTorch (torch) for training/inference without DeepSpeed/Megatron,
  select "pytorch"
- Only select "pytorch-training" if the repo uses distributed training libraries like
  DeepSpeed, Accelerate with FSDP, or PyTorch Lightning as core components
- Only select "jax" if JAX/Flax is the PRIMARY framework, not just a minor utility import
- Only select "vllm" or "sglang" if the repo actually uses those serving frameworks
- Only select "vllm-dev" if the repo IS a fork of vLLM itself

Respond with ONLY a JSON object (no markdown fences, no extra text):
{{"workload": "<key from the catalog>", "reasoning": "<one paragraph explaining why; cite the live evidence you used>"}}"""

    messages = [{"role": "user", "content": prompt}]
    try:
        response, usage = get_llm_response(llm, messages, temperature=0.1, max_tokens=512)
        if response and response[0]:
            log_info(f"LLM image selection: {usage.get('total_tokens', 0)} tokens used "
                     f"(evidence chars={len(evidence)})")
            return _parse_image_selection_response(response[0])
    except Exception as e:
        log_warning(f"LLM image selection failed ({e}), falling back to heuristic")

    return _fallback_image_selection(import_counts)


def _parse_image_selection_response(response_text: str) -> dict:
    """Parse the LLM JSON response into a structured image selection result."""
    try:
        result = load_json_loose(response_text, expected="object")
    except ValueError:
        log_warning("Could not parse LLM image selection response, falling back")
        return _fallback_image_selection({})

    workload = result.get("workload", "pytorch")
    reasoning = result.get("reasoning", "")

    if workload not in ROCM_IMAGE_CATALOG:
        log_warning(f"LLM selected unknown workload '{workload}', falling back to pytorch")
        workload = "pytorch"

    entry = ROCM_IMAGE_CATALOG[workload]
    return {
        "image": f"{entry['image']}:{entry['default_tag']}",
        "tag": entry["default_tag"],
        "workload": workload,
        "description": entry["description"],
        "reasoning": [reasoning] if reasoning else [],
    }


def _fallback_image_selection(import_counts: Dict[str, int]) -> dict:
    """Simple heuristic fallback when LLM is not available or fails."""
    imports_lower = {k.lower(): v for k, v in import_counts.items()}

    priority_checks = [
        (["sglang"], "sglang"),
        (["vllm"], "vllm"),
        (["megatron", "megatron_core"], "megatron"),
        (["deepspeed", "lightning", "pytorch_lightning"], "pytorch-training"),
        (["tensorflow", "keras"], "tensorflow"),
        (["onnxruntime", "onnx"], "onnxruntime"),
    ]

    for pkgs, workload in priority_checks:
        if any(p in imports_lower for p in pkgs):
            entry = ROCM_IMAGE_CATALOG[workload]
            return {
                "image": f"{entry['image']}:{entry['default_tag']}",
                "tag": entry["default_tag"],
                "workload": workload,
                "description": entry["description"],
                "reasoning": [f"Fallback heuristic: detected {workload}-related imports"],
            }

    jax_freq = imports_lower.get("jax", 0) + imports_lower.get("flax", 0)
    torch_freq = imports_lower.get("torch", 0)

    if jax_freq > 0 and torch_freq == 0:
        workload = "jax"
    elif jax_freq > torch_freq * 2 and torch_freq > 0:
        workload = "jax"
    else:
        workload = "pytorch"

    entry = ROCM_IMAGE_CATALOG[workload]
    return {
        "image": f"{entry['image']}:{entry['default_tag']}",
        "tag": entry["default_tag"],
        "workload": workload,
        "description": entry["description"],
        "reasoning": [f"Fallback heuristic: primary framework is {workload}"],
    }


# ── other detectors ──────────────────────────────────────────────────────────

def _recommend_base_image(framework: str) -> Tuple[str, str]:
    """Legacy fallback — used only when select_rocm_image() is not called."""
    if framework in ROCM_IMAGE_CATALOG:
        entry = ROCM_IMAGE_CATALOG[framework]
        return f"{entry['image']}:{entry['default_tag']}", entry["description"]
    entry = ROCM_IMAGE_CATALOG.get("pytorch", {})
    if entry:
        return f"{entry['image']}:{entry['default_tag']}", "Default: PyTorch ROCm image."
    return "rocm/pytorch:latest", "Fallback default."


def _find_entry_scripts(repo_path: str, readme_content: Optional[str]) -> List[str]:
    candidates = []
    entry_patterns = re.compile(
        r"(main|train|run|demo|example|infer|predict|test_run|serve|evaluate|eval|generate|sample)\w*\.py$",
        re.IGNORECASE
    )
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        rel_root = os.path.relpath(root, repo_path)
        if rel_root.count(os.sep) > 2:
            continue
        for f in files:
            if entry_patterns.search(f):
                candidates.append(os.path.relpath(os.path.join(root, f), repo_path))
    if readme_content:
        for m in re.finditer(r"python\s+([\w/\-]+\.py)", readme_content):
            script = m.group(1)
            if script not in candidates and os.path.isfile(os.path.join(repo_path, script)):
                candidates.append(script)
    return candidates[:10]


def _extract_readme_run_commands(readme_content: Optional[str], repo_path: str) -> List[Dict]:
    """
    Extract actual run/execution commands from the README, preserving full
    command lines with arguments (model names, flags, data paths, etc.).

    Returns a list of dicts: {"command": str, "context": str}
    where context is the surrounding text that explains what the command does.
    """
    if not readme_content:
        return []

    results = []
    lines = readme_content.splitlines()

    cmd_pattern = re.compile(
        r"(?:^|\s)((?:CUDA_VISIBLE_DEVICES=\S+\s+)?python[3]?\s+[\w/\-\.]+\.py(?:\s+[^\n`]*)?)",
        re.IGNORECASE,
    )

    for i, line in enumerate(lines):
        for m in cmd_pattern.finditer(line):
            cmd = m.group(1).strip()
            if len(cmd) < 10:
                continue

            context_start = max(0, i - 3)
            context_end = min(len(lines), i + 2)
            context_lines = lines[context_start:context_end]
            context = "\n".join(l for l in context_lines if l.strip())

            if cmd not in [r["command"] for r in results]:
                results.append({"command": cmd, "context": context})

    return results


def _extract_model_references(readme_content: Optional[str], repo_path: str) -> List[Dict]:
    """
    Extract HuggingFace model references from README and config files
    like model2path.json, identifying gated vs likely-ungated models.
    """
    refs = []
    seen = set()

    gated_prefixes = ["meta-llama/", "mistralai/", "google/gemma-7b", "google/gemma-2b"]
    ungated_prefixes = ["lmsys/", "THUDM/", "Salesforce/", "microsoft/", "TinyLlama/",
                        "EleutherAI/", "facebook/", "bigscience/bloomz", "google/gemma-2-"]

    hf_model_re = re.compile(r"['\"]([a-zA-Z0-9_\-]+/[a-zA-Z0-9_\.\-]+)['\"]")

    model2path_path = os.path.join(repo_path, "config", "model2path.json")
    if os.path.isfile(model2path_path):
        content = _read_file(model2path_path, max_chars=5000)
        if content:
            try:
                mapping = json.loads(content)
                for alias, hf_path in mapping.items():
                    if alias not in seen:
                        seen.add(alias)
                        is_gated = any(hf_path.startswith(p) for p in gated_prefixes)
                        is_ungated = any(hf_path.startswith(p) for p in ungated_prefixes)
                        refs.append({
                            "alias": alias,
                            "hf_path": hf_path,
                            "gated": is_gated,
                            "ungated": is_ungated,
                            "source": "config/model2path.json",
                        })
            except (json.JSONDecodeError, AttributeError):
                pass

    if not refs:
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
            if os.path.relpath(root, repo_path).count(os.sep) > 2:
                continue
            for f in files:
                if not (f.endswith(".json") or f.endswith(".py")):
                    continue
                fpath = os.path.join(root, f)
                content = _read_file(fpath, max_chars=8000)
                if not content:
                    continue
                for m in hf_model_re.finditer(content):
                    hf_path = m.group(1)
                    if "/" not in hf_path or hf_path.count("/") > 1:
                        continue
                    if hf_path in seen:
                        continue
                    seen.add(hf_path)
                    is_gated = any(hf_path.startswith(p) for p in gated_prefixes)
                    is_ungated = any(hf_path.startswith(p) for p in ungated_prefixes)
                    refs.append({
                        "alias": hf_path.split("/")[-1],
                        "hf_path": hf_path,
                        "gated": is_gated,
                        "ungated": is_ungated,
                        "source": os.path.relpath(fpath, repo_path),
                    })

    return refs


def _extract_external_assets(readme_content: Optional[str], repo_path: str) -> List[Dict]:
    """
    Scan the README and repo scripts to identify external data, model checkpoints,
    and dataset archives that must be downloaded before the main scripts can run.

    GitHub does not allow files > 25 MB, so any sizable dataset, pretrained
    checkpoint, or annotation archive will always be hosted externally:
    HuggingFace Hub, Google Drive, Baidu Yun, direct wget/curl URLs, etc.
    This function makes those requirements explicit in the plan so the executor
    does not waste turns waiting for files that were never present.

    Returns a list of dicts:
        {
            "kind":     "dataset" | "checkpoint" | "pseudo_mask" | "archive" | "unknown",
            "name":     human-readable label,
            "source":   "huggingface" | "google_drive" | "baidu" | "direct_url" | "script",
            "hf_id":    HF repo id when source=huggingface (e.g. "FudanCVL/Ref-Lerf"),
            "hf_type":  "dataset" | "model" | "space" | "" (the HF repo type),
            "url":      raw URL or Drive link when present,
            "script":   path to a download script in the repo (relative),
            "target_path": where the script expects the asset (e.g. "/data/ref-lerf"),
            "download_cmd": a concrete shell command to download it,
            "note":     any additional context,
        }
    """
    assets: List[Dict] = []
    seen_ids: set = set()

    # ── Pattern registry ────────────────────────────────────────────────────

    # HF dataset references: load_dataset('org/name'), snapshot_download(repo_id='org/name', repo_type='dataset'),
    # huggingface.co/datasets/org/name, hf.co/datasets/..., or bare 'org/name' next to 'dataset'
    hf_dataset_patterns = [
        re.compile(r"load_dataset\s*\(\s*['\"]([a-zA-Z0-9_\-\.]+/[a-zA-Z0-9_\-\.]+)['\"]"),
        re.compile(r"snapshot_download\s*\([^)]*repo_id\s*=\s*['\"]([a-zA-Z0-9_\-\.]+/[a-zA-Z0-9_\-\.]+)['\"][^)]*repo_type\s*=\s*['\"]dataset['\"]"),
        re.compile(r"snapshot_download\s*\([^)]*repo_type\s*=\s*['\"]dataset['\"][^)]*repo_id\s*=\s*['\"]([a-zA-Z0-9_\-\.]+/[a-zA-Z0-9_\-\.]+)['\"]"),
        re.compile(r"huggingface\.co/datasets/([a-zA-Z0-9_\-\.]+/[a-zA-Z0-9_\-\.]+)"),
        re.compile(r"hf\.co/datasets/([a-zA-Z0-9_\-\.]+/[a-zA-Z0-9_\-\.]+)"),
        re.compile(r"huggingface-cli\s+download\s+([a-zA-Z0-9_\-\.]+/[a-zA-Z0-9_\-\.]+)\s+--repo-type\s+dataset"),
    ]

    # HF model/checkpoint references: from_pretrained('org/model'), hub.download(repo_id='...'), etc.
    hf_model_patterns = [
        re.compile(r"from_pretrained\s*\(\s*['\"]([a-zA-Z0-9_\-\.]+/[a-zA-Z0-9_\-\.]+)['\"]"),
        re.compile(r"hf_hub_download\s*\([^)]*repo_id\s*=\s*['\"]([a-zA-Z0-9_\-\.]+/[a-zA-Z0-9_\-\.]+)['\"]"),
        re.compile(r"snapshot_download\s*\([^)]*repo_id\s*=\s*['\"]([a-zA-Z0-9_\-\.]+/[a-zA-Z0-9_\-\.]+)['\"]"),
        re.compile(r"huggingface\.co/([a-zA-Z0-9_\-\.]+/[a-zA-Z0-9_\-\.]+)(?!/tree|/blob|/commit)"),
        re.compile(r"huggingface-cli\s+download\s+([a-zA-Z0-9_\-\.]+/[a-zA-Z0-9_\-\.]+)(?!\s+--repo-type\s+dataset)"),
    ]

    # Google Drive
    gdrive_re = re.compile(r"drive\.google\.com/(?:drive/folders/|file/d/|open\?id=)([a-zA-Z0-9_\-]+)")

    # Baidu Yun / Baidu Pan
    baidu_re = re.compile(r"pan\.baidu\.com/s/([a-zA-Z0-9_\-]+)")

    # Direct wget / curl / aria2 download URLs (not pypi/dockerhub/github)
    wget_re = re.compile(
        r"(?:wget|curl\s+-[LOo]+|aria2c)\s+(?:-[^\s]+\s+)*"
        r"(https?://(?!pypi\.org|hub\.docker\.com|github\.com|raw\.githubusercontent\.com)[^\s'\"]+)"
    )

    # Checkpoint / pretrained weight references in README text
    ckpt_ref_re = re.compile(
        r"(?:pretrained|checkpoint|checkpoints?|weights?|model|download)[^.]*?['\"]?(/[^'\"<>\s]+\.(?:pth|pt|ckpt|bin|safetensors|pkl|npz))",
        re.IGNORECASE,
    )

    # Download scripts shipped with the repo
    download_script_re = re.compile(
        r"(?:bash|sh|python[3]?)\s+((?:\./)?(?:scripts/|tools/|data/)?download[^\s`'\"]*\.(?:sh|py))",
        re.IGNORECASE,
    )

    # Path references that suggest a data dir the script expects but that
    # doesn't exist inside the repo (likely must be downloaded)
    data_path_re = re.compile(
        r"(?:--source_path|--data(?:_path|_dir|set)?|--dataset(?:_path|_root)?|-s\s+)(?:\s+|=)"
        r"(/(?:data|datasets?|inputs?|checkpoints?)[^\s'\"<>]+)",
        re.IGNORECASE,
    )

    def _add(kind: str, name: str, source: str, **kwargs) -> None:
        uid = f"{source}:{name}"
        if uid in seen_ids:
            return
        seen_ids.add(uid)
        entry = {
            "kind": kind,
            "name": name,
            "source": source,
            "hf_id": kwargs.get("hf_id", ""),
            "hf_type": kwargs.get("hf_type", ""),
            "url": kwargs.get("url", ""),
            "script": kwargs.get("script", ""),
            "target_path": kwargs.get("target_path", ""),
            "download_cmd": kwargs.get("download_cmd", ""),
            "note": kwargs.get("note", ""),
        }
        assets.append(entry)

    # ── Scan README ──────────────────────────────────────────────────────────

    texts_to_scan: List[Tuple[str, str]] = []
    if readme_content:
        texts_to_scan.append(("README", readme_content))

    # Also scan top-level shell scripts and Makefiles for download commands
    for fname in ("Makefile", "makefile", "GNUmakefile", "setup.sh", "prepare_data.sh"):
        fc = _read_file(os.path.join(repo_path, fname), max_chars=8000)
        if fc:
            texts_to_scan.append((fname, fc))

    # And any scripts/ or data/ subdirectory with "download" in the filename
    for subdir in ("scripts", "tools", "data", "."):
        dpath = os.path.join(repo_path, subdir)
        if not os.path.isdir(dpath):
            continue
        for fn in os.listdir(dpath):
            if "download" in fn.lower() and fn.endswith((".sh", ".py", ".md")):
                fpath = os.path.join(dpath, fn)
                fc = _read_file(fpath, max_chars=6000)
                if fc:
                    rel = os.path.relpath(fpath, repo_path)
                    texts_to_scan.append((rel, fc))
                    # Register the script itself as a download asset
                    _add("archive", fn, "script",
                         script=rel,
                         download_cmd=f"bash /repo/{rel}",
                         note="Download script found in repo — run this first.")

    for text_label, text in texts_to_scan:
        # HF datasets
        for pat in hf_dataset_patterns:
            for m in pat.finditer(text):
                hf_id = m.group(1)
                _add("dataset", hf_id, "huggingface",
                     hf_id=hf_id, hf_type="dataset",
                     download_cmd=(
                         f"pip install -q huggingface_hub && "
                         f"python3 -c \"from huggingface_hub import snapshot_download; "
                         f"snapshot_download(repo_id='{hf_id}', repo_type='dataset', "
                         f"local_dir='/data/{hf_id.split('/')[-1].lower()}')\""
                     ),
                     note=f"Found in {text_label}")

        # HF models / checkpoints
        for pat in hf_model_patterns:
            for m in pat.finditer(text):
                hf_id = m.group(1)
                # Skip short strings, non-repos, and false positives like 'datasets/org'
                if len(hf_id) < 5 or "/" not in hf_id:
                    continue
                if hf_id.startswith("datasets/") or hf_id.startswith("spaces/"):
                    continue
                # Don't double-count something already found as dataset
                uid = f"huggingface:{hf_id}"
                if uid in seen_ids:
                    continue
                _add("checkpoint", hf_id, "huggingface",
                     hf_id=hf_id, hf_type="model",
                     download_cmd=(
                         f"pip install -q huggingface_hub && "
                         f"huggingface-cli download {hf_id} --local-dir /data/models/{hf_id.split('/')[-1].lower()}"
                     ),
                     note=f"Found in {text_label}")

        # Google Drive
        for m in gdrive_re.finditer(text):
            drive_id = m.group(1)
            _add("archive", f"gdrive:{drive_id[:16]}", "google_drive",
                 url=f"https://drive.google.com/uc?id={drive_id}",
                 download_cmd=(
                     f"pip install -q gdown && "
                     f"gdown https://drive.google.com/uc?id={drive_id} -O /data/gdrive_download"
                 ),
                 note=f"Google Drive link found in {text_label}. Check the README for the target directory.")

        # Baidu Yun
        for m in baidu_re.finditer(text):
            _add("archive", f"baidu:{m.group(1)[:16]}", "baidu",
                 url=f"https://pan.baidu.com/s/{m.group(1)}",
                 note=f"Baidu Yun link in {text_label}. Requires a Baidu account or extraction code (pwd). "
                      "Check the HuggingFace mirror first if one is mentioned alongside it.")

        # Direct wget/curl
        for m in wget_re.finditer(text):
            url = m.group(1)
            fname_guess = url.rstrip("/").split("/")[-1] or "download"
            _add("archive", fname_guess, "direct_url",
                 url=url,
                 download_cmd=f"wget -q -O /data/{fname_guess} '{url}'",
                 note=f"Direct download URL in {text_label}")

        # Download scripts invoked from README
        for m in download_script_re.finditer(text):
            script_path = m.group(1).lstrip("./")
            full = os.path.join(repo_path, script_path)
            if os.path.isfile(full):
                _add("archive", script_path, "script",
                     script=script_path,
                     download_cmd=f"bash /repo/{script_path}",
                     note=f"Download script invoked in {text_label}")

    # ── Scan Python source for from_pretrained / load_dataset calls ──────────
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "node_modules")]
        rel_root = os.path.relpath(root, repo_path)
        if rel_root.count(os.sep) > 2:
            continue
        for fn in files:
            if not fn.endswith(".py"):
                continue
            fpath = os.path.join(root, fn)
            content = _read_file(fpath, max_chars=8000)
            if not content:
                continue
            label = os.path.relpath(fpath, repo_path)
            for pat in hf_dataset_patterns:
                for m in pat.finditer(content):
                    hf_id = m.group(1)
                    _add("dataset", hf_id, "huggingface",
                         hf_id=hf_id, hf_type="dataset",
                         download_cmd=(
                             f"pip install -q huggingface_hub && "
                             f"python3 -c \"from huggingface_hub import snapshot_download; "
                             f"snapshot_download(repo_id='{hf_id}', repo_type='dataset', "
                             f"local_dir='/data/{hf_id.split('/')[-1].lower()}')\""
                         ),
                         note=f"load_dataset call in {label}")

    return assets


def _extract_readme_expected_outcomes(
    readme_content: Optional[str],
    readme_run_commands: List[Dict],
    llm: Optional[str] = None,
) -> List[Dict]:
    """
    Use the LLM to extract expected outcomes / success criteria from the README.

    Many READMEs document what correct output looks like — result tables, sample
    outputs, accuracy thresholds, pass/fail expectations.  The execution agent
    needs this information so it can validate its own output rather than
    declaring success just because a script didn't crash.

    Returns a list of dicts:
        {"command_or_script": str, "expected_outcome": str}
    Falls back to an empty list when no LLM is available or extraction fails.
    """
    if not readme_content or not llm:
        return []

    cmd_list = "\n".join(
        f"  - {rc['command']}" for rc in readme_run_commands
    ) if readme_run_commands else "(no specific commands extracted)"

    prompt = f"""\
You are analyzing a project README to extract **expected outcomes and success criteria**
for its run/test commands.

README CONTENT:
{readme_content}

KNOWN RUN COMMANDS:
{cmd_list}

TASK:
Extract every concrete, verifiable expected outcome from this README.  Look for:
1. Result tables showing which configurations or parameters produce which results.
2. Sample output blocks or expected console output.
3. Prose stating success criteria (e.g. "all tests should pass",
   "you should see accuracy above X").
4. Any documented pass/fail behavior for specific configurations or parameter sets.

For each outcome, state:
- **command_or_script**: the command, script name, or test name it applies to
  (use the closest match from the known run commands above, or the script filename).
- **expected_outcome**: a concise, specific description of what correct output
  looks like.  Include exact values, thresholds, or labels directly from the README.

Respond with ONLY a JSON array (no markdown fences, no extra text).  Example:
[
  {{"command_or_script": "python run_tests.py", "expected_outcome": "All tests pass with exit code 0"}},
  {{"command_or_script": "python benchmark.py", "expected_outcome": "Throughput > 100 samples/sec on GPU"}}
]

If the README contains NO verifiable expected outcomes, return an empty array: []"""

    messages = [{"role": "user", "content": prompt}]
    try:
        response, usage = get_llm_response(llm, messages, temperature=0.1, max_tokens=2048)
        if response and response[0]:
            log_info(f"Expected-outcome extraction: {usage.get('total_tokens', 0)} tokens used")
            text = response[0].strip()
            if text.startswith("```"):
                text = re.sub(r"^```\w*\n?", "", text)
                text = re.sub(r"\n?```$", "", text)
            parsed = json.loads(text)
            if isinstance(parsed, list):
                valid = [
                    e for e in parsed
                    if isinstance(e, dict)
                    and "command_or_script" in e
                    and "expected_outcome" in e
                ]
                if valid:
                    log_info(f"  Extracted {len(valid)} expected outcomes from README")
                return valid
    except (json.JSONDecodeError, Exception) as e:
        log_info(f"Expected-outcome extraction failed ({e}), skipping")

    return []


def _detect_python_version(config_contents: Dict[str, str]) -> Optional[str]:
    if ".python-version" in config_contents:
        ver = config_contents[".python-version"].strip().split("\n")[0].strip()
        if ver:
            return ver
    for fname in ("pyproject.toml", "setup.cfg", "setup.py"):
        content = config_contents.get(fname, "")
        for pattern in [r"python_requires\s*[=:]\s*['\"]([^'\"]+)['\"]",
                        r"requires-python\s*=\s*['\"]([^'\"]+)['\"]"]:
            m = re.search(pattern, content)
            if m:
                return m.group(1)
    return None


def _build_rocm_migration_section(cuda_deps: List[str]) -> str:
    if not cuda_deps:
        return "No CUDA/NVIDIA-specific dependencies detected."
    lines = ["CUDA-to-ROCm Migration Steps:"]
    for dep in cuda_deps:
        dep_lower = dep.lower().replace("_", "-")
        mapping = CUDA_TO_ROCM_MAPPING.get(dep) or CUDA_TO_ROCM_MAPPING.get(dep_lower)
        if mapping:
            lines.append(f"  - {dep}:")
            lines.append(f"      ROCm replacement: {mapping['rocm_package']}")
            lines.append(f"      Install: {mapping['install_cmd']}")
            if mapping.get("notes"):
                lines.append(f"      Notes: {mapping['notes']}")
        elif dep in BANNED_NVIDIA_PACKAGES:
            lines.append(f"  - {dep}: BANNED on ROCm. Skip.")
        else:
            lines.append(f"  - {dep}: Likely CUDA-specific. Investigate.")
    return "\n".join(lines)


# ── planner-side external notes (PR-A/B/C) ──────────────────────────────────

def _planner_external_notes(cuda_deps: List[str],
                            base_image_name: str,
                            llm: Optional[str] = None) -> List[str]:
    """
    Lightweight planner-side external research.

    The planner itself is a one-shot LLM call, not a multi-turn tool-calling
    agent. To still expose the same knowledge surface in planning, we
    pre-compute a compact note from:
      - deterministic lookups: `pypi_versions`, `dockerhub_tags`
      - optional deep_research for the riskiest CUDA-ish deps (cached)
    """
    notes: List[str] = []

    if base_image_name:
        try:
            from tools.external_lookups import dockerhub_tags
            image_repo = base_image_name.split(":")[0]
            body, rc = dockerhub_tags(image_repo, limit=4)
            if rc == 0 and body:
                lines = [ln for ln in body.splitlines()[:6] if ln.strip()]
                notes.append("Docker Hub tags for chosen base image:")
                notes.extend(lines)
        except Exception:
            pass

    for dep in cuda_deps[:2]:
        dep_norm = dep.replace("-", "_").lower()
        try:
            from tools.external_lookups import pypi_versions
            body, rc = pypi_versions(dep_norm, limit=4)
            if rc == 0 and body:
                lines = [ln for ln in body.splitlines()[:6] if ln.strip()]
                notes.append(f"PyPI versions for {dep_norm}:")
                notes.extend(lines)
        except Exception:
            pass

        if (
            llm
            and dep_norm in {"flash_attn", "bitsandbytes", "xformers", "triton"}
            and os.environ.get("AMD_LLM_API_KEY")
        ):
            try:
                from agents.researcher import research
                note = research(
                    f"What is the safest AMD ROCm installation or fallback strategy for "
                    f"`{dep_norm}` in a PyTorch repository on ROCm 7.x? Include exact "
                    f"commands if known.",
                    llm=llm,
                    budget_s=45.0,
                    use_cache=True,
                    profile="repoResearch",
                    context={
                        "dep": dep_norm,
                        "base_image": base_image_name,
                        "cuda_deps": cuda_deps[:6],
                    },
                )
                ans = (note.get("answer") or "").strip()
                cmds = note.get("suggested_commands") or []
                if ans:
                    notes.append(f"Deep research note for {dep_norm}: {ans[:320]}")
                for c in cmds[:3]:
                    notes.append(f"Suggested command: {str(c)[:220]}")
            except Exception:
                pass

    bounded: List[str] = []
    used = 0
    for line in notes:
        if used + len(line) > 2500:
            break
        bounded.append(line)
        used += len(line) + 1
    return bounded


# ── AMD ecosystem recommender ─────────────────────────────────────────────────

def _build_amd_ecosystem_section(
    import_counts: Dict[str, int],
    config_contents: Dict[str, str],
    rocm_mode: bool,
) -> List[str]:
    """
    Match detected Python imports and config-file strings against the AMD ROCm
    repo catalog and produce a concise, actionable section for the plan.

    Only returns content when rocm_mode is True and at least one relevant AMD
    library is identified.
    """
    if not rocm_mode:
        return []

    # Collect all import names and config strings for matching
    import_names = list(import_counts.keys())
    config_strings: List[str] = []
    for content in config_contents.values():
        config_strings.extend(content.split())

    relevant = get_relevant_amd_repos(import_names, config_strings)
    if not relevant:
        return []

    lines: List[str] = []
    lines.append("AMD ROCm Ecosystem — Project-Specific Recommendations:")
    lines.append(
        "  The planner detected the following CUDA/NVIDIA imports and matched them"
        " to AMD-native alternatives. Use these instead of the NVIDIA equivalents."
    )
    lines.append(
        "  Full reference: /Repo2ROCm/build_agent/knowledge/amd_rocm_ecosystem.md"
    )
    lines.append("")

    # Group by category for readability
    by_category: Dict[str, List] = {}
    for entry in relevant:
        cat = entry.get("category", "other")
        by_category.setdefault(cat, []).append(entry)

    category_order = ["rendering", "ml", "dl", "inference", "vision", "math", "tooling", "other"]
    for cat in category_order:
        if cat not in by_category:
            continue
        cat_label = {
            "rendering": "3D Rendering / Gaussian Splatting",
            "ml": "ML Acceleration Libraries",
            "dl": "Deep Learning Primitives",
            "inference": "Inference / Graph Compilers",
            "vision": "Computer Vision",
            "math": "Math Libraries",
            "tooling": "Tooling",
            "other": "Other",
        }.get(cat, cat.title())
        lines.append(f"  [{cat_label}]")
        for entry in by_category[cat]:
            lines.append(f"  • {entry['amd_name']}  (replaces NVIDIA: {entry['nvidia_equiv']})")
            lines.append(f"    Use case: {entry['use_case'][:200]}")
            lines.append(f"    Status:   {entry['status']}")
            if entry.get("install_cmd"):
                # Compact: show just the first non-comment line of install_cmd
                first_cmd = next(
                    (l.strip() for l in entry["install_cmd"].splitlines() if l.strip() and not l.strip().startswith("#")),
                    ""
                )
                if first_cmd:
                    lines.append(f"    Install:  {first_cmd}")
            if entry.get("notes"):
                # Show up to 200 chars of notes
                lines.append(f"    Notes:    {entry['notes'][:200]}")
            lines.append(f"    GitHub:   {entry['github']}")
            lines.append("")

    lines.append(
        "  IMPORTANT: If this is a Gaussian Splatting repo (detects diff_gaussian_rasterization"
        " or simple_knn), use amd_gsplat INSTEAD of trying to compile those submodules from"
        " source — they use CUDA-only headers that are very hard to patch correctly."
    )
    lines.append("")
    return lines


# ── main planner ─────────────────────────────────────────────────────────────

def generate_plan(repo_path: str, full_name: str, rocm_mode: bool = False,
                  llm: Optional[str] = None,
                  no_scale_down: bool = False,
                  paper_pdf_path: Optional[str] = None,
                  paper_corpus: Optional[Any] = None,
                  reproduce_results: bool = False,
                  run_memory: Optional[Any] = None,
                  graphify_provider: Optional[Any] = None,
                  learned_context: str = "",
                  run_mode: str = "env") -> Tuple[str, Optional[str], Dict[str, Any]]:
    """
    Deep-analyze the repository and produce a comprehensive strategic plan.

    Args:
        no_scale_down: If True, skip training parameter detection and scale-down
            sed commands. The agent will run the README commands exactly as-is
            with real data instead of mock data.
        paper_pdf_path: Local path to the paper PDF, when --reproduce-results is on.
        reproduce_results: When True, shortlist paper experiments and splice a
            PAPER REPRODUCTION TARGET section into the plan.

    Returns:
        (plan_text, recommended_image, paper_context) — recommended_image is
        the Docker image string (or None in non-ROCm mode). `paper_context`
        is a dict with keys `experiments` (list of ExperimentCandidate dicts,
        empty when reproduce_results is False) and `title` (paper title or "").
    """
    log_phase("RECONNAISSANCE & PLANNING", f"Analyzing {full_name}")

    # 1. Read config files
    config_contents: Dict[str, str] = {}
    for fname in _CONFIG_FILES:
        content = _read_file(os.path.join(repo_path, fname))
        if content is not None:
            config_contents[fname] = content
            log_info(f"  Found config: {fname}")

    workflows_dir = os.path.join(repo_path, ".github", "workflows")
    if os.path.isdir(workflows_dir):
        for wf in os.listdir(workflows_dir)[:3]:
            wf_content = _read_file(os.path.join(workflows_dir, wf), max_chars=3000)
            if wf_content:
                config_contents[f".github/workflows/{wf}"] = wf_content

    # 2. Read README
    readme_content = None
    for rname in _README_NAMES:
        content = _read_file(os.path.join(repo_path, rname), max_chars=None)
        if content:
            readme_content = content
            log_info(f"  Found README: {rname}")
            break

    repo_readme_context = ""
    repo_config_context = ""
    repo_code_context = ""
    if graphify_provider is not None and getattr(graphify_provider, "enabled", False):
        try:
            repo_readme_context = graphify_provider.query_repo_corpus(
                "README installation usage commands expected outputs models datasets benchmarks",
                scope="readme",
                token_budget=12000,
                max_chunks=10,
                per_chunk_max_chars=2000,
            ) or ""
            repo_config_context = graphify_provider.query_repo_corpus(
                "config training hyperparameters batch size learning rate optimizer scheduler dataset path yaml toml",
                scope="config",
                token_budget=12000,
                max_chunks=10,
                per_chunk_max_chars=2000,
            ) or ""
            repo_code_context = graphify_provider.query_repo_corpus(
                "entrypoint training evaluation metric logging argparse hydra main function model loading config parser",
                scope="code",
                token_budget=12000,
                max_chunks=10,
                per_chunk_max_chars=2000,
            ) or ""
        except Exception as _repo_ctx_e:
            log_warning(f"  Repo corpus query failed: {_repo_ctx_e}")

    # 3. Scan source files
    py_files = _find_python_files(repo_path)
    import_counts = _extract_imports(py_files)
    top_imports = sorted(import_counts.items(), key=lambda x: -x[1])[:30]
    log_info(f"  Scanned {len(py_files)} Python files, {len(import_counts)} unique imports")

    # 4. Detect framework
    framework = _detect_framework(import_counts)
    log_info(f"  Detected framework: {framework}")

    # 5. Detect Python version
    python_version = _detect_python_version(config_contents)

    # 6. CUDA deps
    cuda_deps = _detect_cuda_deps(import_counts, config_contents)

    # 7. Entry scripts
    entry_scripts = _find_entry_scripts(repo_path, readme_content)

    # 7b. Extract run commands and model references from README
    readme_run_commands = _extract_readme_run_commands(readme_content, repo_path)
    if readme_run_commands:
        log_info(f"  Extracted {len(readme_run_commands)} run commands from README")

    model_references = _extract_model_references(readme_content, repo_path)
    if model_references:
        gated_count = sum(1 for r in model_references if r["gated"])
        ungated_count = sum(1 for r in model_references if r["ungated"])
        log_info(f"  Found {len(model_references)} model references ({ungated_count} ungated, {gated_count} gated)")

    # 7b-2. Detect external datasets, checkpoints, archives that must be downloaded.
    # GitHub enforces a 25 MB per-file limit so large data is ALWAYS hosted externally.
    external_assets = _extract_external_assets(readme_content, repo_path)
    if external_assets:
        hf_assets = [a for a in external_assets if a["source"] == "huggingface"]
        script_assets = [a for a in external_assets if a["source"] == "script"]
        drive_assets = [a for a in external_assets if a["source"] == "google_drive"]
        log_info(
            f"  External assets: {len(external_assets)} total "
            f"({len(hf_assets)} HuggingFace, {len(script_assets)} scripts, "
            f"{len(drive_assets)} Google Drive)"
        )

    # 7c. Extract expected outcomes from README
    expected_outcomes: List[Dict] = []
    if llm:
        expected_outcomes = _extract_readme_expected_outcomes(
            readme_content, readme_run_commands, llm,
        )

    # 7d. Shortlist paper experiments for reproduction (if requested)
    paper_experiments: List = []
    paper_title: str = ""
    if reproduce_results:
        try:
            from agents.paper_agent import PaperAgent
            paper_agent = PaperAgent(llm=llm or "")
            paper_experiments, paper_title = paper_agent.shortlist_experiments(
                paper_pdf_path=paper_pdf_path,
                repo_path=repo_path,
                paper_corpus=paper_corpus,
                readme_content=readme_content or "",
                readme_run_commands=readme_run_commands,
                readme_expected_outcomes=expected_outcomes,
                llm=llm,
                run_memory=run_memory,
                graphify_provider=graphify_provider,
            )
            if paper_experiments:
                log_info(
                    f"  Paper shortlist: {len(paper_experiments)} experiments "
                    f"(chosen: {paper_experiments[0].name[:80]!r}, "
                    f"code_available={paper_experiments[0].code_available})"
                )
            else:
                log_warning("  Paper shortlist produced no experiments; reproduction section will be minimal.")
        except Exception as e:
            log_warning(f"  Paper shortlist failed: {e}")
            paper_experiments = []
            paper_title = ""

    # 8. Install mechanisms
    install_mechanisms = []
    if "poetry.lock" in config_contents or ("pyproject.toml" in config_contents and "poetry" in config_contents.get("pyproject.toml", "").lower()):
        install_mechanisms.append("poetry install")
    if "setup.py" in config_contents:
        install_mechanisms.append("pip install -e .")
    if "setup.cfg" in config_contents and "pyproject.toml" in config_contents:
        install_mechanisms.append("pip install -e .")
    if "Pipfile" in config_contents:
        install_mechanisms.append("pipenv install")
    if "environment.yml" in config_contents or "environment.yaml" in config_contents:
        install_mechanisms.append("conda env create -f environment.yml")
    req_files = [f for f in config_contents if f.startswith("requirements")]
    for rf in req_files:
        install_mechanisms.append(f"pip install -r {rf}")
    if not install_mechanisms:
        install_mechanisms.append("pipreqs + manual install")

    # 9. Directory listing
    try:
        top_level = sorted(os.listdir(repo_path))[:40]
    except Exception:
        top_level = []

    # ── DEEP ANALYSIS ────────────────────────────────────────────────────────

    # Python 3.12 compatibility
    py312_issues = _detect_py312_compat_issues(py_files, repo_path)
    if py312_issues:
        log_info(f"  Found {len(py312_issues)} Python 3.12 compatibility issues")

    # Version pin hazards
    pin_hazards = _detect_version_pin_hazards(config_contents)
    if pin_hazards:
        log_info(f"  Found {len(pin_hazards)} risky version pins")

    # Code hazards
    code_hazards = _detect_code_hazards(py_files, repo_path) if rocm_mode else []
    if code_hazards:
        log_info(f"  Found {len(code_hazards)} code-level hazards")

    # Training params (skipped in --no-scale-down mode)
    if no_scale_down:
        training_params = []
        log_info("  Skipping training parameter detection (--no-scale-down)")
    else:
        training_params = _detect_training_params(py_files, repo_path)
        if training_params:
            log_info(f"  Found {len(training_params)} large training parameter values")

    # ── Context-aware image selection (LLM-based) ──────────────────────────
    py_file_contents: Dict[str, str] = {}
    for fpath in py_files[:60]:
        content = _read_file(fpath, max_chars=5000)
        if content:
            py_file_contents[os.path.relpath(fpath, repo_path)] = content

    preinstalled = []
    base_image_name = ""
    image_selection = {}
    if rocm_mode:
        if llm:
            image_selection = _llm_select_rocm_image(
                import_counts=import_counts,
                config_contents=config_contents,
                readme_content=readme_content,
                py_file_contents=py_file_contents,
                learned_context=learned_context,
                llm=llm,
            )
        else:
            image_selection = _fallback_image_selection(import_counts)

        base_image_name = image_selection["image"]
        preinstalled = ROCM_PREINSTALLED_PACKAGES.get(
            base_image_name.split(":")[0], [])
        log_info(f"  Image selection: {base_image_name} "
                 f"(workload={image_selection['workload']})")
        for reason in image_selection.get("reasoning", []):
            log_info(f"    - {reason}")

    install_pkgs, skip_pkgs, flagged_pkgs = _produce_filtered_requirements(
        config_contents, preinstalled, rocm_mode)

    # ── Assemble raw plan ────────────────────────────────────────────────────

    sections = []

    _MODE_BANNERS = {
        "env": (
            "RUN MODE: 1 — ROCm Env Only\n"
            "  Goal: install dependencies, migrate CUDA→ROCm, run a quick smoke test,\n"
            "  then echo ROCM_ENV_VERIFIED. Training params may be scaled down.\n"
            "  Paper reproduction is NOT required in this mode."
        ),
        "reproduce": (
            "RUN MODE: 2 — Paper Reproduce\n"
            "  Goal: download required datasets/checkpoints, run the paper experiment\n"
            "  with the EXACT config from the paper/README (no scale-down), and\n"
            "  compare results against the paper's reported metrics.\n"
            "  env setup is a prerequisite but PAPER_RESULT_REPRODUCED/NOT_REPRODUCED\n"
            "  is the primary success criterion. Do NOT synthesise fake data."
        ),
        "full": (
            "RUN MODE: 3 — Full (Env + Paper Reproduce)\n"
            "  Stage 1 — Echo ROCM_ENV_VERIFIED once the repo runs on the AMD GPU.\n"
            "  Stage 2 — Download required assets, run the paper experiment with the\n"
            "  EXACT paper config (no scale-down), compare against reported metrics,\n"
            "  then echo PAPER_RESULT_REPRODUCED or PAPER_RESULT_NOT_REPRODUCED."
        ),
    }
    banner = _MODE_BANNERS.get(run_mode, _MODE_BANNERS["env"])
    sections.append(banner)
    sections.append("")
    sections.append(f"Repository: {full_name}")
    sections.append(f"Top-level contents: {', '.join(top_level)}")
    sections.append("")

    if python_version:
        sections.append(f"Python Version Required: {python_version}")
    else:
        sections.append("Python Version: Not specified (use container default)")
    sections.append("")

    sections.append("Detected Config Files:")
    for fname in config_contents:
        sections.append(f"  - {fname}")
    sections.append("")

    sections.append("Install Strategy (in order):")
    for i, mech in enumerate(install_mechanisms, 1):
        sections.append(f"  {i}. {mech}")
    sections.append("")

    sections.append(f"Framework: {framework}")
    sections.append(f"Top Imports: {', '.join(pkg for pkg, _ in top_imports[:15])}")
    sections.append("")

    if learned_context:
        sections.append("Learned Prior For Planning:")
        sections.append(learned_context.strip())
        sections.append("")

    # ── External lookup notes (PR-A/B/C, planner-side) ──────────────────────
    # The planner is not a multi-turn tool-calling agent, but we can still give
    # it access to the same knowledge surface by precomputing a compact note.
    ext_notes = _planner_external_notes(
        cuda_deps=sorted(cuda_deps),
        base_image_name=base_image_name if rocm_mode else "",
        llm=llm,
    )
    if ext_notes:
        sections.append("External Lookup Notes (cached, planner-side):")
        sections.extend(f"  {line}" for line in ext_notes)
        sections.append("")

    # ── AMD ecosystem recommendations (project-specific) ────────────────────

    amd_ecosystem_lines = _build_amd_ecosystem_section(
        import_counts=import_counts,
        config_contents=config_contents,
        rocm_mode=rocm_mode,
    )
    if amd_ecosystem_lines:
        sections.extend(amd_ecosystem_lines)
        log_info(f"  AMD ecosystem: {len(amd_ecosystem_lines)} recommendation lines added to plan")

    # ── Filtered requirements ────────────────────────────────────────────────

    if install_pkgs or skip_pkgs or flagged_pkgs:
        sections.append("Filtered Dependency Analysis:")
        if install_pkgs:
            sections.append(f"  INSTALL ({len(install_pkgs)} packages):")
            for p in install_pkgs:
                sections.append(f"    pip install {p}")
        if skip_pkgs:
            sections.append(f"  SKIP ({len(skip_pkgs)} packages):")
            for p in skip_pkgs:
                sections.append(f"    {p}")
        if flagged_pkgs:
            sections.append(f"  SPECIAL HANDLING ({len(flagged_pkgs)} packages):")
            for p in flagged_pkgs:
                sections.append(f"    {p}")
        sections.append("")

    # ── Python 3.12 issues ───────────────────────────────────────────────────

    if py312_issues:
        sections.append(f"CRITICAL - Python 3.12 Compatibility Issues ({len(py312_issues)}):")
        sections.append("  The container uses Python 3.12. These MUST be fixed BEFORE running scripts:")
        for issue in py312_issues:
            sections.append(f"  - {issue['file']}:{issue['line']} — `import {issue['module']}`")
            sections.append(f"    Fix: {issue['fix']}")
            sections.append(f"    Command: {issue['sed']}")
        sections.append("")

    # ── Version pin hazards ──────────────────────────────────────────────────

    if pin_hazards:
        sections.append(f"WARNING - Risky Version Pins ({len(pin_hazards)}):")
        sections.append("  These old pins will likely fail to install on Python 3.12 (no prebuilt wheels):")
        for h in pin_hazards:
            sections.append(f"  - {h['package']}=={h['pinned']} in {h['file']}")
            sections.append(f"    {h['fix']}")
        sections.append("")

    # ── Code hazards ─────────────────────────────────────────────────────────

    if code_hazards:
        seen_kinds: Set[str] = set()
        sections.append(f"Code-Level Hazards ({len(code_hazards)} occurrences):")
        for h in code_hazards:
            if h["kind"] not in seen_kinds:
                seen_kinds.add(h["kind"])
                sections.append(f"  [{h['kind']}] {h['description']}")
                sections.append(f"    Example: {h['file']}:{h['line']} — {h['code']}")
        sections.append("")

    # ── Training params to scale down ────────────────────────────────────────

    if training_params:
        sections.append(f"Training Parameters to Scale Down ({len(training_params)} found, showing top 20):")
        sections.append("  These values MUST be reduced before running scripts to avoid timeouts:")
        for tp in training_params[:20]:
            source_label = tp.get("source", "python")
            sections.append(f"  - {tp['file']}:{tp['line']} — value={tp['value']} ({source_label})")
            sections.append(f"    Code: {tp['code']}")
            sections.append(f"    Fix:  {tp['sed']}")
        if len(training_params) > 20:
            sections.append(f"  ... and {len(training_params) - 20} more (batch-fix with:")
            sections.append("    find /repo -name '*.py' -exec grep -l 'epochs' {} \\; | xargs -I{} sed -i \"s/'epochs': [0-9]*/'epochs': 2/g\" {}")
            sections.append("  )")
        sections.append("")

    # ── ROCm specifics ───────────────────────────────────────────────────────

    if rocm_mode:
        sections.append(f"ROCm Base Image: {base_image_name}")
        if image_selection:
            sections.append(f"  Workload type: {image_selection['workload']}")
            sections.append(f"  Description: {image_selection['description']}")
            if image_selection.get("reasoning"):
                sections.append("  Selection reasoning:")
                for reason in image_selection["reasoning"]:
                    sections.append(f"    - {reason}")
        if preinstalled:
            sections.append(f"  Pre-installed (DO NOT reinstall): {', '.join(preinstalled)}")
        sections.append("")
        sections.append(_build_rocm_migration_section(cuda_deps))
        sections.append("")

    # ── Target scripts ───────────────────────────────────────────────────────

    sections.append("Target Scripts for Verification:")
    if entry_scripts:
        for s in entry_scripts:
            sections.append(f"  - {s}")
    else:
        sections.append("  (none detected — inspect README for usage examples)")
    sections.append("")

    # ── README run commands (CRITICAL for correct verification) ───────────────

    if readme_run_commands:
        sections.append("CRITICAL - Exact Run Commands from README (USE THESE for verification):")
        sections.append("  The README specifies these exact commands. Use them as-is (adapting for ROCm if needed):")
        for rc in readme_run_commands:
            sections.append(f"  Command: {rc['command']}")
            if rc["context"]:
                for ctx_line in rc["context"].splitlines()[:3]:
                    if ctx_line.strip() and not ctx_line.strip().startswith("```"):
                        sections.append(f"    Context: {ctx_line.strip()}")
        sections.append("")

    # ── External assets (datasets, checkpoints, archives) ────────────────────

    if external_assets:
        sections.append(
            "EXTERNAL ASSETS REQUIRED (DOWNLOAD BEFORE RUNNING SCRIPTS):"
        )
        sections.append(
            "  CRITICAL: GitHub does not permit files > 25 MB in a repository."
            " Datasets, pretrained checkpoints, pseudo-mask archives, and"
            " annotation packages are ALWAYS hosted externally (HuggingFace,"
            " Google Drive, direct URLs, etc.). These files will NOT be present"
            " inside /repo. The agent MUST download them before running any"
            " training / inference / evaluation script that references them."
        )
        sections.append("")
        for a in external_assets:
            sections.append(f"  [{a['kind'].upper()}] {a['name']}")
            sections.append(f"    Source:  {a['source']}" + (f" ({a['hf_type']})" if a["hf_type"] else ""))
            if a["hf_id"]:
                sections.append(f"    HF id:   {a['hf_id']}")
            if a["url"]:
                sections.append(f"    URL:     {a['url']}")
            if a["script"]:
                sections.append(f"    Script:  /repo/{a['script']}")
            if a["download_cmd"]:
                sections.append(f"    Command: {a['download_cmd']}")
            if a["target_path"]:
                sections.append(f"    Target:  {a['target_path']}")
            if a["note"]:
                sections.append(f"    Note:    {a['note']}")
            sections.append("")
        sections.append(
            "  STRATEGY: Check each asset with `ls <target_path>` BEFORE"
            " running the script. If it is missing, run the Command above."
            " For HuggingFace assets, also try `huggingface-cli download`."
            " For Google Drive, use `gdown`. For Baidu Yun links,"
            " prefer the HuggingFace mirror when one is mentioned alongside it."
        )
        sections.append(
            "  If the download source is unclear, use `web_search` to find the"
            " canonical download path — e.g."
            " 'heshuting555/ReferSplat Ref-LERF dataset download HuggingFace'."
        )
        sections.append("")

    # ── Model references and gating status ────────────────────────────────────

    if model_references:
        sections.append("HuggingFace Model References (gated vs ungated):")
        has_gated = False
        for ref in model_references:
            status = "GATED (will fail without auth)" if ref["gated"] else "ungated" if ref["ungated"] else "unknown"
            sections.append(f"  - {ref['alias']} -> {ref['hf_path']} [{status}] (from {ref['source']})")
            if ref["gated"]:
                has_gated = True
        if has_gated:
            ungated_models = [r for r in model_references if r["ungated"]]
            if ungated_models:
                sections.append(f"  WARNING: Some models are gated. Prefer ungated models for verification:")
                for u in ungated_models[:5]:
                    sections.append(f"    USE: --model {u['alias']}  (maps to {u['hf_path']}, ungated)")
            else:
                sections.append(f"  WARNING: All models are gated. Substitute with ungated alternatives")
                sections.append(f"    (e.g., TinyLlama/TinyLlama-1.1B-Chat-v1.0 for Llama-based models)")
        sections.append("")

    # ── Expected outcomes (for output validation in --no-scale-down mode) ───

    if expected_outcomes:
        sections.append("EXPECTED OUTCOMES FROM README (VALIDATE YOUR OUTPUT AGAINST THESE):")
        sections.append("  After running each script, check that your output matches these expected results.")
        sections.append("  Do NOT declare ROCM_ENV_VERIFIED if output contradicts these expectations.")
        sections.append("")
        for eo in expected_outcomes:
            sections.append(f"  Script/Command: {eo['command_or_script']}")
            for outcome_line in eo["expected_outcome"].splitlines():
                sections.append(f"    Expected: {outcome_line}")
            sections.append("")

    # ── Paper reproduction target (for --reproduce-results) ──────────────────

    if reproduce_results:
        sections.append("PAPER REPRODUCTION TARGET (from paper.pdf at /repo/paper.pdf):")
        if paper_title:
            sections.append(f"  Paper: {paper_title}")
        if not paper_experiments:
            sections.append("  (No experiments could be shortlisted from the paper automatically.")
            sections.append("   The paper-reproducer sub-agent must open /repo/paper.pdf with the Read tool,")
            sections.append("   pick the shortest runnable experiment whose code exists in the repo,")
            sections.append("   run it with the EXACT paper/README config, and judge the result.)")
        else:
            chosen = paper_experiments[0]
            runtime_str = (
                f"{chosen.est_runtime_minutes:.0f} min"
                if chosen.est_runtime_minutes > 0 else "unknown"
            )
            sections.append(f"  Chosen experiment: {chosen.name}")
            if chosen.section:
                sections.append(f"    Source: {chosen.section}")
            sections.append(
                f"    Reason: shortest runtime with code {'AVAILABLE' if chosen.code_available else 'NOT MATCHED'}"
                f" in repo (~{runtime_str} est.)"
            )
            metric_line = (
                f"{chosen.expected_metric_name}={chosen.expected_metric_value} {chosen.expected_metric_units}".strip()
                if chosen.expected_metric_name else "(none parsed)"
            )
            sections.append(f"    Paper-reported metric: {metric_line}")
            if chosen.hardware:
                sections.append(f"    Paper hardware: {chosen.hardware}")
            if chosen.suggested_command:
                sections.append(f"    Suggested command (EXACT, all non-default flags): {chosen.suggested_command}")
            if chosen.paper_config:
                sections.append(f"    Paper-exact hyperparameters:")
                for k, v in list(chosen.paper_config.items())[:20]:
                    if v not in (None, ""):
                        sections.append(f"      - {k} = {v}")
            if chosen.config_source:
                sections.append(f"    Config source (paper + codebase): {chosen.config_source}")
            if chosen.codebase_config_files:
                sections.append(f"    Codebase config files (read + override these, do NOT guess):")
                for cf in chosen.codebase_config_files[:10]:
                    sections.append(f"      - /repo/{cf}")
            if chosen.missing_flags:
                sections.append(f"    Flags not exposed by script (agent must patch): {', '.join(chosen.missing_flags[:10])}")
            if chosen.matched_files:
                sections.append(f"    Matched files in repo: {', '.join(chosen.matched_files[:5])}")
            sections.append(
                f"    Tolerance: {chosen.tolerance_rule or '<=15% for ratios/speedups, <=3 abs pts for accuracy, <=5% for PPL/throughput'}"
            )
            if chosen.caveats:
                sections.append(f"    Caveats (from paper/README):")
                for cv in chosen.caveats[:6]:
                    sections.append(f"      * {cv}")
            if chosen.notes:
                sections.append(f"    Notes: {chosen.notes}")
            if len(paper_experiments) > 1:
                sections.append("  Fallback experiments (if the chosen one fails):")
                for fb in paper_experiments[1:4]:
                    fb_rt = (
                        f"~{fb.est_runtime_minutes:.0f} min"
                        if fb.est_runtime_minutes > 0 else "unknown"
                    )
                    fb_code = "code available" if fb.code_available else "no direct code match"
                    sections.append(f"    - {fb.name} ({fb_rt}, {fb_code})")
                    if fb.suggested_command:
                        sections.append(f"        cmd: {fb.suggested_command}")
                    for cv in (fb.caveats or [])[:2]:
                        sections.append(f"        caveat: {cv}")
        sections.append("  Verification protocol:")
        sections.append("    1. Complete Stage 1 (ROCM_ENV_VERIFIED) as today.")
        sections.append("    2. Run the chosen experiment with the EXACT paper/README config (no scale-down).")
        sections.append("    3. Capture stdout + artifacts to /repo/paper_experiment.log.")
        sections.append("    4. Delegate to the `paper-reproducer` sub-agent; it will read /repo/paper.pdf,")
        sections.append("       locate the relevant table/figure, compute a numeric delta, and fall back")
        sections.append("       to an LLM-judge verdict when the metric is not directly comparable.")
        sections.append("    5. Based on its JSON verdict, echo exactly ONE of:")
        sections.append("         echo PAPER_RESULT_REPRODUCED metric=<name> actual=<v> expected=<v> delta_pct=<x>")
        sections.append("         echo PAPER_RESULT_NOT_REPRODUCED <one-line reason>")
        sections.append("")

    # ── Execution plan ───────────────────────────────────────────────────────

    if rocm_mode:
        sections.append("Execution Plan (follow in order, skip reconnaissance — already done):")
        step = 1
        if py312_issues:
            sections.append(f"  {step}. Fix Python 3.12 compatibility issues (sed commands above)")
            step += 1
        sections.append(f"  {step}. Verify GPU: python -c \"import torch; print(torch.cuda.is_available())\"")
        step += 1
        sections.append(f"  {step}. Install dependencies (use filtered list above, skip pre-installed)")
        step += 1
        if pin_hazards:
            sections.append(f"  {step}. Handle version pin failures — drop old pins, install latest")
            step += 1
        if flagged_pkgs:
            sections.append(f"  {step}. Install CUDA->ROCm mapped packages (special handling above)")
            step += 1
        if code_hazards:
            sections.append(f"  {step}. Fix code hazards (wandb, cudnn flags, etc.)")
            step += 1
        if training_params:
            sections.append(f"  {step}. Scale down training params before running scripts (sed commands above)")
            step += 1
        if external_assets:
            sections.append(f"  {step}. DOWNLOAD REQUIRED EXTERNAL ASSETS (see 'EXTERNAL ASSETS' section above):")
            sections.append(f"       Check each asset is on disk BEFORE running any script that uses it.")
            sections.append(f"       For HuggingFace: huggingface-cli download <id> --repo-type dataset/model")
            sections.append(f"       For Google Drive: pip install gdown && gdown <url>")
            sections.append(f"       For download scripts: bash /repo/<script>")
            sections.append(f"       If source unclear: web_search '<repo_name> dataset download HuggingFace'")
            step += 1
        if no_scale_down:
            sections.append(f"  {step}. Run the EXACT commands from the README as-is — do NOT scale down, do NOT use mock data")
        else:
            sections.append(f"  {step}. Run target script with minimal args / mock data")
        step += 1
        sections.append(f"  {step}. Verify GPU execution (output must show cuda device, not cpu)")
        step += 1
        if run_mode == 'reproduce':
            # Mode 2: env check is a prerequisite but not the final goal — skip the explicit marker
            sections.append(f"  {step}. GPU confirmed — proceed directly to asset download and paper experiment")
            step += 1
        else:
            # Mode 1 / Mode 3: ROCM_ENV_VERIFIED is an explicit required milestone
            sections.append(f"  {step}. echo ROCM_ENV_VERIFIED")
            step += 1
        if reproduce_results:
            sections.append(f"  {step}. Verify all EXTERNAL ASSETS exist before the paper experiment — do NOT synthesize fake data")
            step += 1
            sections.append(f"  {step}. Run the Chosen experiment from PAPER REPRODUCTION TARGET (exact config, no scale-down)")
            step += 1
            sections.append(f"  {step}. Tee its output to /repo/paper_experiment.log")
            step += 1
            sections.append(f"  {step}. Invoke the paper-reproducer sub-agent to compare vs paper.pdf")
            step += 1
            sections.append(f"  {step}. echo PAPER_RESULT_REPRODUCED <metric=...> OR echo PAPER_RESULT_NOT_REPRODUCED <reason>")
        if no_scale_down:
            sections.append("")
            sections.append("*** NO-SCALE-DOWN MODE ACTIVE ***")
            sections.append("Do NOT reduce epochs, iterations, batch sizes, or any training parameters.")
            sections.append("Do NOT create mock/dummy data. Use the real data paths and commands from the README.")
            sections.append("Run scripts EXACTLY as the README describes, with all original arguments.")
    else:
        sections.append("Execution Plan:")
        sections.append("  1. Install dependencies following Install Strategy above")
        sections.append("  2. Run `runtest` or `poetryruntest` to verify")
        sections.append("  3. Fix errors iteratively until tests pass")
    sections.append("")

    # ── README snippet ───────────────────────────────────────────────────────

    if repo_readme_context:
        sections.append("README Corpus Highlights:")
        sections.append(repo_readme_context)
        sections.append("")
    elif readme_content:
        sections.append("README Content (full file):")
        sections.append(readme_content)
        sections.append("")

    if repo_config_context:
        sections.append("Config Corpus Highlights:")
        sections.append(repo_config_context)
        sections.append("")

    if repo_code_context:
        sections.append("Code Corpus Highlights:")
        sections.append(repo_code_context)
        sections.append("")

    raw_plan = "\n".join(sections)

    # ── Optionally refine with LLM ───────────────────────────────────────────

    if llm:
        plan = _refine_plan_with_llm(raw_plan, full_name, rocm_mode, llm, no_scale_down=no_scale_down)
    else:
        plan = raw_plan

    log_success("Plan generated successfully")
    recommended_image = base_image_name if rocm_mode and base_image_name else None
    paper_context = {
        "experiments": [c.to_dict() for c in paper_experiments] if paper_experiments else [],
        "title": paper_title or "",
    }
    return plan, recommended_image, paper_context


def _refine_plan_with_llm(raw_analysis: str, full_name: str, rocm_mode: bool, llm: str,
                         no_scale_down: bool = False) -> str:
    mode_label = "ROCm GPU migration" if rocm_mode else "environment configuration"

    if no_scale_down:
        scale_down_instruction = (
            "7. **NO-SCALE-DOWN MODE**: Do NOT include any sed commands to reduce epochs/iterations/steps. "
            "Do NOT suggest creating mock or dummy data. The agent must run the README commands "
            "exactly as written, with the original parameters and real data."
        )
        run_instruction = (
            "11. The agent must run the EXACT commands from the README with original arguments. "
            "Do NOT scale down any parameters. Do NOT create mock data."
        )
    else:
        scale_down_instruction = "7. List training parameters that must be scaled down with exact sed commands."
        run_instruction = (
            f"11. {'Describe how to create mock data and run with scaled-down parameters.' if rocm_mode else 'Describe how to run tests.'}"
        )

    prompt = f"""\
You are an expert build engineer. Given the following deep analysis of the repository
"{full_name}", produce a concise, step-by-step strategic plan for {mode_label}.

The plan MUST:
1. State the recommended base Docker image and Python version.
2. List ALL Python 3.12 compatibility fixes that must be applied FIRST (with exact sed commands).
3. List the exact install commands in order, using the FILTERED dependency list (not raw requirements.txt).
4. Flag version pins that will fail and recommend dropping them.
5. {"List all CUDA-to-ROCm migrations needed (package swaps, code patches, env vars)." if rocm_mode else ""}
6. {"List code hazards (wandb, cudnn, hardcoded paths) with fix commands." if rocm_mode else ""}
{scale_down_instruction}
8. **CRITICAL: Include the EXACT run commands from the README, VERBATIM, with all arguments
   (model names, dataset names, flags, etc.).** The execution agent will NOT read the README
   itself, so if the README says `python pred_mine.py --model longchat-v1.5-7b-32k`, that
   EXACT command must appear in the plan. Do NOT summarize or omit these commands.
9. **If the README specifies which model to use, state it explicitly** (e.g., "The README
   recommends model `longchat-v1.5-7b-32k`"). If the analysis shows which models are
   gated vs ungated, include that information and recommend the ungated model.
10. If the raw analysis contains a section called "Learned Prior For Planning",
   preserve the actionable structured guidance that matches the current repo.
   Do not drop it unless it clearly conflicts with current repo evidence.
{run_instruction}

CRITICAL: The execution agent will NOT re-read the README, directory listing, or config files.
The plan must contain ALL information the agent needs to start executing immediately.
Be specific. Use actual package names, file paths, and commands. Keep it under 1500 words.

CRITICAL: If the raw analysis contains a section titled "CRITICAL - Exact Run Commands from README",
you MUST copy those commands into the plan VERBATIM. These are the primary verification commands.
Do NOT replace them with generic `--help` commands.

CRITICAL: If the raw analysis contains "HuggingFace Model References", include the model
gating information in the plan so the agent knows which models to use and which to avoid.

CRITICAL: If the raw analysis contains a section titled "EXPECTED OUTCOMES FROM README",
you MUST copy it into the plan VERBATIM — including every script/command and its expected
outcome.  The execution agent will use this to validate its output.  Do NOT summarize,
paraphrase, or omit any expected outcomes.

CRITICAL: If the raw analysis contains a section titled
"AMD ROCm Ecosystem — Project-Specific Recommendations", copy it VERBATIM into
the plan. Do NOT summarise or drop any library entry. The executor uses these
exact install commands to set up AMD-native alternatives.

CRITICAL: If the raw analysis contains a section titled
"EXTERNAL ASSETS REQUIRED (DOWNLOAD BEFORE RUNNING SCRIPTS)", you MUST:
  a) Copy the entire section VERBATIM into the plan (every asset, every download command).
  b) Add a dedicated step in the Execution Plan BEFORE any training/inference/evaluation
     step: "Download required external assets (see EXTERNAL ASSETS section above)".
  c) Note that GitHub does not allow files > 25 MB — large datasets, pretrained
     checkpoints, and annotation archives will NEVER be present inside /repo and
     must always be fetched externally.
  d) If the download source is unclear for any asset, instruct the agent to run
     web_search("<repo_name> <asset_name> download") to discover the canonical path
     BEFORE attempting to create synthetic / mock data.
  e) For paper-reproduction runs, explicitly state: "Do NOT synthesise fake data
     or mock scenes — verify dataset and checkpoint existence first."

RAW ANALYSIS:
{raw_analysis}

STRATEGIC PLAN:"""

    messages = [{"role": "user", "content": prompt}]
    try:
        response, usage = get_llm_response(llm, messages, temperature=0.2, max_tokens=4096)
        if response and response[0]:
            log_info(f"LLM-refined plan: {usage.get('total_tokens', 0)} tokens used")
            return response[0]
    except Exception as e:
        log_info(f"LLM refinement failed ({e}), using raw plan")

    return raw_analysis


def print_plan(plan: str) -> None:
    console.print("\n")
    console.rule("[bold blue]STRATEGIC PLAN[/bold blue]", style="blue")
    console.print("")
    for line in plan.split("\n"):
        if line.startswith("Repository:") or line.startswith("ROCm Base Image:"):
            console.print(f"[bold cyan]{line}[/bold cyan]")
        elif "CRITICAL" in line or "WARNING" in line or "MUST" in line:
            console.print(f"[bold red]{line}[/bold red]")
        elif line.startswith("  ") and line.strip().startswith(tuple("0123456789")):
            console.print(f"[green]{line}[/green]")
        elif line.strip().startswith("pip install") or line.strip().startswith("sed "):
            console.print(f"[dim]{line}[/dim]")
        elif line.startswith("  - "):
            console.print(f"[yellow]{line}[/yellow]")
        elif line.endswith(":") or line.startswith("CUDA") or line.startswith("Verification") or line.startswith("Execution"):
            console.print(f"[bold]{line}[/bold]")
        else:
            console.print(line)
    console.print("")
    console.rule("[bold blue]END OF PLAN[/bold blue]", style="blue")
    console.print("\n")
