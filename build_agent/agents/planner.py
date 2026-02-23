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
from typing import Dict, List, Optional, Tuple, Set

from knowledge.rocm_knowledge import (
    ROCM_IMAGE_CATALOG,
    ROCM_PREINSTALLED_PACKAGES,
    CUDA_TO_ROCM_MAPPING,
    BANNED_NVIDIA_PACKAGES,
)
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
_MAX_README_CHARS = 6000
_MAX_SOURCE_SCAN_FILES = 120

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

def _read_file(path: str, max_chars: int = _MAX_FILE_CHARS) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", errors="ignore") as f:
            return f.read(max_chars)
    except Exception:
        return None


def _find_python_files(repo_path: str, limit: int = _MAX_SOURCE_SCAN_FILES) -> List[str]:
    py_files = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__" and d != "node_modules"]
        for f in files:
            if f.endswith(".py"):
                py_files.append(os.path.join(root, f))
                if len(py_files) >= limit:
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


def _llm_select_rocm_image(
    import_counts: Dict[str, int],
    config_contents: Dict[str, str],
    readme_content: Optional[str],
    py_file_contents: Dict[str, str],
    llm: str,
) -> dict:
    """
    Use the LLM to analyze the repository and select the best ROCm Docker image.

    The LLM sees the full picture: imports with frequencies, dependency files,
    README context, and code-level import patterns. It picks from the available
    ROCM_IMAGE_CATALOG entries and explains its reasoning.

    Returns dict with: image, tag, workload, description, reasoning
    """
    catalog_desc = _build_image_catalog_description()
    import_summary = _build_import_summary(import_counts)
    code_summary = _build_code_snippets_summary(py_file_contents)

    config_section = ""
    for fname, content in config_contents.items():
        config_section += f"\n--- {fname} ---\n{content[:2000]}\n"

    readme_snippet = ""
    if readme_content:
        readme_snippet = f"\n--- README (first 1500 chars) ---\n{readme_content[:1500]}\n"

    prompt = f"""\
You are an expert build engineer selecting the best ROCm Docker base image for a repository.

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

## Task

Analyze the repository's PRIMARY framework. Look at:
1. Which framework has the MOST imports across files (frequency matters most)
2. What the requirements.txt / setup.py actually lists as dependencies
3. What the README describes the project as
4. Whether a specialized image (sglang, vllm, megatron) is needed, or a general one

A repo might import multiple frameworks (e.g. both torch and jax) but typically one is the
PRIMARY framework used for the core logic, and others are secondary/utility imports.
Choose the image that matches the PRIMARY framework.

IMPORTANT rules:
- If the repo primarily uses PyTorch (torch) for training/inference without DeepSpeed/Megatron,
  select "pytorch"
- Only select "pytorch-training" if the repo uses distributed training libraries like
  DeepSpeed, Accelerate with FSDP, or PyTorch Lightning as core components
- Only select "jax" if JAX/Flax is the PRIMARY framework, not just a minor utility import
- Only select "vllm" or "sglang" if the repo actually uses those serving frameworks
- Only select "vllm-dev" if the repo IS a fork of vLLM itself

Respond with ONLY a JSON object (no markdown fences, no extra text):
{{"workload": "<key from the catalog>", "reasoning": "<one paragraph explaining why>"}}"""

    messages = [{"role": "user", "content": prompt}]
    try:
        response, usage = get_llm_response(llm, messages, temperature=0.1, max_tokens=512)
        if response and response[0]:
            log_info(f"LLM image selection: {usage.get('total_tokens', 0)} tokens used")
            return _parse_image_selection_response(response[0])
    except Exception as e:
        log_warning(f"LLM image selection failed ({e}), falling back to heuristic")

    return _fallback_image_selection(import_counts)


def _parse_image_selection_response(response_text: str) -> dict:
    """Parse the LLM JSON response into a structured image selection result."""
    import json as _json

    text = response_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    try:
        result = _json.loads(text)
    except _json.JSONDecodeError:
        json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if json_match:
            result = _json.loads(json_match.group())
        else:
            log_warning(f"Could not parse LLM image selection response, falling back")
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


# ── main planner ─────────────────────────────────────────────────────────────

def generate_plan(repo_path: str, full_name: str, rocm_mode: bool = False,
                  llm: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """
    Deep-analyze the repository and produce a comprehensive strategic plan.

    Returns:
        (plan_text, recommended_image) — recommended_image is the Docker image
        string (e.g. "rocm/vllm:latest") selected by context-aware analysis,
        or None if not in ROCm mode.
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
        content = _read_file(os.path.join(repo_path, rname), max_chars=_MAX_README_CHARS)
        if content:
            readme_content = content
            log_info(f"  Found README: {rname}")
            break

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

    # Training params
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
        sections.append(f"  {step}. Run target script with minimal args / mock data")
        step += 1
        sections.append(f"  {step}. Verify GPU execution (output must show cuda device, not cpu)")
        step += 1
        sections.append(f"  {step}. echo ROCM_ENV_VERIFIED")
    else:
        sections.append("Execution Plan:")
        sections.append("  1. Install dependencies following Install Strategy above")
        sections.append("  2. Run `runtest` or `poetryruntest` to verify")
        sections.append("  3. Fix errors iteratively until tests pass")
    sections.append("")

    # ── README snippet ───────────────────────────────────────────────────────

    if readme_content:
        sections.append("README Snippet (first 2000 chars):")
        sections.append(readme_content[:2000])
        sections.append("")

    raw_plan = "\n".join(sections)

    # ── Optionally refine with LLM ───────────────────────────────────────────

    if llm:
        plan = _refine_plan_with_llm(raw_plan, full_name, rocm_mode, llm)
    else:
        plan = raw_plan

    log_success("Plan generated successfully")
    recommended_image = base_image_name if rocm_mode and base_image_name else None
    return plan, recommended_image


def _refine_plan_with_llm(raw_analysis: str, full_name: str, rocm_mode: bool, llm: str) -> str:
    mode_label = "ROCm GPU migration" if rocm_mode else "environment configuration"
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
7. List training parameters that must be scaled down with exact sed commands.
8. Name the target script(s) to run for verification, with suggested arguments.
9. {"Describe how to create mock data and run with scaled-down parameters." if rocm_mode else "Describe how to run tests."}

CRITICAL: The execution agent will NOT re-read the README, directory listing, or config files.
The plan must contain ALL information the agent needs to start executing immediately.
Be specific. Use actual package names, file paths, and commands. Keep it under 1200 words.

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
