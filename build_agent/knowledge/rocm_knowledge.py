# ROCm Knowledge Base for Repo2Run
# This module contains structured data about AMD ROCm Docker images,
# pre-installed packages, CUDA-to-ROCm library mappings, and common
# code patterns that need adaptation for ROCm compatibility.

ROCM_IMAGE_CATALOG = {
    # ── Specialized serving / inference images (highest priority) ─────────
    "sglang": {
        "image": "rocm/sgl-dev",
        "tags": ["main", "latest"],
        "default_tag": "main",
        "description": (
            "SGLang runtime for AMD GPUs. Use when the repo IS an SGLang-based "
            "serving/inference project or depends heavily on sglang."
        ),
    },
    "vllm-dev": {
        "image": "rocm/vllm-dev",
        "tags": ["main", "latest"],
        "default_tag": "main",
        "description": (
            "vLLM dev image for AMD GPUs. Use when the repo IS a vLLM fork, extends "
            "vLLM internals, or contributes to vLLM development."
        ),
    },
    "vllm": {
        "image": "rocm/vllm",
        "tags": ["latest"],
        "default_tag": "latest",
        "description": (
            "vLLM serving image for AMD GPUs. Use when the repo builds ON TOP of "
            "vLLM (imports vllm, uses vllm.LLM, uses vllm-based serving) but is "
            "not itself a vLLM fork."
        ),
    },

    # ── Framework-specific images ────────────────────────────────────────
    "jax": {
        "image": "rocm/jax",
        "tags": ["latest"],
        "default_tag": "latest",
        "description": (
            "JAX with ROCm backend. Use when the repo primarily uses JAX/Flax/Optax "
            "and does NOT also heavily depend on PyTorch."
        ),
    },
    "tensorflow": {
        "image": "rocm/tensorflow",
        "tags": [
            "rocm6.3-tf2.17-dev",
            "rocm6.2.4-tf2.16-dev",
            "latest",
        ],
        "default_tag": "latest",
        "description": "TensorFlow with ROCm backend support.",
    },
    "onnxruntime": {
        "image": "rocm/onnxruntime",
        "tags": ["latest"],
        "default_tag": "latest",
        "description": (
            "ONNX Runtime with ROCm backend. Use when the repo primarily does "
            "ONNX model inference (imports onnxruntime, uses .onnx model files)."
        ),
    },

    # ── Training-optimized images ────────────────────────────────────────
    "pytorch-training": {
        "image": "rocm/pytorch-training",
        "tags": ["latest"],
        "default_tag": "latest",
        "description": (
            "Unified PyTorch base container optimized for distributed training "
            "with ROCm. Includes DeepSpeed, Megatron-LM, and FSDP support."
        ),
    },
    "megatron": {
        "image": "rocm/megatron-lm",
        "tags": ["latest"],
        "default_tag": "latest",
        "description": "Megatron-LM for large model training on ROCm.",
    },

    # ── General purpose (default fallback) ───────────────────────────────
    "pytorch": {
        "image": "rocm/pytorch",
        "tags": [
            "rocm6.3_ubuntu22.04_py3.10_pytorch_release_2.4.0",
            "rocm6.2.4_ubuntu22.04_py3.10_pytorch_release_2.3.0",
            "rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.7.1",
            "latest",
        ],
        "default_tag": "latest",
        "description": (
            "General-purpose PyTorch with ROCm. Default fallback for any ML repo "
            "that uses torch/pytorch for training or inference."
        ),
    },
}

# Packages that are typically pre-installed in ROCm images.
# The LLM agent should NOT attempt to reinstall these.
ROCM_PREINSTALLED_PACKAGES = {
    "rocm/pytorch": [
        "torch", "torchvision", "torchaudio",
        "numpy", "apex", "triton",
        "pillow", "pyyaml", "typing-extensions",
        "sympy", "networkx", "filelock", "jinja2",
        "cmake", "ninja", "packaging", "setuptools", "wheel",
    ],
    "rocm/tensorflow": [
        "tensorflow", "numpy", "keras",
        "tensorboard", "protobuf", "grpcio",
        "packaging", "setuptools", "wheel",
    ],
    "rocm/jax": [
        "jax", "jaxlib", "numpy",
        "scipy", "opt-einsum",
        "packaging", "setuptools", "wheel",
    ],
    "rocm/vllm": [
        "torch", "torchvision", "torchaudio",
        "vllm", "numpy", "triton",
        "transformers", "tokenizers", "safetensors",
        "packaging", "setuptools", "wheel",
    ],
    "rocm/vllm-dev": [
        "torch", "torchvision", "torchaudio",
        "vllm", "numpy", "triton",
        "transformers", "tokenizers", "safetensors",
        "ray", "aiohttp", "fastapi", "uvicorn",
        "packaging", "setuptools", "wheel",
    ],
    "rocm/sgl-dev": [
        "torch", "torchvision", "torchaudio",
        "sglang", "numpy", "triton",
        "transformers", "tokenizers", "safetensors",
        "vllm", "flashinfer",
        "packaging", "setuptools", "wheel",
    ],
    "rocm/onnxruntime": [
        "onnxruntime", "numpy",
        "protobuf", "flatbuffers",
        "packaging", "setuptools", "wheel",
    ],
    "rocm/pytorch-training": [
        "torch", "torchvision", "torchaudio",
        "numpy", "apex", "triton", "deepspeed",
        "pillow", "pyyaml", "typing-extensions",
        "cmake", "ninja", "packaging", "setuptools", "wheel",
    ],
    "rocm/megatron-lm": [
        "torch", "torchvision", "torchaudio",
        "numpy", "apex", "triton", "megatron-core",
        "packaging", "setuptools", "wheel",
    ],
}

# Mapping from CUDA/NVIDIA-specific packages to their ROCm alternatives
CUDA_TO_ROCM_MAPPING = {
    "flash-attn": {
        "rocm_package": "flash-attn",
        "install_cmd": (
            "git clone https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention && "
            "cd /tmp/flash-attention && "
            "FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE python setup.py install && "
            "export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE && "
            "echo 'export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE' >> /root/.bashrc"
        ),
        "notes": (
            "Use the MAIN Dao-AILab/flash-attention repo (NOT the ROCm fork). "
            "The Triton backend works out-of-the-box on rocm/pytorch:latest. "
            "No C++ compilation needed -- pure Python/Triton install via setup.py. "
            "NEVER use pip install flash-attn from PyPI -- those are CUDA-only prebuilt wheels. "
            "The ONLY env var needed is FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE (at install AND runtime). "
            "Without it, import flash_attn will fail with ModuleNotFoundError: No module named 'flash_attn_2_cuda'. "
            "Persist to /root/.bashrc so it survives across turns. "
            "NEVER set HSA_OVERRIDE_GFX_VERSION, PYTORCH_ROCM_ARCH, or MAX_JOBS for this install. "
            "These are NOT needed and will cause build failures or incorrect behavior. "
            "If install still fails, use PyTorch SDPA fallback."
        ),
    },
    "flash_attn": {
        "rocm_package": "flash-attn",
        "install_cmd": (
            "git clone https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention && "
            "cd /tmp/flash-attention && "
            "FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE python setup.py install && "
            "export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE && "
            "echo 'export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE' >> /root/.bashrc"
        ),
        "notes": (
            "Same as flash-attn (alternate import name). See flash-attn entry for full instructions. "
            "Use Dao-AILab/flash-attention main repo with Triton backend. "
            "The ONLY env var needed is FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE. "
            "NEVER set HSA_OVERRIDE_GFX_VERSION, PYTORCH_ROCM_ARCH, or MAX_JOBS."
        ),
    },
    "nvidia-ml-py": {
        "rocm_package": "pyrsmi",
        "install_cmd": "pip install pyrsmi",
        "notes": "ROCm equivalent of nvidia-ml-py for GPU monitoring. Also consider using rocm-smi CLI tool.",
    },
    "nvidia-ml-py3": {
        "rocm_package": "pyrsmi",
        "install_cmd": "pip install pyrsmi",
        "notes": "ROCm equivalent of nvidia-ml-py3.",
    },
    "pynvml": {
        "rocm_package": "pyrsmi",
        "install_cmd": "pip install pyrsmi",
        "notes": "ROCm equivalent of pynvml.",
    },
    "bitsandbytes": {
        "rocm_package": "bitsandbytes-rocm",
        "install_cmd": "pip install bitsandbytes-rocm",
        "alt_install_cmd": "git clone https://github.com/ROCm/bitsandbytes.git && cd bitsandbytes && pip install -e .",
        "notes": "For quantized model inference/training on ROCm.",
    },
    "xformers": {
        "rocm_package": "xformers",
        "install_cmd": "pip install xformers --index-url https://download.pytorch.org/whl/rocm6.2",
        "notes": "xformers has ROCm-compatible builds in PyTorch's wheel index.",
    },
    "triton": {
        "rocm_package": "triton",
        "install_cmd": "pip install triton",
        "notes": "Triton is typically pre-installed in ROCm PyTorch images. If needed, install from PyTorch's ROCm wheel index.",
    },
    "deepspeed": {
        "rocm_package": "deepspeed",
        "install_cmd": "pip install deepspeed",
        "notes": "DeepSpeed supports ROCm. Set DS_BUILD_OPS=1 and DS_BUILD_AIO=0 if building from source.",
    },
    "apex": {
        "rocm_package": "apex",
        "install_cmd": "pip install apex",
        "notes": "NVIDIA Apex is typically pre-installed in ROCm PyTorch images. Do not reinstall unless specifically needed.",
    },
    "cupy": {
        "rocm_package": None,
        "install_cmd": None,
        "notes": "CuPy does not have a direct ROCm equivalent. Consider using PyTorch tensors or numpy as alternatives.",
    },
    "pycuda": {
        "rocm_package": None,
        "install_cmd": None,
        "notes": "PyCUDA does not have a direct ROCm equivalent. Use HIP Python bindings or PyTorch instead.",
    },
}

# NVIDIA packages that should NEVER be installed in ROCm environments
BANNED_NVIDIA_PACKAGES = [
    "nvidia-cuda-runtime-cu11", "nvidia-cuda-runtime-cu12",
    "nvidia-cuda-cupti-cu11", "nvidia-cuda-cupti-cu12",
    "nvidia-cuda-nvrtc-cu11", "nvidia-cuda-nvrtc-cu12",
    "nvidia-cublas-cu11", "nvidia-cublas-cu12",
    "nvidia-cufft-cu11", "nvidia-cufft-cu12",
    "nvidia-curand-cu11", "nvidia-curand-cu12",
    "nvidia-cusolver-cu11", "nvidia-cusolver-cu12",
    "nvidia-cusparse-cu11", "nvidia-cusparse-cu12",
    "nvidia-nccl-cu11", "nvidia-nccl-cu12",
    "nvidia-nvjitlink-cu11", "nvidia-nvjitlink-cu12",
    "nvidia-nvtx-cu11", "nvidia-nvtx-cu12",
    "nvidia-cudnn-cu11", "nvidia-cudnn-cu12",
    "nvidia-dali-cuda110", "nvidia-dali-cuda120",
    "torch+cu118", "torch+cu121",
]

# Common CUDA-specific code patterns and their ROCm equivalents
CUDA_CODE_PATTERNS = {
    "nvidia-smi": {
        "cuda_pattern": "nvidia-smi",
        "rocm_replacement": "rocm-smi",
        "notes": "GPU monitoring tool replacement.",
    },
    "cuda_visible_devices": {
        "cuda_pattern": "CUDA_VISIBLE_DEVICES",
        "rocm_replacement": "HIP_VISIBLE_DEVICES",
        "notes": "Environment variable for selecting GPUs. CUDA_VISIBLE_DEVICES also works with ROCm PyTorch but HIP_VISIBLE_DEVICES is the native equivalent.",
    },
    "nccl_backend": {
        "cuda_pattern": "nccl",
        "rocm_replacement": "rccl",
        "notes": "NCCL equivalent for ROCm is RCCL. In PyTorch, torch.distributed still uses 'nccl' as the backend name for both CUDA and ROCm.",
    },
    "torch_cuda_api": {
        "cuda_pattern": "torch.cuda",
        "rocm_replacement": "torch.cuda",
        "notes": "PyTorch ROCm reuses the torch.cuda API. torch.cuda.is_available() returns True on ROCm. No code changes needed for standard PyTorch CUDA API calls.",
    },
    "cuda_device": {
        "cuda_pattern": "torch.device('cuda')",
        "rocm_replacement": "torch.device('cuda')",
        "notes": "Device specification remains 'cuda' in PyTorch ROCm. Do NOT change to 'rocm' or 'hip'.",
    },
}

# Supported AMD GPU architectures
SUPPORTED_GPU_ARCHITECTURES = {
    "gfx908": "MI100",
    "gfx90a": "MI210/MI250/MI250x",
    "gfx942": "MI300A/MI300X/MI325",
    "gfx950": "MI350/MI355 (ROCm 7.0+)",
    "gfx1030": "Navi21-based consumer GPUs",
    "gfx1100": "Navi31-based consumer GPUs (gfx1100/gfx1101)",
    "gfx1200": "Navi44/Navi48-based consumer GPUs (ROCm 6.4.2+)",
    "gfx1151": "Strix Halo/Strix Point APUs (ROCm 7.1+)",
}


def get_rocm_image_for_workload(keywords):
    """
    Legacy keyword-based selector (kept for backward compatibility).
    Prefer select_rocm_image() for deep context-aware selection.
    """
    result = select_rocm_image(
        import_counts={k: 1 for k in keywords},
        config_contents={},
        readme_content=None,
        top_level_files=[],
    )
    return result["image"], result["workload"]


# ── Signals: weighted evidence for each image type ───────────────────────

_IMAGE_SIGNALS = {
    "sglang": {
        "strong_imports": ["sglang", "sglang_router"],
        "strong_deps": ["sglang", "sglang-router"],
        "readme_patterns": [
            r"sglang", r"SGLang", r"sgl[-_]", r"from sglang",
            r"python -m sglang", r"sgl\.gen", r"sgl\.function",
        ],
        "file_patterns": ["sgl_", "sglang"],
        "code_patterns": [r"import sglang", r"from sglang", r"sgl\.gen", r"sgl\.function"],
    },
    "vllm-dev": {
        "strong_imports": [],
        "strong_deps": [],
        "readme_patterns": [
            r"vLLM\s+(?:fork|development|dev|contribution|custom)",
            r"extends?\s+vllm", r"modif(?:y|ied|ication)\s+.*vllm",
        ],
        "file_patterns": [],
        "code_patterns": [r"from vllm\.\w+\.\w+ import", r"class \w+\(vllm\."],
        "structure_signals": ["csrc/", "vllm/", "setup.py"],
    },
    "vllm": {
        "strong_imports": ["vllm"],
        "strong_deps": ["vllm", "vllm-flash-attn"],
        "readme_patterns": [
            r"vLLM", r"vllm\.LLM", r"vllm\.SamplingParams",
            r"from vllm import", r"pip install vllm",
            r"LLM serving", r"model serving",
        ],
        "file_patterns": ["serve", "serving", "endpoint", "api_server"],
        "code_patterns": [
            r"from vllm import", r"import vllm",
            r"vllm\.LLM\(", r"SamplingParams\(",
            r"AsyncLLMEngine", r"vllm\.entrypoints",
        ],
    },
    "jax": {
        "strong_imports": ["jax", "jaxlib", "flax", "optax", "equinox", "orbax"],
        "strong_deps": ["jax", "jaxlib", "flax", "optax", "equinox", "orbax", "chex"],
        "readme_patterns": [r"\bJAX\b", r"\bFlax\b", r"jax\.numpy", r"jnp\."],
        "file_patterns": [],
        "code_patterns": [
            r"import jax", r"from jax import", r"jax\.numpy",
            r"import flax", r"from flax", r"jax\.grad", r"jax\.jit",
            r"jax\.vmap", r"flax\.linen",
        ],
    },
    "tensorflow": {
        "strong_imports": ["tensorflow", "keras", "tf_agents"],
        "strong_deps": ["tensorflow", "tensorflow-gpu", "keras", "tf-agents"],
        "readme_patterns": [r"TensorFlow", r"tensorflow", r"tf\.", r"keras\."],
        "file_patterns": [],
        "code_patterns": [
            r"import tensorflow", r"from tensorflow", r"tf\.keras",
            r"tf\.data", r"tf\.GradientTape", r"tf\.function",
        ],
    },
    "onnxruntime": {
        "strong_imports": ["onnxruntime", "onnx"],
        "strong_deps": ["onnxruntime", "onnxruntime-gpu", "onnx", "onnxmltools"],
        "readme_patterns": [
            r"ONNX\s*Runtime", r"onnxruntime", r"\.onnx\b",
            r"ONNX model", r"onnx export",
        ],
        "file_patterns": [],
        "code_patterns": [
            r"import onnxruntime", r"from onnxruntime",
            r"onnxruntime\.InferenceSession", r"ort\.InferenceSession",
            r"\.onnx",
        ],
    },
    "pytorch-training": {
        "strong_imports": ["deepspeed", "accelerate", "lightning", "pytorch_lightning"],
        "strong_deps": [
            "deepspeed", "accelerate", "pytorch-lightning",
            "lightning", "fairscale", "colossalai",
        ],
        "readme_patterns": [
            r"[Dd]istributed training", r"DeepSpeed", r"FSDP",
            r"multi.?GPU training", r"data.?parallel",
            r"accelerate", r"pytorch.?lightning",
        ],
        "file_patterns": ["ds_config", "deepspeed_config", "accelerate_config"],
        "code_patterns": [
            r"import deepspeed", r"from deepspeed",
            r"torch\.distributed", r"DistributedDataParallel",
            r"FullyShardedDataParallel", r"from accelerate",
            r"import pytorch_lightning", r"import lightning",
        ],
    },
    "megatron": {
        "strong_imports": ["megatron", "megatron_core"],
        "strong_deps": ["megatron-core", "megatron-lm"],
        "readme_patterns": [r"Megatron", r"megatron.?lm", r"tensor.?parallel"],
        "file_patterns": ["megatron"],
        "code_patterns": [
            r"from megatron", r"import megatron",
            r"megatron\.core", r"tensor_parallel",
        ],
    },
}

import re as _re


def select_rocm_image(
    import_counts: dict,
    config_contents: dict,
    readme_content: str = None,
    top_level_files: list = None,
    py_file_contents: dict = None,
) -> dict:
    """
    Context-aware ROCm Docker image selector.

    Instead of matching keywords, this function scores each candidate image
    by analyzing multiple signals:
      - Python imports (weighted by frequency)
      - Dependency declarations in requirements/setup files
      - README content (project description, usage examples)
      - Project file/directory structure
      - Actual source code patterns

    Returns dict with keys: image, tag, workload, description, score, reasoning
    """
    scores: dict = {}
    reasoning: dict = {}
    top_level = top_level_files or []
    all_config_text = "\n".join(config_contents.values()) if config_contents else ""
    imports_lower = {k.lower().replace("-", "_"): v for k, v in import_counts.items()}

    for workload, signals in _IMAGE_SIGNALS.items():
        score = 0.0
        reasons = []

        # Signal 1: Strong imports (high weight)
        for imp in signals.get("strong_imports", []):
            imp_norm = imp.lower().replace("-", "_")
            if imp_norm in imports_lower:
                freq = imports_lower[imp_norm]
                weight = 30 if freq >= 5 else (20 if freq >= 2 else 10)
                score += weight
                reasons.append(f"import:{imp} (freq={freq}, +{weight})")

        # Signal 2: Dependency declarations in config files
        for dep in signals.get("strong_deps", []):
            dep_norm = dep.lower().replace("-", "_")
            config_norm = all_config_text.lower().replace("-", "_")
            if dep_norm in config_norm:
                score += 15
                reasons.append(f"dep:{dep} in configs (+15)")

        # Signal 3: README content analysis
        if readme_content:
            for pat_str in signals.get("readme_patterns", []):
                pat = _re.compile(pat_str, _re.IGNORECASE)
                matches = pat.findall(readme_content)
                if matches:
                    weight = min(len(matches) * 5, 20)
                    score += weight
                    reasons.append(f"readme:/{pat_str}/ x{len(matches)} (+{weight})")

        # Signal 4: File/directory structure
        for fp in signals.get("file_patterns", []):
            fp_lower = fp.lower()
            for tf in top_level:
                if fp_lower in tf.lower():
                    score += 8
                    reasons.append(f"file:{tf} matches '{fp}' (+8)")

        # Signal 5: Structure signals (directories that indicate this IS a fork/project)
        for sp in signals.get("structure_signals", []):
            sp_stripped = sp.rstrip("/")
            for tf in top_level:
                if tf == sp_stripped or tf == sp:
                    score += 25
                    reasons.append(f"structure:{tf} (+25)")

        # Signal 6: Source code patterns (most precise but most expensive)
        if py_file_contents:
            for pat_str in signals.get("code_patterns", []):
                pat = _re.compile(pat_str)
                match_count = 0
                for _fpath, content in py_file_contents.items():
                    match_count += len(pat.findall(content))
                if match_count > 0:
                    weight = min(match_count * 3, 25)
                    score += weight
                    reasons.append(f"code:/{pat_str}/ x{match_count} (+{weight})")

        scores[workload] = score
        reasoning[workload] = reasons

    # ── vllm-dev vs vllm disambiguation ──────────────────────────────────
    # If the repo looks like a vLLM fork (has vllm/ directory + csrc/ or setup.py
    # at top level), boost vllm-dev and suppress vllm
    if scores.get("vllm-dev", 0) > 0 and scores.get("vllm", 0) > 0:
        if scores["vllm-dev"] > scores["vllm"]:
            scores["vllm"] = 0
            reasoning["vllm"] = ["suppressed: vllm-dev scored higher (repo is a vLLM fork)"]

    # ── JAX vs PyTorch disambiguation ────────────────────────────────────
    # If both JAX and PyTorch have signals, check which is primary
    jax_score = scores.get("jax", 0)
    pytorch_score = scores.get("pytorch", 0)
    if jax_score > 0 and pytorch_score > 0:
        torch_freq = imports_lower.get("torch", 0)
        jax_freq = imports_lower.get("jax", 0) + imports_lower.get("flax", 0)
        if jax_freq > torch_freq * 2:
            scores["pytorch"] = 0
            reasoning.setdefault("pytorch", []).append("suppressed: JAX is primary framework")
        elif torch_freq > jax_freq * 2:
            scores["jax"] = 0
            reasoning.setdefault("jax", []).append("suppressed: PyTorch is primary framework")

    # ── Select winner ────────────────────────────────────────────────────
    best_workload = max(scores, key=scores.get) if scores else "pytorch"
    best_score = scores.get(best_workload, 0)

    # If no strong signal for anything, fall back to pytorch
    if best_score < 10:
        has_torch = "torch" in imports_lower
        if has_torch:
            best_workload = "pytorch"
            reasoning.setdefault("pytorch", []).append("fallback: torch in imports, no specialized image matched")
        else:
            best_workload = "pytorch"
            reasoning.setdefault("pytorch", []).append("fallback: no GPU framework detected, using default")

    entry = ROCM_IMAGE_CATALOG.get(best_workload, ROCM_IMAGE_CATALOG["pytorch"])

    return {
        "image": f"{entry['image']}:{entry['default_tag']}",
        "tag": entry["default_tag"],
        "workload": best_workload,
        "description": entry["description"],
        "score": best_score,
        "reasoning": reasoning.get(best_workload, []),
        "all_scores": {k: v for k, v in sorted(scores.items(), key=lambda x: -x[1]) if v > 0},
    }


def get_preinstalled_packages(image_name):
    """
    Given a ROCm Docker image name (e.g., 'rocm/pytorch'),
    return a list of packages that are pre-installed.
    """
    base = image_name.split(":")[0] if ":" in image_name else image_name
    return ROCM_PREINSTALLED_PACKAGES.get(base, [])


def get_rocm_alternative(package_name):
    """
    Given a CUDA/NVIDIA package name, return the ROCm alternative info
    or None if no mapping exists.
    """
    normalized = package_name.lower().replace('-', '_')
    for cuda_pkg, info in CUDA_TO_ROCM_MAPPING.items():
        if cuda_pkg.lower().replace('-', '_') == normalized:
            return info
    return None


def is_banned_package(package_name):
    """Check if a package is an NVIDIA-specific package that should not be installed."""
    normalized = package_name.lower().replace('-', '_').replace('.', '_')
    for banned in BANNED_NVIDIA_PACKAGES:
        if banned.lower().replace('-', '_').replace('.', '_') == normalized:
            return True
    return False


def generate_rocm_prompt_section():
    """
    Generate the ROCm-specific section to be injected into the LLM system prompt.
    Returns a string with ROCm instructions.
    """
    image_list = ""
    for wtype, info in ROCM_IMAGE_CATALOG.items():
        keywords_str = ", ".join(info["keywords"])
        image_list += f"  - If the repo uses [{keywords_str}]: `change_base_image {info['image']}:{info['default_tag']}`\n"

    preinstalled_str = ""
    for img, pkgs in ROCM_PREINSTALLED_PACKAGES.items():
        preinstalled_str += f"  - `{img}`: {', '.join(pkgs)}\n"

    cuda_mapping_str = ""
    for cuda_pkg, info in CUDA_TO_ROCM_MAPPING.items():
        if info["rocm_package"]:
            cuda_mapping_str += f"  - `{cuda_pkg}` -> Install with: `{info['install_cmd']}`\n"
        else:
            cuda_mapping_str += f"  - `{cuda_pkg}` -> No ROCm equivalent. {info['notes']}\n"

    banned_str = ", ".join(f"`{p}`" for p in BANNED_NVIDIA_PACKAGES[:10]) + ", etc."

    prompt = f"""
## AMD ROCm GPU MODE - CRITICAL INSTRUCTIONS

You are configuring this repository to run on **AMD GPUs with ROCm** (not NVIDIA CUDA).
You MUST follow the ROCm-specific workflow below. Do NOT skip steps.

**CRITICAL RULE: Do NOT use `runtest` or `poetryruntest` in ROCm mode. They are DISABLED.**
**CRITICAL RULE: Output EXACTLY ONE ```bash``` block per response. Wait for the real result before your next action.**

### MANDATORY STEP 1: Read the README and understand the project FIRST
Before doing ANYTHING else, your FIRST action MUST be:
```bash
cat /repo/README.md
```
Then in subsequent turns:
- `ls -la /repo` to see the full directory structure
- `find /repo -maxdepth 2 -name "*.py" | head -30` to see Python files
Identify from the README:
   - What the project does (training, inference, data processing, etc.)
   - How to install it (what commands the README says to run)
   - **What example/demo scripts exist** (e.g., `example_mm.py`, `train.py`, `run.py`, `demo.py`, etc.)
   - What dependencies it needs
   - What GPU framework it uses (PyTorch, TensorFlow, JAX, etc.)

**This step is NOT optional.** You must read the README before proceeding.

### MANDATORY STEP 2: Select the correct ROCm base image
Based on what you learned from the README and config files, determine the workload type, then use `change_base_image` to switch to the appropriate ROCm Docker image:
{image_list}  - If the repo does NOT use any GPU frameworks: keep the current Python image (no GPU needed).

**IMPORTANT: Check your CURRENT image first (shown in the `[image=...]` header). If you are ALREADY
on the correct image (e.g., already on `rocm/pytorch:latest`), do NOT issue `change_base_image` again.
Switching to the same image wastes a turn and causes a full container restart for no reason.**

After switching the base image (or confirming you're already on it), run `pip list` to see what is already installed.

### STEP 3: Install dependencies following the README instructions
Follow the installation instructions from the README. Typical steps:
- If the README says `pip install -r requirements.txt`, do that (but skip pre-installed and banned packages).
- If the README says `pip install -e .`, do that.
- If the README says `python setup.py install`, do that.
- Install any additional dependencies the README mentions.
- Use `waitinglist add -p <package> -t pip` and then `download` when possible.

### CRITICAL: uv / poetry / conda can SHADOW the ROCm PyTorch — ALWAYS verify after install!

**This is the #1 cause of scripts running on CPU instead of GPU in the ROCm Docker container.**

When a project uses `uv sync`, `poetry install`, or `conda install`, these tools create their OWN
virtual environment and install their OWN PyTorch from PyPI. The PyPI wheels are CUDA-only or CPU-only —
they do NOT have ROCm support. This SHADOWS the ROCm PyTorch that is pre-installed in the Docker image.

**Symptoms:** After `uv sync` or `poetry install`, `torch.cuda.is_available()` returns `False`,
and scripts load models on CPU even though `rocm-smi` shows GPUs are present.

**MANDATORY: After ANY dependency installation (pip install, uv sync, poetry install), ALWAYS verify:**
```bash
python -c "import torch; print('ROCm:', torch.cuda.is_available(), '| Version:', torch.__version__)"
```
If this prints `ROCm: False`, the PyTorch was overwritten. Fix it immediately:

**Fix for `uv` projects:**
Option A (preferred): Install project without touching torch, then run with system python:
```bash
pip install -e . --no-deps && pip install -e packages/*/  --no-deps 2>/dev/null; true
```
Then run scripts with `python` (NOT `uv run`), setting PYTHONPATH as needed.

Option B: After `uv sync`, force-reinstall the ROCm torch:
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```

**Fix for `poetry` projects:**
```bash
poetry install --no-root
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2 --force-reinstall
```

**Fix for `conda` projects:**
Do NOT run `conda install pytorch`. Instead:
```bash
pip install -r requirements.txt --ignore-installed torch torchvision torchaudio
```

**RULE: After fixing, ALWAYS re-verify `torch.cuda.is_available() == True` before proceeding.**
**If `uv run` uses a separate venv, use `python` directly instead of `uv run`.**

### STEP 4: Verify GPU access and ROCm functionality — MANDATORY, NEVER SKIP
**This step is MANDATORY after every dependency installation.** If GPU verification fails,
you MUST fix it before proceeding. Do NOT continue to run scripts on CPU.

1. **Check GPU hardware:**
   ```bash
   rocm-smi
   ```
   This should show your AMD GPU(s) (MI200, MI250, MI300, etc.).

2. **Verify PyTorch can see ROCm GPUs:**
   ```bash
   python -c "import torch; print('ROCm available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
   ```
   This MUST print `ROCm available: True` and show your GPU details.

   **If it prints `False`:** Your PyTorch installation was overwritten by a non-ROCm version.
   Go back and follow the fix instructions in "CRITICAL: uv / poetry / conda" section above.
   **Do NOT proceed until this prints `True`.**

3. **If the project uses `uv run`, verify THAT python also sees GPU:**
   ```bash
   uv run python -c "import torch; print('ROCm via uv:', torch.cuda.is_available())"
   ```
   If this prints `False` but step 2 printed `True`, then `uv` has its own non-ROCm PyTorch.
   **Solution: Do NOT use `uv run`. Use `python` directly with PYTHONPATH instead.**

### STEP 5: Verify by ACTUALLY RUNNING the project's scripts
**Do NOT use `runtest`. It is disabled in ROCm mode.**
**Running `--help` alone is NOT SUFFICIENT. You MUST actually execute the script.**

Verification steps (you MUST do ALL of these):
1. Verify all core imports work:
   `python -c "from <main_package> import <main_class>; print('OK')"`

2. Run the main script with `--help` to understand its interface and arguments:
   `cd /repo && python <main_script>.py --help`

3. **CRITICAL: Actually run the script.** This is the real verification.
   - Read the README to understand what data/inputs the script expects.
   
   **MANDATORY PRE-RUN CHECK: Scale down epochs, iterations, and data BEFORE running ANY script.**
   
   **This is CRITICAL. Training scripts often default to hundreds or thousands of epochs (e.g., 2000, 500, 100)
   or process very large datasets. Running them at full scale will HANG the agent for hours and waste all remaining turns.
   Your goal is to verify the environment works, NOT to complete full training.**
   
   **THIS APPLIES TO EVERY SINGLE PYTHON SCRIPT YOU RUN — not just the first one.**
   If a project has multiple scripts,
   you MUST scale down EACH script BEFORE running it. Scaling down one script does NOT
   automatically scale down others — they each have their own hardcoded parameters.
   
   **BEST PRACTICE: Batch-patch ALL runnable scripts at once BEFORE running any of them:**
   ```bash
   # Scale down ALL Python scripts in the project at once
   find /repo -name "*.py" -exec grep -l "epochs" {{}} \; | xargs -I{{}} sed -i "s/'epochs': [0-9]*/'epochs': 2/g" {{}}
   find /repo -name "*.py" -exec grep -l "num_train" {{}} \; | xargs -I{{}} sed -i "s/'num_train': [0-9]*/'num_train': 10/g" {{}}
   find /repo -name "*.py" -exec grep -l "num_test" {{}} \; | xargs -I{{}} sed -i "s/'num_test': [0-9]*/'num_test': 5/g" {{}}
   ```
   This ensures you don't forget to scale down any script. Do this ONCE, early in the process.
   
   **BEFORE running EACH training/inference script, you MUST:**
   
   a) **Inspect THAT SPECIFIC script for hardcoded training parameters:**
      ```bash
      grep -n "epochs\|num_epochs\|n_epochs\|max_steps\|max_iter\|num_train\|num_test\|num_samples\|iterations\|total_steps" <script>.py
      ```
   
   b) **If epochs/iterations are passed via CLI arguments** (e.g., `--epochs`), pass small values:
      ```bash
      python <script>.py --epochs 2 --max_steps 5 <other_args>
      ```
   
   c) **If epochs/iterations are HARDCODED in the script** (no argparse, just a dict or variable), you MUST
      use `sed` to reduce them BEFORE running. This is very common — many research scripts hardcode
      `'epochs': 2000` or `num_iterations = 10000` in the `__main__` block. Examples:
      ```bash
      # Reduce hardcoded epochs from any large number to 2
      sed -i "s/'epochs': [0-9]*/'epochs': 2/g" <script>.py
      # Reduce hardcoded iterations
      sed -i "s/iterations = [0-9]*/iterations = 5/g" <script>.py
      # Reduce num_train and num_test to small values
      sed -i "s/'num_train': [0-9]*/'num_train': 10/g" <script>.py
      sed -i "s/'num_test': [0-9]*/'num_test': 5/g" <script>.py
      ```
      **IMPORTANT: Do this for EVERY script you plan to run, not just the first one!**
      Example: If the project has `norm.py` and `norm_DeltaPhi.py`, you must `sed` BOTH:
      ```bash
      sed -i "s/'epochs': [0-9]*/'epochs': 2/g" norm.py norm_DeltaPhi.py
      ```
   
   d) **If the script uses a config file** (YAML, JSON, .cfg), edit it to reduce epochs:
      ```bash
      sed -i 's/epochs: [0-9]*/epochs: 2/' config.yaml
      sed -i 's/max_steps: [0-9]*/max_steps: 5/' config.yaml
      ```
   
   e) **Common parameters to ALWAYS scale down** (target values in parentheses):
      - `epochs` / `num_epochs` / `n_epochs` → (1-2)
      - `max_steps` / `num_steps` / `total_steps` / `iterations` → (3-5)
      - `num_train` / `ntrain` → (5-10)
      - `num_test` / `ntest` → (3-5)
      - `batch_size` → keep as-is or reduce to (2-4) if dataset is tiny
      - `num_workers` → (0 or 1)
      - `save_every` / `eval_every` / `log_every` → (1)
   
   **NEVER run ANY Python script without first checking and reducing these values.
   A script running for 2000 epochs = WASTED RUN. You have limited turns. Be smart.
   If you scaled down script_A.py but forgot script_B.py, running script_B.py WILL timeout.**
   
   **A. For TRAINING scripts (train.py, fine_tune.py, etc.):**
   - If the script requires data download or pre-trained models:
     ```bash
     # Check if data/model downloading is needed
     cd /repo && python <train_script>.py --help | grep -E "data|model|download"
     ```
   - **If large downloads are needed (>1GB models, large datasets):**
     Create a **small dummy dataset** to verify the training loop starts successfully.
     **IMPORTANT: For multiline Python code, ALWAYS write a .py file first, then run it.**
     **Do NOT use multiline `python -c` -- it breaks due to newline handling in the sandbox.**

     Example for text/JSON data -- write a helper script:
     ```bash
     cat > /tmp/create_dummy_data.py << 'PYEOF'
import os, json, torch
os.makedirs('/tmp/dummy_data', exist_ok=True)
data = [{{"input": "test " + str(i), "output": "result " + str(i)}} for i in range(10)]
with open('/tmp/dummy_data/train.json', 'w') as f:
    json.dump(data, f)
print('Created dummy training data with 10 samples')
PYEOF
     ```
     Then run it:
     ```bash
     python /tmp/create_dummy_data.py
     ```

     Example for ImageNet-style image data (class folders with images):
     ```bash
     cat > /tmp/create_dummy_images.py << 'PYEOF'
import os
from PIL import Image
import numpy as np
for split in ['train', 'val']:
    for c in range(2):
        d = f'/tmp/dummy_imagenet/{{split}}/class_{{c}}'
        os.makedirs(d, exist_ok=True)
        for i in range(5):
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img.save(f'{{d}}/img_{{i}}.jpg')
print('Created dummy ImageNet dataset: 2 classes x 5 images x train+val')
PYEOF
     ```
     Then run it:
     ```bash
     python /tmp/create_dummy_images.py
     ```
   - **Remember: You MUST have already scaled down epochs/iterations (see pre-run check above).**
   - Run the training script with the dummy data and **minimal settings** (e.g., `--epochs 1`, `--max_steps 10`):
     ```bash
     cd /repo && python <train_script>.py --data_path /tmp/dummy_data --epochs 1 --max_steps 10 <other_minimal_args>
     ```
   - **IMPORTANT: For training scripts, you DO NOT need to wait for full training to complete.**
     - If the script starts training and shows progress (e.g., "Epoch 1/1, Step 1/10, Loss: 2.345..."), **that is sufficient verification**.
     - You can monitor the first few steps to ensure no errors, then you may interrupt with Ctrl+C if it's clearly working.
     - Interrupting a training script (Ctrl+C or letting it timeout) does **NOT** revert the environment.
   
   **B. For INFERENCE scripts (infer.py, predict.py, generate.py, etc.):**
   - If the script needs input images/data, create minimal mock data using a helper script:
     ```bash
     cat > /tmp/create_test_images.py << 'PYEOF'
import os
from PIL import Image
import numpy as np
os.makedirs('/tmp/test_data', exist_ok=True)
for i in range(3):
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    img.save(f'/tmp/test_data/img_{{i}}.png')
print('Created 3 test images')
PYEOF
     ```
     Then run it:
     ```bash
     python /tmp/create_test_images.py
     ```
   - Then run the script with the mock data and minimal arguments:
     ```bash
     cd /repo && python <infer_script>.py --input_path /tmp/test_data <other_minimal_args>
     ```
   
   **C. General guidelines:**
   - If the script needs to download a large model from HuggingFace (>5GB), create a verification script
     that tests the model loading path with dummy weights or use a smaller model variant if available.
   - **Do NOT use `timeout` commands.** Let the script run naturally, but ONLY after you have
     scaled down epochs/iterations to minimal values (1-5). A properly scaled-down script should
     finish in under 2 minutes. If you forgot to scale down, the script WILL hang for hours.
   - **Do NOT use `--help` as a substitute for actually running.** The `--help` flag only tests import
     paths; it does NOT verify the code actually works end-to-end.
   - If the script fails with an error, debug and fix it before declaring success.
   - **ALWAYS `grep` for epoch/iteration counts in scripts AND config files before running them.**
     Many research repos hardcode `epochs=2000` or `iterations=50000` with no CLI override.
     Use `sed` to patch these values down to 1-2 epochs before execution.

4. Once the script has ACTUALLY RUN and produced real output (not just --help text):
   - For training scripts: Verify it shows "Epoch 1" or "Step 1" with loss/metrics
   - For inference scripts: Verify it produces predictions/outputs/results
   - For data processing: Verify it processes data and saves outputs
   
   **CRITICAL: Confirm the script ran on GPU, NOT CPU.** Look for these indicators in the output:
   - `Using device: cuda` or `Using cuda device` → GOOD (GPU)
   - `device=cpu` or `Loading model on cpu` → BAD (CPU, must fix!)
   - If the output says `device=cpu` or loads the model on CPU, the environment is NOT correctly
     configured. Go back and check if PyTorch has ROCm support (`torch.cuda.is_available()`).
   - If the script has no device indicator in its output, verify manually:
     ```bash
     python -c "import torch; m = torch.zeros(1).cuda(); print('GPU test:', m.device)"
     ```
   
   **Only after confirming GPU usage AND actual script output**, declare success:
   ```bash
   echo ROCM_ENV_VERIFIED
   ```

**IMPORTANT: You MUST show that the script produces actual output ON GPU (not CPU).
Scripts running on CPU in an AMD GPU container means the environment is BROKEN.
`--help` output does NOT count as verification.
Output showing `device=cpu` does NOT count as success.**

### CRITICAL: Multiline commands and PYTHONPATH

**Multiline Python code:** NEVER use multiline `python -c "..."` commands (they break due to
newline handling). Instead, write a `.py` file first using heredoc, then run it:
```bash
cat > /tmp/my_script.py << 'PYEOF'
import torch
print(torch.cuda.is_available())
PYEOF
```
Then in the next turn:
```bash
python /tmp/my_script.py
```

**PYTHONPATH:** If the project's scripts import from sibling directories (e.g., `from utils import ...`
when `utils.py` is in `/repo/src/`), you MUST persist PYTHONPATH to `/root/.bashrc` so it survives
across all subsequent commands. Do this ONCE, then source it:
```bash
echo 'export PYTHONPATH=/repo/src:$PYTHONPATH' >> /root/.bashrc && source /root/.bashrc
```
**IMPORTANT:** A plain `export PYTHONPATH=...` is lost if the environment reverts. Always write to
`/root/.bashrc` so it persists. After writing, verify with:
```bash
echo $PYTHONPATH
```

Common patterns:
- `from utils import ...` when utils.py is in `/repo/src/` -> PYTHONPATH=/repo/src
- `from models import ...` when models/ is in `/repo/src/` -> PYTHONPATH=/repo/src
- Relative imports failing -> check where the package root is and add it to PYTHONPATH
Always check the project structure to determine the correct PYTHONPATH.

### Pre-installed packages - DO NOT reinstall these
The following packages come pre-installed in ROCm Docker images. Do NOT add them to the waiting list or pip install them:
{preinstalled_str}
If a requirements.txt lists `torch`, `torchvision`, `torchaudio`, `numpy`, or other pre-installed packages, SKIP them when adding to the waiting list. You can verify what's installed by running `pip list` inside the container.

### CUDA-to-ROCm package mapping
When the repository requires CUDA-specific packages, use these ROCm alternatives:
{cuda_mapping_str}
### BANNED packages - NEVER install these
The following NVIDIA-specific packages are incompatible with ROCm and must NEVER be installed:
{banned_str}
If you encounter these in requirements.txt, skip them. They are CUDA runtime libraries that are not needed on ROCm.

### Code compatibility rules
- **torch.cuda API**: Works AS-IS on ROCm. `torch.cuda.is_available()` returns True on AMD GPUs. Do NOT change `torch.cuda` to anything else.
- **torch.device('cuda')**: Works AS-IS. Do NOT change device strings to 'rocm' or 'hip'.
- **nvidia-smi**: Replace with `rocm-smi` in any scripts or code.
- **CUDA_VISIBLE_DEVICES**: Works on ROCm, but the native equivalent is `HIP_VISIBLE_DEVICES`.
- **NCCL**: PyTorch distributed still uses 'nccl' as backend name on ROCm (it maps to RCCL internally).
- **Custom CUDA kernels (.cu files)**: These need HIPIFY conversion. If the repo has custom .cu files, they may need to be converted using `hipify-perl` or `hipify-clang`.

### PyTorch index URL for ROCm wheels
If you need to install PyTorch-ecosystem packages not in the base image:
`pip install <package> --index-url https://download.pytorch.org/whl/rocm6.2`

### FlashAttention for ROCm - PROVEN INSTALLATION STEPS

If the project requires `flash-attn` or `flash_attn`:

**Do NOT run `pip install flash-attn` from PyPI -- those are CUDA-only prebuilt wheels and WILL FAIL on ROCm.**

Use the **Triton backend** from the main Dao-AILab repo. This is a pure Python/Triton install --
no C++ compilation, no submodules needed. It works out-of-the-box on `rocm/pytorch:latest`.

**FORBIDDEN environment variables -- NEVER set these for flash-attn installation:**
- `HSA_OVERRIDE_GFX_VERSION` -- NOT needed, causes incorrect GPU targeting
- `PYTORCH_ROCM_ARCH` -- NOT needed, the Triton backend does not compile GPU kernels
- `MAX_JOBS` -- NOT needed, there is no C++ compilation
- `GPU_ARCHS` -- NOT needed

The ONLY environment variable needed is `FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"`.

Follow these steps EXACTLY, one action per turn:

**Step 1: Clone the main flash-attention repository**
```bash
git clone https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention
```

**Step 2: Install with Triton AMD flag**
```bash
cd /tmp/flash-attention && FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" python setup.py install
```
This installs flash_attn as a pure Python package using the Triton backend. No GPU compilation required.
The install takes ~30 seconds (not minutes).

**Step 3: CRITICAL -- Set the runtime environment variable and persist it**
You MUST set `FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"` at runtime, otherwise
`import flash_attn` will fail with `ModuleNotFoundError: No module named 'flash_attn_2_cuda'`.
```bash
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" && echo 'export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"' >> /root/.bashrc
```

**Step 4: Verify the installation**
```bash
FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" python -c "from flash_attn import flash_attn_func; print('flash_attn OK')"
```

---
#### FALLBACK: PyTorch SDPA (if flash_attn install fails)
If the Triton-based install fails, use PyTorch's built-in **Scaled Dot-Product Attention (SDPA)**
as a universal fallback. SDPA works on ALL backends (CUDA, ROCm, CPU) and requires NO extra
installation -- it is part of PyTorch >= 2.0.

Create a compatibility shim that redirects flash_attn calls to torch SDPA:

```bash
cat > /repo/flash_attn_sdpa_fallback.py << 'PYEOF'
# Drop-in SDPA fallback for flash_attn when flash-attention install fails.
import torch
import torch.nn.functional as F

def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
    # flash_attn expects (batch, seqlen, nheads, headdim)
    # SDPA expects (batch, nheads, seqlen, headdim)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale)
    return out.transpose(1, 2)

def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
    q, k, v = qkv.unbind(dim=2)
    return flash_attn_func(q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal)

def flash_attn_kvpacked_func(q, kv, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
    k, v = kv.unbind(dim=2)
    return flash_attn_func(q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal)

def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
    batch_size = cu_seqlens_q.shape[0] - 1
    q = q.view(batch_size, max_seqlen_q, -1, q.shape[-1])
    k = k.view(batch_size, max_seqlen_k, -1, k.shape[-1])
    v = v.view(batch_size, max_seqlen_k, -1, v.shape[-1])
    out = flash_attn_func(q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal)
    return out.view(-1, out.shape[-2], out.shape[-1])

def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
    q, k, v = qkv.unbind(dim=1)
    return flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal)
PYEOF
echo "Created SDPA fallback at /repo/flash_attn_sdpa_fallback.py"
```

Then modify the project's import statements. For example, if the project has:
```python
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
```
Change it to:
```python
try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
except (ImportError, ModuleNotFoundError):
    from flash_attn_sdpa_fallback import flash_attn_varlen_qkvpacked_func
```

**IMPORTANT: SDPA is a FALLBACK. Always try the Triton install FIRST.**
**SDPA does not support all flash_attn features (e.g., variable-length sequences with different lengths per batch).**
**But it is far better than commenting out flash_attn imports and disabling the functionality entirely.**

---
#### KEY RULES:
- **Clone from https://github.com/Dao-AILab/flash-attention.git** (the MAIN repo, not ROCm fork).
- **ALWAYS set `FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"`** both at install time AND at runtime.
- **Persist the env var** to `/root/.bashrc` so it survives across turns.
- The install is pure Python/Triton -- no C++ compilation, takes ~30 seconds.
- **NEVER set `HSA_OVERRIDE_GFX_VERSION`, `PYTORCH_ROCM_ARCH`, `MAX_JOBS`, or `GPU_ARCHS`** -- they are unnecessary and harmful.
- If install fails, use the SDPA fallback instead of commenting out imports.

### Important reminders for ROCm mode
- Always check `pip list` after switching base image to see what's already installed.
- ROCm images are Ubuntu-based and include apt-get. Use apt-get for system packages.
- Do NOT try to install CUDA toolkit, cuDNN, or any nvidia-* system packages.
- If a setup.py or requirements.txt has CUDA version pinning (e.g., torch==2.1.0+cu118), remove the CUDA suffix or skip that package (it's already in the base image).
- **NEVER use `runtest` or `poetryruntest`** -- they are disabled. Use `echo ROCM_ENV_VERIFIED` after verifying.
- **`--help` alone is NOT verification.** You MUST actually run the script with data.
- **Do NOT use `timeout` commands.** Let scripts run after scaling down.
- **ALWAYS scale down epochs/iterations/data BEFORE running EVERY script.** This applies to
  EACH AND EVERY Python script you run, not just the first one. If a project has 3 scripts,
  you must scale down ALL 3 before running them. Use batch-patching early:
  `find /repo -name "*.py" -exec grep -l "epochs" {{}} \; | xargs -I{{}} sed -i "s/'epochs': [0-9]*/'epochs': 2/g" {{}}`
  **Running ANY script with default 2000 epochs = agent timeout = FAILURE.**
- Create mock/dummy input data (images, tensors, etc.) to test the script if real data is not available.
- If the script crashes mid-execution with an error, debug and fix it before declaring success.
- The script MUST produce actual output (not just --help text) to count as verified.

### Handling wandb (Weights & Biases)
Many ML projects use `wandb` for experiment tracking. In the Docker container there is no API key.
If you see `wandb.login(key="FILL IN YOUR W&B KEY")` or similar placeholder in the code:
1. **Comment out or remove** the `wandb.login(...)` call.
2. **Set wandb to offline mode** so it doesn't try to connect:
   ```bash
   export WANDB_MODE=offline && echo 'export WANDB_MODE=offline' >> /root/.bashrc
   ```
   OR modify the `wandb.init()` call in the source code to add `mode="offline"`:
   ```bash
   sed -i 's/wandb.init(/wandb.init(mode="offline", /' /repo/<script>.py
   ```
3. If the script fails with a wandb authentication error, use either approach above.

### Handling hardcoded data paths
Many projects have hardcoded paths for datasets (e.g., `/data/imagenet`, `/ds-sds/images/imagenet`).
When the script fails with `FileNotFoundError` for a data path:
1. **Check if the script accepts a `--data_path` or `--data_dir` argument** to override the path.
2. **If not, use `sed` to replace the hardcoded path** with your dummy data path:
   ```bash
   sed -i "s|/original/hardcoded/path|/tmp/dummy_data|g" /repo/<script>.py
   ```
3. **Always create proper dummy data** that matches the expected structure (e.g., ImageNet-style
   with class subdirectories, JSON files with expected keys, etc.).

### Python 3.12 compatibility
The ROCm `rocm/pytorch:latest` image uses **Python 3.12**. Some older code is incompatible:
- **`import imp`** was removed in Python 3.12. Replace with `import importlib`. If the project uses `imp`, patch it:
  `sed -i 's/import imp/import importlib as imp/' /repo/<file>.py`
  or more precisely fix specific `imp.reload()` -> `importlib.reload()`, etc.
- **`pkg_resources`** may show deprecation warnings. Prefer `importlib.metadata`.
- **`distutils`** was removed. Use `setuptools` equivalents instead.
- **argparse `%i` format in help strings** crashes in Python 3.12 with `TypeError: %i format: a real number is required, not dict`.
  This happens when argparse default values are dicts/lists and the help string uses `%(default)s` with `%i` or `%d`.
  Fix: If `--help` crashes with this error, skip `--help` and run the script directly with arguments.
  Or patch the help string: `sed -i 's/%i/%s/g' /repo/<script>.py`
If a script fails with `ModuleNotFoundError: No module named 'imp'`, this is the cause. Fix it, don't bypass it.

### Common build prerequisites
Some packages require system libraries to build from source:
- **`transformers` with version pins** (needs to build `tokenizers` from Rust): install `apt-get install -y libssl-dev pkg-config` and Rust (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && source ~/.cargo/env`) before `pip install transformers==<version>`.
- If a pinned transformers version fails, try `pip install transformers` (latest) as a fallback.

### Fix tensor shape mismatch warnings in training scripts
When running training verification and you see warnings like:
```
UserWarning: Using a target size (torch.Size([4, 1])) that is different to the input size (torch.Size([4]))
```
This means your test script has a shape mismatch between model output and targets. Fix it BEFORE declaring
success -- it indicates the loss calculation is wrong due to broadcasting. Common fixes:
- If model output is `(batch,)` and target is `(batch, 1)`: add `target = target.squeeze(-1)` or `target = target.view(-1)`
- If model output is `(batch, 1)` and target is `(batch,)`: add `output = output.squeeze(-1)` or `target = target.unsqueeze(-1)`
- Always verify shapes match: `assert output.shape == target.shape, f"Shape mismatch: {{output.shape}} vs {{target.shape}}"`

### Repos requiring external API keys (OpenAI, Gemini, Anthropic, etc.)
Some projects cannot run inference without external API keys (e.g., OpenAI, Gemini, Anthropic).
You will recognize this when you see `openai.api_key`, `genai.configure(api_key=...)`,
`anthropic.Anthropic(api_key=...)`, etc., in the main inference/evaluation scripts.

**In these cases, you CANNOT run the full pipeline.** Instead, verify what you CAN:
1. Verify all imports work: `python -c "from <main_module> import <MainClass>; print('OK')"`
2. Verify GPU/model loading works (load the model but don't call the API endpoint)
3. Check that CLI / argument parsing works: `python <script>.py --help`
4. Once imports, model loading, and GPU access are confirmed, declare:
   ```bash
   echo ROCM_ENV_VERIFIED
   ```
   with a comment in your Thought explaining that the API key is not available in the container
   but the environment is correctly configured.

**Do NOT fabricate or hardcode fake API keys** -- the calls will fail with auth errors, wasting turns.

---

### AMP (Automatic Mixed Precision) API migration for PyTorch 2.x
`torch.cuda.amp` APIs are deprecated in PyTorch 2.x. On `rocm/pytorch:latest` (PyTorch 2.9),
these may emit warnings or fail. The modern replacements are in `torch.amp`:

| Old (deprecated) | New (PyTorch 2.x+) |
|---|---|
| `torch.cuda.amp.autocast()` | `torch.amp.autocast('cuda')` |
| `torch.cuda.amp.GradScaler()` | `torch.amp.GradScaler('cuda')` |
| `@torch.cuda.amp.autocast()` decorator | `with torch.amp.autocast('cuda'):` context |

If a script crashes with `AttributeError: module 'torch.cuda.amp' has no attribute ...`
or shows deprecation warnings about `torch.cuda.amp`, fix with sed:
```bash
sed -i 's/torch\.cuda\.amp\.autocast()/torch.amp.autocast("cuda")/g' /repo/<file>.py
sed -i 's/torch\.cuda\.amp\.GradScaler()/torch.amp.GradScaler("cuda")/g' /repo/<file>.py
```
Note: `torch.amp.autocast('cuda')` works on BOTH NVIDIA CUDA and AMD ROCm with no changes needed.

---

### torch.backends.cudnn crashes on ROCm
Many repos set cuDNN flags for reproducibility:
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
On ROCm, `torch.backends.cudnn` exists but may raise errors when you try to set these attributes,
because ROCm uses MIOpen (not cuDNN) internally.

**Fix**: Guard these flags with a HIP/ROCm check. Use sed to patch the file:
```bash
sed -i 's/torch\.backends\.cudnn\.deterministic\s*=\s*True/if not getattr(torch.version, "hip", None): torch.backends.cudnn.deterministic = True/g' /repo/<file>.py
sed -i 's/torch\.backends\.cudnn\.benchmark\s*=\s*False/if not getattr(torch.version, "hip", None): torch.backends.cudnn.benchmark = False/g' /repo/<file>.py
```
Or write a helper script to patch all files:
```bash
cat > /tmp/fix_cudnn.py << 'PYEOF'
import os, re, sys

def fix_file(path):
    with open(path) as f:
        src = f.read()
    guard = "if not getattr(torch.version, 'hip', None):\\n    "
    new = re.sub(
        r'([ \t]*)(torch\.backends\.cudnn\.(deterministic|benchmark)\s*=\s*(True|False))',
        lambda m: m.group(1) + "if not getattr(torch.version, 'hip', None):\\n" + m.group(1) + "    " + m.group(2),
        src
    )
    if new != src:
        with open(path, 'w') as f:
            f.write(new)
        print(f"Patched: {{path}}")

for root, dirs, files in os.walk(sys.argv[1] if len(sys.argv) > 1 else '/repo'):
    dirs[:] = [d for d in dirs if d not in ('__pycache__', '.git', 'node_modules')]
    for fname in files:
        if fname.endswith('.py'):
            fix_file(os.path.join(root, fname))
PYEOF
python /tmp/fix_cudnn.py /repo
```

---

### flash_attn 2.6.0+ API change: unpad_input returns 5 values
Before `flash_attn` 2.6.0, `unpad_input` returned 4 values:
```python
x, indices, cu_seqlens, max_seqlen = unpad_input(x, attention_mask)
```
From `flash_attn` 2.6.0 onwards it returns **5 values** (added `seqused`):
```python
x, indices, cu_seqlens, max_seqlen, seqused = unpad_input(x, attention_mask)
```
If you see `ValueError: too many values to unpack (expected 4)`, fix with:
```bash
sed -i 's/\(.*\), \(indices\), \(cu_seqlens[^,]*\), \(max_seqlen[^=]*\) = unpad_input(/result = unpad_input(/g' /repo/<file>.py
```
Or the simpler approach — unpack via slicing so it works for both versions:
```bash
grep -rn "= unpad_input(" /repo --include="*.py"
```
Then manually patch each occurrence to:
```python
_result = unpad_input(x, attention_mask)
x, indices, cu_seqlens, max_seqlen = _result[:4]
```
Use sed or write a small patcher:
```bash
sed -i 's/\(^\s*\)\(.*\) = unpad_input(\(.*\))/\1_unpad = unpad_input(\3)\\n\1\2 = _unpad[:4]/g' /repo/<file>.py
```

---

### device_utils.py shim: fixing many .cuda() calls efficiently
When a project has many files with hardcoded `.cuda()`, `device="cuda"`, `torch.cuda.synchronize()`,
`torch.cuda.empty_cache()` etc., the fastest fix is to create a `device_utils.py` shim once,
then use sed to batch-replace the calls.

**Step 1: Create /repo/device_utils.py** (one heredoc, one turn):
```bash
cat > /repo/device_utils.py << 'PYEOF'
import torch

def is_rocm():
    return torch.cuda.is_available() and getattr(torch.version, 'hip', None) is not None

def get_device(idx=None):
    if torch.cuda.is_available():
        return torch.device(f'cuda:{{idx}}' if idx is not None else 'cuda')
    return torch.device('cpu')

def device_count():
    return torch.cuda.device_count() if torch.cuda.is_available() else 0

def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def to_device(tensor, idx=None):
    return tensor.to(get_device(idx))

DEVICE = get_device()
DEVICE_NAME = str(DEVICE)
PYEOF
echo "device_utils.py created at /repo/device_utils.py"
```

**Step 2: Add import to affected files and replace calls** (next turn):
```bash
python /tmp/patch_cuda.py
```
Where `/tmp/patch_cuda.py` is:
```bash
cat > /tmp/patch_cuda.py << 'PYEOF'
import os, re

TARGET_DIR = '/repo/src'  # adjust to where the .py files are

REPLACEMENTS = [
    (r'\.cuda\(\)', '.to(DEVICE)'),
    (r'torch\.cuda\.empty_cache\(\)', 'empty_cache()'),
    (r'torch\.cuda\.synchronize\(\)', 'synchronize()'),
    (r'torch\.cuda\.device_count\(\)', 'device_count()'),
    (r"device\s*=\s*['\"]cuda['\"]", "device=DEVICE_NAME"),
    (r"device\s*=\s*torch\.device\(['\"]cuda['\"]\)", "device=DEVICE"),
]

IMPORT_LINE = "import sys, os; sys.path.insert(0, '/repo'); from device_utils import DEVICE, DEVICE_NAME, get_device, device_count, empty_cache, synchronize\n"

for root, dirs, files in os.walk(TARGET_DIR):
    dirs[:] = [d for d in dirs if d not in ('__pycache__', '.git')]
    for fname in files:
        if not fname.endswith('.py'):
            continue
        path = os.path.join(root, fname)
        with open(path) as f:
            src = f.read()
        new = src
        changed = False
        for pattern, repl in REPLACEMENTS:
            new2 = re.sub(pattern, repl, new)
            if new2 != new:
                changed = True
                new = new2
        if changed:
            if IMPORT_LINE not in new:
                new = IMPORT_LINE + new
            with open(path, 'w') as f:
                f.write(new)
            print(f"Patched: {{path}}")
PYEOF
python /tmp/patch_cuda.py
```

**IMPORTANT:** Only use this approach when many files need patching. For a single file,
just use `sed` directly. Always verify the patched file still imports correctly:
```bash
python -c "import sys; sys.path.insert(0, '/repo'); import <module>; print('OK')"
```

### NEVER disable core functionality to fake success
**CRITICAL: Do NOT comment out, remove, or disable project-critical imports or function calls just to make a script "run".**
For example, if a project's core feature is FlashAttention-based KV cache eviction, do NOT comment out `from flash_attn import ...` and the function calls that use it. That would make the script run but without its core functionality -- this is NOT a valid verification.
Instead:
1. Fix the import issue (e.g., set `FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"` for Triton backend).
2. Use the SDPA fallback (Option 3 above) to provide equivalent functionality.
3. Only if the import is truly optional (e.g., an optimization hint, not core logic), can you add a try/except with a clear warning.

### ENSURE SCRIPTS ALWAYS USE AMD GPU — NEVER allow CPU execution

**The WHOLE POINT of this ROCm environment is to run on AMD GPUs. A script running on CPU = failure.**

**After ANY dependency installation, IMMEDIATELY run this check:**
```bash
python -c "import torch; assert torch.cuda.is_available(), 'FATAL: ROCm GPU not visible to PyTorch — PyTorch was likely overwritten by uv/poetry/conda'; print('GPU OK:', torch.cuda.get_device_name(0))"
```
If this assertion fails, STOP and fix it before proceeding with any script execution.

**Common patterns that cause CPU execution and their fixes:**

1. **Broken device detection logic** — Many scripts use:
   ```python
   device = args.device or "cuda" if torch.cuda.is_available() else "cpu"
   ```
   Due to Python operator precedence, this is evaluated as:
   `device = (args.device or "cuda") if torch.cuda.is_available() else "cpu"`
   When `torch.cuda.is_available()` returns `False` (due to PyTorch being replaced), device becomes `"cpu"`.
   **Fix:** Restore ROCm PyTorch first. The logic itself is fine once `torch.cuda.is_available()` is `True`.

2. **Scripts that hardcode `device='cpu'`** — Some scripts default to CPU:
   ```bash
   grep -rn "device.*=.*['\"]cpu['\"]" /repo/src/ /repo/*.py 2>/dev/null
   ```
   Fix with:
   ```bash
   sed -i "s/device\s*=\s*['\"]cpu['\"]/device = 'cuda' if torch.cuda.is_available() else 'cpu'/g" <file>
   ```

3. **Models loaded without `.to('cuda')` or `.cuda()`** — Look for:
   ```bash
   grep -rn "model = .*\.from_pretrained\|model = .*Model(" /repo/src/ /repo/*.py 2>/dev/null | grep -v "\.cuda\|\.to("
   ```
   If the model is loaded but never moved to GPU, add `.to('cuda')` after the model init.

4. **`CUDA_VISIBLE_DEVICES` set to empty** — Check environment:
   ```bash
   env | grep -i cuda
   env | grep -i hip
   env | grep -i rocr
   ```
   If `CUDA_VISIBLE_DEVICES` is empty or set to a non-existent device, unset it:
   ```bash
   unset CUDA_VISIBLE_DEVICES
   ```

**Checklist before declaring ROCM_ENV_VERIFIED:**
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` prints `True`
- [ ] The script output mentions `cuda` device (not `cpu`)
- [ ] Training/inference produces actual results (loss values, predictions, etc.)
- [ ] If using `uv run`, confirmed it also uses ROCm PyTorch (or switched to `python` directly)
- [ ] Output files are non-empty and contain real results (not just error logs)
"""
    return prompt


def generate_rocm_prompt_section_with_plan():
    """
    Generate a slimmed-down ROCm prompt section for use when an upfront plan exists.

    Omits MANDATORY STEP 1 (Read README) and MANDATORY STEP 2 (Select base image)
    since the planner has already performed this analysis and injected the results
    into the STRATEGIC PLAN section of the system prompt.

    Keeps all technical reference material (flash-attn instructions, banned packages,
    CUDA mapping, code compat rules, etc.) since those are needed during execution.
    """
    preinstalled_str = ""
    for img, pkgs in ROCM_PREINSTALLED_PACKAGES.items():
        preinstalled_str += f"  - `{img}`: {', '.join(pkgs)}\n"

    cuda_mapping_str = ""
    for cuda_pkg, info in CUDA_TO_ROCM_MAPPING.items():
        if info["rocm_package"]:
            cuda_mapping_str += f"  - `{cuda_pkg}` -> Install with: `{info['install_cmd']}`\n"
        else:
            cuda_mapping_str += f"  - `{cuda_pkg}` -> No ROCm equivalent. {info['notes']}\n"

    banned_str = ", ".join(f"`{p}`" for p in BANNED_NVIDIA_PACKAGES[:10]) + ", etc."

    prompt = f"""
## AMD ROCm GPU MODE - CRITICAL INSTRUCTIONS

You are configuring this repository to run on **AMD GPUs with ROCm** (not NVIDIA CUDA).

**CRITICAL RULE: Do NOT use `runtest` or `poetryruntest` in ROCm mode. They are DISABLED.**
**CRITICAL RULE: Output EXACTLY ONE ```bash``` block per response.**

### IMPORTANT: A STRATEGIC PLAN HAS BEEN GENERATED
A comprehensive plan has already analyzed the repository's README, directory structure,
config files, Python imports, and compatibility issues. The plan is included in your prompt.

**DO NOT re-read the README, directory listing, or requirements.txt.**
**DO NOT select a base image — it has already been set based on the plan.**
**Start executing the plan immediately from the first actionable step.**

If the plan includes Python 3.12 compatibility fixes, apply those FIRST.
Then proceed with dependency installation using the filtered package list from the plan.

### CRITICAL: uv / poetry / conda can SHADOW the ROCm PyTorch — ALWAYS verify after install!

When a project uses `uv sync`, `poetry install`, or `conda install`, these tools create their OWN
virtual environment and install their OWN PyTorch from PyPI. The PyPI wheels are CUDA-only or CPU-only —
they do NOT have ROCm support. This SHADOWS the ROCm PyTorch that is pre-installed in the Docker image.

**MANDATORY: After ANY dependency installation (pip install, uv sync, poetry install), ALWAYS verify:**
```bash
python -c "import torch; print('ROCm:', torch.cuda.is_available(), '| Version:', torch.__version__)"
```
If this prints `ROCm: False`, the PyTorch was overwritten. Fix it immediately:

**Fix for `uv` projects:**
```bash
pip install -e . --no-deps && pip install -e packages/*/  --no-deps 2>/dev/null; true
```
Then run scripts with `python` (NOT `uv run`).

**Fix for `poetry` projects:**
```bash
poetry install --no-root
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2 --force-reinstall
```

**Fix for `conda` projects:**
```bash
pip install -r requirements.txt --ignore-installed torch torchvision torchaudio
```

**RULE: After fixing, ALWAYS re-verify `torch.cuda.is_available() == True` before proceeding.**

### Verify by ACTUALLY RUNNING the project's scripts
**Do NOT use `runtest`. It is disabled in ROCm mode.**
**Running `--help` alone is NOT SUFFICIENT. You MUST actually execute the script.**

**MANDATORY PRE-RUN CHECK: Scale down epochs, iterations, and data BEFORE running ANY script.**
Check the plan for specific training parameters that need scaling down.

1. Verify all core imports work:
   `python -c "from <main_package> import <main_class>; print('OK')"`

2. Run the main script with `--help` to understand its interface.

2b. **Choose the right model.** If the script takes a `--model` argument:
   - Check the README (in the plan above) for the recommended model — USE THAT ONE.

3. **CRITICAL: Actually run the script** with mock data and minimal parameters.
   - For training scripts: Use small dummy data, 1-2 epochs, 3-5 max_steps.
   - For inference scripts: Create minimal mock data.
   - **ALWAYS write multiline Python code to a .py file first, then run it.**
   - **Do NOT use multiline `python -c`.**

4. Once the script runs and produces real output ON GPU (not CPU):
   ```bash
   echo ROCM_ENV_VERIFIED
   ```

### Pre-installed packages - DO NOT reinstall these
{preinstalled_str}

### CUDA-to-ROCm package mapping
{cuda_mapping_str}

### BANNED packages - NEVER install these
{banned_str}

### Code compatibility rules
- **torch.cuda API**: Works AS-IS on ROCm. Do NOT change `torch.cuda` calls.
- **torch.device('cuda')**: Works AS-IS. Do NOT change to 'rocm' or 'hip'.
- **nvidia-smi**: Replace with `rocm-smi`.
- **NCCL**: PyTorch distributed still uses 'nccl' as backend name on ROCm.

### PyTorch index URL for ROCm wheels
`pip install <package> --index-url https://download.pytorch.org/whl/rocm6.2`

### FlashAttention for ROCm
If the project requires `flash-attn` or `flash_attn`:

**Do NOT run `pip install flash-attn` from PyPI — CUDA-only wheels.**
**NEVER set HSA_OVERRIDE_GFX_VERSION, PYTORCH_ROCM_ARCH, MAX_JOBS, or GPU_ARCHS — they are NOT needed.**
The ONLY env var needed is `FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"`.

Follow these steps EXACTLY:

1. `git clone https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention`
2. `cd /tmp/flash-attention && FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" python setup.py install`
3. `export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" && echo 'export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"' >> /root/.bashrc`
4. Verify: `FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" python -c "from flash_attn import flash_attn_func; print('OK')"`

If install fails, use PyTorch SDPA fallback (create flash_attn_sdpa_fallback.py shim).

### Python 3.12 compatibility
The ROCm container uses Python 3.12. Check the plan for specific fixes needed.
Common issues: `import imp` (removed), `import distutils` (removed), `from collections import Mapping` (moved).

### Common build prerequisites
- **Old transformers pins** need Rust: `apt-get install -y libssl-dev pkg-config && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y`
- If a pinned version fails, try `pip install <package>` (latest) as fallback.

### Handling wandb
Set `export WANDB_MODE=offline && echo 'export WANDB_MODE=offline' >> /root/.bashrc`

### PYTHONPATH
If scripts import from sibling directories, persist PYTHONPATH:
```bash
echo 'export PYTHONPATH=/repo/src:$PYTHONPATH' >> /root/.bashrc && source /root/.bashrc
```

### NEVER disable core functionality to fake success
Do NOT comment out critical imports just to make a script run.

### ENSURE SCRIPTS ALWAYS USE AMD GPU
After ANY dependency installation: `python -c "import torch; assert torch.cuda.is_available()"`

**Checklist before declaring ROCM_ENV_VERIFIED:**
- [ ] `torch.cuda.is_available()` prints `True`
- [ ] Script output mentions `cuda` device (not `cpu`)
- [ ] Training/inference produces actual results
- [ ] No `GatedRepoError` or `401 Unauthorized` in output — if present, substitute with ungated model and rerun
- [ ] Output files are non-empty and contain real results
"""
    return prompt
