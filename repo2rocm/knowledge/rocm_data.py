"""ROCm domain data tables.

Pure data — no I/O, no logic beyond simple lookups. Anything that needs
heuristics or scoring lives in `repo2rocm/recon/image_select.py`.
"""
from __future__ import annotations

from typing import Any

# ── ROCm Docker images ──────────────────────────────────────────────────────

ROCM_IMAGE_CATALOG: dict[str, dict[str, Any]] = {
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
            "ONNX model inference."
        ),
    },
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

# Packages typically preinstalled in each ROCm image (DO NOT reinstall).
ROCM_PREINSTALLED_PACKAGES: dict[str, list[str]] = {
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

# ── NVIDIA → AMD/ROCm package map ───────────────────────────────────────────

CUDA_TO_ROCM_MAPPING: dict[str, dict[str, Any]] = {
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
            "The Triton backend works out-of-the-box on rocm/pytorch. "
            "ENV: FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE at install AND runtime. "
            "Do NOT pip install flash-attn from PyPI (CUDA-only wheels)."
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
        "notes": "Alias of flash-attn (import name). Same recipe.",
    },
    "nvidia-ml-py": {
        "rocm_package": "pyrsmi",
        "install_cmd": "pip install pyrsmi",
        "notes": "ROCm equivalent of nvidia-ml-py. Also consider the rocm-smi CLI.",
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
        "rocm_package": "bitsandbytes",
        "install_cmd": (
            "git clone https://github.com/ROCm/bitsandbytes.git /tmp/bnb && "
            "cd /tmp/bnb && pip install -e ."
        ),
        "notes": (
            "Use the ROCm fork. The PyPI wheels are CUDA-only. "
            "For quantized model inference/training on ROCm."
        ),
    },
    "xformers": {
        "rocm_package": "xformers",
        "install_cmd": (
            "pip install xformers --index-url https://download.pytorch.org/whl/rocm6.2"
        ),
        "notes": (
            "xformers has ROCm-compatible builds in PyTorch's ROCm wheel index. "
            "If install fails, fall back to torch.nn.functional.scaled_dot_product_attention."
        ),
    },
    "triton": {
        "rocm_package": "triton",
        "install_cmd": "pip install triton",
        "notes": (
            "Triton is preinstalled in ROCm PyTorch images. Do NOT reinstall unless "
            "explicitly required by the project."
        ),
    },
    "deepspeed": {
        "rocm_package": "deepspeed",
        "install_cmd": "pip install deepspeed",
        "notes": (
            "DeepSpeed supports ROCm. Set DS_BUILD_OPS=1 and DS_BUILD_AIO=0 if "
            "building from source."
        ),
    },
    "apex": {
        "rocm_package": "apex",
        "install_cmd": (
            "git clone https://github.com/ROCm/apex /tmp/apex && cd /tmp/apex && "
            "python setup.py install --cpp_ext --cuda_ext"
        ),
        "notes": (
            "Use the ROCm fork. Apex is typically preinstalled in ROCm PyTorch images."
        ),
    },
    "vllm": {
        "rocm_package": "vllm",
        "install_cmd": None,
        "notes": (
            "Do NOT pip install vllm — CUDA-only wheels. Switch base image to "
            "rocm/vllm or rocm/vllm-dev instead."
        ),
    },
    "cupy": {
        "rocm_package": "cupy-rocm-5-0",
        "install_cmd": "pip install cupy-rocm-5-0",
        "notes": (
            "CuPy has ROCm builds via cupy-rocm-* wheels. If incompatible, fall back "
            "to PyTorch tensors or numpy."
        ),
    },
    "pycuda": {
        "rocm_package": None,
        "install_cmd": None,
        "notes": "No direct ROCm equivalent. Use HIP Python bindings or PyTorch.",
    },
    "tensorrt": {
        "rocm_package": None,
        "install_cmd": None,
        "notes": (
            "No ROCm equivalent of TensorRT. Use MIGraphX, ONNXRuntime-ROCm, or "
            "vLLM-ROCm for inference acceleration."
        ),
    },
}

# Packages that should NEVER be installed in ROCm environments.
BANNED_NVIDIA_PACKAGES: list[str] = [
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

# ── CUDA → ROCm code patterns ───────────────────────────────────────────────

CUDA_CODE_PATTERNS: dict[str, dict[str, str]] = {
    "nvidia_smi": {
        "cuda_pattern": "nvidia-smi",
        "rocm_replacement": "rocm-smi",
        "notes": "GPU monitoring tool replacement.",
    },
    "cuda_visible_devices": {
        "cuda_pattern": "CUDA_VISIBLE_DEVICES",
        "rocm_replacement": "HIP_VISIBLE_DEVICES",
        "notes": (
            "CUDA_VISIBLE_DEVICES still works with ROCm PyTorch; HIP_VISIBLE_DEVICES "
            "is the native equivalent."
        ),
    },
    "nccl_backend": {
        "cuda_pattern": "nccl",
        "rocm_replacement": "rccl",
        "notes": (
            "ROCm equivalent is RCCL. PyTorch torch.distributed still uses 'nccl' as "
            "the backend name on both."
        ),
    },
    "torch_cuda_api": {
        "cuda_pattern": "torch.cuda",
        "rocm_replacement": "torch.cuda",
        "notes": (
            "PyTorch ROCm reuses the torch.cuda API. torch.cuda.is_available() "
            "returns True on ROCm. No code changes required for standard usage."
        ),
    },
    "cuda_device": {
        "cuda_pattern": "torch.device('cuda')",
        "rocm_replacement": "torch.device('cuda')",
        "notes": "Keep 'cuda' as the device string in PyTorch ROCm; do NOT use 'rocm'/'hip'.",
    },
}

# ── Supported AMD GPU architectures ─────────────────────────────────────────

SUPPORTED_GPU_ARCHITECTURES: dict[str, str] = {
    "gfx908": "MI100",
    "gfx90a": "MI210/MI250/MI250x",
    "gfx942": "MI300A/MI300X/MI325",
    "gfx950": "MI350/MI355 (ROCm 7.0+)",
    "gfx1030": "Navi21-based consumer GPUs",
    "gfx1100": "Navi31-based consumer GPUs",
    "gfx1200": "Navi44/Navi48-based consumer GPUs (ROCm 6.4.2+)",
    "gfx1151": "Strix Halo/Strix Point APUs (ROCm 7.1+)",
}

# ── Image-scoring signals (for recon.image_select) ──────────────────────────

IMAGE_SIGNALS: dict[str, dict[str, list[str]]] = {
    "sglang": {
        "strong_imports": ["sglang", "sglang_router"],
        "strong_deps": ["sglang", "sglang-router"],
        "readme_patterns": [
            r"sglang", r"SGLang", r"sgl[-_]", r"from sglang",
            r"python -m sglang", r"sgl\.gen", r"sgl\.function",
        ],
    },
    "vllm-dev": {
        "strong_imports": [],
        "strong_deps": [],
        "readme_patterns": [
            r"vLLM\s+(?:fork|development|dev|contribution|custom)",
            r"extends?\s+vllm", r"modif(?:y|ied|ication)\s+.*vllm",
        ],
    },
    "vllm": {
        "strong_imports": ["vllm"],
        "strong_deps": ["vllm", "vllm-flash-attn"],
        "readme_patterns": [
            r"vLLM", r"vllm\.LLM", r"vllm\.SamplingParams",
            r"pip install vllm", r"LLM serving",
        ],
    },
    "jax": {
        "strong_imports": ["jax", "jaxlib", "flax", "optax", "equinox", "orbax"],
        "strong_deps": ["jax", "jaxlib", "flax", "optax", "equinox", "orbax", "chex"],
        "readme_patterns": [r"\bJAX\b", r"\bFlax\b", r"jax\.numpy", r"jnp\."],
    },
    "tensorflow": {
        "strong_imports": ["tensorflow", "keras", "tf_agents"],
        "strong_deps": ["tensorflow", "tensorflow-gpu", "keras", "tf-agents"],
        "readme_patterns": [r"TensorFlow", r"tensorflow", r"tf\.", r"keras\."],
    },
    "onnxruntime": {
        "strong_imports": ["onnxruntime", "onnx"],
        "strong_deps": ["onnxruntime", "onnxruntime-gpu", "onnx", "onnxmltools"],
        "readme_patterns": [r"ONNX\s*Runtime", r"onnxruntime", r"\.onnx\b", r"ONNX model"],
    },
    "pytorch-training": {
        "strong_imports": ["deepspeed", "accelerate", "lightning", "pytorch_lightning"],
        "strong_deps": [
            "deepspeed", "accelerate", "pytorch-lightning",
            "lightning", "fairscale", "colossalai",
        ],
        "readme_patterns": [
            r"[Dd]istributed training", r"DeepSpeed", r"FSDP",
            r"multi.?GPU training", r"accelerate", r"pytorch.?lightning",
        ],
    },
    "megatron": {
        "strong_imports": ["megatron", "megatron_core"],
        "strong_deps": ["megatron-core", "megatron-lm"],
        "readme_patterns": [r"Megatron", r"megatron.?lm", r"tensor.?parallel"],
    },
    "pytorch": {
        "strong_imports": ["torch", "torchvision", "torchaudio"],
        "strong_deps": ["torch", "torchvision", "torchaudio"],
        "readme_patterns": [r"PyTorch", r"\btorch\b", r"\.pt\b", r"\.pth\b"],
    },
}


# ── Convenience lookups ─────────────────────────────────────────────────────


def get_preinstalled_packages(image: str) -> list[str]:
    """Return the preinstalled-package list for a `repo/name[:tag]` string."""
    base = image.split(":")[0]
    return list(ROCM_PREINSTALLED_PACKAGES.get(base, []))


def get_rocm_alternative(package: str) -> dict[str, Any] | None:
    """Return the AMD/ROCm replacement record for a CUDA package, or None."""
    key = package.lower().strip()
    record = CUDA_TO_ROCM_MAPPING.get(key) or CUDA_TO_ROCM_MAPPING.get(key.replace("_", "-"))
    if record is None:
        return None
    return dict(record)


def is_banned_package(package: str) -> bool:
    """True if the package should never be installed on ROCm."""
    name = package.lower().strip().split("[")[0].split("=")[0].split(">")[0].split("<")[0]
    return name in {p.lower() for p in BANNED_NVIDIA_PACKAGES}
