"""
Seed error patterns and rules for the KB — extracted from the static
knowledge in rocm_knowledge.py and common ROCm failure modes.

Run once to populate a fresh KB, or call seed_if_empty() at startup.
"""

from __future__ import annotations

from storage.models import ErrorPattern, Fix, Rule, RuleSource, ErrorSeverity
from storage.kb_store import KBStore


def seed_if_empty(kb: KBStore):
    """Seed the KB only if it has no error patterns yet."""
    stats = kb.get_stats()
    if stats.get("error_patterns_count", 0) > 0:
        return
    seed_error_patterns(kb)
    seed_rules(kb)


def seed_error_patterns(kb: KBStore):
    """Populate KB with known ROCm error patterns and their fixes."""

    _patterns = [
        {
            "error_class": "BANNED_NVIDIA_PACKAGE",
            "description": "Attempted to install a CUDA-only NVIDIA package on ROCm",
            "regex": r"(?:nvidia-cuda-runtime|nvidia-cublas|nvidia-cufft|nvidia-curand|nvidia-cusolver|nvidia-cusparse|nvidia-cudnn|nvidia-nccl|nvidia-nvtx|nvidia-nvjitlink)",
            "fix_commands": None,
            "fix_desc": "Skip installation — these packages are incompatible with ROCm",
        },
        {
            "error_class": "FLASH_ATTN_CUDA_WHEEL",
            "description": "Installed CUDA-only flash-attn from PyPI instead of building with Triton backend",
            "regex": r"(?:flash.attn.*undefined symbol|flash.attn.*cuda.*not found|ImportError.*flash_attn.*cuda)",
            "fix_commands": [
                "pip uninstall flash-attn -y",
                "git clone https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention",
                "cd /tmp/flash-attention && FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE python setup.py install",
                "export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE",
                "echo 'export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE' >> /root/.bashrc",
            ],
            "fix_desc": "Build flash-attn from source with Triton AMD backend",
        },
        {
            "error_class": "HIPBLAS_NOT_INITIALIZED",
            "description": "hipBLAS library not properly initialized — GPU not accessible",
            "regex": r"(?:hipErrorNotInitialized|hipBLAS.*not initialized|HIP.*error.*initialization)",
            "fix_commands": [
                "python -c \"import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')\"",
            ],
            "fix_desc": "Verify GPU accessibility; may need container restart with --device flags",
        },
        {
            "error_class": "NO_BINARY_FOR_GPU",
            "description": "Compiled binary doesn't support the target GPU architecture",
            "regex": r"(?:hipErrorNoBinaryForGpu|no kernel image.*available|unsupported gpu architecture)",
            "fix_commands": [
                "echo 'Need to rebuild with correct GPU architecture target (gfx90a/gfx942)'",
                "export HSA_OVERRIDE_GFX_VERSION=9.0.0",
            ],
            "fix_desc": "Set HSA_OVERRIDE_GFX_VERSION or rebuild targeting correct gfx arch",
        },
        {
            "error_class": "TORCH_CUDA_NOT_AVAILABLE",
            "description": "PyTorch reports CUDA not available on ROCm system",
            "regex": r"(?:torch\.cuda\.is_available\(\).*False|AssertionError.*CUDA.*not available|RuntimeError.*CUDA.*not available)",
            "fix_commands": [
                "python -c \"import torch; print(torch.__version__); print('HIP:', torch.version.hip)\"",
                "ls /opt/rocm/lib/libamdhip64.so* 2>/dev/null || echo 'ROCm libs not found'",
            ],
            "fix_desc": "Check PyTorch ROCm build and ROCm library availability",
        },
        {
            "error_class": "ABI_MISMATCH",
            "description": "C++ ABI mismatch between PyTorch and compiled extension",
            "regex": r"(?:undefined symbol.*_ZN|CXX11.*ABI|abi.*incompatible|GLIBCXX.*not found)",
            "fix_commands": [
                "python -c \"import torch; print('CXX11 ABI:', torch._C._GLIBCXX_USE_CXX11_ABI)\"",
            ],
            "fix_desc": "Rebuild extension with matching CXX11 ABI flag",
        },
        {
            "error_class": "MODULE_NOT_FOUND",
            "description": "Python module not found",
            "regex": r"ModuleNotFoundError:\s+No module named\s+'([^']+)'",
            "fix_commands": None,
            "fix_desc": "Install the missing module or check PYTHONPATH",
        },
        {
            "error_class": "PIP_VERSION_CONFLICT",
            "description": "pip dependency resolution conflict",
            "regex": r"(?:ERROR:.*pip.*resolver|ResolutionImpossible|Cannot install.*because|conflicting dependencies)",
            "fix_commands": None,
            "fix_desc": "Resolve version conflicts using pipdeptree and version pinning",
        },
        {
            "error_class": "NVIDIA_SMI_NOT_FOUND",
            "description": "Code calls nvidia-smi which doesn't exist on ROCm",
            "regex": r"(?:nvidia-smi.*not found|command not found.*nvidia|FileNotFoundError.*nvidia.smi)",
            "fix_commands": [
                "ln -sf $(which rocm-smi) /usr/local/bin/nvidia-smi 2>/dev/null || echo 'Use rocm-smi instead'",
            ],
            "fix_desc": "Replace nvidia-smi calls with rocm-smi",
        },
        {
            "error_class": "CUDNN_NOT_FOUND",
            "description": "Code references cuDNN which is NVIDIA-specific",
            "regex": r"(?:cudnn.*not found|libcudnn.*cannot find|torch\.backends\.cudnn.*error)",
            "fix_commands": None,
            "fix_desc": "cuDNN calls may need to be guarded with HIP detection check",
        },
        {
            "error_class": "TRITON_COMPILE_ERROR",
            "description": "Triton kernel compilation failed on ROCm",
            "regex": r"(?:triton.*CompilationError|triton.*compile.*fail|amdgcn.*error)",
            "fix_commands": None,
            "fix_desc": "Check Triton kernel for AMD-incompatible patterns (warp size, shared memory)",
        },
        {
            "error_class": "HIPIFY_ERROR",
            "description": "hipify-clang conversion produced errors",
            "regex": r"(?:hipify.*error|hip.*conversion.*fail|cuda_runtime\.h.*not found after hipify)",
            "fix_commands": None,
            "fix_desc": "Manual intervention needed for hipify failures",
        },
        {
            "error_class": "OOM_GPU",
            "description": "GPU out of memory",
            "regex": r"(?:OutOfMemoryError|CUDA out of memory|HIP out of memory|torch\.cuda\.OutOfMemoryError)",
            "fix_commands": [
                "echo 'Reduce batch size, sequence length, or model size'",
            ],
            "fix_desc": "Reduce memory consumption parameters",
        },
        {
            "error_class": "SETUPTOOLS_BUILD_FAIL",
            "description": "C/C++ extension build failed during pip install",
            "regex": r"(?:error:.*command.*gcc.*failed|error:.*cl\.exe.*failed|fatal error|subprocess.*CalledProcessError.*setup\.py)",
            "fix_commands": [
                "apt-get update -qq && apt-get install -y -qq build-essential",
            ],
            "fix_desc": "Install build tools and retry",
        },
        {
            "error_class": "WARP_SIZE_MISMATCH",
            "description": "Code assumes NVIDIA warp size of 32, AMD uses 64",
            "regex": r"(?:warp.?size.*32|WARP_SIZE.*=.*32|__shfl.*warp)",
            "fix_commands": None,
            "fix_desc": "AMD GPUs use warp (wavefront) size 64 instead of NVIDIA's 32",
        },
    ]

    for p in _patterns:
        pattern = ErrorPattern(
            signature=p["error_class"].lower(),
            error_class=p["error_class"],
            description=p["description"],
            regex_pattern=p["regex"],
            evidence_count=10,
            confidence=0.8,
        )
        pattern_id = kb.add_error_pattern(pattern, source_attempt="seed")

        if p["fix_commands"]:
            fix = Fix(
                description=p["fix_desc"],
                commands=p["fix_commands"],
                success_rate=0.7,
                evidence_count=5,
            )
            fix_id = kb.add_fix(fix, source_attempt="seed")
            kb.link_error_to_fix(pattern_id, fix_id)


def seed_rules(kb: KBStore):
    """Populate KB with initial executable rules from expert knowledge."""

    _rules = [
        {
            "id": "rule_flash_attn_triton_amd",
            "condition": {
                "package_needed": "flash-attn",
                "rocm_mode": True,
            },
            "action": [
                {"type": "skip_pip", "package": "flash-attn"},
                {"type": "bash", "command": "git clone https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention"},
                {"type": "bash", "command": "cd /tmp/flash-attention && FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE python setup.py install"},
                {"type": "env", "key": "FLASH_ATTENTION_TRITON_AMD_ENABLE", "value": "TRUE"},
            ],
            "confidence": 0.9,
        },
        {
            "id": "rule_ban_nvidia_packages",
            "condition": {
                "package_matches": r"nvidia-cuda-|nvidia-cublas|nvidia-cufft|nvidia-curand|nvidia-cusolver|nvidia-cusparse|nvidia-cudnn|nvidia-nccl|nvidia-nvtx|nvidia-nvjitlink",
                "rocm_mode": True,
            },
            "action": [
                {"type": "skip_install", "reason": "NVIDIA-only package incompatible with ROCm"},
            ],
            "confidence": 0.99,
        },
        {
            "id": "rule_wandb_offline",
            "condition": {
                "code_contains": "wandb.login",
            },
            "action": [
                {"type": "env", "key": "WANDB_MODE", "value": "offline"},
            ],
            "confidence": 0.95,
        },
        {
            "id": "rule_nvidia_smi_to_rocm_smi",
            "condition": {
                "code_contains": "nvidia-smi",
                "rocm_mode": True,
            },
            "action": [
                {"type": "bash", "command": "ln -sf $(which rocm-smi) /usr/local/bin/nvidia-smi 2>/dev/null || true"},
            ],
            "confidence": 0.9,
        },
        {
            "id": "rule_bitsandbytes_rocm",
            "condition": {
                "package_needed": "bitsandbytes",
                "rocm_mode": True,
            },
            "action": [
                {"type": "bash", "command": "pip install bitsandbytes-rocm || pip install bitsandbytes --no-deps"},
            ],
            "confidence": 0.7,
        },
        {
            "id": "rule_sdpa_fallback",
            "condition": {
                "error_pattern": r"flash.*attn.*(?:error|fail|not found)",
                "rocm_mode": True,
            },
            "action": [
                {"type": "env", "key": "ATTN_BACKEND", "value": "sdpa"},
                {"type": "guidance", "text": "Flash attention failed. Try SDPA fallback: set ATTN_BACKEND=sdpa or use torch.nn.functional.scaled_dot_product_attention"},
            ],
            "confidence": 0.75,
        },
        {
            "id": "rule_hsa_override_gfx",
            "condition": {
                "error_pattern": r"hipErrorNoBinaryForGpu|no kernel image",
                "rocm_mode": True,
            },
            "action": [
                {"type": "env", "key": "HSA_OVERRIDE_GFX_VERSION", "value": "9.0.0"},
                {"type": "guidance", "text": "GPU arch mismatch. HSA_OVERRIDE_GFX_VERSION set to 9.0.0; adjust if targeting different arch"},
            ],
            "confidence": 0.7,
        },
    ]

    for r in _rules:
        rule = Rule(
            id=r["id"],
            condition=r["condition"],
            action=r["action"],
            confidence=r["confidence"],
            source=RuleSource.EXPERT.value,
            evidence_count=10,
            success_rate=r["confidence"],
            success_count=int(10 * r["confidence"]),
            failure_count=int(10 * (1 - r["confidence"])),
        )
        kb.add_rule(rule, source_attempt="seed")
