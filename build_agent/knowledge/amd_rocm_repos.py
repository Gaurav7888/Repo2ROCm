"""
AMD ROCm Ecosystem Catalog for Repo2ROCm.

Each entry maps a CUDA/NVIDIA concept (import name, pip package, or library)
to its AMD-native counterpart.

Structure of each entry
------------------------
  nvidia_pkg     : pip package name(s) or import name on NVIDIA
  amd_name       : canonical AMD library name
  category       : "math" | "dl" | "inference" | "vision" | "rendering" | "tooling" | "ml"
  nvidia_equiv   : brief label of what it replaces on the NVIDIA side
  use_case       : what the library does / when you need it
  status         : "preinstalled" | "pip" | "build_from_source" | "hipify"
  github         : canonical AMD GitHub URL
  install_cmd    : exact shell command to obtain the library
  import_triggers: Python import names or file patterns that signal this library is needed
  config_triggers: strings in config / requirements files that signal this library is needed
  pypi_index     : extra-index-url to pass to pip when applicable
  notes          : caveats, version requirements, tips
"""

from typing import Dict, List, Any

AMD_ROCM_REPO_CATALOG: List[Dict[str, Any]] = [

    # ═══════════════════════════════════════════════════════════════════════
    # TOOLING — CUDA → HIP translation
    # ═══════════════════════════════════════════════════════════════════════
    {
        "key": "hipify",
        "amd_name": "HIPIFY",
        "category": "tooling",
        "nvidia_equiv": "N/A — translates CUDA source to HIP source",
        "use_case": (
            "Automatically converts CUDA C++ source files (.cu, .cuh) to portable HIP C++ "
            "when a repo ships custom CUDA kernels that need to be ported to ROCm. "
            "Handles API renaming (cudaMalloc→hipMalloc, etc.) and include paths."
        ),
        "status": "preinstalled",
        "github": "https://github.com/ROCm/HIPIFY",
        "install_cmd": "hipify-perl <file.cu>  OR  hipify-clang <file.cu>  # pre-installed in ROCm images",
        "import_triggers": [],
        "config_triggers": [".cu", ".cuh", "cuda_kernel", "cuda_extension"],
        "pypi_index": "",
        "notes": (
            "Use hipify-perl for quick bulk translation; hipify-clang for semantically "
            "correct translation of complex CUDA code. After translation, search for any "
            "remaining 'cuda' strings with: grep -r 'cuda' /repo --include='*.h' --include='*.cpp'."
        ),
    },

    # ═══════════════════════════════════════════════════════════════════════
    # MATH LIBRARIES — Linear Algebra
    # ═══════════════════════════════════════════════════════════════════════
    {
        "key": "rocblas",
        "amd_name": "rocBLAS",
        "category": "math",
        "nvidia_equiv": "cuBLAS",
        "use_case": (
            "AMD's native BLAS (Basic Linear Algebra Subprograms) library. "
            "Provides GEMM, TRSM, and other dense linear algebra operations on AMD GPUs. "
            "Powers PyTorch's torch.matmul on ROCm."
        ),
        "status": "preinstalled",
        "github": "https://github.com/ROCm/rocm-libraries (projects/rocblas)",
        "install_cmd": "# Pre-installed in all ROCm Docker images. Already linked via PyTorch.",
        "import_triggers": ["cublas", "cublaslt"],
        "config_triggers": ["cublas", "cublaslt"],
        "pypi_index": "",
        "notes": (
            "Direct Python access via torch.backends.cuda.matmul (works on ROCm). "
            "If code calls cuBLAS C API directly, use hipBLAS wrapper instead."
        ),
    },
    {
        "key": "hipblas",
        "amd_name": "hipBLAS",
        "category": "math",
        "nvidia_equiv": "cuBLAS (HIP-portable wrapper)",
        "use_case": (
            "HIP-portable wrapper around rocBLAS (on AMD) and cuBLAS (on NVIDIA). "
            "Use when C++/C source calls cuBLAS APIs directly — replace `cublas` headers "
            "with `hipblas` and rename function calls."
        ),
        "status": "preinstalled",
        "github": "https://github.com/ROCm/rocm-libraries (projects/hipblas)",
        "install_cmd": "# Pre-installed in ROCm images. Header: #include <hipblas/hipblas.h>",
        "import_triggers": ["cublas"],
        "config_triggers": ["cublas"],
        "pypi_index": "",
        "notes": "Python bindings: pip install hipblaslt (part of ROCm distro).",
    },
    {
        "key": "hipblaslt",
        "amd_name": "hipBLASLt",
        "category": "math",
        "nvidia_equiv": "cuBLASLt / CUTLASS",
        "use_case": (
            "High-performance GEMM with epilogue support, mixed precision (FP8/BF16), "
            "and batched operations. Used by flash-attention, vLLM, and large-model training. "
            "Enables TunableOp to benchmark rocBLAS vs hipBLASLt kernels."
        ),
        "status": "preinstalled",
        "github": "https://github.com/ROCm/rocm-libraries (projects/hipblaslt)",
        "install_cmd": "# Pre-installed in ROCm images.",
        "import_triggers": ["cublaslt", "cutlass"],
        "config_triggers": ["cublaslt", "hipblaslt"],
        "pypi_index": "",
        "notes": (
            "Enable via PyTorch TunableOp for auto-selection between rocBLAS and hipBLASLt: "
            "export PYTORCH_TUNABLEOP_ENABLED=1"
        ),
    },
    {
        "key": "composablekernel",
        "amd_name": "Composable Kernel (CK)",
        "category": "math",
        "nvidia_equiv": "CUTLASS",
        "use_case": (
            "AMD's template-based GPU kernel library for GEMM, convolution, attention, "
            "and reduction. The default backend for flash-attention on ROCm. "
            "Used internally by MIOpen and hipBLASLt."
        ),
        "status": "preinstalled",
        "github": "https://github.com/ROCm/rocm-libraries (projects/composablekernel)",
        "install_cmd": "# Internal AMD library, pre-installed in ROCm images.",
        "import_triggers": ["cutlass"],
        "config_triggers": ["composable_kernel", "ck_tile"],
        "pypi_index": "",
        "notes": (
            "When flash-attention uses the CK backend: "
            "FLASH_ATTENTION_TRITON_AMD_ENABLE='' python setup.py install  (CK backend, default)"
        ),
    },

    # ═══════════════════════════════════════════════════════════════════════
    # MATH LIBRARIES — Sparse / FFT / Random / Solver
    # ═══════════════════════════════════════════════════════════════════════
    {
        "key": "rocfft",
        "amd_name": "rocFFT / hipFFT",
        "category": "math",
        "nvidia_equiv": "cuFFT",
        "use_case": "Fast Fourier Transform on AMD GPUs. Used by signal processing, diffusion models, and spectral networks.",
        "status": "preinstalled",
        "github": "https://github.com/ROCm/rocm-libraries (projects/rocfft, projects/hipfft)",
        "install_cmd": "# Pre-installed. HIP wrapper: #include <hipfft/hipfft.h>",
        "import_triggers": ["cufft", "torch.fft"],
        "config_triggers": ["cufft", "hipfft"],
        "pypi_index": "",
        "notes": "torch.fft on ROCm routes through rocFFT automatically.",
    },
    {
        "key": "rocsparse",
        "amd_name": "rocSPARSE / hipSPARSE",
        "category": "math",
        "nvidia_equiv": "cuSPARSE",
        "use_case": "Sparse matrix operations (SpMM, SpMV) on AMD GPUs. Used by GNN frameworks, sparse transformers.",
        "status": "preinstalled",
        "github": "https://github.com/ROCm/rocm-libraries (projects/rocsparse, projects/hipsparse)",
        "install_cmd": "# Pre-installed.",
        "import_triggers": ["cusparse", "torch.sparse"],
        "config_triggers": ["cusparse", "hipsparse"],
        "pypi_index": "",
        "notes": "torch.sparse operations on ROCm use rocSPARSE backend.",
    },
    {
        "key": "rocsolver",
        "amd_name": "rocSOLVER / hipSOLVER",
        "category": "math",
        "nvidia_equiv": "cuSOLVER",
        "use_case": "Dense linear algebra solvers (LU, QR, SVD, eigendecomposition) on AMD GPUs.",
        "status": "preinstalled",
        "github": "https://github.com/ROCm/rocm-libraries (projects/rocsolver, projects/hipsolver)",
        "install_cmd": "# Pre-installed.",
        "import_triggers": ["cusolver", "torch.linalg"],
        "config_triggers": ["cusolver"],
        "pypi_index": "",
        "notes": "torch.linalg.svd etc. route through rocSOLVER on AMD.",
    },
    {
        "key": "rocrand",
        "amd_name": "rocRAND / hipRAND",
        "category": "math",
        "nvidia_equiv": "cuRAND",
        "use_case": "GPU random number generation. Used by dropout, VAEs, diffusion noise schedules.",
        "status": "preinstalled",
        "github": "https://github.com/ROCm/rocm-libraries (projects/rocrand, projects/hiprand)",
        "install_cmd": "# Pre-installed.",
        "import_triggers": ["curand"],
        "config_triggers": ["curand"],
        "pypi_index": "",
        "notes": "torch.randn on ROCm uses rocRAND internally.",
    },
    {
        "key": "hipcub",
        "amd_name": "hipCUB / rocPRIM",
        "category": "math",
        "nvidia_equiv": "CUB / Thrust",
        "use_case": (
            "Device-wide parallel primitives (sort, reduce, scan, histogram). "
            "hipCUB is the HIP-portable wrapper; rocPRIM is the AMD-native implementation."
        ),
        "status": "preinstalled",
        "github": "https://github.com/ROCm/rocm-libraries (projects/hipcub, projects/rocprim)",
        "install_cmd": "# Pre-installed.",
        "import_triggers": ["cub", "thrust"],
        "config_triggers": ["cub", "thrust", "hipcub"],
        "pypi_index": "",
        "notes": (
            "For custom CUDA kernels using cub:: or thrust:: namespaces, include "
            "<hipcub/hipcub.hpp> and use hipcub:: namespace instead."
        ),
    },
    {
        "key": "rocthrust",
        "amd_name": "rocThrust",
        "category": "math",
        "nvidia_equiv": "Thrust",
        "use_case": "High-level parallel algorithms (sort, transform, reduce) on AMD GPUs.",
        "status": "preinstalled",
        "github": "https://github.com/ROCm/rocm-libraries (projects/rocthrust)",
        "install_cmd": "# Pre-installed.",
        "import_triggers": ["thrust"],
        "config_triggers": ["thrust"],
        "pypi_index": "",
        "notes": "Include <thrust/...> still works in HIP — the ROCm Thrust port is API-compatible.",
    },
    {
        "key": "hiptensor",
        "amd_name": "hipTensor",
        "category": "math",
        "nvidia_equiv": "cuTENSOR",
        "use_case": "High-performance tensor contraction library on AMD GPUs.",
        "status": "preinstalled",
        "github": "https://github.com/ROCm/rocm-libraries (projects/hiptensor)",
        "install_cmd": "# Pre-installed in ROCm images.",
        "import_triggers": ["cutensor"],
        "config_triggers": ["cutensor", "hiptensor"],
        "pypi_index": "",
        "notes": "",
    },
    {
        "key": "hipsparselt",
        "amd_name": "hipSPARSELt",
        "category": "math",
        "nvidia_equiv": "cuSPARSELt",
        "use_case": "Structured sparsity GEMM acceleration (2:4 sparsity pattern).",
        "status": "preinstalled",
        "github": "https://github.com/ROCm/rocm-libraries (projects/hipsparselt)",
        "install_cmd": "# Pre-installed.",
        "import_triggers": ["cusparselt"],
        "config_triggers": ["cusparselt", "hipsparselt"],
        "pypi_index": "",
        "notes": "Used by optimized sparse transformer inference.",
    },
    {
        "key": "rocwmma",
        "amd_name": "rocWMMA",
        "category": "math",
        "nvidia_equiv": "WMMA (Warp Matrix Multiply-Accumulate) / nvcuda::wmma",
        "use_case": "Matrix core (tensor core) intrinsics for AMD CDNA/RDNA GPUs.",
        "status": "preinstalled",
        "github": "https://github.com/ROCm/rocm-libraries (projects/rocwmma)",
        "install_cmd": "# Pre-installed.",
        "import_triggers": ["wmma", "nvcuda"],
        "config_triggers": ["wmma", "rocwmma"],
        "pypi_index": "",
        "notes": "Used when custom CUDA kernels call nvcuda::wmma operations directly.",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # DEEP LEARNING PRIMITIVES
    # ═══════════════════════════════════════════════════════════════════════
    {
        "key": "miopen",
        "amd_name": "MIOpen",
        "category": "dl",
        "nvidia_equiv": "cuDNN",
        "use_case": (
            "AMD's deep-learning primitives library: convolutions, batch-norm, pooling, "
            "RNN, attention, and activation functions. The ROCm equivalent of cuDNN. "
            "PyTorch, TensorFlow, and MXNet on ROCm call MIOpen automatically."
        ),
        "status": "preinstalled",
        "github": "https://github.com/ROCm/rocm-libraries (projects/miopen)",
        "install_cmd": "# Pre-installed in all ROCm Docker images.",
        "import_triggers": ["cudnn", "torch.backends.cudnn"],
        "config_triggers": ["cudnn", "miopen"],
        "pypi_index": "",
        "notes": (
            "torch.backends.cudnn.* flags work on ROCm — they route to MIOpen. "
            "However, guard with: if not getattr(torch.version, 'hip', None): to avoid "
            "flags that MIOpen doesn't support."
        ),
    },

    # ═══════════════════════════════════════════════════════════════════════
    # INFERENCE & GRAPH COMPILERS
    # ═══════════════════════════════════════════════════════════════════════
    {
        "key": "migraphx",
        "amd_name": "MIGraphX",
        "category": "inference",
        "nvidia_equiv": "TensorRT",
        "use_case": (
            "AMD's graph compiler for ML inference. Takes ONNX or TF models, applies "
            "operator fusion, arithmetic simplification, and dead-code elimination, "
            "then emits optimized kernels via MIOpen, rocBLAS, or custom HIP. "
            "Python API: import migraphx; p = migraphx.parse_onnx('model.onnx'); "
            "p.compile(migraphx.get_target('gpu'))."
        ),
        "status": "preinstalled",
        "github": "https://github.com/ROCm/AMDMIGraphX",
        "install_cmd": "# Pre-installed in ROCm inference images. pip install migraphx",
        "import_triggers": ["tensorrt", "trt", "onnxruntime"],
        "config_triggers": ["tensorrt", "migraphx"],
        "pypi_index": "",
        "notes": (
            "When repo uses ONNX Runtime on NVIDIA (onnxruntime-gpu with TRT backend), "
            "replace with: pip install onnxruntime-rocm (ROCm EP) or use MIGraphX directly."
        ),
    },

    # ═══════════════════════════════════════════════════════════════════════
    # COMPUTER VISION
    # ═══════════════════════════════════════════════════════════════════════
    {
        "key": "mivisionx",
        "amd_name": "MIVisionX",
        "category": "vision",
        "nvidia_equiv": "NVIDIA Video Codec SDK / VPI / DALI",
        "use_case": (
            "Comprehensive computer vision and machine intelligence toolkit. "
            "Implements Khronos OpenVX and extensions for hardware-accelerated CV ops. "
            "Includes NNEF/ONNX model compiler, video decode (VCN hardware), "
            "and AMD's Inference Engine for rapid CV inference deployment."
        ),
        "status": "preinstalled",
        "github": "https://github.com/ROCm/MIVisionX",
        "install_cmd": "# Pre-installed in ROCm images. Python: from amd import pyAMDVX",
        "import_triggers": ["nvvl", "dali", "nvidia_dali"],
        "config_triggers": ["mivisionx", "openvx"],
        "pypi_index": "",
        "notes": "Use for video loading pipelines, optical flow, or OpenVX-based CV graphs.",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # 3D RENDERING & GAUSSIAN SPLATTING
    # ═══════════════════════════════════════════════════════════════════════
    {
        "key": "amd_gsplat",
        "amd_name": "amd_gsplat (ROCm/gsplat)",
        "category": "rendering",
        "nvidia_equiv": "diff-gaussian-rasterization + simple-knn (Inria 3DGS)",
        "use_case": (
            "AMD's ROCm port of the gsplat library for GPU-accelerated 3D Gaussian Splatting. "
            "Replaces BOTH simple-knn AND diff-gaussian-rasterization submodules that are "
            "found in most Gaussian Splatting repos (3DGS, LangSplat, ReferSplat, etc.). "
            "Provides optimized CUDA→HIP rasterization kernels and KNN lookup."
        ),
        "status": "pip",
        "github": "https://github.com/ROCm/gsplat",
        "install_cmd": (
            "# ROCm 7.0.0:\n"
            "pip install amd_gsplat --extra-index-url=https://pypi.amd.com/rocm-7.0.0/simple/\n"
            "# ROCm 6.4.3:\n"
            "pip install amd_gsplat --extra-index-url=https://pypi.amd.com/rocm-6.4.3/simple/"
        ),
        "import_triggers": [
            "diff_gaussian_rasterization", "simple_knn", "gaussian_rasterization",
            "simple-knn", "diff-gaussian-rasterization",
        ],
        "config_triggers": [
            "diff-gaussian-rasterization", "simple-knn", "gaussian_splatting",
            "diff_gaussian_rasterization", "simple_knn",
        ],
        "pypi_index": "https://pypi.amd.com/rocm-7.0.0/simple/",
        "notes": (
            "CRITICAL for any 3DGS-based repo (3D Gaussian Splatting, LangSplat, ReferSplat, "
            "GaussianGrouping, etc.). Do NOT try to compile simple-knn or diff-gaussian-rasterization "
            "from source on ROCm — they use CUDA-specific headers that require extensive patching. "
            "amd_gsplat is the maintained, production-ready drop-in. "
            "After install, patch imports: replace 'from diff_gaussian_rasterization import ...' "
            "with 'from gsplat import ...' and adjust function signatures per the gsplat API."
        ),
    },

    # ═══════════════════════════════════════════════════════════════════════
    # ML ACCELERATION LIBRARIES
    # ═══════════════════════════════════════════════════════════════════════
    {
        "key": "flash_attn",
        "amd_name": "flash-attention (ROCm fork)",
        "category": "ml",
        "nvidia_equiv": "flash-attn (Tri Dao)",
        "use_case": (
            "Memory-efficient attention for transformers. Critical for LLMs, ViTs, "
            "and any repo that imports flash_attn. Two ROCm backends: "
            "Composable Kernel (CK, default, recommended) and Triton."
        ),
        "status": "build_from_source",
        "github": "https://github.com/Dao-AILab/flash-attention (uses ROCm CK backend automatically)",
        "install_cmd": (
            "pip install ninja\n"
            "git clone https://github.com/Dao-AILab/flash-attention.git && cd flash-attention\n"
            "# CK backend (recommended, default on ROCm):\n"
            "python setup.py install\n"
            "# Triton backend (alternative):\n"
            "# FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE python setup.py install"
        ),
        "import_triggers": ["flash_attn", "flash_attention"],
        "config_triggers": ["flash-attn", "flash_attn", "flash_attention"],
        "pypi_index": "",
        "notes": (
            "Build time: ~20-40 min on first install (compiles CK kernels). "
            "Set MAX_JOBS=4 to limit parallel compilation if OOM. "
            "Requires ROCm 6.0+ and PyTorch ≥ 2.3 with HIP support."
        ),
    },
    {
        "key": "bitsandbytes",
        "amd_name": "bitsandbytes (ROCm/bitsandbytes fork)",
        "category": "ml",
        "nvidia_equiv": "bitsandbytes",
        "use_case": (
            "INT8 and 4-bit quantization (QLoRA, GGUF loading, LLM.int8). "
            "Used by transformers, peft, and any repo doing quantized inference or fine-tuning."
        ),
        "status": "build_from_source",
        "github": "https://github.com/ROCm/bitsandbytes",
        "install_cmd": (
            "git clone --recurse https://github.com/ROCm/bitsandbytes && cd bitsandbytes\n"
            "git checkout rocm_enabled\n"
            "cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH=\"gfx90a;gfx942\" -S .\n"
            "make -j4\n"
            "pip install ."
        ),
        "import_triggers": ["bitsandbytes", "bnb"],
        "config_triggers": ["bitsandbytes", "bitsandbytes-gpu"],
        "pypi_index": "",
        "notes": (
            "Adjust gfx arch for your GPU: MI200=gfx90a, MI300=gfx942, RX7900=gfx1100. "
            "Check with: rocminfo | grep 'gfx'. "
            "After install verify: python -c 'import bitsandbytes; print(bitsandbytes.__version__)'"
        ),
    },
    {
        "key": "xformers",
        "amd_name": "xFormers (ROCm/xformers fork)",
        "category": "ml",
        "nvidia_equiv": "xformers (Meta)",
        "use_case": (
            "Efficient transformer building blocks: memory-efficient attention, "
            "FMHA (Fused Multi-Head Attention), sparse attention, and operator benchmarking. "
            "Required by diffusers, stable-diffusion repos, and many vision transformers."
        ),
        "status": "build_from_source",
        "github": "https://github.com/ROCm/xformers",
        "install_cmd": (
            "pip install xformers --extra-index-url=https://download.pytorch.org/whl/rocm6.3\n"
            "# OR build from source for latest ROCm:\n"
            "git clone https://github.com/ROCm/xformers && cd xformers\n"
            "pip install -e . --no-build-isolation"
        ),
        "import_triggers": ["xformers", "xformers.ops"],
        "config_triggers": ["xformers"],
        "pypi_index": "https://download.pytorch.org/whl/rocm6.3",
        "notes": (
            "Experimental ROCm support. If memory-efficient attention fails, "
            "fall back to: XFORMERS_DISABLED=1 (uses PyTorch's native scaled_dot_product_attention). "
            "PyTorch 2.0+ native SDPA usually works as a drop-in."
        ),
    },
    {
        "key": "triton",
        "amd_name": "Triton (AMD ROCm backend)",
        "category": "ml",
        "nvidia_equiv": "OpenAI Triton (CUDA backend)",
        "use_case": (
            "Python-based GPU kernel DSL. Triton kernels (.triton files or @triton.jit decorated "
            "functions) compile to AMD GCN/CDNA via the ROCm backend. Required by many "
            "custom attention kernels, FlashInfer, and vLLM."
        ),
        "status": "pip",
        "github": "https://github.com/triton-lang/triton (built-in ROCm support since v2.1)",
        "install_cmd": (
            "pip install triton  # ROCm backend is built-in from v2.1+\n"
            "# Verify: python -c \"import triton; print(triton.__version__)\""
        ),
        "import_triggers": ["triton", "triton.language"],
        "config_triggers": ["triton"],
        "pypi_index": "",
        "notes": (
            "Triton kernels written for CUDA usually work unchanged on ROCm — "
            "the backend handles the target-specific lowering. "
            "Verify with TRITON_INTERPRET=1 for debugging."
        ),
    },
    {
        "key": "deepspeed",
        "amd_name": "DeepSpeed (ROCm-compatible)",
        "category": "ml",
        "nvidia_equiv": "DeepSpeed (Microsoft)",
        "use_case": (
            "Distributed training with ZeRO, pipeline parallelism, and mixed-precision. "
            "Works on ROCm via the PyTorch HIP backend. Flash-attention and "
            "custom CUDA ops inside DeepSpeed may need patching."
        ),
        "status": "pip",
        "github": "https://github.com/microsoft/DeepSpeed (ROCm supported upstream)",
        "install_cmd": (
            "DS_BUILD_OPS=0 pip install deepspeed  # skip CUDA op builds\n"
            "# OR with AMD-specific ops:\n"
            "pip install deepspeed --extra-index-url=https://download.pytorch.org/whl/rocm6.3"
        ),
        "import_triggers": ["deepspeed"],
        "config_triggers": ["deepspeed"],
        "pypi_index": "",
        "notes": (
            "Use DS_BUILD_OPS=0 to skip CUDA-specific op compilation (fused Adam, etc.). "
            "The PyTorch-native ops work on ROCm. Check DeepSpeed docs for ROCm-verified configs."
        ),
    },

    # ═══════════════════════════════════════════════════════════════════════
    # PERFORMANCE TUNING
    # ═══════════════════════════════════════════════════════════════════════
    {
        "key": "tunableop",
        "amd_name": "PyTorch TunableOp",
        "category": "tooling",
        "nvidia_equiv": "cuBLAS auto-tuning / CUDA graph optimization",
        "use_case": (
            "Automatically benchmarks and selects the fastest BLAS kernel (rocBLAS vs hipBLASLt) "
            "for each GEMM shape in the model at runtime. Persists the selected kernels to a "
            "CSV file for reuse. Typically gives 5-15% throughput improvement for LLMs on MI300X."
        ),
        "status": "preinstalled",
        "github": "https://github.com/pytorch/pytorch (built-in since PyTorch 2.3)",
        "install_cmd": (
            "# Enable at runtime (no install needed):\n"
            "export PYTORCH_TUNABLEOP_ENABLED=1\n"
            "export PYTORCH_TUNABLEOP_TUNING=1\n"
            "export PYTORCH_TUNABLEOP_FILENAME=tunableop_results.csv\n"
            "# Offline tuning (run once, reuse results):\n"
            "export PYTORCH_TUNABLEOP_ENABLED=1\n"
            "export PYTORCH_TUNABLEOP_TUNING=0\n"
            "export PYTORCH_TUNABLEOP_FILENAME=tunableop_results.csv"
        ),
        "import_triggers": ["torch"],
        "config_triggers": ["tunableop"],
        "pypi_index": "",
        "notes": (
            "TunableOp uses rocBLAS and hipBLASLt as candidate libraries. "
            "Blog: https://rocm.blogs.amd.com/artificial-intelligence/pytorch-tunableop/README.html"
        ),
    },
    {
        "key": "rocroller",
        "amd_name": "rocRoller",
        "category": "tooling",
        "nvidia_equiv": "Cutlass kernel generator",
        "use_case": (
            "AMD's code generator for highly optimized GPU kernels (GEMM, convolution). "
            "Powers the auto-generated kernels in hipBLASLt and Tensile. "
            "Used internally — not typically invoked by ML application code."
        ),
        "status": "preinstalled",
        "github": "https://github.com/ROCm/rocm-libraries (shared/rocroller)",
        "install_cmd": "# Internal AMD tool, pre-installed.",
        "import_triggers": [],
        "config_triggers": [],
        "pypi_index": "",
        "notes": "Used by AMD engineers to generate tuned GEMM kernels for specific GFX targets.",
    },
]


# ── Lookup helpers ────────────────────────────────────────────────────────────

def get_relevant_amd_repos(
    import_names: List[str],
    config_strings: List[str],
) -> List[Dict[str, Any]]:
    """
    Given a list of Python import names and config file strings found in the repo,
    return the subset of AMD_ROCM_REPO_CATALOG entries that are relevant.

    A catalog entry is considered relevant if any of its `import_triggers` appear
    in `import_names` or any of its `config_triggers` appear in `config_strings`.
    """
    import_lower = {s.lower().replace("-", "_") for s in import_names}
    config_lower = {s.lower() for s in config_strings}

    relevant: List[Dict[str, Any]] = []
    seen_keys: set = set()

    for entry in AMD_ROCM_REPO_CATALOG:
        if entry["key"] in seen_keys:
            continue
        matched = False
        for trigger in entry.get("import_triggers", []):
            if trigger.lower().replace("-", "_") in import_lower:
                matched = True
                break
        if not matched:
            for trigger in entry.get("config_triggers", []):
                if trigger.lower() in config_lower:
                    matched = True
                    break
        if matched:
            relevant.append(entry)
            seen_keys.add(entry["key"])

    return relevant


def get_entry_by_key(key: str) -> Dict[str, Any]:
    for entry in AMD_ROCM_REPO_CATALOG:
        if entry["key"] == key:
            return entry
    return {}
