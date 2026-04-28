# AMD ROCm Ecosystem Reference

> **Purpose:** This document is the planner's reference for AMD-native alternatives to
> NVIDIA/CUDA libraries. When the planner detects CUDA-specific imports or packages,
> it consults this guide to recommend the correct AMD counterpart, its install command,
> and any caveats.  
> **Source repos:** [ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries) ·
> [ROCm/gsplat](https://github.com/ROCm/gsplat) ·
> [ROCm/HIPIFY](https://github.com/ROCm/HIPIFY)  
> **Last updated:** 2026-04 (ROCm 7.2.2)

---

## SECTION 1 — Tooling: CUDA → HIP Translation

### HIPIFY

| Field | Value |
|---|---|
| **NVIDIA equiv** | N/A — translates CUDA source to HIP |
| **GitHub** | https://github.com/ROCm/HIPIFY |
| **Status** | Pre-installed in all ROCm Docker images |

**When to use:** Any repo that ships custom `.cu` or `.cuh` files (CUDA kernels, custom ops)
must translate them to HIP before they compile on ROCm. HIPIFY automates most of the renaming.

**Two modes:**
- `hipify-perl file.cu` — fast, regex-based. Good for bulk translation.
- `hipify-clang file.cu` — semantically correct, handles complex macros. Preferred for
  production.

**Workflow:**
```bash
# Find all CUDA source files
find /repo -name "*.cu" -o -name "*.cuh" | head -20

# Quick translation (perl mode)
for f in $(find /repo -name "*.cu"); do
    hipify-perl "$f" > "${f%.cu}.hip"
done

# Verify no CUDA strings remain
grep -r "cuda" /repo --include="*.cpp" --include="*.h" | grep -v "ROCm\|hip\|#"
```

**After translation — always check for:**
- `#include <cuda_runtime.h>` → `#include <hip/hip_runtime.h>`
- `cudaMalloc` / `cudaFree` → `hipMalloc` / `hipFree`
- `__global__` kernel launches: `<<<grid, block>>>` stays the same in HIP
- `cub::` namespace → `hipcub::` namespace

---

## SECTION 2 — Math Libraries (pre-installed in ROCm images)

All libraries in this section are **pre-installed** in the standard ROCm Docker images
(`rocm/pytorch`, `rocm/pytorch-training`, etc.). You do NOT need to install them manually.
PyTorch on ROCm routes through them automatically for standard operations.

### rocBLAS / hipBLAS — Dense Matrix Multiply

| NVIDIA equiv | cuBLAS |
|---|---|
| **GitHub** | https://github.com/ROCm/rocm-libraries (projects/rocblas, projects/hipblas) |

- `rocBLAS` is the AMD-native implementation of BLAS (GEMM, TRSM, etc.).
- `hipBLAS` is the HIP-portable wrapper (same API as cuBLAS, works on both AMD and NVIDIA).
- **Trigger:** imports `cublas`, `cublaslt`, or any direct BLAS C API calls.
- **PyTorch:** `torch.matmul` and `torch.nn.Linear` use rocBLAS automatically on ROCm.
- **C++ code:** Replace `#include <cublas_v2.h>` with `#include <hipblas/hipblas.h>`.

### hipBLASLt — High-Performance GEMM with Epilogue

| NVIDIA equiv | cuBLASLt / CUTLASS |
|---|---|
| **GitHub** | https://github.com/ROCm/rocm-libraries (projects/hipblaslt) |

- Enables FP8/BF16 mixed-precision GEMM, batched operations, and flexible epilogues.
- Used by flash-attention (CK backend), vLLM, and PyTorch TunableOp.
- **Activate TunableOp** to let PyTorch auto-select between rocBLAS and hipBLASLt:
  ```bash
  export PYTORCH_TUNABLEOP_ENABLED=1
  ```

### Composable Kernel (CK) — Template GPU Kernel Library

| NVIDIA equiv | CUTLASS |
|---|---|
| **GitHub** | https://github.com/ROCm/rocm-libraries (projects/composablekernel) |

- AMD's template-based GPU kernel library for GEMM, convolution, attention, reduction.
- **Default backend for flash-attention on ROCm** — install flash-attention and it uses CK automatically.
- Used internally by MIOpen and hipBLASLt.

### rocFFT / hipFFT — Fast Fourier Transform

| NVIDIA equiv | cuFFT |
|---|---|
| **GitHub** | https://github.com/ROCm/rocm-libraries (projects/rocfft, projects/hipfft) |

- `torch.fft.*` on ROCm uses rocFFT automatically.
- Replace `#include <cufft.h>` with `#include <hipfft/hipfft.h>`.

### rocSPARSE / hipSPARSE — Sparse Linear Algebra

| NVIDIA equiv | cuSPARSE |
|---|---|
| **GitHub** | https://github.com/ROCm/rocm-libraries (projects/rocsparse, projects/hipsparse) |

- SpMM, SpMV, sparse GEMM operations.
- `torch.sparse` operations on ROCm use rocSPARSE.

### rocSOLVER / hipSOLVER — Dense Solvers

| NVIDIA equiv | cuSOLVER |
|---|---|
| **GitHub** | https://github.com/ROCm/rocm-libraries (projects/rocsolver, projects/hipsolver) |

- LU, QR, SVD, eigendecomposition.
- `torch.linalg.*` on ROCm uses rocSOLVER.

### rocRAND / hipRAND — Random Number Generation

| NVIDIA equiv | cuRAND |
|---|---|
| **GitHub** | https://github.com/ROCm/rocm-libraries (projects/rocrand, projects/hiprand) |

- `torch.randn`, `torch.rand` on ROCm use rocRAND internally.

### hipCUB / rocPRIM — Parallel Primitives

| NVIDIA equiv | CUB / Thrust |
|---|---|
| **GitHub** | https://github.com/ROCm/rocm-libraries (projects/hipcub, projects/rocprim) |

- Sort, reduce, scan, histogram on GPU.
- Custom CUDA kernels using `cub::` → replace with `hipcub::`.
- `#include <cub/cub.cuh>` → `#include <hipcub/hipcub.hpp>`

### rocThrust — High-Level Parallel Algorithms

| NVIDIA equiv | Thrust |
|---|---|
| **GitHub** | https://github.com/ROCm/rocm-libraries (projects/rocthrust) |

- `thrust::sort`, `thrust::transform`, etc. — API-compatible port.

### hipTensor — Tensor Contraction

| NVIDIA equiv | cuTENSOR |
|---|---|
| **GitHub** | https://github.com/ROCm/rocm-libraries (projects/hiptensor) |

### hipSPARSELt — Structured Sparsity GEMM

| NVIDIA equiv | cuSPARSELt |
|---|---|
| **GitHub** | https://github.com/ROCm/rocm-libraries (projects/hipsparselt) |

- 2:4 structured sparsity acceleration for inference.

### rocWMMA — Matrix Core Intrinsics

| NVIDIA equiv | nvcuda::wmma / tensor core intrinsics |
|---|---|
| **GitHub** | https://github.com/ROCm/rocm-libraries (projects/rocwmma) |

- When CUDA kernels use `nvcuda::wmma::` for tensor core ops.

---

## SECTION 3 — Deep Learning Primitives

### MIOpen — Deep Learning Primitives

| NVIDIA equiv | cuDNN |
|---|---|
| **GitHub** | https://github.com/ROCm/rocm-libraries (projects/miopen) |
| **Status** | Pre-installed |
| **Docs** | https://rocm.docs.amd.com/projects/MIOpen/en/docs-7.2.0/ |

**What it provides:** Convolutions, batch normalization, pooling, RNN/LSTM, attention primitives,
activation functions — all with auto-tuning infrastructure and support for BF16/FP16.

**PyTorch:** All `torch.nn.Conv2d`, `torch.nn.BatchNorm`, etc. call MIOpen on ROCm automatically.

**Critical caveat:** Code that sets `torch.backends.cudnn.*` flags will work, but some flags
(e.g., `benchmark=True`) behave slightly differently on MIOpen. Always guard with:
```python
if not getattr(torch.version, 'hip', None):
    torch.backends.cudnn.benchmark = True
```

---

## SECTION 4 — Inference & Graph Compilers

### MIGraphX — Graph Compiler for ML Inference

| NVIDIA equiv | TensorRT |
|---|---|
| **GitHub** | https://github.com/ROCm/AMDMIGraphX |
| **Status** | Pre-installed in ROCm inference images |
| **Docs** | https://rocm.docs.amd.com/projects/AMDMIGraphX/ |

**What it provides:** ONNX / TensorFlow graph intake → operator fusion, constant folding,
dead-code elimination → optimized code via MIOpen / rocBLAS / custom HIP kernels.
CPU fallback via DNNL/ZenDNN.

**Python API:**
```python
import migraphx
p = migraphx.parse_onnx("model.onnx")
p.compile(migraphx.get_target("gpu"))
result = p.run({"input": input_tensor})
```

**When to use:** Repos that use TensorRT for inference (TensorRT C++ API or `tensorrt` Python package).
Replace with MIGraphX for AMD, or use ONNX Runtime with the ROCm EP.

---

## SECTION 5 — Computer Vision

### MIVisionX — Computer Vision Toolkit

| NVIDIA equiv | NVIDIA Video Codec SDK / VPI / DALI |
|---|---|
| **GitHub** | https://github.com/ROCm/MIVisionX |
| **Status** | Pre-installed in ROCm images |
| **Docs** | https://rocm.docs.amd.com/projects/MIVisionX/ |

**What it provides:**
- Khronos OpenVX hardware-accelerated CV ops
- Neural network model compiler (ONNX, NNEF) for embedded deployment
- AMD Inference Engine for rapid CV model deployment
- VCN hardware video decode

**When to use:** Repos that use DALI (NVIDIA Data Loading Library), NvVL (NVIDIA Video Loader),
or OpenVX for GPU-accelerated data preprocessing / video pipelines.

---

## SECTION 6 — 3D Rendering & Gaussian Splatting

### amd_gsplat — AMD Gaussian Splatting Library

| NVIDIA equiv | diff-gaussian-rasterization + simple-knn (Inria 3DGS submodules) |
|---|---|
| **GitHub** | https://github.com/ROCm/gsplat |
| **PyPI** | `pip install amd_gsplat --extra-index-url=https://pypi.amd.com/rocm-7.0.0/simple/` |

**CRITICAL — use this for any Gaussian Splatting repo.**

Most 3DGS repos (3D Gaussian Splatting, LangSplat, ReferSplat, GaussianGrouping,
GaussianEditor, Splat-SLAM, etc.) include two CUDA-specific submodules:
- `submodules/diff-gaussian-rasterization` — the differentiable rasterizer
- `submodules/simple-knn` — the K-Nearest Neighbours for Gaussian densification

Both contain CUDA-specific headers (`cuda_runtime.h`, `cub`, `cooperative_groups`)
that are very difficult to patch correctly. **Do not try to compile them from source on ROCm.**

Instead, install `amd_gsplat` which provides equivalent, production-ready HIP implementations:

```bash
# ROCm 7.x:
pip install amd_gsplat --extra-index-url=https://pypi.amd.com/rocm-7.0.0/simple/

# ROCm 6.4.x:
pip install amd_gsplat --extra-index-url=https://pypi.amd.com/rocm-6.4.3/simple/

# Verify:
pip show amd_gsplat
python -c "import gsplat; print(gsplat.__version__)"
```

**Patch imports in the repo code:**
```python
# Original (NVIDIA):
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from simple_knn._C import distCUDA2

# AMD replacement (gsplat API):
from gsplat import rasterization
# Note: gsplat has a different API — consult https://github.com/ROCm/gsplat/tree/main/examples
```

**System requirements:**
- ROCm 6.4.3 or 7.0.0
- PyTorch 2.6 (ROCm 6.4.3) or 2.8 (ROCm 7.0.0)
- AMD Instinct MI300X (also works on MI200-series)

---

## SECTION 7 — ML Acceleration Libraries (require installation)

### flash-attention on ROCm

| NVIDIA equiv | flash-attn (Tri Dao) |
|---|---|
| **GitHub** | https://github.com/Dao-AILab/flash-attention (ROCm CK backend built-in) |
| **Docs** | https://rocm.blogs.amd.com/artificial-intelligence/flash-attention/README.html |

**Two backends on ROCm:**
1. **Composable Kernel (CK)** — default, recommended. Better performance on MI300X.
2. **Triton** — alternative, set `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`.

```bash
pip install ninja packaging
git clone https://github.com/Dao-AILab/flash-attention.git && cd flash-attention

# CK backend (recommended):
MAX_JOBS=4 python setup.py install

# Triton backend:
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE MAX_JOBS=4 python setup.py install
```

**Also available (April 2026):** FlashInfer on ROCm — production LLM prefill/decode kernels
for CDNA3/4 with FP8 support. See https://rocm.blogs.amd.com/artificial-intelligence/flashinfer-release2/

### bitsandbytes on ROCm

| NVIDIA equiv | bitsandbytes (Tim Dettmers) |
|---|---|
| **GitHub** | https://github.com/ROCm/bitsandbytes |

Required for QLoRA, LLM.int8(), GPTQ loading, and any repo using 4-bit quantization.

```bash
git clone --recurse https://github.com/ROCm/bitsandbytes && cd bitsandbytes
git checkout rocm_enabled

# Check your GPU arch: rocminfo | grep -i "gfx"
# MI200=gfx90a  MI300=gfx942  RX7900=gfx1100
cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="gfx90a;gfx942" -S .
make -j4 && pip install .
```

### xFormers on ROCm

| NVIDIA equiv | xformers (Meta) |
|---|---|
| **GitHub** | https://github.com/ROCm/xformers |

Required by diffusers (Stable Diffusion, SDXL), many ViT repos.

```bash
# Easiest (pre-built for ROCm 6.3):
pip install xformers --extra-index-url=https://download.pytorch.org/whl/rocm6.3

# Build from source (for latest ROCm):
git clone https://github.com/ROCm/xformers && cd xformers
pip install -e . --no-build-isolation
```

**Fallback:** If xFormers fails, PyTorch 2.0+ `scaled_dot_product_attention` (SDPA) is
a compatible drop-in for most uses. Set `XFORMERS_DISABLED=1` to force PyTorch SDPA.

### Triton on ROCm

| NVIDIA equiv | OpenAI Triton (CUDA backend) |
|---|---|
| **GitHub** | https://github.com/triton-lang/triton |

ROCm backend is built-in from Triton v2.1+. Standard `pip install triton` works.
Triton kernels written for CUDA usually compile unchanged on ROCm.

```bash
pip install triton
python -c "import triton; print(triton.__version__)"
```

### DeepSpeed on ROCm

| NVIDIA equiv | DeepSpeed (Microsoft) |
|---|---|
| **GitHub** | https://github.com/microsoft/DeepSpeed |

ROCm is supported upstream. Skip CUDA-specific ops:

```bash
DS_BUILD_OPS=0 pip install deepspeed
# Check: python -c "import deepspeed; deepspeed.ops.adam.FusedAdam"
```

---

## SECTION 8 — Performance Tuning

### PyTorch TunableOp

| NVIDIA equiv | cuBLAS auto-tuning |
|---|---|
| **Built into** | PyTorch 2.3+ (no installation needed) |
| **Docs** | https://rocm.blogs.amd.com/artificial-intelligence/pytorch-tunableop/README.html |

TunableOp benchmarks rocBLAS and hipBLASLt kernels for every GEMM shape encountered
and persists the winner to a CSV for reuse. Typically 5–15% throughput improvement on MI300X.

```bash
# Online tuning (first run, slow — tunes and saves):
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_TUNING=1
export PYTORCH_TUNABLEOP_FILENAME=/tmp/tunableop_results.csv
python train.py ...  # runs slowly on first pass; saves winners to CSV

# Offline reuse (subsequent runs, fast — loads pre-tuned results):
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_TUNING=0
export PYTORCH_TUNABLEOP_FILENAME=/tmp/tunableop_results.csv
python train.py ...  # loads from CSV, no tuning overhead
```

**Offline tuning workflow** (tune once, deploy fast):
```bash
# Step 1: Record GEMM shapes without tuning
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_TUNING=0
export PYTORCH_TUNABLEOP_RECORD_UNTUNED=1
python train.py --steps 10  # just record shapes

# Step 2: Tune offline
export PYTORCH_TUNABLEOP_TUNING=1
python -c "import torch; torch.cuda.tunable.tune_gemm_in_file('untuned.csv', 'tuned.csv')"

# Step 3: Deploy with pre-tuned results
export PYTORCH_TUNABLEOP_FILENAME=tuned.csv
export PYTORCH_TUNABLEOP_TUNING=0
```

---

## SECTION 9 — Quick Reference: NVIDIA → AMD Mapping Table

| NVIDIA / CUDA | AMD / ROCm | Category | Status |
|---|---|---|---|
| cuBLAS | rocBLAS / hipBLAS | Math | Pre-installed |
| cuBLASLt / CUTLASS | hipBLASLt | Math | Pre-installed |
| CUTLASS templates | Composable Kernel (CK) | Math | Pre-installed |
| cuFFT | rocFFT / hipFFT | Math | Pre-installed |
| cuSPARSE | rocSPARSE / hipSPARSE | Math | Pre-installed |
| cuSOLVER | rocSOLVER / hipSOLVER | Math | Pre-installed |
| cuRAND | rocRAND / hipRAND | Math | Pre-installed |
| CUB | hipCUB | Math | Pre-installed |
| Thrust | rocThrust | Math | Pre-installed |
| cuTENSOR | hipTensor | Math | Pre-installed |
| cuSPARSELt | hipSPARSELt | Math | Pre-installed |
| nvcuda::wmma | rocWMMA | Math | Pre-installed |
| cuDNN | MIOpen | DL primitives | Pre-installed |
| TensorRT | MIGraphX | Inference | Pre-installed |
| DALI / NvVL | MIVisionX | Vision | Pre-installed |
| diff-gaussian-rasterization + simple-knn | amd_gsplat | 3D Rendering | `pip install amd_gsplat --extra-index-url=https://pypi.amd.com/rocm-7.0.0/simple/` |
| flash-attn | flash-attention (CK backend) | ML | Build from source |
| bitsandbytes | ROCm/bitsandbytes | ML | Build from source |
| xformers | ROCm/xformers | ML | pip / build from source |
| triton (CUDA) | triton (ROCm backend) | ML | `pip install triton` |
| DeepSpeed CUDA ops | DeepSpeed (DS_BUILD_OPS=0) | ML | `pip install deepspeed` |
| cuBLAS auto-tune | PyTorch TunableOp | Performance | env vars only |
| CUDA source files | HIP (via HIPIFY) | Tooling | Pre-installed |

---

## SECTION 10 — Domain-Specific Recommendations

### Gaussian Splatting (3DGS) repos

**Signal imports/packages:** `diff_gaussian_rasterization`, `simple_knn`, `gaussian_splatting`

**Action:**
1. `pip install amd_gsplat --extra-index-url=https://pypi.amd.com/rocm-7.0.0/simple/`
2. Patch `from diff_gaussian_rasterization import ...` → `from gsplat import rasterization`
3. Do NOT try to pip-install or build the original submodules — they use CUDA-only CUB/cooperative_groups headers

### LLM Training / Fine-Tuning repos

**Signal imports:** `flash_attn`, `bitsandbytes`, `deepspeed`, `xformers`

**Action priority:**
1. Flash-attention: build from source (CK backend) — ~30 min
2. bitsandbytes: build from ROCm fork — ~10 min
3. xformers: pip from ROCm wheel or build from source
4. Enable TunableOp for GEMM tuning

### Vision / Stable Diffusion repos

**Signal imports:** `xformers`, `diffusers`, `accelerate`

**Action:**
1. `pip install xformers --extra-index-url=https://download.pytorch.org/whl/rocm6.3`
2. Fallback: `XFORMERS_DISABLED=1` → PyTorch SDPA
3. `diffusers` and `accelerate` work unchanged on ROCm

### ONNX Inference repos

**Signal imports:** `onnxruntime`, `tensorrt`

**AMD alternatives:**
- `pip install onnxruntime-rocm` — ONNX Runtime with ROCm Execution Provider
- MIGraphX for graph-compiled inference (`pip install migraphx`)

### Custom CUDA Kernel repos

**Signal files:** `.cu`, `.cuh`, `setup.py` with `CUDAExtension`

**Action:**
1. Run HIPIFY on all `.cu` / `.cuh` files
2. Replace `CUDAExtension` with `HIPExtension` in `setup.py`
3. Check for `cub::`, `thrust::`, `cooperative_groups::` → replace with HIP equivalents
4. Rebuild with: `python setup.py install`
