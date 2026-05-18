---
name: hipify_patterns
description: Common .cu / .cpp patterns that hipify-perl cannot auto-translate.
when_to_use: Use when the repo ships .cu or .cuh files and a build fails post-hipify.
---

# Hipify edge cases

`hipify-perl` and `hipify-clang` handle most CUDA→HIP translation automatically. The
remainder is human work. Here's the residual list.

## Warp size

CUDA assumes warp=32. AMD CDNA/RDNA is warp=64.

```cpp
// Bad on AMD:
__shared__ float buf[32];
// Better:
__shared__ float buf[warpSize];  // hipBuiltin
```

Hardcoded `32` constants in shared-mem sizing, butterfly reductions, and tile
dimensions need manual review.

## __launch_bounds__

`__launch_bounds__(256)` is portable; `__launch_bounds__(256, 8)` (second arg
= min blocks per SM) is heuristic on NVIDIA and meaningless on AMD. Strip the
second arg.

## Cooperative groups

```cpp
auto tile = cooperative_groups::tiled_partition<32>(block);
// On AMD prefer:
auto tile = cooperative_groups::tiled_partition<64>(block);
```

## cuBLAS / cuDNN handles

`cublasCreate` → `hipblasCreate`. `cudnn*` → `miopen*`. hipify handles most of
this, but the API surface diverges for newer cuDNN ops; check `MIOpen`'s docs.

## NCCL ↔ RCCL

`#include <nccl.h>` → `#include <rccl/rccl.h>`. Otherwise the API is identical.

## CUB / Thrust → rocPRIM / hipCUB / rocThrust

`#include <cub/cub.cuh>` → `#include <hipcub/hipcub.hpp>` and use `hipcub::` namespace.

## Build flags

`-arch=sm_80` → `--offload-arch=gfx942` (MI300X) or `gfx90a` (MI250X).
