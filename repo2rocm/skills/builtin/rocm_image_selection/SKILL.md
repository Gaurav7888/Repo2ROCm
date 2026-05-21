---
name: rocm_image_selection
description: Authoritative catalog of ROCm Docker images, when to pick each, preinstalled packages
when_to_use: When choosing a base image, switching base images, or interpreting an install failure that suggests the image is wrong
paths: ["**/Dockerfile*", "**/requirements*.txt", "**/pyproject.toml", "**/setup.py"]
allowed_tools: ["DockerHubTags"]
---

# ROCm Image Selection

## Catalog

| Workload | Image | Default tag | When to use |
|---|---|---|---|
| SGLang serving | `rocm/sgl-dev` | `main` | Repo IS SGLang-based or uses `sglang.launch_server` |
| vLLM development | `rocm/vllm-dev` | `main` | Repo IS a vLLM fork or contributes to vLLM internals |
| vLLM serving | `rocm/vllm` | `latest` | Repo USES vLLM as a library |
| JAX | `rocm/jax` | `latest` | Imports `jax`, `flax`, `optax`; not also heavy on PyTorch |
| TensorFlow | `rocm/tensorflow` | `latest` | Imports `tensorflow` / `keras` |
| ONNX Runtime | `rocm/onnxruntime` | `latest` | Inference with `.onnx` models |
| Distributed training | `rocm/pytorch-training` | `latest` | DeepSpeed / FSDP / Megatron / torchrun |
| Megatron-LM | `rocm/megatron-lm` | `latest` | Repo IS Megatron or imports `megatron.core` |
| General PyTorch | `rocm/pytorch` | `latest` | Default fallback for torch-based repos |

## Selection rules

1. If the repo's primary framework matches one of the specialized images, prefer that.
2. With multiple frameworks present, pick the **most specialized** image that covers the
   primary workload (heavy training → `pytorch-training`; serving → `vllm`).
3. Always confirm the tag exists by calling `DockerHubTags(image="<repo>")` BEFORE
   `ChangeBaseImage`. The `before_change_base_image` hook enforces this.
4. Prefer pinned tags over `:latest` for reproducibility.

## Preinstalled packages (DO NOT reinstall)

| Image | Already there |
|---|---|
| `rocm/pytorch` | torch, torchvision, torchaudio, numpy, apex, triton, pillow, pyyaml, typing-extensions, cmake, ninja |
| `rocm/pytorch-training` | + deepspeed |
| `rocm/vllm` | + vllm, transformers, tokenizers, safetensors |
| `rocm/vllm-dev` | + ray, aiohttp, fastapi, uvicorn |
| `rocm/sgl-dev` | + sglang, vllm, flashinfer |
| `rocm/jax` | jax, jaxlib, numpy, scipy, opt-einsum |
| `rocm/tensorflow` | tensorflow, keras, tensorboard, protobuf |
| `rocm/onnxruntime` | onnxruntime, onnx, protobuf |
| `rocm/megatron-lm` | + megatron-core |

Strip these from any `requirements.txt` before `Download`.
