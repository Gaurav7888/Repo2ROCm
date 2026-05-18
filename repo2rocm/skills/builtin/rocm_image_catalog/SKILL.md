---
name: rocm_image_catalog
description: Authoritative catalog of ROCm Docker images, with selection criteria.
when_to_use: Use when picking a base image for an unknown repo, or when planner_recommended image fails to build.
allowed_tools: []
---

# ROCm Image Catalog

| Workload | Image | When to use |
|---|---|---|
| SGLang serving | `rocm/sgl-dev` | repo is an SGLang fork or uses `sglang.launch_server` |
| vLLM development | `rocm/vllm-dev` | repo IS vLLM or a vLLM fork |
| vLLM serving | `rocm/vllm` | repo uses vLLM as a library |
| JAX | `rocm/jax` | imports `jax`, `flax`, `optax` |
| TensorFlow | `rocm/tensorflow` | imports `tensorflow` |
| ONNX Runtime | `rocm/onnxruntime` | uses `onnxruntime`, `onnx.export`, or `onnxruntime-genai` |
| Distributed training | `rocm/pytorch-training` | DeepSpeed / FSDP / Megatron / torchrun |
| Megatron-LM | `rocm/megatron-lm` | repo IS Megatron or directly imports `megatron.core` |
| General PyTorch | `rocm/pytorch` | default fallback for any torch-based repo |

## Selection rules

1. If repo's primary framework is one of the specialized ones, pick that image.
2. If multiple frameworks present, pick the **most specialized** that covers the
   primary workload. Heavy training → `pytorch-training`; serving → `vllm`.
3. Always confirm the tag exists by calling `DockerHubTags(image="<repo>")` BEFORE
   `ChangeBaseImage`. The `before_change_base_image` hook enforces this.
4. Avoid `:latest` for reproducibility; pick the most recent stable tag (e.g. `rocm6.2_*`).
