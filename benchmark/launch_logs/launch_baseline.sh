#!/bin/bash
set -e
: "${AMD_LLM_API_KEY:?AMD_LLM_API_KEY must be exported before launch}"
cd /home/upandey/rocm/benchmark
exec python3 -u -m harness.main run \
  --tasks-json harness/cache/tasks_kernel_subset.json \
  --runs-dir runs_mode1 \
  --db runs_mode1/progress.sqlite \
  --reports-dir reports_mode1 \
  --approaches repo2rocm \
  --mode env \
  --gpus 0,1,2 \
  --timeout 5400
