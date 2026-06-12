#!/usr/bin/env bash
# Easy list — not verified (11 repos), mode env, sequential
set -euo pipefail
cd /home/gsarkar/Repo2ROCm
API_KEY="${AMD_LLM_API_KEY:-eabfc51a20ba432d90f51158431e022f}"

run_one() {
  local full_name="$1"
  local sha="$2"
  echo "========================================"
  echo "Starting: ${full_name} @ ${sha}"
  echo "========================================"
  python3 -u build_agent/main.py \
    --full_name "${full_name}" \
    --sha "${sha}" \
    --root_path . \
    --llm "claude-sonnet-4" \
    --rocm \
    --mode env \
    --api-key "${API_KEY}"
}

run_one "allenai/understanding_mcqa" "8b1aea4c1bce5e5295f93b79a899a79c2b1fc626"
run_one "Cohere-Labs-Community/m-rewardbench" "708d3bde7bbf110d5297bb0feb6729400ee7ade1"
run_one "sarahmart/HARDMath" "9ade062a80b7d475666b11ae13a2fb0d7b7f0341"
run_one "ZhiningLiu1998/SelfElicit" "2fa9d1d3ab5a14e583de16fa974dfcd617d861f8"
run_one "FYYFU/HeadKV" "0862a0955fe82e9ff611d59541918e02c5def625"
run_one "FFY0/AdaKV" "04497abac4c1a58426f3daf1014578990e225cc5"
run_one "thunlp/FR-Spec" "29d0136b43d372d7d48806db8702cc9c813fdccf"
run_one "ryienh/jl-metric" "ae0a1e4c4be07f8675234207cca5abafb6d1c65c"
run_one "shimo-lab/modelmap" "ce0826b8a494ee5cef04cde7f5eb4ebe51e4d307"
run_one "togethercomputer/moa" "1b5cab0f0905d9da821e37322ac6df96ba65e1a7"
run_one "SongW-SW/CEB" "436943b78fdda84e0fc411e994eecd5646187f17"

echo "All 11 repos finished."
