#!/usr/bin/env bash
# Run the remaining repos via the real repo2rocm CLI, max 2 concurrent.
set -uo pipefail
cd /home/gsarkar/Repo2ROCm
export AMD_LLM_API_KEY="${AMD_LLM_API_KEY:-eabfc51a20ba432d90f51158431e022f}"

MAX_PARALLEL=2

# repo<SP>sha
REPOS=(
  "sarahmart/HARDMath 9ade062a80b7d475666b11ae13a2fb0d7b7f0341"
  "ZhiningLiu1998/SelfElicit 2fa9d1d3ab5a14e583de16fa974dfcd617d861f8"
  "thunlp/FR-Spec 29d0136b43d372d7d48806db8702cc9c813fdccf"
  "ryienh/jl-metric ae0a1e4c4be07f8675234207cca5abafb6d1c65c"
  "shimo-lab/modelmap ce0826b8a494ee5cef04cde7f5eb4ebe51e4d307"
  "togethercomputer/moa 1b5cab0f0905d9da821e37322ac6df96ba65e1a7"
  "SongW-SW/CEB 436943b78fdda84e0fc411e994eecd5646187f17"
)

run_one() {
  local full_name="$1" sha="$2"
  local log="run_$(echo "${full_name}" | tr '/' '_').log"
  echo "[START] ${full_name} @ ${sha} -> ${log}"
  python3 -u -m repo2rocm.cli migrate "${full_name}" \
    --sha "${sha}" --mode functional > "${log}" 2>&1
  echo "[DONE ${?}] ${full_name}"
}

for entry in "${REPOS[@]}"; do
  read -r full_name sha <<<"${entry}"
  run_one "${full_name}" "${sha}" &
  # throttle to MAX_PARALLEL background jobs
  while [ "$(jobs -rp | wc -l)" -ge "${MAX_PARALLEL}" ]; do
    wait -n
  done
done

wait
echo "[ALL REMAINING DONE]"
