#!/usr/bin/env bash
# SWE-bench Verified smoke -- Laguna-S-2.1-NVFP4, native tool-calling scaffold.
# Same config/seed/slice convention as the Ornith n=100 run for comparability.
set -x
cd ~/swebench-laguna
source ~/swebench-lightning/venv/bin/activate
export HOSTED_VLLM_API_BASE="http://csi370295.alcf.anl.gov:8000/v1"
export HOSTED_VLLM_API_KEY="warpcore"
export MSWEA_COST_TRACKING='ignore_errors'

mini-extra swebench \
  --subset verified --split test \
  --shuffle --slice 0:3 \
  -c swebench.yaml \
  -m "hosted_vllm/poolside/Laguna-S-2.1-NVFP4" \
  --environment-class docker \
  -w 3 \
  -o /tmp/laguna_swe_smoke > ~/laguna_swe_smoke.log 2>&1

echo "EXIT=$?" >> ~/laguna_swe_smoke.log
touch /tmp/laguna_swe_smoke/SMOKE_DONE
