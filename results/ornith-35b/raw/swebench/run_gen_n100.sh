#!/usr/bin/env bash
set -x
cd ~/swebench-ornith
source ~/swebench-lightning/venv/bin/activate
export HOSTED_VLLM_API_BASE="http://csi370295.alcf.anl.gov:8000/v1"
export HOSTED_VLLM_API_KEY="warpcore"
export MSWEA_COST_TRACKING='ignore_errors'
mini-extra swebench \
  --subset verified --split test \
  --shuffle --slice 0:100 \
  -c swebench.yaml \
  -m "hosted_vllm/ornith-ai/Ornith-1.0-35B-FP8" \
  --environment-class docker \
  -w 4 \
  -o /tmp/ornith_swe_n100
