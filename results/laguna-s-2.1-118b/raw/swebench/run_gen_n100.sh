#!/usr/bin/env bash
# SWE-bench Verified n=100 (seed-42 shuffle) -- poolside/Laguna-S-2.1-NVFP4.
#
# Comparability contract with the rest of warpcore-benchmarks:
#   * --subset verified --split test --shuffle --slice 0:100
#     mini-swe-agent's --shuffle uses random.seed(42) => DETERMINISTIC, so this is the
#     IDENTICAL instance set Ornith / Lightning / Qwen3.6 were scored on.
#   * config copied verbatim from the Ornith run (step_limit 250, litellm timeout 1800,
#     temperature 0, robust `git add -A && git diff --cached` submit protocol);
#     ONLY model_name differs.
#   * native tool-calling scaffold (default swebench.yaml), which the tool-call probe
#     verified for this model (finish_reason: tool_calls, clean JSON args).
#
# Agent loop + x86 test containers run HERE (Mac mini, x86_64); the model is served on
# warpcore. Restartable: re-run with the SAME -o and NO --redo-existing to skip
# completed instances and only work the remainder.
set -x
cd ~/swebench-laguna
source ~/swebench-lightning/venv/bin/activate
export HOSTED_VLLM_API_BASE="http://csi370295.alcf.anl.gov:8000/v1"
export HOSTED_VLLM_API_KEY="warpcore"
export MSWEA_COST_TRACKING='ignore_errors'

mini-extra swebench \
  --subset verified --split test \
  --shuffle --slice 0:100 \
  -c swebench.yaml \
  -m "hosted_vllm/poolside/Laguna-S-2.1-NVFP4" \
  --environment-class docker \
  -w 4 \
  -o /tmp/laguna_swe_n100 >> ~/laguna_swe_n100.log 2>&1

echo "EXIT=$?" >> ~/laguna_swe_n100.log
touch /tmp/laguna_swe_n100/GEN_DONE
