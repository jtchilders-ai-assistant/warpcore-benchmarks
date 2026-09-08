#!/bin/bash
# Ornith-1.0-35B-FP8 on GB10. compressed-tensors W8A8 FP8.
# GB10 has NO working CUTLASS W8A8 FP8 scaled-mm kernel; force BOTH
# the MoE experts and dense/linear FP8 GEMMs onto Marlin.
#
# Usage:
#   ./launch_ornith.sh throughput   # remote-client quality or on-box throughput
#   ./launch_ornith.sh agentic      # on-box agent harness; preserve host RAM
#
# This explicit profile is required because GB10 GPU and host memory are unified.
# Do not run an on-host agent harness with the throughput profile.
set -euo pipefail

PROFILE="${1:-}"
case "$PROFILE" in
  throughput)
    GPU_MEMORY_UTILIZATION=0.90
    MAX_NUM_SEQS=512
    ;;
  agentic)
    GPU_MEMORY_UTILIZATION=0.55
    MAX_NUM_SEQS=32
    ;;
  *)
    echo "usage: $0 {throughput|agentic}" >&2
    exit 2
    ;;
esac

MODEL="ornith-ai/Ornith-1.0-35B-FP8"
docker rm -f vllm_ornith 2>/dev/null || true
docker run -d --name vllm_ornith --gpus all --network host --ipc host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  vllm/vllm-openai:cu129-nightly-aarch64 \
  --model "$MODEL" \
  --served-model-name "$MODEL" \
  --moe-backend marlin \
  --max-model-len 262144 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --trust-remote-code \
  --host 0.0.0.0 --port 8000
