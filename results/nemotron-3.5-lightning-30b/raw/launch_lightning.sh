#!/usr/bin/env bash
# Launch NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 on warpcore (DGX Spark / GB10) under vLLM.
#
# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT-TOKEN BUDGET RECOMMENDATION (per-request `max_tokens`)
# ─────────────────────────────────────────────────────────────────────────────
# Lightning is a DEEP reasoner. On hard problems it emits a long chain-of-thought
# (which vLLM's nemotron_v3 parser strips from `content` but still counts against
# `max_tokens`), then a short final answer. Measured on GPQA-Diamond:
#
#     budget   truncated   score
#      16k        41%      53.03%
#      32k        21%      66.16%
#      64k         3%      76.26%   <- 97% of items finish; near-full capability
#
# The hard-tail items genuinely need p50~30k / p90~51k / max~54k completion tokens.
# `max_tokens` is a CEILING, not a reservation: it costs nothing for the ~80% of
# requests (and virtually all GSM8K/chat, visible output p50~200 tok) that finish
# fast. Weights are ~18 GiB leaving ~88 GiB KV (23x concurrency @ 256K context),
# so memory is a non-issue. The only real cost of a high ceiling is LATENCY on the
# rare deep request (~13 ms/tok single-stream -> a 30k-token trace is ~4 min).
#
# RECOMMENDED per-request `max_tokens` by workload:
#   * Interactive agent / chat : 8k-16k  (trade the hardest ~10-20% of problems
#                                          for bounded latency)
#   * Batch / offline reasoning: 32k-64k (full capability; let deep problems run)
#   * Never cap below ~4k       : truncates ordinary reasoning and silently drops
#                                 the final answer -> looks like the model is wrong.
#
# The server below sets --max-model-len 262144, so it ALLOWS up to ~256K output;
# the effective budget is chosen PER REQUEST by the client via `max_tokens`.
# There is no vLLM flag to set a *default* request max_tokens, so clients that
# omit it fall back to vLLM's own default -- always send an explicit `max_tokens`.
#
# Example client call with the batch/offline budget:
#   curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
#     -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"..."}],
#          "max_tokens":65536,"temperature":1.0,"top_p":0.95}'
# ─────────────────────────────────────────────────────────────────────────────
set -x
MODEL="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
docker rm -f vllm_lightning 2>/dev/null
docker run -d --rm --name vllm_lightning \
  --gpus all --network host --ipc host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  vllm/vllm-openai:v0.27.1 \
  --model "$MODEL" \
  --served-model-name "$MODEL" \
  --moe-backend marlin \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.9 \
  --max-model-len 262144 \
  --max-num-seqs 128 \
  --reasoning-parser nemotron_v3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --trust-remote-code \
  --host 0.0.0.0 --port 8000
echo "launched, container id above"
