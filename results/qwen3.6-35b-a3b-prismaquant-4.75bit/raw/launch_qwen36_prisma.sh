#!/usr/bin/env bash
set -euo pipefail

MODEL='rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm'
REVISION='e347d86b2a6cba4b54ea6f87ca247f60439eed07'
IMAGE='eugr/spark-vllm@sha256:c154ad0a2575d6c42f8e05cba16ef255ce4ff54d537ad37987ca5b4215cb58b8'
NAME='vllm_qwen36_prisma'

docker rm -f "$NAME" 2>/dev/null || true
exec docker run -d --rm \
  --name "$NAME" \
  --gpus all \
  --network host \
  --ipc host \
  -v /home/jchilders/.cache/vllm:/root/.cache/vllm \
  -v /home/jchilders/.cache/flashinfer:/root/.cache/flashinfer \
  -v /home/jchilders/.cache/b12x:/root/.cache/b12x \
  -v /home/jchilders/.triton:/root/.triton \
  -v /home/jchilders/.tilelang:/root/.tilelang \
  -v /home/jchilders/.cache/huggingface:/root/.cache/huggingface \
  "$IMAGE" \
  vllm serve "$MODEL" \
    --revision "$REVISION" \
    --served-model-name "$MODEL" \
    --trust-remote-code \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.80 \
    --max-num-seqs 128 \
    --enable-prefix-caching \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_xml \
    --enable-auto-tool-choice \
    --host 0.0.0.0 \
    --port 8000
