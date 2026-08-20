#!/bin/bash
# Ornith-1.0-35B-FP8 on GB10. compressed-tensors W8A8 FP8.
# GB10 has NO working CUTLASS w8a8 fp8 scaled-mm kernel (cutlass_gemm_caller Error Internal) ->
# force BOTH the MoE experts AND the dense/linear FP8 GEMMs onto Marlin.
docker rm -f vllm_ornith 2>/dev/null
docker run -d --name vllm_ornith --gpus all --network host --ipc host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  vllm/vllm-openai:cu129-nightly-aarch64 \
  --model ornith-ai/Ornith-1.0-35B-FP8 \
  --served-model-name ornith-ai/Ornith-1.0-35B-FP8 \
  --moe-backend marlin \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --trust-remote-code \
  --host 0.0.0.0 --port 8000
