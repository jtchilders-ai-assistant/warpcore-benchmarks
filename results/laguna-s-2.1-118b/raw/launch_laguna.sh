#!/bin/bash
# Laguna-S-2.1-NVFP4 (117.6B total / 8.5B active MoE) on warpcore GB10
# SIZING: NVFP4 weights ~92.9 GiB of the ~121 GiB unified pool. KV is cheap:
#   36/48 layers are sliding-window(512) => fixed 37.7 MB/seq; only 12 global layers
#   scale with context. FP8 KV => ~3.26 GB/seq @128k, ~0.84 GB/seq @32k.
# GB10 traps: NVFP4 MoE must use MARLIN (CUTLASS -> illegal instruction / init ValueError).
# Client note: reasoning model -> send generous max_tokens (>=4k chat, 32k+ agentic).
# OUTPUT-BUDGET TRAP: --generation-config defaults to `auto`, which loads the
#   checkpoint's generation_config.json; a `max_new_tokens` there becomes a
#   SERVER-WIDE cap on every request that clamps SILENTLY (no error, not in this
#   command, not in the startup banner). Laguna-S ships none, so this run is
#   uncapped -- but the XS variant ships "max_new_tokens": 32768. For eval runs
#   add `--generation-config vllm` to make the budget purely client-controlled,
#   and verify with viz/check_output_budget.py. See PROVENANCE.md 5c.
docker run -d --network host --name vllm_laguna \
  -v $HOME/vllm_patch:/patch \
  -e PYTHONPATH=/patch \
  -v /home/jchilders/.cache/huggingface:/root/.cache/huggingface \
  --gpus all --ipc=host \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  vllm/vllm-openai:cu129-nightly-aarch64 \
  --model poolside/Laguna-S-2.1-NVFP4 \
  --moe-backend marlin \
  --served-model-name poolside/Laguna-S-2.1-NVFP4 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.88 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser poolside_v1 \
  --reasoning-parser poolside_v1 \
  --default-chat-template-kwargs "{\"enable_thinking\": true}" \
  --trust-remote-code \
  --host 0.0.0.0 --port 8000
