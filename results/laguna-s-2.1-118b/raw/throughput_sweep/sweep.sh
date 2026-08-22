#!/bin/bash
# Laguna-S-2.1-NVFP4 throughput sweep on warpcore GB10.
# Uses raw completions + --ignore-eos for clean full-length tok/s (chat backend
# lets the model stop early and understates throughput).
OUT=~/bench/laguna
LOG=$OUT/sweep.log
: > $LOG
for C in 1 2 4 8 16 32 64 128; do
  echo "=================== CONCURRENCY $C ===================" >> $LOG
  docker exec vllm_laguna vllm bench serve \
    --base-url http://localhost:8000 \
    --model poolside/Laguna-S-2.1-NVFP4 \
    --backend openai --endpoint /v1/completions \
    --dataset-name random --random-input-len 512 --random-output-len 256 \
    --ignore-eos --num-prompts $((C*3)) --max-concurrency $C >> $LOG 2>&1
  echo "" >> $LOG
done
touch $OUT/SWEEP_DONE
