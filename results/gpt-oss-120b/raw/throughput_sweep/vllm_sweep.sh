#!/bin/bash
# vLLM throughput sweep on warpcore — climb concurrency until output tok/s plateaus.
# Runs inside container vllm_prebuilt against localhost:8000 (server ceiling, raw completions).
set -u
MODEL="openai/gpt-oss-120b"
IN_LEN=512
OUT_LEN=256
OUTDIR=/tmp/vllm_sweep
mkdir -p "$OUTDIR"

# Concurrency ladder — continues past 32; the Python analyzer decides where to stop.
LEVELS="$@"
if [ -z "$LEVELS" ]; then LEVELS="1 2 4 8 16 32 48 64 96 128"; fi

for C in $LEVELS; do
  # num-prompts: enough to reach steady state at this concurrency (~3x, min 32, cap 384)
  NP=$(( C * 3 )); if [ "$NP" -lt 32 ]; then NP=32; fi; if [ "$NP" -gt 384 ]; then NP=384; fi
  FN="bench_c${C}.json"
  echo "=== concurrency=$C num_prompts=$NP ==="
  docker exec vllm_prebuilt vllm bench serve \
    --base-url http://localhost:8000 \
    --model "$MODEL" \
    --backend openai --endpoint /v1/completions \
    --dataset-name random --random-input-len $IN_LEN --random-output-len $OUT_LEN \
    --ignore-eos \
    --num-prompts $NP --max-concurrency $C \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-dir "$OUTDIR" --result-filename "$FN" \
    2>&1 | grep -E "Successful requests|Benchmark duration|Output token throughput|Total Token throughput|Mean TTFT|Median TTFT|P99 TTFT|Mean TPOT|Median TPOT|Mean ITL" 
  echo "saved $OUTDIR/$FN"
  echo
done
echo "SWEEP_DONE"
