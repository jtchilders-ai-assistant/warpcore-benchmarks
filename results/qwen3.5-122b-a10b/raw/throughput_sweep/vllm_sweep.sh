#!/bin/bash
# vLLM throughput concurrency-plateau sweep for Qwen3.5-122B on warpcore.
# Runs `vllm bench serve` INSIDE the container against localhost:8000 (server ceiling),
# raw completions + --ignore-eos so every request generates full length (512in/256out),
# standard methodology matching the gpt-oss-120b sweep for direct comparison.
set -u
MODEL="Intel/Qwen3.5-122B-A10B-int4-AutoRound"
CONTAINER="vllm_node"
IN_LEN=512
OUT_LEN=256
OUTDIR=/tmp/qwen_sweep
mkdir -p "$OUTDIR"
LOG="$OUTDIR/sweep.log"
rm -f "$OUTDIR/DONE"

# Standard ladder; climb until output tok/s plateaus (<~5% gain).
LEVELS="1 2 4 8 16 32 48 64 96 128 192 256"

{
echo "SWEEP START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "MODEL=$MODEL CONTAINER=$CONTAINER IN=$IN_LEN OUT=$OUT_LEN"
for C in $LEVELS; do
  NP=$(( C * 3 )); if [ "$NP" -lt 32 ]; then NP=32; fi; if [ "$NP" -gt 384 ]; then NP=384; fi
  # assert engine idle before each level (avoid orphan-benchmark pollution)
  RUN=$(curl -s -m 5 http://localhost:8000/metrics 2>/dev/null | grep 'vllm:num_requests_running' | grep -v '^#' | awk '{print $2}')
  echo "=== concurrency=$C num_prompts=$NP (engine_running_before=$RUN) $(date -u +%H:%M:%S) ==="
  docker exec "$CONTAINER" vllm bench serve \
    --base-url http://localhost:8000 \
    --model "$MODEL" \
    --tokenizer Qwen/Qwen3.5-122B-A10B \
    --backend openai --endpoint /v1/completions \
    --dataset-name random --random-input-len $IN_LEN --random-output-len $OUT_LEN \
    --ignore-eos \
    --num-prompts $NP --max-concurrency $C \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    2>&1 | grep -E "Successful requests|Benchmark duration|Output token throughput|Total Token throughput|Request throughput|Mean TTFT|Median TTFT|P99 TTFT|Mean TPOT|Median TPOT|Mean ITL"
  echo
done
echo "SWEEP END $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$LOG" 2>&1

echo "SWEEP_DONE" > "$OUTDIR/DONE"
