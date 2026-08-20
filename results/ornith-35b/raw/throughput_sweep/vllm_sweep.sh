#!/bin/bash
# Reusable vLLM throughput sweep on warpcore — climb concurrency until output tok/s plateaus.
# Runs `vllm bench serve` inside the live container against localhost:8000 (server ceiling, raw
# completions with --ignore-eos so generation is full-length and comparable).
#
# Usage:  MODEL=<hf/id> CONTAINER=<name> ./vllm_sweep.sh 1 2 4 8 16 24 32 48 64 96 128
#   MODEL     - served model id (must match what /v1/models reports)   [required]
#   CONTAINER - docker container running vLLM (vllm_node / vllm_prebuilt) [required]
#   TOKENIZER - OPTIONAL base-repo tokenizer id. REQUIRED for models whose repo declares a
#               novel tokenizer_class the container's vLLM can't import (e.g. Intel AutoRound
#               repos declare `TokenizersBackend`) — `vllm bench serve` loads the tokenizer
#               CLIENT-SIDE to build the random dataset, so it dies the same way the server does
#               ("RuntimeError: Failed to load the tokenizer"). Point it at the BASE model repo
#               (same vocab, standard class), e.g. TOKENIZER=Qwen/Qwen3.5-122B-A10B.
#   args      - concurrency levels to sweep (defaults if none given)
#
# NOTE: to sweep BEYOND the server's --max-num-seqs, restart the server with a higher cap FIRST;
#       requesting c > max_num_seqs just queues and does not measure true higher concurrency.
# NOTE: --save-result writes INSIDE the container; rely on this script's stdout/tee log for results.
set -u
: "${MODEL:?set MODEL=<hf/id>}"
: "${CONTAINER:?set CONTAINER=<docker name>}"
IN_LEN=${IN_LEN:-512}
OUT_LEN=${OUT_LEN:-256}
OUTDIR=${OUTDIR:-/tmp/vllm_sweep}
TOKENIZER=${TOKENIZER:-}
mkdir -p "$OUTDIR"

TOK_ARG=""
if [ -n "$TOKENIZER" ]; then TOK_ARG="--tokenizer $TOKENIZER"; fi

LEVELS="$@"
if [ -z "$LEVELS" ]; then LEVELS="1 2 4 8 16 24 32 48 64 96 128"; fi

for C in $LEVELS; do
  NP=$(( C * 3 )); if [ "$NP" -lt 32 ]; then NP=32; fi; if [ "$NP" -gt 384 ]; then NP=384; fi
  # assert engine idle before each level (avoid orphan-benchmark pollution)
  RUN=$(curl -s -m 5 http://localhost:8000/metrics 2>/dev/null | grep 'vllm:num_requests_running' | grep -v '^#' | awk '{print $2}')
  echo "=== concurrency=$C num_prompts=$NP (engine_running_before=$RUN) ==="
  docker exec "$CONTAINER" vllm bench serve \
    --base-url http://localhost:8000 \
    --model "$MODEL" \
    $TOK_ARG \
    --backend openai --endpoint /v1/completions \
    --dataset-name random --random-input-len $IN_LEN --random-output-len $OUT_LEN \
    --ignore-eos \
    --num-prompts $NP --max-concurrency $C \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    2>&1 | grep -E "Successful requests|Benchmark duration|Output token throughput|Total Token throughput|Request throughput|Mean TTFT|Median TTFT|P99 TTFT|Mean TPOT|Median TPOT|Mean ITL"
  echo
done
echo "SWEEP_DONE" > "$OUTDIR/SWEEP_DONE"
echo "SWEEP_DONE"
