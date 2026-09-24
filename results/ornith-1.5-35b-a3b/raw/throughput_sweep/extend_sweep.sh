#!/usr/bin/env bash
set -euo pipefail
BASE=/home/jchilders/warpcore-benchmark-runs/ornith-1.5/throughput
LOG="$BASE/sweep.log"
MODEL=ornith-ai/Ornith-1.5-35B-A3B-FP8
CONTAINER=vllm_ornith15
finish() { rc=$?; if (( rc == 0 )); then printf 'SWEEP_DONE\n' > "$BASE/SWEEP_DONE"; else printf 'extension failed rc=%s utc=%s\n' "$rc" "$(date -u +%FT%TZ)" >> "$LOG"; fi; exit "$rc"; }
trap finish EXIT
for c in 192 256 384; do
  n=$((3*c))
  running=$(curl -fsS http://127.0.0.1:8000/metrics | python3 -c 'import sys; print(next((line.split()[-1] for line in sys.stdin if line.startswith("vllm:num_requests_running")), "unknown"))')
  printf '\n=== concurrency=%s num_prompts=%s (engine_running_before=%s) ===\n' "$c" "$n" "$running" | tee -a "$LOG"
  docker exec "$CONTAINER" vllm bench serve \
    --base-url http://localhost:8000 \
    --model "$MODEL" \
    --backend openai \
    --endpoint /v1/completions \
    --dataset-name random \
    --random-input-len 512 \
    --random-output-len 256 \
    --ignore-eos \
    --num-prompts "$n" \
    --max-concurrency "$c" 2>&1 | tee -a "$LOG"
done
printf '\nSWEEP_DONE\n' | tee -a "$LOG"
