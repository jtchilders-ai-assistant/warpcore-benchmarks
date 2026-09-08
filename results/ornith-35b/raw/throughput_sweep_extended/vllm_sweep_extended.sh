#!/bin/bash
set -euo pipefail

RUN_DIR="$HOME/warpcore-benchmark-runs/ornith-throughput-extended-20260908"
LOG="$RUN_DIR/sweep.log"
DONE="$RUN_DIR/DONE"
FAILED="$RUN_DIR/FAILED"
MODEL="ornith-ai/Ornith-1.0-35B-FP8"
CONTAINER="vllm_ornith"
mkdir -p "$RUN_DIR"
rm -f "$DONE" "$FAILED"
exec > >(tee -a "$LOG") 2>&1
trap 'rc=$?; if [ "$rc" -ne 0 ]; then printf "%s\n" "$rc" > "$FAILED"; fi' EXIT

actual=$(curl -fsS -m 10 http://localhost:8000/v1/models | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
if [ "$actual" != "$MODEL" ]; then
  echo "wrong served model: $actual (expected $MODEL)"
  exit 2
fi

for C in 192 256 384; do
  running=$(curl -fsS -m 10 http://localhost:8000/metrics | python3 -c 'import sys
vals=[]
for line in sys.stdin:
    if line.startswith("vllm:num_requests_running{") or line.startswith("vllm:num_requests_waiting{"):
        vals.append(float(line.rsplit(None,1)[1]))
print(int(sum(vals)))')
  if [ "$running" -ne 0 ]; then
    echo "engine not idle before c=$C: running+waiting=$running"
    exit 3
  fi
  NP=384
  echo "=== concurrency=$C num_prompts=$NP engine_running_before=$running ==="
  docker exec "$CONTAINER" vllm bench serve \
    --base-url http://localhost:8000 \
    --model "$MODEL" \
    --backend openai --endpoint /v1/completions \
    --dataset-name random --random-input-len 512 --random-output-len 256 \
    --ignore-eos --temperature 0 \
    --num-prompts "$NP" --max-concurrency "$C" \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99
  echo
 done

touch "$DONE"
echo SWEEP_DONE
