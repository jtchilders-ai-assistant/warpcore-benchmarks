#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$HOME/warpcore-benchmark-runs/qwen38-flash-throughput-corrected-20261008"
LOG="$RUN_DIR/sweep.log"
DONE="$RUN_DIR/DONE"
FAILED="$RUN_DIR/FAILED"
MODEL="qwen3.8-flash-next"
TOKENIZER="/hf/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47"
CONTAINER="qwen38-flash"
LEVELS=(1 2 4 8 16 32 64 96 128 192 256)

mkdir -p "$RUN_DIR"
rm -f "$DONE" "$DONE.tmp" "$FAILED" "$LOG" "$RUN_DIR"/level-*.log "$RUN_DIR"/level-*.rc
exec > >(tee -a "$LOG") 2>&1
trap 'rc=$?; if [ "$rc" -ne 0 ]; then printf "%s\n" "$rc" > "$FAILED.tmp"; mv -f "$FAILED.tmp" "$FAILED"; fi' EXIT

actual="$(curl -fsS -m 10 http://127.0.0.1:8000/v1/models | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"
[ "$actual" = "$MODEL" ] || { echo "wrong served model: $actual"; exit 2; }
docker inspect "$CONTAINER" --format 'image={{.Image}} cmd={{json .Config.Cmd}}'
docker exec "$CONTAINER" python3 - "$TOKENIZER" <<'PY'
import json, os, sys
p=sys.argv[1]
for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
    assert os.path.isfile(os.path.join(p,name)), f"missing {name}"
t=json.load(open(os.path.join(p,"tokenizer_config.json")))
print("TOKENIZER_CLASS=" + str(t.get("tokenizer_class")))
PY

completed=0
for C in "${LEVELS[@]}"; do
  curl -fsS -m 10 http://127.0.0.1:8000/health >/dev/null
  idle="$(curl -fsS -m 10 http://127.0.0.1:8000/metrics | python3 -c 'import sys
r=w=0.0
for l in sys.stdin:
    if l.startswith("vllm:num_requests_running{"): r += float(l.rsplit(None,1)[1])
    if l.startswith("vllm:num_requests_waiting{"): w += float(l.rsplit(None,1)[1])
print(f"{r:g} {w:g}")')"
  [ "$idle" = "0 0" ] || { echo "engine not idle before c=$C: $idle"; exit 3; }
  NP=$((C*3)); [ "$NP" -lt 32 ] && NP=32; [ "$NP" -gt 384 ] && NP=384
  L="$RUN_DIR/level-$C.log"
  echo "=== concurrency=$C num_prompts=$NP engine_running_waiting=$idle ==="
  set +e
  timeout 900 docker exec "$CONTAINER" vllm bench serve \
    --base-url http://localhost:8000 --model "$MODEL" --tokenizer "$TOKENIZER" \
    --backend openai --endpoint /v1/completions \
    --dataset-name random --random-input-len 512 --random-output-len 256 \
    --ignore-eos --temperature 0 --num-prompts "$NP" --max-concurrency "$C" \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    >"$L" 2>&1
  rc=$?
  set -e
  printf '%s\n' "$rc" > "$RUN_DIR/level-$C.rc"
  [ "$rc" -eq 0 ] || { echo "benchmark failed c=$C rc=$rc"; exit "$rc"; }
  ! grep -qE 'Traceback|Failed requests:[[:space:]]*[1-9]' "$L" || { echo "benchmark error evidence c=$C"; exit 4; }
  success="$(python3 - "$L" <<'PY'
import re,sys
s=open(sys.argv[1]).read()
m=re.findall(r"Successful requests:\s+(\d+)",s)
print(m[0] if len(m)==1 else "INVALID")
PY
)"
  [ "$success" = "$NP" ] || { echo "successful mismatch c=$C expected=$NP got=$success"; exit 5; }
  grep -E 'Successful requests|Failed requests|Benchmark duration|Request throughput|Output token throughput|Total Token throughput|Mean TTFT|Median TTFT|P99 TTFT|Mean TPOT|Median TPOT|Mean ITL' "$L" | tee -a "$LOG"
  completed=$((completed+1))
done

[ "$completed" -eq "${#LEVELS[@]}" ]
printf 'DONE %d/%d levels=%s\n' "$completed" "${#LEVELS[@]}" "${LEVELS[*]}" > "$DONE.tmp"
mv -f "$DONE.tmp" "$DONE"
echo "SWEEP_COMPLETE $completed/${#LEVELS[@]}"
