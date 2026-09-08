#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$HOME/warpcore-benchmark-runs/lightning-campaign-20260908"
LOG="$RUN_DIR/stage1-throughput.log"
DONE="$RUN_DIR/STAGE1_DONE"
FAILED="$RUN_DIR/STAGE1_FAILED"
MODEL="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
CONTAINER="vllm_lightning"
mkdir -p "$RUN_DIR"
rm -f "$DONE" "$FAILED"
exec > >(tee -a "$LOG") 2>&1
trap 'rc=$?; if [ "$rc" -ne 0 ]; then printf "%s\n" "$rc" > "$FAILED"; fi' EXIT

# Model swap: Ornith campaign is complete. Preserve the image/cache; replace only
# the serving container. max-num-seqs=512 is specific to this throughput stage.
docker rm -f vllm_ornith vllm_lightning 2>/dev/null || true
docker run -d --name "$CONTAINER" \
  --gpus all --network host --ipc host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  vllm/vllm-openai:v0.27.1 \
  --model "$MODEL" \
  --served-model-name "$MODEL" \
  --moe-backend marlin \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.9 \
  --max-model-len 262144 \
  --max-num-seqs 512 \
  --reasoning-parser nemotron_v3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --trust-remote-code \
  --host 0.0.0.0 --port 8000

for i in $(seq 1 120); do
  if actual=$(curl -fsS -m 5 http://localhost:8000/v1/models 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null) && [ "$actual" = "$MODEL" ]; then
    echo "READY model=$actual poll=$i"
    break
  fi
  if [ "$i" -eq 120 ]; then
    echo "endpoint did not become ready"
    exit 4
  fi
  sleep 10
done

# Real completion canary: fail if content is absent or request does not stop.
python3 - <<'PY'
import json, urllib.request
model="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
payload={"model":model,"messages":[{"role":"user","content":"Reply with exactly LIGHTNING_OK"}],"max_tokens":1024,"temperature":0}
req=urllib.request.Request("http://localhost:8000/v1/chat/completions",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
with urllib.request.urlopen(req,timeout=180) as r: d=json.load(r)
c=d["choices"][0]; content=(c.get("message") or {}).get("content") or ""
print("CANARY",repr(content),c.get("finish_reason"))
if not content.strip() or c.get("finish_reason") != "stop": raise SystemExit(5)
PY

for C in 192 256 384; do
  running=$(curl -fsS -m 10 http://localhost:8000/metrics | python3 -c 'import sys
vals=[]
for line in sys.stdin:
    if line.startswith("vllm:num_requests_running{") or line.startswith("vllm:num_requests_waiting{"):
        vals.append(float(line.rsplit(None,1)[1]))
print(int(sum(vals)))')
  [ "$running" -eq 0 ] || { echo "engine busy before c=$C: $running"; exit 6; }
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
echo STAGE1_DONE
