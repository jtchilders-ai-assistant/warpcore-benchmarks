#!/usr/bin/env bash
set -euo pipefail

MODEL="Qwen/Qwen3.6-35B-A3B-FP8"
ENDPOINT="http://csi370295.alcf.anl.gov:8000/v1"
CHAT_URL="${ENDPOINT}/chat/completions"
REPO="$HOME/workspaces/warpcore-benchmarks"
TASK_DIR="$REPO/results/qwen3.6-35b-a3b/raw"
OUT="$HOME/warpcore-benchmark-runs/qwen36-gpqa-thinking-64k-20260909"
LMEVAL="$HOME/workspaces/lmeval-venv/bin/lm_eval"
mkdir -p "$OUT/results"
rm -f "$OUT/DONE" "$OUT/FAILED"

cleanup() {
  rc=$?
  if [[ $rc -eq 0 ]]; then touch "$OUT/DONE"; else printf '%s\n' "$rc" > "$OUT/FAILED"; fi
}
trap cleanup EXIT

python3 - "$ENDPOINT/models" "$MODEL" <<'PY'
import json,sys,urllib.request
with urllib.request.urlopen(sys.argv[1], timeout=15) as f: d=json.load(f)
ids=[x['id'] for x in d.get('data',[])]
assert ids == [sys.argv[2]], (ids,sys.argv[2])
print('exact-model check passed:',ids[0])
PY

export OPENAI_API_KEY=dummy
"$LMEVAL" \
  --model local-chat-completions \
  --model_args "model=${MODEL},base_url=${CHAT_URL},num_concurrent=4,max_retries=8,tokenized_requests=False,timeout=7200" \
  --apply_chat_template \
  --include_path "$TASK_DIR" \
  --tasks gpqa_diamond_cot_zeroshot_clean \
  --gen_kwargs "max_gen_toks=65536,temperature=0" \
  --output_path "$OUT/results" \
  --log_samples \
  --seed 42 \
  > "$OUT/run.log" 2>&1
