#!/bin/bash
set -euo pipefail
ROOT=/Users/jchilders/workspaces/warpcore-benchmarks/.worktrees/qwen35-campaign
OUT="$ROOT/results/qwen3.5-122b-a10b/diagnostic/noncanonical-limit5-gpqa-20260924"
PY="$HOME/workspaces/lmeval-venv/bin/python"
rm -f "$OUT/DONE" "$OUT/FAILED"
export OPENAI_API_KEY=dummy
export LMEVAL_SIDECAR_RUN_ID=noncanonical-limit5-gpqa-20260924
cd "$ROOT"
set +e
"$PY" viz/lmeval_sidecar_runner.py \
  --model sidecar-chat-completions \
  --model_args "base_url=http://csi370295.alcf.anl.gov:8000/v1/chat/completions,model=Intel/Qwen3.5-122B-A10B-int4-AutoRound,num_concurrent=4,max_retries=0,timeout=9000,tokenized_requests=False,sidecar_path=$OUT/response_metadata.jsonl" \
  --apply_chat_template \
  --tasks gpqa_diamond_cot_zeroshot_clean \
  --gen_kwargs max_gen_toks=65536,temperature=0,do_sample=false \
  --output_path "$OUT/raw" \
  --log_samples --num_fewshot 0 --limit 5 \
  --include_path "$ROOT/suite/tasks" \
  > "$OUT/run.log" 2>&1
rc=$?
set -e
printf '%s\n' "$rc" > "$OUT/exit_code"
if [ "$rc" -eq 0 ]; then touch "$OUT/DONE"; else touch "$OUT/FAILED"; fi
exit "$rc"
