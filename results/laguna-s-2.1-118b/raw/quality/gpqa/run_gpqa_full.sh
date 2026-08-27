#!/bin/bash
# GPQA-Diamond FULL run (n=198) for poolside/Laguna-S-2.1-NVFP4 on warpcore.
#
# WHY THIS DIFFERS FROM THE ABANDONED 2026-08-22 ATTEMPT
# The first attempt was killed at 110/198 after ~13 h with 352 TimeoutError
# retries. Root cause was an arithmetic mismatch, not a hang: it ran at
# num_concurrent=16 (the log shows 16, though the script said 5), giving each
# request ~4 tok/s, so a 32k-token answer needed ~2 h against a 3600 s client
# timeout. Long items were cut off and retried FROM SCRATCH, hitting the same
# wall forever.
#
# Fixes here:
#   1. num_concurrent=4  -> measured ~17.3 tok/s single-stream; at c=4 each
#      request keeps a usable share of decode.
#   2. timeout=14400 (4 h) -> comfortably exceeds a worst-case 32k answer at
#      the measured rate (~31 min at full single-stream, with headroom for
#      contention). The old 3600 s was BELOW worst-case generation time.
#   3. LM_EVAL_REASONING_FALLBACK=1 -> this endpoint's poolside_v1 parser logs
#      "Auto-initialization of reasoning token IDs failed" and files whole
#      answers under message.reasoning with content=null. Unpatched, those
#      score 0 (this is what made GSM8K read 83.40% vs a true ~96%).
#
# EXCLUSIVE GPU ACCESS: do not run a throughput sweep or other load against
# the engine while this runs (a prior Laguna sweep was invalidated that way).
set -u
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null)
export LM_EVAL_REASONING_FALLBACK=1
# Record every fallback firing so the card can state exactly how many items
# would have scored 0 unpatched -- required to compare this number honestly
# against cards measured WITHOUT the patch (e.g. Ornith GPQA 69.70%).
export LM_EVAL_REASONING_FALLBACK_LOG=/tmp/lmeval_results/laguna_gpqa_full/fallback_firings.log
source /tmp/lmeval-venv/bin/activate

MODEL="poolside/Laguna-S-2.1-NVFP4"
BASE="http://localhost:8000/v1/chat/completions"
INC="/tmp/lmeval_clean_tasks"
OUT="/tmp/lmeval_results/laguna_gpqa_full"
mkdir -p "$OUT"
cd /tmp

echo "=== GPQA full start $(date -u +%FT%TZ) ==="
OPENAI_API_KEY=dummy lm_eval \
  --model local-chat-completions \
  --model_args "model=${MODEL},base_url=${BASE},num_concurrent=4,max_retries=3,tokenized_requests=False,timeout=14400" \
  --apply_chat_template \
  --include_path "$INC" \
  --tasks gpqa_diamond_cot_zeroshot_clean \
  --gen_kwargs "max_gen_toks=32768,temperature=0" \
  --output_path "$OUT" \
  --log_samples > "${OUT}/gpqa_full.log" 2>&1
rc=$?
echo "=== GPQA full done rc=${rc} $(date -u +%FT%TZ) ==="
touch "${OUT}/DONE"
