#!/bin/bash
# GPQA-Diamond RE-RUN at a 64k output budget (n=198) for poolside/Laguna-S-2.1-NVFP4.
#
# WHY THIS RE-RUN EXISTS
# The completed 32k run (2026-08-27, c=4, 0 TimeoutErrors, 24h21m) scored 40.40%
# but is WITHHELD as a truncation artifact: 93/198 items (47.0%) emitted no
# answer line, and 78 of those ran to the 32,768-token ceiling mid-reasoning
# (generation length p50 2,043 chars, p90 129,887, max 172,139). The number
# measured the output budget, not the model. Precedent: Lightning moved
# 16k 53.03% -> 64k 76.26% on budget alone.
#
# WHAT CHANGED vs run_gpqa_full.sh (the 32k run)
#   1. max_gen_toks 32768 -> 65536.
#   2. timeout 14400 -> 30000 s. THIS IS LOAD-BEARING, NOT COSMETIC.
#      At c=4 each request gets ~17.3/4 = 4.33 tok/s, so a full 64k answer needs
#      ~15,153 s (4.21 h) -- which EXCEEDS the old 14400 s (4 h) deadline.
#      Reusing 14400 would cut off exactly the long items this re-run exists to
#      rescue and retry them from scratch, recreating the 2026-08-22 retry storm
#      (352 TimeoutErrors, run abandoned at 110/198). 30000 s = ~2x worst case.
#   3. Fresh output dir; the 32k artifacts are already committed to the repo.
#
# UNCHANGED ON PURPOSE (comparability):
#   num_concurrent=4 (validated: 0 timeouts over 24 h), temperature=0, same
#   clean task YAML, same fallback patch + firing log.
#
# NOTE ON SERVING CONFIG: no container restart. Laguna-S ships NO
# generation_config max_new_tokens, so there is no server-wide cap to clear
# (verified 2026-08-27 via viz/check_output_budget.py: PASS, 65536 accepted,
# max_model_len 131072). --generation-config vllm would NOT be the right tool
# anyway -- it only suppresses the sampling whitelist and would discard this
# checkpoint's eos_token_id [2, 24] handling (token 24 = </assistant>).
#
# EXCLUSIVE GPU ACCESS: do not run a throughput sweep or any other load against
# the engine while this runs.
set -u
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null)
export LM_EVAL_REASONING_FALLBACK=1
# Record every fallback firing (95 at 32k). Required to compare honestly against
# cards measured WITHOUT the patch (e.g. Ornith GPQA 69.70%).
OUT="/home/jchilders/gpqa64k/laguna_gpqa_64k"
mkdir -p "$OUT"
export LM_EVAL_REASONING_FALLBACK_LOG="${OUT}/fallback_firings.log"
source /tmp/lmeval-venv/bin/activate

MODEL="poolside/Laguna-S-2.1-NVFP4"
BASE="http://localhost:8000/v1/chat/completions"
INC="/tmp/lmeval_clean_tasks"
cd /tmp

echo "=== GPQA 64k start $(date -u +%FT%TZ) ==="
echo "=== budget=65536 concurrency=4 timeout=30000 ==="
OPENAI_API_KEY=dummy lm_eval \
  --model local-chat-completions \
  --model_args "model=${MODEL},base_url=${BASE},num_concurrent=4,max_retries=3,tokenized_requests=False,timeout=30000" \
  --apply_chat_template \
  --include_path "$INC" \
  --tasks gpqa_diamond_cot_zeroshot_clean \
  --gen_kwargs "max_gen_toks=65536,temperature=0" \
  --output_path "$OUT" \
  --log_samples > "${OUT}/gpqa_64k.log" 2>&1
rc=$?
echo "=== GPQA 64k done rc=${rc} $(date -u +%FT%TZ) ===" >> "${OUT}/gpqa_64k.log"
touch "${OUT}/DONE"
