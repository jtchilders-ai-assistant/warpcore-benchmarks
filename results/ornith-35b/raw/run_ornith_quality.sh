#!/bin/bash
# Full quality suite for Ornith-1.0-35B-FP8 on warpcore (lm-eval 0.4.12, chat endpoint).
# Runs GSM8K-clean, IFEval, GPQA-Diamond-clean as SEPARATE processes (a scorer crash in one
# can't discard the others). Reasoning model -> generous max_gen_toks; GPQA at low concurrency.
# Touches DONE flag per task and a global ALL_DONE at the end.
set -u
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null)
source /tmp/lmeval-venv/bin/activate
MODEL="ornith-ai/Ornith-1.0-35B-FP8"
BASE="http://localhost:8000/v1/chat/completions"
INC="/tmp/lmeval_tasks/gpqa_clean"
OUT="/tmp/lmeval_results/ornith35b"
mkdir -p "$OUT"
cd /tmp

run_task () {
  local name="$1" task="$2" conc="$3" toks="$4"
  echo "=== [$name] start $(date -u +%H:%M:%S) ==="
  OPENAI_API_KEY=dummy lm_eval \
    --model local-chat-completions \
    --model_args "model=${MODEL},base_url=${BASE},num_concurrent=${conc},max_retries=8,tokenized_requests=False,timeout=3600" \
    --apply_chat_template \
    --include_path "$INC" \
    --tasks "$task" \
    --gen_kwargs "max_gen_toks=${toks},temperature=0" \
    --output_path "${OUT}/${name}" \
    --log_samples > "${OUT}/${name}.log" 2>&1
  echo "=== [$name] done rc=$? $(date -u +%H:%M:%S) ==="
  touch "${OUT}/${name}.DONE"
}

# GSM8K and IFEval tolerate higher concurrency; GPQA long-tail wants c=5.
run_task gsm8k  gsm8k_cot_zeroshot_clean          8  8192
run_task ifeval ifeval                            8  8192
run_task gpqa   gpqa_diamond_cot_zeroshot_clean   5  32768

touch "${OUT}/ALL_DONE"
echo "ALL_DONE $(date -u +%H:%M:%S)"
