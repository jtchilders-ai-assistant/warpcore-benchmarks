#!/usr/bin/env bash
# Re-run Qwen3.6-35B-A3B-FP8 on SWE-bench Verified n=100 (seed-42 shuffle).
#
# WHY: the 2026-08-04 run lost 22/100 instances to a 120 s Docker pull timeout on a
# cold image cache. Those instances never started a container -- the model was never
# invoked -- but were scored as zeros. See TODO.md section 2a and PROVENANCE.md.
#
# WHAT THIS FIXES vs the original run:
#   1. Pre-flight verifies all 100 images are cached, and REFUSES to launch otherwise.
#   2. pull_timeout raised 120s -> 1800s so a cold pull cannot silently zero an instance.
#   3. Uses Ornith's committed config verbatim, so the comparison is apples-to-apples
#      (identical step_limit/cost_limit/timeout AND the robust `git add -A` submit).
#
# The instance set is fixed by `--shuffle --slice 0:100`: mini-swe-agent sorts by
# instance_id then shuffles with a hardcoded seed 42, so this is the exact same 100
# instances Ornith and Lightning ran.
#
# Usage:  ./run_qwen36_swebench_rerun.sh [output_dir]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-/tmp/qwen36_swe_n100_rerun}"
MODEL="hosted_vllm/Qwen/Qwen3.6-35B-A3B-FP8"
ENDPOINT="${HOSTED_VLLM_API_BASE:-http://csi370295.alcf.anl.gov:8000/v1}"
CONFIG="$REPO/results/qwen3.6-35b-a3b/raw/swebench/swebench_qwen36_rerun_config.yaml"
INSTANCES="$REPO/results/qwen3.6-35b-a3b/raw/swebench/preds_shuffle100.json"
WORKERS="${WORKERS:-4}"

echo "=== Qwen3.6-35B SWE-bench Verified n=100 re-run ==="
echo "output   : $OUT"
echo "endpoint : $ENDPOINT"
echo "config   : $CONFIG"
echo

# --- 1. endpoint must be live BEFORE we burn hours ------------------------------
echo "--- checking vLLM endpoint ---"
if ! curl -sf --max-time 10 "${ENDPOINT%/v1}/v1/models" >/dev/null; then
  echo "ERROR: no vLLM endpoint at $ENDPOINT" >&2
  echo "Serve Qwen3.6-35B-A3B-FP8 on warpcore first." >&2
  exit 1
fi
SERVED=$(curl -sf --max-time 10 "${ENDPOINT%/v1}/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
echo "endpoint OK, serving: $SERVED"
if [[ "$MODEL" != *"$SERVED"* ]]; then
  echo "ERROR: endpoint serves '$SERVED' but this script expects '$MODEL'." >&2
  echo "Refusing to launch -- a mismatched model would silently produce a wrong-model score." >&2
  exit 1
fi
echo

# --- 2. image pre-flight: the whole point of this script ------------------------
echo "--- pre-flight: SWE-bench container images ---"
python3 "$REPO/viz/swebench_preflight.py" --instances "$INSTANCES" --pull --pull-timeout 1800
echo

# --- 3. launch -------------------------------------------------------------------
if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: missing config $CONFIG" >&2
  exit 1
fi

echo "--- launching generation (${WORKERS} workers) ---"
echo "NOTE: mini-swe-agent resumes from an existing preds.json in the output dir."
echo "      Delete $OUT to force a clean run."
echo

export HOSTED_VLLM_API_BASE="$ENDPOINT"
export HOSTED_VLLM_API_KEY="${HOSTED_VLLM_API_KEY:-warpcore}"
export MSWEA_COST_TRACKING='ignore_errors'

mini-extra swebench \
  --subset verified --split test \
  --shuffle --slice 0:100 \
  -c "$CONFIG" \
  -m "$MODEL" \
  --environment-class docker \
  -w "$WORKERS" \
  -o "$OUT"

echo
echo "=== generation complete: $OUT ==="
echo "Next: grade with the swebench harness, then copy preds.json, the report,"
echo "exit_statuses, minisweagent.log, and this config into"
echo "results/qwen3.6-35b-a3b/raw/swebench/ per PROVENANCE.md."
