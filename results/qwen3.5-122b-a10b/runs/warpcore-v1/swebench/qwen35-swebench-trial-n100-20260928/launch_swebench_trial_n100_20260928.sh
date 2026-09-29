#!/bin/bash
set -uo pipefail
REPO=/Users/jchilders/workspaces/warpcore-benchmarks/.worktrees/qwen35-campaign
PY=/Users/jchilders/swebench-run/venv/bin/python
RUN_ID=qwen35-swebench-trial-n100-20260928
RUN_DIR="$REPO/results/qwen3.5-122b-a10b/runs/warpcore-v1/swebench/$RUN_ID"
STATE=/Users/jchilders/qwen35-swebench-trial-n100-20260928
ENDPOINT=http://csi370295.alcf.anl.gov:8000/v1
MODEL=Intel/Qwen3.5-122B-A10B-int4-AutoRound
mkdir -p "$STATE"
rm -f "$STATE/DONE" "$STATE/FAILED"
exec >"$STATE/launcher.log" 2>&1
cd "$REPO" || { printf 'cd failed\n' >"$STATE/FAILED"; exit 1; }
if [[ -e "$RUN_DIR" ]]; then
  printf 'run directory already exists\n' >"$STATE/FAILED"
  exit 1
fi
if ! curl -fsS --max-time 10 "$ENDPOINT/models" | /usr/bin/grep -Fq "$MODEL"; then
  printf 'endpoint/model unavailable\n' >"$STATE/FAILED"
  exit 1
fi
"$PY" - <<'PY' || { printf 'harness prerequisite failed\n' >"$STATE/FAILED"; exit 1; }
import minisweagent, swebench
assert minisweagent.__version__ == "2.4.6"
assert swebench.__version__ == "4.1.0"
PY
"$PY" viz/swebench_preflight.py --instances suite/swebench/instances-seed42-n100.json || {
  printf 'image preflight failed\n' >"$STATE/FAILED"
  exit 1
}
/usr/bin/make run-swebench \
  PYTHON="$PY" \
  SUITE=suite/warpcore-v1.yaml \
  ADAPTER=adapters/qwen3.5-122b-a10b.yaml \
  ENDPOINT="$ENDPOINT" \
  PROMPT_TOKENS=gsm8k=256,ifeval=373,gpqa_diamond=2808 \
  WORKERS=4 \
  RUN_ID="$RUN_ID" \
  NONCANONICAL_TRIAL=1
rc=$?
if [[ $rc -eq 0 ]]; then
  printf 'completed\n' >"$STATE/DONE"
else
  printf 'exit_code=%s\n' "$rc" >"$STATE/FAILED"
fi
exit "$rc"
