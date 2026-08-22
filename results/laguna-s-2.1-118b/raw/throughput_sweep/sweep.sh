#!/bin/bash
# Laguna-S-2.1-NVFP4 throughput sweep on warpcore GB10 -- EXCLUSIVE GPU ACCESS REQUIRED.
# Raw completions + --ignore-eos for clean full-length tok/s.
# NOTE: a prior run was contaminated by an overlapping lm-eval smoke test (cf. ISSUES #6);
# this version refuses to start a level unless the engine is idle.
OUT=~/bench/laguna
LOG=$OUT/sweep.log
: > $LOG

wait_idle () {
  for _ in $(seq 1 60); do
    r=$(curl -s -m 10 http://localhost:8000/metrics \
        | awk "/^vllm:num_requests_running/{print \$2} /^vllm:num_requests_waiting{/{print \$2}" \
        | paste -sd+ | bc 2>/dev/null)
    [ "${r%%.*}" = "0" ] && return 0
    sleep 10
  done
  echo "WARNING: engine not idle before level" >> $LOG
}

for C in 1 2 4 8 16 32 64 128; do
  wait_idle
  echo "=================== CONCURRENCY $C ===================" >> $LOG
  docker exec vllm_laguna vllm bench serve \
    --base-url http://localhost:8000 \
    --model poolside/Laguna-S-2.1-NVFP4 \
    --backend openai --endpoint /v1/completions \
    --dataset-name random --random-input-len 512 --random-output-len 256 \
    --ignore-eos --num-prompts $((C*3)) --max-concurrency $C >> $LOG 2>&1
  echo "" >> $LOG
done
touch $OUT/SWEEP_DONE
