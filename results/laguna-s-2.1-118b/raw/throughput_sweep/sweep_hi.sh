#!/bin/bash
# Extend the Laguna sweep past the default --max-num-seqs 128 scheduler cap.
# c=128 was still climbing +40.8%, so 233 tok/s is a FLOOR not a plateau.
OUT=~/bench/laguna
LOG=$OUT/sweep_hi.log
: > $LOG
wait_idle () {
  for _ in $(seq 1 60); do
    r=$(curl -s -m 10 http://localhost:8000/metrics \
        | awk "/^vllm:num_requests_running/{a=\$2} /^vllm:num_requests_waiting{/{b=\$2} END{print a+b}")
    [ "${r%%.*}" = "0" ] && return 0
    sleep 10
  done
}
for C in 192 256; do
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
touch $OUT/SWEEPHI_DONE
