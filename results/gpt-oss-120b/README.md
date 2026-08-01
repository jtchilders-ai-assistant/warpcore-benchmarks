# openai/gpt-oss-120b — Warpcore Benchmark Card

**Date:** 2026-07-27
**Model:** `openai/gpt-oss-120b` (MXFP4)
**Host:** Warpcore, NVIDIA DGX Spark / GB10 — see [../../HARDWARE.md](../../HARDWARE.md)
**Serving:** vLLM `0.23.1rc1.dev961+gbc6fbf472.d20260708`, container `vllm_prebuilt`
(image `eugr/spark-vllm:latest`), MARLIN MoE + TRITON attention
**Endpoint:** `http://csi370295.alcf.anl.gov:8000/v1/chat/completions`

---

## Quality — lm-evaluation-harness 0.4.9.1

Settings: `--model local-chat-completions --apply_chat_template`, 0-shot CoT, `temperature=0`,
`max_gen_toks=8192` (16384 for GPQA). Full test sets (no `--limit`).

| Benchmark | Metric | Score | Stderr | n |
| --------- | ------ | ----- | ------ | - |
| GSM8K (CoT, 0-shot) | exact_match, flexible-extract | **83.70%** | ±1.02 | 1319 |
| IFEval | prompt-level strict acc | **83.73%** | ±1.59 | 541 |
| IFEval | inst-level strict acc | 89.09% | — | 541 |
| IFEval | prompt-level loose acc | 86.69% | ±1.46 | 541 |
| IFEval | inst-level loose acc | 91.01% | — | 541 |
| GPQA-Diamond (CoT, clean-extract) | exact_match, answer-line | **72.73%** | ±3.17 | 198 |

**Notes**
- **GSM8K strict-match reads 0% — format artifact, not capability.** gpt-oss doesn't emit the rigid
  `#### <n>` tail; flexible-extract is the correct metric (verified to pull the model's real answer).
- **GPQA** used a custom clean-extraction task (see `raw/gpqa_clean_task.yaml`) with an anchored
  "The answer is (X)" regex; both filters agree at 72.73%, and parsing was verified deterministic.
  72.7% on GPQA-Diamond is a strong, near-frontier result for a 120B open model.

## Agentic — Zapier AutomationBench (public set)

Domain: **sales**, `--toolset api`, `--reasoning-effort high`. **Smoke only (5 tasks).** Not the full
100-task domain. AutomationBench is a brutal cross-app business-workflow benchmark — the public
leaderboard tops out ~30% strict (Claude Opus 4.8); gpt-oss is not tuned for it.

| Run | Tasks | Avg partial credit | Strict pass rate | Wall clock | Input tok | Output tok | Tool calls |
| --- | ----- | ------------------ | ---------------- | ---------- | --------- | ---------- | ---------- |
| c=5  | 5 | 15.0% | 0% (0/5) | 10m47s | — | — | — |
| c=32 | 5 | 8.6% | 0% (0/5) | 8m39s (518.9s) | 1,248,257 | 29,421 | 113 |

Per-task token usage (c=32 run) — note the very heavy **input** load (tool results fed back across up
to 50 agent steps):

| Task | Input tok | Output tok | Total |
| ---- | --------- | ---------- | ----- |
| sales.multi_hop_lookup | 416,987 | 8,887 | 425,874 |
| sales.negative_selection | 324,050 | 5,862 | 329,912 |
| sales.recency_selection | 38,626 | 2,343 | 40,969 |
| sales.priority_selection | 194,488 | 4,583 | 199,071 |
| sales.format_ambiguity | 274,106 | 7,746 | 281,852 |

Aggregate model time (c=32): 1890.2 s across 5 concurrent tasks (`total_model_time_s`). This is
wall-clock-under-concurrency, **not** a clean tok/s measurement — see Throughput below.

## Throughput / latency — `vllm bench serve` concurrency sweep

Measured 2026-07-28, **on-box** (inside `vllm_prebuilt` against `localhost:8000` → server ceiling,
network excluded). Raw-completions path (`--backend openai --endpoint /v1/completions --ignore-eos`)
so gpt-oss generates the full length (the chat path lets it stop early and understates tok/s).
Fixed shape: **512 input / 256 output tokens** per request. Concurrency was increased until output
throughput **plateaued** (not capped at an arbitrary number).

| Concurrency | Output tok/s | Δ vs prev | req/s | Mean TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) |
| ----------: | -----------: | --------: | ----: | -------------: | ------------: | -------------: |
| 1   | 34.1  |   —    | 0.133 | 71    | 76    | 29.2  |
| 2   | 56.9  | +66.9% | 0.222 | 131   | 166   | 34.8  |
| 4   | 85.6  | +50.4% | 0.335 | 197   | 230   | 46.1  |
| 8   | 120.9 | +41.2% | 0.472 | 231   | 256   | 65.5  |
| 16  | 168.2 | +39.1% | 0.657 | 304   | 355   | 94.3  |
| 32  | 233.7 | +38.9% | 0.913 | 1132  | 3483  | 132.9 |
| 48  | 285.3 | +22.1% | 1.115 | 1447  | 4777  | 163.1 |
| 64  | 338.5 | +18.6% | 1.322 | 1359  | 5061  | 184.2 |
| 96  | 422.9 | +24.9% | 1.652 | 2396  | 9381  | 218.0 |
| 128 | 499.1 | +18.0% | 1.950 | 2131  | 9965  | 248.3 |
| 192 | 682.7 | +36.8% | 2.667 | 1035  | 1340  | 277.9 |
| **256** | **708.9** | **+3.8%** | 2.769 | 1120 | 1495 | 276.8 |

**Plateau: ~700–710 tok/s output**, reached at **concurrency ~192–256**. Every step through c=192 added
double-digit throughput; c=192→256 added only **+3.8%**, so 256 is at/just past the knee. Notably 32 was
**not** the ceiling — it delivers 233 tok/s, only ~1/3 of the sustained max.

**Interpreting the two operating points:**
- **Single-stream (c=1):** 34.1 tok/s/user, TTFT 71 ms, TPOT 29 ms — snappy interactive latency.
- **Max throughput (c≈192–256):** ~700 tok/s aggregate, but per-token latency degrades ~10× (TPOT ~278 ms)
  and TTFT is second-scale. This is the batch/offline regime, not interactive.
- The bandwidth-limited GB10 decode path means TPOT rises steadily with batch size; pick the concurrency
  for your SLO (e.g. keep TPOT < ~100 ms → stay at/below c≈16).

All 12 levels completed with **zero failed requests**. Raw per-level JSON + the parsed
[`throughput_sweep.csv`](raw/throughput_sweep/throughput_sweep.csv) and the sweep script are in
[`raw/throughput_sweep/`](raw/throughput_sweep/).

### Reproduce the sweep
```bash
# on-box, inside the container; climbs concurrency, saves JSON per level
ssh warpcore 'bash /tmp/vllm_sweep.sh 1 2 4 8 16 32 48 64 96 128 192 256'
# each level: vllm bench serve --backend openai --endpoint /v1/completions --ignore-eos \
#   --dataset-name random --random-input-len 512 --random-output-len 256 \
#   --num-prompts <~3x conc> --max-concurrency <conc> --save-result ...
```

## Agentic coding — pi-30 (Fleet-30)

[`rick-stevens-ai/pi-30`](https://github.com/rick-stevens-ai/pi-30): 30 agentic-coding problems, each
solved via a full `pi` agent tool-loop (read/write/bash), graded **solely by verifier exit codes**.
Run from a client Mac against the warpcore endpoint, `PI_TIMEOUT=360`, single-shot canonical.

| Model | pi-30 score | Failures |
| ----- | :---------: | -------- |
| **gpt-oss-120b** | **30 / 30** | none |
| Nemotron-3-Super-120B | 30 / 30 | none |
| Qwen3.6-35B-A3B | 29 / 30 | P2 (LRU cache) |

**⚠ Serving-stack caveat (this materially changes the score):** gpt-oss-120b MUST be served on the
crash-fixed **`eugr/spark-vllm:latest`** image (MARLIN MXFP4 MoE + TRITON attention). The
`spark-vllm-docker` recipe's default `vllm-node-mxfp4` build hits a `cudaErrorIllegalAddress` (CUTLASS
MXFP4 MoE kernel bug) partway through the sustained tool-calling workload — on the crashing build this
same model scored a bogus **6/30** (engine died ~P7; P15/P23 throughput collapsed 250–1000×). On the
crash-fixed image it is a clean **30/30**, verified beforehand with a 60-way concurrent
guided-decoding stress test (60/60 OK, engine alive). **The serving container is a first-class
benchmark variable — the same weights score 6/30 or 30/30 depending on it.** See
[../../ISSUES.md](../../ISSUES.md). Raw per-problem log: [`raw/pi30/RESULTS.txt`](raw/pi30/RESULTS.txt).

## Reproduce

```bash
python3 -m venv lmeval-venv && source lmeval-venv/bin/activate
pip install "lm-eval[api]" "numpy<2" langdetect immutabledict nltk

# GSM8K + IFEval (run each task separately — see ISSUES.md #3)
OPENAI_API_KEY=dummy lm_eval --model local-chat-completions \
  --model_args "model=openai/gpt-oss-120b,base_url=http://csi370295.alcf.anl.gov:8000/v1/chat/completions,num_concurrent=12,tokenized_requests=False,timeout=900" \
  --apply_chat_template --tasks gsm8k_cot_zeroshot --gen_kwargs "max_gen_toks=8192,temperature=0" \
  --output_path results/final_gsm8k --log_samples

# GPQA (gated; needs HF_TOKEN; low concurrency for the long-generation tail — see ISSUES.md #4)
HF_TOKEN=... OPENAI_API_KEY=dummy lm_eval --model local-chat-completions \
  --model_args "model=openai/gpt-oss-120b,base_url=http://csi370295.alcf.anl.gov:8000/v1/chat/completions,num_concurrent=4,max_retries=5,tokenized_requests=False,timeout=1200" \
  --apply_chat_template --include_path raw/ --tasks gpqa_diamond_cot_zeroshot_clean \
  --gen_kwargs "max_gen_toks=16384,temperature=0" --output_path results/final_gpqa --log_samples
```

Raw harness output (aggregate results JSON, AutomationBench JSON, custom GPQA task config) is in
[`raw/`](raw/).
