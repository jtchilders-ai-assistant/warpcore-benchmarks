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

## Throughput / latency — **TODO**

The quality harnesses above record **no tok/s, TTFT, or ITL**. lm-eval measures only correctness;
AutomationBench records token *counts* and aggregate model time but not per-request latency or
sustained tok/s.

Planned: run `vllm bench serve` inside the container across a concurrency sweep and record:
- output tok/s (mean, peak, max sustained)
- TTFT (time to first token)
- ITL / TPOT (inter-token latency)
- per-concurrency scaling (c=1, 8, 16, 32, …)

Prior ad-hoc baseline (from the maintainer's ops notes, to be re-measured properly here):
c=1 → ~34 tok/s/user, TTFT ~316 ms, ITL ~29 ms; c=32 → ~219 tok/s aggregate output, TPOT ~137 ms,
TTFT rose to ~2.3 s (prefill queuing).

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
