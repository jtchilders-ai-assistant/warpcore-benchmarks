# nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 — Warpcore Benchmark Card

**Original campaign:** 2026-07-30

**Controlled current-profile recovery:** 2026-09-09

**Model:** `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` (NVFP4, MoE — 120B total / ~12B active)

**Host:** Warpcore, NVIDIA DGX Spark / GB10 — see [../../HARDWARE.md](../../HARDWARE.md)

**Current profile (used unchanged for the 2026-09-09 GPQA recovery, throughput sweep, and clean
GSM8K run):** vLLM `0.27.1`, container `vllm_node`, **MARLIN** MoE backend, FP8 KV cache,
`--max-num-seqs 128`, `--max-model-len 262144`, checkpoint `super_v3` reasoning-parser plugin,
and `qwen3_coder` tool parser. Historical July scores below used vLLM
`0.17.1rc1.dev96+g57431d823.d20260312`, MARLIN/TRITON, FP8 KV, and initially
`--max-num-seqs 24`; they are labeled separately where comparability matters.
**Endpoint:** `http://csi370295.alcf.anl.gov:8000/v1/chat/completions`

> **Serving note (important):** the stock `spark-vllm-docker` recipe forces `--moe-backend cutlass`,
> which **fails to initialize** on this checkpoint (`ValueError: NvFp4 MoE backend 'VLLM_CUTLASS' does
> not support ... no act_and_mul MLP layer`). Letting vLLM auto-select picks `FLASHINFER_CUTLASS`,
> which loads but **crashes on the first decode** with `cudaErrorIllegalInstruction`. The GB10-stable
> path is **`--moe-backend marlin`** (survives sustained concurrent load). See [../../ISSUES.md](../../ISSUES.md).

---

## Quality — lm-evaluation-harness 0.4.9.1

Settings: `--model local-chat-completions --apply_chat_template`, 0-shot CoT, `temperature=0`,
`max_gen_toks=8192` (16384 for GPQA). Full test sets (no `--limit`). Nemotron is a reasoning model
(emits a `reasoning` channel; final answer in `content`).

| Benchmark | Metric | Nemotron-3-Super-120B | gpt-oss-120b | n |
| --------- | ------ | :-------------------: | :----------: | - |
| GSM8K (CoT, 0-shot), historical July run | exact_match, flexible-extract | **76.65%** ±1.17 | 83.70% | 1319 |
| GSM8K (CoT, 0-shot), clean current-profile run | exact_match, answer-line | **95.83%** ±0.55 | — | 1319 |
| GSM8K (CoT, 0-shot), clean current-profile run | exact_match, flexible-fallback | **96.89%** ±0.48 | — | 1319 |
| IFEval | prompt-level strict acc | **85.40%** ±1.52 | 83.73% | 541 |
| IFEval | inst-level strict acc | 88.13% | 89.09% | 541 |
| IFEval | prompt-level loose acc | 88.35% ±1.38 | 86.69% | 541 |
| IFEval | inst-level loose acc | 90.05% | 91.01% | 541 |
| GPQA-Diamond (CoT, clean-extract), original 16K | exact_match, answer-line | **63.64%** ±3.43 | 72.73% | 198 |
| GPQA-Diamond, corrected 64K composite | exact_match, answer-line | **73.74%** | 72.73% | 198 |

**Notes**
- **Clean GSM8K (2026-09-09):** the strict answer-line task scored **1264/1319 = 95.83%**;
  its more permissive fallback scored **1278/1319 = 96.89%**. Raw evidence is retained under
  `raw/quality/gsm8k_clean_2026-09-09/`. One item (`doc_id=119`) had empty content and is counted
  wrong in both aggregates. An exact-prompt replay reproduced the empty result with
  `finish_reason=length`, 8,192 completion tokens, and 26,634 characters retained in the reasoning
  field, confirming a budget-limited residual rather than a parser-empty response. The replay is
  diagnostic only and is not silently substituted into the published full-run score.
- **Historical GSM8K strict-match reads 0% — format artifact, not capability** (same as gpt-oss): the reasoning
  model doesn't emit the rigid `#### <n>` tail. flexible-extract is the correct metric.
- **GPQA** used the same custom clean-extraction task as the gpt-oss card (anchored
  `The answer is (X)` regex, see `raw/gpqa_diamond_cot_zeroshot_clean.yaml`). The original 16K run
  scored **63.64% (126/198)** and returned empty content on 56 items. Replaying exactly those 56
  stored prompts at 64K recovered 25 final answers, 20 correct, producing the measured corrected
  composite **73.74% (146/198)**. The other 31 all ended `finish_reason=length` at exactly 65,536
  completion tokens and remain counted wrong. Thus 73.74% is the 64K operational score, not a
  no-limit capability ceiling. Retained evidence: `raw/quality/gpqa/replay_64k_2026-09-09/`.
- **Compute cost (notable):** the 56-item replay required **44,858.7 s (12.46 h)** at concurrency 8;
  its median completion length was the full 65,536-token ceiling, and the longest completed response
  used 61,044 tokens. This model's tail cost is operationally significant on one GB10.

## Comparison summary vs the gpt-oss-120b baseline

On GPQA-Diamond, the corrected 64K composite is **73.74% (146/198)** versus gpt-oss-120b's
**72.73%**—a 1.01-point descriptive difference, not evidence of superiority without a paired analysis.
Nemotron-3-Super's clean current-profile GSM8K score is **95.83% (1264/1319)** on the anchored
answer-line metric (**96.89%** with the explicitly labeled flexible fallback). This is not directly
comparable to the old gpt-oss figure until gpt-oss is rerun through the same clean task. Nemotron-Super
is operationally expensive: 31/198 GPQA prompts still fail to emit a final answer at 64K, and its
56-item recovery alone consumed 12.46 hours at concurrency 8. On the current evidence it is
competitive on GPQA, strong on IFEval and clean GSM8K, but not yet a clear replacement for gpt-oss on
this hardware because throughput and latency differ substantially and clean-task cross-model parity
is incomplete.

## Throughput / latency — `vllm bench serve` concurrency sweep

The clean **current-profile** sweep (2026-09-09) ran end-to-end under the same vLLM 0.27.1 profile as
the GPQA recovery and clean GSM8K run. It used the on-box raw-completions path with fixed **512 input /
256 output** tokens, `--ignore-eos`, and concurrency 1→128. Every request succeeded and the engine was
idle before every level.

| Concurrency | Successful | Output tok/s | Mean TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) |
| ----------: | ---------: | -----------: | -------------: | ------------: | -------------: |
| 1 | 32/32 | 15.75 | 486 | 644 | 61.83 |
| 2 | 32/32 | 27.40 | 793 | 1,147 | 70.15 |
| 4 | 32/32 | 45.13 | 1,223 | 1,394 | 84.18 |
| 8 | 32/32 | 64.65 | 1,982 | 2,582 | 116.35 |
| 16 | 48/48 | 95.42 | 3,048 | 4,876 | 156.06 |
| 24 | 72/72 | 117.73 | 3,702 | 7,177 | 189.55 |
| 32 | 96/96 | 137.09 | 4,219 | 9,536 | 217.03 |
| 48 | 144/144 | 163.66 | 5,180 | 14,382 | 272.84 |
| 64 | 192/192 | 184.27 | 6,112 | 19,275 | 322.64 |
| 96 | 288/288 | 218.23 | 7,933 | 29,626 | 406.79 |
| **128** | **384/384** | **244.70** | **9,822** | **39,456** | **481.09** |

Throughput was still rising at c=128, so **244.70 output tok/s is a measured floor, not a proven
hardware ceiling or plateau**. Latency deteriorated sharply: at c=128, P99 TTFT was 39.5 s and mean
TPOT was 481 ms. The result is therefore an offline saturation operating point, not an interactive
recommendation. Raw evidence: [`raw/throughput_sweep_current_profile.log`](raw/throughput_sweep_current_profile.log)
and [`raw/vllm_sweep_current_profile.sh`](raw/vllm_sweep_current_profile.sh).

### Historical July sweep (different vLLM build/profile)

Measured 2026-07-30/31, **on-box** (inside `vllm_node` against `localhost:8000` → server ceiling,
network excluded). Raw-completions path (`--backend openai --endpoint /v1/completions --ignore-eos`),
fixed shape **512 input / 256 output** tokens. Concurrency swept 1→128 until output tok/s plateaus.
(The 1→24 levels were run at `--max-num-seqs 24`; 32→128 after raising the server to
`--max-num-seqs 128` — the server was restarted between the two halves, hence the c=32 warmup spike.)

| Concurrency | Output tok/s | Δ vs prev | Mean TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) |
| ----------: | -----------: | --------: | -------------: | ------------: | -------------: |
| 1 | 15.2 | — | 447 | 455 | 64.1 |
| 2 | 26.7 | +74.8% | 718 | 895 | 72.5 |
| 4 | 42.1 | +58.0% | 1422 | 1455 | 89.8 |
| 8 | 62.3 | +47.8% | 2279 | 2630 | 119.9 |
| 16 | 86.6 | +39.1% | 3377 | 5030 | 171.9 |
| 24 | 111.3 | +28.5% | 4091 | 7514 | 199.9 |
| 32 | 110.5 | −0.8%¹ | 9466 | 27157 | 252.9 |
| 48 | 140.6 | +27.3% | 5698 | 15113 | 318.9 |
| 64 | 156.2 | +11.1% | 6703 | 20416 | 382.8 |
| 96 | 184.5 | +18.1% | 8720 | 31720 | 484.1 |
| **128** | **189.9** | +2.9% | 14403 | 123619 | 575.1 |

¹ c=32 is a warm-up transient right after the server restart (note the 27 s P99 TTFT); the clean trend
resumes at c=48.

**Plateau: ~185–190 tok/s output, reached at concurrency ~96–128** (c=96→128 adds only +2.9%, the knee).
Three operating points:
- **Single-stream (c=1):** 15.2 tok/s/user, TTFT 447 ms, TPOT 64 ms — snappy.
- **Balanced (c≈24):** ~111 tok/s aggregate, TPOT ~200 ms, P99 TTFT ~7.5 s.
- **Max throughput (c≈128):** ~190 tok/s aggregate, but latency collapses — TPOT ~575 ms and
  P99 TTFT ~124 s (deep batch/offline regime; not usable interactively).

**Historical comparison vs the gpt-oss-120b card (~709 tok/s @ c≈256):** under the July profile,
**gpt-oss-120b sustained ~3.7× Nemotron's measured throughput** (~709 vs ~190 tok/s) and was far
snappier per stream (c=1: gpt-oss 34 tok/s / TTFT 71 ms vs Nemotron 15 tok/s / TTFT 447 ms). The
current-profile Nemotron sweep above supersedes ~190 for present operations but does not establish a
ceiling because its c=128 endpoint was still rising. Historical raw per-level output:
[`raw/throughput_sweep/sweep.log`](raw/throughput_sweep/sweep.log) (c1–24) and
[`sweep_hi.log`](raw/throughput_sweep/sweep_hi.log) (c32–128).

## Agentic coding — pi-30 (Fleet-30)

[`rick-stevens-ai/pi-30`](https://github.com/rick-stevens-ai/pi-30): 30 agentic-coding problems, each
solved via a full `pi` agent tool-loop (read/write/bash), graded **solely by verifier exit codes**.
Run from a client Mac against the warpcore endpoint, `PI_TIMEOUT=600` (raised from the default 360 s
for Nemotron's long reasoning tails), single-shot canonical.

| Model | pi-30 score | Failures |
| ----- | :---------: | -------- |
| **Nemotron-3-Super-120B** | **30 / 30** | none |
| Qwen3.6-35B-A3B | 29 / 30 | P2 (LRU cache) |
| gpt-oss-120b | 30 / 30 | none (on the crash-fixed image) |

**Nemotron-3-Super ties gpt-oss-120b for the top agentic-coding score** — a perfect 30/30, including
P5 (101.6 GFLOP/s) and the LRU-cache problem (P2) that Qwen3.6 missed. This is a notable **reversal**
of the quality-benchmark ranking (where Nemotron was weakest): the iterate-until-green agent loop
rewards its verbose, methodical reasoning, whereas GSM8K/GPQA penalize its verbosity/latency. Raw
per-problem log: [`raw/pi30/RESULTS.txt`](raw/pi30/RESULTS.txt).

## Reproduce

```bash
python3 -m venv lmeval-venv && source lmeval-venv/bin/activate
pip install "lm-eval[api]" "numpy<2" langdetect immutabledict nltk
# Apply the crash-survival patches (see the lm-eval-vllm-endpoint skill / ISSUES.md) before long runs.

MODEL="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
BASE="http://csi370295.alcf.anl.gov:8000/v1/chat/completions"

# GSM8K + IFEval (c=8 ok)
OPENAI_API_KEY=dummy lm_eval --model local-chat-completions \
  --model_args "model=${MODEL},base_url=${BASE},num_concurrent=8,max_retries=3,tokenized_requests=False,timeout=900" \
  --apply_chat_template --tasks gsm8k_cot_zeroshot --gen_kwargs "max_gen_toks=8192,temperature=0" \
  --output_path results/final_gsm8k --log_samples

# GPQA (gated; needs HF_TOKEN; LOW concurrency + long timeout for the very long reasoning tails)
HF_TOKEN=... OPENAI_API_KEY=dummy lm_eval --model local-chat-completions \
  --model_args "model=${MODEL},base_url=${BASE},num_concurrent=4,max_retries=8,tokenized_requests=False,timeout=3600" \
  --apply_chat_template --include_path raw/ --tasks gpqa_diamond_cot_zeroshot_clean \
  --gen_kwargs "max_gen_toks=16384,temperature=0" --output_path results/final_gpqa --log_samples
```

Raw harness output (aggregate results JSON, custom GPQA task config) is in [`raw/`](raw/).
