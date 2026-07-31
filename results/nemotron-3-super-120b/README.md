# nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 — Warpcore Benchmark Card

**Date:** 2026-07-30
**Model:** `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` (NVFP4, MoE — 120B total / ~12B active)
**Host:** Warpcore, NVIDIA DGX Spark / GB10 — see [../../HARDWARE.md](../../HARDWARE.md)
**Serving:** vLLM `0.17.1rc1.dev96+g57431d823.d20260312`, container `vllm_node`,
**MARLIN** NVFP4 MoE backend + TRITON attention, `--kv-cache-dtype fp8`, `--max-num-seqs 24`,
`--max-model-len 262144`
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
| GSM8K (CoT, 0-shot) | exact_match, flexible-extract | **76.65%** ±1.17 | 83.70% | 1319 |
| IFEval | prompt-level strict acc | **85.40%** ±1.52 | 83.73% | 541 |
| IFEval | inst-level strict acc | 88.13% | 89.09% | 541 |
| IFEval | prompt-level loose acc | 88.35% ±1.38 | 86.69% | 541 |
| IFEval | inst-level loose acc | 90.05% | 91.01% | 541 |
| GPQA-Diamond (CoT, clean-extract) | exact_match, answer-line | **63.64%** ±3.43 | 72.73% | 198 |

**Notes**
- **GSM8K strict-match reads 0% — format artifact, not capability** (same as gpt-oss): the reasoning
  model doesn't emit the rigid `#### <n>` tail. flexible-extract is the correct metric.
- **GPQA** used the same custom clean-extraction task as the gpt-oss card (anchored
  `The answer is (X)` regex, see `raw/gpqa_diamond_cot_zeroshot_clean.yaml`). Both the `answer-line`
  and `flexible-fallback` filters agree at **63.64%**, so parsing is deterministic.
- **Verbose-reasoning / null-content behavior (notable):** on the hardest GPQA items Nemotron-3-Super
  frequently exhausts even a 16384-token budget in its reasoning channel **without emitting a final
  answer** (vLLM returns `content: null`; lm-eval scores these 0). This depresses the GPQA score
  somewhat but is a real, reproducible characteristic of the model on this hardware, not a harness bug.
- **Compute cost (notable):** GPQA-Diamond took **9h15m** (198 items, per-item 3–6 min) because of
  these very long reasoning traces — dramatically more expensive to evaluate than gpt-oss-120b.

## Comparison summary vs the gpt-oss-120b baseline

gpt-oss-120b is the stronger model on this suite: it wins **GSM8K (+7.1)** and **GPQA-Diamond (+9.1)**
decisively, and ties inst-level IFEval. Nemotron-3-Super-120B's only win is **prompt-level IFEval
(+1.7)**. Combined with its much heavier per-query compute cost (long reasoning tails, high
null-content rate on hard questions), Nemotron-3-Super is **not** an upgrade over the incumbent
gpt-oss-120b on this hardware for these workloads.

## Throughput / latency — `vllm bench serve` concurrency sweep

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

**Comparison vs the gpt-oss-120b card (~709 tok/s @ c≈256):** now an apples-to-apples uncapped sweep —
**gpt-oss-120b sustains ~3.7× Nemotron's peak throughput** (~709 vs ~190 tok/s) and is far snappier
per stream (c=1: gpt-oss 34 tok/s / TTFT 71 ms vs Nemotron 15 tok/s / TTFT 447 ms). This is consistent
with Nemotron's heavier per-token compute (larger active-expert footprint). Raw per-level output:
[`raw/throughput_sweep/sweep.log`](raw/throughput_sweep/sweep.log) (c1–24) and
[`sweep_hi.log`](raw/throughput_sweep/sweep_hi.log) (c32–128).

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
