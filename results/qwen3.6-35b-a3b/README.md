# Qwen/Qwen3.6-35B-A3B-FP8 — Warpcore Benchmark Card

**Date:** 2026-08-01
**Model:** `Qwen/Qwen3.6-35B-A3B-FP8` (FP8, MoE — ~35B total / ~3B active)
**Host:** Warpcore, NVIDIA DGX Spark / GB10 — see [../../HARDWARE.md](../../HARDWARE.md)
**Serving:** vLLM `0.17.1rc1.dev96+g57431d823.d20260312`, container `vllm_node`,
**TRITON** FP8 MoE backend + flashinfer attention, `--kv-cache-dtype fp8`, default `--max-num-seqs`
(256), `--max-model-len 262144`, `--reasoning-parser qwen3`
**Endpoint:** `http://csi370295.alcf.anl.gov:8000/v1/chat/completions`

> Only ~37 GB of weights — by far the smallest of the three carded models (¼ the size of the 120B
> models), leaving large KV-cache headroom on the GB10. FP8 MoE served cleanly on the auto-selected
> TRITON backend — no CUTLASS/NVFP4 crash issues (those are NVFP4-specific; see [../../ISSUES.md](../../ISSUES.md)).

---

## Quality — lm-evaluation-harness 0.4.9.1

Settings: `--model local-chat-completions --apply_chat_template`, 0-shot CoT, `temperature=0`.
Full test sets (no `--limit`).

| Benchmark | Metric | Qwen3.6-35B | gpt-oss-120b | Nemotron-3-Super-120B | n |
| --------- | ------ | :---------: | :----------: | :-------------------: | - |
| GSM8K (CoT, 0-shot) | exact_match, flexible | **97.04%** ±0.47 | 83.70% | 76.65% | 1319 |
| IFEval | prompt-level strict acc | 84.84% ±1.52 | 83.73% | 85.40% | 541 |
| IFEval | inst-level strict acc | 87.65% | 89.09% | 88.13% | 541 |
| IFEval | prompt-level loose acc | 87.43% | 86.69% | 88.35% | 541 |
| IFEval | inst-level loose acc | 89.33% | 91.01% | 90.05% | 541 |
| GPQA-Diamond (CoT, clean-extract) | exact_match, answer-line | **82.32%** ±2.72 | 72.73% | 63.64% | 198 |

**Headline:** at **~¼ the size** of the two 120B models, Qwen3.6-35B-A3B **wins GSM8K by 13–20 points**
and **GPQA-Diamond by 10–19 points**, and effectively ties on IFEval. It is the clear value standout
of the three models on this suite.

### Methodology notes (important — read before comparing)

Both GSM8K and GPQA required care because `--reasoning-parser qwen3` routes chain-of-thought into a
separate `reasoning_content` channel, which interacts badly with number/MCQ extraction. All results
above are **verified** (parse checked against the model's actual stated answers on smoke samples).

- **GSM8K used a custom clean-extract task** (`raw/gsm8k_cot_zeroshot_clean.yaml`), not the stock
  `gsm8k_cot_zeroshot` the gpt-oss/Nemotron cards used. Reason: with the reasoning parser on, stock
  flexible-extract returned a **bogus 39.88%** — 53% of responses had `content: null` (answer went to
  the reasoning channel) and `$`/bold formatting defeated the stock regex. The custom task's
  flexible-fallback (last-number, format-robust) gives **97.04%**, verified on a smoke (39/40
  last-number == gold, the one miss a real model error). This "last number in output" definition is the
  same as stock flexible-extract, just format-robust, so it stays comparable.
- **GPQA is reported in NON-THINKING mode (`enable_thinking=false`): 82.32%.** In the model's **default
  thinking mode**, GPQA scored only **33.84%** — but that is an *artifact*: on hard GPQA questions the
  model exhausts the 16 384-token budget inside the reasoning channel and emits **no final answer**
  (`content: null` on **61%** of items, `finish_reason=length`). With thinking off the model still
  produces full step-by-step reasoning **in `content`**, finishes every item (**0% null**), and scores
  82.32% (163/198, verified). The 33.84% thinking-mode figure is retained in `raw/` as an operational
  data point: **running Qwen3.6 in default thinking mode with a ≤16k budget is not viable for hard
  long-form MCQ on this hardware** — either disable thinking or budget ≥32k tokens.

## Throughput / latency — `vllm bench serve` concurrency sweep

Measured 2026-08-01, **on-box** (inside `vllm_node` against `localhost:8000` → server ceiling, network
excluded). Raw-completions path (`--backend openai --endpoint /v1/completions --ignore-eos`), fixed
shape **512 input / 256 output** tokens. Concurrency swept 1→128 (no `--max-num-seqs` cap on this
recipe).

| Concurrency | Output tok/s | Δ vs prev | Mean TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) |
| ----------: | -----------: | --------: | -------------: | ------------: | -------------: |
| 1 | 47.5 | — | 155 | 170 | 20.5 |
| 2 | 75.1 | +58.3% | 254 | 301 | 25.7 |
| 4 | 112.2 | +49.4% | 391 | 460 | 34.2 |
| 8 | 146.9 | +30.9% | 1554 | 4178 | 48.6 |
| 16 | 211.1 | +43.7% | 1206 | 1291 | 71.4 |
| 24 | 244.9 | +16.0% | 1968 | 2082 | 90.7 |
| 32 | 280.0 | +14.3% | 2603 | 2725 | 104.5 |
| 48 | 329.4 | +17.7% | 3408 | 3996 | 132.8 |
| 64 | 370.7 | +12.5% | 4525 | 5350 | 155.4 |
| 96 | 436.9 | +17.9% | 6055 | 8080 | 196.6 |
| **128** | **486.7** | +11.4% | 6925 | 10960 | 236.5 |

**Peak ~487 tok/s at c=128** (still climbing +11.4% at the last step — the true plateau is a bit higher;
128 chosen to match the Nemotron sweep range). Operating points:
- **Single-stream (c=1):** 47.5 tok/s/user, TTFT 155 ms, TPOT 21 ms — the snappiest of the three models.
- **Max measured (c=128):** ~487 tok/s aggregate, TPOT ~237 ms, P99 TTFT ~11 s.

**Comparison at matched concurrency (all on-box, 512/256):** Qwen3.6-35B is by far the fastest — its
single-stream 47.5 tok/s beats gpt-oss (34 tok/s) and triples Nemotron (15 tok/s), and its c=128
throughput (~487 tok/s) is **~2.6× Nemotron's** (~190 @ c128). gpt-oss-120b still reaches a higher
absolute peak (~709 tok/s, but only at c≈256). Given Qwen3.6 is ¼ the size with far lower latency AND
higher quality scores, it is the efficiency winner on this hardware. Raw per-level output:
[`raw/throughput_sweep/sweep.log`](raw/throughput_sweep/sweep.log).

## Agentic coding — pi-30 (Fleet-30)

[`rick-stevens-ai/pi-30`](https://github.com/rick-stevens-ai/pi-30): 30 agentic-coding problems, each
solved via a full `pi` agent tool-loop (read/write/bash), graded **solely by verifier exit codes**.
Run from a client Mac against the warpcore endpoint, `PI_TIMEOUT=360`, single-shot canonical.

| Model | pi-30 score | Failures |
| ----- | :---------: | -------- |
| Nemotron-3-Super-120B | 30 / 30 | none |
| **Qwen3.6-35B-A3B** | **29 / 30** | P2 (LRU cache) |
| gpt-oss-120b | 29 / 30 | P5 (matmul GFLOP/s — GB10 throughput ceiling) |

Qwen3.6-35B ties gpt-oss-120b at **29/30** — matching the 120B incumbent on agentic coding at ¼ the
size. Its one miss (P2, LRU cache) is a genuine model error (staging verified). Notably it **passed
P5** (60.0 GFLOP/s matmul), gpt-oss's one failure. Raw per-problem log:
[`raw/pi30/RESULTS.txt`](raw/pi30/RESULTS.txt).

## Reproduce

```bash
python3 -m venv lmeval-venv && source lmeval-venv/bin/activate
pip install "lm-eval[api]" "numpy<2" langdetect immutabledict nltk
# Apply the crash-survival + None-guard patches (see lm-eval-vllm-endpoint skill / ISSUES.md).

MODEL="Qwen/Qwen3.6-35B-A3B-FP8"
BASE="http://csi370295.alcf.anl.gov:8000/v1/chat/completions"

# GSM8K (custom clean task; report flexible-fallback)
OPENAI_API_KEY=dummy lm_eval --model local-chat-completions \
  --model_args "model=${MODEL},base_url=${BASE},num_concurrent=6,max_retries=8,tokenized_requests=False,timeout=1800" \
  --apply_chat_template --include_path raw/ --tasks gsm8k_cot_zeroshot_clean \
  --gen_kwargs "max_gen_toks=8192,temperature=0" --output_path results/gsm8k --log_samples

# GPQA no-think (needs the LMEVAL_NO_THINK payload patch injecting chat_template_kwargs.enable_thinking=false)
HF_TOKEN=... LMEVAL_NO_THINK=1 OPENAI_API_KEY=dummy lm_eval --model local-chat-completions \
  --model_args "model=${MODEL},base_url=${BASE},num_concurrent=8,max_retries=8,tokenized_requests=False,timeout=1800" \
  --apply_chat_template --include_path raw/ --tasks gpqa_diamond_cot_zeroshot_clean \
  --gen_kwargs "max_gen_toks=16384,temperature=0" --output_path results/gpqa_nothink --log_samples
```

Raw harness output (results JSON for both GPQA modes, custom task configs) is in [`raw/`](raw/).
