# Intel/Qwen3.5-122B-A10B-int4-AutoRound — Warpcore Benchmark Card

**Date:** 2026-08-10
**Model:** `Intel/Qwen3.5-122B-A10B-int4-AutoRound` — MoE, **122B total / ~10B active**,
**hybrid Mamba + attention** arch (`Qwen3_5MoeForConditionalGeneration`). Intel AutoRound
**INT4** (W4) quantization of `Qwen/Qwen3.5-122B-A10B`. This is the model behind the
[Reddit "Qwen 3.5 122B A10B running 50 tok/s on DGX Spark"](https://www.reddit.com/r/LocalLLaMA/comments/1sko0ft/)
report; this card is the independent Warpcore measurement of that claim.
**Host:** Warpcore, NVIDIA DGX Spark / GB10 — see [../../HARDWARE.md](../../HARDWARE.md)
**Serving:** vLLM (container `vllm_node`, image `vllm-node`), **MARLIN** INT4 MoE kernel
(`GPTQMarlinLinearMethod`, vLLM reports `quantization=inc`), `--enable-prefix-caching`,
`--max-model-len 262144`, `--gpu-memory-utilization 0.8`, `--max-num-batched-tokens 8192`,
`--reasoning-parser qwen3`, `--tool-call-parser qwen3_xml`, `--enable-auto-tool-choice`,
`--chat-template unsloth.jinja`, `--tensor-parallel-size 1` (single GB10, solo mode).
**Endpoint:** `http://csi370295.alcf.anl.gov:8000/v1` (raw completions used for the sweep).

> **Single-Spark, INT4 is the only fit.** At INT4 the weights load in **62.65 GiB**, leaving
> **26.26 GiB for KV cache → 285,056 tokens (4.14× concurrency at the full 262,144-token context)**.
> Fits comfortably on one GB10. The **FP8** sibling (`Qwen/Qwen3.5-122B-A10B-FP8`, ~125 GiB of
> weights) does **NOT** fit a single 128 GB Spark — that recipe is written for a 2-Spark cluster
> (`tensor_parallel: 2` + Ray). INT4 corroborates the Reddit poster's "quant 4, int4 with MTP
> headers" choice as the practical sweet spot.

> **Serving notes / gotchas** (full write-up in [../../ISSUES.md](../../ISSUES.md)):
> - **Tokenizer class trap.** The Intel int4 repo's `tokenizer_config.json` declares
>   `tokenizer_class: TokenizersBackend`, which this container's (older) vLLM/transformers does not
>   recognize → `ValueError: Tokenizer class TokenizersBackend does not exist`. **Fix:** override the
>   tokenizer to the base repo, `--tokenizer Qwen/Qwen3.5-122B-A10B` (identical `Qwen2Tokenizer`
>   vocab). The **`vllm bench serve` client also needs its own `--tokenizer` flag** for the same
>   reason (it loads the tokenizer to build the random dataset).
> - **Solo override needed.** Every recipe in `spark-vllm-docker` defaults to `tensor_parallel: 2`
>   + Ray (2-Spark cluster). For single-Warpcore, a solo recipe with `tensor_parallel: 1` and no Ray
>   backend is required; the `-tp` CLI shorthand collides with `-t` in the launcher, so set it in the
>   recipe YAML, not on the command line.
> - **INT4 MoE runs through MARLIN** (`Using MarlinLinearKernel for GPTQMarlinLinearMethod`) — the
>   GB10-stable path, consistent with the other quantized MoE models on this box.
> - **Reasoning-parser split is imperfect.** The model emits `<think>…</think>` inline but the
>   `qwen3` reasoning parser + `unsloth.jinja` template do not cleanly route it into
>   `reasoning_content` (it stays empty). Output `content` is correct; only the *separation* is off.
> - **MTP not enabled.** This baseline was run **without** Multi-Token Prediction / speculative
>   decoding for a clean number; enabling MTP is the lever to chase the Reddit "~50 tok/s" figure
>   (see *Not yet measured* below).

---

## Smoke test (functional verification) — PASS

Verified end-to-end before benchmarking:

| Check | Result |
| ----- | ------ |
| Arithmetic (`2+2`, small budget) | ✅ `content: "4"`, `finish_reason: stop` |
| Factual (`capital of France`) | ✅ correct, `finish_reason: stop` |
| Reasoning-parser split (`qwen3`) | ⚠️ works but `reasoning_content` empty — `<think>` tags appear inline in content (cosmetic, see notes) |
| Stability under load | ✅ 384/384 requests OK at c=128/192/256, zero failures across the full sweep |

## Throughput / latency — `vllm bench serve` concurrency sweep

Measured 2026-08-10, **on-box** (inside `vllm_node` against `localhost:8000` → server ceiling,
network excluded). Raw-completions path (`--backend openai --endpoint /v1/completions --ignore-eos`),
fixed shape **512 input / 256 output** tokens. Concurrency swept 1→256. `--max-num-seqs` = vLLM
default (256), matching the cap under which gpt-oss-120b's plateau was measured.

| Concurrency | Output tok/s | Δ vs prev | Mean TTFT (ms) | Median TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) |
| ----------: | -----------: | --------: | -------------: | ---------------: | ------------: | -------------: |
| 1 | 26.9 | — | 356 | 352 | 376 | 35.9 |
| 2 | 44.2 | +64.0% | 574 | 577 | 597 | 43.2 |
| 4 | 63.0 | +42.5% | 1031 | 1034 | 1169 | 59.7 |
| 8 | 81.5 | +29.4% | 2695 | 2119 | 5289 | 88.0 |
| 16 | 109.4 | +34.3% | 3388 | 3540 | 3551 | 133.5 |
| 32 | 141.1 | +28.9% | 6249 | 5727 | 7220 | 203.1 |
| 48 | 168.7 | +19.6% | 7811 | 8570 | 10736 | 254.6 |
| 64 | 186.9 | +10.7% | 9193 | 9190 | 14478 | 307.4 |
| 96 | 208.7 | +11.7% | 11620 | 11014 | 22691 | 415.4 |
| 128 | 226.2 | +8.4% | 13555 | 11895 | 30638 | 513.6 |
| **192** | **227.7** | **+0.7%** | 58250 | 24568 | 167018 | 525.6 |
| 256 | 223.2 | −2.0% | 107955 | 159715 | 183445 | 536.3 |

**Sustained ceiling ≈ 228 tok/s at c≈128–192.** Throughput flattens at c=128→192 (+0.7%) and
**regresses** at c=256 (−2.0%) while TTFT explodes (mean 108 s, P99 183 s) — past ~192 concurrent
requests the engine is purely queuing, not doing more useful work. Three operating points:
- **Single-stream (c=1):** **26.9 tok/s/user**, TTFT **356 ms**, TPOT **36 ms** — this is the number
  that matches the Reddit report (~30 tok/s at int4, no MTP). Snappy for interactive chat.
- **Interactive SLO (TPOT < 100 ms): c ≤ 8** — ~81 tok/s aggregate, TPOT ~88 ms, median TTFT ~2.1 s.
  At c=16 TPOT crosses 100 ms.
- **Max aggregate (c≈192):** **~228 tok/s**, but TPOT ~526 ms and multi-second-to-minute TTFT —
  offline/batch only.

Raw per-level output: [`raw/throughput_sweep/sweep.log`](raw/throughput_sweep/sweep.log).
Sweep script: [`raw/throughput_sweep/vllm_sweep.sh`](raw/throughput_sweep/vllm_sweep.sh).

### Comparison vs the other Warpcore models

| Model | Size | c=1 tok/s | c=1 TTFT | Peak tok/s | at concurrency |
| ----- | ---- | --------: | -------: | ---------: | -------------- |
| nvidia/Nemotron-3.5-Lightning-30B-A3B | 30B / 3B act | 73.9 | 136 ms | ~719 (cap) | c=128 (still climbing) |
| openai/gpt-oss-120b | 120B / ~5B act | 34 | 71 ms | ~709 | c≈256 |
| Qwen/Qwen3.6-35B-A3B-FP8 | 35B / 3B act | — | — | ~487 | c=128 |
| **Intel/Qwen3.5-122B-A10B-int4** | **122B / ~10B act** | **26.9** | **356 ms** | **~228** | **c≈192 (plateau)** |
| nvidia/Nemotron-3-Super-120B-A12B | 120B / 12B act | 15 | 447 ms | ~190 | c≈128 |

**Where Qwen3.5-122B lands:** its **~10B active params/token** make it the second-heaviest decoder on
this box (after Nemotron-3-Super's 12B), and on the bandwidth-bound GB10 that active-parameter count is
what sets both single-stream speed and aggregate ceiling. At **26.9 tok/s single-stream** it sits just
above Nemotron-3-Super (15) and well below the 3B-active models. Its **~228 tok/s aggregate peak** is
~⅓ of gpt-oss-120b's, as expected: gpt-oss (~5B active) and the 3B-active MoEs decode roughly 2–3×
more tokens per unit bandwidth. **The trade is capability/footprint, not batch throughput** — this is
a 122B-parameter model running in 63 GiB with 256K context on a single Spark. For interactive
single-user or low-concurrency chat it is comfortable; for high-fan-out serving the smaller-active
models dominate. **MTP (not enabled here) is the untapped lever** — the Reddit report's ~50 tok/s used
it, so single-stream could plausibly ~1.5–2× with speculative decoding.

## Quality

**Not yet measured on Warpcore.** No lm-eval quality card (GSM8K / IFEval / GPQA-Diamond) or agentic
(pi-30 / SWE-bench) numbers have been produced for this model yet — this card covers **serving
bring-up + a functional smoke test + a throughput sweep**. Run the lm-eval pipeline
(`lm-eval-vllm-endpoint` skill) against the chat endpoint next to fill this in (note the reasoning
parser caveat above — verify `<think>` handling before trusting MCQ extraction).

## Not yet measured / next steps

- **MTP / speculative decoding.** The headline lever to reproduce the Reddit ~50 tok/s single-stream
  figure. Baseline here is deliberately MTP-off for a clean number.
- **Quality suite** (GSM8K, IFEval, GPQA-Diamond) via lm-eval against the chat endpoint.
- **Agentic** (pi-30, SWE-bench Verified 100-sample) — verify tool-calling (`qwen3_xml` parser) returns
  native `finish_reason: tool_calls` first, or every task scores 0.
- **Fix the `<think>` reasoning-parser split** (chat-template / parser mismatch) so `reasoning_content`
  is populated correctly — required for clean MCQ answer extraction in lm-eval.

## Reproduce

Serving (on warpcore, single-Spark solo recipe in `~/spark-vllm-docker`):
```bash
# solo recipe = tensor_parallel:1, no Ray, tokenizer override to the base repo
./run-recipe.sh qwen3.5-122b-int4-autoround-solo --solo --daemon
# key overrides baked into the solo recipe:
#   --tensor-parallel-size 1
#   --tokenizer Qwen/Qwen3.5-122B-A10B      # avoids TokenizersBackend load failure
#   --gpu-memory-utilization 0.8 --max-model-len 262144
```

Throughput sweep (raw completions, `--ignore-eos`, 512-in/256-out, on-box against `localhost:8000`):
```bash
docker exec vllm_node vllm bench serve \
  --base-url http://localhost:8000 \
  --model Intel/Qwen3.5-122B-A10B-int4-AutoRound \
  --tokenizer Qwen/Qwen3.5-122B-A10B \
  --backend openai --endpoint /v1/completions \
  --dataset-name random --random-input-len 512 --random-output-len 256 --ignore-eos \
  --num-prompts <3×C, cap 384> --max-concurrency <C> \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99
# swept C = 1 2 4 8 16 32 48 64 96 128 192 256
```
