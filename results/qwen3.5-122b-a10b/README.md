# Intel/Qwen3.5-122B-A10B-int4-AutoRound — Warpcore Benchmark Card

**Date:** 2026-09-27
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
| nvidia/Nemotron-3.5-Lightning-30B-A3B | 30B / 3B act | 73.9 | 136 ms | 926 (floor) | c=384 (still climbing) |
| openai/gpt-oss-120b | 120B / ~5B act | 34 | 71 ms | ~709 | c≈256 |
| Qwen/Qwen3.6-35B-A3B-FP8 | 35B / 3B act | — | — | ~487 | c=128 |
| **Intel/Qwen3.5-122B-A10B-int4** | **122B / ~10B act** | **26.9** | **356 ms** | **~228** | **c≈192 (plateau)** |
| nvidia/Nemotron-3-Super-120B-A12B | 120B / 12B act | 15.75 | 486 ms | **244.70 (floor)** | c=128 (still climbing) |

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

## Canonical `warpcore-v1` results

All quality runs used the contract runner against the chat-completions endpoint at temperature 0,
concurrency 4, measured aggregate throughput 70.04 output tok/s, and a 9,000 s client timeout.
The adapter pins model revision `3045d02bb737effc4581da91bddbad3be02934e4`, vLLM
`0.29.1rc1.dev427+g0748d3bd5.d20260920`, and image digest
`sha256:c154ad0a2575d6c42f8e05cba16ef255ce4ff54d537ad37987ca5b4215cb58b8`.

- **GSM8K:** **1,278/1,319 = 96.89%** answer-line exact match. Fourteen empty responses
  exhausted the frozen 8,192-token output ceiling and remain counted wrong; one other response
  ended `length` with nonempty content. Publication validation passed.
- **IFEval:** **465/541 = 85.95% prompt-strict**; instruction-strict 87.89%, prompt-loose
  88.17%, instruction-loose 89.33%. Thirty-eight responses exhausted the frozen 65,536-token
  output ceiling before emitting final content and remain counted wrong. The first launch failed
  before inference because loading the pinned task through a custom path broke its harness-relative
  import; it remains invalid. The replacement run completed all 541 prompts and passed publication
  validation.
- **GPQA-Diamond:** **164/198 = 82.83%** answer-line exact match. Four responses exhausted
  the frozen 65,536-token output ceiling without final content and remain counted wrong.
  Publication validation passed.
- **SWE-bench Verified:** **not measured canonically.** The mandatory suite-owned n=20
  qualification produced 17 nonempty `Submitted` patches, two `LimitsExceeded` outcomes at the
  250-step limit, and one `ContextWindowExceededError`. The latter requested 229,377 input plus
  32,768 output tokens, totaling 262,145 against the 262,144-token context. Because the contract
  requires 20/20 `Submitted`, no qualification record was sealed and canonical n=100 was not
  launched.

`Submitted` means a gradeable patch was produced; it does **not** mean the issue was solved. For
diagnostic context only, official grading of the nonqualifying n=20 run found 14 resolved, 3
unresolved, and 3 empty-patch instances. **14/20 is not the canonical suite score** and is not
published as the SWE-bench cell. The complete gate decision is retained in
[`qualification_decision.json`](qualification/warpcore-v1/swebench/qualification_decision.json).

## Throughput refresh — 2026-09-24

A fresh 512-input/256-output sweep on the campaign serving profile completed **1,280/1,280**
requests across c=1,2,4,8,16,32,48,64,96,128. It measured **27.18 tok/s at c=1**,
**70.04 tok/s at c=4**, and **234.54 tok/s at c=128**. The c=96→128 gain was still 3.70%,
so c=128 is a measured near-knee floor, not a demonstrated plateau; P99 TTFT there was 128.1 s.
The preserved log and script are under [`raw/throughput_sweep_20260924/`](raw/throughput_sweep_20260924/).
The older 2026-08 sweep above remains historical context and is not substituted for the fresh
campaign measurement.

## Remaining optional work

- **MTP / speculative decoding.** The baseline remains MTP-off; this is the lever relevant to the
  Reddit ~50 tok/s single-stream claim.
- A future SWE-bench n=100 campaign requires a new, fresh n=20 qualification that satisfies the
  unchanged 20/20 submission gate. Reusing or sealing the failed qualification is prohibited.

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
# swept C = 1 2 4 8 16 32 48 64 96 128
```
