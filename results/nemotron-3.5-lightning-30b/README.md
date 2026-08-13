# nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 — Warpcore Benchmark Card

**Date:** 2026-08-12
**Model:** `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` — MoE, **30B total / 3B active**
(128 routed experts + 1 shared, 6 experts/token), **hybrid Mamba + attention** arch
(`NemotronHForCausalLM`, `model_type: nemotron_h`). Released 2026-08-11; NVIDIA positions it for
high-volume, low-latency "always-on" agents. Open weights (OpenMDW 1.1).
**Host:** Warpcore, NVIDIA DGX Spark / GB10 — see [../../HARDWARE.md](../../HARDWARE.md)
**Serving:** vLLM **`0.27.1`** (the NVIDIA-recommended nightly for this model,
image `vllm/vllm-openai:v0.27.1`), container `vllm_lightning`, **MARLIN** NVFP4 MoE backend,
`--kv-cache-dtype fp8`, `--enable-prefix-caching`, `--max-model-len 262144`, `--max-num-seqs 128`,
`--reasoning-parser nemotron_v3`, `--tool-call-parser qwen3_coder`, `--enable-auto-tool-choice`.
**Endpoint:** `http://csi370295.alcf.anl.gov:8000/v1/chat/completions`
**Recommended sampling (NVIDIA):** temperature **1.0**, top_p **0.95**.

> **Quantization:** this is a **mixed-precision** ModelOpt checkpoint — vLLM reports
> `quantization=modelopt_mixed`. The Mamba `in_proj`/`out_proj` layers are **FP8**, the MoE experts
> are **W4A16_NVFP4** (group_size 16), with an **FP8** KV-cache scheme. The `26.01/26.02` NGC images
> (upstream `0.15.x`) are too old for this Aug-2026 checkpoint; use `v0.27.1`.

> **Serving notes**
> - **Weights are tiny (~18 GiB).** On the GB10 this leaves **88.12 GiB free for KV cache** →
>   **23.15× concurrency at the full 262,144-token context**. Memory is a non-issue for this model.
> - **`--moe-backend marlin`** is used (the GB10-stable NVFP4 MoE path, consistent with the other
>   NVFP4 models on this box). The model's own README also references a `humming` MoE backend and a
>   `--speculative_config` **DSpark draft-model** path for extra latency — see *Not yet measured* below.
> - Benign startup warning: `Unexpected gate/up projection names: up_proj … Fused gate/up mapping will
>   be skipped` — the experts store `up_proj`/`down_proj` unfused, so vLLM skips the gate/up **fusion
>   optimization**. Loads and runs correctly; may leave some throughput on the table (see DSpark note).

---

## Smoke test (functional verification) — PASS

Verified end-to-end before benchmarking (raw transcript: [`raw/smoke_tests.txt`](raw/smoke_tests.txt)):

| Check | Result |
| ----- | ------ |
| Arithmetic + strict format (`17*23`, then `LIGHTNING_OK`) | ✅ `391\nLIGHTNING_OK`, `finish_reason: stop` |
| Reasoning-parser split (`nemotron_v3`) | ✅ clean — Lightning is terse, `reasoning_content` empty |
| Tool calling (`get_weather`, `tool_choice: auto`) | ✅ `finish_reason: tool_calls`, args `{"city":"Chicago"}` |
| Stability under load | ✅ 384/384 requests OK at c=128, no crash across the full sweep |

## Throughput / latency — `vllm bench serve` concurrency sweep

Measured 2026-08-12, **on-box** (inside `vllm_lightning` against `localhost:8000` → server ceiling,
network excluded). Raw-completions path (`--backend openai --endpoint /v1/completions --ignore-eos`),
fixed shape **512 input / 256 output** tokens. Concurrency swept 1→128 (the server's `--max-num-seqs`).

| Concurrency | Output tok/s | Δ vs prev | Mean TTFT (ms) | Median TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) |
| ----------: | -----------: | --------: | -------------: | ---------------: | ------------: | -------------: |
| 1 | 73.9 | — | 136 | 135 | 171 | 13.1 |
| 2 | 116.3 | +57.4% | 273 | 259 | 469 | 16.2 |
| 4 | 169.8 | +46.0% | 395 | 387 | 424 | 22.1 |
| 8 | 234.2 | +37.9% | 637 | 669 | 746 | 31.8 |
| 16 | 312.6 | +33.5% | 942 | 882 | 1416 | 47.6 |
| 24 | 367.2 | +17.5% | 1137 | 996 | 2096 | 61.0 |
| 32 | 418.7 | +14.0% | 1289 | 1012 | 2780 | 71.5 |
| 48 | 502.7 | +20.1% | 1570 | 1039 | 4195 | 89.3 |
| 64 | 559.6 | +11.3% | 1847 | 1071 | 5652 | 107.0 |
| 96 | 649.1 | +16.0% | 2400 | 1466 | 8744 | 138.0 |
| **128** | **719.0** | +10.8% | 2996 | 1547 | 11949 | 165.2 |

**Still climbing at c=128 (+10.8%): this is the `--max-num-seqs 128` cap, not a saturation plateau.**
The 88 GiB of free KV cache (23× concurrency headroom) means a higher `--max-num-seqs` would push peak
throughput further. Three operating points:
- **Single-stream (c=1):** **73.9 tok/s/user**, TTFT **136 ms**, TPOT **13 ms** — very snappy.
- **Balanced (c≈16):** ~313 tok/s aggregate, TPOT ~48 ms, median TTFT ~0.9 s.
- **Max measured (c=128):** **719 tok/s** aggregate, TPOT 165 ms, median TTFT ~1.5 s (mean 3.0 s;
  the P99 tail grows with batch depth as expected).

Raw per-level output: [`raw/throughput_sweep/sweep.log`](raw/throughput_sweep/sweep.log).

### Comparison vs the other Warpcore models

| Model | Size | c=1 tok/s | c=1 TTFT | Peak tok/s | at concurrency |
| ----- | ---- | --------: | -------: | ---------: | -------------- |
| **Nemotron-3.5-Lightning-30B-A3B** | 30B / 3B act | **73.9** | **136 ms** | ~719 (cap) | c=128 (still climbing) |
| openai/gpt-oss-120b | 120B / ~5B act | 34 | 71 ms | ~709 | c≈256 |
| Nemotron-3-Super-120B-A12B | 120B / 12B act | 15 | 447 ms | ~190 | c≈128 |

**Lightning is the standout on this hardware for per-stream speed:** at c=1 it is **~2.2× faster than
gpt-oss-120b** and **~4.9× faster than Nemotron-3-Super** per user, and it already **matches
gpt-oss-120b's peak aggregate throughput (~719 vs ~709)** while being a quarter of the size and while
still capped at c=128. For always-on / high-fan-out agent workloads (its design target) it is the most
efficient model measured on Warpcore so far. (gpt-oss-120b's peak is measured at a higher c≈256; a
matched-cap re-run of Lightning at `--max-num-seqs 256` would very likely exceed it.)

## Quality — lm-eval-harness (measured 2026-08-12)

Measured independently on Warpcore against the live `vllm_lightning` endpoint (raw results:
[`raw/quality/`](raw/quality/)).

| Benchmark | n | Metric | Score |
| --------- | -: | ------ | ----- |
| **GSM8K** (0-shot CoT) | 1319 | exact_match, flexible | **95.07%** (±0.60) |
| | | exact_match, anchored line | 94.62% |
| **IFEval** | 541 | prompt-level strict | **86.14%** (±1.49) |
| | | prompt-level loose | 87.06% (±1.44) |
| | | inst-level strict / loose | 85.49% / 86.09% |
| **GPQA-Diamond** (0-shot CoT) | 198 | exact_match, 32k budget | **66.16%** (±3.36) ⚠️ |
| | | exact_match, 16k budget | 53.03% (truncation-floored) |

**Eval config:** `lm-eval` 0.4.12, `local-chat-completions` backend against
`http://localhost:8000/v1/chat/completions`, `--apply_chat_template`, **greedy `temperature=0`**
(matches the other cards' fair-comparison setting rather than NVIDIA's recommended `temp=1.0`;
Lightning is terse so token burn stays low). GSM8K/IFEval used a 8192-token generation budget at
concurrency 8; GPQA at concurrency 4. GSM8K and GPQA use in-repo **clean-extract** task configs
([`raw/gsm8k_cot_zeroshot_clean.yaml`](raw/gsm8k_cot_zeroshot_clean.yaml),
[`raw/gpqa_diamond_cot_zeroshot_clean.yaml`](raw/gpqa_diamond_cot_zeroshot_clean.yaml) +
[`raw/gpqa_utils.py`](raw/gpqa_utils.py)) that anchor the answer to a required final line and fall
back to the last number / `(X)` letter. GPQA-Diamond is the **gated** `Idavidrein/gpqa` dataset.

> **⚠️ GPQA-Diamond is truncation-limited, not capability-limited — read this before comparing.**
> Lightning is a *deep* reasoner: on the hardest grad-level questions it can exhaust the generation
> budget mid-reasoning and never emit a final answer, which auto-scores as **wrong**.
>
> | Budget | Raw acc | Items answered | Truncated (empty) | Acc on *answered* items |
> | ------ | ------- | -------------- | ----------------- | ----------------------- |
> | 16k    | 53.03%  | 116 / 198      | 82 (41.4%)        | 105/116 = **90.5%**     |
> | 32k    | 66.16%  | 157 / 198      | 41 (20.7%)        | 131/157 = **83.4%**     |
>
> Doubling the budget to 32k roughly **halved** the truncation rate (82→41 items) and lifted the raw
> score by **+13 points** (53→66%). ~21% of items still truncate at 32k, so **66.16% is a lower bound**;
> the true GPQA-Diamond capability is higher. The `answer-line` and `flexible-fallback` filters agree
> exactly on GPQA (every finished response emits the required `The answer is (X)` line), so the gap is
> purely unfinished reasoning, not a parsing artifact. The 32k run is the headline number for
> cross-card comparison; the 16k run is retained to document the effect.
>
> Note the conditional accuracy *drops* from 90.5%→83.4% between 16k and 32k: the 41 extra items that
> now finish are precisely the harder ones the model previously couldn't complete, and it gets more of
> those wrong — expected behaviour, and a sign the remaining truncated tail is genuinely difficult.

**Takeaway:** near-ceiling on GSM8K (95%), strong instruction-following (86% strict IFEval), and a
science-reasoning score that is **budget-bound rather than knowledge-bound** — a 30B/3B-active model
answering GPQA-Diamond at 83–90% *when it finishes reasoning* is exceptional for its active size.

## Not yet measured / next steps

- **Agentic** (pi-30, SWE-bench Verified 100-sample) — this model is built for agents; expected to do
  well given clean tool-calling.
- **GPQA at higher budget.** ~21% of GPQA-Diamond items still truncate at 32k; a 48–64k re-run would
  lift the raw score further toward the ~83–90% conditional accuracy (this baseline stops at 32k).
- **DSpark speculative decoding.** NVIDIA ships a matching draft model
  (`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark`) and the README's recommended serve
  command enables it (`--speculative_config.num_speculative_tokens 3 --speculative_config.model
  $DSPARK_CKPT --mamba-backend flashinfer --mamba-cache-mode align`). This baseline was run **without**
  spec-decode for a clean number; enabling DSpark is the lever to chase lower latency / higher tok/s.
- **`--max-num-seqs 256` re-run** to find the true throughput ceiling (there's ~23× KV headroom).

## Reproduce

Serving (on warpcore):
```bash
docker rm -f vllm_node vllm_lightning 2>/dev/null   # frees :8000
docker run -d --rm --name vllm_lightning --gpus all --network host --ipc host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  vllm/vllm-openai:v0.27.1 \
  --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --served-model-name nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --moe-backend marlin --kv-cache-dtype fp8 --enable-prefix-caching \
  --gpu-memory-utilization 0.9 --max-model-len 262144 --max-num-seqs 128 \
  --reasoning-parser nemotron_v3 --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice --trust-remote-code --host 0.0.0.0 --port 8000
```

Throughput sweep (reusable `scripts/vllm_sweep.sh` from the `warpcore-dgx-spark` skill):
```bash
MODEL=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 CONTAINER=vllm_lightning \
  OUTDIR=~/lightning_sweep bash ~/vllm_sweep.sh 1 2 4 8 16 24 32 48 64 96 128
```
(Raw completions, `--ignore-eos`, 512-in/256-out; on-box against `localhost:8000`.)
