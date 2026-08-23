# poolside/Laguna-S-2.1-NVFP4 on Warpcore

| | |
| --- | --- |
| **Date** | 2026-08-22 |
| **Model** | [`poolside/Laguna-S-2.1-NVFP4`](https://huggingface.co/poolside/Laguna-S-2.1-NVFP4) — 117.6B total / ~8.5B active MoE (256 experts, top-10 routed) |
| **Host** | Warpcore — NVIDIA DGX Spark (GB10, **sm_121**), ~121 GiB unified memory ([HARDWARE.md](../../HARDWARE.md)) |
| **Serving** | vLLM `0.27.2rc1.dev193+gaa9903490` (`vllm/vllm-openai:cu129-nightly-aarch64`), NVFP4 experts via **MARLIN** + BF16 experts via **TRITON**, FP8 KV, 128k context |
| **Endpoint** | `http://csi370295.alcf.anl.gov:8000/v1` (`--served-model-name poolside/Laguna-S-2.1-NVFP4`) |

> **Bring-up required patching vLLM's MoE backend selection.** This is a *mixed-quantization*
> checkpoint and the GB10 is **sm_121**, a combination that vLLM cannot currently serve out of the box.
> Full diagnosis in [ISSUES.md #14](../../ISSUES.md#14-gb10-is-sm_121-and-flashinfer-ships-no-sm_121-cubin--no-kernel-image-is-available);
> the shim and launch script are archived in [`raw/`](raw/). Short version:
>
> - Layers **0–39** experts are NVFP4, layers **40–47** experts stay **BF16** (they are in
>   `quantization_config.ignore`). vLLM builds two MoE method objects with **disjoint** legal backends,
>   but `--moe-backend` is a single **global** flag: `marlin` is rejected by the unquantized group,
>   `triton` by the NVFP4 group.
> - Omitting the flag lets both groups auto-select **FlashInfer**, which initialises fine and then dies
>   on the **first decode** with `no kernel image is available for execution on the device` — the GB10
>   is sm_121 but the image only ships sm_80/90/100/120 cubins, and FlashInfer's kernels are arch-exact
>   SASS with no PTX fallback. (The error is wrapped as `MemoryError`; **it is not an OOM**.)
> - Fix: `--moe-backend marlin` **plus** a `sitecustomize.py` aliasing marlin→TRITON for the
>   unquantized group only, gated on device capability `(12,1)`.

## Sizing — why only NVFP4 fits

Vendor's model card claims the NVFP4 weights are "roughly 71 GB". **That is ~30% low.** Summing the
actual `.safetensors` files gives **99.7 GB (92.9 GiB)**, because the `ignore` list keeps attention
projections, `mlp.gate`, shared experts, layer-0 dense MLP and all of layers 40–47's experts in BF16
(HF metadata: `BF16: 23.3B params` + `U8: 94.2B packed`).

| Variant | Size | Fits one GB10? |
| --- | --- | --- |
| BF16 | 235.1 GB | No |
| FP8 | 131.3 GB | No |
| **NVFP4** | **99.7 GB (92.9 GiB)** | **Yes, barely** |
| INT4 | 99.7 GB | Yes |

**Measured at runtime** (not estimated): weights + non-torch **94.48 GiB**, peak activation 4.11 GiB,
CUDA graphs 0.47 GiB, leaving **8.44 GiB of KV = 333,604 tokens** (2.55× concurrency at 128k) out of
the 121.63 GiB pool. vLLM reports headroom to raise KV to 16.32 GiB via `--kv-cache-memory`.

> **"8B active" does not make it small.** Active params govern *speed*; total params govern
> *residency*. The router picks a different top-10 of 256 experts per token, so all 117.6B must be
> resident. This question comes up every time — the answer is always total, not active.

KV is unusually cheap here because 36 of 48 layers are sliding-window(512) and only 12 are global:
a fixed 37.7 MB/seq for the SWA layers plus 24.6 kB/tok for the global ones → ~3.26 GB/seq at 128k,
vs ~12.9 GB/seq if you (wrongly) applied the all-global formula.

## Smoke tests

| Check | Result |
| --- | --- |
| Single request | `content: 'WARPCORE_OK'`, `finish_reason: stop` |
| Tool calling (`poolside_v1` parser) | `finish_reason: tool_calls` → `get_weather {"city": "Chicago"}` |
| Concurrent guided decoding (60 req @ c=16) | **60/60 OK, 0 FAIL in 20.2 s**, engine alive afterwards |

The third test is the one that matters on this box: concurrent structured-output/tool-calling is the
workload that kills bad GB10 MoE kernels, and it is exactly where the FlashInfer path failed. Probe
archived at [`raw/stress_guided_decoding_laguna.py`](raw/stress_guided_decoding_laguna.py).

## Throughput sweep

`vllm bench serve`, raw completions + `--ignore-eos` (the chat backend lets the model stop early and
understates tok/s). 512-token input, 256-token output, 3 prompts per concurrency level.

| Concurrency | Output tok/s | Δ vs prev | Mean TTFT (ms) | Median TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | tok/s per user |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 17.65 | — | 156.1 | 157.7 | 158.4 | 56.3 | 17.65 |
| 2 | 29.34 | +66.2% | 299.1 | 309.9 | 409.8 | 67.2 | 14.67 |
| 4 | 42.17 | +43.7% | 386.5 | 426.6 | 475.3 | 93.7 | 10.54 |
| 8 | 60.52 | +43.5% | 509.1 | 554.3 | 556.7 | 130.7 | 7.57 |
| 16 | 78.00 | +28.9% | 1545.3 | 906.8 | 3392.9 | 199.6 | 4.88 |
| 32 | 112.54 | +44.3% | 2369.6 | 1230.6 | 6189.9 | 275.4 | 3.52 |
| 64 | 165.64 | +47.2% | 3770.6 | 1359.0 | 10108.2 | 370.8 | 2.59 |
| 128 | 233.27 | +40.8% | 7570.1 | 3826.5 | 30366.8 | 514.4 | 1.82 |
| 192 | 258.77 | **+10.9%** | 28118.2 | — | 154313.3 | 557.1 | 1.35 |
| 256 | 266.34 | **+2.9%** | 78215.9 | — | 171099.0 | 565.2 | 1.04 |

**Peak is ~266 tok/s at c≈256, and the curve is flat by c≈192.** The sweep to c=128 looked like it was
still climbing (+40.8% on the last step), but extending it settled the question: c=192 adds only
**+10.9%** and c=256 a further **+2.9%** — the engine is saturated, not scheduler-capped. During the
c=256 run live `/metrics` showed only **49 requests running with 207 queued**, *fewer* admitted
concurrently than at c=192, which is KV-cache pressure rather than queueing headroom. With just
8.44 GiB of KV (333,604 tokens) left after 94.48 GiB of weights, this model runs out of cache long
before it runs out of scheduler slots. **Practical peak: ~259 tok/s at c≈192** — the last 2.9% costs
2.8× the TTFT and is not worth taking.

> **Correction.** An earlier revision of this card reported the peak as "≥333 tok/s, still climbing",
> extrapolated from a live `/metrics` sample (164 running / 28 queued at 333.5 tok/s) taken *during*
> the c=192 run. That instantaneous reading was measured mid-run while the queue was draining and did
> not survive contact with the completed benchmark: c=192 finished at **258.77 tok/s**. Instantaneous
> `generation_tokens_total` deltas overstate sustained throughput — trust the completed
> `vllm bench serve` number.

Latency past c=128 is not usable interactively: mean TTFT is **28 s** at c=192 and **78 s** at c=256,
with P99s of 154 s and 171 s. Anything above c=128 is a batch-only regime.

**Interactive SLO (mean TPOT < 100 ms) holds only to c=4.** Single-stream decode is **17.65 tok/s** —
slow, and expected: ~8.5B active params/token on a bandwidth-bound box, with the NVFP4 experts on
Marlin (a compatibility path, not a speed path) and 12 of 48 layers doing full global attention.

> ⚠️ **A first sweep was discarded as contaminated** and is preserved as
> [`raw/throughput_sweep/sweep_CONTAMINATED.log`](raw/throughput_sweep/sweep_CONTAMINATED.log). An
> lm-eval smoke test was accidentally run against the same engine mid-sweep, producing a nonsense flat
> spot (c=8 at 42.18 tok/s vs c=4 at 40.99). This is [ISSUES #6](../../ISSUES.md) recurring. The rerun
> ([`sweep.sh`](raw/throughput_sweep/sweep.sh)) polls `vllm:num_requests_running/waiting` and refuses
> to start a level unless the engine is idle. Killing the old run also required
> `docker exec vllm_laguna pkill -f "vllm bench serve"` — an orphaned bench process survived
> `tmux kill-session` and runs as root inside the container.

### Comparison vs other Warpcore models

| Model | Active params | c=1 tok/s | c=1 TTFT | Peak tok/s |
| --- | ---: | ---: | ---: | --- |
| nvidia/Nemotron-3.5-Lightning-30B-A3B-NVFP4 | ~3B | — | — | ~719 (c=128, capped) |
| openai/gpt-oss-120b | ~5B | ~34 | ~316 ms | ~709 (c≈256) |
| Qwen/Qwen3.6-35B-A3B-FP8 | ~3B | — | — | ~487 (c=128) |
| ornith-ai/Ornith-1.0-35B-FP8 | ~3B | 36.95 | — | ~464 (c=128, capped) |
| Intel/Qwen3.5-122B-A10B-int4 | ~10B | 26.9 | ~120 ms | ~228 (c≈192) |
| **poolside/Laguna-S-2.1-NVFP4** | **~8.5B** | **17.65** | **156 ms** | **~259 (c≈192)** |

Laguna lands at the slow end, which is what the architecture predicts on this bandwidth-bound GB10:
active-params-per-token is the dominant lever, and at ~8.5B it sits near Qwen3.5-122B's ~10B (26.9
tok/s) rather than the ~3B models. Its single-stream number is *below* Qwen3.5-122B despite fewer
active params — the plausible causes are the Marlin NVFP4 path being a compatibility rather than fast
kernel, the mixed BF16/NVFP4 expert stack (layers 40–47 run unquantized through Triton, moving ~2×
the bytes), and 12 global-attention layers. Not yet isolated.

## Quality

lm-eval 0.4.12, `local-chat-completions`, greedy (`temperature=0`), the repo's clean task configs.

| Benchmark | Score | n | Budget | Notes |
| --- | ---: | ---: | ---: | --- |
| GSM8K-clean | **96.13%** (corrected, full set) | 1319 | 8k, c=32 | raw harness output 83.40% — see defect below |
| IFEval prompt-strict | **75.79%** (floor) | 541 | 8k, c=32 | loose 80.59%; underestimate, same defect |
| IFEval inst-strict | **81.41%** (floor) | 541 | 8k, c=32 | loose 85.01%; underestimate, same defect |
| GPQA-Diamond-clean | **not reported** | — | 32k, c=16 | run abandoned — see below |

### The GSM8K number was a harness defect, not a capability result

**14.1% of GSM8K questions (186/1319) came back with completely empty `content`** and scored zero,
dragging the raw harness output to 83.40%. The root cause is below; this section records the correction.

All 186 affected questions were **re-served and graded with the identical task filters**, reading the
`reasoning` field where vLLM had actually put the answers:

| | answer-line | flexible-fallback |
| --- | ---: | ---: |
| Recovered text | 186/186 (100.0%) | 186/186 (100.0%) |
| Correct among recovered | **168 (90.3%)** | 165 (88.7%) |

Corrected full-set score: **(1100 + 168) / 1319 = 96.13%** (flexible-fallback 95.91%).

| Subset | Score | n |
| --- | ---: | ---: |
| Raw harness output | 83.40% | 1319 |
| Items that got a response | 97.09% | 1133 |
| **Recovered items** | **90.32%** | **186** |
| **Corrected total** | **96.13%** | **1319** |

**The recovered items were measurably harder than the served ones** — 90.3% vs 97.09%. That is not
noise: it means the earlier "97.09% on served items" figure was a **biased estimate**, optimistic by
~0.96 points, because the defect did not drop questions uniformly at random. Estimating a score by
excluding failures assumes the excluded items resemble the kept ones, and here they demonstrably did
not. Measuring them directly was necessary; extrapolating would have overstated the model.

**96.13% is the number to use.** It is a direct measurement of all 1319 items, not an extrapolation.
For repo comparison that places Laguna just below Ornith's 97.19%, not above it as the exclusion
estimate implied.

IFEval's scores are still **floors**, not corrected values — 29/541 (5.4%) were hit by the same defect
and have not been re-served. Its true scores are modestly higher than reported.

### ROOT CAUSE (confirmed): the answers go into `message.reasoning`, which lm-eval never reads

This is **not** a model capability problem and **not** lost work. vLLM generates a complete, correct
answer and places it in a response field the harness does not look at.

A question that returned empty content, re-served and dumped in full:

```json
{"role": "assistant",
 "content": null,
 "reasoning": "A candle melts by 2 centimeters every hour... From 1:00 PM to 5:00 PM is 4 hours...
               2 cm/hour x 4 hours = 8 cm\n\nThe answer is 8"}
```

`content: null`, `finish_reason: "stop"`, 129 completion tokens billed — and the correct answer,
**including the exactly-formatted `The answer is 8` line the task asks for**, sitting in `reasoning`.

The two halves of the defect:

1. **Server side.** `--reasoning-parser poolside_v1` fails to initialize its reasoning token IDs:
   ```
   WARNING [vllm.py:1689] Auto-initialization of reasoning token IDs failed. Please check whether
   your reasoning parser has implemented the `reasoning_start_str` and `reasoning_end_str`.
   ```
   Without those delimiters the parser cannot find where reasoning ends and the answer begins, so on
   some generations it classifies the *entire* output as reasoning and emits `content: null`.
   Note the field is `reasoning`, **not** the OpenAI-conventional `reasoning_content` — probes that
   only check `reasoning_content` see nothing and wrongly conclude the output vanished.

2. **Client side.** lm-eval's `ChatCompletions` parser is a single line:
   ```python
   tmp[choices["index"]] = choices["message"]["content"]
   ```
   It has no notion of `reasoning` (`"reasoning" in source == False`). `content: null` becomes an
   empty string, which scores zero.

Verified by bisection: **the raw `/v1/completions` endpoint returns 1579 characters of normal text
for the same prompt**, while `/v1/chat/completions` returns `content: null`. The tokens are produced;
the chat layer misfiles them.

This is the same family as the empty-`reasoning_content` defect on the `qwen3` parser in
[ISSUES.md #13](../../ISSUES.md), and it means **any** lm-eval score taken against a vLLM endpoint
with a partially-initialized reasoning parser is silently depressed.

Hypotheses tested and falsified along the way — recorded so nobody re-treads them:

| Hypothesis | Test | Result |
| --- | --- | --- |
| Truncation at the 8k budget | response-length distribution | **Ruled out** — p99 ≈ 240 tok, none near ceiling |
| Request errors / retries | client + server logs | **Ruled out** — 0 errors, 0 retries, HTTP 200 |
| Per-question / prompt-specific | replayed a failing question alone | **Ruled out** — same prompt succeeds and fails |
| Concurrency or load | 64 requests at c=32 | **Ruled out** — 0/64 empty |
| lm-eval `until: ["\n\nQ:"]` stop string | replay with and without stop | **Ruled out** — 60% empty even *without* it |

The reproduction that mattered: running the **full lm-eval path** on 200 questions reproduced the
defect at **11.5%**, while hand-rolled direct API calls never reproduced it — which localized the bug
to the chat-completions response handling rather than the engine.

Analysis scripts: [`raw/quality/gsm8k_empty_content_analysis.py`](raw/quality/gsm8k_empty_content_analysis.py),
[`raw/quality/empty_content_load_probe.py`](raw/quality/empty_content_load_probe.py),
[`raw/quality/where_tokens_go.py`](raw/quality/where_tokens_go.py).

IFEval shows the same defect at lower rate (29/541 = 5.4% empty), so its scores are also mild
underestimates.

### GPQA-Diamond was abandoned, not completed

The 32k/c=16 run was **killed at 110/198 after ~13 hours** and no score is reported. It was not
producing usable throughput: **352 `TimeoutError`/retry events**, with items 103→110 alone consuming
over four hours.

The engine was **not** wedged — it was generating steadily at ~70 tok/s with 17 requests running and
nothing queued. The failure is an arithmetic mismatch, not a hang: at c=16 each request gets roughly
**4 tok/s**, so a 32k-token reasoning answer needs **~2 hours**, but lm-eval's client timeout is
**3600 s**. Long items were therefore cut off and retried *from scratch*, and the retries hit the same
wall — the run was burning GPU hours re-generating work it then discarded, and would never converge.

Fixing this requires raising the client `timeout` well past the worst-case generation time and cutting
concurrency so each request gets a larger share of decode (c=4 gives ~4× the per-request rate). That
rerun is pending.

## Agentic

- **SWE-bench Verified** (n=100, seed-42 shuffle) — pending, to be driven from the Mac mini against
  this endpoint.
- **pi-30 — excluded, and will stay excluded.** pi-30 runs its agent processes *on Warpcore itself*
  and needs ~40 GiB of host headroom. With 94.5 GiB of weights resident there is ~7 GiB free, so the
  kernel OOM-killer would take vLLM mid-run (the unified-memory failure mode in ISSUES). This is a
  hard incompatibility between this model's footprint and that benchmark's design, not a skipped step.

## Reproduce

Serve (full script: [`raw/launch_laguna.sh`](raw/launch_laguna.sh), shim:
[`raw/sm121_moe_sitecustomize.py`](raw/sm121_moe_sitecustomize.py)):

```bash
docker run -d --network host --name vllm_laguna \
  -v $HOME/vllm_patch:/patch \
  -e PYTHONPATH=/patch \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  --gpus all --ipc=host \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  vllm/vllm-openai:cu129-nightly-aarch64 \
  --model poolside/Laguna-S-2.1-NVFP4 \
  --moe-backend marlin \
  --served-model-name poolside/Laguna-S-2.1-NVFP4 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.88 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser poolside_v1 \
  --reasoning-parser poolside_v1 \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --trust-remote-code \
  --host 0.0.0.0 --port 8000
```

Confirm the backends before trusting anything — you want **both** of these, and no FlashInfer:

```
[nvfp4.py:244]       Using 'MARLIN' NvFp4 MoE backend out of potential backends: [...]
[unquantized.py:282] Using TRITON Unquantized MoE backend out of potential backends: [...]
```

Startup takes ~20 min: at 92.85 GiB the checkpoint exceeds available page cache, so vLLM disables
auto-prefetch and shard loads run ~17–40 s each. The endpoint returns HTTP 000 throughout — this is
disk-bound, not wedged (`vmstat 1 3` shows high `bi`, low `si/so`).

Throughput sweep ([`raw/throughput_sweep/sweep.sh`](raw/throughput_sweep/sweep.sh)):

```bash
docker exec vllm_laguna vllm bench serve \
  --base-url http://localhost:8000 \
  --model poolside/Laguna-S-2.1-NVFP4 \
  --backend openai --endpoint /v1/completions \
  --dataset-name random --random-input-len 512 --random-output-len 256 \
  --ignore-eos --num-prompts $((C*3)) --max-concurrency $C
```

Quality (lm-eval 0.4.12, clean task configs):

```bash
lm_eval --model local-chat-completions \
  --model_args "model=poolside/Laguna-S-2.1-NVFP4,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=4,max_retries=8,tokenized_requests=False,timeout=3600" \
  --apply_chat_template --include_path /tmp/lmeval_clean_tasks --log_samples \
  --tasks gsm8k_cot_zeroshot_clean --gen_kwargs "max_gen_toks=8192,temperature=0"
```

## Next steps

1. Finish the high-concurrency sweep (c=192/256) to pin the real throughput ceiling.
2. Run the full quality suite (GSM8K, IFEval, GPQA-Diamond) in tmux on the box.
3. SWE-bench Verified n=100 (seed 42) from the Mac mini for comparability with the other cards.
4. Investigate whether the mixed BF16 expert layers are the single-stream bottleneck — if so, the
   uniformly-quantized INT4 sibling may serve faster at the same footprint.
