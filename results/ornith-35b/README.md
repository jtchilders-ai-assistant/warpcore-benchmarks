# ornith-ai/Ornith-1.0-35B-FP8 — Warpcore Benchmark Card

**Date:** 2026-08-18 → 2026-08-20; throughput extension 2026-09-08
**Model:** `ornith-ai/Ornith-1.0-35B-FP8` — MoE, **~35B total**, **hybrid Mamba + attention**
(`qwen3_5_moe` arch), **256K context**, vision-capable. Released by DeepReinforce as an
**agentic-coding** model. Quantization: **compressed-tensors W8A8 FP8** (~34.85 GiB of weights).
**Host:** Warpcore, NVIDIA DGX Spark / GB10 — see [../../HARDWARE.md](../../HARDWARE.md)
**Serving:** vLLM **`0.27.2rc1`** (image `vllm/vllm-openai:cu129-nightly-aarch64`, build
`aa99034`, transformers 5.15.0), container `vllm_ornith`, **MARLIN for both the MoE experts and the
dense/linear FP8 GEMMs**, `--enable-prefix-caching`, `--max-model-len 262144`,
`--tool-call-parser qwen3_xml`, `--reasoning-parser qwen3`.
**Endpoint:** `http://csi370295.alcf.anl.gov:8000/v1/chat/completions`
**Sampling used for all measurements:** greedy, `temperature=0` (fair-comparison setting used by every
card in this repo).

> **⚠️ The bring-up blocker: GB10 has no working CUTLASS W8A8 FP8 kernel — you need TWO Marlin
> switches, not one.** The engine core refuses to initialize
> (`cutlass_gemm_caller … Error Internal` → `RuntimeError`, EngineCore init fails) unless **both**
> the MoE path *and* the dense/linear path are forced onto Marlin:
> ```bash
> -e VLLM_TEST_FORCE_FP8_MARLIN=1   # CompressedTensorsW8A8Fp8 -> MarlinFP8ScaledMMLinearKernel
> --moe-backend marlin              # MoE experts -> Marlin
> ```
> `--moe-backend marlin` alone is **not enough** — that only covers the experts (repo
> [ISSUES.md #8](../../ISSUES.md)); the dense FP8 linear layers crash separately
> ([ISSUES.md #13](../../ISSUES.md)). Despite the `VLLM_TEST_` prefix this is the
> production-correct setting on this SM. Cost is load time only (Marlin repack ~19 s/shard,
> **251 s** to load 34.85 GiB), with no measurable correctness impact.

> **Serving notes**
> - **Weights ~34.85 GiB.** At `--gpu-memory-utilization 0.90` this leaves a **1,265,487-token KV
>   cache = 4.83× concurrency at the full 262,144-token context.**
> - **`--reasoning-parser qwen3` returns an EMPTY `reasoning_content`** while `content` comes back
>   correctly CoT-stripped. Output is right; only the field split is wrong. Don't build tooling that
>   depends on `reasoning_content` being populated.
> - **This model thinks a lot, even on trivial prompts** — 218 reasoning tokens to answer a one-word
>   question. Send a generous `max_tokens` (≥4k chat, **64k for offline reasoning benchmarks**) or the answer is
>   truncated to empty and silently scores wrong.
> - **Single box:** standing this model up took the Nemotron-3.5-Lightning endpoint down.

---

## Smoke test (functional verification) — PASS

| Check | Result |
| ----- | ------ |
| Basic generation | ✅ clean, `finish_reason: stop` |
| Tool calling (`qwen3_xml` parser) | ✅ `finish_reason: tool_calls`, well-formed arguments |
| Reasoning-parser split (`qwen3`) | ⚠️ `content` clean-stripped, but `reasoning_content` empty |
| GPQA-Diamond probe (`--limit 5`) | ✅ **5/5 correct**, `answer-line` == `flexible-fallback` (no parse artifact) |

The `--limit 5` GPQA probe took 15 m 49 s for 5 items at c=5 / 32k budget (~3.2 min/item), which
correctly predicted the ~10.5 h full-GPQA wall time below.

## Quality — lm-eval-harness (measured 2026-08-18 → 08-19)

Run on Warpcore against the live `vllm_ornith` endpoint. Raw results: [`raw/quality/`](raw/quality/).

| Benchmark | n | Metric | Score |
| --------- | -: | ------ | ----- |
| **GSM8K** (0-shot CoT, clean) | 1319 | exact_match, **anchored answer-line** | **97.19%** (±0.45) |
| | | exact_match, flexible-fallback | 97.12% (±0.46) |
| **IFEval** | 541 | **prompt-level strict** | **88.54%** (479/541; ±1.37)¹ |
| | | inst-level strict | **91.49%** (763/834) |
| | | prompt-level loose / inst-level loose | **90.76%** (491/541; ±1.24) / **92.93%** (775/834) |
| **GPQA-Diamond** (0-shot CoT, clean) | 198 | exact_match, answer-line | **80.81%** (160/198; ±2.80)¹ |
| | | exact_match, flexible-fallback | **80.81%** (160/198; ±2.80) — *identical* |

> **64k replay correction.** The original GPQA run used 32k and scored 42 empty final responses as
> zero; the original IFEval run used 8k and scored 28 as zero. We replayed their **byte-identical
> stored chat messages** at 64k, preserving greedy temperature and stop strings, then applied the same
> clean-task / lm-eval 0.4.12 scoring. GPQA recovered content for 27/42 and 22 new correct answers,
> changing **69.70% → 80.81%**. IFEval recovered content for 17/28 and exact evaluator scoring added
> 16 prompt-strict and 26 instruction-strict successes, changing **85.58% → 88.54%** prompt-strict.
> Replay artifacts are in [`raw/quality/gpqa/replay_2026-09-06/`](raw/quality/gpqa/replay_2026-09-06/)
> and [`raw/quality/ifeval/replay_2026-09-07/`](raw/quality/ifeval/replay_2026-09-07/).
>
> **Residual limit:** 15/198 GPQA (7.6%) and 11/541 IFEval (2.0%) still reached the **64k** ceiling
> without final content. These are reported as budget-limited model outcomes, not silently discarded;
> the scores are defensible at the standardized 64k setting but not unconstrained capability estimates.

**Eval config:** `lm-eval` 0.4.12 (with the None-guard + gather-survive patches), backend
`local-chat-completions` against `http://localhost:8000/v1/chat/completions`, `--apply_chat_template`,
greedy `temperature=0`. **Offline reasoning-model policy: GPQA-Diamond and IFEval use a 65536-token
maximum-generation ceiling.** It is a ceiling, not a token reservation; short items terminate normally.
GSM8K remains 8192 tokens. GPQA uses the in-repo **clean-extract** task config (shared with the Lightning
card) which anchors the final answer line and falls back to `(X)` letter extraction. GPQA-Diamond is the
gated `Idavidrein/gpqa` dataset.

**Takeaway:** **97.19% GSM8K is the highest grade-school-math score in this repo** (edging Qwen3.6-35B's
97.04%). At the common 64k offline-reasoning ceiling, Ornith reaches **88.54% IFEval prompt-strict** and
**80.81% GPQA-Diamond**—above Lightning-at-64k (76.26%) and below Qwen3.6's non-thinking-mode 82.32%
GPQA figure, which is not directly comparable until its thinking-mode 64k run. Ornith's separate edge
remains agentic coding (below).

## Throughput / latency — `vllm bench serve` concurrency sweep

Measured 2026-08-19 **on-box** (inside `vllm_ornith` against `localhost:8000` → server ceiling, network
excluded). Raw-completions path (`--backend openai --endpoint /v1/completions --ignore-eos`), fixed
shape **512 input / 256 output** tokens.

| Concurrency | Output tok/s | Δ vs prev | Mean TTFT (ms) | Median TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) |
| ----------: | -----------: | --------: | -------------: | ---------------: | ------------: | -------------: |
| 1 | 36.95 | — | 167 | 165 | 209 | 26.5 |
| 2 | 67.48 | +82.6% | 277 | 277 | 326 | 28.7 |
| 4 | 104.95 | +55.5% | 442 | 486 | 543 | 36.5 |
| 8 | 148.22 | +41.2% | 720 | 691 | 1001 | 51.3 |
| 16 | 208.17 | +40.4% | 1037 | 987 | 1919 | 72.9 |
| 24 | 244.34 | +17.4% | 1231 | 1076 | 2904 | 93.5 |
| 32 | 266.68 | +9.1% | 1447 | 1115 | 3861 | 114.3 |
| 48 | 317.74 | +19.1% | 1857 | 1209 | 5946 | 143.6 |
| 64 | 359.03 | +13.0% | 2266 | 1283 | 8195 | 168.8 |
| 96 | 418.15 | +16.5% | 3095 | 1419 | 12783 | 216.1 |
| 128 | 464.38 | +11.1% | 3986 | 1551 | 17822 | 257.7 |
| 192 | 549.38 | +18.3% | 7819 | 1799 | 29262 | 311.9 |
| **256** | **559.48** | +1.8% | 13712 | 8543 | 42269 | 342.8 |
| 384 | 557.78 | −0.3% | 50272 | 30151 | 125004 | 345.2 |

The extension establishes a measured throughput plateau at **~559 tok/s around c=256**. Going from
c=256 to c=384 changes output throughput by −0.3% while median TTFT grows from 8.5 s to 30.2 s and
P99 TTFT from 42.3 s to 125.0 s. The earlier c=128 result was therefore a measured floor, but the
old claim that it proved a `--max-num-seqs 128` cap was too strong. Three operating points:
- **Single-stream (c=1):** **36.95 tok/s/user**, TTFT **165 ms**, TPOT **26.5 ms**.
- **Balanced (c≈16):** ~208 tok/s aggregate, TPOT ~73 ms, median TTFT ~1.0 s.
- **Maximum throughput (c≈256):** **559 tok/s** aggregate, TPOT 343 ms, median TTFT 8.54 s.
- **Do not use c=384:** throughput is unchanged while median TTFT is 30.2 s and P99 is 125 s.

Raw output: [`raw/throughput_sweep/sweep.log`](raw/throughput_sweep/sweep.log) for c=1–128 and
[`raw/throughput_sweep_extended/sweep.log`](raw/throughput_sweep_extended/sweep.log) for c=192–384.

### Comparison vs the other Warpcore models

| Model | Size | c=1 tok/s | c=1 TTFT | Peak tok/s | at concurrency |
| ----- | ---- | --------: | -------: | ---------: | -------------- |
| nvidia/Nemotron-3.5-Lightning-30B-A3B | 30B / 3B act | 73.9 | 136 ms | 926 (floor) | c=384 (still climbing) |
| openai/gpt-oss-120b | 120B / ~5B act | 34 | 71 ms | ~709 | c≈256 |
| Qwen/Qwen3.6-35B-A3B | 35B / 3B act | — | — | ~487 | c=128 |
| **ornith-ai/Ornith-1.0-35B-FP8** | **~35B MoE (FP8)** | **36.95** | **165 ms** | **~559** | **c≈256** |
| Intel/Qwen3.5-122B-A10B-int4 | 122B / 10B act | 26.9 | — | ~228 | c≈192 |
| nvidia/Nemotron-3-Super-120B-A12B | 120B / 12B act | 15 | 447 ms | ~190 | c≈128 |

Ornith lands in the **middle of the pack on raw speed** — roughly half Lightning's per-stream rate
(36.95 vs 73.9 tok/s) despite being a similar total size. Two reasons, both structural on this
bandwidth-bound GB10: (1) **FP8 W8A8 weights are 2× the bytes of Lightning's NVFP4**, so every token
moves ~35 GiB instead of ~18 GiB through memory; (2) **the Marlin FP8 path is a compatibility fallback,
not the fast path** for this SM. Active-params-per-token remains the dominant lever here — but so does
weight *precision*, and Ornith pays for FP8. It buys that back in agentic quality.

## Agentic coding — SWE-bench Verified (n=100 shuffled)

**Measured 2026-08-19 → 08-20.** [SWE-bench Verified](https://www.swebench.com/) via
[`mini-swe-agent`](https://github.com/SWE-agent/mini-swe-agent) (bash-only agent loop): given a real
GitHub issue + repo, the model must produce a patch that makes the repo's hidden test suite pass,
graded pass/fail inside a per-instance x86 Docker container. The agent loop and test containers run on
the **client Mac mini (x86_64)**; the model is served on Warpcore.

### **Result: 73 / 100 resolved = 73.0%** — the best agentic-coding score in this repo by a wide margin.

| | |
|---|---|
| Resolved | **73 / 100 = 73.0%** |
| Unresolved (genuine model failures) | 18 |
| Non-submissions | 9 empty patches; historical notes classified 8 `LimitsExceeded` and 1 `ContextWindowExceededError`, but the underlying exit-status artifacts are lost |
| Harness / grading errors | **0** |
| Sample | full n=100 (`--shuffle`, **seed 42**, `--slice 0:100`) — **identical instance set** to the Lightning and Qwen3.6 runs |
| Repos spanned | 11 |
| Baseline (Nemotron-3.5-Lightning-30B) | 51/100 — **Ornith is ahead by +22** |
| Baseline (Qwen3.6-35B) | 44/100 — **Ornith is ahead by +29** |
| Head-to-head vs Lightning (shared 100) | **46 both · 27 Ornith-only · 5 Lightning-only · 22 neither** |
| Agent | mini-swe-agent, robust-submit scaffold, `temp=0`, per-step `timeout=1800`, `step_limit 250`, 4 workers |
| Serving | vLLM `0.27.2rc1` aarch64 nightly, Marlin (both paths), `qwen3_xml` tool-call + `qwen3` reasoning parsers, **`--gpu-memory-utilization 0.55`** |
| Wall time | 11 h 06 m (generation) + 21 m (grading) |
| Raw | [`raw/swebench/`](raw/swebench/) (report JSON, preds, agent config, run script) |

**Per-repo breakdown** (resolved / attempted, over all 100):

| Repo | Resolved | Attempted |
| ---- | -------: | --------: |
| django | 44 | 56 |
| sphinx-doc | 8 | 10 |
| sympy | 7 | 10 |
| scikit-learn | 4 | 5 |
| astropy | 2 | 5 |
| pytest-dev | 2 | 4 |
| pydata (xarray) | 2 | 3 |
| matplotlib | 1 | 2 |
| psf (requests) | 1 | 2 |
| pylint-dev | 1 | 2 |
| pallets (flask) | 1 | 1 |
| **Total** | **73** | **100** |

**The aggregate is clean with respect to the retained grading report: 0 harness errors and 91/100
instances received a test verdict.** The nine non-submission IDs are retained, but their original
per-instance exit statuses are not: trajectories and the run log were lost from `/tmp`. Historical
notes classified eight as `LimitsExceeded` and one as `ContextWindowExceededError`, but that breakdown
is no longer independently auditable from repository artifacts. Notably, no serving failures appear in
the retained report. The robust submit step (`git add -A && git diff --cached`, carried over from the
Lightning run) produced **zero patch-apply errors**.

**The head-to-head is the striking part.** On the identical 100 instances, Ornith resolves **27 that
Lightning misses** while losing only **5** that Lightning gets. That is not sampling noise — it is a
real capability gap on agentic patch generation, and it is consistent with Ornith being purpose-built
by DeepReinforce for agentic coding. Note the honest inversion this creates: Ornith is *ahead of
Lightning but behind Qwen3.6* on GPQA-Diamond general reasoning (**80.81%** vs 76.26% / 82.32%) yet far *ahead* on
SWE-bench. **Benchmark-suite rank does not transfer across task families** — pick the model for the
job, not for the leaderboard.

### Caveats (read before quoting the number)
- **n=100 shuffled, not the full 500 → indicative (95% Wilson CI 63.6–80.7, roughly ±9 pp), not
  leaderboard-final.** Same convention and
  same seeded instance set as every other SWE-bench number in this repo, so cross-model comparisons
  here are exact even though the absolute score is a sample.
- **The sample is django-heavy (56/100)** and Ornith is especially strong there (44/56 = 79%). Django
  instances skew slightly easier, so a balanced 500-item run would likely land somewhat lower.
- **9 instances never submitted a patch.** Their exact exit causes are no longer auditable because the
  trajectories were lost; the historical 8-limit/1-context breakdown should not be treated as a
  repository-verifiable result. A larger-budget re-run is required to measure whether 73 is conservative.

## Agentic coding — pi-30 (measured 2026-08-20)

[`rick-stevens-ai/pi-30`](https://github.com/rick-stevens-ai/pi-30): a 30-problem agent-loop coding
benchmark driven by the `pi` CLI (v0.84.2). The model must *use tools* (read/write files, run code) to
iteratively fix or build a solution; verdicts come from **verifier exit codes**, never model prose.
Problems span iterate-until-green pytest fixes, oracle-matching, generator+critic loops, and best-of-N
tournaments.

| | |
|---|---|
| **Score** | **30 / 30 passed** — a perfect run, no retries needed at the problem level |
| Harness | `pi` 0.84.2, `PI_TIMEOUT=600`, temp per pi defaults |
| Serving | `vllm/vllm-openai:cu129-nightly-aarch64`, Marlin (both FP8 paths), **`--gpu-memory-utilization 0.55`** (host-headroom config — pi-30's agent processes run on the same box) |
| Tool-calling | `qwen3_xml` parser, registered as pi provider `warpcore` with `reasoning: true`, `maxTokens: 32768` |
| Wall time | ~1 h 15 m (12:58 → 14:13 CDT) |
| Raw | [`raw/pi30/`](raw/pi30/) (RESULTS.txt, SUMMARY.txt, log, run script) |

**Nothing was close to failing.** Every iterate-until-green problem converged in ≤3 rounds (P4 was the
slowest at 3), both critic+verifier problems passed (P6 in 1, **P24 in 2**), and all five best-of-N
tournaments (P9 77.076, P10 85.6026, P19 346.5107, P27 0.7951, P30 9.9340) produced a valid champion.
Measurement problems: P5 **98.49 GFLOP/s**, P15 **52.84 Mtok/s**, P23 **163.57 Mn/s**.

**P24 is worth calling out:** the token-bucket rate-limiter problem is the single problem
[Nemotron-3.5-Lightning failed](../nemotron-3.5-lightning-30b/README.md#agentic-coding--pi-30-measured-2026-08-14)
(it didn't seed the bucket to capacity, so a t=0 burst of 5 was throttled to 1). Ornith got it right in
2 critic rounds. Combined with the SWE-bench head-to-head, this is a consistent signal rather than a
one-off.

30/30 matches gpt-oss-120b and Nemotron-3-Super-120B and beats Lightning/Qwen3.6 (29/30) — but note
pi-30 is now **saturated at the top of this repo's model set** and no longer discriminates between good
agentic models. SWE-bench Verified is the benchmark with headroom; treat pi-30 as a pass/fail gate.

## Not yet measured / next steps

- **Full SWE-bench Verified (n=500)** — 73/100 on a shuffle is strong enough to be worth confirming at
  full scale; this is the leaderboard-final number.
- **Raise the SWE-bench step/cost limit in a provenance-complete re-run** — historical notes attribute
  8 of 9 non-submissions to `LimitsExceeded`, but the lost trajectories prevent verification.
- ~~**`--max-num-seqs > 128` re-run**~~ — completed 2026-09-08; plateau is ~559 tok/s at c≈256.
- **Vision.** The checkpoint is vision-capable; no multimodal benchmark has been run on Warpcore.
- **A non-Marlin FP8 path.** If a future vLLM/CUDA build lands a working CUTLASS W8A8 FP8 kernel for
  SM 12.1, re-measure throughput — the current 36.95 tok/s c=1 is on a compatibility fallback kernel.
- **`reasoning_content` parser fix** — `qwen3` leaves it empty; worth a bug report.

## Reproduce

Serving (on warpcore) — full script: [`raw/launch_ornith.sh`](raw/launch_ornith.sh). It requires an
explicit workload profile so unified-memory headroom cannot be selected accidentally:
```bash
./launch_ornith.sh throughput   # util=0.90, max-num-seqs=512; remote clients/on-box sweep
./launch_ornith.sh agentic      # util=0.55, max-num-seqs=32; on-box agent processes
```
Both profiles force Marlin for the MoE experts and dense/linear FP8 GEMMs, serve a 262,144-token
window, enable prefix caching, and use the `qwen3_xml` tool-call and `qwen3` reasoning parsers.
The **agentic** profile reproduces the host-headroom settings used for SWE-bench and pi-30; the
**throughput** profile reproduces the extended sweep's serving configuration. GB10 GPU and host memory
share the same 121 GiB pool, so these profiles are not interchangeable.

Clients must always send an explicit `max_tokens` — at least 4k for chat and **64k for offline
reasoning benchmarks**.

> **CORRECTION (2026-08-26):** *"vLLM has no default-request-budget flag"* is **wrong**. With
> `--generation-config auto` (the default), vLLM loads the checkpoint's `generation_config.json` and
> any `max_new_tokens` there becomes a **server-wide cap on all requests** — silently clamping, with
> nothing in the launch command to show for it. **Ornith-1.0-35B-FP8 was verified to carry no
> `max_new_tokens`, so this card's scores are unaffected.** Check before every run with
> [`viz/check_output_budget.py`](../../viz/check_output_budget.py). Full explanation on the
> [Lightning card](../nemotron-3.5-lightning-30b/README.md#recommended-output-token-budget-max_tokens-and-whether-its-a-realistic-serving-setting).

Quality (on warpcore, in a detached tmux — the suite takes ~16.6 h) —
[`raw/run_ornith_quality.sh`](raw/run_ornith_quality.sh):
```bash
source /tmp/lmeval-venv/bin/activate      # lm-eval 0.4.12 + None-guard/gather-survive patches
lm_eval --model local-chat-completions \
  --model_args model=ornith-ai/Ornith-1.0-35B-FP8,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=4,max_retries=8,tokenized_requests=False,timeout=7200 \
  --tasks ifeval --apply_chat_template \
  --gen_kwargs max_gen_toks=65536,temperature=0 --output_path /tmp/lmeval_results/ornith35b/ifeval
# gpqa:   --tasks gpqa_diamond_cot_zeroshot_clean --gen_kwargs max_gen_toks=65536,temperature=0 (num_concurrent=4, timeout=7200)
```

Throughput sweep (reusable `scripts/vllm_sweep.sh` from the `warpcore-dgx-spark` skill, copied to
[`raw/throughput_sweep/vllm_sweep.sh`](raw/throughput_sweep/vllm_sweep.sh)):
```bash
MODEL=ornith-ai/Ornith-1.0-35B-FP8 CONTAINER=vllm_ornith OUTDIR=/tmp/ornith_sweep \
  bash ~/vllm_sweep.sh 1 2 4 8 16 24 32 48 64 96 128
```

SWE-bench Verified (n=100 shuffle) — agent loop + x86 test containers on the **client Mac mini**, model
served on Warpcore at `--gpu-memory-utilization 0.55`. Generation
([`raw/swebench/run_gen_n100.sh`](raw/swebench/run_gen_n100.sh),
config [`raw/swebench/swebench_ornith_config.yaml`](raw/swebench/swebench_ornith_config.yaml)):
```bash
source ~/swebench-lightning/venv/bin/activate
export HOSTED_VLLM_API_BASE="http://csi370295.alcf.anl.gov:8000/v1"
export HOSTED_VLLM_API_KEY="warpcore"
export MSWEA_COST_TRACKING='ignore_errors'
mini-extra swebench --subset verified --split test --shuffle --slice 0:100 \
  -c swebench.yaml -m "hosted_vllm/ornith-ai/Ornith-1.0-35B-FP8" \
  --environment-class docker -w 4 -o /tmp/ornith_swe_n100
```
Grading (local Docker only — safe to run while the endpoint is busy with another benchmark):
```bash
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path /tmp/ornith_swe_n100/preds.json \
  --max_workers 4 --run_id ornith_n100 --namespace swebench
```
Writes `hosted_vllm__ornith-ai__Ornith-1.0-35B-FP8.ornith_n100.json`, copied here to
[`raw/swebench/swebench_verified_n100_results.json`](raw/swebench/swebench_verified_n100_results.json).
The scaffold uses the **robust submit** step (`git add -A && git diff --cached`) rather than
mini-swe-agent's stock `cat patch.txt` — see the Lightning card for why that matters.
