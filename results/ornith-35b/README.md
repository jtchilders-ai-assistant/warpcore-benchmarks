# ornith-ai/Ornith-1.0-35B-FP8 — Warpcore Benchmark Card

**Date:** 2026-08-18 → 2026-08-20
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
>   question. Send a generous `max_tokens` (≥4k chat, 32k+ for GPQA/agentic) or the answer is
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
| **IFEval** | 541 | **prompt-level strict** | **85.58%** (±1.51) |
| | | inst-level strict | 88.37% |
| | | prompt-level loose / inst-level loose | 87.80% (±1.41) / 89.81% |
| **GPQA-Diamond** (0-shot CoT, clean) | 198 | exact_match, answer-line | **69.70%** (±3.27) |
| | | exact_match, flexible-fallback | **69.70%** (±3.27) — *identical* |

**Eval config:** `lm-eval` 0.4.12 (with the None-guard + gather-survive patches), backend
`local-chat-completions` against `http://localhost:8000/v1/chat/completions`, `--apply_chat_template`,
greedy `temperature=0`. GSM8K + IFEval: 8192-token budget at concurrency 8. GPQA-Diamond: **32768**-token
budget at concurrency 5 (the model's thinking budget demands it). GSM8K and GPQA use the in-repo
**clean-extract** task configs (shared with the Lightning card:
[`../nemotron-3.5-lightning-30b/raw/gsm8k_cot_zeroshot_clean.yaml`](../nemotron-3.5-lightning-30b/raw/gsm8k_cot_zeroshot_clean.yaml),
[`../nemotron-3.5-lightning-30b/raw/gpqa_diamond_cot_zeroshot_clean.yaml`](../nemotron-3.5-lightning-30b/raw/gpqa_diamond_cot_zeroshot_clean.yaml))
which anchor the answer to a required final line and fall back to the last number / `(X)` letter.
GPQA-Diamond is the **gated** `Idavidrein/gpqa` dataset. Total quality wall time ≈ **16.6 h**
(GSM8K 3 h 13 m; GPQA alone ran 06:18 → 16:45 ≈ 10.5 h).

> **These numbers are trustworthy, and here's the evidence.** The failure mode this repo has been
> burned by before is a *budget-truncated* reasoning model scoring an artificial zero on items it
> could actually solve ([Lightning's GPQA budget curve](../nemotron-3.5-lightning-30b/README.md#quality--lm-eval-harness-measured-2026-08-12)).
> That did **not** happen here: only **2 length-truncations across all of GPQA-Diamond**, so the 32k
> budget was adequate and 69.70% is capability-bound, not budget-bound. Independently, the
> `answer-line` and `flexible-fallback` filters return **exactly the same value** on GPQA (and differ
> by only 0.08 pt on GSM8K), which rules out a parsing artifact — and the score is consistent with the
> pre-run 5/5 smoke probe.

**Takeaway:** **97.19% GSM8K is the highest grade-school-math score in this repo** (edging Qwen3.6-35B's
97.04%), IFEval is mid-pack at 85.58% prompt-strict, and GPQA-Diamond at 69.70% sits above
Nemotron-3-Super-120B (63.64%) but below Qwen3.6-35B (82.32%) and Lightning-at-64k (76.26%). Ornith is
**not** the strongest general-knowledge reasoner here — its edge is agentic coding (below).

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
| **128** | **464.38** | +11.1% | 3986 | 1551 | 17822 | 257.7 |

**Monotonic, no collapse, and still climbing at c=128 (+11.1%) — 464 tok/s is a floor set by the
`--max-num-seqs 128` cap, not a plateau.** Three operating points:
- **Single-stream (c=1):** **36.95 tok/s/user**, TTFT **165 ms**, TPOT **26.5 ms**.
- **Balanced (c≈16):** ~208 tok/s aggregate, TPOT ~73 ms, median TTFT ~1.0 s.
- **Max measured (c=128):** **464 tok/s** aggregate, TPOT 258 ms, median TTFT 1.55 s (P99 17.8 s — the
  tail grows with batch depth as expected).

Raw per-level output: [`raw/throughput_sweep/sweep.log`](raw/throughput_sweep/sweep.log).

### Comparison vs the other Warpcore models

| Model | Size | c=1 tok/s | c=1 TTFT | Peak tok/s | at concurrency |
| ----- | ---- | --------: | -------: | ---------: | -------------- |
| nvidia/Nemotron-3.5-Lightning-30B-A3B | 30B / 3B act | 73.9 | 136 ms | ~719 (cap) | c=128 (still climbing) |
| openai/gpt-oss-120b | 120B / ~5B act | 34 | 71 ms | ~709 | c≈256 |
| Qwen/Qwen3.6-35B-A3B | 35B / 3B act | — | — | ~487 | c=128 |
| **ornith-ai/Ornith-1.0-35B-FP8** | **~35B MoE (FP8)** | **36.95** | **165 ms** | **~464 (cap)** | **c=128 (still climbing)** |
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
| Non-submissions | 9 empty patches (8 `LimitsExceeded`, 1 `ContextWindowExceededError`) |
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

**This is a clean number: 0 harness errors and 91/100 instances received a fair test verdict.** The 9
non-submissions are genuine model/agent-budget outcomes (8 hit the step/cost limit, 1 blew the context
window), not serving failures — notably the **vLLM/GB10 long-context wedge that plagued the Lightning
SWE-bench run did not recur** on this stack (`0.27.2rc1` aarch64 + Marlin FP8 at util 0.55). The robust
submit step (`git add -A && git diff --cached`, carried over from the Lightning run) again produced
**zero patch-apply errors**.

**The head-to-head is the striking part.** On the identical 100 instances, Ornith resolves **27 that
Lightning misses** while losing only **5** that Lightning gets. That is not sampling noise — it is a
real capability gap on agentic patch generation, and it is consistent with Ornith being purpose-built
by DeepReinforce for agentic coding. Note the honest inversion this creates: Ornith is *behind*
Lightning and Qwen3.6 on GPQA-Diamond general reasoning (69.70% vs 76.26% / 82.32%) yet far *ahead* on
SWE-bench. **Benchmark-suite rank does not transfer across task families** — pick the model for the
job, not for the leaderboard.

### Caveats (read before quoting the number)
- **n=100 shuffled, not the full 500 → indicative (±~5%), not leaderboard-final.** Same convention and
  same seeded instance set as every other SWE-bench number in this repo, so cross-model comparisons
  here are exact even though the absolute score is a sample.
- **The sample is django-heavy (56/100)** and Ornith is especially strong there (44/56 = 79%). Django
  instances skew slightly easier, so a balanced 500-item run would likely land somewhat lower.
- **9 instances never submitted a patch** (agent budget, not model incapability on the merits). With a
  larger step/cost limit the ceiling is higher than 73.

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
- **Raise the SWE-bench step/cost limit** — 8 of the 9 non-submissions were `LimitsExceeded`, so the
  73/100 is a mild underestimate of the model's reach.
- **`--max-num-seqs > 128` re-run** to find the true throughput ceiling (still climbing +11% at the cap).
- **Vision.** The checkpoint is vision-capable; no multimodal benchmark has been run on Warpcore.
- **A non-Marlin FP8 path.** If a future vLLM/CUDA build lands a working CUTLASS W8A8 FP8 kernel for
  SM 12.1, re-measure throughput — the current 36.95 tok/s c=1 is on a compatibility fallback kernel.
- **`reasoning_content` parser fix** — `qwen3` leaves it empty; worth a bug report.

## Reproduce

Serving (on warpcore) — full script: [`raw/launch_ornith.sh`](raw/launch_ornith.sh):
```bash
docker rm -f vllm_ornith 2>/dev/null
docker run -d --name vllm_ornith --gpus all --network host --ipc host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  vllm/vllm-openai:cu129-nightly-aarch64 \
  --model ornith-ai/Ornith-1.0-35B-FP8 \
  --served-model-name ornith-ai/Ornith-1.0-35B-FP8 \
  --moe-backend marlin \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --trust-remote-code --host 0.0.0.0 --port 8000
```
Use `--gpu-memory-utilization 0.55` for the **agentic** runs (host headroom — see the Lightning card's
unified-memory OOM note; the GB10's KV cache shares the same 121 GiB pool as system RAM). Clients must
always send an explicit `max_tokens` (vLLM has no default-request-budget flag) — ≥4k chat, 32k+ reasoning.

Quality (on warpcore, in a detached tmux — the suite takes ~16.6 h) —
[`raw/run_ornith_quality.sh`](raw/run_ornith_quality.sh):
```bash
source /tmp/lmeval-venv/bin/activate      # lm-eval 0.4.12 + None-guard/gather-survive patches
lm_eval --model local-chat-completions \
  --model_args model=ornith-ai/Ornith-1.0-35B-FP8,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=8,max_retries=8,tokenized_requests=False,timeout=3600 \
  --tasks gsm8k_cot_zeroshot_clean --apply_chat_template \
  --gen_kwargs max_gen_toks=8192,temperature=0 --output_path /tmp/lmeval_results/ornith35b/gsm8k
# ifeval: same, --tasks ifeval
# gpqa:   --tasks gpqa_diamond_cot_zeroshot_clean --gen_kwargs max_gen_toks=32768,temperature=0 (num_concurrent=5)
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
