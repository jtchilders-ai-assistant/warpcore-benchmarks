# Why the Laguna SWE-bench run is slow — measured, 2026-08-23

Question asked: *"Why is this taking so long? Is it because of the lack of memory?"*

**Answer: no, it is not memory.** Evidence and the actual causes below.

## Memory is not the constraint

Sampled against the live run at w=4:

| Signal | Value | Interpretation |
| --- | ---: | --- |
| `num_preemptions_total` (Δ over 60 s) | **0** | nothing evicted from KV cache |
| `num_requests_waiting` | **0** | no queueing for capacity |
| `kv_cache_usage_perc` | **0.36–0.41** | under half of the 8.44 GiB KV pool in use |
| `prefix_cache_hits / queries` | **93.5%** | prefix reuse already near-optimal |

A memory-bound server shows rising preemptions and a non-zero wait queue. Neither is
present. KV cache is small (8.44 GiB, from `--gpu-memory-utilization` after ~93 GiB of
weights) but it is **not** the active bottleneck at this concurrency.

## Cause 1 — memory BANDWIDTH (the dominant term)

Laguna-S-2.1 is a **256-expert MoE, 10 experts active per token**, 117.6 B params total,
~93 GiB on disk at NVFP4.

GB10 uses unified LPDDR5X shared between CPU and GPU at roughly **273 GB/s**. Decode is
memory-bound: each token requires reading the active expert weights.

```
active params ≈ 10–21 B depending on expert/dense split
=> ~9–18 GB read per token
=> ceiling ≈ 273 / 13.5 ≈ 15–20 tok/s single-stream
measured single-stream: 17.65 tok/s
```

The model is running **at approximately its hardware ceiling on this box**. `nvidia-smi`
shows 96% GPU utilization at only 22 W — busy stalling on memory, not computing.

Note: a *dense* 118 B model would cap near 2.9 tok/s. We measure 6× that, which is what
confirms the MoE-activation model is the right one. (An earlier dense estimate in this
analysis was wrong and was corrected by reading `num_experts=256, num_experts_per_tok=10`
from the model config.)

## Cause 2 — the agent loop is inherently serial

Each SWE-bench instance is 40–250 **sequential** steps. Every step re-prefills a growing
conversation then decodes at ~9 tok/s under run concurrency. Measured
**prefill:decode ratio = 8.6** — most token throughput goes to re-reading context, not
producing new output. Nothing inside a single instance parallelizes.

## Cause 3 — verbosity failures (the real threat to score validity)

**4 of the first 19 instances (21%) exit `RepeatedFormatError`.** The trajectory shows the
mechanism exactly — the harness pushes back three times and then aborts:

> `Your previous response reached the output token limit (finish_reason=length) before you
> produced a tool call, so it was cut off. Respond more concisely and finish with exactly
> one bash tool call.`

The model spends its output budget on `reasoning` and never emits the tool call. This is
the **same reasoning-field pathology** that depressed its GSM8K and IFEval scores
(ISSUES.md #15) — but unlike those cases it is **not** a harness artifact that can be
recovered post hoc. The tool call genuinely was never produced, so the step does no work.

Each such instance yields an **empty patch** and is scored unresolved. If the 21% rate
holds, ~21/100 instances are lost to output-formatting failure before coding ability is
even tested. **The final score must be published as a conservative floor**, with the
RepeatedFormatError count reported alongside it.

## Correction to an earlier claim in SMOKE_AND_SETUP.md

That file states *"46% of completions finish on `length`"*, citing
`vllm:request_success_total{finished_reason="length"}`. **That number is not valid for this
run.** Those counters are cumulative over the entire server lifetime and include the
GSM8K / IFEval / GPQA lm-eval runs, which used far smaller `max_tokens`.

Sampling **deltas** over a live 100 s SWE-bench window instead
(`finish_reason_delta.py`) shows completions finishing on `stop` with typical lengths of
200–2000 tokens.

The 5400 s timeout fix remains **empirically justified** — 1800 s did fire twice in the
smoke, and there have been **0 timeouts in 11 h** since the change — but the stated
rationale overstated how often full-cap steps occur. Long steps are real but not the
common case.

## Runtime: earlier estimate was wrong by ~4×

| | |
| --- | --- |
| Earlier projection (from 3 smoke instances) | ~14 h |
| **Observed** | 19/100 in 11.3 h = **1.68 inst/h** |
| **Projected total** | **~60 h (2.5 days)** |

The smoke sample was small and unrepresentative. Quote observed run rate, never a
three-instance extrapolation.

## Measured concurrency headroom

Probed against the live server while the run was in flight:

| Workers | Aggregate | vs w=4 | Per-request | Full 32768-tok step needs | Safe at 5400 s? |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 4 (current) | 37.4 tok/s | 1.00× | 8.79 tok/s | 3727 s | yes |
| 8 | 57.2 tok/s | 1.53× | 7.23 tok/s | 4532 s | yes |
| 12 | 66.6 tok/s | 1.78× | 5.37 tok/s | 6102 s | **no** |

Scaling is sub-linear because per-request bandwidth share falls as workers rise. w=8 would
cut the projection to ~37 h and is safe under the current timeout; **w=12 would reintroduce
the timeout bug** unless the deadline is raised again.

**Decision: left at w=4.** Restarting a healthy run is a state change not worth making
without the owner's say-so, and w=4 exactly matches Ornith's worker count, keeping
throughput conditions identical to the baseline. Resume is free if that changes (same
`-o`, no `--redo-existing`).

## Reusable scripts

- `concurrency_headroom.py` — marginal aggregate throughput vs added concurrency, live
- `finish_reason_delta.py` — finish-reason + length distribution over a **window**, not
  lifetime cumulative counters
