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
| **GPQA-Diamond** (0-shot CoT) | 198 | exact_match, **64k budget** | **76.26%** ✅ |
| | | exact_match, 32k budget | 66.16% (±3.36) |
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

> **⚠️ GPQA-Diamond is output-budget-limited, not capability-limited — read this before comparing.**
> Lightning is a *deep* reasoner: on hard grad-level questions it emits a long chain-of-thought (which
> vLLM's `nemotron_v3` parser strips from `content` but which still counts against `max_tokens`), then a
> short final answer. If the budget is exhausted mid-reasoning, `content` comes back empty and the item
> auto-scores **wrong** — so a low budget measures the *budget*, not the model.
>
> | Budget | Raw acc | Items answered | Truncated | Acc on *answered* items |
> | ------ | ------- | -------------- | --------- | ----------------------- |
> | 16k    | 53.03%  | 116 / 198      | 82 (41.4%)| 105/116 = 90.5%         |
> | 32k    | 66.16%  | 157 / 198      | 41 (20.7%)| 131/157 = 83.4%         |
> | **64k**| **76.26%** | **192 / 198** | **6 (3.0%)** | 151/192 = 78.7%      |
>
> Each budget increase roughly halved the truncation rate and lifted the raw score (+13 pts to 32k, +10
> more to 64k). At **64k, 97% of items finish** and the score is essentially capability-bound; the last
> 6 items still truncate (a handful genuinely need >64k). The `answer-line` and `flexible-fallback`
> filters agree exactly on GPQA, so the gap was never a parsing artifact — purely unfinished reasoning.
> **64k is the headline** cross-card number; 16k/32k are retained to document the curve.
>
> **How many tokens does GPQA actually need?** Replaying the 41 items that truncated at 32k with a 64k
> budget: 35 finished, needing **p50 ≈ 30.5k, p90 ≈ 51.2k, max ≈ 53.5k completion tokens** (verified
> `finish_reason: stop`; the visible answer is still only ~200 tokens — the rest is stripped CoT). This
> is legitimate long reasoning reaching correct answers, not a repetition loop.

**Takeaway:** near-ceiling on GSM8K (95%), strong instruction-following (86% strict IFEval), and — with
an adequate output budget — **76%+ on GPQA-Diamond**, which puts a 30B/3B-active model just behind
Qwen3.6-35B (82%) and clearly ahead of Nemotron-3-Super-120B (64%). Its GPQA capability is
**budget-bound, not knowledge-bound**.

### Recommended output-token budget (`max_tokens`) — and whether it's a realistic serving setting

Raising the budget is **not score-gaming**: the extra tokens are real reasoning that reaches correct
answers, and `max_tokens` is a **ceiling, not a reservation** — it costs nothing for the ~80% of GPQA
items (and virtually all GSM8K/chat, visible output p50 ~200 tok) that finish fast. Memory is a
non-issue on this box (~18 GiB weights, ~88 GiB KV, 23× concurrency at 256K context). The only real
cost of a high ceiling is **latency on the rare deep request** (~13 ms/token single-stream → a
30k-token trace ≈ 4 min). So the right budget depends on the workload:

| Workload | Recommended `max_tokens` | Trade-off |
| -------- | ------------------------ | --------- |
| Interactive agent / chat | **8k–16k** | bounds latency; the hardest ~10–20% of deep problems get truncated |
| Batch / offline reasoning | **32k–64k** | full capability; deep problems run to completion |
| (never below ~4k) | — | truncates ordinary reasoning and silently drops the final answer |

The server permits any of these — `--max-model-len 262144` allows up to ~256K output — and the
effective budget is chosen **per request** by the client (there is no vLLM flag for a *default* request
`max_tokens`, so always send one explicitly). This guidance is baked into the serving launch script
([`raw/launch_lightning.sh`](raw/launch_lightning.sh)). **Bottom line:** 32–64k is a setting we'd
genuinely run for a reasoning/agentic workload, so the 76.26% number reflects a real deployment, not a
benchmark-only config.

## Agentic coding — pi-30 (measured 2026-08-14)

[`rick-stevens-ai/pi-30`](https://github.com/rick-stevens-ai/pi-30): a 30-problem agent-loop coding
benchmark driven by the `pi` CLI (v0.84.2). The model must *use tools* (read/write files, run code) to
iteratively fix or build a solution; verdicts come from **verifier exit codes**, never model prose.
Problems span iterate-until-green pytest fixes, oracle-matching, generator+critic loops, and best-of-N
tournaments.

| | |
|---|---|
| **Score** | **29 / 30 passed** |
| Harness | `pi` 0.84.2, `PI_TIMEOUT=600` (deep reasoner needs the headroom), temp per pi defaults |
| Serving | `vllm/vllm-openai:v0.27.1`, `--moe-backend marlin`; **pi-30 headroom config** `--gpu-memory-utilization 0.55 --max-num-seqs 32` (see note) |
| Tool-calling | `qwen3_coder` parser — clean, 16/16 concurrent tool-call stress passed |
| Raw | [`raw/pi30/`](raw/pi30/) (RESULTS.txt, SUMMARY.txt, log) |

**The single miss — P24 (token-bucket rate limiter):** a genuine correctness bug, not infra. The bucket
must start **full** so a burst of 5 requests is allowed at t=0; Lightning's implementation allowed only 1
(it didn't seed the bucket to capacity). Verifier: `BURST: allowed 1 at t=0, expected exactly 5`.
Everything else — including all five best-of-N tournaments (P9/P10/P19/P27/P30), both measurement
problems (P15 52.6 Mtok/s, P23 331 Mn/s), and the P5 GFLOP/s hardware problem (71.9 GFLOP/s) — passed.

29/30 is a strong agentic result and consistent with a model *designed* for tool use. Note the cross-model
caveat: pi-30 rank does not track quality rank (a weaker-on-GPQA model can still ace pi-30), so treat this
as an independent signal.

> **⚠️ Serving note — unified-memory OOM (why the config differs from the throughput card).** The first
> pi-30 attempt crashed mid-run and scored a bogus ~3/30. Root cause: on the GB10 the GPU KV cache shares
> the **same 121 GiB pool** as system RAM. At `--gpu-memory-utilization 0.9` vLLM held ~116 GiB, leaving
> only ~5 GiB for the host — and pi-30 runs its agent processes (node `pi` + python verifiers + pytest,
> with tournaments fanning out 3–4 parallel `pi` calls) **on the same host**, so the kernel OOM-killer took
> vLLM (which, launched `--rm`, vanished without a trace). The fix is to leave host headroom:
> `--gpu-memory-utilization 0.55 --max-num-seqs 32` → ~46 GiB free, and the 30B model still gets ~50 GiB KV
> (ample for pi-30's short contexts). Raw inference concurrency was *not* the cause — the endpoint survived
> 16 concurrent tool-call requests at 0.9. Keep the high-util config for pure remote-client serving only.

## Agentic coding — SWE-bench Verified (measured 2026-08-17)

[SWE-bench Verified](https://www.swebench.com/) via [`mini-swe-agent`](https://github.com/SWE-agent/mini-swe-agent)
(bash-only agent loop): given a real GitHub issue + repo, the model must produce a patch that makes the
repo's hidden test suite pass, graded pass/fail inside a per-instance x86 Docker container. The agent loop
and test containers run on the **client Mac mini (x86_64)**; the model is served on Warpcore.

**Result: 28/55 resolved = 50.9%** — a representative random sample (`--shuffle`, same seed as the
Qwen3.6 run) spanning **11 repos**, scored with **0 harness errors and 0 empty patches** (every instance
got a fair test verdict).

| | |
|---|---|
| Resolved | **28 / 55 (50.9%)** |
| Sample | representative random (`--shuffle --slice 0:100`, reached 55/100 before the endpoint wedged) |
| Repos spanned | 11 (django 32, sphinx 5, sympy 4, scikit-learn 4, astropy/pydata/pylint 2 each, others 1) |
| Scoring cleanliness | 0 errors, 0 empty patches — all 55 fairly evaluated |
| Baseline (Qwen3.6-35B) | 44% (n=100) — **Lightning is ahead** |
| Agent | mini-swe-agent 2.4.6, native tool-calling, `temp=0`, per-step `timeout=1800` |
| Serving | `vllm/vllm-openai:v0.27.1`, marlin; SWE-hardened `--gpu-memory-utilization 0.80 --max-model-len 131072 --max-num-seqs 8` |

**Per-repo breakdown** (resolved / attempted): django 19/32, scikit-learn 3/4, sphinx 2/5, sympy 1/4,
matplotlib 1/1, pallets 1/1, pylint 1/2, astropy 0/2, pydata 0/2, pytest 0/1, requests 0/1. The gradient
is the expected one — strong on django/sklearn, weak on the hard tail (sympy, astropy) — and mirrors the
Qwen3.6 pattern.

### Caveats (read before quoting the number)
- **n=55, not 100 → indicative (±~7%), not leaderboard-final.** The target was n=100; the run reached 55
  before the serving endpoint wedged (below). 50.9% is a legitimate representative-sample number, but it
  is not the full Verified score.
- **Sample is django-heavy (32/55 = 58%).** Django instances skew slightly easier, so the true
  full-sample number could be a touch lower. It spans the same 11 repos as the baseline, so it is
  *representative*, just not perfectly balanced.
- **The number is a conservative floor** — 11 of the 55 were wall-clock `Timeout` at the default budget,
  concentrated on hard repos; a larger time budget would likely recover a few.

### Why it stopped at 55: a recurring vLLM/GB10 engine wedge (serving bug, not the model)
Under sustained SWE-bench load the served engine repeatedly entered a **deadlock**: `GET /v1/models`
returns 200 instantly (metadata only) but a real `POST /v1/chat/completions` **hangs past the timeout
(HTTP 000)** with the GPU pegged ~96%. The agent workers block on the never-returning completion and
progress silently stops (up to ~12 h overnight before a watchdog caught it). This recurred **three times
across three serving configs**, including a deliberately hardened build (`gpu-util 0.80` +
`max-model-len 131072` + `max-num-seqs 8`, giving ~40× KV headroom for even 128K-token requests, verified
by a 3×-concurrent-60K-token stress probe that passed). Because it survives generous KV headroom, this is
a **genuine vLLM/GB10 engine bug under long-context MoE generation, not a tunable KV-pressure issue** —
and it is **model-independent serving instability**, not a Lightning capability limit. Diagnose by hitting
BOTH endpoints (never trust `/models` alone); recover with `docker rm -f` + relaunch, then prove a real
completion returns 200 fast before resuming. `--shuffle` is seeded (`random.seed(42)`), so a resume with
the same `-o` dir skips completed instances and re-runs only the remainder.

### Harness fix: `cat patch.txt` submit was silently zeroing solved instances
mini-swe-agent's stock `swebench.yaml` submit step is a multi-command ritual ending in
`echo COMPLETE… && cat patch.txt`, and the submitted patch is whatever that prints. When Lightning
deviated from the ritual (edited the file a different way, or `cat`'d the wrong thing), the submission
captured **raw file-contents or an error string instead of a diff** → the grader reported
`Patch Apply Failed: **** Only garbage was found in the patch input` and the instance scored a hard zero.
An initial partial score showed **18 such "errors" out of 57** — the model had done the work, the harness
just never received a valid diff. Fix: a robust submit that regenerates the diff from git state
regardless of how the model edited:
```yaml
    echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached
```
After this fix, **all 55 instances produced valid diffs (0 patch-apply errors)** — the 50.9% above is
post-fix and clean. Raw results, the robust config, and the launch script are under
[`raw/`](raw/).

## Not yet measured / next steps

- **Complete SWE-bench to n=100** — requires resolving the vLLM/GB10 long-context wedge above (a newer
  vLLM build where the engine deadlock is fixed), then re-running the same shuffle to fill the remaining
  45 instances. The 55 already done would be reused.
- **GPQA at 128k budget** for the last 6 items that still truncate at 64k (would close the remaining 3%;
  76.26% @64k is already 97%-answered and essentially capability-bound).
- **DSpark speculative decoding.** NVIDIA ships a matching draft model
  (`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark`) and the README's recommended serve
  command enables it (`--speculative_config.num_speculative_tokens 3 --speculative_config.model
  $DSPARK_CKPT --mamba-backend flashinfer --mamba-cache-mode align`). This baseline was run **without**
  spec-decode for a clean number; enabling DSpark is the lever to chase lower latency / higher tok/s.
- **`--max-num-seqs 256` re-run** to find the true throughput ceiling (there's ~23× KV headroom).

## Reproduce

Serving (on warpcore) — see [`raw/launch_lightning.sh`](raw/launch_lightning.sh) for the full script
with the output-token-budget guidance baked in as comments:
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
The server allows up to ~256K output; choose the per-request `max_tokens` by workload (**8k–16k**
interactive, **32k–64k** batch/reasoning — see the budget section above). Clients must send an explicit
`max_tokens` (vLLM has no default-request-budget flag).

Throughput sweep (reusable `scripts/vllm_sweep.sh` from the `warpcore-dgx-spark` skill):
```bash
MODEL=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 CONTAINER=vllm_lightning \
  OUTDIR=~/lightning_sweep bash ~/vllm_sweep.sh 1 2 4 8 16 24 32 48 64 96 128
```
(Raw completions, `--ignore-eos`, 512-in/256-out; on-box against `localhost:8000`.)
