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
| gpt-oss-120b | 30 / 30 | none (on the crash-fixed image) |

At **29/30**, Qwen3.6-35B is within one problem of the two 120B models (both 30/30) on agentic coding —
a strong showing at ¼ the size. Its one miss (P2, LRU cache) is a genuine model error (staging
verified). Notably it passed several problems cleanly on the first try. Raw per-problem log:
[`raw/pi30/RESULTS.txt`](raw/pi30/RESULTS.txt).

## Agentic coding — SWE-bench Verified

[SWE-bench Verified](https://www.swebench.com/): resolve real GitHub issues by producing a patch that
makes the repo's hidden test suite pass. Run with [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)
v2.4.6 (bash-only agent loop, no custom scaffolding). **The agent and the x86 test-execution Docker
containers run on a client Mac (x86_64); only the model is served on Warpcore** — this keeps results
leaderboard-comparable (Warpcore is aarch64 and can't run the x86 SWE-bench images itself). Native
tool-calling via `--tool-call-parser qwen3_xml`; `temperature=0`; default 250-step agent limit;
per-request timeout 1800 s.

**Reported on a representative random 100-instance sample** (`--shuffle`, seed-free) of the 500-instance
Verified set — **not** the full set. A contiguous slice was explicitly avoided: the first 50 instances
alphabetically are only 2 repos (astropy + django, the easier end) and over-scored at 52%. The random
100-sample spans **11 repos** and is the honest indicative number.

| Metric | Value |
| ------ | :---: |
| **Resolved** | **44 / 100 = 44%** (±~5% at n=100) |
| Unresolved (patch applied, tests failed) | 22 |
| Empty patch (no fix submitted) | 29 |
| Harness error | 5 |

**Per-repo (resolved / sampled):**

| Repo | Resolved | Repo | Resolved |
| ---- | :------: | ---- | :------: |
| django | 26 / 56 | pydata/xarray | 2 / 3 |
| sphinx | 5 / 10 | matplotlib | 1 / 2 |
| scikit-learn | 3 / 5 | pylint | 1 / 2 |
| astropy | 2 / 5 | pytest | 1 / 4 |
| sympy | 2 / 10 | pallets/flask | 1 / 1 |
| requests (psf) | 0 / 2 | | |

**Reading it:** 44% on a representative sample is a **strong result for a 35B model** — SWE-bench
Verified is a hard, execution-graded coding benchmark, and this is competitive with much larger models.
Consistent with pi-30 (29/30), Qwen3.6-35B is a capable agentic coder well above its weight class.

**Caveat — the score is a conservative floor.** 29 of the 100 attempts submitted an **empty patch**, the
majority because the agent hit the wall-clock timeout mid-work on the harder repos (notably **sympy,
2/10**) before it could submit — Qwen3.6's lengthy chain-of-thought eats the per-instance budget on hard
problems. A larger per-instance time budget would likely recover several of those, so 44% is the
low-generosity number. Defaults were kept for comparability.

> This is a **100-sample**, not the full 500 — treat 44% as indicative (±~5%), not the exact
> leaderboard figure. Raw report, predictions, and per-instance exit statuses:
> [`raw/swebench/`](raw/swebench/).

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

### SWE-bench Verified (agent + scoring)

Agent and x86 test containers on a client Mac; model served on Warpcore. mini-swe-agent v2.4.6,
`swebench` 4.1.0 eval harness. (Serve Qwen3.6 with `--tool-call-parser qwen3_xml --enable-auto-tool-choice`.)

```bash
# 1) Agent phase — representative random 100-sample, writes preds.json
export OPENAI_API_KEY=dummy OPENAI_API_BASE=http://csi370295.alcf.anl.gov:8000/v1
export MSWEA_COST_TRACKING=ignore_errors
CFG=$(python -c "import minisweagent,os;print(os.path.join(os.path.dirname(minisweagent.__file__),'config/benchmarks/swebench.yaml'))")
mini-extra swebench --subset verified --split test --shuffle --slice 0:100 --workers 3 \
  -m "openai/Qwen/Qwen3.6-35B-A3B-FP8" \
  -c "$CFG" -c model.model_kwargs.temperature=0 -c model.model_kwargs.timeout=1800 \
  -o swebench_out

# 2) Scoring — builds x86 test containers, applies patches, runs test suites
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path swebench_out/preds.json \
  --max_workers 4 --run_id qwen_shuffle100 --namespace swebench
# -> openai__Qwen__Qwen3.6-35B-A3B-FP8.qwen_shuffle100.json  (resolved_instances / 100)
```

Notes: `swebench` pulls in a hard `import modal` (cloud path) that fails to build on some hosts — install
with `--no-deps` plus the real deps and stub the `modal` package if you only run local Docker eval. The
default 250-step agent limit is required (lower caps cause premature `LimitsExceeded`); the 1800 s
per-request timeout accommodates long reasoning. The full pipeline (with all pitfalls) is captured in the
Hermes skill `swebench-vllm-endpoint`. Raw report + predictions: [`raw/swebench/`](raw/swebench/).
