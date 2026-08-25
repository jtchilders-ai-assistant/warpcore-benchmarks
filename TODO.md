# TODO — re-runs needed for a systematic, apples-to-apples comparison

**Status as of 2026-08-24.** This file tracks the work required to turn the per-model cards in
[`results/`](results/) into a *systematic* comparison. Models here were benchmarked over roughly a
month (2026-07-27 → 2026-08-24), and the harness, the serving stack, and our understanding of the
failure modes all changed underneath us. Several published numbers are therefore **not comparable to
each other**, and a few are **known-wrong in a direction we can quantify**.

Nothing below is a claim that a model is better or worse than its card says. Each item states what
was measured, why it is suspect, and what measurement would settle it.

> **In flight right now:** Laguna-S-2.1-NVFP4 SWE-bench Verified n=100 is **running** on the Mac mini
> (`screen -r laguna_swe_n100`, started 2026-08-22 23:22, 60/100 instances written to
> `/tmp/laguna_swe_n100/preds.json` as of 2026-08-24 17:27). Do not restart the endpoint
> (`vllm_laguna` on warpcore) or take the GPU for another model until it finishes — see
> [§0](#0-in-flight-do-not-disturb).

---

## 0. In flight — do not disturb

- [ ] **Laguna-S-2.1-NVFP4 — SWE-bench Verified, n=100 seed-42 shuffle**
  - Host: Mac mini (`csi0359637.cels.anl.gov`), `screen -ls` → `16175.laguna_swe_n100`.
  - Output: `/tmp/laguna_swe_n100/` (60 instance dirs, 39 non-empty patches so far).
  - Endpoint: `vllm_laguna` container on warpcore, up 2 days.
  - It is slow by design — Laguna is 17.65 tok/s single-stream, the slowest model in the repo
    (see [the card](results/laguna-s-2.1-118b/README.md#throughput-sweep)).
  - **On completion:** grade with the SWE-bench harness, write the card's Agentic section, and add
    the row to the top-level README. Then the GPU is free for §1.

---

## 1. Known-wrong scores — re-serve required (highest value per GPU-hour)

The root cause is [ISSUES.md #15](ISSUES.md): with `--reasoning-parser`, vLLM can emit
`content: null` with the full answer in `message.reasoning`; lm-eval reads only `content`, so those
items score **zero without ever being answered**. HTTP 200, `finish_reason: stop`, tokens billed, no
retries, no truncation — invisible unless you count empties in `samples_*.jsonl`.

Audit of every retained sample file (recomputed 2026-08-24, not copied from the cards):

| Model / task | Published | Served-only | Empty | In ISSUES #15? |
| --- | ---: | ---: | ---: | --- |
| **Nemotron-3-Super GPQA-Diamond** (16k) | **63.64%** | 88.73% | **56/198 = 28.3%** | ❌ **no — new** |
| **Lightning IFEval** prompt-strict | **86.14%** | 94.33% | **47/541 = 8.7%** | ❌ **no — new** |
| Ornith GPQA-Diamond (32k) | 69.70% | 88.46% | 42/198 = 21.2% | ✅ yes, uncorrected |
| Ornith IFEval prompt-strict | 85.58% | 90.25% | 28/541 = 5.2% | ✅ yes, floor |
| Laguna IFEval prompt-strict | 75.79% | — | 29/541 = 5.4% | ✅ yes, floor |
| Laguna GSM8K | 83.40% → **96.13%** | 97.09% | 186/1319 = 14.1% | ✅ **corrected** |
| Lightning GSM8K | 95.07% | 96.83% | 24/1319 = 1.8% | negligible |
| Ornith GSM8K | 97.19% | 97.27% | 1/1319 = 0.1% | negligible |
| Lightning GPQA-D @32k | 66.16% | 83.44% | 41/198 = 20.7% | recovered → 76.26% @64k |

> ⚠️ **The "served-only" column is an upper bound, not a fix.** On Laguna the 186 recovered items
> scored **90.3%** versus **97.09%** for the items that returned content — the defect does *not* drop
> questions uniformly at random, so exclusion-based estimates are optimistically biased. Only a
> re-serve produces a defensible number. **Do not publish the served-only figures.**

- [ ] **1a. Nemotron-3-Super-120B — re-serve the 56 empty GPQA-Diamond items**
  - Worst empty rate in the repo (28.3%). Published 63.64% is an underestimate of unknown size;
    the true value is somewhere in (63.64%, 88.73%).
  - Confounded: it also ran at a **16k** budget, and Lightning's curve shows 16k costs a deep
    reasoner ~23 points on GPQA. Truncation and the `reasoning`-field defect leave *identical*
    evidence in the sample file, so **replay at 64k** and separate the two: record
    `finish_reason` and `completion_tokens` per item.
  - Stakes: it currently loses GPQA to gpt-oss-120b by 9.1 points. The artifact is larger than the gap.
  - Method: `results/nemotron-3.5-lightning-30b/raw/gpqa_64k_replay.py` (proven — recovered 35/41
    truncated items, p50 30,525 completion tokens, max 53,519), plus
    `results/laguna-s-2.1-118b/raw/quality/recover_empties_via_reasoning_field.py`.
  - Samples retained at `warpcore:/tmp/lmeval_nemotron/gpqa/.../samples_*.jsonl`.
  - Cost: model bring-up + ~2–3 h.

- [ ] **1b. Ornith-1.0-35B — re-serve the 42 empty GPQA-Diamond items**
  - Already flagged in ISSUES #15 as the outstanding uncorrected score. 69.70% published,
    88.46% served-only, truth in between.
  - Samples at `warpcore:/tmp/lmeval_results/ornith35b/gpqa/.../samples_*.jsonl`.
  - Cost: bring-up + ~2 h. Budget is *not* the confound here (only 2 length-truncations across all
    of GPQA at 32k), so this is a clean single-cause recovery.

- [ ] **1c. Lightning IFEval — re-serve the 47 empty items** (and Ornith's 28, Laguna's 29)
  - Smaller effect (8.7% / 5.2% / 5.4%) but IFEval is the one benchmark where all five models are
    within ~10 points of each other, so an 8.7% zero-rate is enough to reorder the table.
  - Lightning samples are committed at
    `results/nemotron-3.5-lightning-30b/raw/quality/ifeval/.../samples_*.jsonl`.

- [ ] **1d. Add Nemotron-3-Super GPQA and Lightning IFEval to the ISSUES #15 impact table**
  - The table currently lists only Laguna and Ornith. Two more affected scores were found in the
    2026-08-24 audit. ISSUES #15 says "any score in this repo taken through lm-eval against a vLLM
    endpoint with a reasoning parser is suspect until audited" — this closes out the audit for
    every model whose samples survive.

- [ ] **1e. Standing rule: always run lm-eval with `--log_samples`, and commit the sample files**
  - **gpt-oss-120b and Qwen3.6-35B sample files were not retained** (checked both hosts — only
    `results_*.json` survives). Their empty-response rates are **permanently unauditable**.
    That is why §3 lists them for full re-runs rather than cheap replays.
  - Also: run a `--limit 40` empty-content smoke on any model served with a model-specific
    `--reasoning-parser` *before* committing to a multi-hour sweep. Minutes of cost, days of
    protection. Check the server startup log for
    `WARNING [vllm.py:1689] Auto-initialization of reasoning token IDs failed`.

---

## 2. Not comparable as printed — methodology drifted between runs

### 2a. SWE-bench: the robust-submit fix landed *after* the Qwen3.6 run

mini-swe-agent's stock `swebench.yaml` submits with `cat patch.txt`, which intermittently captures
raw file contents instead of a diff → "Patch Apply Failed: only garbage found" false zeros. The
Lightning run replaced it with `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff
--cached`. Qwen3.6 ran before that fix.

Verified directly from `results/qwen3.6-35b-a3b/raw/swebench/preds_shuffle100.json`: **4 of its 5
harness errors are raw file contents or the literal string `Need to create patch first` in
`model_patch`** — the exact signature of the capture bug.

Full accounting of the identical 100 instances (all three runs cover the same seed-42 set):

| | Qwen3.6 | Lightning | Ornith |
| --- | ---: | ---: | ---: |
| **fair test verdict** | **66** | **88** (98 after recovery) | **91** |
| empty patch | 29 (22 = `TimeoutExpired`) | 11 | 9 |
| harness error | 5 | 1 | 0 |
| **resolved** | **44** | **47** (51 after recovery) | **73** |

Qwen3.6 was scored on **66 real attempts out of 100**; Ornith on 91. Presenting 44 / 51 / 73 as a
like-for-like ranking is not defensible.

> **RESOLVED 2026-08-25 — the config was recovered, and it changes the diagnosis.** An earlier
> revision of this section speculated that Qwen3.6 might have run under a shorter timeout budget
> than Ornith. **That was wrong.** mini-swe-agent 2.4.6 embeds the fully-resolved config in every
> `info.config` of every trajectory; all 78 surviving Qwen3.6 trajectories carry an *identical*
> config, recovered to
> `results/qwen3.6-35b-a3b/raw/swebench/swebench_qwen36_config.RECONSTRUCTED.yaml` (regenerate with
> `viz/reconstruct_qwen_config.py`). Its limits are **the same as Ornith's**: `step_limit: 250`,
> `cost_limit: 3.0`, per-command `timeout: 60`, LLM `timeout: 1800`, `wall_time_limit_seconds: 0`.
> The limits were never the problem. See the corrected root cause below.

**Root cause of the 22 `TimeoutExpired` exits: Docker image pulls, not the model.** Recovered from
the run log (`raw/swebench/minisweagent_shuffle100.log`):

```
subprocess.TimeoutExpired: Command '['docker', 'run', '-d', ..., 'sleep', '2h']'
  timed out after 120 seconds
```

That is `DockerEnvironment.pull_timeout` (default 120 s), raised while *starting the container* —
before the agent takes a single step. Unlike per-command timeouts, which `docker.py` catches and
feeds back as an observation, this call is **not** wrapped in `try/except`, so it escapes to
`Agent.handle_uncaught_exception` and becomes the instance's exit status.

Two independent confirmations:

- **All 22 `TimeoutExpired` instances have no trajectory file at all; all 78 others do.** A perfect
  22/78 split. The container never started, so no agent ran.
- The log records **44 pull timeouts** across those 22 instances — each was attempted twice and
  failed both times, on a cold image cache.

**These 22 instances measure the Mac mini's Docker image-pull throughput, not Qwen3.6.** The model
was never invoked. This is a pure harness artifact, and a stronger claim than "the timeout budget
may have been unfair": the model provably did not participate in 22% of its own benchmark.

**Second recovered finding: Qwen3.6 used the *stock* submit command.** The log's config-spec line
shows it built from `minisweagent/config/benchmarks/swebench.yaml` with only
`temperature=0, timeout=1800` overridden — i.e. no custom config file. Stock submits with
`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt`; Ornith's committed config uses
`... && git add -A && git diff --cached`. This confirms §2a's robust-submit claim from the run's own
artifacts rather than from recollection, and explains the 5 harness errors whose `model_patch`
holds raw file contents.

**Decomposing the 29-point Ornith − Qwen3.6 gap.** Splitting the score into *completion* (did the
agent produce a gradeable patch?) and *patch quality* (given one, did it resolve?):

| | Qwen3.6 | Ornith |
| --- | ---: | ---: |
| produced a graded patch | 66/100 | 91/100 |
| **of those, resolved** | **66.7%** | **80.2%** |
| headline score | 44 | 73 |

Holding Qwen3.6's own patch quality fixed and giving it Ornith's completion rate yields
**60.7 resolved**. So of the 29-point gap:

- **~16.7 points (57%) is completion** — and we now know 22 of the 29 non-completions are
  **infrastructure**, not the model;
- **~12.3 points (43%) is genuine patch quality.**

Ornith is really better — 80.2% vs 66.7% on graded patches is a solid margin — but the headline
gap is roughly **double the capability difference**. The README currently reports only the
headline.

Caveat on the decomposition: it assumes the pull-failed instances would have resolved at the same
rate as the ones that ran. Since those 22 were selected by *image pull latency* — unrelated to
problem difficulty — that assumption is far safer here than for a genuine model timeout. The
7 `LimitsExceeded` exits are real model failures and stay counted against it.

- [ ] **2a-i. Re-run Qwen3.6-35B SWE-bench n=100** with the robust submit and a warm image cache
      (`docker pull` all 100 images first, or raise `pull_timeout`). The config is now recovered, so
      the re-run can match Ornith's exactly. Cost ~11 h generation + ~20 min grading.
- [ ] **2a-ii. Record the fair-verdict count (`completed_instances`) next to every SWE-bench score**
      in the top-level README table. A resolve rate over 66 attempts and one over 91 are different
      measurements and the table should say so. Report **both** raw resolves and
      resolves-among-graded; the first is the deployment answer, the second isolates capability.
- [ ] **2a-iii. Commit Ornith's `exit_statuses_*.yaml`** — it is the only model missing the
      exit-status breakdown in `raw/swebench/`, so its 9 non-submissions can't be re-audited from
      the repo alone.
- [x] **2a-iv. Recover Qwen3.6's agent config.** ~~Neither config nor launch script is committed.~~
      **Done 2026-08-25**: reconstructed from the trajectories' embedded `info.config` and committed
      alongside the run log. Limits confirmed identical to Ornith's; the gap is pull-timeout
      infrastructure, not an unfair budget.
- [ ] **2a-v. Raise `pull_timeout` (or pre-pull images) for every future SWE-bench run.** 120 s is
      the mini-swe-agent default and it is too tight for a cold cache on this host — it silently
      converted 22 instances into zeros. Pre-pulling is the cheaper fix and belongs in the run
      script.


### 2b. gpt-oss-120b SWE-bench was blocked by a serving bug that no longer applies

79/100 instances aborted with `RepeatedFormatError` caused by vLLM's `--tool-call-parser openai`
corrupting tool-call JSON arguments mid-run — median **12 successful shell commands** before the
abort, i.e. the model was actively solving. Every model since used `qwen3_xml` / `qwen3_coder` and
got 0–1 harness errors.

- [ ] **2b-i. Re-run gpt-oss-120b SWE-bench n=100** on the current stack with a working tool-call
      parser. It is the only model in the repo with no agentic-coding number, and its pi-30 30/30
      says it is capable. Cost ~11 h.

### 2c. GSM8K: two different tasks are in the same column

gpt-oss-120b (83.70%) and Nemotron-3-Super (76.65%) were measured with the **stock**
`gsm8k_cot_zeroshot` `flexible-extract` filter. Every model since used the in-repo **clean-extract**
task with an anchored `The answer is <n>` final line. Note both stock runs report
`exact_match,strict-match = 0.0`, which is the stock task's strict filter failing outright on
reasoning-model output — the same class of parse artifact the clean task was written to fix.

- [ ] **2c-i. Re-run gpt-oss-120b and Nemotron-3-Super GSM8K** on the clean-extract task
      (`results/nemotron-3.5-lightning-30b/raw/gsm8k_cot_zeroshot_clean.yaml`). ~3 h each.
- [ ] **2c-ii. Fix the committed `gsm8k_cot_zeroshot_clean.yaml` before re-using it.** The copy at
      `results/qwen3.6-35b-a3b/raw/gsm8k_cot_zeroshot_clean.yaml` still has **both** bugs from the
      skill reference: `dataset_path: gsm8k` (rejected by lm-eval ≥ 0.4.12 — must be
      `openai/gsm8k`) and the chained normalize-regex in the `answer-line` filter that truncates the
      answer to its **first digit**. That second bug is exactly why Qwen3.6's `answer-line` reads
      **18.88%** while `flexible-fallback` reads **97.04%** on the same items. Ship one anchored
      regex with `group_select: -1` and push comma/`$`/`.` stripping onto the metric via
      `regexes_to_ignore`.
- [ ] **2c-iii. Fix the committed `gpqa_clean_task.yaml`** (`results/gpt-oss-120b/raw/`) — its
      `flexible-fallback` uses `multi_choice_regex`, which raises `KeyError: 'choices'` at
      *scoring* time on lm-eval ≥ 0.4.12, discarding a completed run. Replace with a plain
      `\(([A-D])\)` regex.

### 2d. Output budget is a first-class variable and it is not held constant

Lightning's measured GPQA-Diamond curve: **16k → 53.03%** (41% truncated), **32k → 66.16%** (21%),
**64k → 76.26%** (3%). A 23-point swing from budget alone.

Current budgets in the repo:

| Model | GPQA budget | Concurrency |
| --- | ---: | ---: |
| gpt-oss-120b | 16,384 | 4 |
| Nemotron-3-Super-120B | 16,384 | 6 |
| Qwen3.6-35B | 16,384 | 8 |
| Ornith-1.0-35B | 32,768 | 5 |
| Lightning (reported) | 65,536 | — |

- [ ] **2d-i. Standardise on a 64k GPQA budget** for reasoning models, or publish the budget in the
      README table next to every GPQA score. Right now the column silently mixes three budgets.
- [ ] **2d-ii. Re-run Qwen3.6-35B GPQA in *thinking* mode at 64k.** The published **82.32%** is the
      **non-thinking** run. Thinking mode at 16k scored **33.84%** — that is a truncation artifact,
      not a capability measurement, and the model's intended mode has never been measured properly.
      ~8 h.

### 2e. Do the timeout arithmetic before launching

```
per_request_tok_s  = aggregate_tok_s_at_C / C
worst_case_seconds = max_gen_toks / per_request_tok_s
```

If `worst_case_seconds > client timeout`, **the run cannot converge** — long items are cut off and
retried from scratch, and every retry hits the same wall. This burned ~13 h on Laguna GPQA (c=16,
~4 tok/s per request, 32k budget = ~2.2 h worst case against `timeout=3600`; **352 TimeoutError /
retry events**, abandoned at 110/198). Raising concurrency to "go faster" is what creates the storm.

- [ ] **2e-i. Add a preflight check to the quality runner** that computes the above from the model's
      own throughput sweep and refuses to launch if the arithmetic doesn't close. Note Qwen3.6's
      GSM8K run inherited `num_concurrent=6` and Nemotron-3-Super's `num_concurrent=8` — neither was
      derived from that model's own sweep.

---

## 3. Throughput methodology

- [ ] **3a. Retire the "still climbing at the top of the sweep ⇒ `--max-num-seqs` cap" heuristic.**
      It was **falsified on Laguna**: a +40.8% step at c=128 looked exactly like a cap, but c=192
      added only +10.9% and c=256 only +2.9%. The real peak was ~14% above c=128, not the large
      headroom the slope implied. Also, `SchedulerConfig.max_num_seqs` defaulted to 128 while the
      engine admitted 150–172 concurrently — the documented default is not the live ceiling.
      **Extend the sweep past the knee, or report the top measured point as a measured point.**
- [ ] **3b. Re-sweep Lightning (~719 tok/s) and Ornith (~464 tok/s) past c=128.** Both peaks are
      labelled "floors capped by `--max-num-seqs 128`" on the strength of the heuristic 3a just
      retired. Either extend to c=256/384 or restate them as measured points. ~2 h each.
- [ ] **3c. Re-sweep Nemotron-3-Super cleanly.** Its curve was run across a **server restart at two
      different `--max-num-seqs` values** (24 for c=1→24, then 128 for c=32→128), which is why there
      is a c=32 warmup spike. That is not a single clean curve. ~2 h.
- [ ] **3d. Never extrapolate a peak from an instantaneous `/metrics` delta.** A 20–30 s sample of
      `vllm:generation_tokens_total` mid-run gave 333.5 tok/s where the completed `vllm bench serve`
      finished at 258.77 — a 29% overstatement, published then retracted. Live metrics are for
      liveness and diagnosis only.

---

## 4. Never measured

- [ ] **4a. Qwen3.5-122B-A10B-int4 — the entire quality + agentic suite.** Serving is verified and
      throughput is measured (~228 tok/s at c≈192, 26.9 tok/s single-stream); GSM8K, IFEval,
      GPQA-D, pi-30 and SWE-bench have never been run.
- [ ] **4b. Qwen3.5-122B with MTP / speculative decoding enabled.** This is the model behind the
      Reddit "50 tok/s on DGX Spark" report; we measured 26.9 tok/s single-stream without spec
      decode. That is the lever toward the reported figure and it is untested.
- [ ] **4c. Laguna GPQA-Diamond.** Abandoned at 110/198 after ~13 h to the timeout storm in §2e.
      Re-run at c=4 with a timeout that clears the worst case.
- [ ] **4d. AutomationBench.** Listed in the README's "What's measured" section; no model has a score.

---

## 5. Reporting hygiene (cheap, do alongside the re-runs)

- [ ] **5a. Put the measurement date and harness version in every README table row.** Runs span
      2026-07-27 → 2026-08-24 across at least four vLLM builds (`v0.27.1`, `v0.27.2rc1`,
      `cu129-nightly-aarch64`, `eugr/spark-vllm:latest`).
- [ ] **5b. Report empty-response rate as a column alongside every lm-eval score.** It is the single
      statistic that would have caught all five affected scores on the day they were produced.
- [ ] **5c. Mark superseded numbers in place rather than replacing them.** Lightning's committed
      SWE-bench report says **47** resolved; the card says **51** after re-running 11 wedge-denied
      instances. Both are real and the repo should show the provenance of the correction, not just
      the final figure.
- [ ] **5d. pi-30 is saturated and should be retired as a discriminator.** Four models at 29–30/30.
      Keep it as a bring-up smoke test; stop reporting it as a capability comparison.

---

## 6. Artifact & provenance standard (makes everything above enforceable)

The re-runs in §1–§4 are wasted effort if the new runs are as unreproducible as the old ones. The
standard is written up in **[PROVENANCE.md](PROVENANCE.md)**: a per-run `manifest.json`, a required
artifact list per benchmark, and rules for recording a configuration that changes mid-run.

Two gaps that block the apples-to-apples comparison specifically:

- [ ] **6a. Ornith has no `exit_statuses_*.yaml`.** Best score in the repo; its 9 non-submissions
      cannot be audited from the repo alone. (Same as 2a-iii — listed here because it is an
      artifact gap, not a re-run.)
- [x] **6b. Qwen3.6 has no agent config and no launch script.** **Recovered 2026-08-25** — see
      §2a. mini-swe-agent embeds the resolved config in every trajectory, so the effective config
      was reconstructible from the run's own output. The launch script is still absent, but the
      recovered config plus the log's config-spec line make it redundant.

Mid-run configuration changes are already a live problem, not a hypothetical:

- [ ] **6c. Lightning's SWE-bench run spans two vLLM builds** — `launch_lightning_swe.sh` pins
      `vllm/vllm-openai:v0.27.1`, `launch_lightning_swe_nightly.sh` pins `cu129-nightly-aarch64`,
      after the engine wedged under long-context load. **Nothing records which instances ran under
      which build.** Reconstruct the split from run logs if possible; otherwise annotate the score
      as segmented with the boundary unrecorded.
- [ ] **6d. Fix the misleading segment filenames.** `exit_statuses_n55.yaml` contains **23**
      instances; `exit_statuses_all_segments.yaml` and `exit_statuses_shuffle100_final_segment.yaml`
      both contain 28 despite the first implying it is the union. Name files for their real
      contents.
- [ ] **6i. Reconcile `launch_ornith.sh` with the run it documents.** The script pins
      `--gpu-memory-utilization 0.90`; the card says the agentic runs used **0.55** for host
      headroom, and the card itself prints both values in different sections. A committed launch
      script that does not reproduce its own run is worse than none, because it looks
      authoritative. Same audit for every other `launch_*.sh` in the repo.

Tooling, so the standard is cheaper to follow than to skip:

- [ ] **6e. `make manifest MODEL=<m> BENCH=<b>`** — scaffold `manifest.json`, auto-filling engine
      version/args probed from the live endpoint, client host, repo commit, timestamps. Unknowns
      written as the literal `"unrecorded"`, never guessed.
- [ ] **6f. `make check-artifacts`** — fail if any `results/*/raw/*/` lacks a required artifact for
      its benchmark. Wire into CI so gaps surface at commit time.
- [ ] **6g. Normalize quality artifact paths.** Some models use `raw/quality/<task>/`, others
      `raw/<task>_results.json`. Pick `raw/<benchmark>/` and move the rest.
- [ ] **6h. Commit `samples_*.jsonl` for every lm-eval run** (gzipped if size is a concern). It is
      the only artifact that permits after-the-fact detection of the ISSUES #15 defect.

---

## Suggested order

Ranked by information gained per GPU-hour. §0 blocks everything that needs the warpcore GPU.

| Order | Item | Cost | Unblocks |
| --: | --- | --- | --- |
| 1 | §0 Laguna SWE-bench finishes + graded | in flight | the GPU |
| 2 | §6a/6b artifact recovery (Ornith exit_statuses, Qwen3.6 config) | ~1 h, no GPU | whether 2a-i is even needed |
| 3 | §1a Nemotron-3-Super GPQA re-serve @64k | ~3 h | a possibly-wrong head-to-head vs gpt-oss |
| 4 | §1b Ornith GPQA re-serve | ~2 h | the last known-suspect score |
| 5 | §2c-ii/iii task-YAML fixes | ~30 min | every subsequent lm-eval run |
| 6 | §6e/6f manifest + artifact check | ~2 h, no GPU | every re-run below lands reproducible |
| 7 | §2a-i Qwen3.6 SWE-bench re-run | ~11 h | the most misleading number in the table |
| 8 | §2b-i gpt-oss SWE-bench | ~11 h | the only missing agentic score |
| 9 | §2c-i GSM8K clean-task re-runs (×2) | ~6 h | GSM8K column comparability |
| 10 | §2d-ii Qwen3.6 GPQA thinking @64k | ~8 h | Qwen3.6's real reasoning ceiling |
| 11 | §3b/3c throughput re-sweeps | ~6 h | three "floor" peaks |
| 12 | §4a Qwen3.5-122B full suite | ~20 h | the one model with no quality data |

Items 2 and 6 are new and deliberately placed early: they cost no GPU time, and doing them *before*
the re-runs is what makes the re-runs worth doing. Item 2 in particular may show that Qwen3.6 ran
under a shorter timeout than Ornith, which would change how §2a-i is configured.

**Definition of done for "systematic":** every model in the top-level README table measured with the
same task configs, the same output budget, a concurrency derived from its own throughput sweep, a
verified-zero (or explicitly reported) empty-response rate, and — for SWE-bench — the same scaffold,
the same seed-42 instance set, and the fair-verdict count published next to the resolve rate.
Every run additionally carries a `manifest.json` conforming to [PROVENANCE.md](PROVENANCE.md), so
the settings behind any number can be recovered from the repo without asking a person.

