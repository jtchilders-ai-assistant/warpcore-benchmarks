# TODO — re-runs needed for a systematic, apples-to-apples comparison

**Status as of 2026-09-08.** This file tracks the work required to turn the per-model cards in
[`results/`](results/) into a *systematic* comparison. Models here were benchmarked over roughly a
month (2026-07-27 → 2026-09-08), and the harness, the serving stack, and our understanding of the
failure modes all changed underneath us. Several published numbers are therefore **not comparable to
each other**, and a few are **known-wrong in a direction we can quantify**.

Nothing below is a claim that a model is better or worse than its card says. Each item states what
was measured, why it is suspect, and what measurement would settle it.

---

## 1. Known-wrong scores — re-serve required (highest value per GPU-hour)

The root cause is [ISSUES.md #15](ISSUES.md): with `--reasoning-parser`, vLLM can emit
`content: null` with the full answer in `message.reasoning`; lm-eval reads only `content`, so those
items score **zero without ever being answered**. HTTP 200, `finish_reason: stop`, tokens billed, no
retries, no truncation — invisible unless you count empties in `samples_*.jsonl`.

Audit of every retained sample file (recomputed 2026-09-08, not copied from the cards).
**Note: Ornith empties at 64k are 64k-budget residuals (`finish_reason=length`), NOT parser empties
(`finish_reason=stop`). See ISSUES.md #15 for the classification table.**

| Model / task | Published | Served-only | Empty | In ISSUES #15? |
| --- | ---: | ---: | ---: | --- |
| **Nemotron-3-Super GPQA-Diamond** (64k composite) | **73.74%** | — | **31/198 = 15.7%** | ⚠️ budget residual (`finish=length`) — corrected; residuals counted wrong |
| **Lightning IFEval 64k composite** prompt-strict | **93.35%** | — | **5/541 = 0.9%** | ⚠️ budget residual (`finish=length`) — re-serve would not help |
| Ornith GPQA-Diamond **64k composite** | **80.81%** | — | **15/198 = 7.6%** | ⚠️ budget residual (`finish=length`) — re-serve would not help |
| Ornith IFEval prompt-strict **64k composite** | **88.54%** | — | **11/541 = 2.0%** | ⚠️ budget residual (`finish=length`) — re-serve would not help |
| Laguna IFEval prompt-strict | 75.79% | — | 29/541 = 5.4% | ✅ yes, floor |
| Laguna GSM8K | 83.40% → **96.13%** | 97.09% | 186/1319 = 14.1% | ✅ **corrected** |
| Lightning GSM8K | 95.07% | 96.83% | 24/1319 = 1.8% | negligible |
| Ornith GSM8K | 97.19% | 97.27% | 1/1319 = 0.1% | negligible |
| Lightning GPQA-D @32k | 66.16% | 83.44% | 41/198 = 20.7% | recovered → 76.26% @64k |

> ⚠️ **The "served-only" column is an upper bound, not a fix.** On Laguna the 186 recovered items
> scored **90.3%** versus **97.09%** for the items that returned content — the defect does *not* drop
> questions uniformly at random, so exclusion-based estimates are optimistically biased. Only a
> re-serve produces a defensible number. **Do not publish the served-only figures.**

- [x] **1a. Nemotron-3-Super-120B — re-serve the 56 empty GPQA-Diamond items**
  - Completed 2026-09-09 at the standardized 64k ceiling. Byte-identical replay recovered content
    for 25/56 and 20 new correct answers: **63.64% → 73.74% (146/198)**. The other 31 all reached
    exactly 65,536 completion tokens with `finish_reason=length` and remain counted wrong.
  - The replay had zero HTTP errors but required **12.46 h at concurrency 8**; median completion length
    was the full 65,536-token ceiling. This is a measured operational score, not a no-limit ceiling.
  - Raw replay responses, completion usage, summary, and corrected composite are retained under
    `results/nemotron-3-super-120b/raw/quality/gpqa/replay_64k_2026-09-09/`.

- [x] **1b. Ornith-1.0-35B — re-serve the 42 empty GPQA-Diamond items**
  - Completed 2026-09-06 at the standardized 64k ceiling. Byte-identical replay recovered content
    for 27/42 and 22 new correct answers: **69.70% → 80.81% (160/198)**. 15 items (7.6%) still
    reached 64k without final content and remain counted as wrong; raw replay artifacts are committed.

- [x] **1c. Re-serve the Ornith and Lightning empty IFEval items**
  - Ornith completed 2026-09-06–07. Exact lm-eval scoring changed prompt-strict
    **85.58% → 88.54% (479/541)**; 11 items (2.0%) still reached 64k without final content.
  - Lightning completed 2026-09-08. The 64k replay recovered content for 42/47 parser empties and
    39 prompt-strict successes: **86.14% → 93.35% (505/541)**. Five items (0.9%) reached 64k
    without final content; all are budget residuals. Raw replay and corrected composite artifacts are
    committed under `results/nemotron-3.5-lightning-30b/raw/quality/ifeval/`. Laguna remains open.

- [x] **1d. Add Nemotron-3-Super GPQA and Lightning IFEval to the ISSUES #15 impact table**
  - Done: both rows added to ISSUES.md #15 impact table, distinguishing parser empties
    (`finish=stop`) from Ornith's 64k budget residuals (`finish=length`). Also added
    Nemotron-Super GPQA and Lightning IFEval as open items requiring re-serve.

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

- [ ] **2a-i. Re-run Qwen3.6-35B SWE-bench n=100.** Script ready:
      `viz/run_qwen36_swebench_rerun.sh`. It uses
      `results/qwen3.6-35b-a3b/raw/swebench/swebench_qwen36_rerun_config.yaml`, derived from
      Ornith's committed config with **exactly two** deliberate differences (`model_name`, and
      `pull_timeout: 1800`) — verified by a structural diff, so the robust `git add -A` submit and
      all limits are Ornith's verbatim. Blocks until the endpoint serves the expected model and all
      100 images are present. Cost ~11 h generation + ~20 min grading; needs the GPU.
- [ ] **2a-ii. Record the fair-verdict count (`completed_instances`) next to every SWE-bench score**
      in the top-level README table. A resolve rate over 66 attempts and one over 91 are different
      measurements and the table should say so. Report **both** raw resolves and
      resolves-among-graded; the first is the deployment answer, the second isolates capability.
- [x] **2a-iii. Classify Ornith's missing `exit_statuses_*.yaml` as unrecoverable.** The original
      trajectories were written under `/tmp/ornith_swe_n100` and reaped by macOS on 2026-08-24;
      no run log or exit-status artifact survives locally, remotely, in git, or in stashes. The nine
      empty-patch IDs remain derivable from the committed results JSON, but their per-instance causes
      cannot be reconstructed honestly. A true exit-status file now requires a full re-run.
- [x] **2a-iv. Recover Qwen3.6's agent config.** ~~Neither config nor launch script is committed.~~
      **Done 2026-08-25**: reconstructed from the trajectories' embedded `info.config` and committed
      alongside the run log. Limits confirmed identical to Ornith's; the gap is pull-timeout
      infrastructure, not an unfair budget.
- [x] **2a-v. Guard the image cache instead of pre-pulling it.** **Done 2026-08-25.** The intended
      fix here was to pre-pull the 100 task images. **On inspection that was unnecessary: all 100 are
      already cached** on the Mac mini (verified against the instance set, including all 22 that
      originally failed; cold-start of two of them takes ~0.5 s). They were warmed by the *later*
      Ornith and Lightning runs — which is also why only Qwen3.6, run first on 2026-08-04 against a
      cold cache, shows this failure mode. **The cause was run order, not the model.**

      The real risk is that this cache is an *implicit, unprotected* precondition: a
      `docker system prune`, a Docker Desktop cleanup, or a new machine silently restores the
      failure, and the damage looks like model failures rather than infrastructure. So the fix is a
      verified precondition, not a pre-pull:

      - `viz/swebench_preflight.py` — checks every image for a given instance set, `--pull`s what is
        missing, and exits non-zero otherwise. A fast no-op when warm. Tested on all three paths
        (warm pass, missing→exit 1, genuine cold pull).
      - `pull_timeout: 1800` in the re-run config, up from the mini-swe-agent default of 120 s.
        A single cold pull measured **51 s** on this host, so 120 s was only ~2× headroom on one
        image — far too tight when many pull concurrently. This is the setting that converts an
        infrastructure hiccup into a silent zero.


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

- [x] **2c-i-a. Re-run Nemotron-3-Super GSM8K** on the clean-extract task
      (`results/nemotron-3.5-lightning-30b/raw/gsm8k_cot_zeroshot_clean.yaml`).
  - **Completed 2026-09-09:** answer-line **95.83% (1264/1319)**; flexible fallback
    **96.89% (1278/1319)**, with one empty response counted wrong. Raw samples, aggregate JSON, run
    log, and the campaign manifest are retained under its `raw/` tree.
- [ ] **2c-i-b. Re-run gpt-oss-120b GSM8K** on the same clean-extract task (~3 h).
  - Do not compare its historical stock-task 83.70% directly with the clean-task Nemotron score.
- [x] **2c-ii. Fix the committed `gsm8k_cot_zeroshot_clean.yaml` before re-using it.** Completed 2026-09-08 (commit 4c10dde). Fixed
      `dataset_path: gsm8k` → `openai/gsm8k` and replaced the chained normalize-regex with a single
      anchored regex plus `group_select: -1` and `regexes_to_ignore` for comma/`$`/`.` stripping.
- [x] **2c-iii. Fix the committed `gpqa_clean_task.yaml`** Completed 2026-09-08 (commit 4c10dde). Replaced
      `multi_choice_regex` `flexible-fallback` (raised `KeyError: 'choices'` on lm-eval ≥ 0.4.12)
      with a plain `\(([A-D])\)` regex.

### 2d. Output budget is a first-class variable and it is not held constant

Lightning's measured GPQA-Diamond curve: **16k → 53.03%** (41% truncated), **32k → 66.16%** (21%),
**64k → 76.26%** (3%). A 23-point swing from budget alone.

Current budgets in the repo:

| Model | GPQA budget | Concurrency |
| --- | ---: | ---: |
| gpt-oss-120b | 16,384 | 4 |
| Nemotron-3-Super-120B | 16,384 | 6 |
| Qwen3.6-35B | 16,384 | 8 |
| Ornith-1.0-35B | **65,536 composite** | 4 |
| **Laguna-S-2.1** | **32,768 / 65,536 measured** | **4** |
| Lightning (reported) | 65,536 | — |

- [x] **2d-0. Re-run Laguna-S-2.1 GPQA-Diamond at 64k.** Completed 2026-09-08. Scored **37.88%**
      (75/198) — statistically indistinguishable from the 32k run (40.40%, McNemar χ²=0.41, p=0.52,
      95% CI [−8.7, +3.7]). Items at the ceiling rose from 95→104 when the budget doubled, ruling out
      truncation as the cause; this is non-termination (looping reasoning). See ISSUES.md #17 for the
      full diagnosis. **Raising the budget further is not recommended.** The score (37.88%) is
      published in the card with explicit caveats. *(See also §4c, which was a duplicate of this item.)*

- [x] **2d-i. Standardise on a 64k output ceiling for offline reasoning benchmarks.**
      Adopted 2026-09-08: GPQA-Diamond and IFEval for reasoning models now use
      `max_gen_toks=65536`; GSM8K remains 8192. `RUNBOOK.md` and the Ornith runner encode it.
      The ceiling is not a reservation; report residual `finish_reason=length` rates.
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

- [x] **2e-i. Add a preflight check to the quality runner.** Completed 2026-09-08:
      `viz/quality_preflight.py` fail-closes on served-model identity, usable endpoint output,
      requested generation ceiling, and timeout feasibility with a 2× safety factor. The Make
      interface requires explicit `MODE=live|arithmetic`; `MAX_TOKENS_PROBE` is forwarded explicitly
      for deep reasoners whose final content cannot fit the default canary. Regression tests cover
      the Python gate and Make wrapper.

---

## 3. Throughput methodology

- [x] **3a. Retire the "still climbing at the top of the sweep ⇒ `--max-num-seqs` cap" heuristic.**
      It was **falsified on Laguna**: a +40.8% step at c=128 looked exactly like a cap, but c=192
      added only +10.9% and c=256 only +2.9%. The real peak was ~14% above c=128, not the large
      headroom the slope implied. Also, `SchedulerConfig.max_num_seqs` defaulted to 128 while the
      engine admitted 150–172 concurrently — the documented default is not the live ceiling.
      **Extend the sweep past the knee, or report the top measured point as a measured point.**
      Documentation now uses this rule and no longer attributes a rising endpoint to a cap without
      direct evidence.
- [x] **3b. Re-sweep Lightning past c=128.** Completed 2026-09-08 under one unchanged
      `--max-num-seqs 512` configuration: **836.51 / 852.33 / 926.18 tok/s** at c=192/256/384,
      with 384/384 successful requests at every point. Because c=384 still gains 8.7%, 926.18 tok/s
      is reported as a **short-context measured floor**, not a plateau or hardware ceiling. Median
      TTFT at c=384 is 19.9 s. Raw log and launcher are retained under
      `results/nemotron-3.5-lightning-30b/raw/throughput_sweep_extended/`.
      **Ornith complete 2026-09-08:** 549.38 tok/s at c=192, a measured peak of
      **559.48 tok/s at c=256**, and 557.78 tok/s at c=384. Throughput plateaus while median TTFT
      rises 8.54 s → 30.15 s from c=256→384; use c≈256 for maximum throughput and lower concurrency
      for latency-sensitive service. Raw log and launcher are retained under
      `results/ornith-35b/raw/throughput_sweep_extended/`.
- [ ] **3c. Re-sweep Nemotron-3-Super cleanly.** Its curve was run across a **server restart at two
      different `--max-num-seqs` values** (24 for c=1→24, then 128 for c=32→128), which is why there
      is a c=32 warmup spike. That is not a single clean curve. ~2 h.
- [x] **3d. Never extrapolate a peak from an instantaneous `/metrics` delta.** A 20–30 s sample of
      `vllm:generation_tokens_total` mid-run gave 333.5 tok/s where the completed `vllm bench serve`
      finished at 258.77 — a 29% overstatement, published then retracted. Live metrics are for
      liveness and diagnosis only. This is now an explicit policy in `AGENTS.md` and `RUNBOOK.md`.

---

## 4. Never measured

- [ ] **4a. Qwen3.5-122B-A10B-int4 — the entire quality + agentic suite.** Serving is verified and
      throughput is measured (~228 tok/s at c≈192, 26.9 tok/s single-stream); GSM8K, IFEval,
      GPQA-D, pi-30 and SWE-bench have never been run.
- [ ] **4b. Qwen3.5-122B with MTP / speculative decoding enabled.** This is the model behind the
      Reddit "50 tok/s on DGX Spark" report; we measured 26.9 tok/s single-stream without spec
      decode. That is the lever toward the reported figure and it is untested.
- [x] **4c. Laguna GPQA-Diamond.** *(Duplicate of §2d-0 — see that entry for results.)*
      Both 32k and 64k runs are complete. The 64k run scored 37.88%, statistically indistinguishable
      from the 32k run (40.40%), confirming non-termination rather than truncation. Published with caveats.
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

- [x] **6a. Ornith's missing `exit_statuses_*.yaml` is permanently classified.** The trajectories
      and run log are gone, so the nine known empty-patch IDs cannot be assigned truthful causes.
      The provenance table retains this as a historical gap; future runs must write to `$HOME`,
      retain trajectories, and commit exit statuses. (Same finding as §2a-iii.)
- [x] **6b. Qwen3.6 has no agent config and no launch script.** **Recovered 2026-08-25** — see
      §2a. mini-swe-agent embeds the resolved config in every trajectory, so the effective config
      was reconstructible from the run's own output. The launch script is still absent, but the
      recovered config plus the log's config-spec line make it redundant.

Mid-run configuration changes are already a live problem, not a hypothetical:

- [x] **6c. Lightning's SWE-bench run spans two vLLM builds** — `launch_lightning_swe.sh` pins
      `vllm/vllm-openai:v0.27.1`, `launch_lightning_swe_nightly.sh` pins `cu129-nightly-aarch64`,
      after the engine wedged under long-context load. **Nothing records which instances ran under
      which build.** Repository and session-history searches found no surviving per-instance build
      boundary, so the card now states that the aggregate is segmented and the boundary is
      unrecorded; no IDs were guessed.
- [x] **6d. Fix the misleading segment filenames.** Completed 2026-09-08:
      the 23-entry file is now `exit_statuses_resume_segment_n23.yaml`, the 28-entry final segment is
      `exit_statuses_final_segment_n28.yaml`, and the repeated-mapping historical concatenation is
      explicitly named `exit_statuses_segments_raw_concatenated.yaml`. Consumers and provenance
      guidance point at the truthful names.
- [x] **6j. Reconcile `launch_ornith.sh` with the run it documents.** Completed 2026-09-08:
      the launcher now requires an explicit `throughput` profile (`util=0.90`, `max-num-seqs=512`)
      or `agentic` profile (`util=0.55`, `max-num-seqs=32`). The card identifies which measured runs
      used each profile, eliminating the prior contradictory single command.

Tooling, so the standard is cheaper to follow than to skip:

- [x] **6e. `make manifest MODEL=<m> BENCH=<b>`** — scaffold `manifest.json`, auto-filling engine
      version/args probed from the live endpoint, client host, repo commit, timestamps. Unknowns
      written as the literal `"unrecorded"`, never guessed. **Done 2026-09-03**
      (`viz/manifest_scaffold.py`). Probes `/v1/models` with `--endpoint`, scans the committed launch
      script for image/args/env, refuses to overwrite without `--force`. Deliberately leaves
      `launch_script_matches_run` as `"unrecorded"` — the committed script is not evidence of what ran.
- [x] **6f. `make check-artifacts`** — fail if any `results/*/raw/*/` lacks a required artifact for
      its benchmark. Wire into CI so gaps surface at commit time. **Done 2026-09-03.**
      `viz/audit_provenance.py` previously printed 17 gaps and ended in an unconditional `return 0`;
      it is now **ratcheted** against `viz/data/provenance_baseline.json` (17 accepted, any new gap
      exits 1) and runs in `.github/workflows/provenance.yml` — the repo's first CI. `STRICT=1`
      fails on the whole backlog; clearing it and deleting the baseline is the goal.
- [ ] **6g. Normalize quality artifact paths.** Some models use `raw/quality/<task>/`, others
      `raw/<task>_results.json`. Pick `raw/<benchmark>/` and move the rest.
- [x] **6h. Commit `samples_*.jsonl` for every lm-eval run** (gzipped if size is a concern). It is
      the only artifact that permits after-the-fact detection of the ISSUES #15 defect.
      **Addressed 2026-09-08:** `.gitignore` blanket `*.jsonl` exclusion removed; only the
      explicit `samples_*.jsonl` pattern is retained (for scratch/tmp locations). Future runs must
      use `--log_samples` and commit `samples_*.jsonl.gz` under `results/`. The committed slim
      `*.per_item.csv` files are **supplemental** (2.7–19 KB vs multi-MB JSONL) for CI audit
      coverage on already-committed runs; they do not replace the full JSONL for future runs.
      Per_item.csv preserves auditability for runs whose JSONL already exists — it cannot recover
      the 5 of 7 models whose samples were never kept.
- [ ] **6i. Make `make samples` a hard CI gate.** It exits 1 above 2% empty responses but is
      warn-only in CI because three committed Lightning tasks already breach it (GPQA 41.4%,
      GPQA-32k 20.7%, IFEval 8.7%). Flipping it is the definition of done for the ISSUES #15 re-serve
      backlog.
      *(Note: the label `6i` is used twice in this file -- see also line ~400, "Reconcile
      `launch_ornith.sh`". Left as-is rather than renumbered, since both are referenced elsewhere.)*
- [x] **6j. Preflight the endpoint before launching a quality run.** DONE.
      `viz/preflight_serving.py` sends 3 short probes and refuses to launch when answers are being
      dropped. `make preflight-serving` (live) / `make preflight-selftest` (fixtures, in CI).

      Exit 0 = usable, **1 = defect, 2 = could not probe**. Exit 2 is deliberately not success: an
      unprobeable endpoint must never read as a pass.

      **Empty content alone does NOT mean a broken parser**, and assuming so would block healthy
      endpoints. Measured on live warpcore 2026-09-03 (`RedHatAI/Muse-Glimmer-30B-NVFP4`): the same
      healthy model returns `content=None, finish=length` at `max_tokens=64` and `content='4',
      finish=stop` at 512 -- it needs ~109 completion tokens before emitting any answer. So the
      classifier keys on `finish_reason`:

      | signature | verdict |
      | --- | --- |
      | `finish=length` + empty | BUDGET -- raise `max_gen_toks`, parser is fine |
      | `finish=stop` + empty + `reasoning` populated | PARSER -- ISSUES #15 |
      | `finish=stop` + empty + nothing anywhere | EMPTY -- nothing generated |

      Checks both `reasoning` and `reasoning_content`; vLLM emits the former, so a probe checking
      only the conventional field sees nothing and wrongly concludes the output vanished.

      Verified: self-test 6/6 branches; live endpoint exit 0; unreachable exit 2; live endpoint at
      `--max-tokens 64` exit 1 diagnosed BUDGET; `tests/fake_broken_endpoint.py` serving a real
      ISSUES #15 payload over HTTP exit 1 diagnosed PARSER. Mutation test (`if finish == "length"`
      -> `if False`) flips `budget_starved` to `parser` and fails the self-test, proving the guard
      is load-bearing.

      Still open: call this from the quality runner so it *cannot* be skipped (P1 #5).

---

## 7. Follow-on study — long-context quality and serving concurrency

Do this **after the current cross-model quality/throughput campaign, but before expanding to
AutomationBench or another benchmark family**. It does not block finishing the current comparison.
The existing 512-input/256-output sweeps remain useful, but must be labelled as short-context
saturation measurements rather than deployable long-context operating points.

- [ ] **7a. Record the context-capacity profile for every deployment.** Capture the model-declared
      context limit, configured `--max-model-len`, measured `kv_cache_size_tokens`, vLLM's reported
      full-window concurrency, `--max-num-seqs`, KV dtype, memory utilization, and prefix-caching state.
- [ ] **7b. Add controlled long-context serving sweeps.** Test actual total sequence lengths of 32K,
      64K, 128K, and 256K where supported. At each length, sweep concurrency below, near, and above
      the predicted KV-capacity boundary. Report completed-harness prefill/output throughput, TTFT,
      TPOT/ITL, end-to-end latency, running/waiting requests, incremental preemptions, failures, and
      latency-constrained goodput. Do not infer throughput from instantaneous `/metrics` deltas.
- [ ] **7c. Measure effective context, not merely accepted context.** Start with RULER at matched
      lengths because its deterministic retrieval, multi-hop, aggregation, and variable-tracking
      tasks isolate context-length degradation. Follow with a small HELMET or LongBench v2 subset for
      external validity; do not start with the full, heterogeneous suite.
- [ ] **7d. Isolate server-limit effects with paired deployments.** For selected models, compare
      `--max-model-len=131072`, `262144`, and the largest feasible model-supported value using
      identical overlapping request lengths and concurrency points. Changing only the server ceiling
      while changing the workload would confound the result. Record restart configuration and rerun
      serving/output-budget preflight after every profile change.
- [ ] **7e. Add a realistic long-context coding workload only after 7b/7c.** Use a fixed repository
      snapshot, tasks requiring evidence from distant files, and deterministic tests. Keep quality and
      systems results separate: score all attempted tasks, and also report correct tasks/hour.
- [ ] **7f. Account for verbose reasoning explicitly.** Retain prompt tokens, reasoning tokens,
      visible-answer tokens, total sequence tokens, finish reason, correctness, and latency. Evaluate
      prefill-heavy, decode-heavy, and mixed agentic workloads separately. Standardize the maximum
      allowed budget rather than forcing models to emit equal reasoning lengths; report tokens per
      correct answer and correct answers/hour.

Target summary metrics per model: **effective context** (longest tested length meeting a declared
RULER quality threshold) and **long-context goodput** at 128K/256K under declared latency,
preemption, and error constraints. These supplement—not replace—the short-context throughput ceiling.

---

## Suggested order

Ranked by information gained per GPU-hour, from current state (2026-09-09).
Completed items (§0, §1a/§3c/§2c-i-a Nemotron-3-Super campaign, §2d-0/§4c, §2c-ii/iii, §1d, §6h)
are done and excluded. Ornith throughput is cross-referenced rather than duplicated (see §3b). The
Lightning IFEval replay in §1c is complete.

| Order | Item | Cost | Unblocks |
| --: | --- | --- | --- |
| 1 | §2d-ii Qwen3.6-35B GPQA thinking mode @64k | ~8 h | Qwen3.6's real reasoning ceiling |
| 2 | §2a-i Qwen3.6 SWE-bench re-run | ~11 h | the most misleading number in the table |
| 3 | §2b-i gpt-oss SWE-bench re-run | ~11 h | the only missing agentic score |
| 4 | §2c-i gpt-oss GSM8K clean-task re-run | ~3 h | GSM8K column comparability |
| 5 | §4a Qwen3.5-122B full suite (GSM8K, IFEval, GPQA-D, SWE-bench) | ~20 h | the one model with no quality data |

**Nemotron-3-Super consolidated campaign completed 2026-09-09.** Corrected GPQA is **73.74%**,
clean answer-line GSM8K is **95.83%**, and the current-profile short-context sweep reached
**244.70 output tok/s at c=128**. Because throughput was still rising at the final point, that value
is a measured floor, not a proven ceiling.

**Ornith and Lightning short-context extensions are complete.** Ornith plateaus at ~559 tok/s around
c=256. Lightning reaches 926.18 tok/s at c=384, but the final point still rises 8.7%, so it remains a
measured floor rather than a demonstrated hardware ceiling.

**Definition of done for "systematic":** every model in the top-level README table measured with the
same task configs, the same output budget, a concurrency derived from its own throughput sweep, a
verified-zero (or explicitly reported) empty-response rate, and — for SWE-bench — the same scaffold,
the same seed-42 instance set, and the fair-verdict count published next to the resolve rate.
Every run additionally carries a `manifest.json` conforming to [PROVENANCE.md](PROVENANCE.md), so
the settings behind any number can be recovered from the repo without asking a person.

