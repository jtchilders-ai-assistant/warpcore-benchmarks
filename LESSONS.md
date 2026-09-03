# Lessons Learned → What to Systematize

Retrospective across 7 benchmarking sessions (~940 messages, 2026-08-21 → 2026-08-31) covering
gpt-oss-120b, Nemotron-3-Super, Nemotron-3.5-Lightning, Qwen3.6, Qwen3.5-122B, Ornith-1.0, Laguna-S-2.1,
Muse-Glimmer-30B. 70 distinct findings were extracted and clustered into the 7 failure modes below.

**The governing observation:** this repo's *documentation* of methodology is excellent —
`PROVENANCE.md`, `ISSUES.md` (17 entries) and the README footnotes are more rigorous than most
published model cards. What is missing is **enforcement**. Every rule is prose that a human must
remember. Every defect below was caught by a human noticing something, late, after GPU-hours were
already spent. The work is to convert prose rules into exit codes.

Verified state of enforcement as of 2026-09-03:

| Claimed control | Reality |
| --- | --- |
| `PROVENANCE.md` §4 says run `make manifest` / `make check-artifacts` | ~~Neither target exists~~ → **both implemented 2026-09-03** (§4 did flag them as not-yet-existing) |
| `viz/audit_provenance.py` docstring: "Exits nonzero if a REPORTED number lacks its artifact" | ~~Ends in unconditional `return 0`~~ → **fixed**: ratcheted, exits 1 on any new gap |
| CI | ~~`.github/workflows/` does not exist~~ → **added** `.github/workflows/provenance.yml` |
| `make check` (figures) | **Works** — exit 0, regenerated figures match committed. This is the one real gate |

---

## 1. Silent scoring failures — the highest-severity class

A model can return HTTP 200, `finish_reason: "stop"`, billed completion tokens, and an **empty
answer** that scores zero. No error, no retry, no exception. Root cause is ISSUES #15: with
`--reasoning-parser`, vLLM may fail to initialize reasoning delimiters and emit `content: null`
with the answer in `message.reasoning`; lm-eval reads only `content`.

Measured empty rates: Nemotron-3-Super GPQA-D **28.3%**, Ornith GPQA-D **21.2%**,
Laguna GSM8K **14.1%**, Lightning IFEval **8.7%**, Laguna IFEval 5.4%, Ornith IFEval 5.2%.
Laguna GSM8K corrected 83.40% → **96.13%** once re-served. That is a 12.7-point error in a
published number.

Two compounding traps:
- **The startup warning was already in the log.** `Auto-initialization of reasoning token IDs
  failed` was printed before the benchmarks ran and was not acted on.
- **Excluding empty items is not a fix.** On Laguna, recovered items scored 90.3% vs 97.09% for
  served items — the defect is *not* missing-at-random, so served-only rates are optimistically
  biased upper bounds, not scores.

**Systematize**
- `scripts/preflight_serving.py` — after server ready, grep the vLLM log for the
  auto-init failure string, then send a 10-item probe and assert `content` non-null rate ≥ 99%.
  Refuse to start the suite otherwise. Cost: ~30 s against 13 h of wasted GPU time.
- `scripts/validate_samples.py` — post-run, compute `empty_content_rate` per task; exit 1
  above 2%. Wire into CI so no sample file can be merged with a silent-zero defect.
- Make `LM_EVAL_REASONING_FALLBACK=1` the **default in the runner**, not an env var someone must
  remember. It currently appears only in Laguna's two scripts.

## 2. Primary evidence is being destroyed

`samples_*.jsonl` is in `.gitignore` (line 5). Result: only **2 of 7 models** have any sample file
committed (laguna 2, lightning 4, everyone else 0). The empty-response defect above is therefore
**permanently unauditable for 5 of 7 models** — those scores can never be verified or corrected.

Worse, the repo-wide auditor `audit_empty_responses.py` globs `/tmp/lmeval_results/**`, and `/tmp`
on this Mac is reaped. The one tool designed to check every model points at a directory that no
longer exists. Same for 8 other committed analysis scripts (`gpqa_64k_replay.py` → `/tmp/trunc_prompts.json`,
`traj_triage.py` → `/tmp/laguna_swe_smoke/`, etc.).

**Systematize**
- Commit a slim `per_item.csv` per task — `item_id, correct, finish_reason, completion_tokens,
  empty_content, at_cap` — ~50 KB instead of 5 MB. Preserves every audit-relevant field, dodges the
  file-size objection that motivated the gitignore.
- Analysis scripts take the artifact path as **argv**, defaulting to the committed repo path. Never
  a hardcoded `/tmp`. Staging goes under `$HOME`, which is not reaped.

## 3. Denominators are not published, and they reorder the ranking

SWE-bench headline rates conflate *completion rate* with *patch quality*. Computed from the
committed `*results.json`:

| Model | Headline | Fair verdicts | resolved \| graded |
| --- | ---: | ---: | ---: |
| ornith-35b | 73/100 | 91 | 80.2% |
| laguna-s-2.1-118b | 55/100 | 65 | **84.6%** |
| nemotron-3.5-lightning | 51/100 | 98 | 52.0% |
| qwen3.6-35b-a3b | 44/100 | 66 | 66.7% |

**Ranking by headline: Ornith > Laguna. Ranking by resolved-given-graded: Laguna > Ornith.** The
README publishes only the first. TODO 2a-ii (publish `completed_instances`) is still open. Neither
column is "the" right one — but publishing one without the other hides that the ordering is a
choice.

Non-completions are also heterogeneous and must not be pooled: Qwen3.6's 22 non-completions were
**Docker image-pull timeouts** (infrastructure), Laguna's were 23% `RepeatedFormatError`
(model burns budget in the reasoning field, never emits a tool call), gpt-oss's 79/100 were a
`--tool-call-parser openai` JSON-corruption bug. Only the middle one is a capability signal.

**Systematize** — `scripts/swebench_report.py` emitting `submitted / completed / resolved`,
`resolved|graded`, and an exit-status histogram per model; README generated from it, never hand-typed.

## 4. Task-config and serving-config drift inside a single comparison table

Verified by md5 of committed task YAMLs — **two distinct GPQA configs and two distinct GSM8K
configs are feeding the same README columns**:

- GPQA `77ba1295…` (uses `multi_choice_regex`): gpt-oss, Nemotron-3-Super, Qwen3.6
- GPQA `89a722b5…` (plain `\(([A-D])\)`): Laguna, Lightning
- GSM8K `ad4d9e5b…` (still `dataset_path: gsm8k`, TODO 2c-ii open): Qwen3.6
- GSM8K `f51e8434…` (fixed `dataset_path: openai/gsm8k`): Lightning

Serving config drifts too: Ornith's committed `launch_ornith.sh` pins
`--gpu-memory-utilization 0.90`, but the SWE-bench run that produced 73/100 used **0.55**. The
committed script does not reproduce the published number.

Root cause is visible in the source: `run_laguna_quality.sh` and `run_ornith_quality.sh` are
**hand-edited copies of each other**, differing only in model name, paths and comments, with
per-task `(concurrency, max_gen_toks)` as inline literals. Copy-paste is the drift mechanism.

**Systematize**
- One parameterized `scripts/run_quality.sh --model-config configs/<model>.yaml`. Per-model YAML
  holds model id, endpoint, quant, `gpu_memory_utilization`, and per-task budgets. Delete the clones.
- Hash every task YAML and the serving config into the run manifest. CI asserts all models in a
  comparison column share one task-config hash, else the cell is marked non-comparable.

## 5. Throughput numbers that measure the harness, not the model

- **Contamination:** a concurrent lm-eval smoke test polluted a sweep; c=8 read 42.18 vs c=4 40.99
  tok/s and looked saturated. ISSUES #6 is the same failure recurring.
- **Instantaneous sampling:** ≥333 tok/s was reported off a mid-run metric delta; sustained was
  **258.77**.
- **Premature stop:** sweeps stopping at c=128 (vLLM's `max_num_seqs` default) were labeled
  "saturated" while still climbing **+40.8%**. The heuristic "still climbing ⇒ max_num_seqs cap"
  was later **falsified** — the real ceiling was KV-cache saturation, and c=256 admitted *fewer*
  concurrent requests than c=192. Lightning's 719 and Ornith's 464 are still labeled floors on
  the falsified heuristic.
- **Unrepresentative shape:** all sweeps use 512-in/256-out synthetic requests. These are not
  deployable operating points for a 32k-context reasoning workload.
- Only **1 of 7** models has a parsed `throughput_sweep.csv`; the rest are unstructured logs.
- Aggregate tok/s vs per-stream tok/s **inverts the model ranking** — the choice must be stated.

**Systematize** — a sweep harness that (a) polls `vllm:num_requests_running`/`_waiting` to 0 before
each level and refuses to start otherwise, (b) reads only the final `Output token throughput` line,
(c) auto-extends the range while the last delta > 10%, (d) records `request_input_tokens`/
`request_output_tokens` and a `peak_type` enum (`confirmed_plateau` | `max_num_seqs_floor` |
`unknown`), (e) always emits the normalized CSV. Add a `trap` that reaps orphans via
`docker exec pkill` — a root-owned orphan inside the container survived `tmux kill-session` and
silently invalidated a sweep.

## 6. Budgets and timeouts derived by guess instead of arithmetic

A GPQA run was killed at 110/198 **after 13 hours** because the client timeout was 3600 s while
32k tokens at ~4 tok/s needs ~2 h per item. A SWE-bench smoke test inherited a 1800 s litellm
timeout that was 2× too small. Separately, 46% of GPQA generations hit `finish_reason=length`, so
exhausting the cap was routine, not exceptional — the timeout cascade was predictable from a
10-item probe.

Output budget is itself a confound: Lightning's GPQA went **53.03% @16k → 76.26% @64k**. A 16k
number measures the budget, not the model. But budget is not always the answer — Laguna's 32k vs
64k runs were statistically indistinguishable (−2.53 pp, McNemar p=0.52) because its failure mode is
**non-termination**, not truncation. Distinguishing the two requires the token census the repo
already built (`compare_gpqa_budgets.py`).

Also: character length is **not** a proxy for token count. An interim analysis used one
5.25 chars/token ratio when the true range was 2.05–3.98, inflating the assumed ceiling ~70% and
producing a wrong "0 items at the ceiling" claim.

**Systematize** — `scripts/probe_budget.py --n 10` before every long run: measure the
`finish_reason` distribution and per-request tok/s at the planned concurrency, then assert
`configured_timeout >= max_tokens / measured_tok_s * 2.0` and refuse to launch otherwise. Always
tokenize with the model's own tokenizer; never estimate tokens from characters.

## 7. Checkpoint and infrastructure trust

Findings from the Muse-Glimmer and Laguna bring-ups that belong in a pre-flight gate:

- A quantized checkpoint **silently missing its vision tower** produced garbage image embeddings
  that looked like a model capability failure. Correcting it required diffing safetensors manifests
  across ~60 repos. The wrong hypothesis ("synthetic-image artifact") survived until the weights
  were swapped.
- Incoherent metadata is a cheap red flag: **6.03B params reported for a 30B model**.
- A missing `quant_method` key caused silent `quantization=None` at load; scale-tensor name
  mismatch (`weight_scale_inv` vs `weight_scale`) caused a hard load failure.
- Vendor model card said ~71 GB; actual was **92.9 GiB** (+30%), which disabled auto-prefetch and
  made loading disk-bound at ~37 s/shard.
- GB10 is **sm_121**; shipped FlashInfer cubins cover sm_80/90/100/120 only — crash appears
  **40 minutes in**, at first decode, after the full weight load.
- **A documented CLI default is not the observed runtime behavior.** A `max_num_batched_tokens`
  default change was asserted from release-note prose; reading `vllm/config/scheduler.py` in both
  images showed `DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048` unchanged. Pinning 8192 "for parity" was a
  4× raise that cost **35% of KV cache** (333,604 → 216,350 tokens). Verify version claims by
  reading the source tree in both images, never the changelog.
- A reasoning-parser smoke test checked `message.reasoning_content`; vLLM 0.28.0 uses
  `message.reasoning`. A working parser was reported broken. Always read
  `m.get("reasoning") or m.get("reasoning_content") or ""`.
- `/v1/models` returns 200 before the engine can actually serve. Readiness must be a real
  completion request.
- The inference endpoint was **unauthenticated and publicly routable**, and had already been
  scanned by ANL IT.

**Systematize** — `scripts/preflight_checkpoint.py <hf-id>`: assert declared params vs safetensors
sum, `quant_method` present, expected module groups present (vision tower when applicable), scale
tensor naming, measured on-disk size vs model card, and local arch vs `torch.cuda.get_arch_list()`.
All of it runs before a 40-minute weight load, not after.

---

## Priority

**P0 — cheap, prevents wrong published numbers** ✅ **IMPLEMENTED 2026-09-03**
1. ~~Make `viz/audit_provenance.py` exit nonzero on gaps.~~ Done — ratcheted against
   `viz/data/provenance_baseline.json`: the 17 known gaps are accepted, any **new** gap exits 1.
   Verified by hiding a real artifact (`ornith` preds) → exit 1 naming it; restoring → exit 0.
2. ~~Add CI.~~ Done — `.github/workflows/provenance.yml` runs `make check` + `check-artifacts` +
   the sample validator.
3. ~~Add the missing `make manifest` / `make check-artifacts` targets.~~ Done, plus
   `make samples` and `make ci`.
4. ~~Commit slim `per_item.csv`.~~ Done — 6 files, 2.7–19 KB each vs multi-MB JSONL.

**P1 — prevents the next run from being wasted**
5. `preflight_serving.py` (reasoning-parser probe + real-completion readiness).
6. `probe_budget.py` (timeout arithmetic from measured tok/s).
7. Single parameterized quality runner + per-model config YAML; delete the copy-paste clones.

**P2 — makes the comparison table defensible**
8. `swebench_report.py` with completion-rate columns and exit-status histograms.
9. Sweep harness with idle-gate, auto-extend, and `peak_type`.
10. Task-config hash equality check per README column.

## Open corrections these findings imply

Tracked properly in `TODO.md`; listed here because they are consequences of the above, not new work.

- Nemotron-3-Super GPQA-D 63.64% — 28.3% empty, **not a capability measurement**. Re-serve.
- Ornith GPQA-D 69.70% — 21.2% empty. Re-serve (served-only 88.46% is an upper bound, not a fix).
- Lightning IFEval — 8.7% empty, not yet recorded in ISSUES.md.
- gpt-oss & Nemotron-3-Super GSM8K — stock flexible-extract task, not the clean task the others use.
- gpt-oss SWE-bench — no valid score exists (79/100 aborted on the parser bug).
- Qwen3.6 thinking-mode GPQA 33.84% — truncation artifact at 16k.
- Ornith/Lightning throughput peaks — labeled floors on a heuristic that was later falsified.
- Laguna README peak cell says `~259 (c≈192)`; its own committed artifacts say **266.3 @ c=256**.

## Verified non-findings

Two candidate criticisms were checked and are **wrong** — the repo already handles both correctly,
and they are recorded here so they are not "fixed" later:

- *"n=100 CIs are stated as ±5%."* False. The README states **±9 pp** Wilson (±8 pp with FPC).
  Independently recomputed: Ornith 73/100 → 63.6–80.7, i.e. ±8.6 pp. Correct as published.
- *"The Lightning-vs-Qwen3.6 +7 pp gap is stated as a win."* False. The README explicitly says the
  two are "not statistically distinguishable", gives CI [−3, +17], and reports McNemar χ²=1.2,
  p≈0.27.
