# Artifact & provenance standard

What every (model × benchmark) run must leave behind for its number to be quotable in the
top-level README, and how to record a configuration that changes while a run is in flight.

This exists because several published numbers in this repo cannot currently be reproduced or
audited from the committed artifacts alone. The gaps are catalogued in [§5](#5-current-coverage)
and tracked in [TODO.md](TODO.md).

**Rule of thumb:** a number is quotable when someone with this repo, the model weights, and the
hardware could re-derive it without asking us a question. If answering "what settings produced
this?" requires a person, the run is not reproducible yet.

---

## 1. The run manifest

Every run writes exactly one `manifest.json` next to its results, at
`results/<model>/raw/<benchmark>/manifest.json`.

It is the single source of truth for *how* a number was produced. Scores live in the harness
output; the manifest records everything the harness does not.

```json
{
  "schema_version": 1,
  "model": {
    "id": "ornith-ai/Ornith-1.0-35B-FP8",
    "revision": "<HF commit sha, or 'unrecorded'>",
    "quantization": "compressed-tensors W8A8 FP8"
  },
  "benchmark": {
    "name": "swebench-verified",
    "sample": "n=100, --shuffle, seed 42, --slice 0:100",
    "harness": "mini-swe-agent",
    "harness_version": "2.4.6",
    "grading_harness": "swebench 4.0.5",
    "run_id": "ornith_n100"
  },
  "serving": {
    "engine": "vLLM",
    "version": "0.27.2rc1.dev193",
    "image": "vllm/vllm-openai:cu129-nightly-aarch64",
    "image_digest": "sha256:...",
    "host": "warpcore (DGX Spark GB10, aarch64)",
    "launch_script": "../launch_ornith.sh",
    "args": ["--max-model-len 262144", "--moe-backend marlin", "--gpu-memory-utilization 0.55",
             "--enable-prefix-caching", "--tool-call-parser qwen3_xml", "--reasoning-parser qwen3"],
    "env": {"VLLM_TEST_FORCE_FP8_MARLIN": "1", "VLLM_MARLIN_USE_ATOMIC_ADD": "1"},
    "launch_script_matches_run": false,
    "launch_script_note": "committed launch_ornith.sh pins 0.90; the SWE-bench run used 0.55 for host headroom. Script never updated — see §5.",
    "generation_config": "auto",
    "checkpoint_max_new_tokens": "absent"
  },
  "client": {
    "host": "csi0359637 (Mac mini, x86_64)",
    "concurrency": 4,
    "agent_config": "./swebench_ornith_config.yaml",
    "limits": {"step_limit": 250, "cost_limit": 3.0, "per_step_timeout_s": 1800}
  },
  "timing": {"started": "2026-08-19T13:51:00-05:00", "finished": "2026-08-20T00:57:00-05:00"},
  "segments": [],
  "repo_commit": "<git rev-parse HEAD at run time>",
  "notes": ""
}
```

Generate it with `make manifest`, which fills in what can be probed automatically (see §4).
Anything genuinely unknown is written as the literal string `"unrecorded"` — **never guessed, never
omitted.** A missing key is ambiguous; `"unrecorded"` is a fact.

---

## 2. Required artifacts per benchmark

Beyond the manifest. Paths are relative to `results/<model>/raw/<benchmark>/`.

### All benchmarks

| Artifact | Why |
| --- | --- |
| `manifest.json` | §1 |
| launch script (or a copy) | The serving config is half the measurement |

### SWE-bench

| Artifact | Why |
| --- | --- |
| `*_report.json` / `*_results.json` | Scores **and** `completed_instances`, the denominator |
| `preds*.json` | The actual patches — lets anyone re-grade or audit an empty patch |
| `exit_statuses*.yaml` | **Distinguishes "wrong" from "never finished."** Non-negotiable |
| agent config `.yaml` | Step/cost/timeout limits decide the completion rate |
| run script `.sh` | The invocation, including slice and seed |

Without `exit_statuses`, a 44 and a 73 are not comparable and the difference cannot be
attributed. This is the single most valuable artifact and the one most often missing.

### lm-eval quality (GSM8K, IFEval, GPQA)

| Artifact | Why |
| --- | --- |
| `results_*.json` | lm-eval already records `config.model_args`, `lm_eval_version`, `n-samples`, `task_hashes`, `chat_template_sha`, `total_evaluation_time_seconds` — keep it **whole**, do not summarize |
| `samples_*.jsonl` | Required to audit empty/truncated generations (this is how ISSUES #15 was found) |
| custom task `.yaml` + any `utils.py` | A "GSM8K score" means nothing without the extraction filter |

`samples_*.jsonl` is large but it is the only way to detect the `message.reasoning` defect after
the fact. If size is a problem, commit it gzipped rather than dropping it.

**Record the effective output ceiling.** A quality score is only meaningful against a known token
budget, and the budget is not fully described by `max_gen_toks`. With `--generation-config auto`
(vLLM's default) the checkpoint's own `generation_config.json` is loaded at startup, and a
`max_new_tokens` there becomes a **server-wide cap on every request** — it clamps silently, returns no
error, and appears nowhere in the launch command or the startup banner. Every lm-eval manifest must
therefore carry `serving.generation_config` and `serving.checkpoint_max_new_tokens` (`"absent"` when
the checkpoint sets none). Verify with `viz/check_output_budget.py` before the run; see §5c.

### Throughput

| Artifact | Why |
| --- | --- |
| sweep log + sweep script | Concurrency ladder, input/output lengths, `--ignore-eos` |
| Note whether anything else shared the GPU | A contaminated sweep already had to be discarded once ([ISSUES #6](ISSUES.md)) |

### pi-30

| Artifact | Why |
| --- | --- |
| `RESULTS.txt` / `SUMMARY.txt` + runner | Per-problem pass/fail, timeout setting |

---

## 3. When settings change mid-run

This is not hypothetical. **Lightning's 51/100 spans two different vLLM builds** —
`launch_lightning_swe.sh` uses `vllm/vllm-openai:v0.27.1`, `launch_lightning_swe_nightly.sh` uses
`cu129-nightly-aarch64` — because the engine wedged and the run was restarted on a nightly. Nothing
in the committed artifacts says which instances ran under which build.

A mid-run change is legitimate; an *unrecorded* one is not.

### Rules

1. **Never edit a manifest in place.** Append to `segments[]`:

```json
"segments": [
  {"seq": 1, "started": "...", "ended": "...", "instances": ["django__django-11551", "..."],
   "serving": {"version": "0.27.1", "image": "vllm/vllm-openai:v0.27.1"},
   "reason": "initial run"},
  {"seq": 2, "started": "...", "ended": "...", "instances": ["sympy__sympy-20916", "..."],
   "serving": {"version": "0.27.2rc1.dev193", "image": "vllm/vllm-openai:cu129-nightly-aarch64"},
   "reason": "engine wedge under long-context load; restarted on nightly (ISSUES #11)"}
]
```

2. **Segments list instance IDs, not just counts.** "23 of 100 were re-run" is not auditable;
   the ID list is. Name segmented exit-status files for their segment
   (`exit_statuses_seg2_nightly.yaml`), and make the count in the filename match the contents —
   `exit_statuses_n55.yaml` currently contains 23 instances, which is worse than no name at all.

3. **A changed setting makes the run *segmented*, not *invalid*.** Report the score with a
   footnote naming the segments. Only a change that plausibly alters the score — engine version,
   quantization, step/cost/timeout limits, context length, sampling params — needs a segment.
   Restarting the same config after a crash does not.

4. **State the direction of the effect** if you can bound it, or say you can't. "Re-ran 11
   wedged instances on a fresh endpoint, recovering 4 resolves" is useful. "Some instances were
   re-run" is not.

5. **If a benchmark cannot be completed under one config, that is a result.** Record it as a
   partial with the completion count, rather than stitching a clean-looking number out of pieces.

---

## 4. Making this cheap

The standard only survives if it is easier to follow than to skip.

- `make manifest MODEL=<m> BENCH=<b>` — scaffolds `manifest.json`, auto-filling engine version and
  args probed from the live endpoint (`/v1/models`, container labels), client host, repo commit,
  and timestamps. Prompts only for what it cannot see.
- `make check-artifacts` — fails if any `results/*/raw/*/` is missing a required artifact for its
  benchmark. Runs in CI so gaps surface at commit time, not review time.
- Wrap the launch script so it writes its own resolved args into the manifest. The launch script
  and the manifest disagreeing is a bug the tooling should catch.

Neither target exists yet; both are tracked in TODO.md §6.

---

## 5. Current coverage

Audited from the committed tree by [`viz/audit_provenance.py`](viz/audit_provenance.py) — run it
rather than trusting this table to stay current. `-` = missing, `↺` = recovered by reconstruction
(see §5a).

| Model | manifest | report | preds | exit_statuses | trajectories | agent cfg | run script | launch |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| laguna-s-2.1-118b | ✅ | ✅ | ✅ | ↺ | ✅ | ✅ | ✅ | ✅ |
| ornith-35b | **–** | ✅ | ✅ | **–** | **✗ lost** | ✅ | ✅ | ✅ |
| nemotron-3.5-lightning-30b | **–** | ✅ | ✅ | ✅ | **✗ lost** | ✅ | **–** | ✅ |
| qwen3.6-35b-a3b | **–** | ✅ | ✅ | ✅ | **✗ lost** | ↺ | **–** | **–** |
| gpt-oss-120b | **–** | *n/a, blocked* | *n/a* | ✅ | **✗ lost** | **–** | **–** | **–** |
| nemotron-3-super-120b | **–** | *no SWE-bench run* | | | | | | **–** |
| qwen3.5-122b-a10b | **–** | *no SWE-bench run* | | | | | | **–** |

gpt-oss is published as **blocked / no score** (a vLLM tool-call bug, see `raw/DIAGNOSIS.json`), so
its absent preds and report are correct behaviour, not a gap.

**Laguna is currently the only model meeting this standard in full.** The rest predate it. Three
findings from the 2026-08-26 audit:

- **Trajectories for every earlier model are gone.** They were left in `/tmp` and macOS's reaper
  deleted the file contents (Ornith's 100 instance directories still exist, empty, dated 2026-08-24).
  This is unrecoverable: the §5a reconstruction trick that saved Qwen3.6's config only worked
  *because* its trajectories still existed. Laguna's are committed as a 6.0 MB
  `trajectories.tar.gz` — that is now the rule, not an optimization. **Never leave the primary
  evidence in `/tmp`.**
- **Ornith still has no `exit_statuses`** — the best score in the repo, and its 9 non-submissions
  cannot be audited from the repo alone. Its trajectories are now gone, so unlike Laguna's this
  cannot be regenerated. It is permanently a partially-unauditable number.
- **No model except Laguna has a manifest.** Serving engine version, image digest, and resolved
  runtime args are unrecorded for every earlier run.

**A harness-written `exit_statuses` file can itself be incomplete.** Laguna's is marked `↺` because
mini-swe-agent writes it periodically: the on-disk copy captured 96 of 100 instances, missing the
last four to finish. It was regenerated from the trajectories and cross-checked against the graded
report (65 submitted / 35 empty, reconciling exactly). Prefer deriving this file from trajectories
over trusting the harness's snapshot.

**A committed launch script is not automatically the config that ran.** `launch_ornith.sh` pins
`--gpu-memory-utilization 0.90`, but the Ornith card states the agentic runs used **0.55** for host
headroom (pi-30's agent processes share the box) — the script was simply never updated. The card
itself carries both values in different sections. So the repo contains a launch script that does
not reproduce the run it is filed under, and nothing flags the discrepancy.

Laguna's manifest sets `launch_script_matches_run: verified` on the strength of a **two-way** check:
the committed script is byte-identical to the copy on the serving host, *and* its flags match the
args read back from the live container via `docker inspect`. Either check alone would have missed
the Ornith failure mode.

This is the strongest argument for the manifest: a script records an *intention* at the time it was
written, while a manifest records what was *resolved at run time*. When they disagree, the manifest
should say so explicitly (`launch_script_matches_run: false`) rather than leaving a reader to find
the contradiction — or, more likely, not find it.

### 5a. Recovering provenance after the fact

A missing config is not always fatal. **mini-swe-agent ≥ 2.4.6 embeds the fully-resolved config in
`info.config` of every `*.traj.json`**, so the effective configuration of a SWE-bench run can be
reconstructed from its own output — and that reconstruction is *better evidence than a config file*,
because it records what actually executed rather than what someone meant to run.

This was used to recover Qwen3.6's config (§2a of TODO.md) after it turned out never to have been
committed. `viz/reconstruct_qwen_config.py` reads every trajectory, asserts the config is identical
across all of them (ignoring the per-instance container image), and refuses to emit anything if it
varies — a non-uniform run is itself a finding, and silently emitting the first config would hide it.

Two rules follow:

1. **Always retain trajectories and the run log**, even when a run is superseded. They are the
   fallback provenance, and the run log is where harness-level failures — the ones that never reach
   a trajectory — are recorded. **Retaining means committing them, not leaving them on disk.** This
   rule was already written down and still failed in practice: the trajectories for Ornith,
   Lightning, Qwen3.6 and gpt-oss were left in `/tmp` and reaped by the OS (§5). The Qwen3.6 recovery
   above succeeded only because it happened *before* the reaper ran. Compressed, a 100-instance run
   is ~6 MB — there is no space argument for leaving it outside the repo.
2. **Mark reconstructed artifacts as reconstructed**, in the filename and in a header comment
   stating what they were derived from. A recovered config must never be mistaken for a
   contemporaneous one.

The 22 lost Qwen3.6 instances show why the log matters as much as the trajectories: those instances
have **no trajectory at all** — the container never started — so the log is the *only* artifact that
explains them. Absence of a trajectory is itself data, and a coverage check (§6f) should treat a
gap between "instances attempted" and "trajectories present" as a finding rather than an accident.

### 5b. Verify preconditions before a run, not after

The Qwen3.6 failure was invisible at scoring time: 22 instances came back as zeros that looked
exactly like model failures. Nothing in the report distinguished "the model tried and failed" from
"the model was never invoked". That is the expensive kind of bug — it does not announce itself, it
just quietly biases a number that then gets published.

The generalisable rule: **anything a run silently depends on must be checked before the run starts
and recorded in the manifest.** For SWE-bench on this host that means:

- **Container images present.** `viz/swebench_preflight.py` verifies every image in the instance set
  and refuses to launch if any is missing. Cheap insurance: a measured cold pull took **51 s** for a
  single image on this host, so the mini-swe-agent default `pull_timeout` of 120 s has almost no
  headroom once several pull concurrently.
- **The endpoint serves the model you think it does.** `run_qwen36_swebench_rerun.sh` reads
  `/v1/models` and aborts on mismatch. This is not hypothetical — during testing it correctly
  refused to launch because the endpoint was still serving Laguna.
- **Config equivalence to the comparison baseline.** The re-run config is derived from Ornith's with
  a structural diff asserting exactly two intended changes, so scaffold drift cannot quietly
  re-enter a comparison that is supposed to isolate the model.

A precondition that is merely *usually true* — like a warm image cache — is a latent failure. Either
verify it at launch or record in the manifest that it was unverified.

Quality artifacts are present for every evaluated model but live at inconsistent paths
(`raw/quality/<task>/` for some, `raw/<task>_results.json` for others). Normalizing to
`raw/<benchmark>/` is tracked in TODO.md §6.

---

### 5c. The output-token ceiling is a precondition, not a client setting

The repo asserted in two places (Lightning and Ornith cards) that *"vLLM has no
default-request-budget flag"*, and therefore that `max_tokens` is purely the client's business. **That
was wrong**, and it is a §5b-class hazard: a thing a run silently depends on, which does not announce
itself when it fails.

`vllm serve --help=generation-config`:

> "If `max_new_tokens` is specified in generation config, then it sets a **server-wide limit on the
> number of output tokens for all requests**."

`--generation-config` defaults to `auto`, so **the checkpoint decides**. A model shipping
`max_new_tokens` in its `generation_config.json` caps every request on the server, regardless of what
the client asks for. There is no error and no warning — the generation is simply cut short, exactly
like a model that ran out of things to say.

Why that is dangerous here: the Lightning card's own budget curve (16k → 53.03%, 32k → 66.16%,
64k → 76.26%) shows a truncated budget moves a GPQA score by **23 points**. A silent server cap
produces the same depression while the client's `max_gen_toks=65536` makes the log look correct.

Real instance, same host: `poolside/Laguna-XS-2.1-NVFP4` ships `"max_new_tokens": 32768`;
`poolside/Laguna-S-2.1-NVFP4` ships none.

**Audit result (2026-08-26):** every checkpoint benchmarked in this repo — Lightning-30B, Ornith-35B,
Qwen3.5-122B, Qwen3.6-35B, Nemotron-3-Super-120B (FP8 + NVFP4), gpt-oss-120b, Laguna-S-2.1 — was
checked and **all carry `max_new_tokens: absent`. No published score is affected.** The only capped
checkpoint on the box, Laguna-XS, has never been benchmarked.

**The rule.** Before any run whose score depends on a generous budget:

```bash
python3 viz/check_output_budget.py \
  --model poolside/Laguna-S-2.1-NVFP4 \
  --generation-config ~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/snapshots/*/generation_config.json \
  --require-budget 32768
```

Exit 0 = the budget is permitted; **1** = a checkpoint cap or endpoint bound makes it unreachable;
**2** = undetermined (endpoint unreadable — never treated as a pass). Record the result in the
manifest.

**Removing a cap: prefer `--override-generation-config` over `--generation-config vllm`.** Verified
against the engine source (`vllm/config/model.py`, v0.27.2rc1) on 2026-08-27:

- `--generation-config vllm` only suppresses **sampling** defaults. The cap is applied in
  `get_diff_sampling_param()`, which reads a six-key whitelist (`repetition_penalty`, `temperature`,
  `top_k`, `top_p`, `min_p`, `max_new_tokens`) and short-circuits to `{}` when the source is `vllm`.
- It does **not** stop the checkpoint config from being read. `try_get_generation_config()` — the
  method that supplies special tokens — branches on `if self.generation_config in {"auto", "vllm"}`,
  i.e. **both** values load the file.
- Because it discards the whole sampling block, it also drops the checkpoint's `temperature`/`top_p`/
  `top_k`, silently changing serving defaults for every other client on that endpoint.

So the targeted fix is to override just the offending key:

```bash
--override-generation-config '{"max_new_tokens": 65536}'
```

**Check `eos_token_id` before changing generation-config handling on a chat model.** A checkpoint can
declare stop tokens the tokenizer does not. `poolside/Laguna-S-2.1-NVFP4` sets `eos_token_id: [2, 24]`
where token 24 is `</assistant>` (the chat stop token) while `tokenizer_config.json` supplies only
token 2 — so any change that alters which config supplies EOS risks generations that never terminate.

Note the asymmetry that makes this worth a dedicated check: an over-large request is **rejected
loudly** (`max_tokens=200000` → HTTP 400 against `max_model_len=131072`), while a checkpoint cap
**passes quietly**. Only the failure mode that corrupts data is the silent one.
