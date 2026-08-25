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
    "launch_script_note": "committed launch_ornith.sh pins 0.90; the SWE-bench run used 0.55 for host headroom. Script never updated — see §5."
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

Audited from the committed tree. `-` = missing, `↺` = recovered by reconstruction (see §5a).

| Model | report | preds | exit_statuses | agent cfg | run script | launch |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| ornith-35b | ✅ | ✅ | **–** | ✅ | ✅ | ✅ |
| nemotron-3.5-lightning-30b | ✅ | ✅ | ✅ | ✅ | **–** | ✅ |
| qwen3.6-35b-a3b | ✅ | ✅ | ✅ | ↺ | **–** | **–** |
| laguna-s-2.1-118b (in flight) | **–** | ✅ | ✅ | ✅ | ✅ | ✅ |
| gpt-oss-120b | **–** | **–** | ✅ | **–** | **–** | **–** |
| nemotron-3-super-120b | *no SWE-bench run* | | | | | **–** |
| qwen3.5-122b-a10b | *no SWE-bench run* | | | | | **–** |

The two that matter most:

- **Ornith has no `exit_statuses`** — the best score in the repo, and its 9 non-submissions
  cannot be audited from the repo alone.
- **Qwen3.6's agent config was never committed** — recovered 2026-08-25 from its own trajectories
  (§5a). The recovered config proves its limits matched Ornith's, so its 22 `TimeoutExpired` exits
  were a 120 s Docker `pull_timeout`, not an unfair budget. Had the trajectories and run log been
  discarded along with the config, that would have been unknowable — and the natural guess (an
  unfair timeout) was in fact wrong.

**A committed launch script is not automatically the config that ran.** `launch_ornith.sh` pins
`--gpu-memory-utilization 0.90`, but the Ornith card states the agentic runs used **0.55** for host
headroom (pi-30's agent processes share the box) — the script was simply never updated. The card
itself carries both values in different sections. So the repo contains a launch script that does
not reproduce the run it is filed under, and nothing flags the discrepancy.

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
   a trajectory — are recorded.
2. **Mark reconstructed artifacts as reconstructed**, in the filename and in a header comment
   stating what they were derived from. A recovered config must never be mistaken for a
   contemporaneous one.

The 22 lost Qwen3.6 instances show why the log matters as much as the trajectories: those instances
have **no trajectory at all** — the container never started — so the log is the *only* artifact that
explains them. Absence of a trajectory is itself data, and a coverage check (§6f) should treat a
gap between "instances attempted" and "trajectories present" as a finding rather than an accident.

Quality artifacts are present for every evaluated model but live at inconsistent paths
(`raw/quality/<task>/` for some, `raw/<task>_results.json` for others). Normalizing to
`raw/<benchmark>/` is tracked in TODO.md §6.
