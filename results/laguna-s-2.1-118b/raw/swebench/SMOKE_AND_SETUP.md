# SWE-bench Verified (n=100, seed-42 shuffle) — poolside/Laguna-S-2.1-NVFP4

Status: **generation IN PROGRESS** (started 2026-08-22 23:22 CDT). No score yet.
This file records the preflight + smoke evidence so the eventual number is auditable.

## Setup

| Piece | Where | Detail |
| --- | --- | --- |
| Model / inference | warpcore (GB10, aarch64) | vLLM, NVFP4, `vllm_laguna` |
| Agent scaffold | Mac mini (x86_64) | mini-swe-agent, **native tool-calling** |
| Test containers | Mac mini (x86_64) | 142 sweb images already cached from prior runs |

x86 gate: SWE-bench's per-instance test images are amd64-only. This Mac mini **is**
`x86_64` (verified `uname -m`, Docker 29.3.0, 1.5 TiB free), so the harness runs locally —
no ssh hop, and warpcore's aarch64 never touches the test containers.

## Scaffold choice — verified, not assumed

poolside models are **tool-call-native**; running them on the bash-in-content
(backticks) scaffold produces a degenerate loop and a fake 0/N (documented for
Laguna-XS-2.1). Laguna-S is served with `--enable-auto-tool-choice --tool-call-parser
poolside_v1`, so the mini-swe-agent DEFAULT tool-calling config is correct. Probed
before committing any hours (`laguna_toolcall_probe.py`):

```
finish_reason : tool_calls
tool_calls    : 1  ->  {"command": "ls -la /testbed"}   PARSED OK
content       : ''        <- empty, text went to `reasoning`
```

Note the empty `content`: this model exhibits the same `reasoning`-field defect that
depressed its lm-eval scores (see ISSUES.md #15). It is **harmless here** because the
tool-calling scaffold reads `tool_calls`, not `content`. That is luck, not design — a
bash-in-content run of this model would have scored near zero for purely mechanical
reasons.

## Comparability: one deliberate config delta

The config is copied **byte-for-byte** from the Ornith n=100 run (`step_limit: 250`,
`max_tokens: 32768`, `temperature: 0`, robust `git add -A && git diff --cached` submit
protocol), changing only `model_name` — and one required fix:

```diff
-    timeout: 1800
+    timeout: 5400
```

**Why this was necessary, measured not guessed.** The first smoke attempt returned 1/3
Submitted with two `litellm.Timeout` failures at 1800 s. The wedged-engine hypothesis was
tested and **ruled out** (`/chat` 200 in 0.67 s, 0 aborts, 0 errors server-side). The real
cause is arithmetic:

```
measured per-request rate under load : 8.79 tok/s
max_tokens allowed per step          : 32768
=> a full-length step needs          : 3727 s  (62 min)
=> configured client timeout         : 1800 s  (30 min)
```

Corroboration: `finished_reason="length"` accounts for **2243/4915 (46%)** of this
model's completions — long steps are routine, not rare.

> **CORRECTION (2026-08-23):** that 46% figure is **not valid for this run**. Those are
> *cumulative* counters spanning the whole server lifetime, including the GSM8K / IFEval /
> GPQA lm-eval runs at much smaller `max_tokens`. Sampling **deltas** over a live
> SWE-bench window shows steps finishing on `stop` at 200–2000 tokens. The 5400 s fix is
> still empirically justified (1800 s fired twice in the smoke; **0 timeouts in 11 h**
> since), but full-cap steps are rarer than stated here. See `WHY_SLOW.md`.

At 1800 s litellm was killing
**healthy, still-generating** requests and retrying, burning another 30 min each.

This is a **client-side deadline, not a capability knob**: it grants no extra steps, no
extra attempts, and no larger token budget. The instance set (seed-42 shuffle), step
limit, and sampling are identical to Ornith/Lightning/Qwen3.6. Ornith did not need it
because it generates ~2× faster per token.

## Smoke result (3 instances, default step_limit 250)

| Instance | Exit status | Tool calls | Distinct | JSON errors | Patch |
| --- | --- | ---: | ---: | ---: | --- |
| django__django-14672 | Submitted | 43 | 41 | 0 | valid `diff --git` (512 B) |
| django__django-11299 | Submitted | 88 | 83 | 0 | valid `diff --git` (700 B) |
| sphinx-doc__sphinx-10449 | LimitsExceeded | 254 | **254** | 0 | empty |

**2/3 submitted valid diffs, 0 timeouts, 0 tool-call JSON corruption.** Both known
failure modes are absent: the Laguna scaffold mismatch and the gpt-oss vLLM tool-call
JSON bug.

The sphinx `LimitsExceeded` is **not** a degenerate loop — 254 tool calls, **254
distinct** (zero repeats). The model did varied, non-repeating work and ran out of steps
at the cap. Per the harness rule, judge by trajectory, not exit status: this is a genuine
"hard instance, ran out of budget" outcome and it will show up as an empty patch in the
final tally. Expect the headline score to be a **conservative floor**.

## Timing

Mean instance duration at w=3 was **34.5 min** (11.7 / 45.8 / 46.1). At w=4 the n=100
projection is **~14 h**, degrading if per-request throughput drops as workers climb.
Laguna is the slowest model in this repo (~17 tok/s single-stream, 259 tok/s aggregate
peak), so this run is expected to take substantially longer than Ornith's ~11 h.

## Files

- `swebench_laguna_config.yaml` — the exact config used
- `run_gen_n100.sh` / `run_smoke.sh` — runners (restartable: same `-o`, no `--redo-existing`)
- `laguna_toolcall_probe.py` — the scaffold gate
- `timeout_arithmetic.py` — the measurement that justified 5400 s
- `traj_triage.py` — tool-call-aware loop/submission triage
- `smoke_preds.json` — smoke predictions
