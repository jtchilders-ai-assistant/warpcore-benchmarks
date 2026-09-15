# Warpcore v1 Apples-to-Apples Benchmark Design

**Status:** Approved design, pending implementation  
**Date:** 2026-09-15  
**Decision owner:** Taylor Childers  
**Repository:** `jtchilders-ai-assistant/warpcore-benchmarks`

## 1. Goal

Create a versioned, executable measurement contract for Warpcore model comparisons. A result may enter the canonical comparison matrix only when the repository can prove that all model-independent experimental variables match the contract and that all required evidence survived the run.

The design standardizes the experiment while allowing the minimum model-specific serving configuration needed for correct execution. It does not pretend that one vLLM image, parser, quantization backend, or context configuration is valid for every checkpoint.

## 2. Governing decision

Warpcore uses **standardized experiments with model-specific serving adapters**.

### 2.1 Variables fixed by the suite

The suite owns all variables that define the scientific experiment:

- benchmark and dataset revision;
- exact instance IDs and ordering;
- prompt and chat-template policy;
- task YAML and scoring-code hashes;
- sampling parameters;
- output-token ceiling;
- client harness and version;
- retry and timeout policy;
- expected denominator;
- artifact schema;
- empty, error, and timeout classification;
- statistical reporting method.

An adapter may not override these values.

### 2.2 Variables owned by a serving adapter

A model adapter records only model-specific requirements:

- model ID and immutable checkpoint revision;
- quantization format and compatible kernels;
- serving image and immutable digest;
- vLLM version;
- reasoning parser and parser plugin;
- tool-call parser;
- tokenizer override;
- model-supported and configured context length;
- GPU-memory utilization;
- `max-num-seqs` and other capacity controls;
- required environment variables and compatibility patches.

These differences are explicit experimental metadata. They are not hidden, and they may not alter prompts, scores, budgets, instance sets, or denominators.

## 3. Claims and boundaries

### 3.1 Capability claims

Quality and agentic results compare model behavior under one fixed client-side experiment. They do not claim that serving stacks are identical. The manifest must expose serving differences so a reader can assess whether a serving adapter plausibly affected the result.

### 3.2 Systems claims

Throughput and latency compare complete declared serving profiles on the same Warpcore hardware. They are deployment measurements, not architecture-only claims. A profile name includes the model, checkpoint revision, image digest, engine arguments, and workload shape.

### 3.3 Noncanonical results

Historical, superseded, diagnostic, or invalid results remain visible as evidence but cannot drive canonical rankings. No result is silently upgraded to canonical merely because its headline fields resemble the current suite.

## 4. Versioned baseline suite

The first immutable suite is `warpcore-v1`.

### 4.1 GSM8K

- Task: clean anchored-answer extraction.
- Dataset revision: pinned by the suite manifest.
- Generation ceiling: 8,192 output tokens.
- Sampling: temperature 0.
- Required evidence: aggregate result, compressed raw samples, per-item audit table, task YAML, scoring utility, run log, command record, manifest, status, completion sentinel.

### 4.2 IFEval

- Task: standard IFEval task pinned by harness and dataset revision.
- Generation ceiling: 65,536 output tokens for the v1 reasoning-model comparison.
- Sampling: temperature 0.
- Canonical headline: prompt-level strict accuracy.
- Required evidence: same quality-run evidence set as GSM8K.

The ceiling is a maximum, not a forced generation length. Residual `finish_reason=length` responses remain wrong at the declared operational ceiling and must be reported.

### 4.3 GPQA-Diamond

- Task: canonical clean answer-line extraction.
- Generation ceiling: 65,536 output tokens.
- Sampling: temperature 0.
- Canonical headline: answer-line exact match.
- Required evidence: same quality-run evidence set as GSM8K, including completion-token and finish-reason data for empty responses.

### 4.4 SWE-bench Verified

- Dataset: SWE-bench Verified.
- Instance set: fixed seed-42 shuffled 100-instance set stored explicitly in the suite.
- Agent scaffold: one pinned scaffold and submit protocol.
- Limits: pinned step, cost, command, model-request, Docker-pull, and wall-time policies.
- Client: Mac mini / Tribble, because test containers are x86.
- Canonical headline: resolved divided by all 100 assigned instances.
- Required disposition: resolved, submitted-but-wrong, model non-submission, and infrastructure failure.
- Cross-model inference: paired comparison on identical instance IDs using an exact McNemar/binomial convention named in the output.

An infrastructure-adjusted “fair” rate may characterize one run but may not rank two models because the retained populations differ.

### 4.5 Throughput

- Host: Warpcore, on-box client.
- Endpoint: raw completions.
- Workload: 512 input tokens and 256 output tokens.
- Sampling: explicit temperature 0.
- Termination: `--ignore-eos`.
- Concurrency sweep: continue beyond the apparent knee until throughput plateaus or regresses; otherwise report the final point as a measured floor.
- Report: output tokens/s, TTFT, TPOT/ITL, end-to-end latency, success count, and tested concurrency.

Short-context saturation is not evidence of long-context capacity or deployable long-context goodput.

### 4.6 Excluded from v1 rankings

- pi-30 is an optional bring-up smoke test only; it is saturated and cannot rank current models.
- AutomationBench is excluded until it receives its own approved, versioned measurement contract.
- Long-context quality and goodput form a separate future suite and do not block v1 closure.

Any change to a dataset, instance set, prompt, scoring rule, generation ceiling, harness behavior, or denominator creates a new suite version. Historical v1 results are never rewritten to conform to v2.

## 5. Repository architecture

```text
suite/
  warpcore-v1.yaml
  tasks/
    gpqa_diamond_clean_v3.yaml
    gpqa_utils.py
    gsm8k_clean_v1.yaml
  swebench/
    instances-seed42-n100.json
    scaffold.yaml
  schemas/
    adapter.schema.json
    manifest.schema.json
    result-status.schema.json

adapters/
  <model-slug>.yaml

scripts/
  validate_suite.py
  preflight_campaign.py
  run_quality.py
  run_swebench.py
  validate_campaign.py
  publish_campaign.py

results/<model>/runs/<suite-version>/<benchmark>/<run-id>/
  manifest.json
  status.json
  command.txt
  run.log
  raw/
  samples_*.jsonl.gz
  per_item.csv
  DONE
```

Historical result paths remain untouched. Migration registers their lifecycle and comparability status; it does not move or rewrite primary evidence.

## 6. Suite specification

`suite/warpcore-v1.yaml` is the single source of truth for experiment-level settings. It contains:

- suite ID and schema version;
- benchmark membership;
- immutable task and utility-file hashes;
- dataset identifiers and revisions;
- fixed instance-set digest;
- sampling settings;
- generation ceilings;
- required harness versions;
- retry and timeout policy;
- expected item counts;
- required artifact classes;
- scoring and publication metrics;
- allowed lifecycle states;
- statistical comparison policy.

The suite validator computes hashes from files and rejects stale declared hashes. A changed canonical file without a suite-version change is a CI failure.

## 7. Serving adapter contract

A serving adapter is declarative and schema-validated. Example:

```yaml
schema_version: 1
model:
  slug: gpt-oss-120b
  id: openai/gpt-oss-120b
  revision: <immutable-hugging-face-sha>
serving:
  image: eugr/spark-vllm@sha256:<digest>
  engine: vllm
  engine_version: <resolved-version>
  quantization: mxfp4
  reasoning_parser: openai_gptoss
  tool_call_parser: openai
  tokenizer: null
  moe_backend: marlin
  max_model_len: 131072
  gpu_memory_utilization: 0.90
  max_num_seqs: <validated-value>
  environment: {}
```

The schema rejects unknown keys and every experiment-level key, including task names, prompts, dataset selectors, generation ceilings, sampling settings, scoring filters, instance sets, and denominators.

Mutable image tags alone are insufficient. Before publication, the manifest must contain the resolved image digest and effective engine arguments. Unknown historical values use the literal `"unrecorded"`; canonical new runs fail rather than publish with an unknown required value.

## 8. Campaign lifecycle

A campaign follows this state machine:

```text
planned -> preflight_passed -> running -> completed -> validated -> published
                              \-> failed
```

State transitions are append-only records in `status.json`; a later state does not erase earlier failure or retry history.

### 8.1 Planned

The runner resolves suite version, adapter, model revision, task hashes, expected instances, output directory, and intended command without starting work.

### 8.2 Preflight passed

All applicable gates must pass:

1. exact model identity from `/v1/models`;
2. real completion with usable content;
3. parser classification using the full response object;
4. configured context and output-ceiling feasibility;
5. checkpoint generation-config cap inspection;
6. timeout arithmetic using measured throughput and a conservative tail policy;
7. limit-5 task smoke through the real harness with raw-response extraction checked;
8. SWE-bench image-cache, instance-set, scaffold, and submit-protocol checks when applicable;
9. clean output destination and sufficient disk space.

Exit 1 means a diagnosed defect. Exit 2 means inconclusive or unreachable. Both block launch.

### 8.3 Running and completed

Long clients run under `/usr/bin/screen` on the Mac mini. Throughput sweeps run under `tmux` on Warpcore. Logs and artifacts live under stable `$HOME` paths, never `/tmp`. The runner writes `DONE` only after the harness exits successfully and the expected raw files exist.

A vanished multiplexer session is not completion evidence.

### 8.4 Validated

Validation is offline and fail-closed. It proves:

- suite and adapter schema validity;
- declared hashes equal actual files;
- expected item IDs and counts are complete;
- no unexpected duplicates exist;
- every item has a classified disposition;
- raw samples and required provenance exist;
- aggregate scores reproduce from per-item evidence;
- empty, timeout, request-error, and finish-reason counts reconcile;
- manifest effective settings match the resolved run;
- no prohibited experiment override entered through an adapter or command line;
- secret scanning passes on staged artifacts.

### 8.5 Published

Publication is generated, not hand-edited. Only validated `current` runs from the requested suite version may populate the canonical matrix. The generated entry links to the manifest and per-item evidence.

## 9. Artifact lifecycle

Each run is assigned exactly one lifecycle state:

- `current`: canonical publication candidate and hard-gated;
- `superseded`: replaced by a named run or corrected composite, retained as evidence;
- `historical`: predates the contract and is not automatically canonical;
- `diagnostic`: investigates a failure but is not a score;
- `replay`: re-serves a named subset from a named base run;
- `invalid`: violated the measurement contract and cannot be published.

A replay must list exact item IDs, reference its base run, preserve the original prompts and scoring policy, and produce a composite manifest. The composite becomes `current` only after validation; the base remains `superseded` or `historical`.

Hard CI thresholds apply only to `current` publication candidates. Other states remain visible in audit output. Evidence is never deleted or hidden to make CI green.

## 10. Failure semantics

### 10.1 Quality tasks

- `finish_reason=length` with empty visible content: budget residual; count wrong at the declared ceiling and report it.
- `finish_reason=stop` with answer text stranded in a reasoning field: parser/field-routing defect; the run is invalid unless affected items are exactly recovered.
- request, transport, or server errors: infrastructure failure; the run is not canonical until exact-item recovery or a full rerun completes.
- empty stop with no content anywhere: explicit empty generation; retain and classify rather than silently retrying it away.

Served-only rates are diagnostic upper bounds and never capability scores.

### 10.2 SWE-bench

- submitted and graded: model outcome;
- step, cost, or context limit: model/config operational outcome and remains in the full denominator;
- container pull failure before model invocation: infrastructure outcome;
- malformed captured patch caused by scaffold: harness outcome;
- server/parser failure: serving outcome.

All 100 assigned instances remain in the headline denominator. Disposition categories explain why misses occurred; they do not erase misses.

### 10.3 Segmented campaigns

If an effective setting changes mid-run, the campaign records a new segment with exact instance IDs, serving image/version/arguments, reason, and timing. A segmented run may be reported only when the effect is bounded or explicitly stated as unknown. Configurations are never silently blended.

## 11. Publication and statistical rules

The canonical table includes:

- model and checkpoint revision;
- suite version;
- score and full denominator;
- empty/error rate or disposition summary;
- UTC measurement date;
- serving-profile link;
- validated manifest link.

Historical values appear only in a clearly labeled legacy view or model-card history.

For SWE-bench, cross-model claims require identical instance sets and paired statistics. Report discordant-pair counts, exact p-value, and interval for the paired difference. “Not distinguishable” means insufficient evidence, not equality.

For reweighted or subset estimates, report uncertainty and label the target population. Do not rank models using different fair denominators.

For throughput, label every value with workload shape and concurrency. A rising final point is a floor, not a ceiling. Report latency alongside aggregate throughput.

## 12. Initial implementation and closure sequence

1. Add `warpcore-v1` suite specification, schemas, canonical task files, and fixed SWE-bench instance list.
2. Add initial adapters for gpt-oss-120b and Qwen3.6-35B.
3. Implement suite, adapter, manifest, and lifecycle validators with negative tests.
4. Implement one parameterized quality runner with mandatory preflight and durable launch.
5. Implement one parameterized SWE-bench runner with cache, identity, instance-set, and scaffold gates.
6. Implement lifecycle-aware post-run validation and generated publication.
7. Register existing artifacts as `current`, `superseded`, `historical`, `diagnostic`, `replay`, or `invalid` without moving them.
8. Run the first closure campaigns through the finished machinery:
   - Qwen3.6 SWE-bench rerun;
   - gpt-oss SWE-bench rerun;
   - gpt-oss clean GSM8K rerun.
9. Publish the first canonical `warpcore-v1` comparison matrix.
10. Add Qwen3.5-122B and later models only through the same contract.

The three closure reruns are not run before the contract and validators exist; otherwise they could create another generation of bespoke artifacts.

## 13. Acceptance criteria

P1 is complete when:

1. Two different models can execute from separate serving adapters under the same immutable suite.
2. Adapter attempts to override experiment-level settings fail validation.
3. Suite file drift without a version change fails CI.
4. Live preflight exit 1 or 2 prevents harness launch.
5. Every expected item and required artifact is retained under a stable run directory.
6. Aggregate results reproduce from per-item evidence.
7. Empty, timeout, error, and SWE-bench disposition counts reconcile exactly.
8. Only validated `current` runs enter the canonical comparison matrix.
9. Historical and superseded failures remain visible but do not control current publication gates.
10. Qwen3.6 SWE-bench, gpt-oss SWE-bench, and gpt-oss clean GSM8K have canonical v1 results or an explicitly retained, evidence-backed blocked status.
11. Local verification, remote CI, and secret scanning are green on the final `main` SHA.

## 14. Explicit non-goals

P1 does not:

- force one serving image or parser across incompatible models;
- erase or rewrite historical evidence;
- make historical runs reproducible when primary artifacts are lost;
- add AutomationBench;
- implement long-context quality or goodput studies;
- run full SWE-bench Verified n=500;
- claim that one serving profile isolates model architecture from systems effects.

Those require separate approved suite versions or studies.
