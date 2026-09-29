# SWE-bench Submission Reporting and Warpcore v2 Launch Design

**Status:** Approved direction, detailed design for maintainer review  
**Date:** 2026-09-29  
**Decision owner:** Taylor Childers  
**Repository:** `jtchilders-ai-assistant/warpcore-benchmarks`

## 1. Goal

Make the top-level SWE-bench comparison directly answer two questions for the same frozen 100 instances:

1. How many instances produced a nonempty patch that reached grading?
2. How many of all 100 assigned instances were resolved?

For future campaigns, remove the separate n=20 launch qualification and run the full frozen n=100 campaign after cheap live preflight, while retaining the in-run systemic-failure circuit breaker and all evidence/publication gates.

## 2. Delivery boundaries

This design is delivered as two independently reviewable increments:

1. **Reporting increment:** update the README and generated SWE-bench data/figures to use each model's newest, best-provenance n=100 run. Show `submitted / 100` and `resolved / 100`; any displayed percentage is `resolved / 100`. Preserve older runs in footnotes and provenance rather than rewriting them.
2. **Protocol increment:** introduce `warpcore-v2` for future runs. Keep `warpcore-v1` immutable and retain all v1 qualification code and evidence for historical reproducibility. V2 removes n=20 qualification as a launch prerequisite, but keeps live preflight, exact frozen n=100 identity, the systemic-failure circuit breaker, official grading, complete reconciliation, secret scanning, and publication validation.

The reporting increment may merge before the protocol increment. No historical run is relabeled merely because the display changes.

## 3. Reporting semantics

### 3.1 Headline columns

The top-level model table will replace the single overloaded SWE-bench cell with two columns:

- **SWE submitted:** number of assigned instances with a nonempty patch that reached a terminal grader verdict, shown as `N/100`.
- **SWE resolved:** number of all assigned instances resolved by official grading, shown as `N/100` and optionally `N%` because the denominator is always 100.

The table will not show conditional `resolved / submitted` percentages. Those remain useful diagnostic details in model cards and footnotes, but they are not cross-model headline statistics because submission is selected by model behavior.

### 3.2 Definitions

For a selected n=100 report:

- `resolved = len(resolved_ids)`
- `submitted-but-wrong = len(unresolved_ids)`
- `submitted = resolved + submitted-but-wrong`
- `empty/error/incomplete` remain outside `submitted` and still remain failures in the full denominator
- `resolved percentage = resolved / 100`

A generated nonempty patch that fails before a terminal grader verdict is not counted as submitted in the headline. It must remain visible as a grading error in provenance. This makes “submitted” mean “gradeable patch with a verdict,” matching the user-facing comparison question and preventing ambiguous treatment of historical grader errors.

### 3.3 Selected run per model

Use the newest, best-provenance completed n=100 run for each model:

- Ornith 1.0: historical retained n=100 report — submitted 91/100, resolved 73/100.
- Laguna S 2.1: historical retained n=100 report — submitted 65/100, resolved 55/100.
- Nemotron 3.5 Lightning: historical retained n=100 report — submitted 98/100, resolved 51/100.
- Qwen3.6: validated `warpcore-v1` run `qwen36-swebench-n100-20260919` — submitted 87/100, resolved 57/100. The older 44/100 run remains historical and is disclosed in its footnote.
- Qwen3.5: completed diagnostic run `qwen35-swebench-trial-n100-20260928` — submitted 76/100, resolved 57/100, visibly labeled diagnostic because it was not eligible under the frozen v1 qualification policy.

Models without a completed, gradeable n=100 result show `not measured` in both columns.

### 3.4 Derived-data architecture

`viz/common.py` remains the single source of selected SWE-bench result paths. It will point to the selected best-provenance reports, including normalized v1 paths where appropriate. A shared parser will derive the decomposition from either historical aggregate reports or normalized `grading_results.json` without hand-entered counts.

Generated data and figures must consume that parser. Tests must assert:

- exact selected source path per model;
- exact set equality for the shared seed-42 n=100 inventory;
- submitted and resolved counts above;
- `resolved <= submitted <= 100`;
- no headline conditional percentage;
- Qwen3.5 remains diagnostic and excluded from the canonical matrix.

### 3.5 Figure and prose changes

The SWE-bench figure's decomposition remains resolved / submitted-but-wrong / no terminal graded patch. Labels and titles must avoid claims such as “most accurate coder” based on conditional accuracy. Comparative prose will use full-denominator resolved counts and paired per-item statistics only.

The existing infrastructure-adjusted fair-rate artifact remains available for causal diagnosis, but fair rates leave the top-level model table. They do not answer the requested all-100 comparison.

## 4. Warpcore v2 SWE-bench launch policy

### 4.1 Versioning

Removing the n=20 gate changes launch behavior and therefore cannot modify frozen `warpcore-v1`. Add a new suite identity, `warpcore-v2`, with its own suite file and immutable hashes. V1 files, schemas, qualification artifacts, runner behavior, and historical manifests remain valid and reproducible.

V2 initially keeps all benchmark experiments identical to v1 except SWE-bench launch authorization. This isolates the policy change and preserves comparability of the n=100 measurements.

### 4.2 V2 launch gates

A v2 n=100 SWE-bench campaign may launch only after:

1. suite and adapter schema validation;
2. exact model identity verification at `/v1/models`;
3. a real bounded generation/tool-call probe through the production API path;
4. x86 Docker-host and full frozen-image cache preflight;
5. exact scaffold, instance-set, adapter, image, model-revision, and serving-profile hashes;
6. durable Screen execution and stable run directory.

There is no separate n=20 task run and no qualification seal.

### 4.3 In-run protection

The existing suite-owned systemic-failure circuit breaker remains mandatory. It observes early completed instances and terminates a campaign when retained evidence proves a systemic parser, transport, server, or harness failure. Thresholds remain suite-owned and cannot be overridden by adapters or ordinary CLI flags.

The breaker is not a hidden smaller benchmark: completed early instances remain part of the same frozen n=100 campaign. If it trips, the campaign is failed/diagnostic and no score is published.

### 4.4 Publication

A completed v2 run publishes only after all 100 IDs reconcile across predictions, trajectories, exit statuses, grader partitions, manifest, and status history. The canonical headline remains `resolved / 100`. Publication also records submitted, submitted-but-wrong, empty patch, grading error, incomplete, and exit-status counts.

Canonical eligibility requires lifecycle `current`, successful validation, complete evidence, no unresolved infrastructure integrity defect, and the standard secret scan. The removal of n=20 does not weaken completion or publication checks.

## 5. Compatibility and migration

- Do not delete `viz/swebench_qualification.py`, its schema, tests, or v1 Make targets; they remain part of v1 reproduction.
- Make runner qualification behavior suite-driven: v1 requires its qualification record; v2 does not define one and therefore proceeds through v2 preflight and breaker gates.
- Reject unknown suite policy shapes. Absence of qualification is allowed only for a suite explicitly declaring the v2 launch policy, never as a generic bypass flag.
- Remove the temporary `--noncanonical-trial` need for ordinary v2 launches; retain compatibility sufficient to interpret the committed Qwen3.5 v1 diagnostic command and evidence.
- Do not promote Qwen3.5's historical diagnostic run retroactively. Future equivalent runs may be canonical under v2 after satisfying v2 gates.

## 6. Verification and acceptance

### Reporting increment

- README shows separate `SWE submitted` and `SWE resolved` columns.
- Every shown value derives from a committed official grader artifact.
- Qwen3.6 uses the validated 2026-09-19 v1 run; the 44/100 run is only historical context.
- Qwen3.5 is shown as 76/100 submitted and 57/100 resolved with a diagnostic marker.
- Figures and generated JSON are regenerated from the selected sources.
- Full tests, `make ci`, artifact audit, and secret scan pass.

### Protocol increment

- A v1 dry run without a valid qualification record still fails before campaign state is created.
- A v2 dry run with all preflight inputs succeeds without consulting or creating a qualification record.
- A v2 launch cannot bypass preflight, identity, image-cache, scaffold, or circuit-breaker checks.
- A circuit-breaker negative control terminates a synthetic systemic-failure campaign and writes no publication sentinel.
- A complete synthetic v2 n=100 run validates and publishes only at `resolved / 100` with full decomposition.
- V1 canonical artifacts continue validating against their immutable snapshots.
- Full tests, `make ci`, publication validators, artifact audit, secret scan, and independent exact-SHA review pass before merge.

## 7. Non-goals

- Re-running historical models solely to improve missing provenance.
- Ranking models by `resolved / submitted` or infrastructure-adjusted fair rates.
- Changing the frozen seed-42 n=100 instance set, scaffold, budgets, or grader.
- Relabeling old diagnostic/historical runs as canonical.
- Deleting the v1 qualification implementation or evidence.
