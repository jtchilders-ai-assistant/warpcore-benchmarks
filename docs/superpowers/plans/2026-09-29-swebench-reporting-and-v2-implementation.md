# SWE-bench Reporting and Warpcore v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish source-derived SWE-bench submitted/resolved counts on a common n=100 denominator and add a fail-closed `warpcore-v2` launch policy that replaces the separate n=20 qualification with production preflight plus the existing in-run circuit breaker.

**Architecture:** First, make one parser derive a report decomposition from committed prediction and grading artifacts and make README/figure data consume it. Second, version launch authorization in the suite contract: v1 explicitly requires a qualification seal, while v2 explicitly declares direct n=100 launch after preflight. Unknown or absent policy shapes fail closed. Existing v1 code and evidence remain reproducible.

**Tech Stack:** Python 3, pytest, YAML/JSON Schema, mini-swe-agent/SWE-bench artifacts, Make, deterministic Matplotlib generation, Git/GitHub Actions.

---

## Mandatory semantic correction

The approved-direction spec currently contradicts the user's stated reporting convention. Before implementation, correct §3 so:

- **submitted** means an assigned instance whose retained prediction contains a nonempty `model_patch`, whether or not grading later produced a verdict;
- **resolved** means an ID in official `resolved_ids`;
- every headline percentage is `resolved / 100`;
- Qwen3.6's selected validated run is **89/100 submitted, 57/100 resolved** (the two nonempty grading-error patches remain submissions);
- Qwen3.5 remains **76/100 submitted, 57/100 resolved**, diagnostic/noncanonical under v1;
- grader-verdict coverage is separately `resolved + unresolved`, never relabeled as submission count.

For older runs lacking retained per-prediction or exit-status evidence, derive only what the official aggregate artifact proves and disclose the weaker provenance. Never invent an exit-status classification.

## File map

- Modify `docs/superpowers/specs/2026-09-29-swebench-reporting-and-v2-launch-design.md`: correct submission semantics and selected counts.
- Create `viz/swebench_reporting.py`: parse legacy/normalized artifacts into one validated decomposition.
- Modify `viz/common.py`: select the strongest retained n=100 source bundle per model.
- Modify `viz/collect_matrix.py`: emit source-derived submitted/resolved/decomposition fields.
- Modify `viz/fig2_swebench.py`: consume the shared decomposition and remove conditional-rate headline framing.
- Modify `README.md`, `PROVENANCE.md`, and affected model cards: display and explain the comparable counts and historical selections.
- Regenerate `viz/data/bench_matrix.json`, `viz/data/swebench_fair.json` only if its source schema changes, and `viz/out/fig2_swebench.{png,svg}`.
- Create `tests/test_swebench_reporting.py`: source selection, counts, vocabulary, and lifecycle tests.
- Create `suite/warpcore-v2.yaml`: clone v1 experiment identity while explicitly changing only SWE launch authorization.
- Modify `suite/schemas/suite.schema.json`: require a closed launch-policy vocabulary.
- Modify `viz/contract.py`: validate suite-policy semantics and referenced files.
- Modify `viz/run_swebench.py`: require qualification only when the suite declares it.
- Modify `viz/create_campaign.py` and `Makefile` only where needed to route v2 without weakening v1.
- Create `tests/test_warpcore_v2.py` and extend `tests/test_swebench_qualification.py`: v1/v2 positive and negative controls.
- Modify `AGENTS.md`, `RUNBOOK.md`, and `PROVENANCE.md`: operationally distinguish v1 and v2.

---

## Increment A — comparable SWE-bench reporting

### Task 1: Correct the contract prose before code

**Files:**
- Modify: `docs/superpowers/specs/2026-09-29-swebench-reporting-and-v2-launch-design.md`

- [ ] Replace the terminal-verdict definition of submitted with nonempty-patch semantics.
- [ ] Change Qwen3.6 selected-run count from `87/100` to `89/100`; keep resolved at `57/100` and disclose 2 grading errors.
- [ ] State that older aggregate-only reports may prove nonempty count only through their retained aggregate partitions and carry an explicit provenance caveat.
- [ ] Run:

```bash
python3 - <<'PY'
from pathlib import Path
p=Path('docs/superpowers/specs/2026-09-29-swebench-reporting-and-v2-launch-design.md').read_text()
assert '89/100' in p and 'nonempty' in p
assert 'submitted 87/100' not in p
PY
git diff --check
```

- [ ] Commit:

```bash
git add docs/superpowers/specs/2026-09-29-swebench-reporting-and-v2-launch-design.md docs/superpowers/plans/2026-09-29-swebench-reporting-and-v2-implementation.md
git commit -m "docs: finalize SWE-bench submission semantics"
```

### Task 2: Specify the shared reporting parser with failing tests

**Files:**
- Create: `tests/test_swebench_reporting.py`
- Create later: `viz/swebench_reporting.py`

- [ ] Write tests importing `load_swebench_report` and `selected_swebench_reports` and asserting this result shape:

```python
{
    "expected": 100,
    "submitted": 89,
    "resolved": 57,
    "unresolved": 30,
    "empty_patch": 11,
    "grading_error": 2,
    "incomplete": 0,
    "source_paths": [...],
    "provenance": "normalized-complete",
}
```

- [ ] Assert the selected sources and values:
  - Ornith: submitted 91, resolved 73;
  - Laguna: submitted 65, resolved 55;
  - Nemotron: submitted 98, resolved 51;
  - Qwen3.6 validated v1: submitted 89, resolved 57, grading errors 2;
  - Qwen3.5 diagnostic v1: submitted 76, resolved 57, grading errors 0.
- [ ] Assert all selected reports use the same frozen 100 IDs where item-level IDs exist; where old evidence cannot prove exact prediction inventory, require a provenance limitation instead of fabricating equality.
- [ ] Assert `resolved <= submitted <= expected`, all disjoint grading partitions reconcile to 100, and predictions' nonempty-patch count is authoritative when predictions exist.
- [ ] Assert the older Qwen3.6 44/100 run is not selected but remains addressable as historical evidence.
- [ ] Run RED:

```bash
python3 -m pytest -q tests/test_swebench_reporting.py
```

Expected: import failure because `viz/swebench_reporting.py` does not exist.

### Task 3: Implement the source-derived reporting parser

**Files:**
- Create: `viz/swebench_reporting.py`
- Modify: `viz/common.py`

- [ ] Implement immutable source bundles in `viz/common.py`; each bundle names official grading evidence and optional prediction/exit-status evidence. Do not encode result counts there.
- [ ] Implement `load_swebench_report(repo, bundle)` to:
  - parse legacy and normalized official reports;
  - calculate resolved/unresolved/empty/error/incomplete partitions;
  - count nonempty `model_patch` values when predictions are retained;
  - otherwise use only an aggregate field/partition that actually proves nonempty patches and mark provenance limitations;
  - reject duplicate/overlapping IDs, foreign IDs, non-100 totals, impossible inequalities, and disagreement between predictions and aggregate counts;
  - retain exact source paths and lifecycle labels.
- [ ] Implement `selected_swebench_reports(repo)` as the single source of the chosen run per model.
- [ ] Run GREEN:

```bash
python3 -m pytest -q tests/test_swebench_reporting.py
```

- [ ] Mutation-check the Qwen3.6 semantics by temporarily changing one retained nonempty prediction to empty in a temporary copied fixture and require the parser test to fail; do not modify committed raw evidence.
- [ ] Commit:

```bash
git add viz/common.py viz/swebench_reporting.py tests/test_swebench_reporting.py
git commit -m "feat: derive comparable SWE-bench submission counts"
```

### Task 4: Drive generated data and figure from the parser

**Files:**
- Modify: `viz/collect_matrix.py`
- Modify: `viz/fig2_swebench.py`
- Modify: relevant generation tests under `tests/`

- [ ] Add failing tests asserting generated SWE entries contain `submitted`, `resolved`, `expected=100`, complete disposition fields, source paths, and lifecycle/provenance labels.
- [ ] Add a failing test proving `resolved_rate == resolved / expected`, never `resolved / submitted`.
- [ ] Add a failing test proving figure labels use full-denominator language and distinguish empty/error/incomplete from submitted-but-unresolved.
- [ ] Run RED:

```bash
python3 -m pytest -q tests/test_swebench_reporting.py tests/test_check_generated.py
```

- [ ] Replace duplicated SWE parsing in `collect_matrix.py` and `fig2_swebench.py` with calls to `swebench_reporting.py`.
- [ ] Preserve fair-rate data as diagnostic output, but remove it and conditional accuracy from the headline comparison.
- [ ] Run GREEN and deterministic generation:

```bash
python3 -m pytest -q tests/test_swebench_reporting.py tests/test_check_generated.py
make data
make figs
make check
```

- [ ] Inspect generated diffs; reject any unrelated model/task changes.
- [ ] Commit source and generated outputs together:

```bash
git add viz/collect_matrix.py viz/fig2_swebench.py viz/data/bench_matrix.json viz/out/fig2_swebench.png viz/out/fig2_swebench.svg tests
git commit -m "feat: generate full-denominator SWE-bench reporting"
```

### Task 5: Update publication prose without relabeling history

**Files:**
- Modify: `README.md`
- Modify: `PROVENANCE.md`
- Modify: `results/qwen3.6-35b-a3b/README.md`
- Modify: `results/qwen3.5-122b-a10b/README.md`
- Modify other model cards only if their selected-run explanation is stale.

- [ ] Add RED prose tests requiring separate `SWE submitted` and `SWE resolved` columns and the five artifact-derived pairs.
- [ ] Require Qwen3.5's visible diagnostic/noncanonical marker and canonical-matrix exclusion.
- [ ] Require a Qwen3.6 footnote preserving the older 71-submitted/44-resolved historical run and its grading-error caveat.
- [ ] Require Ornith's weaker exit-status provenance disclosure.
- [ ] Reject headline `resolved/submitted` percentages and fair-rate ranking language.
- [ ] Run RED, patch prose, then run GREEN:

```bash
python3 -m pytest -q tests/test_swebench_reporting.py tests/test_qwen35_campaign_evidence.py
```

- [ ] Commit:

```bash
git add README.md PROVENANCE.md results/*/README.md tests/test_swebench_reporting.py
git commit -m "docs: report SWE-bench submitted and resolved out of 100"
```

### Task 6: Verify and independently review Increment A

- [ ] Run:

```bash
python3 -m pytest -q
make check
make check-artifacts
make ci
git diff --check origin/main...HEAD
```

- [ ] Run the repository's configured Gitleaks/full-history secret scan; record exact command and output without exposing any discovered secret value.
- [ ] Independently reconcile every README count from committed artifacts with a separate script, not the new parser alone.
- [ ] Request independent review focused on semantics, source selection, denominator accounting, generated artifacts, and lifecycle labels.
- [ ] Fix every Critical/Important finding with a failing regression test, rerun the complete gates, and obtain re-review of the new exact SHA.

---

## Increment B — explicit warpcore-v2 launch authorization

### Task 7: Add RED contract tests for a closed launch-policy vocabulary

**Files:**
- Create: `tests/test_warpcore_v2.py`
- Modify later: `suite/schemas/suite.schema.json`, `viz/contract.py`, `suite/warpcore-v2.yaml`

- [ ] Test that v1 explicitly declares qualification-required authorization.
- [ ] Test that v2 explicitly declares direct-n100-after-preflight authorization.
- [ ] Test that missing, unknown, misspelled, contradictory, or adapter-overridden policy fails closed.
- [ ] Test v2 preserves v1 dataset revisions, seed-42 n=100 IDs/order, scaffold, generation ceilings, grader, retry, timeout, and circuit-breaker policy; permit only suite ID/design reference/launch authorization differences.
- [ ] Run RED:

```bash
python3 -m pytest -q tests/test_warpcore_v2.py
```

Expected: failures because v2 and the policy schema do not exist.

### Task 8: Create the v2 suite and validate policy semantics

**Files:**
- Create: `suite/warpcore-v2.yaml`
- Modify: `suite/schemas/suite.schema.json`
- Modify: `viz/contract.py`

- [ ] Add a required SWE launch authorization enum, for example:

```yaml
swebench:
  launch_authorization:
    mode: direct_n100_after_preflight
```

V1 must explicitly use `qualification_seal`; do not infer policy from suite ID or missing keys.
- [ ] Clone v1 into v2 and change only identity/design reference/authorization fields plus hashes necessarily affected by versioned identity.
- [ ] Validate referenced paths and reject forbidden qualification fields in direct mode or missing v1 qualification inputs in seal mode.
- [ ] Run GREEN plus both suite validators:

```bash
python3 -m pytest -q tests/test_warpcore_v2.py
python3 viz/validate_suite.py suite/warpcore-v1.yaml --repo .
python3 viz/validate_suite.py suite/warpcore-v2.yaml --repo .
```

- [ ] Commit:

```bash
git add suite/warpcore-v1.yaml suite/warpcore-v2.yaml suite/schemas/suite.schema.json viz/contract.py tests/test_warpcore_v2.py
git commit -m "feat: define warpcore-v2 SWE-bench launch policy"
```

### Task 9: Make qualification authorization suite-driven without a bypass

**Files:**
- Modify: `viz/run_swebench.py`
- Modify: `tests/test_swebench_qualification.py`
- Modify: `tests/test_warpcore_v2.py`

- [ ] Add RED production-path tests proving:
  - v1 missing/stale/wrong-model qualification still exits nonzero before `command.txt`, state transition, generation, grading, or `DONE`;
  - v1 valid qualification proceeds;
  - v2 never reads or creates `qualification.json` and proceeds only after successful preflight;
  - v2 preflight exit 1 or 2 blocks before generation and `DONE`;
  - unknown policy blocks;
  - `--noncanonical-trial` remains restricted to diagnostic v1 compatibility and cannot authorize a current v2 run.
- [ ] Run RED:

```bash
python3 -m pytest -q tests/test_swebench_qualification.py tests/test_warpcore_v2.py
```

- [ ] Parse launch mode from the validated suite in `SwebenchRunner`; branch only on the closed enum.
- [ ] Keep `run_qualification()` and v1 artifact verification unchanged for v1 reproduction.
- [ ] Ensure dry-run output states the exact authorization path and remains side-effect free.
- [ ] Run GREEN and sabotage the policy branch temporarily to prove the v1-no-seal and v2-preflight-failure tests bite.
- [ ] Commit:

```bash
git add viz/run_swebench.py tests/test_swebench_qualification.py tests/test_warpcore_v2.py
git commit -m "feat: enforce suite-driven SWE-bench launch authorization"
```

### Task 10: Preserve every production preflight and circuit-breaker gate

**Files:**
- Modify only as required: `viz/create_campaign.py`, `Makefile`, `tests/test_warpcore_v2.py`

- [ ] Add RED tests that exercise the real command construction and prove v2 still requires model identity, bounded generation/tool call, Docker/x86/image cache, scaffold hash, adapter hash, model/image revision, serving profile, Screen, and stable run directory.
- [ ] Reuse existing preflight and breaker implementation; do not create weaker v2 duplicates.
- [ ] Add/retain a synthetic systemic-failure test where the breaker terminates the owned generation process, writes `circuit_breaker.json`, marks failed/invalid, preserves partial evidence, skips grading, and writes no `DONE`.
- [ ] Add a success-path synthetic n=100 test reconciling all predictions, trajectories, exit statuses, grading partitions, manifest fields, status history, and a final `DONE` written last.
- [ ] Run RED/GREEN:

```bash
python3 -m pytest -q tests/test_warpcore_v2.py tests/test_swebench_circuit_breaker.py tests/test_run_swebench.py
```

Use the exact existing circuit-breaker test filename if discovery shows a different name.
- [ ] Commit:

```bash
git add viz/create_campaign.py Makefile tests/test_warpcore_v2.py tests/test_swebench_circuit_breaker.py
git commit -m "test: prove warpcore-v2 preflight and breaker gates"
```

### Task 11: Validate v2 publication and preserve v1/Qwen history

**Files:**
- Modify as required: `viz/validate_campaign.py`, `viz/publish_campaign.py`, publication tests, `AGENTS.md`, `RUNBOOK.md`, `PROVENANCE.md`

- [ ] Add RED tests proving a complete current/validated v2 run publishes resolved/100 plus the full decomposition, while incomplete, infrastructure-invalid, diagnostic, or zero-verdict runs are rejected.
- [ ] Prove v1 canonical fixtures continue validating and Qwen3.5's diagnostic v1 run remains rejected with no qualification seal and no canonical-matrix cell.
- [ ] Ensure publisher derives submitted from nonempty predictions, not verdict count, and rejects disagreement.
- [ ] Update operator docs: v1 requires n=20 qualification; v2 performs no separate task sample but requires all production preflight checks and the same in-run breaker.
- [ ] Run RED/GREEN:

```bash
python3 -m pytest -q tests/test_warpcore_v2.py tests/test_publish_swebench_score.py tests/test_qwen35_campaign_evidence.py
```

- [ ] Run the real publisher into a scratch path and compare canonical keys/values against the committed matrix; no unrelated historical cell may disappear or change.
- [ ] Commit:

```bash
git add viz/validate_campaign.py viz/publish_campaign.py tests AGENTS.md RUNBOOK.md PROVENANCE.md
git commit -m "feat: validate and publish warpcore-v2 campaigns"
```

### Task 12: Full verification, exact-SHA review, PR, CI, and cleanup

- [ ] Remove generated test residue from suite-owned directories, then run fresh:

```bash
python3 -m pytest -q
make check
make check-artifacts
make ci
python3 viz/validate_suite.py suite/warpcore-v1.yaml --repo .
python3 viz/validate_suite.py suite/warpcore-v2.yaml --repo .
git diff --check origin/main...HEAD
git status --short
```

- [ ] Run publication negative controls for Qwen3.5 diagnostic evidence and malformed/incomplete v2 fixtures.
- [ ] Run Gitleaks against the worktree and all commits reachable from the candidate branch; report only pass/fail and detector metadata, never secret values.
- [ ] Commit any final verified corrections, then record candidate SHA.
- [ ] Dispatch two independent exact-SHA reviews:
  1. specification/scientific-integrity review;
  2. adversarial code/security/compatibility review.
- [ ] Treat any Critical/Important finding as blocking. Repair through RED/GREEN, rerun all gates, commit a new SHA, and repeat both exact-SHA reviews.
- [ ] Push using the `jtchilders-ai-assistant` GitHub identity and open a PR describing both increments, semantic correction, tests, risk, and explicit non-promotion of Qwen3.5.
- [ ] Read the PR back and verify base, head branch, exact head SHA, file list, and body.
- [ ] Wait for all GitHub checks attached to that exact SHA and inspect any failures; never claim green from local evidence alone.
- [ ] Squash merge only after exact-SHA approval and green CI.
- [ ] Verify provider merge SHA, fetch, and prove local `main == origin/main == GitHub main`.
- [ ] Remove the feature worktree and delete superseded local/remote feature branches only after merge verification.

## Final acceptance matrix

- [ ] README values are artifact-derived: Ornith 91/73, Laguna 65/55, Nemotron 98/51, Qwen3.6 89/57, Qwen3.5 76/57 (submitted/resolved out of 100).
- [ ] Any headline percentage equals resolved/100.
- [ ] Nonempty grading-error patches count as submitted but not resolved.
- [ ] Historical source limitations are explicit; no missing evidence is reconstructed.
- [ ] Qwen3.5 remains diagnostic/noncanonical and excluded from the canonical matrix.
- [ ] Frozen v1 still requires a valid qualification seal.
- [ ] V2 requires explicit direct-n100 authorization, complete preflight, and the existing breaker; no generic bypass exists.
- [ ] Complete v2 evidence reconciles all 100 IDs before publication.
- [ ] Full tests, CI-equivalent target, validators, artifact audit, generated-output check, whitespace check, and secret scan pass on the exact reviewed SHA.
- [ ] Remote CI is green and local/remote/provider SHAs agree before cleanup.
