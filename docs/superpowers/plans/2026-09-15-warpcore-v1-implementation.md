# Warpcore v1 Comparison Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the minimum executable `warpcore-v1` contract required to resume canonical model testing without creating another generation of incomparable artifacts.

**Architecture:** Add immutable suite inputs and strict JSON Schemas, validate those inputs with a small standard-library-first Python module, and wrap the existing proven `viz/` preflight and harness scripts rather than replacing them. New runs use normalized run directories and transactional state records; old `results/<model>/raw/...` artifacts remain readable and untouched. Publication is a separate fail-closed step that consumes only validated `current` runs.

**Tech Stack:** Python 3.9+, PyYAML 6, jsonschema 4, pytest, GNU Make, GitHub Actions, Bash launch wrappers, lm-eval, mini-swe-agent/SWE-bench.

**Authoritative design:** `docs/superpowers/specs/2026-09-15-warpcore-v1-apples-to-apples-design.md`

---

## Delivery discipline

Each task is a separately reviewed increment. For every task:

1. Write the failing test first and run it to observe the expected failure.
2. Implement only enough production behavior to make the test pass.
3. Run the task tests, then `/usr/bin/python3 -m pytest -q`, then `make ci` with the pinned interpreter environment.
4. Run `git diff --check` and inspect the complete diff.
5. Commit, review, push to `main`, wait for both GitHub workflows, and verify local `main` = `origin/main` = GitHub `main` before starting the next task.

Do not launch a full benchmark campaign until Tasks 1–7 are complete.

---

### Task 1: Freeze canonical suite inputs and schemas

**Files:**
- Create: `suite/warpcore-v1.yaml`
- Create: `suite/tasks/gpqa_diamond_clean_v3.yaml`
- Create: `suite/tasks/gpqa_utils.py`
- Create: `suite/tasks/gsm8k_clean_v1.yaml`
- Create: `suite/swebench/instances-seed42-n100.json`
- Create: `suite/swebench/scaffold.yaml`
- Create: `suite/schemas/suite.schema.json`
- Create: `suite/schemas/adapter.schema.json`
- Create: `suite/schemas/manifest.schema.json`
- Create: `suite/schemas/result-status.schema.json`
- Create: `tests/test_suite_contract.py`

- [ ] **Step 1: Write failing fixture/schema tests**

Tests must assert:

```python
assert suite["suite_id"] == "warpcore-v1"
assert suite["benchmarks"]["gsm8k"]["generation_ceiling"] == 8192
assert suite["benchmarks"]["ifeval"]["generation_ceiling"] == 65536
assert suite["benchmarks"]["gpqa_diamond"]["generation_ceiling"] == 65536
assert len(instances) == 100
assert len(set(instances)) == 100
assert declared_sha256 == sha256(file_bytes).hexdigest()
```

Also validate one minimal valid document and reject one invalid document against each JSON Schema.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
/usr/bin/python3 -m pytest tests/test_suite_contract.py -q
```

Expected: failure because `suite/warpcore-v1.yaml` and the schemas do not exist.

- [ ] **Step 3: Copy—not reinterpret—the approved canonical inputs**

Use the exact retained sources:

- GPQA task and utility from `results/qwen3.6-35b-a3b/raw/quality/gpqa_thinking_64k_2026-09-09/`.
- GSM8K task from `results/qwen3.6-35b-a3b/raw/gsm8k_cot_zeroshot_clean.yaml` after comparing it against the other retained variant and documenting why this variant is canonical.
- SWE-bench scaffold from `results/qwen3.6-35b-a3b/raw/swebench/swebench_qwen36_rerun_config.yaml`, replacing the model name with a runner-injected value and retaining the robust `git add -A && git diff --cached` submission protocol.
- Exact instance IDs from the keys of `results/qwen3.6-35b-a3b/raw/swebench/preds_shuffle100.json`, sorted into a JSON array only after confirming they equal the seed-42 n=100 set used by the existing campaign.

The suite file must contain explicit hashes, expected counts, fixed sampling, endpoints/workload shapes, retry and timeout policy, and the statistical policy. It must not contain mutable aliases where an immutable revision is available.

- [ ] **Step 4: Define strict schemas**

All object schemas use `"additionalProperties": false`. The adapter schema permits model/serving fields only; attempts to add `generation_ceiling`, task selection, sampling, scoring, instance IDs, or denominator fields must fail. The status schema separates execution state from lifecycle state. The manifest schema requires suite ID, suite-input hashes, adapter hash, model revision, image digest, effective serving arguments, item inventory, timing, and artifact inventory.

- [ ] **Step 5: Run GREEN verification**

```bash
/usr/bin/python3 -m pytest tests/test_suite_contract.py -q
```

Expected: all Task 1 tests pass.

- [ ] **Step 6: Commit**

```bash
git add suite tests/test_suite_contract.py
git commit -m "feat: freeze warpcore-v1 suite contract"
```

---

### Task 2: Implement strict contract validation and CI drift enforcement

**Files:**
- Create: `viz/contract.py`
- Create: `viz/validate_suite.py`
- Create: `tests/test_validate_suite.py`
- Modify: `Makefile`
- Modify: `.github/workflows/provenance.yml`

- [ ] **Step 1: Write failing tests for valid and invalid contracts**

Cover:

```python
validate_suite(valid_suite) == []
```

and failures for:

- stale task hash;
- changed canonical file without a suite-version/hash update;
- duplicate or missing SWE-bench IDs;
- expected count mismatch;
- unknown suite keys;
- adapter containing `generation_ceiling` or another experiment-level key;
- relative path escaping the repository;
- missing required schema.

Use temporary directories and real files; do not mock hashing or schema validation.

- [ ] **Step 2: Verify RED**

```bash
/usr/bin/python3 -m pytest tests/test_validate_suite.py -q
```

Expected: import or behavior failures because the validator is absent.

- [ ] **Step 3: Implement the validator**

`viz/contract.py` provides focused helpers:

```python
def load_yaml(path: Path) -> dict: ...
def sha256_file(path: Path) -> str: ...
def validate_json(instance: dict, schema_path: Path) -> list[str]: ...
def validate_suite(repo: Path, suite_path: Path) -> list[str]: ...
def validate_adapter(repo: Path, adapter_path: Path) -> list[str]: ...
```

`viz/validate_suite.py` is the CLI. It prints every error and exits 1 for a diagnosed defect, 2 for unreadable/inconclusive input, and 0 only when all checks pass.

- [ ] **Step 4: Prove negative controls**

Copy the suite to a temporary tree, mutate one byte in a canonical task, and verify exit 1. Restore the fixture and verify exit 0. Also inject a prohibited adapter key and verify exit 1.

- [ ] **Step 5: Wire enforcement into local and remote CI**

Add a `contract` Make target and make `ci` depend on it. Add a named GitHub Actions step invoking `make contract`; do not create a second command path with different semantics.

- [ ] **Step 6: Verify and commit**

```bash
/usr/bin/python3 -m pytest tests/test_validate_suite.py -q
env -u VIRTUAL_ENV PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin make ci
git add viz/contract.py viz/validate_suite.py tests/test_validate_suite.py Makefile .github/workflows/provenance.yml
git commit -m "feat: enforce warpcore-v1 contract in CI"
```

---

### Task 3: Add and validate initial serving adapters

**Files:**
- Create: `adapters/gpt-oss-120b.yaml`
- Create: `adapters/qwen3.6-35b-a3b.yaml`
- Create: `tests/test_adapters.py`
- Modify: `viz/validate_suite.py`

- [ ] **Step 1: Write failing adapter tests**

Assert both adapters validate, slugs match result-directory names, required identities are not `unrecorded`, image references include a digest, and `max_model_len` can hold each enabled benchmark’s prompt-plus-output ceiling. Add negative cases for mutable image-only tags, model slug aliases, missing checkpoint revision, and forbidden experiment keys.

- [ ] **Step 2: Verify RED**

```bash
/usr/bin/python3 -m pytest tests/test_adapters.py -q
```

- [ ] **Step 3: Populate adapters from observed evidence**

Read the retained manifests and launch scripts. Never guess missing revisions or image digests. If an immutable value cannot be verified from repository or live Warpcore evidence, keep the adapter noncanonical and make validation fail loudly with the missing field named; resolve the fact before enabling a campaign.

- [ ] **Step 4: Add cross-validation**

The validator must reject adapters whose configured context cannot accommodate the tokenized task prompt plus suite output ceiling. It must also reject duplicate model slugs and unsupported adapter schema versions.

- [ ] **Step 5: Verify and commit**

```bash
/usr/bin/python3 -m pytest tests/test_adapters.py tests/test_validate_suite.py -q
env -u VIRTUAL_ENV PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin make ci
git add adapters tests/test_adapters.py viz/validate_suite.py
git commit -m "feat: add initial warpcore serving adapters"
```

---

### Task 4: Add transactional run-state and manifest creation

**Files:**
- Create: `viz/campaign_state.py`
- Create: `viz/create_campaign.py`
- Create: `tests/test_campaign_state.py`
- Modify: `viz/manifest_scaffold.py`

- [ ] **Step 1: Write failing state-machine tests**

Cover legal transitions:

```text
planned -> preflight_passed -> running -> completed -> validated -> published
running -> failed
```

Reject skipped stages, state rewrites, malformed timestamps, duplicate terminal transitions, and publication when lifecycle is not `current`. Verify that execution state and lifecycle are separate fields and failed execution maps to lifecycle `invalid` unless explicitly `diagnostic`.

- [ ] **Step 2: Verify RED**

```bash
/usr/bin/python3 -m pytest tests/test_campaign_state.py -q
```

- [ ] **Step 3: Implement atomic creation and append-only transitions**

Create normalized directories at:

```text
results/<model>/runs/<suite-version>/<benchmark>/<run-id>/
```

Write JSON through a sibling temporary file followed by `os.replace`. `create_campaign.py` resolves and records all suite/adapter hashes before any harness command is generated. Refuse an existing output directory unless an explicit resume path proves identity and inventory equality.

- [ ] **Step 4: Preserve legacy behavior**

Do not break `make manifest MODEL=... BENCH=...` for historical paths. Factor shared manifest fields into helpers while keeping the legacy CLI operational and tested.

- [ ] **Step 5: Verify and commit**

```bash
/usr/bin/python3 -m pytest tests/test_campaign_state.py -q
/usr/bin/python3 -m pytest -q
env -u VIRTUAL_ENV PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin make ci
git add viz/campaign_state.py viz/create_campaign.py viz/manifest_scaffold.py tests/test_campaign_state.py
git commit -m "feat: add transactional benchmark campaign records"
```

---

### Task 5: Build the contract-aware quality runner

**Files:**
- Create: `viz/run_quality.py`
- Create: `tests/test_run_quality.py`
- Modify: `viz/quality_preflight.py`
- Modify: `Makefile`

- [ ] **Step 1: Write failing command-construction tests**

Given suite, adapter, benchmark, endpoint, throughput, concurrency, and timeout, assert the generated command:

- uses the canonical task path;
- uses the suite generation ceiling and temperature;
- includes `--log_samples`;
- points at the normalized run directory;
- does not accept CLI overrides for task, ceiling, sampling, scoring, or instances;
- runs the existing quality preflight before the harness;
- does not create `DONE` on nonzero harness exit;
- refuses a long launch outside `/usr/bin/screen`, except in `--dry-run` or test mode.

- [ ] **Step 2: Verify RED**

```bash
/usr/bin/python3 -m pytest tests/test_run_quality.py -q
```

- [ ] **Step 3: Implement with dependency-injected process execution**

Do not duplicate `preflight_serving.py`, `check_output_budget.py`, or timeout arithmetic. Call the existing `QualityPreflightGate`, then transition state. Write the exact argv to `command.txt` with shell-safe quoting. On success, verify required raw files and then write `DONE`; on failure, record the exit code and transition to `failed`.

- [ ] **Step 4: Add Make entry point and negative integration tests**

Add `make run-quality SUITE=... ADAPTER=... BENCH=...` with explicit required variables. A dry-run test must inspect the exact generated command without network/GPU work.

- [ ] **Step 5: Verify and commit**

```bash
/usr/bin/python3 -m pytest tests/test_run_quality.py tests/test_quality_preflight.py tests/test_quality_preflight_make_probe.py -q
env -u VIRTUAL_ENV PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin make ci
git add viz/run_quality.py viz/quality_preflight.py tests/test_run_quality.py Makefile
git commit -m "feat: add contract-aware quality runner"
```

---

### Task 6: Build the contract-aware SWE-bench runner

**Files:**
- Create: `viz/run_swebench.py`
- Create: `tests/test_run_swebench.py`
- Modify: `viz/swebench_preflight.py`
- Modify: `Makefile`

- [ ] **Step 1: Write failing tests**

Assert the runner:

- uses exactly the 100 IDs in the suite artifact;
- verifies image cache, model identity, scaffold hash, submit protocol, and x86 Docker host;
- injects only the adapter model ID into the fixed scaffold;
- preserves fixed step/cost/request/pull/wall-time limits;
- writes into a normalized stable run directory;
- refuses launch outside `/usr/bin/screen`, except for dry-run/tests;
- resumes only when existing instance IDs and hashes match;
- records all 100 terminal dispositions.

- [ ] **Step 2: Verify RED**

```bash
/usr/bin/python3 -m pytest tests/test_run_swebench.py -q
```

- [ ] **Step 3: Implement the runner by wrapping proven tools**

Reuse `viz/swebench_preflight.py` and the robust existing Qwen rerun command semantics. Do not copy a model-specific script. Separate generation from grading in the state history, preserve trajectories, and only mark `completed` after grading artifacts and dispositions exist.

- [ ] **Step 4: Verify and commit**

```bash
/usr/bin/python3 -m pytest tests/test_run_swebench.py -q
env -u VIRTUAL_ENV PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin make ci
git add viz/run_swebench.py viz/swebench_preflight.py tests/test_run_swebench.py Makefile
git commit -m "feat: add contract-aware swebench runner"
```

---

### Task 7: Validate campaigns and generate canonical publication data

**Files:**
- Create: `viz/validate_campaign.py`
- Create: `viz/publish_campaign.py`
- Create: `tests/test_validate_campaign.py`
- Create: `tests/test_publish_campaign.py`
- Modify: `viz/validate_samples.py`
- Modify: `viz/audit_provenance.py`
- Modify: `viz/collect_matrix.py`
- Modify: `Makefile`

- [ ] **Step 1: Write failing post-run validation tests**

Use compact synthetic fixtures to reject:

- missing/duplicate item IDs;
- missing raw samples or manifest fields;
- aggregate mismatch with per-item evidence;
- unclassified empty, timeout, transport, parser, or SWE-bench failure;
- stale suite/adapter hashes;
- forbidden effective override;
- lifecycle other than `current` during publication;
- secret-scan failure;
- `DONE` without successful harness exit and complete artifacts.

Also prove historical fixtures remain visible without being subject to current-run hard gates.

- [ ] **Step 2: Verify RED**

```bash
/usr/bin/python3 -m pytest tests/test_validate_campaign.py tests/test_publish_campaign.py -q
```

- [ ] **Step 3: Implement lifecycle-aware readers**

Update discovery once so `validate_samples.py`, `audit_provenance.py`, and `collect_matrix.py` understand both historical and normalized layouts. Current v1 runs fail closed; historical debt stays in the ratchet and cannot become canonical implicitly.

- [ ] **Step 4: Implement transactional publication**

`publish_campaign.py` reads validated manifests and produces canonical matrix data. It emits explicit `not measured` cells, never imputes values, never computes a cross-task overall rank, and emits paired comparisons only for identical item sets.

- [ ] **Step 5: Prove publication is non-vacuous**

A valid current fixture must appear. Mutating it to `historical`, changing one task hash, deleting one item, or changing one instance ID must remove/block it. Restore and verify it reappears.

- [ ] **Step 6: Verify and commit**

```bash
/usr/bin/python3 -m pytest tests/test_validate_campaign.py tests/test_publish_campaign.py -q
/usr/bin/python3 -m pytest -q
env -u VIRTUAL_ENV PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin make ci
git add viz/validate_campaign.py viz/publish_campaign.py viz/validate_samples.py viz/audit_provenance.py viz/collect_matrix.py tests/test_validate_campaign.py tests/test_publish_campaign.py Makefile
git commit -m "feat: gate and publish canonical campaigns"
```

---

### Task 8: Register historical artifacts without rewriting evidence

**Files:**
- Create: `results/registry.json`
- Create: `tests/test_result_registry.py`
- Modify: `viz/audit_provenance.py`
- Modify: `viz/collect_matrix.py`
- Modify: `README.md`
- Modify: `PROVENANCE.md`
- Modify: `RUNBOOK.md`

- [ ] **Step 1: Write failing registry tests**

Assert every existing published source referenced by `viz/collect_matrix.py` has exactly one registry entry and status. Reject unknown paths, duplicate IDs, current status without v1 validation, and missing supersession/base-run links for replay composites.

- [ ] **Step 2: Verify RED**

```bash
/usr/bin/python3 -m pytest tests/test_result_registry.py -q
```

- [ ] **Step 3: Classify from evidence**

Populate statuses mechanically where evidence proves them; use `historical` or `invalid` rather than inferring missing provenance. Do not move, edit, or synthesize raw historical artifacts. Preserve the existing 14-gap ratchet and explain how it coexists with hard gates on new `current` runs.

- [ ] **Step 4: Update operator documentation**

Make `RUNBOOK.md` point to contract runners for new campaigns, preserve legacy forensic instructions, and state that direct model-specific scripts cannot create canonical results. Update `README.md` to distinguish the generated canonical v1 matrix from legacy evidence.

- [ ] **Step 5: Verify and commit**

```bash
/usr/bin/python3 -m pytest tests/test_result_registry.py -q
env -u VIRTUAL_ENV PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin make ci
git add results/registry.json tests/test_result_registry.py viz/audit_provenance.py viz/collect_matrix.py README.md PROVENANCE.md RUNBOOK.md
git commit -m "docs: classify historical benchmark evidence"
```

---

### Task 9: Execute closure-run readiness checks

**Files:**
- Modify only if a proven defect is found in Tasks 1–8.
- Produce local dry-run/preflight output; do not commit generated credentials or transient logs.

- [ ] **Step 1: Verify actual host and endpoint state**

```bash
hostname
uname -m
curl -fsS --max-time 10 http://csi370295.alcf.anl.gov:8000/v1/models
ssh -o BatchMode=yes -o ConnectTimeout=8 warpcore 'hostname; uname -m; docker ps --format "{{.Names}} {{.Image}} {{.Status}}"; curl -fsS --max-time 10 http://localhost:8000/v1/models'
```

Use SSH alias `warpcore`, not the FQDN. Do not change the live server merely to satisfy a stale adapter.

- [ ] **Step 2: Exercise both adapters without launching full campaigns**

Run contract validation, dry-run command generation, and live preflight for the model actually served. If the other model is not served, report it as unavailable; do not swap models until the completed increment is committed and the next closure campaign is selected.

- [ ] **Step 3: Run full repository verification**

```bash
git diff --check
/usr/bin/python3 -m pytest -q
env -u VIRTUAL_ENV PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin make ci
```

- [ ] **Step 4: Final independent review**

Review the complete implementation against all eleven design acceptance criteria. Any unmet criterion keeps P1 open.

- [ ] **Step 5: Commit readiness fixes, push, and verify remote state**

After the final reviewed increment lands on `main`, verify local SHA, `origin/main`, GitHub `main`, provenance workflow, and secret-scan workflow all agree and pass.

---

## After P1

Run closure campaigns sequentially because Warpcore serves one model at a time:

1. Qwen3.6 SWE-bench n=100 rerun.
2. gpt-oss SWE-bench n=100 rerun.
3. gpt-oss clean GSM8K rerun.

Each campaign must pass live preflight, complete under durable execution, validate offline, and publish transactionally before the next result is treated as canonical.
