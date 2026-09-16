"""tests/test_task7_submitted_and_scoring_provenance.py

Strict-TDD tests for two repair areas:

  S1  Submitted-count reconciliation (SWE authoritative validator)
      manifest.item_inventory.submitted must exist and equal:
        • expected (from manifest.item_inventory.expected)
        • len(frozen_ids)
      Missing, bool, non-int, lower, and higher values all fail.
      Preds, statuses, grading dispositions, and trajectories must still
      each reconcile exactly with frozen IDs.

  S2  Quality per_item submitted validation
      manifest.item_inventory.submitted must exist as a non-bool int and
      equal both the unique item rows in per_item.csv and expected.
      Preserve duplicate/missing/foreign rejection (existing gates).

  S3  Benchmark-aware scoring_implementation provenance
      required_evidence 'scoring_implementation' follows the actual frozen
      suite logic:
        • If utils_file exists (GPQA):  utils_file path must be in
          manifest.suite_input_hashes.
        • If no utils_file but task_file exists (GSM8K, IFEval in current
          suite): task_file hash entry satisfies scorer provenance.
        • If neither: require suite.required_harness.lm_eval_revision and
          verify that exact revision is represented (not just prose).
      Positive cases: current suite GSM8K (task_file only), IFEval (task_file
      only), GPQA (utils_file).  Negative mutations: wrong path, missing
      entry, absent both files.

SCOPE: Only viz/validate_campaign.py — minimum relevant blocks.
Do NOT touch completion_token_counts, finish_reasons, full_response_fields.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import pathlib
import sys
import tempfile
import unittest

_TESTS_DIR = pathlib.Path(__file__).parent
_REPO_ROOT = _TESTS_DIR.parent
_VIZ_DIR = _REPO_ROOT / "viz"
for _p in (str(_TESTS_DIR), str(_VIZ_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import validate_campaign  # noqa: E402

# ---------------------------------------------------------------------------
# Shared suite definitions (mirrors the actual warpcore-v1.yaml structure)
# ---------------------------------------------------------------------------

_FAKE_FROZEN_IDS = [f"django__django-{i:04d}" for i in range(1, 6)]  # 5 IDs
_N_FROZEN = len(_FAKE_FROZEN_IDS)

# Suite YAML content for SWE-bench tests (minimal + frozen instance set ref)
_SWE_SUITE_CONTENT_TEMPLATE = """\
suite_id: warpcore-v1
suite_schema_version: 1
benchmarks:
  swebench:
    description: "SWE-bench test"
    instance_set_file: "suite/swebench/instances-test.json"
    instances_sha256: "{sha256}"
    expected_item_count: {n}
    required_evidence:
      - "preds_json"
      - "exit_statuses"
      - "run_log"
      - "command_txt"
      - "manifest_json"
      - "status_json"
      - "done_sentinel"
"""

# Suite YAML for GSM8K tests (task_file only, no utils_file)
_GSM8K_SUITE_CONTENT_TEMPLATE = """\
suite_id: warpcore-v1
suite_schema_version: 1
benchmarks:
  gsm8k:
    description: "GSM8K test"
    task_file: "suite/tasks/gsm8k_clean_v1.yaml"
    task_sha256: "{task_sha256}"
    expected_item_count: 5
    required_evidence:
      - "task_yaml"
      - "scoring_implementation"
      - "aggregate_result"
      - "samples_jsonl_gz"
      - "per_item_csv"
      - "run_log"
      - "command_txt"
      - "manifest_json"
      - "status_json"
      - "done_sentinel"
"""

# Suite YAML for IFEval tests (task_file only, no utils_file)
_IFEVAL_SUITE_CONTENT_TEMPLATE = """\
suite_id: warpcore-v1
suite_schema_version: 1
required_harness:
  lm_eval_revision: "6d642546f4688648fced259eb3302efd36ece5af"
benchmarks:
  ifeval:
    description: "IFEval test"
    task_file: "suite/tasks/ifeval_v4.yaml"
    task_sha256: "{task_sha256}"
    expected_item_count: 5
    required_evidence:
      - "task_yaml"
      - "scoring_implementation"
      - "aggregate_result"
      - "samples_jsonl_gz"
      - "per_item_csv"
      - "run_log"
      - "command_txt"
      - "manifest_json"
      - "status_json"
      - "done_sentinel"
"""

# Suite YAML for GPQA tests (has utils_file)
_GPQA_SUITE_CONTENT_TEMPLATE = """\
suite_id: warpcore-v1
suite_schema_version: 1
benchmarks:
  gpqa_diamond:
    description: "GPQA Diamond test"
    task_file: "suite/tasks/gpqa_diamond_clean_v3.yaml"
    task_sha256: "{task_sha256}"
    utils_file: "suite/tasks/gpqa_utils.py"
    utils_sha256: "{utils_sha256}"
    expected_item_count: 5
    required_evidence:
      - "task_yaml"
      - "scoring_implementation"
      - "aggregate_result"
      - "samples_jsonl_gz"
      - "per_item_csv"
      - "run_log"
      - "command_txt"
      - "manifest_json"
      - "status_json"
      - "done_sentinel"
"""


# ---------------------------------------------------------------------------
# Helper: write frozen instance set JSON
# ---------------------------------------------------------------------------

def _write_frozen_ids(path: pathlib.Path, ids: list) -> str:
    """Write the frozen IDs JSON and return its SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(ids)
    path.write_text(data)
    return hashlib.sha256(data.encode()).hexdigest()


def _make_swe_suite(repo: pathlib.Path, ids: list | None = None) -> tuple:
    """Create suite YAML + frozen instance file, return (suite_path, ids)."""
    ids = ids or _FAKE_FROZEN_IDS
    instances_path = repo / "suite" / "swebench" / "instances-test.json"
    sha256 = _write_frozen_ids(instances_path, ids)

    suite_content = _SWE_SUITE_CONTENT_TEMPLATE.format(sha256=sha256, n=len(ids))
    suite_path = repo / "suite" / "warpcore-v1.yaml"
    suite_path.parent.mkdir(parents=True, exist_ok=True)
    suite_path.write_text(suite_content)
    return suite_path, ids


def _make_adapter(repo: pathlib.Path, slug: str = "test-model") -> pathlib.Path:
    path = repo / "adapters" / f"{slug}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("adapter_schema_version: 1\ncampaign_status: canonical\n")
    return path


def _make_swe_run(
    repo: pathlib.Path,
    suite_path: pathlib.Path,
    ids: list,
    *,
    submitted: object = None,  # sentinel: use len(ids) by default
    slug: str = "swe-model",
    include_grading: bool = True,
    extra_manifest: dict | None = None,
) -> pathlib.Path:
    """Build a minimal valid SWE-bench run directory."""
    adapter_path = _make_adapter(repo, slug)
    adapter_hash = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
    suite_hash = hashlib.sha256(suite_path.read_bytes()).hexdigest()

    run_id = "run-2026-swe-test"
    run_dir = (
        repo / "results" / slug / "runs" / "warpcore-v1" / "swebench" / run_id
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    _submitted = len(ids) if submitted is None else submitted

    manifest = {
        "schema_version": 1,
        "suite_id": "warpcore-v1",
        "suite_schema_version": 1,
        "run_id": run_id,
        "benchmark": "swebench",
        "adapter_hash": adapter_hash,
        "suite_input_hashes": {"suite/warpcore-v1.yaml": suite_hash},
        "serving_profile_digest": "sha256:" + "c" * 64,
        "model": {"slug": slug, "id": f"org/{slug}", "revision": "a" * 40},
        "serving": {
            "image_digest": "sha256:" + "d" * 64,
            "engine": "vllm",
            "engine_version": "0.6.6",
            "effective_args": [],
            "environment": {},
            "hardware_id": "dgx-spark-gb10",
        },
        "item_inventory": {
            "expected": len(ids),
            "submitted": _submitted,
            "instance_ids_hash": hashlib.sha256(
                json.dumps(sorted(ids), sort_keys=True).encode()
            ).hexdigest(),
        },
        "timing": {
            "started_utc": "2026-09-15T12:00:00Z",
            "completed_utc": "2026-09-15T14:00:00Z",
        },
        "artifact_inventory": {
            "preds_json": True,
            "exit_statuses_json": True,
            "grading_results_json": include_grading,
            "run_log": True,
            "command_txt": True,
            "done_sentinel": True,
            "samples_jsonl_gz": False,
            "per_item_csv": False,
        },
    }
    if extra_manifest:
        manifest.update(extra_manifest)

    status = {
        "schema_version": 1,
        "run_id": run_id,
        "suite_id": "warpcore-v1",
        "execution_state": "completed",
        "lifecycle": "current",
        "history": [
            {"state": "planned", "timestamp": "2026-09-15T12:00:00Z"},
            {"state": "preflight_passed", "timestamp": "2026-09-15T13:00:00Z"},
            {"state": "running", "timestamp": "2026-09-15T14:00:00Z"},
            {"state": "completed", "timestamp": "2026-09-15T15:00:00Z"},
        ],
    }

    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (run_dir / "status.json").write_text(json.dumps(status, indent=2))
    (run_dir / "DONE").write_text("done\n")
    (run_dir / "command.txt").write_text("python3 run_swebench.py\n")

    raw = run_dir / "raw"
    raw.mkdir(exist_ok=True)
    (raw / "run.log").write_text("exit 0\n")
    (raw / "preds.json").write_text(
        json.dumps({iid: {"model_patch": f"diff {iid}"} for iid in ids})
    )
    (raw / "exit_statuses.json").write_text(
        json.dumps({iid: "submitted" for iid in ids})
    )

    if include_grading:
        grading = {
            "resolved_ids": [],
            "unresolved_ids": ids,
            "empty_patch_ids": [],
            "error_ids": [],
            "incomplete_ids": [],
        }
        (raw / "grading_results.json").write_text(json.dumps(grading))

    traj_dir = raw / "trajectories"
    traj_dir.mkdir(exist_ok=True)
    for iid in ids:
        (traj_dir / f"{iid}.traj").write_text(json.dumps({"steps": []}))

    return run_dir


# ---------------------------------------------------------------------------
# Helper: quality run builder
# ---------------------------------------------------------------------------

def _make_quality_suite(
    repo: pathlib.Path,
    benchmark: str,
    *,
    task_file_rel: str,
    utils_file_rel: str | None = None,
    with_harness_revision: bool = False,
) -> tuple:
    """
    Create suite YAML + task/utils files; return (suite_path, task_sha256, utils_sha256_or_None).
    """
    task_path = repo / task_file_rel
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_text(f"# fake task for {benchmark}\ntask: {benchmark}\n")
    task_sha256 = hashlib.sha256(task_path.read_bytes()).hexdigest()

    utils_sha256 = None
    if utils_file_rel:
        utils_path = repo / utils_file_rel
        utils_path.parent.mkdir(parents=True, exist_ok=True)
        utils_path.write_text(f"# fake utils for {benchmark}\n")
        utils_sha256 = hashlib.sha256(utils_path.read_bytes()).hexdigest()

    suite_path = repo / "suite" / "warpcore-v1.yaml"
    suite_path.parent.mkdir(parents=True, exist_ok=True)

    if benchmark == "gsm8k":
        content = _GSM8K_SUITE_CONTENT_TEMPLATE.format(task_sha256=task_sha256)
    elif benchmark == "ifeval":
        content = _IFEVAL_SUITE_CONTENT_TEMPLATE.format(task_sha256=task_sha256)
    elif benchmark == "gpqa_diamond":
        content = _GPQA_SUITE_CONTENT_TEMPLATE.format(
            task_sha256=task_sha256,
            utils_sha256=utils_sha256 or ("0" * 64),
        )
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    suite_path.write_text(content)
    return suite_path, task_sha256, utils_sha256


def _make_quality_run(
    repo: pathlib.Path,
    suite_path: pathlib.Path,
    benchmark: str,
    *,
    task_file_rel: str,
    utils_file_rel: str | None = None,
    task_sha256: str,
    utils_sha256: str | None = None,
    n_items: int = 5,
    submitted: object = None,
    slug: str = "qual-model",
    extra_suite_input_hashes: dict | None = None,
) -> pathlib.Path:
    """Build a minimal valid quality run directory."""
    adapter_path = _make_adapter(repo, slug)
    adapter_hash = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
    suite_hash = hashlib.sha256(suite_path.read_bytes()).hexdigest()

    run_id = "run-2026-qual-test"
    run_dir = (
        repo / "results" / slug / "runs" / "warpcore-v1" / benchmark / run_id
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    ids = [f"item-{i}" for i in range(n_items)]
    _submitted = n_items if submitted is None else submitted

    # Build suite_input_hashes: always include suite YAML + task file
    suite_input_hashes: dict[str, str] = {
        "suite/warpcore-v1.yaml": suite_hash,
        task_file_rel: task_sha256,
    }
    if utils_file_rel and utils_sha256:
        suite_input_hashes[utils_file_rel] = utils_sha256
    if extra_suite_input_hashes:
        suite_input_hashes.update(extra_suite_input_hashes)

    manifest = {
        "schema_version": 1,
        "suite_id": "warpcore-v1",
        "suite_schema_version": 1,
        "run_id": run_id,
        "benchmark": benchmark,
        "adapter_hash": adapter_hash,
        "suite_input_hashes": suite_input_hashes,
        "serving_profile_digest": "sha256:" + "c" * 64,
        "model": {"slug": slug, "id": f"org/{slug}", "revision": "a" * 40},
        "serving": {
            "image_digest": "sha256:" + "d" * 64,
            "engine": "vllm",
            "engine_version": "0.6.6",
            "effective_args": [],
            "environment": {},
            "hardware_id": "dgx-spark-gb10",
        },
        "item_inventory": {
            "expected": n_items,
            "submitted": _submitted,
            "instance_ids_hash": hashlib.sha256(
                json.dumps(sorted(ids), sort_keys=True).encode()
            ).hexdigest(),
        },
        "timing": {
            "started_utc": "2026-09-15T12:00:00Z",
            "completed_utc": "2026-09-15T14:00:00Z",
        },
        "artifact_inventory": {
            "samples_jsonl_gz": True,
            "per_item_csv": True,
            "run_log": True,
            "command_txt": True,
            "done_sentinel": True,
        },
    }

    status = {
        "schema_version": 1,
        "run_id": run_id,
        "suite_id": "warpcore-v1",
        "execution_state": "completed",
        "lifecycle": "current",
        "history": [
            {"state": "planned", "timestamp": "2026-09-15T12:00:00Z"},
            {"state": "preflight_passed", "timestamp": "2026-09-15T13:00:00Z"},
            {"state": "running", "timestamp": "2026-09-15T14:00:00Z"},
            {"state": "completed", "timestamp": "2026-09-15T15:00:00Z"},
        ],
    }

    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (run_dir / "status.json").write_text(json.dumps(status, indent=2))
    (run_dir / "DONE").write_text("done\n")
    (run_dir / "run.log").write_text("harness exit 0\n")
    (run_dir / "command.txt").write_text(f"python3 run_quality.py --task {benchmark}\n")

    raw = run_dir / "raw"
    raw.mkdir(exist_ok=True)

    # Write per_item.csv
    rows = [
        {
            "item_id": iid,
            "score": 1.0,
            "disposition": "correct",
            "finish_reason": "stop",
            "response_chars": 100,
            "empty_content": False,
        }
        for iid in ids
    ]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    (run_dir / "per_item.csv").write_text(buf.getvalue())

    # Write samples JSONL (with finish_reason for evidence)
    with gzip.open(raw / f"samples_{benchmark}.jsonl.gz", "wt") as fh:
        for row in rows:
            rec = {
                "item_id": row["item_id"],
                "finish_reason": "stop",
                "completion_tokens": 100,
                "resps": ["answer"],
                "filtered_resps": ["answer"],
            }
            fh.write(json.dumps(rec) + "\n")

    # Write aggregate result
    score_key = {
        "gsm8k": "exact_match,flexible-fallback",
        "ifeval": "prompt_level_strict_acc,none",
        "gpqa_diamond": "exact_match,answer-line",
    }.get(benchmark, "exact_match,flexible-fallback")
    (raw / f"results_{benchmark}.json").write_text(
        json.dumps({"results": {benchmark: {score_key: 1.0}}, "n_samples": n_items})
    )

    return run_dir


# ===========================================================================
# S1: Submitted-count reconciliation — SWE authoritative validator
# ===========================================================================

class TestS1SweSubmittedReconciliation(unittest.TestCase):
    """manifest.item_inventory.submitted must exist and equal expected and len(frozen_ids).

    The validator's _validate_swebench_frozen_ids function must enforce this.
    Missing, bool, non-int, lower, and higher values must all fail.
    Existing preds/statuses/grading/trajectories reconciliation must still pass.
    """

    def _run(self, submitted, ids=None):
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            suite_path, frozen_ids = _make_swe_suite(repo, ids)
            run_dir = _make_swe_run(
                repo, suite_path, frozen_ids, submitted=submitted
            )
            adapter_path = repo / "adapters" / "swe-model.yaml"
            return validate_campaign.validate(run_dir, suite_path, adapter_path)

    def test_submitted_equals_frozen_count_passes(self):
        """Positive: submitted == len(frozen_ids) == expected passes."""
        result = self._run(submitted=_N_FROZEN)
        errors_about_submitted = [
            e for e in result.errors if "submitted" in e.lower()
        ]
        self.assertEqual(
            errors_about_submitted, [],
            f"Valid submitted count should produce no submitted-related errors; "
            f"got: {errors_about_submitted}",
        )

    def test_submitted_missing_fails(self):
        """submitted key absent from item_inventory must fail."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            suite_path, frozen_ids = _make_swe_suite(repo)
            run_dir = _make_swe_run(
                repo, suite_path, frozen_ids, submitted=_N_FROZEN
            )
            # Remove submitted from manifest
            m = json.loads((run_dir / "manifest.json").read_text())
            del m["item_inventory"]["submitted"]
            (run_dir / "manifest.json").write_text(json.dumps(m))
            adapter_path = repo / "adapters" / "swe-model.yaml"

            result = validate_campaign.validate(run_dir, suite_path, adapter_path)
            self.assertFalse(result.passed, "Missing submitted must fail")
            submitted_errors = [e for e in result.errors if "submitted" in e.lower()]
            self.assertTrue(
                len(submitted_errors) > 0,
                f"Must report error for missing submitted; got errors: {result.errors}",
            )

    def test_submitted_bool_true_fails(self):
        """submitted=True (a bool, not int) must fail — bools are not valid counts."""
        result = self._run(submitted=True)
        self.assertFalse(result.passed, "Bool submitted must fail")
        submitted_errors = [e for e in result.errors if "submitted" in e.lower()]
        self.assertTrue(
            len(submitted_errors) > 0,
            f"Must report error for bool submitted=True; got: {result.errors}",
        )

    def test_submitted_bool_false_fails(self):
        """submitted=False (bool) must fail even if frozen count is 0."""
        result = self._run(submitted=False)
        self.assertFalse(result.passed, "Bool False submitted must fail")
        submitted_errors = [e for e in result.errors if "submitted" in e.lower()]
        self.assertTrue(
            len(submitted_errors) > 0,
            f"Must report error for bool submitted=False; got: {result.errors}",
        )

    def test_submitted_string_fails(self):
        """submitted='5' (string) must fail — only int is valid."""
        result = self._run(submitted="5")
        self.assertFalse(result.passed, "String submitted must fail")
        submitted_errors = [e for e in result.errors if "submitted" in e.lower()]
        self.assertTrue(
            len(submitted_errors) > 0,
            f"Must report error for string submitted; got: {result.errors}",
        )

    def test_submitted_lower_than_frozen_fails(self):
        """submitted < len(frozen_ids) must fail."""
        result = self._run(submitted=_N_FROZEN - 1)
        self.assertFalse(result.passed, "Low submitted must fail")
        submitted_errors = [e for e in result.errors if "submitted" in e.lower()]
        self.assertTrue(
            len(submitted_errors) > 0,
            f"Must report error for submitted < frozen count; got: {result.errors}",
        )

    def test_submitted_higher_than_frozen_fails(self):
        """submitted > len(frozen_ids) must fail."""
        result = self._run(submitted=_N_FROZEN + 1)
        self.assertFalse(result.passed, "High submitted must fail")
        submitted_errors = [e for e in result.errors if "submitted" in e.lower()]
        self.assertTrue(
            len(submitted_errors) > 0,
            f"Must report error for submitted > frozen count; got: {result.errors}",
        )

    def test_submitted_zero_with_nonempty_frozen_fails(self):
        """submitted=0 when frozen IDs is non-empty must fail."""
        result = self._run(submitted=0)
        self.assertFalse(result.passed, "submitted=0 with non-empty frozen must fail")
        submitted_errors = [e for e in result.errors if "submitted" in e.lower()]
        self.assertTrue(
            len(submitted_errors) > 0,
            f"Must report error for submitted=0 with nonempty frozen; got: {result.errors}",
        )

    def test_preds_still_reconcile_with_frozen_ids(self):
        """Existing preds.json reconciliation still fires: foreign ID in preds must fail."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            suite_path, frozen_ids = _make_swe_suite(repo)
            run_dir = _make_swe_run(
                repo, suite_path, frozen_ids, submitted=_N_FROZEN
            )
            raw = run_dir / "raw"
            # Inject a foreign ID into preds.json
            preds = {iid: {"model_patch": f"diff {iid}"} for iid in frozen_ids}
            preds["foreign__repo-9999"] = {"model_patch": "diff foreign"}
            (raw / "preds.json").write_text(json.dumps(preds))
            # Also update exit_statuses and traj so only preds has the foreign ID
            adapter_path = repo / "adapters" / "swe-model.yaml"

            result = validate_campaign.validate(run_dir, suite_path, adapter_path)
            self.assertFalse(result.passed, "Foreign pred ID must still fail")
            preds_errors = [e for e in result.errors if "preds" in e.lower() or "foreign" in e.lower()]
            self.assertTrue(
                len(preds_errors) > 0,
                f"Must report foreign pred ID error; got: {result.errors}",
            )


# ===========================================================================
# S2: Quality per_item submitted validation
# ===========================================================================

class TestS2QualitySubmittedValidation(unittest.TestCase):
    """manifest.item_inventory.submitted must be a non-bool int equal to
    unique item rows in per_item.csv and to expected.

    This checks _check_item_ids (quality per_item path).
    Preserve: duplicate/missing/foreign ID rejection.
    """

    def _run_gsm8k(self, submitted, n_items=5):
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            suite_path, task_sha256, _ = _make_quality_suite(
                repo, "gsm8k",
                task_file_rel="suite/tasks/gsm8k_clean_v1.yaml",
            )
            run_dir = _make_quality_run(
                repo, suite_path, "gsm8k",
                task_file_rel="suite/tasks/gsm8k_clean_v1.yaml",
                task_sha256=task_sha256,
                n_items=n_items,
                submitted=submitted,
            )
            adapter_path = repo / "adapters" / "qual-model.yaml"
            return validate_campaign.validate(run_dir, suite_path, adapter_path)

    def test_submitted_equals_unique_rows_passes(self):
        """Positive: submitted == n_items == len(unique per_item rows) passes."""
        result = self._run_gsm8k(submitted=5)
        submitted_errors = [e for e in result.errors if "submitted" in e.lower()]
        self.assertEqual(
            submitted_errors, [],
            f"Valid submitted=5 with 5 items should produce no submitted-related errors; "
            f"got: {submitted_errors}",
        )

    def test_submitted_missing_from_item_inventory_fails(self):
        """submitted absent from item_inventory must fail."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            suite_path, task_sha256, _ = _make_quality_suite(
                repo, "gsm8k",
                task_file_rel="suite/tasks/gsm8k_clean_v1.yaml",
            )
            run_dir = _make_quality_run(
                repo, suite_path, "gsm8k",
                task_file_rel="suite/tasks/gsm8k_clean_v1.yaml",
                task_sha256=task_sha256,
                n_items=5,
                submitted=5,
            )
            m = json.loads((run_dir / "manifest.json").read_text())
            del m["item_inventory"]["submitted"]
            (run_dir / "manifest.json").write_text(json.dumps(m))
            adapter_path = repo / "adapters" / "qual-model.yaml"

            result = validate_campaign.validate(run_dir, suite_path, adapter_path)
            self.assertFalse(result.passed, "Missing submitted must fail for quality run")
            submitted_errors = [e for e in result.errors if "submitted" in e.lower()]
            self.assertTrue(
                len(submitted_errors) > 0,
                f"Must report submitted-missing error; got: {result.errors}",
            )

    def test_submitted_bool_rejected_for_quality(self):
        """submitted=True (bool) must be rejected even if it equals 1."""
        result = self._run_gsm8k(submitted=True, n_items=1)
        self.assertFalse(result.passed, "Bool submitted must fail for quality run")
        submitted_errors = [e for e in result.errors if "submitted" in e.lower()]
        self.assertTrue(
            len(submitted_errors) > 0,
            f"Must report bool submitted error for quality run; got: {result.errors}",
        )

    def test_submitted_lower_than_row_count_fails(self):
        """submitted < unique per_item row count must fail."""
        result = self._run_gsm8k(submitted=3, n_items=5)
        self.assertFalse(result.passed, "submitted < row count must fail")
        submitted_errors = [e for e in result.errors if "submitted" in e.lower()]
        self.assertTrue(
            len(submitted_errors) > 0,
            f"Must report submitted < row-count error; got: {result.errors}",
        )

    def test_submitted_higher_than_row_count_fails(self):
        """submitted > unique per_item row count must fail."""
        result = self._run_gsm8k(submitted=7, n_items=5)
        self.assertFalse(result.passed, "submitted > row count must fail")
        submitted_errors = [e for e in result.errors if "submitted" in e.lower()]
        self.assertTrue(
            len(submitted_errors) > 0,
            f"Must report submitted > row-count error; got: {result.errors}",
        )

    def test_duplicate_item_id_still_rejected(self):
        """Duplicate item IDs in per_item.csv must still fail (existing gate preserved)."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            suite_path, task_sha256, _ = _make_quality_suite(
                repo, "gsm8k",
                task_file_rel="suite/tasks/gsm8k_clean_v1.yaml",
            )
            run_dir = _make_quality_run(
                repo, suite_path, "gsm8k",
                task_file_rel="suite/tasks/gsm8k_clean_v1.yaml",
                task_sha256=task_sha256,
                n_items=5,
                submitted=5,
            )
            # Inject duplicate row into per_item.csv
            per_item_path = run_dir / "per_item.csv"
            rows = list(csv.DictReader(io.StringIO(per_item_path.read_text())))
            rows.append(rows[0])  # duplicate first row
            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
            per_item_path.write_text(buf.getvalue())
            adapter_path = repo / "adapters" / "qual-model.yaml"

            result = validate_campaign.validate(run_dir, suite_path, adapter_path)
            self.assertFalse(result.passed, "Duplicate item ID must fail")
            dup_errors = [
                e for e in result.errors
                if "duplicate" in e.lower() or "item_id" in e.lower()
            ]
            self.assertTrue(
                len(dup_errors) > 0,
                f"Must report duplicate ID error; got: {result.errors}",
            )


# ===========================================================================
# S3: Benchmark-aware scoring_implementation provenance
# ===========================================================================

class TestS3ScoringImplementationProvenance(unittest.TestCase):
    """scoring_implementation evidence follows the actual frozen suite logic:

      GPQA:        utils_file exists → utils_file path in suite_input_hashes
      GSM8K:       no utils_file, task_file exists → task_file satisfies provenance
      IFEval:      no utils_file, task_file exists → task_file satisfies provenance
      Neither:     require required_harness.lm_eval_revision (and pin exists)

    Positive (current-suite cases) must pass; negative mutations must fail.
    """

    # ── GPQA positive: utils_file in suite_input_hashes ────────────────────

    def test_gpqa_utils_file_in_hashes_passes(self):
        """GPQA: utils_file present and its hash recorded → scoring provenance satisfied."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            suite_path, task_sha256, utils_sha256 = _make_quality_suite(
                repo, "gpqa_diamond",
                task_file_rel="suite/tasks/gpqa_diamond_clean_v3.yaml",
                utils_file_rel="suite/tasks/gpqa_utils.py",
            )
            run_dir = _make_quality_run(
                repo, suite_path, "gpqa_diamond",
                task_file_rel="suite/tasks/gpqa_diamond_clean_v3.yaml",
                utils_file_rel="suite/tasks/gpqa_utils.py",
                task_sha256=task_sha256,
                utils_sha256=utils_sha256,
            )
            adapter_path = repo / "adapters" / "qual-model.yaml"

            result = validate_campaign.validate(run_dir, suite_path, adapter_path)
            scoring_errors = [
                e for e in result.errors if "scoring_implementation" in e.lower()
            ]
            self.assertEqual(
                scoring_errors, [],
                f"GPQA utils_file in hashes should satisfy scoring provenance; "
                f"got: {scoring_errors}",
            )

    def test_gpqa_utils_file_missing_from_hashes_fails(self):
        """GPQA: utils_file declared in suite but NOT in manifest.suite_input_hashes → fail."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            suite_path, task_sha256, utils_sha256 = _make_quality_suite(
                repo, "gpqa_diamond",
                task_file_rel="suite/tasks/gpqa_diamond_clean_v3.yaml",
                utils_file_rel="suite/tasks/gpqa_utils.py",
            )
            run_dir = _make_quality_run(
                repo, suite_path, "gpqa_diamond",
                task_file_rel="suite/tasks/gpqa_diamond_clean_v3.yaml",
                utils_file_rel="suite/tasks/gpqa_utils.py",
                task_sha256=task_sha256,
                utils_sha256=utils_sha256,
            )
            # Remove utils_file from suite_input_hashes in manifest
            m = json.loads((run_dir / "manifest.json").read_text())
            hashes = m.get("suite_input_hashes", {})
            hashes.pop("suite/tasks/gpqa_utils.py", None)
            m["suite_input_hashes"] = hashes
            (run_dir / "manifest.json").write_text(json.dumps(m))
            adapter_path = repo / "adapters" / "qual-model.yaml"

            result = validate_campaign.validate(run_dir, suite_path, adapter_path)
            self.assertFalse(result.passed, "Missing utils hash must fail for GPQA")
            scoring_errors = [
                e for e in result.errors if "scoring_implementation" in e.lower()
            ]
            self.assertTrue(
                len(scoring_errors) > 0,
                f"Must report scoring_implementation error for missing utils hash; "
                f"got: {result.errors}",
            )

    # ── GSM8K positive: task_file satisfies scoring provenance (no utils_file) ──

    def test_gsm8k_task_file_satisfies_scoring_provenance(self):
        """GSM8K: no utils_file in suite, task_file in suite_input_hashes → scoring passes.

        RED: current code fails with 'utils_file absent' even when task_file is declared.
        GREEN: task_file in manifest.suite_input_hashes satisfies scoring_implementation.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            suite_path, task_sha256, _ = _make_quality_suite(
                repo, "gsm8k",
                task_file_rel="suite/tasks/gsm8k_clean_v1.yaml",
            )
            run_dir = _make_quality_run(
                repo, suite_path, "gsm8k",
                task_file_rel="suite/tasks/gsm8k_clean_v1.yaml",
                task_sha256=task_sha256,
            )
            adapter_path = repo / "adapters" / "qual-model.yaml"

            result = validate_campaign.validate(run_dir, suite_path, adapter_path)
            scoring_errors = [
                e for e in result.errors if "scoring_implementation" in e.lower()
            ]
            self.assertEqual(
                scoring_errors, [],
                f"GSM8K task_file in hashes should satisfy scoring_implementation; "
                f"scoring_errors: {scoring_errors}\nall_errors: {result.errors}",
            )

    def test_gsm8k_task_file_missing_from_hashes_fails(self):
        """GSM8K: task_file declared but NOT in manifest.suite_input_hashes → fail."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            suite_path, task_sha256, _ = _make_quality_suite(
                repo, "gsm8k",
                task_file_rel="suite/tasks/gsm8k_clean_v1.yaml",
            )
            run_dir = _make_quality_run(
                repo, suite_path, "gsm8k",
                task_file_rel="suite/tasks/gsm8k_clean_v1.yaml",
                task_sha256=task_sha256,
            )
            # Remove task_file from suite_input_hashes
            m = json.loads((run_dir / "manifest.json").read_text())
            hashes = m.get("suite_input_hashes", {})
            hashes.pop("suite/tasks/gsm8k_clean_v1.yaml", None)
            m["suite_input_hashes"] = hashes
            (run_dir / "manifest.json").write_text(json.dumps(m))
            adapter_path = repo / "adapters" / "qual-model.yaml"

            result = validate_campaign.validate(run_dir, suite_path, adapter_path)
            self.assertFalse(result.passed, "Missing task hash must fail for GSM8K scoring provenance")
            scoring_errors = [
                e for e in result.errors if "scoring_implementation" in e.lower()
            ]
            self.assertTrue(
                len(scoring_errors) > 0,
                f"Must report scoring_implementation error for missing task hash in GSM8K; "
                f"got: {result.errors}",
            )

    # ── IFEval positive: task_file satisfies scoring provenance (no utils_file) ──

    def test_ifeval_task_file_satisfies_scoring_provenance(self):
        """IFEval: no utils_file in suite, task_file in suite_input_hashes → scoring passes.

        RED: current code fails with 'utils_file absent' even when task_file is declared.
        GREEN: task_file in manifest.suite_input_hashes satisfies scoring_implementation.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            suite_path, task_sha256, _ = _make_quality_suite(
                repo, "ifeval",
                task_file_rel="suite/tasks/ifeval_v4.yaml",
            )
            run_dir = _make_quality_run(
                repo, suite_path, "ifeval",
                task_file_rel="suite/tasks/ifeval_v4.yaml",
                task_sha256=task_sha256,
                slug="ifeval-model",
            )
            adapter_path = repo / "adapters" / "ifeval-model.yaml"

            result = validate_campaign.validate(run_dir, suite_path, adapter_path)
            scoring_errors = [
                e for e in result.errors if "scoring_implementation" in e.lower()
            ]
            self.assertEqual(
                scoring_errors, [],
                f"IFEval task_file in hashes should satisfy scoring_implementation; "
                f"scoring_errors: {scoring_errors}\nall_errors: {result.errors}",
            )

    def test_ifeval_task_file_missing_from_hashes_fails(self):
        """IFEval: task_file declared but NOT in manifest.suite_input_hashes → fail."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            suite_path, task_sha256, _ = _make_quality_suite(
                repo, "ifeval",
                task_file_rel="suite/tasks/ifeval_v4.yaml",
            )
            run_dir = _make_quality_run(
                repo, suite_path, "ifeval",
                task_file_rel="suite/tasks/ifeval_v4.yaml",
                task_sha256=task_sha256,
                slug="ifeval-model",
            )
            m = json.loads((run_dir / "manifest.json").read_text())
            hashes = m.get("suite_input_hashes", {})
            hashes.pop("suite/tasks/ifeval_v4.yaml", None)
            m["suite_input_hashes"] = hashes
            (run_dir / "manifest.json").write_text(json.dumps(m))
            adapter_path = repo / "adapters" / "ifeval-model.yaml"

            result = validate_campaign.validate(run_dir, suite_path, adapter_path)
            self.assertFalse(result.passed, "Missing task hash must fail for IFEval scoring provenance")
            scoring_errors = [
                e for e in result.errors if "scoring_implementation" in e.lower()
            ]
            self.assertTrue(
                len(scoring_errors) > 0,
                f"Must report scoring_implementation error for missing task hash in IFEval; "
                f"got: {result.errors}",
            )

    def test_gpqa_wrong_path_in_hashes_fails(self):
        """GPQA: utils_file has a different path (renamed) in suite_input_hashes → fail."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            suite_path, task_sha256, utils_sha256 = _make_quality_suite(
                repo, "gpqa_diamond",
                task_file_rel="suite/tasks/gpqa_diamond_clean_v3.yaml",
                utils_file_rel="suite/tasks/gpqa_utils.py",
            )
            run_dir = _make_quality_run(
                repo, suite_path, "gpqa_diamond",
                task_file_rel="suite/tasks/gpqa_diamond_clean_v3.yaml",
                utils_file_rel="suite/tasks/gpqa_utils.py",
                task_sha256=task_sha256,
                utils_sha256=utils_sha256,
            )
            # Replace utils path with wrong key
            m = json.loads((run_dir / "manifest.json").read_text())
            hashes = m.get("suite_input_hashes", {})
            hashes.pop("suite/tasks/gpqa_utils.py", None)
            hashes["suite/tasks/WRONG_utils.py"] = utils_sha256  # wrong path
            m["suite_input_hashes"] = hashes
            (run_dir / "manifest.json").write_text(json.dumps(m))
            adapter_path = repo / "adapters" / "qual-model.yaml"

            result = validate_campaign.validate(run_dir, suite_path, adapter_path)
            self.assertFalse(result.passed, "Wrong utils path in hashes must fail for GPQA")
            scoring_errors = [
                e for e in result.errors if "scoring_implementation" in e.lower()
            ]
            self.assertTrue(
                len(scoring_errors) > 0,
                f"Must report scoring_implementation error for wrong path; got: {result.errors}",
            )


# ===========================================================================
# S3b: Neither utils_file nor task_file → require lm_eval_revision
# ===========================================================================

class TestS3bNoFilesRequiresHarnessRevision(unittest.TestCase):
    """When a benchmark has neither utils_file nor task_file, the harness revision
    must be present in suite.required_harness.lm_eval_revision.

    This is a defence-in-depth case — not present in the current suite (all benchmarks
    have task_file), but the gate must not blindly accept prose about the harness.

    We exercise the _check_suite_required_evidence function directly (unit-level)
    to avoid the manifest JSON schema enum constraint on benchmark names, which would
    block the full-stack validate() path for unknown benchmark names.
    """

    def _make_suite_no_files(self, tmp: pathlib.Path, include_harness_revision: bool) -> dict:
        """Return a suite_data dict with no task_file or utils_file for 'no_file_bench'."""
        suite: dict = {
            "suite_id": "warpcore-v1",
            "suite_schema_version": 1,
            "benchmarks": {
                "no_file_bench": {
                    "description": "Benchmark with no task/utils file",
                    "expected_item_count": 5,
                    "required_evidence": [
                        "scoring_implementation",
                        "run_log",
                        "command_txt",
                    ],
                }
            },
        }
        if include_harness_revision:
            suite["required_harness"] = {
                "lm_eval_revision": "6d642546f4688648fced259eb3302efd36ece5af"
            }
        return suite

    def test_no_task_or_utils_file_without_harness_revision_fails(self):
        """No task_file, no utils_file, no required_harness → fail scoring provenance.

        The validator must not silently accept 'scoring is in the harness' claims without
        a pinned revision. Tested via _check_suite_required_evidence directly.

        RED: current code emits error 'utils_file absent' but does not distinguish
             "no utils_file, no task_file, no harness" from "harness present" cases.
        GREEN: must report scoring_implementation error when no files and no harness pin.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            run_dir = repo / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            # Minimal manifest with no suite_input_hashes for the task or utils
            manifest_path = run_dir / "manifest.json"
            manifest_path.write_text(json.dumps({"suite_input_hashes": {}}))

            suite_data = self._make_suite_no_files(repo, include_harness_revision=False)

            errors: list = []
            validate_campaign._check_suite_required_evidence(
                run_dir, suite_data, "no_file_bench", errors
            )
            scoring_errors = [e for e in errors if "scoring_implementation" in e.lower()]
            self.assertTrue(
                len(scoring_errors) > 0,
                f"Must report scoring_implementation error when no files and no harness pin; "
                f"got errors: {errors}",
            )

    def test_no_task_or_utils_file_with_harness_revision_passes(self):
        """No task_file, no utils_file, but required_harness.lm_eval_revision is present.

        When neither file is declared but a pinned harness revision exists in the suite,
        scoring provenance is satisfied via the harness pin.

        RED: current code always fails 'utils_file absent' regardless of harness pin.
        GREEN: harness revision present → no scoring_implementation error.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            run_dir = repo / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = run_dir / "manifest.json"
            manifest_path.write_text(json.dumps({"suite_input_hashes": {}}))

            suite_data = self._make_suite_no_files(repo, include_harness_revision=True)

            errors: list = []
            validate_campaign._check_suite_required_evidence(
                run_dir, suite_data, "no_file_bench", errors
            )
            scoring_errors = [e for e in errors if "scoring_implementation" in e.lower()]
            self.assertEqual(
                scoring_errors, [],
                f"With harness revision, scoring_implementation should pass; "
                f"got scoring errors: {scoring_errors}\nall errors: {errors}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
