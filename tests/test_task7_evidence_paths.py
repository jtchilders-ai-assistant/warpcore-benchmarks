"""tests/test_task7_evidence_paths.py — RED tests for Task 7 evidence-path defects.

Tests must initially FAIL (RED) to prove gaps exist, then pass (GREEN) after fixes.

Defects targeted:

  A) Benchmark-aware run_log path:
     - Quality benchmark requires run_dir/run.log (correct, already works)
     - SWE-bench requires run_dir/raw/run.log (not yet implemented)
     - Validator must NOT require root run.log for SWE when raw/run.log exists
     - Evidence name "run_log" must be benchmark-aware in _EVIDENCE_NAME_MAP

  B) suite_input_hashes path containment:
     - Paths in suite_input_hashes must resolve under the repo root
     - Symlink escape must fail even when the external file's hash matches

  C) SWE identity metadata — mandatory instance_ids_hash:
     - manifest.item_inventory.instance_ids_hash must be REQUIRED (not optional)
       for SWE-bench runs — validator must fail if absent (not silently pass)
     - create_campaign must populate instance_ids_hash for SWE from frozen suite file

  D) No-op evidence checks:
     - task_yaml and scoring_implementation must verify required suite files and
       hashes via suite config/manifest hashes — not silently pass (None no-op)
     - completion_token_counts, finish_reasons, full_response_fields must inspect
       actual retained lm-eval sample shape and either verify concrete fields or
       fail closed; fabricating fields is forbidden
     - Unknown evidence names must be rejected (already works, keep as regression guard)
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
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
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_suite_yaml(repo: pathlib.Path, content: str | None = None) -> pathlib.Path:
    p = repo / "suite" / "warpcore-v1.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    if content is None:
        content = "suite_id: warpcore-v1\nsuite_schema_version: 1\n"
    p.write_text(content)
    return p


def _make_adapter_yaml(repo: pathlib.Path, slug: str = "test-model") -> pathlib.Path:
    p = repo / "adapters" / f"{slug}.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("adapter_schema_version: 1\ncampaign_status: canonical\n")
    return p


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_swebench_manifest(
    run_dir: pathlib.Path,
    suite_hash: str,
    adapter_hash: str,
    n_instances: int = 3,
    instance_ids: list[str] | None = None,
    include_instance_ids_hash: bool = True,
    suite_input_hashes: dict | None = None,
) -> dict:
    """Build a minimal SWE-bench manifest dict."""
    if instance_ids is None:
        instance_ids = [f"repo__repo-{i}" for i in range(n_instances)]
    run_id = run_dir.name
    inv: dict = {"expected": len(instance_ids), "submitted": len(instance_ids)}
    if include_instance_ids_hash:
        inv["instance_ids_hash"] = _sha256(
            json.dumps(sorted(instance_ids), sort_keys=True).encode()
        )
    return {
        "schema_version": 1,
        "suite_id": "warpcore-v1",
        "suite_schema_version": 1,
        "run_id": run_id,
        "benchmark": "swebench",
        "adapter_hash": adapter_hash,
        "suite_input_hashes": suite_input_hashes or {"suite/warpcore-v1.yaml": suite_hash},
        "serving_profile_digest": "sha256:" + "c" * 64,
        "model": {"slug": "sweb-model", "id": "org/sweb-model", "revision": "a" * 40},
        "serving": {
            "image_digest": "sha256:" + "d" * 64,
            "engine": "vllm",
            "engine_version": "0.6.6",
            "effective_args": [],
            "environment": {},
            "hardware_id": "dgx-spark-gb10",
        },
        "item_inventory": inv,
        "timing": {
            "started_utc": "2026-09-15T12:00:00Z",
            "completed_utc": "2026-09-15T14:00:00Z",
        },
        "artifact_inventory": {
            "preds_json": True,
            "exit_statuses_json": True,
            "run_log": True,
            "command_txt": True,
            "done_sentinel": True,
        },
    }


def _make_quality_manifest(
    run_dir: pathlib.Path,
    suite_hash: str,
    adapter_hash: str,
    n_items: int = 3,
    benchmark: str = "gsm8k",
    suite_input_hashes: dict | None = None,
) -> dict:
    """Build a minimal quality manifest dict."""
    item_ids = [f"item-{i}" for i in range(n_items)]
    run_id = run_dir.name
    return {
        "schema_version": 1,
        "suite_id": "warpcore-v1",
        "suite_schema_version": 1,
        "run_id": run_id,
        "benchmark": benchmark,
        "adapter_hash": adapter_hash,
        "suite_input_hashes": suite_input_hashes or {"suite/warpcore-v1.yaml": suite_hash},
        "serving_profile_digest": "sha256:" + "c" * 64,
        "model": {"slug": "test-model", "id": "org/test-model", "revision": "a" * 40},
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
            "submitted": n_items,
            "instance_ids_hash": _sha256(
                json.dumps(sorted(item_ids), sort_keys=True).encode()
            ),
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


def _make_status(
    run_id: str,
    suite_id: str = "warpcore-v1",
    execution_state: str = "completed",
    lifecycle: str = "current",
) -> dict:
    chain = ["planned", "preflight_passed", "running", "completed", "validated"]
    idx = chain.index(execution_state) if execution_state in chain else 3
    history = [
        {"state": s, "timestamp": f"2026-09-15T{12 + i:02d}:00:00Z"}
        for i, s in enumerate(chain[: idx + 1])
    ]
    return {
        "schema_version": 1,
        "run_id": run_id,
        "suite_id": suite_id,
        "execution_state": execution_state,
        "lifecycle": lifecycle,
        "history": history,
    }


def _write_swebench_run(
    repo: pathlib.Path,
    *,
    n_instances: int = 3,
    include_root_run_log: bool = False,
    include_raw_run_log: bool = True,
    include_instance_ids_hash: bool = True,
    suite_input_hashes: dict | None = None,
    model_slug: str = "sweb-model",
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """Write a minimal SWE-bench run directory. Returns (run_dir, suite_path, adapter_path)."""
    suite_path = _make_suite_yaml(repo)
    suite_hash = _sha256(suite_path.read_bytes())
    adapter_path = _make_adapter_yaml(repo, model_slug)
    adapter_hash = _sha256(adapter_path.read_bytes())

    run_id = "run-2026-09-15T12-00-00"
    run_dir = repo / "results" / model_slug / "runs" / "warpcore-v1" / "swebench" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    instance_ids = [f"repo__repo-{i}" for i in range(n_instances)]
    manifest = _make_swebench_manifest(
        run_dir,
        suite_hash,
        adapter_hash,
        instance_ids=instance_ids,
        include_instance_ids_hash=include_instance_ids_hash,
        suite_input_hashes=suite_input_hashes,
    )
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (run_dir / "status.json").write_text(
        json.dumps(_make_status(run_id), indent=2)
    )
    (run_dir / "DONE").write_text("done\n")
    (run_dir / "command.txt").write_text("python3 run_swebench.py\n")
    if include_root_run_log:
        (run_dir / "run.log").write_text("harness exit 0\n")

    raw_dir = run_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    if include_raw_run_log:
        (raw_dir / "run.log").write_text("harness exit 0\n")

    # Minimal SWE-bench evidence
    preds = {iid: {"model_patch": "diff"} for iid in instance_ids}
    (raw_dir / "preds.json").write_text(json.dumps(preds))
    statuses = {iid: 0 for iid in instance_ids}
    (raw_dir / "exit_statuses.json").write_text(json.dumps(statuses))

    grading = {
        "resolved_ids": instance_ids,
        "unresolved_ids": [],
        "empty_patch_ids": [],
        "error_ids": [],
        "incomplete_ids": [],
    }
    (raw_dir / "grading_results.json").write_text(json.dumps(grading))

    traj_dir = raw_dir / "trajectories"
    traj_dir.mkdir(exist_ok=True)
    for iid in instance_ids:
        (traj_dir / f"{iid}.traj").write_text("{}")

    return run_dir, suite_path, adapter_path


# ---------------------------------------------------------------------------
# Defect A: Benchmark-aware run_log paths
# ---------------------------------------------------------------------------

class TestDefectA_BenchmarkAwareRunLog(unittest.TestCase):
    """
    Defect A: The evidence name "run_log" must be benchmark-aware.

    - Quality benchmarks: run_log → run_dir/run.log  (is_raw=False)
    - SWE-bench: run_log → run_dir/raw/run.log  (is_raw=True)

    The validator must NOT fail SWE-bench validation when raw/run.log exists
    and root run.log is absent. Currently the _EVIDENCE_NAME_MAP has run_log
    as (run.log, False) universally — this always checks run_dir/run.log for
    all benchmarks, which is wrong for SWE.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _validate(self, run_dir, suite_path, adapter_path):
        return validate_campaign.validate(run_dir, suite_path, adapter_path)

    def test_swebench_with_raw_run_log_only_should_pass(self):
        """
        SWE run with raw/run.log but NO root run.log must pass validation.

        RED: currently fails because run_log is mapped to (run.log, False)
        regardless of benchmark, so validator incorrectly requires run_dir/run.log.
        """
        run_dir, suite_path, adapter_path = _write_swebench_run(
            self.repo,
            include_root_run_log=False,
            include_raw_run_log=True,
        )
        result = self._validate(run_dir, suite_path, adapter_path)
        run_log_errors = [e for e in result.errors if "run.log" in e and "run_log" in e.lower()]
        self.assertFalse(
            run_log_errors,
            f"SWE run with raw/run.log should not produce run.log validation errors; "
            f"got: {run_log_errors}",
        )

    def test_swebench_missing_raw_run_log_should_fail(self):
        """
        SWE run with NEITHER raw/run.log NOR root run.log must fail.

        This confirms the SWE-specific check fires correctly.
        """
        run_dir, suite_path, adapter_path = _write_swebench_run(
            self.repo,
            include_root_run_log=False,
            include_raw_run_log=False,
        )
        result = self._validate(run_dir, suite_path, adapter_path)
        run_log_errors = [
            e for e in result.errors
            if "run.log" in e or "run_log" in e.lower()
        ]
        self.assertTrue(
            run_log_errors,
            "SWE run missing both run_dir/run.log and run_dir/raw/run.log must produce "
            "an error referencing run.log",
        )
        self.assertFalse(
            result.passed,
            "SWE run with no run.log anywhere must not pass validation",
        )

    def test_quality_run_requires_root_run_log(self):
        """
        Quality run still requires run_dir/run.log (not raw/run.log).

        This is the existing correct behavior — confirm it is not broken.
        """
        suite_path = _make_suite_yaml(self.repo)
        suite_hash = _sha256(suite_path.read_bytes())
        adapter_path = _make_adapter_yaml(self.repo)
        adapter_hash = _sha256(adapter_path.read_bytes())
        run_id = "run-2026-09-15T12-00-00"
        run_dir = (
            self.repo / "results" / "test-model" / "runs" / "warpcore-v1" / "gsm8k" / run_id
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        n = 3
        item_ids = [f"item-{i}" for i in range(n)]
        manifest = _make_quality_manifest(run_dir, suite_hash, adapter_hash, n_items=n)
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (run_dir / "status.json").write_text(
            json.dumps(_make_status(run_id), indent=2)
        )
        (run_dir / "DONE").write_text("done\n")
        (run_dir / "command.txt").write_text("lm_eval\n")
        # Deliberately omit run.log — should fail
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        rows = [
            {"item_id": iid, "score": 1.0, "disposition": "correct",
             "finish_reason": "stop", "response_chars": 100, "empty_content": False}
            for iid in item_ids
        ]
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
        (run_dir / "per_item.csv").write_text(buf.getvalue())
        with gzip.open(raw_dir / "samples_gsm8k.jsonl.gz", "wt") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        (raw_dir / "results_test.json").write_text(json.dumps({
            "results": {"gsm8k": {"exact_match,flexible-fallback": 1.0}},
            "n_samples": n,
        }))

        result = self._validate(run_dir, suite_path, adapter_path)
        run_log_errors = [e for e in result.errors if "run.log" in e]
        self.assertTrue(
            run_log_errors,
            "Quality run missing run.log must produce a run.log error",
        )

    def test_evidence_name_map_run_log_is_benchmark_aware(self):
        """
        The _EVIDENCE_NAME_MAP must differentiate run_log path by benchmark.

        RED: currently run_log is a single entry with is_raw=False.
        After fix, the map or the evidence check function must handle
        SWE-bench specifically: run.log under raw/.
        """
        # Check that the module exposes benchmark-aware logic for run_log
        # Either via a different _EVIDENCE_NAME_MAP structure,
        # or via _check_suite_required_evidence being benchmark-aware.
        # The test probes the actual behavior by checking _check_suite_required_evidence
        # does NOT add a run.log error when SWE run has raw/run.log.
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp)
            run_dir, suite_path, adapter_path = _write_swebench_run(
                repo,
                include_root_run_log=False,
                include_raw_run_log=True,
            )
            suite_data = {"benchmarks": {"swebench": {"required_evidence": ["run_log"]}}}
            errors: list[str] = []
            validate_campaign._check_suite_required_evidence(
                run_dir, suite_data, "swebench", errors
            )
            run_log_errors = [e for e in errors if "run.log" in e]
            self.assertFalse(
                run_log_errors,
                f"_check_suite_required_evidence must not report run.log missing for SWE "
                f"when raw/run.log exists; errors: {run_log_errors}",
            )


# ---------------------------------------------------------------------------
# Defect B: suite_input_hashes path containment (symlink escape)
# ---------------------------------------------------------------------------

class TestDefectB_SuiteInputHashContainment(unittest.TestCase):
    """
    Defect B: suite_input_hashes paths must resolve under the repo root.

    Symlink escape must fail even when the external file's hash matches
    the recorded hash. The validator currently resolves suite_input_hashes
    relative to the repo root but does NOT verify path containment — a
    symlink pointing outside the repo would pass if the hash matches.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = pathlib.Path(self._tmp.name)
        self._external = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()
        self._external.cleanup()

    def test_symlink_escape_in_suite_input_hashes_fails_even_with_matching_hash(self):
        """
        A suite_input_hashes entry pointing outside the repo via a symlink
        must fail validation even when the external file's hash matches.

        RED: currently _verify_hashes resolves abs_path = repo / rel_path
        but does NOT call abs_path.resolve() to catch symlinks, so a symlink
        to an external file with a matching hash passes silently.
        """
        # Create an external file outside the repo
        ext_dir = pathlib.Path(self._external.name)
        ext_file = ext_dir / "external_task.yaml"
        ext_file.write_text("task: external\n")
        ext_hash = _sha256(ext_file.read_bytes())

        # Create a symlink inside the repo pointing to the external file
        suite_dir = self.repo / "suite"
        suite_dir.mkdir(parents=True, exist_ok=True)
        task_dir = self.repo / "suite" / "tasks"
        task_dir.mkdir(parents=True, exist_ok=True)
        symlink = task_dir / "escaped_task.yaml"
        symlink.symlink_to(ext_file)

        # Build run with suite_input_hashes referencing the symlink path
        # but the hash matches the external file (so it "passes" hash check)
        suite_path = _make_suite_yaml(self.repo)
        suite_hash = _sha256(suite_path.read_bytes())
        adapter_path = _make_adapter_yaml(self.repo)
        adapter_hash = _sha256(adapter_path.read_bytes())
        run_id = "run-escape-test"
        run_dir = (
            self.repo / "results" / "test-model" / "runs"
            / "warpcore-v1" / "gsm8k" / run_id
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        # suite_input_hashes has the symlink path with its external hash
        suite_input_hashes = {
            "suite/warpcore-v1.yaml": suite_hash,
            "suite/tasks/escaped_task.yaml": ext_hash,  # hash matches external file
        }
        manifest = _make_quality_manifest(
            run_dir, suite_hash, adapter_hash, suite_input_hashes=suite_input_hashes
        )
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (run_dir / "status.json").write_text(json.dumps(_make_status(run_id), indent=2))
        (run_dir / "DONE").write_text("done\n")
        (run_dir / "run.log").write_text("ok\n")
        (run_dir / "command.txt").write_text("lm_eval\n")

        n = 3
        item_ids = [f"item-{i}" for i in range(n)]
        rows = [
            {"item_id": iid, "score": 1.0, "disposition": "correct",
             "finish_reason": "stop", "response_chars": 100, "empty_content": False}
            for iid in item_ids
        ]
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
        (run_dir / "per_item.csv").write_text(buf.getvalue())
        with gzip.open(raw_dir / "samples_gsm8k.jsonl.gz", "wt") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        (raw_dir / "results_test.json").write_text(json.dumps({
            "results": {"gsm8k": {"exact_match,flexible-fallback": 1.0}},
            "n_samples": n,
        }))

        result = validate_campaign.validate(run_dir, suite_path, adapter_path)
        containment_errors = [
            e for e in result.errors
            if "outside" in e.lower() or "escape" in e.lower()
            or "contain" in e.lower() or "symlink" in e.lower()
            or "traversal" in e.lower()
        ]
        self.assertTrue(
            containment_errors,
            f"suite_input_hashes with symlink escape should produce a containment error; "
            f"errors were: {result.errors}",
        )
        self.assertFalse(
            result.passed,
            "Validation with symlink-escaped suite_input_hashes must not pass",
        )

    def test_internal_suite_input_hashes_pass_containment(self):
        """
        suite_input_hashes referencing real files inside the repo must pass containment.
        This is the green-path regression check.
        """
        suite_path = _make_suite_yaml(self.repo)
        suite_hash = _sha256(suite_path.read_bytes())
        adapter_path = _make_adapter_yaml(self.repo)
        adapter_hash = _sha256(adapter_path.read_bytes())

        # Create a real task file inside the repo
        task_dir = self.repo / "suite" / "tasks"
        task_dir.mkdir(parents=True, exist_ok=True)
        task_file = task_dir / "real_task.yaml"
        task_file.write_text("task: real\n")
        task_hash = _sha256(task_file.read_bytes())

        run_id = "run-containment-pass"
        run_dir = (
            self.repo / "results" / "test-model" / "runs"
            / "warpcore-v1" / "gsm8k" / run_id
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        suite_input_hashes = {
            "suite/warpcore-v1.yaml": suite_hash,
            "suite/tasks/real_task.yaml": task_hash,
        }
        n = 3
        item_ids = [f"item-{i}" for i in range(n)]
        manifest = _make_quality_manifest(
            run_dir, suite_hash, adapter_hash, suite_input_hashes=suite_input_hashes
        )
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (run_dir / "status.json").write_text(json.dumps(_make_status(run_id), indent=2))
        (run_dir / "DONE").write_text("done\n")
        (run_dir / "run.log").write_text("ok\n")
        (run_dir / "command.txt").write_text("lm_eval\n")

        rows = [
            {"item_id": iid, "score": 1.0, "disposition": "correct",
             "finish_reason": "stop", "response_chars": 100, "empty_content": False}
            for iid in item_ids
        ]
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
        (run_dir / "per_item.csv").write_text(buf.getvalue())
        with gzip.open(raw_dir / "samples_gsm8k.jsonl.gz", "wt") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        (raw_dir / "results_test.json").write_text(json.dumps({
            "results": {"gsm8k": {"exact_match,flexible-fallback": 1.0}},
            "n_samples": n,
        }))

        result = validate_campaign.validate(run_dir, suite_path, adapter_path)
        containment_errors = [
            e for e in result.errors
            if "outside" in e.lower() or "escape" in e.lower()
            or "contain" in e.lower() or "symlink" in e.lower()
            or "traversal" in e.lower()
        ]
        self.assertFalse(
            containment_errors,
            f"suite_input_hashes with real internal files must not produce containment errors; "
            f"got: {containment_errors}",
        )


# ---------------------------------------------------------------------------
# Defect C: Mandatory SWE instance_ids_hash in manifest
# ---------------------------------------------------------------------------

class TestDefectC_MandatorySWEInstanceIdsHash(unittest.TestCase):
    """
    Defect C: instance_ids_hash must be REQUIRED for SWE-bench manifests.

    Currently _validate_swebench_frozen_ids checks recorded_hash only when
    it is non-empty (line: `if recorded_hash:`). A SWE manifest without
    instance_ids_hash silently passes this gate. It must instead FAIL.

    Additionally, create_campaign must populate instance_ids_hash for SWE
    campaigns using the frozen suite instance set.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _make_frozen_suite(self, instance_ids: list[str]) -> tuple[pathlib.Path, str]:
        """Create suite YAML with frozen SWE-bench instance set. Returns (suite_path, suite_hash)."""
        instance_file = self.repo / "suite" / "swe_instances.json"
        instance_file.parent.mkdir(parents=True, exist_ok=True)
        instance_file.write_text(json.dumps(instance_ids))
        instance_hash = _sha256(instance_file.read_bytes())

        suite_content = (
            "suite_id: warpcore-v1\n"
            "suite_schema_version: 1\n"
            "benchmarks:\n"
            "  swebench:\n"
            f"    expected_item_count: {len(instance_ids)}\n"
            f"    instance_set_file: suite/swe_instances.json\n"
            f"    instances_sha256: {instance_hash}\n"
        )
        suite_path = self.repo / "suite" / "warpcore-v1.yaml"
        suite_path.write_text(suite_content)
        return suite_path, _sha256(suite_path.read_bytes())

    def test_swebench_manifest_missing_instance_ids_hash_must_fail(self):
        """
        A SWE-bench manifest without item_inventory.instance_ids_hash must fail.

        RED: currently _validate_swebench_frozen_ids skips the hash check when
        recorded_hash is empty/absent, so a manifest missing instance_ids_hash
        silently passes. It must instead fail closed with an explicit error.
        """
        instance_ids = [f"repo__repo-{i}" for i in range(3)]
        suite_path, suite_hash = self._make_frozen_suite(instance_ids)
        adapter_path = _make_adapter_yaml(self.repo)
        adapter_hash = _sha256(adapter_path.read_bytes())

        run_id = "run-missing-iid-hash"
        run_dir = (
            self.repo / "results" / "sweb-model" / "runs"
            / "warpcore-v1" / "swebench" / run_id
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        # Build SWE-bench run WITHOUT instance_ids_hash
        manifest = _make_swebench_manifest(
            run_dir,
            suite_hash,
            adapter_hash,
            instance_ids=instance_ids,
            include_instance_ids_hash=False,  # omit the hash
        )
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (run_dir / "status.json").write_text(json.dumps(_make_status(run_id), indent=2))
        (run_dir / "DONE").write_text("done\n")
        (run_dir / "command.txt").write_text("python3 run_swebench.py\n")

        raw_dir = run_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        (raw_dir / "run.log").write_text("ok\n")
        preds = {iid: {"model_patch": "diff"} for iid in instance_ids}
        (raw_dir / "preds.json").write_text(json.dumps(preds))
        statuses = {iid: 0 for iid in instance_ids}
        (raw_dir / "exit_statuses.json").write_text(json.dumps(statuses))
        grading = {
            "resolved_ids": instance_ids,
            "unresolved_ids": [],
            "empty_patch_ids": [],
            "error_ids": [],
            "incomplete_ids": [],
        }
        (raw_dir / "grading_results.json").write_text(json.dumps(grading))
        traj_dir = raw_dir / "trajectories"
        traj_dir.mkdir(exist_ok=True)
        for iid in instance_ids:
            (traj_dir / f"{iid}.traj").write_text("{}")

        result = validate_campaign.validate(run_dir, suite_path, adapter_path)
        iid_hash_errors = [
            e for e in result.errors
            if "instance_ids_hash" in e.lower() or "instance id" in e.lower()
        ]
        self.assertTrue(
            iid_hash_errors,
            f"SWE manifest missing instance_ids_hash must produce an error; "
            f"errors were: {result.errors}",
        )
        self.assertFalse(
            result.passed,
            "SWE run missing instance_ids_hash must not pass validation",
        )

    def test_swebench_manifest_with_correct_instance_ids_hash_passes(self):
        """
        A SWE-bench manifest WITH correct instance_ids_hash must pass the hash gate.
        """
        instance_ids = [f"repo__repo-{i}" for i in range(3)]
        suite_path, suite_hash = self._make_frozen_suite(instance_ids)
        adapter_path = _make_adapter_yaml(self.repo)
        adapter_hash = _sha256(adapter_path.read_bytes())

        run_id = "run-has-iid-hash"
        run_dir = (
            self.repo / "results" / "sweb-model" / "runs"
            / "warpcore-v1" / "swebench" / run_id
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        manifest = _make_swebench_manifest(
            run_dir,
            suite_hash,
            adapter_hash,
            instance_ids=instance_ids,
            include_instance_ids_hash=True,
        )
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (run_dir / "status.json").write_text(json.dumps(_make_status(run_id), indent=2))
        (run_dir / "DONE").write_text("done\n")
        (run_dir / "command.txt").write_text("python3 run_swebench.py\n")

        raw_dir = run_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        (raw_dir / "run.log").write_text("ok\n")
        preds = {iid: {"model_patch": "diff"} for iid in instance_ids}
        (raw_dir / "preds.json").write_text(json.dumps(preds))
        statuses = {iid: 0 for iid in instance_ids}
        (raw_dir / "exit_statuses.json").write_text(json.dumps(statuses))
        grading = {
            "resolved_ids": instance_ids,
            "unresolved_ids": [],
            "empty_patch_ids": [],
            "error_ids": [],
            "incomplete_ids": [],
        }
        (raw_dir / "grading_results.json").write_text(json.dumps(grading))
        traj_dir = raw_dir / "trajectories"
        traj_dir.mkdir(exist_ok=True)
        for iid in instance_ids:
            (traj_dir / f"{iid}.traj").write_text("{}")

        result = validate_campaign.validate(run_dir, suite_path, adapter_path)
        iid_hash_errors = [
            e for e in result.errors
            if "instance_ids_hash" in e.lower() and "missing" in e.lower()
        ]
        self.assertFalse(
            iid_hash_errors,
            f"SWE manifest with correct instance_ids_hash must not produce hash-missing errors; "
            f"got: {iid_hash_errors}",
        )

    def test_validate_swebench_frozen_ids_requires_hash_when_frozen_set_available(self):
        """
        _validate_swebench_frozen_ids must require instance_ids_hash when
        the frozen set is available — not silently skip the check.

        Directly tests the internal function to confirm the gate fires.
        """
        import hashlib, json
        frozen_ids = frozenset(["repo__a", "repo__b", "repo__c"])
        manifest_without_hash = {
            "item_inventory": {
                "expected": 3,
                "submitted": 3,
                # No instance_ids_hash key
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp) / "run"
            run_dir.mkdir()
            raw_dir = run_dir / "raw"
            raw_dir.mkdir()
            errors: list[str] = []
            validate_campaign._validate_swebench_frozen_ids(
                run_dir, manifest_without_hash, frozen_ids, errors
            )
            iid_errors = [
                e for e in errors
                if "instance_ids_hash" in e.lower()
            ]
            self.assertTrue(
                iid_errors,
                f"_validate_swebench_frozen_ids must error when instance_ids_hash is absent; "
                f"errors: {errors}",
            )


# ---------------------------------------------------------------------------
# Defect D: No-op evidence checks
# ---------------------------------------------------------------------------

class TestDefectD_NoOpEvidenceChecks(unittest.TestCase):
    """
    Defect D: No evidence name may be accepted as a no-op.

    Currently task_yaml, scoring_implementation, completion_token_counts,
    finish_reasons, and full_response_fields all map to None in _EVIDENCE_NAME_MAP,
    causing _check_suite_required_evidence to silently skip them (continue when
    file_hint is None). Each must perform a real, non-trivially-falsifiable check.

    task_yaml / scoring_implementation:
      Must verify the suite config declares the corresponding file AND
      that the manifest suite_input_hashes records that file's hash.
      A run that lists these in required_evidence but lacks the hash entry
      must FAIL — it may not silently pass.

    completion_token_counts / finish_reasons / full_response_fields:
      Must inspect actual retained lm-eval sample records and verify the
      required fields are present in at least one sample record. If no
      samples exist or the fields are absent, the check must fail closed.
      Fabricating fields in the frozen suite to evade enforcement is forbidden.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _make_suite_with_evidence(
        self,
        benchmark: str = "gsm8k",
        required_evidence: list[str] | None = None,
        task_file: str | None = None,
        utils_file: str | None = None,
    ) -> pathlib.Path:
        """Build suite YAML with specified required_evidence for a benchmark."""
        bench_block = {}
        if task_file:
            bench_block["task_file"] = task_file
        if utils_file:
            bench_block["utils_file"] = utils_file
        if required_evidence is not None:
            bench_block["required_evidence"] = required_evidence
        suite = {
            "suite_id": "warpcore-v1",
            "suite_schema_version": 1,
            "benchmarks": {benchmark: bench_block},
        }
        import yaml as _yaml
        content = _yaml.dump(suite)
        p = self.repo / "suite" / "warpcore-v1.yaml"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return p

    def test_task_yaml_evidence_fails_when_hash_not_in_manifest(self):
        """
        required_evidence: [task_yaml] must fail if the task file hash is
        absent from manifest.suite_input_hashes.

        RED: currently task_yaml maps to None → silently no-op passes.
        """
        task_dir = self.repo / "suite" / "tasks"
        task_dir.mkdir(parents=True, exist_ok=True)
        task_file = task_dir / "gsm8k_v1.yaml"
        task_file.write_text("task: gsm8k\n")
        suite_path = self._make_suite_with_evidence(
            benchmark="gsm8k",
            required_evidence=["task_yaml"],
            task_file="suite/tasks/gsm8k_v1.yaml",
        )
        suite_data = {
            "suite_id": "warpcore-v1",
            "suite_schema_version": 1,
            "benchmarks": {
                "gsm8k": {
                    "task_file": "suite/tasks/gsm8k_v1.yaml",
                    "required_evidence": ["task_yaml"],
                }
            },
        }
        # Build run with suite_input_hashes that does NOT include the task file
        suite_hash = _sha256(suite_path.read_bytes())
        adapter_path = _make_adapter_yaml(self.repo)
        adapter_hash = _sha256(adapter_path.read_bytes())
        run_id = "run-task-yaml-test"
        run_dir = (
            self.repo / "results" / "test-model" / "runs"
            / "warpcore-v1" / "gsm8k" / run_id
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = _make_quality_manifest(
            run_dir,
            suite_hash,
            adapter_hash,
            # suite_input_hashes MISSING the task file hash
            suite_input_hashes={"suite/warpcore-v1.yaml": suite_hash},
        )
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        # Directly test _check_suite_required_evidence
        errors: list[str] = []
        validate_campaign._check_suite_required_evidence(
            run_dir, suite_data, "gsm8k", errors
        )
        task_yaml_errors = [
            e for e in errors
            if "task_yaml" in e.lower() or "task file" in e.lower() or "task_file" in e.lower()
        ]
        self.assertTrue(
            task_yaml_errors,
            f"task_yaml evidence check must produce an error when task file hash is "
            f"absent from manifest; errors: {errors}",
        )

    def test_scoring_implementation_evidence_fails_when_hash_not_in_manifest(self):
        """
        required_evidence: [scoring_implementation] must fail if the utils file hash
        is absent from manifest.suite_input_hashes.

        RED: currently scoring_implementation maps to None → silently no-op.
        """
        utils_dir = self.repo / "suite" / "utils"
        utils_dir.mkdir(parents=True, exist_ok=True)
        utils_file = utils_dir / "scoring.py"
        utils_file.write_text("def score(): pass\n")
        suite_path = self._make_suite_with_evidence(
            benchmark="gsm8k",
            required_evidence=["scoring_implementation"],
            utils_file="suite/utils/scoring.py",
        )
        suite_data = {
            "suite_id": "warpcore-v1",
            "suite_schema_version": 1,
            "benchmarks": {
                "gsm8k": {
                    "utils_file": "suite/utils/scoring.py",
                    "required_evidence": ["scoring_implementation"],
                }
            },
        }
        errors: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp) / "run"
            run_dir.mkdir()
            validate_campaign._check_suite_required_evidence(
                run_dir, suite_data, "gsm8k", errors
            )
        scoring_errors = [
            e for e in errors
            if "scoring_implementation" in e.lower()
            or "utils_file" in e.lower()
            or "scoring" in e.lower()
        ]
        self.assertTrue(
            scoring_errors,
            f"scoring_implementation evidence must produce an error when utils file hash "
            f"is absent from manifest; errors: {errors}",
        )

    def test_completion_token_counts_requires_real_sample_inspection(self):
        """
        required_evidence: [completion_token_counts] must inspect samples and
        verify completion_tokens or token_count fields are present.

        When samples are absent or the field is missing, it must fail closed.

        RED: currently completion_token_counts → None → silently no-op.
        """
        suite_data = {
            "suite_id": "warpcore-v1",
            "suite_schema_version": 1,
            "benchmarks": {
                "gsm8k": {"required_evidence": ["completion_token_counts"]}
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp) / "run"
            raw_dir = run_dir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)

            # Write samples WITHOUT completion_token_counts field
            records = [
                {"doc_id": 0, "resps": [["answer"]], "exact_match": 1.0}
                for _ in range(3)
            ]
            with gzip.open(raw_dir / "samples_gsm8k.jsonl.gz", "wt") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            errors: list[str] = []
            validate_campaign._check_suite_required_evidence(
                run_dir, suite_data, "gsm8k", errors
            )
            token_errors = [
                e for e in errors
                if "completion_token" in e.lower() or "token_count" in e.lower()
                or "token" in e.lower()
            ]
            self.assertTrue(
                token_errors,
                f"completion_token_counts must fail when field absent from samples; "
                f"errors: {errors}",
            )

    def test_finish_reasons_requires_real_sample_inspection(self):
        """
        required_evidence: [finish_reasons] must inspect samples for finish_reason field.
        When the field is absent, validation must fail closed.

        RED: currently finish_reasons → None → silently no-op.
        """
        suite_data = {
            "suite_id": "warpcore-v1",
            "suite_schema_version": 1,
            "benchmarks": {
                "gsm8k": {"required_evidence": ["finish_reasons"]}
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp) / "run"
            raw_dir = run_dir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)

            # Write samples WITHOUT finish_reason field
            records = [
                {"doc_id": i, "resps": [["answer"]], "exact_match": 1.0}
                for i in range(3)
            ]
            with gzip.open(raw_dir / "samples_gsm8k.jsonl.gz", "wt") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            errors: list[str] = []
            validate_campaign._check_suite_required_evidence(
                run_dir, suite_data, "gsm8k", errors
            )
            fr_errors = [
                e for e in errors
                if "finish_reason" in e.lower() or "finish reason" in e.lower()
            ]
            self.assertTrue(
                fr_errors,
                f"finish_reasons must fail when finish_reason field absent from samples; "
                f"errors: {errors}",
            )

    def test_full_response_fields_requires_real_sample_inspection(self):
        """
        required_evidence: [full_response_fields] must inspect samples for
        resps/filtered_resps fields or equivalent raw response fields.
        When the field is absent, validation must fail closed.

        RED: currently full_response_fields → None → silently no-op.
        """
        suite_data = {
            "suite_id": "warpcore-v1",
            "suite_schema_version": 1,
            "benchmarks": {
                "gsm8k": {"required_evidence": ["full_response_fields"]}
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp) / "run"
            raw_dir = run_dir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)

            # Write samples WITHOUT resps/filtered_resps fields
            records = [
                {"doc_id": i, "exact_match": 1.0}  # no resps field
                for i in range(3)
            ]
            with gzip.open(raw_dir / "samples_gsm8k.jsonl.gz", "wt") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            errors: list[str] = []
            validate_campaign._check_suite_required_evidence(
                run_dir, suite_data, "gsm8k", errors
            )
            resp_errors = [
                e for e in errors
                if "resps" in e.lower() or "response" in e.lower()
                or "full_response" in e.lower()
            ]
            self.assertTrue(
                resp_errors,
                f"full_response_fields must fail when resps/response fields absent from samples; "
                f"errors: {errors}",
            )

    def test_no_op_evidence_names_are_not_accepted_silently(self):
        """
        None of the previously-None evidence names may pass silently when
        the underlying evidence is absent.

        This is a combined regression guard: for each evidence name that was
        previously a no-op, confirm that passing it in required_evidence for a
        run with no supporting artifacts produces at least one error.
        """
        previously_noop = [
            "task_yaml",
            "scoring_implementation",
            "completion_token_counts",
            "finish_reasons",
            "full_response_fields",
        ]
        for ev_name in previously_noop:
            with self.subTest(evidence=ev_name):
                suite_data = {
                    "suite_id": "warpcore-v1",
                    "benchmarks": {
                        "gsm8k": {
                            "required_evidence": [ev_name],
                            "task_file": "suite/tasks/nonexistent.yaml",
                            "utils_file": "suite/utils/nonexistent.py",
                        }
                    },
                }
                with tempfile.TemporaryDirectory() as tmp:
                    run_dir = pathlib.Path(tmp) / "run"
                    run_dir.mkdir()
                    errors: list[str] = []
                    validate_campaign._check_suite_required_evidence(
                        run_dir, suite_data, "gsm8k", errors
                    )
                    self.assertTrue(
                        errors,
                        f"Evidence name {ev_name!r} must not be a silent no-op; "
                        f"expected at least one error for empty run but got none",
                    )

    def test_sample_finish_reason_without_sidecar_does_not_pass(self):
        """Sample-only finish_reason is not request-path response evidence."""
        suite_data = {
            "suite_id": "warpcore-v1",
            "benchmarks": {
                "gsm8k": {"required_evidence": ["finish_reasons"]}
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp) / "run"
            raw_dir = run_dir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)

            records = [
                {"doc_id": i, "resps": [["answer"]], "exact_match": 1.0,
                 "finish_reason": "stop"}
                for i in range(3)
            ]
            with gzip.open(raw_dir / "samples_gsm8k.jsonl.gz", "wt") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            errors: list[str] = []
            validate_campaign._check_suite_required_evidence(
                run_dir, suite_data, "gsm8k", errors
            )
            fr_errors = [
                e for e in errors
                if "finish_reason" in e.lower()
            ]
            self.assertTrue(
                fr_errors,
                f"finish_reasons without a reconciled sidecar must error; "
                f"got: {fr_errors}",
            )

    def test_sample_completion_tokens_without_sidecar_do_not_pass(self):
        """Sample-only token counts are not request-path response evidence."""
        suite_data = {
            "suite_id": "warpcore-v1",
            "benchmarks": {
                "gsm8k": {"required_evidence": ["completion_token_counts"]}
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp) / "run"
            raw_dir = run_dir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)

            records = [
                {"doc_id": i, "resps": [["answer"]], "exact_match": 1.0,
                 "completion_tokens": 42}
                for i in range(3)
            ]
            with gzip.open(raw_dir / "samples_gsm8k.jsonl.gz", "wt") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            errors: list[str] = []
            validate_campaign._check_suite_required_evidence(
                run_dir, suite_data, "gsm8k", errors
            )
            token_errors = [
                e for e in errors
                if "completion_token" in e.lower() or "token_count" in e.lower()
            ]
            self.assertTrue(
                token_errors,
                f"completion_token_counts without a reconciled sidecar must error; "
                f"got: {token_errors}",
            )

    def test_sample_resps_without_sidecar_do_not_pass_full_response_check(self):
        """lm-eval's reduced resps field is not the retained full API response."""
        suite_data = {
            "suite_id": "warpcore-v1",
            "benchmarks": {
                "gsm8k": {"required_evidence": ["full_response_fields"]}
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp) / "run"
            raw_dir = run_dir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)

            records = [
                {"doc_id": i, "resps": [["The answer is 42."]], "exact_match": 1.0}
                for i in range(3)
            ]
            with gzip.open(raw_dir / "samples_gsm8k.jsonl.gz", "wt") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            errors: list[str] = []
            validate_campaign._check_suite_required_evidence(
                run_dir, suite_data, "gsm8k", errors
            )
            resp_errors = [
                e for e in errors
                if "resps" in e.lower() or "full_response" in e.lower()
            ]
            self.assertTrue(
                resp_errors,
                f"full_response_fields without a reconciled sidecar must error; got: {resp_errors}",
            )


# ---------------------------------------------------------------------------
# Additional regression guards
# ---------------------------------------------------------------------------

class TestUnknownEvidenceNameRejection(unittest.TestCase):
    """Regression guard: unknown evidence names must still be rejected (fail-closed)."""

    def test_unknown_evidence_name_is_rejected(self):
        suite_data = {
            "suite_id": "warpcore-v1",
            "benchmarks": {
                "gsm8k": {"required_evidence": ["totally_unknown_evidence_xyz"]}
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp) / "run"
            run_dir.mkdir()
            errors: list[str] = []
            validate_campaign._check_suite_required_evidence(
                run_dir, suite_data, "gsm8k", errors
            )
            self.assertTrue(
                errors,
                "Unknown evidence name must be rejected with an error",
            )
            unknown_errors = [e for e in errors if "unknown" in e.lower()]
            self.assertTrue(
                unknown_errors,
                f"Error for unknown evidence name must mention 'unknown'; got: {errors}",
            )


class TestRecursiveQualityEvidenceDiscovery(unittest.TestCase):
    """lm-eval stores quality artifacts below a model-named raw subdirectory."""

    def test_suite_required_globs_accept_nested_lmeval_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp) / "run"
            nested = run_dir / "raw" / "org__model"
            nested.mkdir(parents=True)
            (nested / "results_test.json").write_text("{}")
            with gzip.open(nested / "samples_test.jsonl.gz", "wt") as fh:
                fh.write(json.dumps({"doc_id": 0}) + "\n")

            suite_data = {
                "benchmarks": {
                    "gsm8k": {
                        "required_evidence": ["aggregate_result", "samples_jsonl_gz"]
                    }
                }
            }
            errors: list[str] = []
            validate_campaign._check_suite_required_evidence(
                run_dir, suite_data, "gsm8k", errors
            )

            self.assertEqual(errors, [])

    def test_aggregate_reconciliation_accepts_nested_lmeval_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp) / "run"
            nested = run_dir / "raw" / "org__model"
            nested.mkdir(parents=True)
            (nested / "results_test.json").write_text(json.dumps({
                "results": {"gsm8k": {"exact_match,answer-line": 1.0}}
            }))
            errors: list[str] = []
            validate_campaign._check_aggregate_reconciliation(
                run_dir,
                [{"item_id": "0", "score": "1.0"}],
                "gsm8k",
                errors,
                fail_on_missing=True,
            )

            self.assertEqual(errors, [])

    def test_gsm8k_reconciliation_uses_canonical_answer_line_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp) / "run"
            nested = run_dir / "raw" / "org__model"
            nested.mkdir(parents=True)
            (nested / "results_test.json").write_text(json.dumps({
                "results": {
                    "gsm8k": {
                        "exact_match,answer-line": 0.0,
                        "exact_match,flexible-fallback": 1.0,
                    }
                }
            }))
            errors: list[str] = []
            validate_campaign._check_aggregate_reconciliation(
                run_dir,
                [{"item_id": "0", "score": "0.0"}],
                "gsm8k",
                errors,
                fail_on_missing=True,
            )

            self.assertEqual(errors, [])

    def test_publication_reconciliation_rejects_benchmark_without_canonical_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp) / "run"
            raw_dir = run_dir / "raw"
            raw_dir.mkdir(parents=True)
            (raw_dir / "results_test.json").write_text(json.dumps({
                "results": {"new_benchmark": {"score,none": 1.0}}
            }))
            errors: list[str] = []
            validate_campaign._check_aggregate_reconciliation(
                run_dir,
                [{"item_id": "0", "score": "1.0"}],
                "new_benchmark",
                errors,
                fail_on_missing=True,
            )

            self.assertTrue(errors)
            self.assertIn("No canonical aggregate metric", errors[0])


if __name__ == "__main__":
    unittest.main()
