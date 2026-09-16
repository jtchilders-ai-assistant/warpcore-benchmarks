"""tests/test_validate_campaign.py — Failing RED tests for Task 7 campaign validator.

Tests for viz/validate_campaign.py which validates a completed campaign run before
it can be transitioned to 'validated' or 'published'.

Requirements covered:
  V1:  reject missing/duplicate item IDs
  V2:  reject missing raw samples or missing manifest fields
  V3:  reject aggregate mismatch with per-item evidence
  V4:  reject unclassified empty/timeout/transport/parser/SWE-bench failure
  V5:  reject stale suite/adapter hashes
  V6:  reject forbidden effective override (adapter overrides suite-owned fields)
  V7:  reject lifecycle other than 'current' during publication gate
  V8:  reject secret-scan failure (credentials in staged artifacts)
  V9:  reject DONE without successful harness exit and complete artifacts
  V10: historical fixtures remain visible without current-run hard gates (no blocking)
"""
from __future__ import annotations

import gzip
import json
import pathlib
import sys
import tempfile
import unittest

_TESTS_DIR = pathlib.Path(__file__).parent
_REPO = _TESTS_DIR.parent
_VIZ_DIR = _REPO / "viz"
for _p in (str(_TESTS_DIR), str(_VIZ_DIR), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import validate_campaign  # noqa: E402

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_ADAPTER_HASH = "a" * 64  # 64-char hex
_SUITE_HASH = "b" * 64
_PROFILE_DIGEST = "sha256:" + "c" * 64
_IMAGE_DIGEST = "sha256:" + "d" * 64
_REVISION = "e" * 40


def _valid_manifest(
    run_id="run-test",
    suite_id="warpcore-v1",
    benchmark="gsm8k",
    expected=10,
    submitted=10,
    adapter_hash=_ADAPTER_HASH,
    suite_hash=_SUITE_HASH,
    profile_digest=_PROFILE_DIGEST,
    image_digest=_IMAGE_DIGEST,
    revision=_REVISION,
) -> dict:
    return {
        "schema_version": 1,
        "suite_id": suite_id,
        "suite_schema_version": 1,
        "run_id": run_id,
        "benchmark": benchmark,
        "adapter_hash": adapter_hash,
        "suite_input_hashes": {
            "suite/warpcore-v1.yaml": suite_hash,
        },
        "serving_profile_digest": profile_digest,
        "model": {"slug": "test-model", "id": "testorg/TestModel", "revision": revision},
        "serving": {
            "image_digest": image_digest,
            "engine": "vllm",
            "engine_version": "0.6.6",
            "effective_args": [],
            "environment": {},
            "hardware_id": "dgx-spark-gb10",
        },
        "item_inventory": {"expected": expected, "submitted": submitted},
        "timing": {"started_utc": "2026-09-15T12:00:00Z", "completed_utc": "2026-09-15T14:00:00Z"},
        "artifact_inventory": {
            "samples_jsonl_gz": True,
            "per_item_csv": True,
            "run_log": True,
            "command_txt": True,
            "done_sentinel": True,
        },
    }


def _valid_status(
    run_id="run-test",
    suite_id="warpcore-v1",
    execution_state="completed",
    lifecycle="current",
) -> dict:
    history = [{"state": "planned", "timestamp": "2026-09-15T12:00:00Z"}]
    if execution_state == "failed":
        history += [
            {"state": "preflight_passed", "timestamp": "2026-09-15T13:00:00Z"},
            {"state": "running", "timestamp": "2026-09-15T14:00:00Z"},
            {"state": "failed", "timestamp": "2026-09-15T15:00:00Z"},
        ]
    else:
        states = ["planned", "preflight_passed", "running", "completed", "validated", "published"]
        idx = states.index(execution_state)
        for i, s in enumerate(states[1:idx+1], start=1):
            history.append({"state": s, "timestamp": f"2026-09-15T{12+i:02d}:00:00Z"})
    return {
        "schema_version": 1,
        "run_id": run_id,
        "suite_id": suite_id,
        "execution_state": execution_state,
        "lifecycle": lifecycle,
        "history": history,
    }


def _valid_per_item(n: int = 10, item_ids: list = None) -> list:
    """Return n valid per-item records with unique IDs."""
    ids = item_ids or list(range(n))
    return [
        {
            "item_id": str(i),
            "score": 1.0,
            "disposition": "correct",
            "finish_reason": "stop",
            "response_chars": 100,
            "empty_content": False,
        }
        for i in ids
    ]


def _valid_aggregate(n: int = 10, score: float = 1.0) -> dict:
    """Return a valid aggregate result."""
    return {
        "results": {
            "gsm8k": {
                "exact_match,flexible-fallback": score,
                "exact_match_stderr,flexible-fallback": 0.0,
            }
        },
        "n_samples": n,
    }


def _build_run_dir(
    tmp: pathlib.Path,
    suite_hash: str = _SUITE_HASH,
    adapter_hash: str = _ADAPTER_HASH,
    manifest: dict = None,
    status: dict = None,
    per_item: list = None,
    aggregate: dict = None,
    write_done: bool = True,
    write_samples: bool = True,
    write_run_log: bool = True,
    write_command: bool = True,
    n_items: int = 10,
    suite_yaml_content: str = None,
    adapter_yaml_content: str = None,
) -> tuple:
    """Build a minimal valid run directory in tmp. Returns (run_dir, suite_path, adapter_path)."""
    repo = tmp / "repo"
    suite_dir = repo / "suite"
    suite_dir.mkdir(parents=True, exist_ok=True)

    # Write suite YAML with correct hash
    suite_yaml = suite_yaml_content or f"suite_id: warpcore-v1\n# hash placeholder\n"
    suite_path = suite_dir / "warpcore-v1.yaml"
    suite_path.write_text(suite_yaml)

    # Compute actual hash if not overridden
    import hashlib
    actual_suite_hash = hashlib.sha256(suite_path.read_bytes()).hexdigest()

    # Adapter
    adapters_dir = repo / "adapters"
    adapters_dir.mkdir(parents=True, exist_ok=True)
    adapter_yaml = adapter_yaml_content or "adapter_schema_version: 1\ncampaign_status: canonical\n"
    adapter_path = adapters_dir / "test-model.yaml"
    adapter_path.write_text(adapter_yaml)
    actual_adapter_hash = hashlib.sha256(adapter_path.read_bytes()).hexdigest()

    # Build manifest with real hashes (unless caller provides specific hashes)
    use_suite_hash = suite_hash if suite_hash != _SUITE_HASH else actual_suite_hash
    use_adapter_hash = adapter_hash if adapter_hash != _ADAPTER_HASH else actual_adapter_hash

    if manifest is None:
        manifest = _valid_manifest(
            suite_hash=use_suite_hash,
            adapter_hash=use_adapter_hash,
            expected=n_items,
            submitted=n_items,
        )

    if status is None:
        status = _valid_status()

    # Run dir
    run_dir = repo / "results" / "test-model" / "runs" / "warpcore-v1" / "gsm8k" / "run-test"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    (run_dir / "status.json").write_text(json.dumps(status))

    if write_done:
        (run_dir / "DONE").write_text("done\n")

    if write_run_log:
        (run_dir / "run.log").write_text("harness log\n")

    if write_command:
        (run_dir / "command.txt").write_text("python3 run_quality.py\n")

    # Raw subdir
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    if write_samples:
        # Write a gzipped samples file
        samples_path = raw_dir / "samples_gsm8k.jsonl.gz"
        with gzip.open(samples_path, "wt") as f:
            for rec in (per_item or _valid_per_item(n_items)):
                f.write(json.dumps(rec) + "\n")

    if aggregate is not None:
        (raw_dir / "results_test.json").write_text(json.dumps(aggregate))

    # Per-item CSV
    import csv
    import io
    items = per_item or _valid_per_item(n_items)
    csv_buf = io.StringIO()
    w = csv.DictWriter(csv_buf, fieldnames=["item_id", "score", "disposition",
                                            "finish_reason", "response_chars", "empty_content"])
    w.writeheader()
    w.writerows(items)
    (run_dir / "per_item.csv").write_text(csv_buf.getvalue())

    return run_dir, suite_path, adapter_path


# ---------------------------------------------------------------------------
# Test: module exists and has expected interface
# ---------------------------------------------------------------------------

class TestValidateCampaignModule(unittest.TestCase):
    """Smoke-test: the module exists and exports the expected API."""

    def test_validate_function_exists(self):
        """validate_campaign module must export a validate() function."""
        self.assertTrue(hasattr(validate_campaign, "validate"),
                        "validate_campaign must export validate()")

    def test_validate_returns_result_with_errors(self):
        """validate() must return an object with an .errors attribute."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, suite_path, adapter_path = _build_run_dir(pathlib.Path(tmp))
            result = validate_campaign.validate(run_dir, suite_path, adapter_path)
            self.assertTrue(hasattr(result, "errors"),
                            "validate() result must have .errors attribute")

    def test_validate_returns_result_with_passed(self):
        """validate() must return an object with a .passed attribute."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, suite_path, adapter_path = _build_run_dir(pathlib.Path(tmp))
            result = validate_campaign.validate(run_dir, suite_path, adapter_path)
            self.assertTrue(hasattr(result, "passed"),
                            "validate() result must have .passed attribute")

    def test_valid_fixture_passes(self):
        """A fully valid fixture must pass validation."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, suite_path, adapter_path = _build_run_dir(pathlib.Path(tmp))
            result = validate_campaign.validate(run_dir, suite_path, adapter_path)
            self.assertTrue(result.passed,
                            f"Valid fixture must pass; errors: {result.errors}")


# ---------------------------------------------------------------------------
# Test: V1 — missing/duplicate item IDs
# ---------------------------------------------------------------------------

class TestItemIDRejection(unittest.TestCase):

    def test_duplicate_item_ids_rejected(self):
        """Duplicate item IDs in per_item evidence must be rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            # Create 10 items but IDs 0..9 with item 0 duplicated
            dup_items = _valid_per_item(10)
            dup_items[5] = dup_items[0].copy()  # same item_id as index 0
            run_dir, suite_path, adapter_path = _build_run_dir(
                pathlib.Path(tmp), per_item=dup_items
            )
            result = validate_campaign.validate(run_dir, suite_path, adapter_path)
            self.assertFalse(result.passed, "Duplicate item IDs must be rejected")
            self.assertTrue(
                any("duplicate" in e.lower() or "dup" in e.lower() for e in result.errors),
                f"Expected 'duplicate' in errors; got: {result.errors}"
            )

    def test_missing_item_ids_rejected(self):
        """Fewer items than expected must be rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            # Manifest says expected=10 but only 7 items in per_item
            short_items = _valid_per_item(7)
            manifest = _valid_manifest(expected=10, submitted=7)
            # Fix hashes
            import hashlib
            tmp_path = pathlib.Path(tmp)
            suite_path_tmp = tmp_path / "repo" / "suite" / "warpcore-v1.yaml"
            suite_path_tmp.parent.mkdir(parents=True, exist_ok=True)
            suite_path_tmp.write_text("suite_id: warpcore-v1\n")
            actual_suite_hash = hashlib.sha256(suite_path_tmp.read_bytes()).hexdigest()
            adapter_path_tmp = tmp_path / "repo" / "adapters" / "test-model.yaml"
            adapter_path_tmp.parent.mkdir(parents=True, exist_ok=True)
            adapter_path_tmp.write_text("adapter_schema_version: 1\ncampaign_status: canonical\n")
            actual_adapter_hash = hashlib.sha256(adapter_path_tmp.read_bytes()).hexdigest()
            manifest["suite_input_hashes"]["suite/warpcore-v1.yaml"] = actual_suite_hash
            manifest["adapter_hash"] = actual_adapter_hash

            run_dir, suite_path, adapter_path = _build_run_dir(
                pathlib.Path(tmp), per_item=short_items, manifest=manifest, n_items=7
            )
            result = validate_campaign.validate(run_dir, suite_path, adapter_path)
            self.assertFalse(result.passed, "Missing items vs expected count must be rejected")
            self.assertTrue(
                any("item" in e.lower() or "count" in e.lower() or "missing" in e.lower()
                    for e in result.errors),
                f"Expected item-count error; got: {result.errors}"
            )


# ---------------------------------------------------------------------------
# Test: V2 — missing raw samples or manifest fields
# ---------------------------------------------------------------------------

class TestMissingArtifacts(unittest.TestCase):

    def test_missing_samples_rejected(self):
        """Missing raw samples file must be rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, suite_path, adapter_path = _build_run_dir(
                pathlib.Path(tmp), write_samples=False
            )
            result = validate_campaign.validate(run_dir, suite_path, adapter_path)
            self.assertFalse(result.passed, "Missing samples must be rejected")
            self.assertTrue(
                any("sample" in e.lower() or "artifact" in e.lower() for e in result.errors),
                f"Expected artifact error; got: {result.errors}"
            )

    def test_missing_run_log_rejected(self):
        """Missing run.log must be rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, suite_path, adapter_path = _build_run_dir(
                pathlib.Path(tmp), write_run_log=False
            )
            result = validate_campaign.validate(run_dir, suite_path, adapter_path)
            self.assertFalse(result.passed, "Missing run.log must be rejected")

    def test_missing_done_sentinel_rejected(self):
        """Missing DONE sentinel must be rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, suite_path, adapter_path = _build_run_dir(
                pathlib.Path(tmp), write_done=False
            )
            result = validate_campaign.validate(run_dir, suite_path, adapter_path)
            self.assertFalse(result.passed, "Missing DONE sentinel must be rejected")
            self.assertTrue(
                any("done" in e.lower() or "sentinel" in e.lower() for e in result.errors),
                f"Expected DONE-sentinel error; got: {result.errors}"
            )

    def test_missing_manifest_field_rejected(self):
        """Manifest missing required field must be rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _valid_manifest()
            del manifest["adapter_hash"]  # remove required field
            run_dir, suite_path, adapter_path = _build_run_dir(
                pathlib.Path(tmp), manifest=manifest
            )
            result = validate_campaign.validate(run_dir, suite_path, adapter_path)
            self.assertFalse(result.passed, "Manifest missing required field must be rejected")

    def test_missing_command_txt_rejected(self):
        """Missing command.txt must be rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, suite_path, adapter_path = _build_run_dir(
                pathlib.Path(tmp), write_command=False
            )
            result = validate_campaign.validate(run_dir, suite_path, adapter_path)
            self.assertFalse(result.passed, "Missing command.txt must be rejected")


# ---------------------------------------------------------------------------
# Test: V3 — aggregate mismatch with per-item evidence
# ---------------------------------------------------------------------------

class TestAggregateMismatch(unittest.TestCase):

    def test_aggregate_mismatch_rejected(self):
        """Aggregate score that does not match per-item sum must be rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            # All items correct (score=1.0) but aggregate claims 0.5
            per_item = _valid_per_item(10)
            # Per-item: 10/10 = 100%
            # Aggregate: claims 50%
            aggregate = _valid_aggregate(10, score=0.5)
            run_dir, suite_path, adapter_path = _build_run_dir(
                pathlib.Path(tmp), per_item=per_item, aggregate=aggregate
            )
            result = validate_campaign.validate(run_dir, suite_path, adapter_path)
            self.assertFalse(result.passed,
                             "Aggregate mismatch with per-item evidence must be rejected")
            self.assertTrue(
                any("mismatch" in e.lower() or "aggregate" in e.lower()
                    or "reconcil" in e.lower() for e in result.errors),
                f"Expected aggregate-mismatch error; got: {result.errors}"
            )


# ---------------------------------------------------------------------------
# Test: V4 — unclassified empty/timeout/transport/parser/SWE-bench failure
# ---------------------------------------------------------------------------

class TestUnclassifiedFailures(unittest.TestCase):

    def test_unclassified_empty_response_rejected(self):
        """Items with empty content and unclassified disposition must be rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            items = _valid_per_item(10)
            # Item 3 is empty but not classified
            items[3] = {
                "item_id": "3",
                "score": 0.0,
                "disposition": "unclassified",  # not classified
                "finish_reason": "stop",
                "response_chars": 0,
                "empty_content": True,
            }
            run_dir, suite_path, adapter_path = _build_run_dir(
                pathlib.Path(tmp), per_item=items
            )
            result = validate_campaign.validate(run_dir, suite_path, adapter_path)
            self.assertFalse(result.passed,
                             "Unclassified empty response must be rejected")
            self.assertTrue(
                any("unclassif" in e.lower() or "classif" in e.lower()
                    or "disposition" in e.lower() for e in result.errors),
                f"Expected classification error; got: {result.errors}"
            )

    def test_unclassified_timeout_rejected(self):
        """Timeout items not classified must be rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            items = _valid_per_item(10)
            items[2] = {
                "item_id": "2",
                "score": 0.0,
                "disposition": "unclassified",
                "finish_reason": "timeout",
                "response_chars": 0,
                "empty_content": True,
            }
            run_dir, suite_path, adapter_path = _build_run_dir(
                pathlib.Path(tmp), per_item=items
            )
            result = validate_campaign.validate(run_dir, suite_path, adapter_path)
            self.assertFalse(result.passed, "Unclassified timeout must be rejected")


# ---------------------------------------------------------------------------
# Test: V5 — stale suite/adapter hashes
# ---------------------------------------------------------------------------

class TestStaleHashes(unittest.TestCase):

    def test_stale_suite_hash_rejected(self):
        """Manifest suite hash that does not match actual file must be rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            # Use a wrong suite hash in the manifest (file changed since recording)
            wrong_hash = "f" * 64
            manifest = _valid_manifest(suite_hash=wrong_hash)
            run_dir, suite_path, adapter_path = _build_run_dir(
                pathlib.Path(tmp), manifest=manifest
            )
            result = validate_campaign.validate(run_dir, suite_path, adapter_path)
            self.assertFalse(result.passed, "Stale suite hash must be rejected")
            self.assertTrue(
                any("hash" in e.lower() or "stale" in e.lower()
                    or "mismatch" in e.lower() for e in result.errors),
                f"Expected hash error; got: {result.errors}"
            )

    def test_stale_adapter_hash_rejected(self):
        """Manifest adapter hash that does not match actual file must be rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            wrong_adapter_hash = "9" * 64
            manifest = _valid_manifest(adapter_hash=wrong_adapter_hash)
            run_dir, suite_path, adapter_path = _build_run_dir(
                pathlib.Path(tmp), manifest=manifest
            )
            result = validate_campaign.validate(run_dir, suite_path, adapter_path)
            self.assertFalse(result.passed, "Stale adapter hash must be rejected")
            self.assertTrue(
                any("hash" in e.lower() or "adapter" in e.lower() for e in result.errors),
                f"Expected adapter hash error; got: {result.errors}"
            )


# ---------------------------------------------------------------------------
# Test: V6 — forbidden effective override
# ---------------------------------------------------------------------------

class TestForbiddenOverride(unittest.TestCase):

    def test_forbidden_task_override_rejected(self):
        """Effective args containing task override must be rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _valid_manifest()
            # Inject a forbidden override in effective_args
            manifest["serving"]["effective_args"] = ["--tasks=my_custom_task"]
            run_dir, suite_path, adapter_path = _build_run_dir(
                pathlib.Path(tmp), manifest=manifest
            )
            result = validate_campaign.validate(run_dir, suite_path, adapter_path)
            self.assertFalse(result.passed, "Forbidden task override must be rejected")
            self.assertTrue(
                any("override" in e.lower() or "forbidden" in e.lower()
                    or "task" in e.lower() for e in result.errors),
                f"Expected override error; got: {result.errors}"
            )

    def test_forbidden_prompt_override_rejected(self):
        """Effective args containing prompt override must be rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _valid_manifest()
            manifest["serving"]["effective_args"] = ["--system_instruction=custom"]
            run_dir, suite_path, adapter_path = _build_run_dir(
                pathlib.Path(tmp), manifest=manifest
            )
            result = validate_campaign.validate(run_dir, suite_path, adapter_path)
            self.assertFalse(result.passed, "Forbidden prompt override must be rejected")


# ---------------------------------------------------------------------------
# Test: V7 — lifecycle other than 'current' during publication gate
# ---------------------------------------------------------------------------

class TestLifecycleGate(unittest.TestCase):

    def test_historical_lifecycle_blocks_publication(self):
        """A run with lifecycle='historical' must not pass the publication gate."""
        with tempfile.TemporaryDirectory() as tmp:
            status = _valid_status(lifecycle="historical", execution_state="validated")
            run_dir, suite_path, adapter_path = _build_run_dir(
                pathlib.Path(tmp), status=status
            )
            result = validate_campaign.validate(
                run_dir, suite_path, adapter_path, for_publication=True
            )
            self.assertFalse(result.passed,
                             "Historical lifecycle must block publication gate")
            self.assertTrue(
                any("lifecycle" in e.lower() or "current" in e.lower()
                    or "historical" in e.lower() for e in result.errors),
                f"Expected lifecycle error; got: {result.errors}"
            )

    def test_superseded_lifecycle_blocks_publication(self):
        """A run with lifecycle='superseded' must not pass the publication gate."""
        with tempfile.TemporaryDirectory() as tmp:
            status = _valid_status(lifecycle="superseded", execution_state="validated")
            run_dir, suite_path, adapter_path = _build_run_dir(
                pathlib.Path(tmp), status=status
            )
            result = validate_campaign.validate(
                run_dir, suite_path, adapter_path, for_publication=True
            )
            self.assertFalse(result.passed,
                             "Superseded lifecycle must block publication gate")

    def test_current_lifecycle_passes_publication(self):
        """A run with lifecycle='current' must pass the lifecycle check."""
        with tempfile.TemporaryDirectory() as tmp:
            status = _valid_status(lifecycle="current", execution_state="validated")
            run_dir, suite_path, adapter_path = _build_run_dir(
                pathlib.Path(tmp), status=status
            )
            result = validate_campaign.validate(
                run_dir, suite_path, adapter_path, for_publication=True
            )
            # Lifecycle gate should pass (other gates may fail but not lifecycle)
            lifecycle_errors = [e for e in result.errors
                                if "lifecycle" in e.lower() or "current" in e.lower()]
            self.assertEqual(lifecycle_errors, [],
                             f"lifecycle='current' must not trigger lifecycle error; "
                             f"got: {lifecycle_errors}")

    def test_historical_passes_without_publication_gate(self):
        """Historical lifecycle must return eligible=False, not certified.

        B5 design: historical runs must not be certified by the v1 validator.
        validate() must return passed=False + eligible=False for historical runs.
        The key semantics: the run is NOT hard-blocked in an error sense, but
        it is ineligible for certification. Publication excludes it.
        """
        with tempfile.TemporaryDirectory() as tmp:
            status = _valid_status(lifecycle="historical", execution_state="completed")
            run_dir, suite_path, adapter_path = _build_run_dir(
                pathlib.Path(tmp), status=status
            )
            result = validate_campaign.validate(
                run_dir, suite_path, adapter_path, for_publication=False
            )
            # B5: Historical runs must not be certified (passed=False, eligible=False)
            self.assertFalse(
                result.passed,
                "Historical run must not receive passed=True from the v1 validator."
            )
            self.assertFalse(
                getattr(result, "eligible", True),
                "Historical run must have eligible=False from the v1 validator."
            )
            # The error message must explain ineligibility, not just 'lifecycle' or 'publication'
            self.assertGreater(
                len(result.errors), 0,
                "Historical run must have at least one error explaining ineligibility."
            )


# ---------------------------------------------------------------------------
# Test: V8 — secret-scan failure
# ---------------------------------------------------------------------------

class TestSecretScan(unittest.TestCase):

    def test_credential_in_command_txt_rejected(self):
        """command.txt containing what looks like an API key must be rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, suite_path, adapter_path = _build_run_dir(pathlib.Path(tmp))
            # Write a command that looks like it has a credential
            (run_dir / "command.txt").write_text(
                "python3 run_quality.py --api-key sk-1234567890abcdef1234567890abcdef\n"
            )
            result = validate_campaign.validate(run_dir, suite_path, adapter_path)
            self.assertFalse(result.passed, "Secret in command.txt must be rejected")
            self.assertTrue(
                any("secret" in e.lower() or "credential" in e.lower()
                    or "api" in e.lower() or "key" in e.lower() for e in result.errors),
                f"Expected secret-scan error; got: {result.errors}"
            )

    def test_clean_artifacts_pass_secret_scan(self):
        """Clean artifacts must pass secret scan."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, suite_path, adapter_path = _build_run_dir(pathlib.Path(tmp))
            result = validate_campaign.validate(run_dir, suite_path, adapter_path)
            secret_errors = [e for e in result.errors
                             if "secret" in e.lower() or "credential" in e.lower()]
            self.assertEqual(secret_errors, [],
                             f"Clean artifacts must not trigger secret-scan errors; "
                             f"got: {secret_errors}")


# ---------------------------------------------------------------------------
# Test: V9 — DONE without successful harness exit
# ---------------------------------------------------------------------------

class TestDoneSentinel(unittest.TestCase):

    def test_done_requires_successful_status(self):
        """DONE sentinel present but status=failed must be rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            status = _valid_status(execution_state="failed")
            run_dir, suite_path, adapter_path = _build_run_dir(
                pathlib.Path(tmp), status=status, write_done=True
            )
            result = validate_campaign.validate(run_dir, suite_path, adapter_path)
            self.assertFalse(result.passed,
                             "DONE with failed status must be rejected")
            self.assertTrue(
                any("done" in e.lower() or "failed" in e.lower()
                    or "sentinel" in e.lower() for e in result.errors),
                f"Expected DONE/failed conflict error; got: {result.errors}"
            )

    def test_done_without_complete_artifacts_rejected(self):
        """DONE but missing artifact inventory fields marked True must be rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _valid_manifest()
            # Claim samples present but don't actually write them
            manifest["artifact_inventory"]["samples_jsonl_gz"] = True
            run_dir, suite_path, adapter_path = _build_run_dir(
                pathlib.Path(tmp), manifest=manifest, write_samples=False
            )
            result = validate_campaign.validate(run_dir, suite_path, adapter_path)
            self.assertFalse(result.passed,
                             "DONE without complete artifacts must be rejected")


# ---------------------------------------------------------------------------
# Test: V10 — Historical fixtures remain visible without hard gates
# ---------------------------------------------------------------------------

class TestHistoricalFixtures(unittest.TestCase):

    def test_historical_run_dir_visible_in_discovery(self):
        """Historical layout (results/<model>/raw/) must be discoverable."""
        # This tests that discover_runs() finds historical layout paths
        self.assertTrue(hasattr(validate_campaign, "discover_runs"),
                        "validate_campaign must export discover_runs()")

    def test_historical_run_not_blocked_without_publication_gate(self):
        """Historical run validated without for_publication=True must not be hard-blocked."""
        with tempfile.TemporaryDirectory() as tmp:
            # Historical layout: results/<model>/raw/
            repo = pathlib.Path(tmp) / "repo"
            model_dir = repo / "results" / "hist-model" / "raw"
            model_dir.mkdir(parents=True, exist_ok=True)

            # Create a minimal historical manifest (may have 'unrecorded' revision)
            hist_manifest = {
                "schema_version": 1,
                "suite_id": "warpcore-v1",
                "suite_schema_version": 1,
                "run_id": "hist-run-001",
                "benchmark": "gsm8k",
                "adapter_hash": "u" * 64,
                "suite_input_hashes": {"suite/warpcore-v1.yaml": "u" * 64},
                "serving_profile_digest": "sha256:" + "u" * 64,
                "model": {
                    "slug": "hist-model",
                    "id": "hist/hist-model",
                    "revision": "unrecorded",  # historical: revision was not recorded
                },
                "serving": {
                    "image_digest": "unrecorded",
                    "engine": "vllm",
                    "engine_version": "unrecorded",
                    "effective_args": [],
                    "environment": {},
                    "hardware_id": "dgx-spark-gb10",
                },
                "item_inventory": {"expected": 5, "submitted": 5},
                "timing": {
                    "started_utc": "2026-08-01T12:00:00Z",
                    "completed_utc": "2026-08-01T14:00:00Z",
                },
                "artifact_inventory": {
                    "samples_jsonl_gz": False,
                    "per_item_csv": False,
                    "run_log": False,
                    "command_txt": False,
                    "done_sentinel": False,
                },
            }
            (model_dir / "manifest.json").write_text(json.dumps(hist_manifest))
            hist_status = _valid_status(
                run_id="hist-run-001", lifecycle="historical", execution_state="completed"
            )
            (model_dir / "status.json").write_text(json.dumps(hist_status))

            # discover_runs should find this
            runs = validate_campaign.discover_runs(repo)
            slugs = [r.get("model_slug", "") for r in runs]
            self.assertIn("hist-model", slugs,
                          f"Historical run must be discoverable; found slugs: {slugs}")


if __name__ == "__main__":
    unittest.main()
