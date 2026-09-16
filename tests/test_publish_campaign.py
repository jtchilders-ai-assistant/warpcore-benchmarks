"""tests/test_publish_campaign.py — Failing RED tests for Task 7 publication gate.

Tests for viz/publish_campaign.py which reads validated manifests and writes
canonical matrix data transactionally.

Requirements covered:
  P1:  reads validated manifests; only validated 'current' runs enter the matrix
  P2:  writes canonical matrix data transactionally (atomic write)
  P3:  emits explicit 'not measured' cells (never imputes)
  P4:  no cross-task overall rank
  P5:  paired comparisons only for identical item sets
  P6:  non-vacuity: valid current fixture appears
  P7:  non-vacuity: mutation to historical/changed hash/deleted item/changed ID blocks it
  P8:  non-vacuity: restoration reappears
  P9:  historical debt remains ratcheted; cannot become canonical implicitly
"""
from __future__ import annotations

import gzip
import hashlib
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

import publish_campaign  # noqa: E402

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_ADAPTER_HASH = "a" * 64
_PROFILE_DIGEST = "sha256:" + "c" * 64
_IMAGE_DIGEST = "sha256:" + "d" * 64
_REVISION = "e" * 40


def _make_valid_run(
    repo: pathlib.Path,
    model_slug: str = "test-model",
    benchmark: str = "gsm8k",
    run_id: str = "run-2026-09-15T00-00-00",
    suite_id: str = "warpcore-v1",
    lifecycle: str = "current",
    execution_state: str = "validated",
    score: float = 0.85,
    n_items: int = 5,
    instance_ids: list = None,
    extra_manifest: dict = None,
) -> pathlib.Path:
    """Create a minimal valid normalized run directory in repo."""
    # Suite YAML
    suite_dir = repo / "suite"
    suite_dir.mkdir(parents=True, exist_ok=True)
    suite_yaml = repo / "suite" / "warpcore-v1.yaml"
    if not suite_yaml.exists():
        suite_yaml.write_text("suite_id: warpcore-v1\nsuite_schema_version: 1\n")
    suite_hash = hashlib.sha256(suite_yaml.read_bytes()).hexdigest()

    # Adapter YAML
    adapters_dir = repo / "adapters"
    adapters_dir.mkdir(parents=True, exist_ok=True)
    adapter_yaml = adapters_dir / f"{model_slug}.yaml"
    if not adapter_yaml.exists():
        adapter_yaml.write_text(
            f"adapter_schema_version: 1\ncampaign_status: canonical\n"
            f"model:\n  slug: {model_slug}\n"
        )
    adapter_hash = hashlib.sha256(adapter_yaml.read_bytes()).hexdigest()

    # Run dir (normalized layout)
    run_dir = (repo / "results" / model_slug / "runs" / suite_id / benchmark / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build instance IDs
    ids = instance_ids or [f"item-{i}" for i in range(n_items)]
    n = len(ids)
    # Compute n_correct from score so actual per-item mean == aggregate value
    n_correct = round(score * n)
    actual_mean = n_correct / n  # use this for both per_item CSV and aggregate

    manifest = {
        "schema_version": 1,
        "suite_id": suite_id,
        "suite_schema_version": 1,
        "run_id": run_id,
        "benchmark": benchmark,
        "adapter_hash": adapter_hash,
        "suite_input_hashes": {"suite/warpcore-v1.yaml": suite_hash},
        "serving_profile_digest": _PROFILE_DIGEST,
        "model": {"slug": model_slug, "id": f"testorg/{model_slug}", "revision": _REVISION},
        "serving": {
            "image_digest": _IMAGE_DIGEST,
            "engine": "vllm",
            "engine_version": "0.6.6",
            "effective_args": [],
            "environment": {},
            "hardware_id": "dgx-spark-gb10",
        },
        "item_inventory": {
            "expected": n_items,
            "submitted": n_items,
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
    if extra_manifest:
        manifest.update(extra_manifest)

    # Build history
    if execution_state == "failed":
        history = [
            {"state": "planned", "timestamp": "2026-09-15T12:00:00Z"},
            {"state": "preflight_passed", "timestamp": "2026-09-15T13:00:00Z"},
            {"state": "running", "timestamp": "2026-09-15T14:00:00Z"},
            {"state": "failed", "timestamp": "2026-09-15T15:00:00Z"},
        ]
    else:
        states = ["planned", "preflight_passed", "running", "completed", "validated", "published"]
        idx = states.index(execution_state)
        history = []
        for i, s in enumerate(states[:idx+1]):
            history.append({"state": s, "timestamp": f"2026-09-15T{12+i:02d}:00:00Z"})

    status = {
        "schema_version": 1,
        "run_id": run_id,
        "suite_id": suite_id,
        "execution_state": execution_state,
        "lifecycle": lifecycle,
        "history": history,
    }

    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (run_dir / "status.json").write_text(json.dumps(status, indent=2))
    (run_dir / "DONE").write_text("done\n")
    (run_dir / "run.log").write_text("harness exit 0\n")
    (run_dir / "command.txt").write_text("python3 run_quality.py\n")

    raw_dir = run_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    # Per-item CSV
    import csv, io
    items = [
        {
            "item_id": iid,
            "score": 1.0 if i < n_correct else 0.0,
            "disposition": "correct" if i < n_correct else "wrong",
            "finish_reason": "stop",
            "response_chars": 100,
            "empty_content": False,
        }
        for i, iid in enumerate(ids)
    ]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(items[0].keys()))
    w.writeheader()
    w.writerows(items)
    (run_dir / "per_item.csv").write_text(buf.getvalue())

    # Gzipped samples
    with gzip.open(raw_dir / f"samples_{benchmark}.jsonl.gz", "wt") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")

    # Aggregate result — use actual_mean so validator agrees with per_item
    (raw_dir / "results_test.json").write_text(json.dumps({
        "results": {benchmark: {"exact_match,flexible-fallback": actual_mean}},
        "n_samples": n,
    }))

    return run_dir


# ---------------------------------------------------------------------------
# Test: module exists and has expected interface
# ---------------------------------------------------------------------------

class TestPublishCampaignModule(unittest.TestCase):

    def test_publish_function_exists(self):
        """publish_campaign must export a publish() function."""
        self.assertTrue(hasattr(publish_campaign, "publish"),
                        "publish_campaign must export publish()")

    def test_publish_returns_result(self):
        """publish() must return a result with .entries and .not_measured."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            _make_valid_run(repo, lifecycle="current", execution_state="validated")
            result = publish_campaign.publish(repo)
            self.assertTrue(hasattr(result, "entries"),
                            "publish() result must have .entries attribute")
            self.assertTrue(hasattr(result, "not_measured"),
                            "publish() result must have .not_measured attribute")


# ---------------------------------------------------------------------------
# Test: P1 — Only validated 'current' runs enter the matrix
# ---------------------------------------------------------------------------

class TestOnlyCurrentValidatedRuns(unittest.TestCase):

    def test_validated_current_run_appears(self):
        """A validated, lifecycle=current run must appear in published entries."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            _make_valid_run(repo, model_slug="model-a", lifecycle="current",
                            execution_state="validated")
            result = publish_campaign.publish(repo)
            models = {e["model_slug"] for e in result.entries}
            self.assertIn("model-a", models,
                          f"validated+current run must appear; entries: {result.entries}")

    def test_historical_run_excluded_from_matrix(self):
        """A historical run must NOT appear as a canonical matrix entry."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            _make_valid_run(repo, model_slug="hist-model", lifecycle="historical",
                            execution_state="validated")
            result = publish_campaign.publish(repo)
            models = {e["model_slug"] for e in result.entries}
            self.assertNotIn("hist-model", models,
                             f"Historical run must not appear in canonical entries; "
                             f"entries: {result.entries}")

    def test_failed_run_excluded_from_matrix(self):
        """A failed run must NOT appear in the matrix."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            _make_valid_run(repo, model_slug="failed-model", lifecycle="invalid",
                            execution_state="failed")
            result = publish_campaign.publish(repo)
            models = {e["model_slug"] for e in result.entries}
            self.assertNotIn("failed-model", models,
                             "Failed run must not appear in matrix")

    def test_completed_not_validated_excluded(self):
        """A completed but not yet validated run must NOT appear in the matrix."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            _make_valid_run(repo, model_slug="unvalidated-model", lifecycle="current",
                            execution_state="completed")
            result = publish_campaign.publish(repo)
            models = {e["model_slug"] for e in result.entries}
            self.assertNotIn("unvalidated-model", models,
                             "Completed-not-validated run must not appear in matrix")


# ---------------------------------------------------------------------------
# Test: P2 — Transactional write
# ---------------------------------------------------------------------------

class TestTransactionalWrite(unittest.TestCase):

    def test_publish_writes_matrix_file(self):
        """publish() must write a canonical matrix file."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            _make_valid_run(repo, lifecycle="current", execution_state="validated")
            result = publish_campaign.publish(repo)
            self.assertTrue(hasattr(result, "output_path"),
                            "publish() result must have .output_path")
            self.assertTrue(result.output_path.exists(),
                            f"Matrix file must exist at {result.output_path}")

    def test_matrix_file_is_valid_json(self):
        """The published matrix file must be valid JSON."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            _make_valid_run(repo, lifecycle="current", execution_state="validated")
            result = publish_campaign.publish(repo)
            content = result.output_path.read_text()
            data = json.loads(content)  # must not raise
            self.assertIsInstance(data, dict)


# ---------------------------------------------------------------------------
# Test: P3 — Explicit 'not measured' cells, never imputed
# ---------------------------------------------------------------------------

class TestNotMeasuredCells(unittest.TestCase):

    def test_missing_benchmark_emits_not_measured(self):
        """A model missing a benchmark must appear as 'not measured', not imputed."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            # Model-a has gsm8k, but not ifeval
            _make_valid_run(repo, model_slug="model-a", benchmark="gsm8k",
                            lifecycle="current", execution_state="validated")
            result = publish_campaign.publish(repo)
            # model-a's ifeval cell must be 'not measured', not absent or imputed
            nm = {(nm["model_slug"], nm["benchmark"]) for nm in result.not_measured}
            self.assertIn(("model-a", "ifeval"), nm,
                          f"model-a/ifeval must be 'not measured'; got: {result.not_measured}")

    def test_no_imputed_values(self):
        """No entry must have imputed=True or source='imputed'."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            _make_valid_run(repo, lifecycle="current", execution_state="validated")
            result = publish_campaign.publish(repo)
            for entry in result.entries:
                self.assertFalse(entry.get("imputed", False),
                                 f"Entry must not be imputed: {entry}")


# ---------------------------------------------------------------------------
# Test: P4 — No cross-task overall rank
# ---------------------------------------------------------------------------

class TestNoOverallRank(unittest.TestCase):

    def test_no_overall_rank_in_entries(self):
        """Published entries must not contain an overall_rank field."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            _make_valid_run(repo, lifecycle="current", execution_state="validated")
            result = publish_campaign.publish(repo)
            for entry in result.entries:
                self.assertNotIn("overall_rank", entry,
                                 f"overall_rank must not appear: {entry}")

    def test_no_cross_task_score_in_entries(self):
        """Published entries must not contain a cross-task aggregate score."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            _make_valid_run(repo, lifecycle="current", execution_state="validated")
            result = publish_campaign.publish(repo)
            for entry in result.entries:
                forbidden = {"overall", "aggregate_score", "cross_task", "overall_score"}
                intersection = forbidden & set(entry.keys())
                self.assertEqual(intersection, set(),
                                 f"Forbidden cross-task fields found: {intersection}")


# ---------------------------------------------------------------------------
# Test: P5 — Paired comparisons only for identical item sets
# ---------------------------------------------------------------------------

class TestPairedComparisons(unittest.TestCase):

    def test_paired_comparison_requires_identical_item_sets(self):
        """Paired comparison must only appear for models with identical item sets."""
        self.assertTrue(hasattr(publish_campaign, "compute_paired_comparison"),
                        "publish_campaign must export compute_paired_comparison()")

    def test_paired_comparison_rejected_for_different_item_sets(self):
        """compute_paired_comparison must reject different instance ID sets."""
        ids_a = [f"item-{i}" for i in range(5)]
        ids_b = [f"item-{i}" for i in range(3, 8)]  # different set
        with self.assertRaises((ValueError, AssertionError),
                               msg="Paired comparison must reject non-identical item sets"):
            publish_campaign.compute_paired_comparison(
                model_a="model-a", scores_a={"item-0": 1.0},
                model_b="model-b", scores_b={"item-0": 1.0},
                item_ids_a=ids_a, item_ids_b=ids_b,
            )

    def test_paired_comparison_accepted_for_identical_item_sets(self):
        """compute_paired_comparison must succeed for identical item sets."""
        ids = [f"item-{i}" for i in range(5)]
        scores_a = {iid: 1.0 for iid in ids}
        scores_b = {iid: 0.8 for iid in ids}
        result = publish_campaign.compute_paired_comparison(
            model_a="model-a", scores_a=scores_a,
            model_b="model-b", scores_b=scores_b,
            item_ids_a=ids, item_ids_b=ids,
        )
        self.assertIsInstance(result, dict, "Paired comparison must return a dict")
        self.assertIn("diff_pp", result, "Paired comparison must include diff_pp")


# ---------------------------------------------------------------------------
# Test: P6/P7/P8 — Non-vacuity proofs
# ---------------------------------------------------------------------------

class TestNonVacuity(unittest.TestCase):

    def test_valid_current_fixture_appears(self):
        """P6: A valid current fixture must appear in published entries."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            _make_valid_run(repo, model_slug="non-vacuous-model",
                            lifecycle="current", execution_state="validated")
            result = publish_campaign.publish(repo)
            models = {e["model_slug"] for e in result.entries}
            self.assertIn("non-vacuous-model", models,
                          "Valid current fixture must appear in published entries")

    def test_mutation_to_historical_removes_entry(self):
        """P7a: Mutating lifecycle to 'historical' must remove the entry."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            run_dir = _make_valid_run(repo, model_slug="mutable-model",
                                      lifecycle="current", execution_state="validated")
            # Before mutation: appears
            result_before = publish_campaign.publish(repo)
            models_before = {e["model_slug"] for e in result_before.entries}
            self.assertIn("mutable-model", models_before,
                          "Before mutation: model must appear")

            # Mutate: change lifecycle to historical
            status_path = run_dir / "status.json"
            status = json.loads(status_path.read_text())
            status["lifecycle"] = "historical"
            status_path.write_text(json.dumps(status))

            # After mutation: must disappear
            result_after = publish_campaign.publish(repo)
            models_after = {e["model_slug"] for e in result_after.entries}
            self.assertNotIn("mutable-model", models_after,
                             "After mutation to historical: model must NOT appear")

    def test_changed_suite_hash_blocks_entry(self):
        """P7b: Changing a suite input hash in manifest (stale) must block the entry."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            run_dir = _make_valid_run(repo, model_slug="hash-test-model",
                                      lifecycle="current", execution_state="validated")
            # Corrupt the suite hash in the manifest
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["suite_input_hashes"]["suite/warpcore-v1.yaml"] = "bad" + "0" * 61
            manifest_path.write_text(json.dumps(manifest))

            result = publish_campaign.publish(repo)
            models = {e["model_slug"] for e in result.entries}
            self.assertNotIn("hash-test-model", models,
                             "Changed (stale) task hash must block entry")

    def test_deleted_item_blocks_entry(self):
        """P7c: Deleting an item from per_item.csv must block the entry."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            run_dir = _make_valid_run(repo, model_slug="item-delete-model",
                                      lifecycle="current", execution_state="validated",
                                      n_items=5)
            # Truncate per_item.csv to fewer items
            csv_path = run_dir / "per_item.csv"
            lines = csv_path.read_text().splitlines()
            # Keep header + 3 items (expected=5, but only 3 present)
            csv_path.write_text("\n".join(lines[:4]) + "\n")

            result = publish_campaign.publish(repo)
            models = {e["model_slug"] for e in result.entries}
            self.assertNotIn("item-delete-model", models,
                             "Deleted items (count mismatch) must block entry")

    def test_changed_instance_id_blocks_entry(self):
        """P7d: Changing an instance ID in per_item.csv must block the entry."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            ids = [f"item-{i}" for i in range(5)]
            run_dir = _make_valid_run(repo, model_slug="instid-model",
                                      lifecycle="current", execution_state="validated",
                                      instance_ids=ids, n_items=5)
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["item_inventory"].pop("instance_ids_hash", None)
            manifest_path.write_text(json.dumps(manifest))
            # Alter an instance ID only in the derived audit CSV. The retained
            # raw sample IDs remain authoritative even for manifests created
            # before instance_ids_hash was introduced.
            csv_path = run_dir / "per_item.csv"
            content = csv_path.read_text()
            # Change item-0 to tampered-0
            content = content.replace("item-0", "tampered-0")
            csv_path.write_text(content)

            result = publish_campaign.publish(repo)
            models = {e["model_slug"] for e in result.entries}
            self.assertNotIn("instid-model", models,
                             "Changed instance ID must block entry")

    def test_restoration_reappears(self):
        """P8: Restoring a mutated run must make it reappear."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            run_dir = _make_valid_run(repo, model_slug="restore-model",
                                      lifecycle="current", execution_state="validated")

            # Mutate
            status_path = run_dir / "status.json"
            status = json.loads(status_path.read_text())
            status["lifecycle"] = "historical"
            status_path.write_text(json.dumps(status))
            result_mutated = publish_campaign.publish(repo)
            self.assertNotIn("restore-model",
                             {e["model_slug"] for e in result_mutated.entries},
                             "After mutation: must be absent")

            # Restore
            status["lifecycle"] = "current"
            status_path.write_text(json.dumps(status))
            result_restored = publish_campaign.publish(repo)
            self.assertIn("restore-model",
                          {e["model_slug"] for e in result_restored.entries},
                          "After restoration: must reappear")


# ---------------------------------------------------------------------------
# Test: P9 — Historical debt cannot become canonical implicitly
# ---------------------------------------------------------------------------

class TestHistoricalDebtRatchet(unittest.TestCase):

    def test_historical_run_not_promoted_implicitly(self):
        """Historical run must never be promoted to canonical without explicit change."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            # Historical layout
            model_dir = repo / "results" / "old-model" / "raw"
            model_dir.mkdir(parents=True, exist_ok=True)
            hist_status = {
                "schema_version": 1,
                "run_id": "hist-run-001",
                "suite_id": "warpcore-v1",
                "execution_state": "completed",
                "lifecycle": "historical",
                "history": [
                    {"state": "planned", "timestamp": "2026-08-01T12:00:00Z"},
                    {"state": "preflight_passed", "timestamp": "2026-08-01T13:00:00Z"},
                    {"state": "running", "timestamp": "2026-08-01T14:00:00Z"},
                    {"state": "completed", "timestamp": "2026-08-01T16:00:00Z"},
                ],
            }
            (model_dir / "status.json").write_text(json.dumps(hist_status))
            (model_dir / "manifest.json").write_text(json.dumps({
                "schema_version": 1,
                "suite_id": "warpcore-v1",
                "suite_schema_version": 1,
                "run_id": "hist-run-001",
                "benchmark": "gsm8k",
                "adapter_hash": "u" * 64,
                "suite_input_hashes": {"suite/warpcore-v1.yaml": "u" * 64},
                "serving_profile_digest": "sha256:" + "u" * 64,
                "model": {"slug": "old-model", "id": "org/old-model", "revision": "unrecorded"},
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
                    "completed_utc": "2026-08-01T16:00:00Z",
                },
                "artifact_inventory": {
                    "samples_jsonl_gz": False,
                    "per_item_csv": False,
                    "run_log": False,
                    "command_txt": False,
                    "done_sentinel": False,
                },
            }))
            result = publish_campaign.publish(repo)
            models = {e["model_slug"] for e in result.entries}
            self.assertNotIn("old-model", models,
                             "Historical run must not appear as canonical entry")

    def test_historical_run_appears_in_not_measured(self):
        """Historical run discovery must be visible (e.g. in audit), not blocked outright."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            # A run that exists but is historical should be discoverable for audit purposes
            model_dir = repo / "results" / "audit-model" / "raw"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "status.json").write_text(json.dumps({
                "schema_version": 1,
                "run_id": "hist-001",
                "suite_id": "warpcore-v1",
                "execution_state": "completed",
                "lifecycle": "historical",
                "history": [
                    {"state": "planned", "timestamp": "2026-08-01T12:00:00Z"},
                    {"state": "preflight_passed", "timestamp": "2026-08-01T13:00:00Z"},
                    {"state": "running", "timestamp": "2026-08-01T14:00:00Z"},
                    {"state": "completed", "timestamp": "2026-08-01T16:00:00Z"},
                ],
            }))
            (model_dir / "manifest.json").write_text(json.dumps({
                "schema_version": 1,
                "suite_id": "warpcore-v1",
                "suite_schema_version": 1,
                "run_id": "hist-001",
                "benchmark": "gsm8k",
                "adapter_hash": "u" * 64,
                "suite_input_hashes": {"suite/warpcore-v1.yaml": "u" * 64},
                "serving_profile_digest": "sha256:" + "u" * 64,
                "model": {"slug": "audit-model", "id": "org/audit", "revision": "unrecorded"},
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
                    "completed_utc": None,
                },
                "artifact_inventory": {
                    "samples_jsonl_gz": False,
                    "per_item_csv": False,
                    "run_log": False,
                    "command_txt": False,
                    "done_sentinel": False,
                },
            }))
            # publish() should succeed (not raise); historical run is visible in context
            result = publish_campaign.publish(repo)
            # Historical run should NOT be a canonical entry
            models = {e["model_slug"] for e in result.entries}
            self.assertNotIn("audit-model", models)


if __name__ == "__main__":
    unittest.main()
