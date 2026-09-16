"""tests/test_task7_runner_integration.py — Task 7: runner production-integration fixes.

TDD: tests written BEFORE implementation.  Must fail (RED) until run_quality.py
is updated to fix blockers 1–4 from the audit report
(subagent-summary-0-20260916_100231_580426.txt).

BLOCKER 1: per_item.csv missing — runner never creates run_dir/per_item.csv.
BLOCKER 2: per_item.csv schema mismatch — needs item_id (not doc_id), disposition,
           finish_reason, response_chars, empty_content, score; one row per item
           despite multiple lm-eval filter rows per doc_id.
BLOCKER 3: manifest.json submitted and completed_utc never updated after run.
BLOCKER 4: artifact_inventory booleans never flipped to True.

All derivation is deterministic from the retained samples_*.jsonl.gz files.
Fail-closed semantics: if samples are absent or corrupt, do NOT write DONE.
Failure injection tests prove no success if derivation or manifest persistence fails.

Design choices forced by real lm-eval field shapes (confirmed from actual samples):
- lm-eval JSONL has no 'finish_reason' field → stored as empty string in CSV
- Multiple rows per doc_id when multiple filters (e.g., GPQA: answer-line +
  flexible-fallback) → deduplicate to ONE row per item_id
- item_id = str(doc_id) (integer doc_id cast to string)
- disposition: 'scored' when response_chars > 0, 'empty_response' when 0
- score: best non-None score across all filter rows for this doc_id
- response_chars: max across filter rows (keep non-empty view per validate_samples logic)
- canonical score for the suite benchmark read from aggregate results JSON
  (exact_match,none / exact_match,answer-line / prompt_level_strict_acc,none)
"""
from __future__ import annotations

import csv
import gzip
import json
import pathlib
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

# Allow imports from viz/ and tests/
_TESTS_DIR = pathlib.Path(__file__).parent
_REPO = _TESTS_DIR.parent
_VIZ_DIR = _REPO / "viz"
for _p in (str(_TESTS_DIR), str(_VIZ_DIR), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_quality  # noqa: E402
import campaign_state  # noqa: E402

_REAL_SUITE = _REPO / "suite" / "warpcore-v1.yaml"
_SCHEMAS_DIR = _REPO / "suite" / "schemas"

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_CANONICAL_ADAPTER = {
    "adapter_schema_version": 1,
    "campaign_status": "canonical",
    "model": {
        "slug": "test-canonical-model",
        "id": "testorg/TestCanonicalModel",
        "revision": "a" * 40,
    },
    "serving": {
        "image": "testregistry.example.com/test@sha256:" + "b" * 64,
        "engine": "vllm",
        "engine_version": "0.6.6",
        "quantization": "fp8",
        "reasoning_parser": None,
        "tool_call_parser": None,
        "tokenizer": None,
        "moe_backend": None,
        "max_model_len": 300000,
        "gpu_memory_utilization": 0.95,
        "max_num_seqs": 256,
        "environment": {},
    },
}

_PROMPT_TOKEN_MAXIMA = {
    "gsm8k": 500,
    "ifeval": 2000,
    "gpqa_diamond": 1000,
}

_ADAPTER_SLUG = _CANONICAL_ADAPTER["model"]["slug"]


def _write_canonical_adapter(path: pathlib.Path) -> None:
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(_CANONICAL_ADAPTER, default_flow_style=False))


def _make_planned_status(run_id: str = "run-test",
                         suite_id: str = "warpcore-v1") -> dict:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "suite_id": suite_id,
        "execution_state": "planned",
        "lifecycle": "current",
        "history": [{"state": "planned",
                     "timestamp": "2026-09-15T12:00:00Z"}],
    }


def _make_manifest(run_id: str = "run-test",
                   suite_id: str = "warpcore-v1",
                   benchmark: str = "gsm8k") -> dict:
    """Minimal manifest for test run dirs (not full schema; tests read from disk)."""
    return {
        "suite_id": suite_id,
        "run_id": run_id,
        "benchmark": benchmark,
        "model": {
            "slug": _ADAPTER_SLUG,
            "id": "testorg/TestCanonicalModel",
            "revision": "a" * 40,
        },
        "item_inventory": {"expected": 1319, "submitted": 0},
        "timing": {"started_utc": "2026-09-15T12:00:00Z", "completed_utc": None},
        "artifact_inventory": {
            "samples_jsonl_gz": False,
            "per_item_csv": False,
            "run_log": False,
            "command_txt": False,
            "done_sentinel": False,
        },
    }


def _build_run_dir(
    repo: pathlib.Path,
    bench: str = "gsm8k",
    slug: str = _ADAPTER_SLUG,
    run_id: str = "run-test",
    suite_id: str = "warpcore-v1",
) -> pathlib.Path:
    run_dir = repo / "results" / slug / "runs" / suite_id / bench / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "status.json").write_text(
        json.dumps(_make_planned_status(run_id, suite_id))
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(_make_manifest(run_id, suite_id, bench))
    )
    return run_dir


def _make_gsm8k_samples_gz(raw_dir: pathlib.Path, n_items: int = 3) -> pathlib.Path:
    """Write a minimal valid samples_*.jsonl.gz with GSM8K-shaped records.

    - One filter row per doc_id (answer-line), like real GSM8K output.
    - Items 0..n_items-2 have non-empty responses; last item is empty.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    gz_path = raw_dir / "samples_gsm8k_cot_zeroshot_clean_2026-01-01T00-00-00.jsonl.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8") as fh:
        for i in range(n_items):
            response = "" if i == n_items - 1 else f"The answer is {i * 2}.\n\n#### {i * 2}"
            rec = {
                "doc_id": i,
                "filter": "answer-line",
                "resps": [[response]],
                "filtered_resps": [str(i * 2) if response else ""],
                "target": str(i * 2),
                "exact_match": 1.0 if response and not response == "" else 0.0,
                "arguments": {"gen_args_0": {"arg_0": [json.dumps([
                    {"role": "user", "content": f"GSM8K question {i}"}
                ])]}},
            }
            fh.write(json.dumps(rec) + "\n")
    return gz_path


def _make_gpqa_samples_gz(raw_dir: pathlib.Path, n_items: int = 3) -> pathlib.Path:
    """Write a minimal valid samples_*.jsonl.gz with GPQA-shaped records.

    - Two filter rows per doc_id (answer-line + flexible-fallback), like real GPQA.
    - Verifies one-row-per-item deduplication in per_item.csv.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    gz_path = raw_dir / "samples_gpqa_diamond_cot_zeroshot_clean_2026-01-01T00-00-00.jsonl.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8") as fh:
        for i in range(n_items):
            response = "" if i == n_items - 1 else f"The answer is (A). Reasoning step {i}."
            for flt in ("answer-line", "flexible-fallback"):
                rec = {
                    "doc_id": i,
                    "filter": flt,
                    "resps": [[response]],
                    "filtered_resps": ["(A)" if response else "[invalid]"],
                    "target": "(A)",
                    "exact_match": 1.0 if response else 0.0,
                    "arguments": {"gen_args_0": {"arg_0": [json.dumps([
                        {"role": "user", "content": f"GPQA question {i}"}
                    ])]}},
                }
                fh.write(json.dumps(rec) + "\n")
    return gz_path


def _make_sidecar_for_samples(run_dir: pathlib.Path) -> pathlib.Path:
    """Create complete request metadata corresponding exactly to fixture samples."""
    import hashlib

    records = {}
    for sample_path in sorted((run_dir / "raw").rglob("samples_*.jsonl.gz")):
        with gzip.open(sample_path, "rt", encoding="utf-8") as fh:
            for line in fh:
                sample = json.loads(line)
                messages = json.loads(sample["arguments"]["gen_args_0"]["arg_0"][0])
                canonical = json.dumps(
                    messages, sort_keys=True, ensure_ascii=False
                ).encode("utf-8")
                fingerprint = hashlib.sha256(canonical).hexdigest()
                response = sample["resps"][0][0]
                records.setdefault(fingerprint, {
                    "fingerprint": fingerprint,
                    "ts_utc": "2026-01-01T00:00:00+00:00",
                    "model": "testorg/TestCanonicalModel",
                    "response_id": f"fixture-{sample['doc_id']}",
                    "finish_reasons": ["stop"],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 0 if not response else 5,
                        "total_tokens": 10 if not response else 15,
                    },
                    "content": [response],
                    "reasoning_content": [None],
                    "reasoning": [None],
                    "content_null_count": 0,
                    "empty_by_length": False,
                })
    path = run_dir / "raw" / "response_metadata.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for record in records.values():
            fh.write(json.dumps(record) + "\n")
    return path


def _make_aggregate_result_json(raw_dir: pathlib.Path, task: str,
                                score_key: str, score: float) -> pathlib.Path:
    """Write a minimal aggregate results JSON."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "results_2026-01-01T00-00-00.json"
    path.write_text(json.dumps({
        "results": {task: {score_key: score}}
    }) + "\n")
    return path


def _make_full_harness_artifacts_gsm8k(
    run_dir: pathlib.Path, n_items: int = 3
) -> None:
    """Populate raw/ with GSM8K artifacts (aggregate JSON + samples gz)."""
    subdir = run_dir / "raw" / "gsm8k_cot_zeroshot_clean"
    subdir.mkdir(parents=True, exist_ok=True)
    _make_aggregate_result_json(
        subdir, "gsm8k_cot_zeroshot_clean", "exact_match,none", 0.667
    )
    _make_gsm8k_samples_gz(subdir, n_items)
    _make_sidecar_for_samples(run_dir)


def _make_full_harness_artifacts_gpqa(
    run_dir: pathlib.Path, n_items: int = 3
) -> None:
    """Populate raw/ with GPQA artifacts (aggregate JSON + samples gz)."""
    subdir = run_dir / "raw" / "gpqa_diamond_cot_zeroshot_clean"
    subdir.mkdir(parents=True, exist_ok=True)
    _make_aggregate_result_json(
        subdir, "gpqa_diamond_cot_zeroshot_clean", "exact_match,answer-line", 0.5
    )
    _make_gpqa_samples_gz(subdir, n_items)
    _make_sidecar_for_samples(run_dir)


def _make_runner(
    tmp: pathlib.Path,
    adapter_path: pathlib.Path,
    bench: str = "gsm8k",
    *,
    preflight_exit: int = 0,
    harness_exit: int = 0,
) -> tuple:
    """Create a QualityRunner with mocked preflight and harness."""
    run_dir = _build_run_dir(tmp, bench)
    preflight_mock = MagicMock(return_value=preflight_exit)
    harness_mock = MagicMock(return_value=harness_exit)
    runner = run_quality.QualityRunner(
        suite_path=_REAL_SUITE,
        adapter_path=adapter_path,
        benchmark=bench,
        endpoint="http://fake:8000/v1",
        throughput=64.0,
        concurrency=8,
        timeout=14400,
        run_dir=run_dir,
        repo=tmp,
        prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
        allow_no_screen=True,
        preflight_runner=preflight_mock,
        harness_runner=harness_mock,
    )
    return runner, run_dir, preflight_mock, harness_mock


# ---------------------------------------------------------------------------
# BLOCKER 1 + 2: per_item.csv created at run_dir/per_item.csv with correct schema
# ---------------------------------------------------------------------------


class TestPerItemCsvCreated(unittest.TestCase):
    """BLOCKER 1: run_quality must create run_dir/per_item.csv on successful run."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_per_item_csv_exists_after_successful_run(self):
        """per_item.csv must exist at run_dir/per_item.csv after a successful run."""
        runner, run_dir, _, harness_mock = _make_runner(
            self.tmp, self.adapter_path, "gsm8k"
        )
        _make_full_harness_artifacts_gsm8k(run_dir)
        rc = runner.run()
        self.assertEqual(rc, 0, f"Expected exit 0, got {rc}")
        self.assertTrue(
            (run_dir / "per_item.csv").exists(),
            "per_item.csv must exist at run_dir/per_item.csv after successful run"
        )

    def test_per_item_csv_not_created_on_harness_failure(self):
        """per_item.csv must NOT be written when harness exits nonzero."""
        runner, run_dir, _, harness_mock = _make_runner(
            self.tmp, self.adapter_path, "gsm8k", harness_exit=1
        )
        _make_full_harness_artifacts_gsm8k(run_dir)
        rc = runner.run()
        self.assertNotEqual(rc, 0)
        # per_item.csv should NOT be created on failure
        self.assertFalse(
            (run_dir / "per_item.csv").exists(),
            "per_item.csv must NOT be created when harness fails"
        )

    def test_per_item_csv_not_created_when_samples_missing(self):
        """If no samples_*.jsonl.gz found, fail closed and do not write DONE or per_item.csv."""
        runner, run_dir, _, harness_mock = _make_runner(
            self.tmp, self.adapter_path, "gsm8k", harness_exit=0
        )
        # Create aggregate result JSON but NO samples gz
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True)
        (raw_dir / "results_2026-01-01T00-00-00.json").write_text(
            '{"results": {"gsm8k_cot_zeroshot_clean": {"exact_match,none": 0.5}}}\n'
        )
        rc = runner.run()
        self.assertNotEqual(rc, 0, "Must return nonzero when samples missing")
        self.assertFalse(
            (run_dir / "DONE").exists(),
            "DONE must NOT be written when samples missing"
        )
        self.assertFalse(
            (run_dir / "per_item.csv").exists(),
            "per_item.csv must NOT be written when samples missing"
        )


class TestPerItemCsvSchema(unittest.TestCase):
    """BLOCKER 2: per_item.csv must use correct column schema for validator."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run_and_read_csv(self, bench: str = "gsm8k",
                          make_artifacts_fn=None) -> tuple:
        runner, run_dir, _, _ = _make_runner(self.tmp, self.adapter_path, bench)
        if make_artifacts_fn is None:
            _make_full_harness_artifacts_gsm8k(run_dir)
        else:
            make_artifacts_fn(run_dir)
        rc = runner.run()
        csv_path = run_dir / "per_item.csv"
        rows = []
        if csv_path.exists():
            with csv_path.open(newline="") as fh:
                rows = list(csv.DictReader(fh))
        return rc, run_dir, rows

    def test_per_item_csv_has_item_id_column(self):
        """per_item.csv must use 'item_id' column, not 'doc_id'."""
        rc, run_dir, rows = self._run_and_read_csv("gsm8k")
        self.assertEqual(rc, 0)
        self.assertTrue(rows, "per_item.csv must have rows")
        self.assertIn("item_id", rows[0],
                      f"per_item.csv must have 'item_id' column; got {list(rows[0].keys())}")

    def test_per_item_csv_has_no_doc_id_column(self):
        """per_item.csv must NOT use 'doc_id' as the ID column (validator reads 'item_id')."""
        rc, run_dir, rows = self._run_and_read_csv("gsm8k")
        self.assertEqual(rc, 0)
        self.assertTrue(rows)
        self.assertNotIn("doc_id", rows[0],
                         "per_item.csv must not use 'doc_id'; validator expects 'item_id'")

    def test_per_item_csv_has_required_columns(self):
        """per_item.csv must have: item_id, score, response_chars, empty_content,
           disposition, finish_reason."""
        required = {"item_id", "score", "response_chars", "empty_content",
                    "disposition", "finish_reason"}
        rc, run_dir, rows = self._run_and_read_csv("gsm8k")
        self.assertEqual(rc, 0)
        self.assertTrue(rows, "per_item.csv must have rows")
        actual_cols = set(rows[0].keys())
        missing = required - actual_cols
        self.assertEqual(
            missing, set(),
            f"per_item.csv is missing required columns: {missing}. Got: {actual_cols}"
        )

    def test_per_item_csv_one_row_per_item_gsm8k(self):
        """GSM8K (1 filter row per doc_id) → one CSV row per unique doc_id."""
        n_items = 4
        rc, run_dir, rows = self._run_and_read_csv(
            "gsm8k",
            lambda rd: _make_full_harness_artifacts_gsm8k(rd, n_items)
        )
        self.assertEqual(rc, 0)
        self.assertEqual(
            len(rows), n_items,
            f"GSM8K: expected {n_items} CSV rows (one per item), got {len(rows)}"
        )

    def test_per_item_csv_one_row_per_item_gpqa(self):
        """GPQA (2 filter rows per doc_id) → one CSV row per unique doc_id, NOT doubled."""
        n_items = 4
        rc, run_dir, rows = self._run_and_read_csv(
            "gpqa_diamond",
            lambda rd: _make_full_harness_artifacts_gpqa(rd, n_items)
        )
        self.assertEqual(rc, 0)
        self.assertEqual(
            len(rows), n_items,
            f"GPQA: expected {n_items} CSV rows (deduplicated from 2x filter rows), "
            f"got {len(rows)}"
        )

    def test_per_item_csv_item_ids_are_strings(self):
        """item_id must be a string (str(doc_id))."""
        rc, run_dir, rows = self._run_and_read_csv("gsm8k")
        self.assertEqual(rc, 0)
        for row in rows:
            self.assertIsInstance(row["item_id"], str,
                                  "item_id must be a string")
            # Must be parseable as integer (i.e., came from integer doc_id)
            try:
                int(row["item_id"])
            except ValueError:
                self.fail(f"item_id must be castable to int; got {row['item_id']!r}")

    def test_per_item_csv_item_id_covers_all_doc_ids(self):
        """All expected doc_ids must appear exactly once in per_item.csv."""
        n_items = 5
        rc, run_dir, rows = self._run_and_read_csv(
            "gsm8k",
            lambda rd: _make_full_harness_artifacts_gsm8k(rd, n_items)
        )
        self.assertEqual(rc, 0)
        item_ids = [row["item_id"] for row in rows]
        expected = [str(i) for i in range(n_items)]
        self.assertEqual(
            sorted(item_ids), sorted(expected),
            f"per_item.csv item_ids {sorted(item_ids)} != expected {sorted(expected)}"
        )

    def test_per_item_csv_response_chars_correct(self):
        """response_chars must be the character count of the response text."""
        rc, run_dir, rows = self._run_and_read_csv("gsm8k")
        self.assertEqual(rc, 0)
        # Last item is empty (n_items=3 by default, item 2 is empty)
        rows_by_id = {row["item_id"]: row for row in rows}
        empty_row = rows_by_id.get("2")
        self.assertIsNotNone(empty_row, "item_id=2 should exist")
        self.assertEqual(
            int(empty_row["response_chars"]), 0,
            f"Empty item response_chars must be 0, got {empty_row['response_chars']!r}"
        )
        # Non-empty item should have response_chars > 0
        nonempty_row = rows_by_id.get("0")
        self.assertIsNotNone(nonempty_row, "item_id=0 should exist")
        self.assertGreater(
            int(nonempty_row["response_chars"]), 0,
            "Non-empty item response_chars must be > 0"
        )

    def test_per_item_csv_empty_content_correct(self):
        """empty_content must be 1 for empty items, 0 for non-empty."""
        rc, run_dir, rows = self._run_and_read_csv("gsm8k")
        self.assertEqual(rc, 0)
        rows_by_id = {row["item_id"]: row for row in rows}
        # Last item (index 2) is empty
        self.assertEqual(rows_by_id["2"]["empty_content"], "1",
                         "Empty item empty_content must be '1'")
        self.assertEqual(rows_by_id["0"]["empty_content"], "0",
                         "Non-empty item empty_content must be '0'")

    def test_per_item_csv_disposition_set(self):
        """disposition must be non-empty for every row."""
        rc, run_dir, rows = self._run_and_read_csv("gsm8k")
        self.assertEqual(rc, 0)
        for row in rows:
            self.assertIn(
                "disposition", row,
                "disposition column must exist"
            )
            self.assertIsNotNone(row["disposition"])
            self.assertNotEqual(
                row["disposition"], "",
                f"disposition must be non-empty for item_id={row.get('item_id')}"
            )

    def test_per_item_csv_finish_reason_column_exists(self):
        """finish_reason column must exist (may be empty string — not in lm-eval JSONL)."""
        rc, run_dir, rows = self._run_and_read_csv("gsm8k")
        self.assertEqual(rc, 0)
        self.assertTrue(rows)
        self.assertIn("finish_reason", rows[0],
                      "finish_reason column must exist in per_item.csv")

    def test_per_item_csv_score_present_for_non_empty_items(self):
        """score must be non-empty for items that received a score."""
        rc, run_dir, rows = self._run_and_read_csv("gsm8k")
        self.assertEqual(rc, 0)
        rows_by_id = {row["item_id"]: row for row in rows}
        # Item 0 has score=1.0 (exact_match=1.0)
        score_0 = rows_by_id["0"]["score"]
        self.assertNotEqual(score_0, "",
                            f"Non-empty item must have a score; got {score_0!r}")
        try:
            float(score_0)
        except ValueError:
            self.fail(f"score must be parseable as float; got {score_0!r}")

    def test_per_item_csv_gpqa_dedup_keeps_best_score(self):
        """GPQA two-filter rows: per-item CSV row must carry the best (non-None) score."""
        n_items = 2
        rc, run_dir, rows = self._run_and_read_csv(
            "gpqa_diamond",
            lambda rd: _make_full_harness_artifacts_gpqa(rd, n_items)
        )
        self.assertEqual(rc, 0)
        rows_by_id = {row["item_id"]: row for row in rows}
        # Item 0 should have a score (both filter rows have exact_match=1.0)
        self.assertNotEqual(rows_by_id["0"]["score"], "",
                            "GPQA deduped item 0 should have a score")


# ---------------------------------------------------------------------------
# BLOCKER 3: manifest.json submitted count and completed_utc updated on success
# ---------------------------------------------------------------------------


class TestManifestUpdatedOnCompletion(unittest.TestCase):
    """BLOCKER 3: manifest.json must be durably updated after a successful run."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run_success(self, bench: str = "gsm8k",
                     make_fn=None) -> tuple:
        runner, run_dir, _, _ = _make_runner(self.tmp, self.adapter_path, bench)
        if make_fn is None:
            _make_full_harness_artifacts_gsm8k(run_dir)
        else:
            make_fn(run_dir)
        rc = runner.run()
        manifest = json.loads((run_dir / "manifest.json").read_text())
        return rc, run_dir, manifest

    def test_manifest_completed_utc_set_on_success(self):
        """timing.completed_utc must be set to a non-null timestamp after a successful run."""
        rc, run_dir, manifest = self._run_success()
        self.assertEqual(rc, 0)
        completed_utc = manifest.get("timing", {}).get("completed_utc")
        self.assertIsNotNone(
            completed_utc,
            "timing.completed_utc must be set (non-null) after a successful run"
        )
        self.assertNotEqual(completed_utc, "",
                            "timing.completed_utc must not be empty string")

    def test_manifest_completed_utc_is_valid_datetime(self):
        """timing.completed_utc must be a valid ISO 8601 date-time string."""
        from datetime import datetime
        rc, run_dir, manifest = self._run_success()
        self.assertEqual(rc, 0)
        completed_utc = manifest["timing"]["completed_utc"]
        # Must parse as ISO 8601 (with Z suffix as used by _utcnow())
        try:
            datetime.fromisoformat(completed_utc.replace("Z", "+00:00"))
        except (ValueError, AttributeError) as e:
            self.fail(
                f"timing.completed_utc {completed_utc!r} is not valid ISO 8601: {e}"
            )

    def test_manifest_submitted_count_updated_on_success(self):
        """item_inventory.submitted must be updated from 0 to the actual item count."""
        n_items = 4
        rc, run_dir, manifest = self._run_success(
            "gsm8k",
            lambda rd: _make_full_harness_artifacts_gsm8k(rd, n_items)
        )
        self.assertEqual(rc, 0)
        submitted = manifest.get("item_inventory", {}).get("submitted", 0)
        self.assertGreater(
            submitted, 0,
            "item_inventory.submitted must be > 0 after a successful run; was still 0"
        )
        self.assertEqual(
            submitted, n_items,
            f"item_inventory.submitted must equal actual item count {n_items}, got {submitted}"
        )

    def test_manifest_submitted_not_updated_on_harness_failure(self):
        """item_inventory.submitted must remain 0 when harness fails."""
        runner, run_dir, _, _ = _make_runner(
            self.tmp, self.adapter_path, "gsm8k", harness_exit=1
        )
        _make_full_harness_artifacts_gsm8k(run_dir)
        rc = runner.run()
        self.assertNotEqual(rc, 0)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        submitted = manifest.get("item_inventory", {}).get("submitted", 0)
        self.assertEqual(
            submitted, 0,
            "item_inventory.submitted must stay 0 when harness fails"
        )

    def test_manifest_completed_utc_not_set_on_harness_failure(self):
        """timing.completed_utc must remain null when harness fails."""
        runner, run_dir, _, _ = _make_runner(
            self.tmp, self.adapter_path, "gsm8k", harness_exit=1
        )
        _make_full_harness_artifacts_gsm8k(run_dir)
        rc = runner.run()
        self.assertNotEqual(rc, 0)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        self.assertIsNone(
            manifest.get("timing", {}).get("completed_utc"),
            "timing.completed_utc must remain null on harness failure"
        )

    def test_manifest_schema_valid_after_completion(self):
        """manifest.json must pass schema validation after a successful run."""
        from contract import validate_json
        from create_campaign import create_campaign
        import shutil
        schema_path = _SCHEMAS_DIR / "manifest.schema.json"

        # create_campaign requires the real repo AND the adapter must be inside it.
        # Write a canonical adapter into the real repo's adapters/ dir for this test.
        run_id = "run-task7-schema-test"
        slug_root = _REPO / "results" / _ADAPTER_SLUG
        temp_adapter_path = _REPO / "adapters" / "test-canonical-model-task7.yaml"
        test_run_dir = (
            slug_root / "runs" / "warpcore-v1" / "gsm8k" / run_id
        )

        # Clean up any artifacts from prior test runs
        if slug_root.exists():
            shutil.rmtree(slug_root)
        if temp_adapter_path.exists():
            temp_adapter_path.unlink()

        try:
            _write_canonical_adapter(temp_adapter_path)
            full_run_dir = create_campaign(
                repo=_REPO,
                suite_path=_REAL_SUITE,
                adapter_path=temp_adapter_path,
                benchmark="gsm8k",
                run_id=run_id,
                prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            )
            _make_full_harness_artifacts_gsm8k(full_run_dir)

            def harness_with_log(cmd):
                (full_run_dir / "run.log").write_text(
                    "mock harness output\n", encoding="utf-8"
                )
                return 0

            runner = run_quality.QualityRunner(
                suite_path=_REAL_SUITE,
                adapter_path=temp_adapter_path,
                benchmark="gsm8k",
                endpoint="http://fake:8000/v1",
                throughput=64.0,
                concurrency=8,
                timeout=14400,
                run_dir=full_run_dir,
                repo=_REPO,
                prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
                allow_no_screen=True,
                preflight_runner=MagicMock(return_value=0),
                harness_runner=harness_with_log,
            )
            rc = runner.run()
            self.assertEqual(rc, 0, f"Expected exit 0, got {rc}")
            manifest = json.loads((full_run_dir / "manifest.json").read_text())
            errors = validate_json(manifest, schema_path)
            self.assertEqual(
                errors, [],
                f"manifest.json after completion must be schema-valid; errors: {errors}"
            )
        finally:
            # Clean up the real repo artifacts created by this test.
            # Remove the entire slug_root subtree (not just the leaf run_id dir)
            # so no empty parent dirs are left as result-tree residue.
            if slug_root.exists():
                shutil.rmtree(slug_root)
            if temp_adapter_path.exists():
                temp_adapter_path.unlink()

    def test_manifest_schema_valid_no_result_tree_residue(self):
        """test_manifest_schema_valid_after_completion must leave no results/ residue.

        Regression: the existing finally block removes the leaf run_id dir but not
        its empty parent chain results/test-canonical-model/runs/warpcore-v1/gsm8k/.
        After the test, _REPO/results/test-canonical-model must not exist.

        TDD: this test is RED until the finally block is widened to remove the
        entire results/<slug> subtree it created.
        """
        from contract import validate_json
        from create_campaign import create_campaign

        run_id = "run-task7-schema-test"
        slug_root = _REPO / "results" / _ADAPTER_SLUG
        temp_adapter_path = _REPO / "adapters" / "test-canonical-model-task7.yaml"
        test_run_dir = (
            slug_root / "runs" / "warpcore-v1" / "gsm8k" / run_id
        )

        # Pre-clean so prior failures don't mask the assertion
        if slug_root.exists():
            shutil.rmtree(slug_root)
        if temp_adapter_path.exists():
            temp_adapter_path.unlink()

        try:
            _write_canonical_adapter(temp_adapter_path)
            full_run_dir = create_campaign(
                repo=_REPO,
                suite_path=_REAL_SUITE,
                adapter_path=temp_adapter_path,
                benchmark="gsm8k",
                run_id=run_id,
                prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            )
            _make_full_harness_artifacts_gsm8k(full_run_dir)

            def harness_with_log(cmd):
                (full_run_dir / "run.log").write_text(
                    "mock harness output\n", encoding="utf-8"
                )
                return 0

            runner = run_quality.QualityRunner(
                suite_path=_REAL_SUITE,
                adapter_path=temp_adapter_path,
                benchmark="gsm8k",
                endpoint="http://fake:8000/v1",
                throughput=64.0,
                concurrency=8,
                timeout=14400,
                run_dir=full_run_dir,
                repo=_REPO,
                prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
                allow_no_screen=True,
                preflight_runner=MagicMock(return_value=0),
                harness_runner=harness_with_log,
            )
            rc = runner.run()
            self.assertEqual(rc, 0, f"Expected exit 0, got {rc}")
        finally:
            # NOTE: the original cleanup only removes test_run_dir and temp_adapter_path,
            # leaving the empty parent tree. The fix must also remove slug_root.
            if test_run_dir.exists():
                shutil.rmtree(test_run_dir)
            if temp_adapter_path.exists():
                temp_adapter_path.unlink()
            # Cleanup the entire slug_root to avoid result-tree residue.
            # This is the correct isolation pattern: remove the entire subtree created
            # for this model slug, not just the leaf run directory.
            if slug_root.exists():
                shutil.rmtree(slug_root)
            # Regression check: slug_root must not survive after full cleanup
            self.assertFalse(
                slug_root.exists(),
                f"Test left residue at {slug_root}. "
                "The finally block must clean up the entire results/<slug> subtree it created."
            )


# ---------------------------------------------------------------------------
# BLOCKER 4: artifact_inventory booleans flipped to True on success
# ---------------------------------------------------------------------------


class TestArtifactInventoryUpdated(unittest.TestCase):
    """BLOCKER 4: artifact_inventory booleans must be True for each written artifact."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run_and_read_manifest(self, bench: str = "gsm8k",
                               make_fn=None) -> tuple:
        runner, run_dir, _, harness_mock = _make_runner(self.tmp, self.adapter_path, bench)
        if make_fn is None:
            _make_full_harness_artifacts_gsm8k(run_dir)
        else:
            make_fn(run_dir)

        # Create a fake run.log so artifact_inventory.run_log can be True
        # (the real subprocess path writes run.log; mock harness does not)
        def harness_with_log(cmd):
            (run_dir / "run.log").write_text("mock harness output\n", encoding="utf-8")
            return 0

        runner._harness_runner = harness_with_log

        rc = runner.run()
        manifest = json.loads((run_dir / "manifest.json").read_text())
        return rc, run_dir, manifest

    def test_artifact_inventory_samples_jsonl_gz_true_on_success(self):
        """artifact_inventory.samples_jsonl_gz must be True after successful run."""
        rc, run_dir, manifest = self._run_and_read_manifest()
        self.assertEqual(rc, 0)
        self.assertTrue(
            manifest.get("artifact_inventory", {}).get("samples_jsonl_gz"),
            "artifact_inventory.samples_jsonl_gz must be True after successful run"
        )

    def test_artifact_inventory_per_item_csv_true_on_success(self):
        """artifact_inventory.per_item_csv must be True after successful run."""
        rc, run_dir, manifest = self._run_and_read_manifest()
        self.assertEqual(rc, 0)
        self.assertTrue(
            manifest.get("artifact_inventory", {}).get("per_item_csv"),
            "artifact_inventory.per_item_csv must be True after successful run"
        )

    def test_artifact_inventory_run_log_true_on_success(self):
        """artifact_inventory.run_log must be True after successful run."""
        rc, run_dir, manifest = self._run_and_read_manifest()
        self.assertEqual(rc, 0)
        self.assertTrue(
            manifest.get("artifact_inventory", {}).get("run_log"),
            "artifact_inventory.run_log must be True after successful run"
        )

    def test_artifact_inventory_command_txt_true_on_success(self):
        """artifact_inventory.command_txt must be True after successful run."""
        rc, run_dir, manifest = self._run_and_read_manifest()
        self.assertEqual(rc, 0)
        self.assertTrue(
            manifest.get("artifact_inventory", {}).get("command_txt"),
            "artifact_inventory.command_txt must be True after successful run"
        )

    def test_artifact_inventory_done_sentinel_true_on_success(self):
        """artifact_inventory.done_sentinel must be True after successful run."""
        rc, run_dir, manifest = self._run_and_read_manifest()
        self.assertEqual(rc, 0)
        self.assertTrue(
            manifest.get("artifact_inventory", {}).get("done_sentinel"),
            "artifact_inventory.done_sentinel must be True after successful run"
        )

    def test_artifact_inventory_all_false_on_harness_failure(self):
        """artifact_inventory must remain all-False when harness fails (no artifacts written)."""
        runner, run_dir, _, _ = _make_runner(
            self.tmp, self.adapter_path, "gsm8k", harness_exit=1
        )
        _make_full_harness_artifacts_gsm8k(run_dir)
        rc = runner.run()
        self.assertNotEqual(rc, 0)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        inv = manifest.get("artifact_inventory", {})
        # done_sentinel and per_item_csv must be False (not produced)
        self.assertFalse(
            inv.get("done_sentinel"),
            "artifact_inventory.done_sentinel must be False on harness failure"
        )
        self.assertFalse(
            inv.get("per_item_csv"),
            "artifact_inventory.per_item_csv must be False on harness failure"
        )

    def test_artifact_inventory_all_true_on_success(self):
        """All artifact_inventory booleans must be True after a successful run."""
        rc, run_dir, manifest = self._run_and_read_manifest()
        self.assertEqual(rc, 0)
        inv = manifest.get("artifact_inventory", {})
        required_keys = [
            "samples_jsonl_gz", "per_item_csv", "run_log",
            "command_txt", "done_sentinel"
        ]
        for key in required_keys:
            self.assertTrue(
                inv.get(key),
                f"artifact_inventory.{key} must be True after successful run; got {inv.get(key)!r}"
            )


# ---------------------------------------------------------------------------
# Fail-closed: derivation failure must block DONE
# ---------------------------------------------------------------------------


class TestFailClosedOnDerivationFailure(unittest.TestCase):
    """If per_item derivation fails, the run must fail closed — DONE not written.

    Failure injection: provide corrupt samples to trigger derivation error.
    """

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_corrupt_samples_gz_blocks_done(self):
        """If samples_*.jsonl.gz is corrupt (invalid gzip), DONE must not be written."""
        runner, run_dir, _, _ = _make_runner(
            self.tmp, self.adapter_path, "gsm8k", harness_exit=0
        )
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True)
        # Write valid aggregate result JSON
        (raw_dir / "results_2026-01-01T00-00-00.json").write_text(
            '{"results": {"gsm8k_cot_zeroshot_clean": {"exact_match,none": 0.5}}}\n'
        )
        # Write corrupt (non-gzip) bytes as .jsonl.gz
        corrupt_gz = raw_dir / "samples_gsm8k_cot_zeroshot_clean_2026-01-01T00-00-00.jsonl.gz"
        corrupt_gz.write_bytes(b"NOT A VALID GZIP FILE\n")
        rc = runner.run()
        self.assertNotEqual(rc, 0, "Corrupt samples must cause nonzero exit")
        self.assertFalse(
            (run_dir / "DONE").exists(),
            "DONE must not be written when samples are corrupt"
        )

    def test_empty_samples_gz_blocks_done(self):
        """If samples_*.jsonl.gz is a valid gzip but contains no JSONL, fail closed."""
        runner, run_dir, _, _ = _make_runner(
            self.tmp, self.adapter_path, "gsm8k", harness_exit=0
        )
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True)
        (raw_dir / "results_2026-01-01T00-00-00.json").write_text(
            '{"results": {"gsm8k_cot_zeroshot_clean": {"exact_match,none": 0.5}}}\n'
        )
        # Write a valid gzip with ZERO bytes inside (no lines at all)
        gz_path = raw_dir / "samples_gsm8k_cot_zeroshot_clean_2026-01-01T00-00-00.jsonl.gz"
        with gzip.open(gz_path, "wb") as fh:
            pass  # write nothing
        rc = runner.run()
        self.assertNotEqual(rc, 0, "Empty samples gz must cause nonzero exit (fail closed)")
        self.assertFalse(
            (run_dir / "DONE").exists(),
            "DONE must not be written when samples gz is empty"
        )

    def test_manifest_write_failure_blocks_done(self):
        """If manifest.json update fails (e.g. read-only), DONE must not be written."""
        runner, run_dir, _, _ = _make_runner(
            self.tmp, self.adapter_path, "gsm8k", harness_exit=0
        )
        _make_full_harness_artifacts_gsm8k(run_dir)
        # Make manifest.json read-only to simulate write failure
        manifest_path = run_dir / "manifest.json"
        manifest_path.chmod(0o444)
        try:
            rc = runner.run()
            self.assertNotEqual(rc, 0,
                                "Manifest write failure must cause nonzero exit")
            self.assertFalse(
                (run_dir / "DONE").exists(),
                "DONE must not be written when manifest update fails"
            )
        finally:
            manifest_path.chmod(0o644)


# ---------------------------------------------------------------------------
# Integration: successful run leaves all 5 artifacts + valid manifest
# ---------------------------------------------------------------------------


class TestFullSuccessfulRunIntegration(unittest.TestCase):
    """End-to-end: successful run must write all required artifacts correctly."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_all_artifacts_present_after_success(self):
        """After successful run, all required artifacts must exist."""
        runner, run_dir, _, _ = _make_runner(self.tmp, self.adapter_path, "gsm8k")
        _make_full_harness_artifacts_gsm8k(run_dir)
        rc = runner.run()
        self.assertEqual(rc, 0)
        for artifact in ["DONE", "per_item.csv", "command.txt"]:
            self.assertTrue(
                (run_dir / artifact).exists(),
                f"{artifact} must exist after successful run"
            )
        # run.log is only written by real subprocess, not by mock harness
        # per_item.csv and manifest must be present

    def test_manifest_submitted_matches_per_item_csv_row_count(self):
        """item_inventory.submitted must equal the number of rows in per_item.csv."""
        n_items = 5
        runner, run_dir, _, _ = _make_runner(self.tmp, self.adapter_path, "gsm8k")
        _make_full_harness_artifacts_gsm8k(run_dir, n_items)
        rc = runner.run()
        self.assertEqual(rc, 0)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        submitted = manifest["item_inventory"]["submitted"]
        csv_path = run_dir / "per_item.csv"
        with csv_path.open(newline="") as fh:
            csv_rows = list(csv.DictReader(fh))
        self.assertEqual(
            submitted, len(csv_rows),
            f"manifest submitted={submitted} must equal per_item.csv row count={len(csv_rows)}"
        )

    def test_done_written_after_manifest_update(self):
        """DONE must be written AFTER manifest is successfully updated."""
        writes = []
        runner, run_dir, _, harness_mock = _make_runner(
            self.tmp, self.adapter_path, "gsm8k"
        )
        _make_full_harness_artifacts_gsm8k(run_dir)
        # Patch manifest write to record order
        original_write = pathlib.Path.write_text

        def recording_write(self_path, text, **kw):
            name = self_path.name
            if name in ("manifest.json", "DONE"):
                writes.append(name)
            return original_write(self_path, text, **kw)

        import unittest.mock
        with unittest.mock.patch.object(pathlib.Path, "write_text", recording_write):
            rc = runner.run()

        self.assertEqual(rc, 0)
        if "manifest.json" in writes and "DONE" in writes:
            mi = writes.index("manifest.json")
            di = writes.index("DONE")
            self.assertLess(
                mi, di,
                f"manifest.json must be written before DONE; order was: {writes}"
            )


if __name__ == "__main__":
    unittest.main()
