"""tests/test_task7_authoritative_validator.py — Adversarial RED tests for Task 7.

These tests must FAIL (RED) against the current shallow implementation and pass (GREEN)
after the authoritative, fail-closed validator is implemented.

Spec gaps targeted (from task description):
  A1  Schema validation silently skips when contract import fails or schema absent —
      must block (fail-closed), not silently pass.
  A2  _verify_hashes lacks resolved-path containment: symlink escape, absolute-path
      injection, and ../.. traversal in suite_input_hashes are not rejected.
  A3  Validator does not parse/reconcile SWE grading categories — only checks file
      existence. Must validate: disjoint/exhaustive categories, all 100 IDs covered,
      no foreign IDs, grading_results always required for SWE publication.
  A4  Runner _update_manifest_on_completion silently swallows malformed grading JSON
      (except … pass), leaving submitted=0. Must fail-closed.
  A5  manifest.artifact_inventory.done_sentinel is set True BEFORE DONE is written —
      the manifest claim precedes reality. Must be set only after DONE exists.
  A6  Validate_campaign does not cross-check preds.json/exit_statuses.json ID sets
      against the frozen suite IDs (only checks file presence, not content).
  A7  Grading_results disjoint/exhaustive check: missing IDs → not just "covered" but
      reconciled so all expected = resolved + submitted_but_wrong + model_non_submission
      + infrastructure_failure (exactly 100 total) is enforced.
  A8  grading_results.json required for SWE publication regardless of manifest claim.
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
from unittest.mock import MagicMock

_TESTS_DIR = pathlib.Path(__file__).parent
_REPO_ROOT = _TESTS_DIR.parent
_VIZ_DIR = _REPO_ROOT / "viz"
for _p in (str(_TESTS_DIR), str(_VIZ_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import validate_campaign  # noqa: E402
import run_swebench       # noqa: E402
from schema_helpers import install_test_qualification  # noqa: E402

_REAL_SUITE = _REPO_ROOT / "suite" / "warpcore-v1.yaml"
_REAL_INSTANCES = _REPO_ROOT / "suite" / "swebench" / "instances-seed42-n100.json"

_FAKE_IDS = ["django__django-001", "django__django-002", "django__django-003"]

_CANONICAL_ADAPTER = {
    "adapter_schema_version": 1,
    "campaign_status": "canonical",
    "model": {
        "slug": "test-swe-adv",
        "id": "testorg/TestSWEAdv",
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

_PROMPT_TOKEN_MAXIMA = {"gsm8k": 500, "ifeval": 2000, "gpqa_diamond": 1000}


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _write_canonical_adapter(path: pathlib.Path) -> None:
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(_CANONICAL_ADAPTER, default_flow_style=False))


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


def _build_minimal_swebench_run(
    repo: pathlib.Path,
    *,
    model_slug: str = "sweb-adv",
    instance_ids: list | None = None,
    lifecycle: str = "current",
    execution_state: str = "completed",
    include_grading: bool = True,
    grading_override: dict | None = None,
    include_trajectories: bool = True,
    extra_manifest: dict | None = None,
) -> pathlib.Path:
    """Build a minimal SWE-bench run directory for validator testing."""
    suite_yaml = _make_suite_yaml(repo)
    suite_hash = hashlib.sha256(suite_yaml.read_bytes()).hexdigest()
    adapter_yaml = _make_adapter_yaml(repo, model_slug)
    adapter_hash = hashlib.sha256(adapter_yaml.read_bytes()).hexdigest()

    ids = instance_ids or _FAKE_IDS
    run_id = "run-2026-09-15T12-00-00"
    run_dir = repo / "results" / model_slug / "runs" / "warpcore-v1" / "swebench" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 1,
        "suite_id": "warpcore-v1",
        "suite_schema_version": 1,
        "run_id": run_id,
        "benchmark": "swebench",
        "adapter_hash": adapter_hash,
        "suite_input_hashes": {"suite/warpcore-v1.yaml": suite_hash},
        "serving_profile_digest": "sha256:" + "c" * 64,
        "model": {"slug": model_slug, "id": f"org/{model_slug}", "revision": "a" * 40},
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
            "submitted": len(ids),
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

    chain = ["planned", "preflight_passed", "running", "completed", "validated"]
    idx = chain.index(execution_state) if execution_state in chain else 3
    history = [{"state": s, "timestamp": f"2026-09-15T{12+i:02d}:00:00Z"}
               for i, s in enumerate(chain[:idx + 1])]
    status = {
        "schema_version": 1,
        "run_id": run_id,
        "suite_id": "warpcore-v1",
        "execution_state": execution_state,
        "lifecycle": lifecycle,
        "history": history,
    }

    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (run_dir / "status.json").write_text(json.dumps(status, indent=2))
    (run_dir / "DONE").write_text("done\n")
    (run_dir / "command.txt").write_text("python3 run_swebench.py\n")

    raw_dir = run_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    (raw_dir / "run.log").write_text("swebench exit 0\n")

    preds = {iid: {"model_patch": f"diff {iid}"} for iid in ids}
    (raw_dir / "preds.json").write_text(json.dumps(preds))

    statuses = {iid: "submitted" for iid in ids}
    (raw_dir / "exit_statuses.json").write_text(json.dumps(statuses))

    if include_grading:
        if grading_override is not None:
            grading = grading_override
        else:
            grading = {
                "resolved_ids": [],
                "unresolved_ids": ids,
                "empty_patch_ids": [],
                "error_ids": [],
                "incomplete_ids": [],
            }
        (raw_dir / "grading_results.json").write_text(json.dumps(grading))

    if include_trajectories:
        traj_dir = raw_dir / "trajectories"
        traj_dir.mkdir(exist_ok=True)
        for iid in ids:
            (traj_dir / f"{iid}.traj").write_text(json.dumps({"steps": []}))

    return run_dir


def _build_minimal_quality_run(
    repo: pathlib.Path,
    *,
    model_slug: str = "qual-adv",
    n_items: int = 5,
    lifecycle: str = "current",
    execution_state: str = "completed",
) -> pathlib.Path:
    """Build a minimal quality run for validator testing."""
    suite_yaml = _make_suite_yaml(repo)
    suite_hash = hashlib.sha256(suite_yaml.read_bytes()).hexdigest()
    adapter_yaml = _make_adapter_yaml(repo, model_slug)
    adapter_hash = hashlib.sha256(adapter_yaml.read_bytes()).hexdigest()

    ids = [f"item-{i}" for i in range(n_items)]
    run_id = "run-2026-09-15T12-00-00"
    run_dir = repo / "results" / model_slug / "runs" / "warpcore-v1" / "gsm8k" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 1,
        "suite_id": "warpcore-v1",
        "suite_schema_version": 1,
        "run_id": run_id,
        "benchmark": "gsm8k",
        "adapter_hash": adapter_hash,
        "suite_input_hashes": {"suite/warpcore-v1.yaml": suite_hash},
        "serving_profile_digest": "sha256:" + "c" * 64,
        "model": {"slug": model_slug, "id": f"org/{model_slug}", "revision": "a" * 40},
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

    chain = ["planned", "preflight_passed", "running", "completed", "validated"]
    idx = chain.index(execution_state) if execution_state in chain else 3
    history = [{"state": s, "timestamp": f"2026-09-15T{12+i:02d}:00:00Z"}
               for i, s in enumerate(chain[:idx + 1])]
    status = {
        "schema_version": 1,
        "run_id": run_id,
        "suite_id": "warpcore-v1",
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

    rows = [{"item_id": iid, "score": 1.0, "disposition": "correct",
             "finish_reason": "stop", "response_chars": 100, "empty_content": False}
            for iid in ids]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    (run_dir / "per_item.csv").write_text(buf.getvalue())

    with gzip.open(raw_dir / "samples_gsm8k.jsonl.gz", "wt") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    (raw_dir / "results_test.json").write_text(json.dumps({
        "results": {"gsm8k": {"exact_match,flexible-fallback": 1.0}},
        "n_samples": n_items,
    }))

    return run_dir


# ===========================================================================
# A1 — Schema validation must fail-closed (not silently skip)
# ===========================================================================

class TestA1SchemaValidationFailClosed(unittest.TestCase):
    """Schema validation must fail-closed when contract import fails OR schema absent.

    Current: _validate_manifest_schema / _validate_status_schema both do:
      `except ImportError: pass`  and  `if not schema.exists(): return`
    — silently accepting invalid manifests when the validator can't run.

    Required: missing schema or unavailable contract helper must BLOCK, not silently pass.
    A manifest claiming schema_version=99 must NEVER pass just because the schema file
    is absent.
    """

    def test_schema_absent_does_not_silently_pass_manifest(self):
        """When schema file is absent, validation must not silently pass a clearly invalid manifest.

        RED: current code does `if not _MANIFEST_SCHEMA.exists(): return` — passes anything.
        GREEN: absence of schema must be a blocking error.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            run_dir = _build_minimal_quality_run(repo)
            suite_yaml = _make_suite_yaml(repo)
            adapter_yaml = _make_adapter_yaml(repo, "qual-adv")

            # Monkeypatch schema paths to non-existent files
            real_manifest_schema = validate_campaign._MANIFEST_SCHEMA
            real_status_schema = validate_campaign._STATUS_SCHEMA
            try:
                # Point schema paths to nonexistent paths so the "if not exists: return" fires
                validate_campaign._MANIFEST_SCHEMA = repo / "nonexistent" / "manifest.schema.json"
                validate_campaign._STATUS_SCHEMA = repo / "nonexistent" / "result-status.schema.json"

                result = validate_campaign.validate(run_dir, suite_yaml, adapter_yaml)
                # KEY ASSERTION: must not silently pass when schema is missing
                # The validator is supposed to BLOCK if it cannot verify the schema
                self.assertFalse(
                    result.passed,
                    "validate() must fail-closed when schema files are absent. "
                    "Currently silently skips and passes — this is a blocking gap. "
                    f"errors: {result.errors}"
                )
                schema_errors = [e for e in result.errors
                                 if "schema" in e.lower() or "contract" in e.lower()]
                self.assertTrue(
                    len(schema_errors) > 0,
                    f"Must report schema-unavailability error; got: {result.errors}"
                )
            finally:
                validate_campaign._MANIFEST_SCHEMA = real_manifest_schema
                validate_campaign._STATUS_SCHEMA = real_status_schema

    def test_contract_import_failure_does_not_silently_pass_manifest(self):
        """When contract module can't be imported, validation must BLOCK, not silently skip.

        RED: current code does `except ImportError: pass` in _validate_manifest_schema.
        GREEN: ImportError on contract must be a blocking error.

        Tests the schema paths exist in the real repo (integration path).
        """
        schema_dir = _REPO_ROOT / "suite" / "schemas"
        if not schema_dir.exists():
            self.skipTest("Schema directory not present; skip schema-integration test")

        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            run_dir = _build_minimal_quality_run(repo)
            suite_yaml = _make_suite_yaml(repo)
            adapter_yaml = _make_adapter_yaml(repo, "qual-adv")

            # Corrupt manifest to have invalid schema_version
            manifest = json.loads((run_dir / "manifest.json").read_text())
            manifest["schema_version"] = 99  # must not pass
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            # Temporarily hide the contract module to trigger ImportError path
            import sys
            original_contract = sys.modules.get("contract")
            sys.modules["contract"] = None  # type: ignore[assignment]
            try:
                result = validate_campaign.validate(run_dir, suite_yaml, adapter_yaml)
                # The `except ImportError: pass` is the gap — currently passes
                # After fix: must fail when contract unavailable AND schemas exist
                # (because we can't verify the schema at all — fail-closed)
                self.assertFalse(
                    result.passed,
                    "validate() must fail-closed when contract module is unavailable "
                    "AND schema files exist but cannot be verified. "
                    "Currently silently skips schema validation via `except ImportError: pass`. "
                    f"errors: {result.errors}"
                )
            finally:
                if original_contract is None:
                    del sys.modules["contract"]
                else:
                    sys.modules["contract"] = original_contract


# ===========================================================================
# A2 — Path containment in _verify_hashes
# ===========================================================================

class TestA2PathContainmentVerifyHashes(unittest.TestCase):
    """Suite input hash traversal must be contained inside repo root.

    Current _verify_hashes: abs_path = repo / rel_path  — no check that
    abs_path.resolve() is still inside repo.resolve().
    A crafted rel_path can escape via symlinks or ../.. traversal.
    """

    def test_dotdot_traversal_in_suite_input_hash_blocked(self):
        """../../etc/passwd in suite_input_hashes must be blocked.

        Acceptable mechanisms: schema regex rejection, explicit containment check,
        or path-traversal detection — any of these count. Critical: must not pass.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            run_dir = _build_minimal_quality_run(repo)
            suite_yaml = _make_suite_yaml(repo)
            adapter_yaml = _make_adapter_yaml(repo, "qual-adv")

            # Inject a traversal path
            manifest = json.loads((run_dir / "manifest.json").read_text())
            manifest["suite_input_hashes"] = {
                "../../etc/passwd": "a" * 64,
            }
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            result = validate_campaign.validate(run_dir, suite_yaml, adapter_yaml)
            # Must not pass — any blocking mechanism (schema regex, containment check) is OK
            self.assertFalse(result.passed, "Path traversal must fail validation")
            # Must have at least one error (schema violation or containment error)
            self.assertTrue(
                len(result.errors) > 0,
                f"Must report at least one error for ../../ traversal; got: {result.errors}"
            )

    def test_absolute_path_in_suite_input_hash_blocked(self):
        """An absolute path like /etc/passwd in suite_input_hashes must be blocked.

        Acceptable mechanisms: schema regex rejection, explicit containment check.
        Critical: must not pass.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            run_dir = _build_minimal_quality_run(repo)
            suite_yaml = _make_suite_yaml(repo)
            adapter_yaml = _make_adapter_yaml(repo, "qual-adv")

            manifest = json.loads((run_dir / "manifest.json").read_text())
            manifest["suite_input_hashes"] = {
                "/etc/passwd": "a" * 64,
            }
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            result = validate_campaign.validate(run_dir, suite_yaml, adapter_yaml)
            self.assertFalse(result.passed, "Absolute path traversal must fail validation")
            self.assertTrue(
                len(result.errors) > 0,
                f"Must report at least one error for absolute path; got: {result.errors}"
            )

    def test_symlink_escape_in_suite_hash_path_blocked(self):
        """A symlink at a repo-internal path pointing outside repo must be rejected.

        RED: _verify_hashes uses abs_path = repo / rel_path, reads it without
             checking if the resolved path exits repo.
        GREEN: must resolve the path and verify it's still inside repo.resolve().
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            run_dir = _build_minimal_quality_run(repo)
            suite_yaml = _make_suite_yaml(repo)
            adapter_yaml = _make_adapter_yaml(repo, "qual-adv")

            # Create a file OUTSIDE the repo
            outside_file = pathlib.Path(tmp) / "outside_secret.txt"
            outside_file.write_text("secret content\n")

            # Create a symlink INSIDE repo pointing outside
            repo_inner = repo / "suite"
            repo_inner.mkdir(parents=True, exist_ok=True)
            symlink_path = repo_inner / "symlinked_secret.txt"
            try:
                symlink_path.symlink_to(outside_file)
            except (OSError, NotImplementedError):
                self.skipTest("Symlinks not supported on this filesystem")

            # Inject the symlink path into suite_input_hashes
            manifest = json.loads((run_dir / "manifest.json").read_text())
            # Record the hash of the target file content (as if attacker knows it)
            correct_hash = hashlib.sha256(outside_file.read_bytes()).hexdigest()
            manifest["suite_input_hashes"] = {
                "suite/symlinked_secret.txt": correct_hash,
            }
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            result = validate_campaign.validate(run_dir, suite_yaml, adapter_yaml)
            # Must not silently pass when a symlink escapes the repo root
            # Either: containment error OR hash mismatch (acceptable fallback)
            # Critical: must NOT return passed=True
            self.assertFalse(
                result.passed,
                "Suite hash path that resolves via symlink outside repo must not pass. "
                f"errors: {result.errors}"
            )


# ===========================================================================
# A3 — SWE grading_results content validation (not just existence)
# ===========================================================================

class TestA3SwebenchGradingContentValidation(unittest.TestCase):
    """validate_campaign must parse and validate grading_results.json content.

    Current: _check_required_files only checks file existence for swebench.
    Required:
      - disjoint categories: no ID appears in more than one category
      - exhaustive coverage: all expected IDs appear across categories
      - no foreign IDs: IDs not in the frozen set must be rejected
      - grading_results.json always required for SWE publication
    """

    def test_duplicate_id_across_grading_categories_fails(self):
        """An ID appearing in both resolved_ids and unresolved_ids must be rejected.

        RED: current validate_campaign doesn't parse grading_results.json content.
        GREEN: must detect and reject duplicate IDs across grading categories.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            # Duplicate: ids[0] in both resolved and unresolved
            grading = {
                "resolved_ids": [_FAKE_IDS[0]],
                "unresolved_ids": [_FAKE_IDS[0], _FAKE_IDS[1], _FAKE_IDS[2]],
                "empty_patch_ids": [],
                "error_ids": [],
                "incomplete_ids": [],
            }
            run_dir = _build_minimal_swebench_run(repo, grading_override=grading)
            suite_yaml = _make_suite_yaml(repo)
            adapter_yaml = _make_adapter_yaml(repo, "sweb-adv")

            result = validate_campaign.validate(run_dir, suite_yaml, adapter_yaml)
            self.assertFalse(
                result.passed,
                "Duplicate instance ID across grading categories must fail validation. "
                f"errors: {result.errors}"
            )
            dup_errors = [e for e in result.errors
                          if "duplic" in e.lower() or "overlap" in e.lower()
                          or "disjoint" in e.lower() or "multiple" in e.lower()]
            self.assertTrue(
                len(dup_errors) > 0,
                f"Must report duplicate/overlap error; got: {result.errors}"
            )

    def test_missing_instance_in_grading_categories_fails(self):
        """An expected instance not covered in any grading category must fail.

        RED: current validator doesn't parse grading_results content.
        GREEN: must detect and reject runs where expected instances have no disposition.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            # ids[2] is missing from all grading categories
            grading = {
                "resolved_ids": [],
                "unresolved_ids": [_FAKE_IDS[0], _FAKE_IDS[1]],  # missing ids[2]
                "empty_patch_ids": [],
                "error_ids": [],
                "incomplete_ids": [],
            }
            run_dir = _build_minimal_swebench_run(repo, grading_override=grading)
            suite_yaml = _make_suite_yaml(repo)
            adapter_yaml = _make_adapter_yaml(repo, "sweb-adv")

            result = validate_campaign.validate(run_dir, suite_yaml, adapter_yaml)
            self.assertFalse(
                result.passed,
                "Missing instance ID in grading categories must fail validation. "
                f"errors: {result.errors}"
            )
            coverage_errors = [e for e in result.errors
                               if "missing" in e.lower() or "cover" in e.lower()
                               or "disposition" in e.lower() or "grading" in e.lower()]
            self.assertTrue(
                len(coverage_errors) > 0,
                f"Must report coverage error for missing instance; got: {result.errors}"
            )

    def test_foreign_id_in_grading_categories_fails(self):
        """An instance ID not in the expected set in grading_results must fail.

        RED: current validator doesn't parse grading_results content.
        GREEN: must detect and reject foreign IDs in grading categories.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            foreign_id = "not-in-expected-set__foreign-123"
            # Include all expected + one foreign
            grading = {
                "resolved_ids": [],
                "unresolved_ids": _FAKE_IDS + [foreign_id],  # foreign ID
                "empty_patch_ids": [],
                "error_ids": [],
                "incomplete_ids": [],
            }
            run_dir = _build_minimal_swebench_run(repo, grading_override=grading)
            suite_yaml = _make_suite_yaml(repo)
            adapter_yaml = _make_adapter_yaml(repo, "sweb-adv")

            result = validate_campaign.validate(run_dir, suite_yaml, adapter_yaml)
            self.assertFalse(
                result.passed,
                "Foreign instance ID in grading categories must fail validation. "
                f"errors: {result.errors}"
            )
            foreign_errors = [e for e in result.errors
                              if "foreign" in e.lower() or "unexpected" in e.lower()
                              or "unknown" in e.lower() or "not in" in e.lower()
                              or "grading" in e.lower()]
            self.assertTrue(
                len(foreign_errors) > 0,
                f"Must report foreign ID error; got: {result.errors}"
            )

    def test_grading_results_required_for_swe_publication_even_if_manifest_claims_false(self):
        """grading_results.json must always be required for SWE-bench publication.

        Even if manifest.artifact_inventory.grading_results_json = False,
        a for_publication=True call must require and validate grading_results.json.

        RED: current validator trusts manifest claim, doesn't require grading content.
        GREEN: must always require grading for SWE publication.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            # Build run with grading but set manifest claim to False
            run_dir = _build_minimal_swebench_run(
                repo,
                extra_manifest={"artifact_inventory": {
                    "preds_json": True,
                    "exit_statuses_json": True,
                    "grading_results_json": False,  # claim it's absent
                    "run_log": True,
                    "command_txt": True,
                    "done_sentinel": True,
                    "samples_jsonl_gz": False,
                    "per_item_csv": False,
                }},
            )
            # Remove the grading file to match the manifest claim
            grading_path = run_dir / "raw" / "grading_results.json"
            if grading_path.exists():
                grading_path.unlink()

            suite_yaml = _make_suite_yaml(repo)
            adapter_yaml = _make_adapter_yaml(repo, "sweb-adv")

            result = validate_campaign.validate(run_dir, suite_yaml, adapter_yaml,
                                               for_publication=True)
            self.assertFalse(
                result.passed,
                "SWE-bench for_publication=True must require grading_results.json "
                "regardless of manifest.artifact_inventory.grading_results_json claim. "
                f"errors: {result.errors}"
            )
            grading_errors = [e for e in result.errors if "grading" in e.lower()]
            self.assertTrue(
                len(grading_errors) > 0,
                f"Must report missing grading_results error; got: {result.errors}"
            )

    def test_malformed_grading_results_json_fails(self):
        """grading_results.json that is not valid JSON must fail validation.

        RED: current validator doesn't parse grading_results content.
        GREEN: malformed JSON must fail with a parsing error.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            run_dir = _build_minimal_swebench_run(repo)
            suite_yaml = _make_suite_yaml(repo)
            adapter_yaml = _make_adapter_yaml(repo, "sweb-adv")

            # Write malformed JSON
            (run_dir / "raw" / "grading_results.json").write_text("not valid json {{{")

            result = validate_campaign.validate(run_dir, suite_yaml, adapter_yaml)
            self.assertFalse(
                result.passed,
                "Malformed grading_results.json must fail validation. "
                f"errors: {result.errors}"
            )
            parse_errors = [e for e in result.errors
                            if "json" in e.lower() or "parse" in e.lower()
                            or "grading" in e.lower() or "invalid" in e.lower()]
            self.assertTrue(
                len(parse_errors) > 0,
                f"Must report parse error; got: {result.errors}"
            )

    def test_valid_swebench_run_with_correct_grading_passes(self):
        """Regression: valid SWE-bench run with correct grading content must still pass.

        GREEN after fix: must not break valid runs.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            grading = {
                "resolved_ids": [_FAKE_IDS[0]],
                "unresolved_ids": [_FAKE_IDS[1]],
                "empty_patch_ids": [],
                "error_ids": [],
                "incomplete_ids": [_FAKE_IDS[2]],
            }
            run_dir = _build_minimal_swebench_run(repo, grading_override=grading)
            suite_yaml = _make_suite_yaml(repo)
            adapter_yaml = _make_adapter_yaml(repo, "sweb-adv")

            result = validate_campaign.validate(run_dir, suite_yaml, adapter_yaml)
            self.assertTrue(
                result.passed,
                f"Valid SWE-bench run with correct grading must pass; errors: {result.errors}"
            )


# ===========================================================================
# A4 — Runner: malformed grading must fail-closed, not silently leave submitted=0
# ===========================================================================

class TestA4RunnerMalformedGradingFailClosed(unittest.TestCase):
    """Runner _update_manifest_on_completion must fail-closed on malformed grading.

    Current: `except (json.JSONDecodeError, OSError): pass` silently leaves submitted=0.
    Required: malformed or unreadable grading_results.json must block completion.
    """

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-swe-adv.yaml"
        _write_canonical_adapter(self.adapter_path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_runner_with_malformed_grading(self, grading_content: str):
        """Build a SwebenchRunner where grading_results.json has invalid content."""
        instance_ids = json.loads(_REAL_INSTANCES.read_text())

        run_id = "run-malformed-grading"
        slug = _CANONICAL_ADAPTER["model"]["slug"]
        run_dir = self.tmp / "results" / slug / "runs" / "warpcore-v1" / "swebench" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        status = {
            "schema_version": 1,
            "run_id": run_id,
            "suite_id": "warpcore-v1",
            "execution_state": "planned",
            "lifecycle": "current",
            "history": [{"state": "planned", "timestamp": "2026-09-15T12:00:00Z"}],
        }
        manifest = {
            "suite_id": "warpcore-v1",
            "run_id": run_id,
            "benchmark": "swebench",
            "model": {"slug": slug, "id": "testorg/TestSWEAdv", "revision": "a" * 40},
            "item_inventory": {"expected": len(instance_ids), "submitted": 0},
            "timing": {"started_utc": "2026-09-15T12:00:00Z", "completed_utc": None},
            "artifact_inventory": {
                "preds_json": False, "exit_statuses_json": False,
                "grading_results_json": False, "run_log": False,
                "command_txt": False, "done_sentinel": False,
            },
        }
        (run_dir / "status.json").write_text(json.dumps(status))
        (run_dir / "manifest.json").write_text(json.dumps(manifest))
        (run_dir / "command.txt").write_text("mock command\n")

        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        preds = {iid: {"model_patch": f"diff"} for iid in instance_ids}
        (raw_dir / "preds.json").write_text(json.dumps(preds))
        statuses = {iid: "submitted" for iid in instance_ids}
        (raw_dir / "exit_statuses.json").write_text(json.dumps(statuses))
        traj_dir = raw_dir / "trajectories"
        traj_dir.mkdir()
        for iid in instance_ids:
            (traj_dir / f"{iid}.traj").write_text(json.dumps({"steps": []}))
        (raw_dir / "run.log").write_text("mock log\n")

        # Write malformed grading content
        (raw_dir / "grading_results.json").write_text(grading_content)

        def gen_runner(cfg, rd): return 0
        def grad_runner(preds_path, rd): return 0

        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://fake:8000/v1",
            run_dir=run_dir,
            repo=self.tmp,
            allow_no_screen=True,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            preflight_runner=MagicMock(return_value=0),
            generation_runner=gen_runner,
            grading_runner=grad_runner,
        )
        return runner, run_dir

    def test_malformed_grading_json_runner_fails_closed(self):
        """Runner must fail-closed (not exit 0) when grading_results.json is malformed JSON.

        RED: current _update_manifest_on_completion does `except (json.JSONDecodeError, OSError): pass`
             so it silently leaves submitted=0 and proceeds to write DONE.
        GREEN: malformed grading must cause exit-nonzero, not silent submitted=0.
        """
        runner, run_dir = self._make_runner_with_malformed_grading("not valid json {{{")
        rc = runner.run()
        self.assertNotEqual(
            rc, 0,
            "Runner must NOT exit 0 when grading_results.json is malformed. "
            "Currently silently swallows the error and proceeds. "
            f"run_dir={run_dir}"
        )
        # Also verify DONE was not written
        self.assertFalse(
            (run_dir / "DONE").exists(),
            "DONE must NOT be written when grading_results.json is malformed"
        )

    def test_malformed_grading_submitted_stays_zero_but_run_fails(self):
        """When grading JSON is invalid, run must fail, not silently proceed with submitted=0.

        The current behavior swallows the error and writes DONE with submitted=0.
        After fix: submitted stays 0 AND run exits nonzero.
        """
        runner, run_dir = self._make_runner_with_malformed_grading("{invalid}")
        rc = runner.run()
        # Primary assertion: exit nonzero (fail-closed)
        self.assertNotEqual(rc, 0, "Runner must fail when grading JSON is invalid")
        # Corollary: even if manifest was somehow written, submitted should be 0
        manifest = json.loads((run_dir / "manifest.json").read_text())
        submitted = manifest.get("item_inventory", {}).get("submitted", 0)
        # submitted=0 is acceptable ONLY if the run actually failed (nonzero rc)
        # This verifies the runner didn't "succeed" with a wrong count
        if rc == 0:
            self.fail(
                f"Runner exited 0 with malformed grading — submitted={submitted} "
                f"(should have failed)"
            )


# ===========================================================================
# A5 — done_sentinel must not be set True before DONE actually exists
# ===========================================================================

class TestA5DoneSentinelTiming(unittest.TestCase):
    """manifest.artifact_inventory.done_sentinel must only be True after DONE exists.

    Current: _update_manifest_on_completion sets art["done_sentinel"] = True
    unconditionally, then writes manifest, then _write_done writes DONE.
    This means between manifest write and DONE write, manifest claims DONE exists
    but it doesn't (race condition / inconsistent state).

    Required: the manifest must only ever claim done_sentinel=True AFTER DONE
    actually exists on disk. This means done_sentinel should be set to True
    AFTER _write_done succeeds, not before.
    """

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-swe-adv.yaml"
        _write_canonical_adapter(self.adapter_path)
        # A live launch is gated on a SWE-bench qualification record; install a
        # valid one so the DONE-ordering assertions are reached at all.
        install_test_qualification(
            repo=self.tmp,
            adapter_path=self.adapter_path,
            endpoint="http://fake:8000/v1",
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_done_sentinel_not_set_true_in_manifest_before_done_written(self):
        """After manifest write (step 0) but before DONE write (step 1),
        artifact_inventory.done_sentinel must NOT be True in the on-disk manifest.

        RED: current _update_manifest_on_completion sets done_sentinel=True then writes
             manifest — the manifest already claims DONE before DONE exists.
        GREEN: done_sentinel must be set True only AFTER DONE exists on disk.
        """
        instance_ids = json.loads(_REAL_INSTANCES.read_text())
        slug = _CANONICAL_ADAPTER["model"]["slug"]
        run_id = "run-done-timing"
        run_dir = self.tmp / "results" / slug / "runs" / "warpcore-v1" / "swebench" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        status = {
            "schema_version": 1,
            "run_id": run_id,
            "suite_id": "warpcore-v1",
            "execution_state": "planned",
            "lifecycle": "current",
            "history": [{"state": "planned", "timestamp": "2026-09-15T12:00:00Z"}],
        }
        manifest_base = {
            "suite_id": "warpcore-v1",
            "run_id": run_id,
            "benchmark": "swebench",
            "model": {"slug": slug, "id": "testorg/TestSWEAdv", "revision": "a" * 40},
            "item_inventory": {"expected": len(instance_ids), "submitted": 0},
            "timing": {"started_utc": "2026-09-15T12:00:00Z", "completed_utc": None},
            "artifact_inventory": {
                "preds_json": False, "exit_statuses_json": False,
                "grading_results_json": False, "run_log": False,
                "command_txt": False, "done_sentinel": False,
            },
        }
        (run_dir / "status.json").write_text(json.dumps(status))
        (run_dir / "manifest.json").write_text(json.dumps(manifest_base))
        (run_dir / "command.txt").write_text("mock command\n")

        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        preds = {iid: {"model_patch": ""} for iid in instance_ids}
        (raw_dir / "preds.json").write_text(json.dumps(preds))
        statuses = {iid: "submitted" for iid in instance_ids}
        (raw_dir / "exit_statuses.json").write_text(json.dumps(statuses))
        traj_dir = raw_dir / "trajectories"
        traj_dir.mkdir()
        for iid in instance_ids:
            (traj_dir / f"{iid}.traj").write_text(json.dumps({"steps": []}))
        (raw_dir / "run.log").write_text("mock log\n")
        grading = {
            "resolved_ids": [],
            "unresolved_ids": instance_ids,
            "empty_patch_ids": [],
            "error_ids": [],
            "incomplete_ids": [],
        }
        (raw_dir / "grading_results.json").write_text(json.dumps(grading))

        # Track: read manifest.done_sentinel at the moment BEFORE DONE is written
        manifest_state_before_done = {}

        original_write_done = run_swebench.SwebenchRunner._write_done

        def intercepting_write_done(self_runner, done_path):
            # Read manifest JUST before DONE is written
            current_manifest = json.loads((self_runner.run_dir / "manifest.json").read_text())
            manifest_state_before_done["done_sentinel"] = (
                current_manifest.get("artifact_inventory", {}).get("done_sentinel")
            )
            # Now do the actual write
            original_write_done(self_runner, done_path)

        def gen_runner(cfg, rd): return 0
        def grad_runner(preds_path, rd): return 0

        import unittest.mock
        with unittest.mock.patch.object(
            run_swebench.SwebenchRunner, "_write_done", intercepting_write_done
        ):
            runner = run_swebench.SwebenchRunner(
                suite_path=_REAL_SUITE,
                adapter_path=self.adapter_path,
                endpoint="http://fake:8000/v1",
                run_dir=run_dir,
                repo=self.tmp,
                allow_no_screen=True,
                prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
                preflight_runner=MagicMock(return_value=0),
                generation_runner=gen_runner,
                grading_runner=grad_runner,
            )
            rc = runner.run()

        self.assertEqual(rc, 0, f"Runner must succeed for this test to be meaningful; rc={rc}")
        self.assertIn("done_sentinel", manifest_state_before_done,
                      "Intercept must have captured manifest state")

        # KEY ASSERTION: done_sentinel must NOT be True in manifest before DONE is written
        self.assertFalse(
            manifest_state_before_done["done_sentinel"],
            "manifest.artifact_inventory.done_sentinel must be False/absent when "
            "_write_done is called (DONE hasn't been written yet). "
            "Current implementation sets it True unconditionally BEFORE writing DONE — "
            "this creates a window where manifest claims DONE but DONE doesn't exist. "
            f"Got done_sentinel={manifest_state_before_done['done_sentinel']!r} in manifest "
            "at the moment _write_done was called."
        )

    def test_done_sentinel_true_in_manifest_after_done_written(self):
        """After successful completion, done_sentinel must be True in final manifest."""
        instance_ids = json.loads(_REAL_INSTANCES.read_text())
        slug = _CANONICAL_ADAPTER["model"]["slug"]
        run_id = "run-done-after"
        run_dir = self.tmp / "results" / slug / "runs" / "warpcore-v1" / "swebench" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        status = {
            "schema_version": 1, "run_id": run_id, "suite_id": "warpcore-v1",
            "execution_state": "planned", "lifecycle": "current",
            "history": [{"state": "planned", "timestamp": "2026-09-15T12:00:00Z"}],
        }
        manifest_base = {
            "suite_id": "warpcore-v1", "run_id": run_id, "benchmark": "swebench",
            "model": {"slug": slug, "id": "testorg/TestSWEAdv", "revision": "a" * 40},
            "item_inventory": {"expected": len(instance_ids), "submitted": 0},
            "timing": {"started_utc": "2026-09-15T12:00:00Z", "completed_utc": None},
            "artifact_inventory": {
                "preds_json": False, "exit_statuses_json": False,
                "grading_results_json": False, "run_log": False,
                "command_txt": False, "done_sentinel": False,
            },
        }
        (run_dir / "status.json").write_text(json.dumps(status))
        (run_dir / "manifest.json").write_text(json.dumps(manifest_base))
        (run_dir / "command.txt").write_text("mock command\n")
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / "preds.json").write_text(json.dumps({iid: {} for iid in instance_ids}))
        (raw_dir / "exit_statuses.json").write_text(json.dumps({iid: "submitted" for iid in instance_ids}))
        traj_dir = raw_dir / "trajectories"
        traj_dir.mkdir()
        for iid in instance_ids:
            (traj_dir / f"{iid}.traj").write_text(json.dumps({"steps": []}))
        (raw_dir / "run.log").write_text("mock log\n")
        (raw_dir / "grading_results.json").write_text(json.dumps({
            "resolved_ids": [], "unresolved_ids": instance_ids,
            "empty_patch_ids": [], "error_ids": [], "incomplete_ids": [],
        }))

        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://fake:8000/v1",
            run_dir=run_dir, repo=self.tmp, allow_no_screen=True,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            preflight_runner=MagicMock(return_value=0),
            generation_runner=lambda cfg, rd: 0,
            grading_runner=lambda pp, rd: 0,
        )
        rc = runner.run()
        self.assertEqual(rc, 0)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        self.assertTrue(
            manifest.get("artifact_inventory", {}).get("done_sentinel"),
            "Final manifest must have done_sentinel=True after successful completion"
        )


# ===========================================================================
# Regression: valid runs must still pass after all fixes
# ===========================================================================

class TestA9RegressionValidRuns(unittest.TestCase):
    """Regression: valid runs must continue to pass after all adversarial fixes."""

    def test_valid_swebench_run_passes(self):
        """A fully valid SWE-bench run must pass validate() after fixes."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            grading = {
                "resolved_ids": [_FAKE_IDS[0]],
                "unresolved_ids": [_FAKE_IDS[1]],
                "empty_patch_ids": [],
                "error_ids": [],
                "incomplete_ids": [_FAKE_IDS[2]],
            }
            run_dir = _build_minimal_swebench_run(repo, grading_override=grading)
            suite_yaml = _make_suite_yaml(repo)
            adapter_yaml = _make_adapter_yaml(repo, "sweb-adv")
            result = validate_campaign.validate(run_dir, suite_yaml, adapter_yaml)
            self.assertTrue(
                result.passed,
                f"Valid SWE-bench run must pass after fixes; errors: {result.errors}"
            )

    def test_valid_quality_run_passes(self):
        """A fully valid quality run must pass validate() after fixes."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            run_dir = _build_minimal_quality_run(repo)
            suite_yaml = _make_suite_yaml(repo)
            adapter_yaml = _make_adapter_yaml(repo, "qual-adv")
            result = validate_campaign.validate(run_dir, suite_yaml, adapter_yaml)
            self.assertTrue(
                result.passed,
                f"Valid quality run must pass after fixes; errors: {result.errors}"
            )


# ===========================================================================
# Task 7 NEW RED tests — Suite-driven evidence & frozen SWE inventory gate
# ===========================================================================
# Spec gaps targeted:
#   T1  Suite instance_set_file must be loaded and its SHA-256 verified against
#       suite yaml instances_sha256; a mismatch must block validation.
#   T2  Frozen ID set from suite (not preds.json) is the authority:
#       preds.json with foreign IDs not in the frozen set must fail.
#   T3  preds.json must have EXACT key equality to frozen IDs (no missing, no extra).
#   T4  exit_statuses.json must have EXACT key equality to frozen IDs.
#   T5  Trajectories: a trajectory file must exist for every frozen ID (not just dir).
#   T6  manifest.item_inventory.instance_ids_hash must match the suite-frozen set
#       digest, not just any consistent set.
#   T7  manifest.item_inventory.expected must equal the suite expected_item_count
#       (not just any number).
#   T8  Adapter path containment in _verify_hashes: adapter symlink escaping repo
#       must be blocked.
#   T9  Suite-driven required_evidence: if suite YAML lists a required_evidence
#       name that maps to a missing file, validation must fail regardless of
#       manifest artifact_inventory booleans.
#   T10 Unknown required_evidence name in suite YAML must be rejected.
# ===========================================================================

def _make_full_suite_yaml(
    repo: pathlib.Path,
    frozen_ids: list,
    inst_hash: str | None = None,
    extra_benchmark_fields: dict | None = None,
) -> pathlib.Path:
    """Build a suite YAML with a real swebench section pointing to the instance file."""
    import yaml  # type: ignore[import]
    instances_file = repo / "suite" / "swebench" / "instances.json"
    instances_file.parent.mkdir(parents=True, exist_ok=True)
    instances_file.write_text(json.dumps(frozen_ids))
    if inst_hash is None:
        inst_hash = hashlib.sha256(instances_file.read_bytes()).hexdigest()

    swebench_cfg = {
        "description": "SWE-bench test",
        "instance_set_file": "suite/swebench/instances.json",
        "instances_sha256": inst_hash,
        "expected_item_count": len(frozen_ids),
        "required_evidence": [
            "preds_json", "exit_statuses", "run_log",
            "command_txt", "manifest_json", "status_json", "done_sentinel",
        ],
    }
    if extra_benchmark_fields:
        swebench_cfg.update(extra_benchmark_fields)

    suite_data = {
        "suite_id": "warpcore-v1",
        "suite_schema_version": 1,
        "benchmarks": {"swebench": swebench_cfg},
    }
    suite_yaml = repo / "suite" / "warpcore-v1.yaml"
    suite_yaml.parent.mkdir(parents=True, exist_ok=True)
    suite_yaml.write_text(yaml.dump(suite_data, default_flow_style=False))
    return suite_yaml


def _build_swebench_run_with_frozen(
    repo: pathlib.Path,
    frozen_ids: list,
    preds_ids: list | None = None,
    statuses_ids: list | None = None,
    traj_ids: list | None = None,
    include_grading: bool = False,
    grading_override: dict | None = None,
    extra_manifest: dict | None = None,
    model_slug: str = "swe-frozen-test",
) -> pathlib.Path:
    """Build a SWE run directory referencing frozen_ids; preds/statuses/traj can differ."""
    preds_ids = preds_ids if preds_ids is not None else frozen_ids
    statuses_ids = statuses_ids if statuses_ids is not None else frozen_ids
    traj_ids = traj_ids if traj_ids is not None else frozen_ids

    suite_yaml = repo / "suite" / "warpcore-v1.yaml"
    adapter_yaml = repo / "adapters" / f"{model_slug}.yaml"
    adapter_yaml.parent.mkdir(parents=True, exist_ok=True)
    adapter_yaml.write_text("adapter_schema_version: 1\ncampaign_status: canonical\n")
    adapter_hash = hashlib.sha256(adapter_yaml.read_bytes()).hexdigest()
    suite_hash = hashlib.sha256(suite_yaml.read_bytes()).hexdigest()

    frozen_ids_hash = hashlib.sha256(
        json.dumps(sorted(frozen_ids), sort_keys=True).encode()
    ).hexdigest()

    run_id = "run-2026-09-15T10-00-00"
    run_dir = (
        repo / "results" / model_slug / "runs" / "warpcore-v1" / "swebench" / run_id
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 1,
        "suite_id": "warpcore-v1",
        "suite_schema_version": 1,
        "run_id": run_id,
        "benchmark": "swebench",
        "adapter_hash": adapter_hash,
        "suite_input_hashes": {"suite/warpcore-v1.yaml": suite_hash},
        "serving_profile_digest": "sha256:" + "c" * 64,
        "model": {"slug": model_slug, "id": f"org/{model_slug}", "revision": "a" * 40},
        "serving": {
            "image_digest": "sha256:" + "d" * 64,
            "engine": "vllm",
            "engine_version": "0.6.6",
            "effective_args": [],
            "environment": {},
            "hardware_id": "dgx-spark-gb10",
        },
        "item_inventory": {
            "expected": len(frozen_ids),
            "submitted": len(preds_ids),
            "instance_ids_hash": frozen_ids_hash,
        },
        "timing": {
            "started_utc": "2026-09-15T10:00:00Z",
            "completed_utc": "2026-09-15T12:00:00Z",
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
            {"state": "planned", "timestamp": "2026-09-15T10:00:00Z"},
            {"state": "preflight_passed", "timestamp": "2026-09-15T10:01:00Z"},
            {"state": "running", "timestamp": "2026-09-15T10:02:00Z"},
            {"state": "completed", "timestamp": "2026-09-15T12:00:00Z"},
        ],
    }

    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (run_dir / "status.json").write_text(json.dumps(status, indent=2))
    (run_dir / "DONE").write_text("done\n")
    (run_dir / "command.txt").write_text("python3 run_swebench.py\n")

    raw_dir = run_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    (raw_dir / "run.log").write_text("swebench exit 0\n")
    preds = {iid: {"model_patch": f"diff {iid}"} for iid in preds_ids}
    (raw_dir / "preds.json").write_text(json.dumps(preds))
    statuses_data = {iid: "submitted" for iid in statuses_ids}
    (raw_dir / "exit_statuses.json").write_text(json.dumps(statuses_data))

    traj_dir = raw_dir / "trajectories"
    traj_dir.mkdir(exist_ok=True)
    for iid in traj_ids:
        (traj_dir / f"{iid}.traj").write_text(json.dumps({"steps": []}))

    if include_grading:
        if grading_override is not None:
            grading = grading_override
        else:
            grading = {
                "resolved_ids": [],
                "unresolved_ids": list(frozen_ids),
                "empty_patch_ids": [],
                "error_ids": [],
                "incomplete_ids": [],
            }
        (raw_dir / "grading_results.json").write_text(json.dumps(grading))

    return run_dir


class TestT1SuiteInstanceHashVerification(unittest.TestCase):
    """T1: Suite instance_set_file SHA-256 must be loaded and verified.

    If the instance file on disk has a hash that does not match the
    instances_sha256 recorded in the suite YAML, validation must fail.
    Currently the validator never reads the instance file at all.
    """

    def test_instance_file_hash_mismatch_fails(self):
        """Suite instances_sha256 mismatch must block validation (currently passes = RED).

        RED: validator doesn't load/check instance file hash from suite YAML.
        GREEN: must verify SHA-256 of instance_set_file against suite instances_sha256.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            frozen_ids = ["django__django-001", "django__django-002", "django__django-003"]

            # Write instance file with wrong hash in suite yaml
            wrong_hash = "a" * 64
            suite_yaml = _make_full_suite_yaml(
                repo, frozen_ids, inst_hash=wrong_hash
            )
            adapter_yaml = _make_adapter_yaml(repo, "swe-frozen-test")
            run_dir = _build_swebench_run_with_frozen(
                repo, frozen_ids, model_slug="swe-frozen-test"
            )

            result = validate_campaign.validate(run_dir, suite_yaml, adapter_yaml)
            self.assertFalse(
                result.passed,
                "Validation must fail when suite instances_sha256 does not match "
                "the actual SHA-256 of instance_set_file. Currently passes = gap. "
                f"errors: {result.errors}"
            )
            hash_errors = [
                e for e in result.errors
                if "sha256" in e.lower() or "hash" in e.lower() or "instance" in e.lower()
            ]
            self.assertTrue(
                len(hash_errors) > 0,
                f"Must report instance file hash error; got: {result.errors}"
            )

    def test_instance_file_missing_fails(self):
        """Suite instance_set_file path that doesn't exist must fail.

        RED: validator doesn't load the instance file at all.
        GREEN: must detect and report missing instance_set_file.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            frozen_ids = ["django__django-001", "django__django-002"]

            suite_yaml = _make_full_suite_yaml(repo, frozen_ids)
            # Delete the instance file after creating suite yaml
            inst_file = repo / "suite" / "swebench" / "instances.json"
            inst_file.unlink()

            adapter_yaml = _make_adapter_yaml(repo, "swe-frozen-test")
            run_dir = _build_swebench_run_with_frozen(
                repo, frozen_ids, model_slug="swe-frozen-test"
            )

            result = validate_campaign.validate(run_dir, suite_yaml, adapter_yaml)
            self.assertFalse(
                result.passed,
                "Validation must fail when instance_set_file is missing. "
                f"errors: {result.errors}"
            )

    def test_instance_file_correct_hash_passes(self):
        """Regression: correct instance hash must pass (GREEN regression guard)."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            frozen_ids = ["django__django-001", "django__django-002", "django__django-003"]
            suite_yaml = _make_full_suite_yaml(repo, frozen_ids)  # uses real hash
            adapter_yaml = _make_adapter_yaml(repo, "swe-frozen-test")
            run_dir = _build_swebench_run_with_frozen(
                repo, frozen_ids, model_slug="swe-frozen-test"
            )
            result = validate_campaign.validate(run_dir, suite_yaml, adapter_yaml)
            self.assertTrue(
                result.passed,
                f"Valid run with correct instance hash must pass; errors: {result.errors}"
            )


class TestT2T3T4FrozenIDAuthorityForSWE(unittest.TestCase):
    """T2/T3/T4: Suite frozen ID set is the authority for preds/statuses/grading.

    Current validator uses preds.json keys as the expected ID set.
    Required: load frozen IDs from suite instance_set_file; preds.json,
    exit_statuses.json must have EXACT key equality to frozen IDs.
    """

    def test_preds_with_foreign_ids_fails(self):
        """preds.json containing IDs not in the frozen suite set must fail.

        RED: validator uses preds.json keys as authority — foreign IDs pass unchecked.
        GREEN: must reject preds.json keys not in suite frozen set.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            frozen_ids = ["django__django-001", "django__django-002", "django__django-003"]
            foreign_ids = ["foreign-001", "foreign-002", "foreign-003"]

            suite_yaml = _make_full_suite_yaml(repo, frozen_ids)
            adapter_yaml = _make_adapter_yaml(repo, "swe-frozen-test")
            run_dir = _build_swebench_run_with_frozen(
                repo, frozen_ids,
                preds_ids=foreign_ids,     # wrong IDs
                statuses_ids=foreign_ids,
                traj_ids=foreign_ids,
                model_slug="swe-frozen-test",
            )

            result = validate_campaign.validate(run_dir, suite_yaml, adapter_yaml)
            self.assertFalse(
                result.passed,
                "preds.json with foreign IDs (not in suite frozen set) must fail. "
                "Currently passes because validator uses preds keys as authority = gap. "
                f"errors: {result.errors}"
            )
            id_errors = [
                e for e in result.errors
                if "foreign" in e.lower() or "frozen" in e.lower()
                or "preds" in e.lower() or "instance" in e.lower()
            ]
            self.assertTrue(
                len(id_errors) > 0,
                f"Must report foreign/frozen ID error for preds; got: {result.errors}"
            )

    def test_preds_missing_frozen_id_fails(self):
        """preds.json missing an ID from the frozen suite set must fail.

        RED: validator doesn't cross-check preds keys against frozen set.
        GREEN: every frozen ID must appear in preds.json.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            frozen_ids = ["django__django-001", "django__django-002", "django__django-003"]
            partial_preds = frozen_ids[:2]  # missing frozen_ids[2]

            suite_yaml = _make_full_suite_yaml(repo, frozen_ids)
            adapter_yaml = _make_adapter_yaml(repo, "swe-frozen-test")
            run_dir = _build_swebench_run_with_frozen(
                repo, frozen_ids,
                preds_ids=partial_preds,
                statuses_ids=partial_preds,
                traj_ids=frozen_ids,  # trajectories exist for all
                model_slug="swe-frozen-test",
                extra_manifest={
                    "item_inventory": {
                        "expected": len(frozen_ids),
                        "submitted": len(partial_preds),
                        "instance_ids_hash": hashlib.sha256(
                            json.dumps(sorted(frozen_ids), sort_keys=True).encode()
                        ).hexdigest(),
                    }
                },
            )

            result = validate_campaign.validate(run_dir, suite_yaml, adapter_yaml)
            self.assertFalse(
                result.passed,
                "preds.json missing a frozen ID must fail validation. "
                f"errors: {result.errors}"
            )

    def test_exit_statuses_with_foreign_ids_fails(self):
        """exit_statuses.json containing IDs not in frozen set must fail.

        RED: exit_statuses.json is not cross-checked against frozen IDs.
        GREEN: must reject exit_statuses.json keys not in suite frozen set.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            frozen_ids = ["django__django-001", "django__django-002", "django__django-003"]
            foreign_ids = ["foreign-001", "foreign-002", "foreign-003"]

            suite_yaml = _make_full_suite_yaml(repo, frozen_ids)
            adapter_yaml = _make_adapter_yaml(repo, "swe-frozen-test")
            run_dir = _build_swebench_run_with_frozen(
                repo, frozen_ids,
                preds_ids=frozen_ids,       # preds correct
                statuses_ids=foreign_ids,   # statuses wrong
                traj_ids=frozen_ids,
                model_slug="swe-frozen-test",
            )

            result = validate_campaign.validate(run_dir, suite_yaml, adapter_yaml)
            self.assertFalse(
                result.passed,
                "exit_statuses.json with foreign IDs must fail. "
                "Currently not cross-checked against frozen set = gap. "
                f"errors: {result.errors}"
            )

    def test_preds_malformed_shape_fails(self):
        """preds.json that is not a dict (e.g. a list) must fail validation.

        RED: validator doesn't check shape of preds.json.
        GREEN: preds.json must be a dict; non-dict must be rejected.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            frozen_ids = ["django__django-001", "django__django-002"]

            suite_yaml = _make_full_suite_yaml(repo, frozen_ids)
            adapter_yaml = _make_adapter_yaml(repo, "swe-frozen-test")
            run_dir = _build_swebench_run_with_frozen(
                repo, frozen_ids, model_slug="swe-frozen-test"
            )
            # Overwrite preds.json with a list (malformed shape)
            (run_dir / "raw" / "preds.json").write_text(json.dumps(list(frozen_ids)))

            result = validate_campaign.validate(run_dir, suite_yaml, adapter_yaml)
            self.assertFalse(
                result.passed,
                "preds.json that is a list (not a dict) must fail. "
                f"errors: {result.errors}"
            )

    def test_exit_statuses_malformed_shape_fails(self):
        """exit_statuses.json that is not a dict must fail validation.

        RED: validator doesn't check shape of exit_statuses.json.
        GREEN: exit_statuses.json must be a dict.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            frozen_ids = ["django__django-001", "django__django-002"]

            suite_yaml = _make_full_suite_yaml(repo, frozen_ids)
            adapter_yaml = _make_adapter_yaml(repo, "swe-frozen-test")
            run_dir = _build_swebench_run_with_frozen(
                repo, frozen_ids, model_slug="swe-frozen-test"
            )
            # Overwrite exit_statuses.json with a list
            (run_dir / "raw" / "exit_statuses.json").write_text(
                json.dumps(list(frozen_ids))
            )

            result = validate_campaign.validate(run_dir, suite_yaml, adapter_yaml)
            self.assertFalse(
                result.passed,
                "exit_statuses.json that is a list (not a dict) must fail. "
                f"errors: {result.errors}"
            )

    def test_correct_frozen_ids_passes(self):
        """Regression: run with correct frozen IDs in preds/statuses must pass."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            frozen_ids = ["django__django-001", "django__django-002", "django__django-003"]
            suite_yaml = _make_full_suite_yaml(repo, frozen_ids)
            adapter_yaml = _make_adapter_yaml(repo, "swe-frozen-test")
            run_dir = _build_swebench_run_with_frozen(
                repo, frozen_ids, model_slug="swe-frozen-test"
            )
            result = validate_campaign.validate(run_dir, suite_yaml, adapter_yaml)
            self.assertTrue(
                result.passed,
                f"Valid run with correct frozen IDs must pass; errors: {result.errors}"
            )


class TestT5TrajectoryPerFrozenID(unittest.TestCase):
    """T5: A trajectory file must exist for every frozen ID.

    Current: validator only checks trajectories/ directory exists.
    Required: every frozen ID must have a .traj file in trajectories/.
    """

    def test_missing_trajectory_for_frozen_id_fails(self):
        """Missing trajectory file for a frozen ID must fail validation.

        RED: validator only checks traj dir exists, not per-ID coverage.
        GREEN: every frozen ID must have raw/trajectories/<id>.traj.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            frozen_ids = ["django__django-001", "django__django-002", "django__django-003"]
            # Provide trajectories for only the first 2
            suite_yaml = _make_full_suite_yaml(repo, frozen_ids)
            adapter_yaml = _make_adapter_yaml(repo, "swe-frozen-test")
            run_dir = _build_swebench_run_with_frozen(
                repo, frozen_ids,
                traj_ids=frozen_ids[:2],  # missing frozen_ids[2]
                model_slug="swe-frozen-test",
            )

            result = validate_campaign.validate(run_dir, suite_yaml, adapter_yaml)
            self.assertFalse(
                result.passed,
                "Missing trajectory for a frozen ID must fail. "
                "Current validator only checks traj dir exists = gap. "
                f"errors: {result.errors}"
            )
            traj_errors = [
                e for e in result.errors
                if "traj" in e.lower() or "trajectory" in e.lower()
            ]
            self.assertTrue(
                len(traj_errors) > 0,
                f"Must report trajectory coverage error; got: {result.errors}"
            )

    def test_all_trajectories_present_passes(self):
        """Regression: all trajectory files present must pass."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            frozen_ids = ["django__django-001", "django__django-002", "django__django-003"]
            suite_yaml = _make_full_suite_yaml(repo, frozen_ids)
            adapter_yaml = _make_adapter_yaml(repo, "swe-frozen-test")
            run_dir = _build_swebench_run_with_frozen(
                repo, frozen_ids,
                traj_ids=frozen_ids,  # all present
                model_slug="swe-frozen-test",
            )
            result = validate_campaign.validate(run_dir, suite_yaml, adapter_yaml)
            self.assertTrue(
                result.passed,
                f"Run with all trajectories must pass; errors: {result.errors}"
            )


class TestT6ManifestInstanceIdsHashAgainstFrozen(unittest.TestCase):
    """T6: manifest.item_inventory.instance_ids_hash must match suite-frozen set digest.

    Current: no cross-check of instance_ids_hash against the frozen suite set.
    The instance_ids_hash must equal SHA-256(json.dumps(sorted(frozen_ids))).
    """

    def test_instance_ids_hash_wrong_for_frozen_set_fails(self):
        """manifest.item_inventory.instance_ids_hash not matching frozen set must fail.

        RED: validator doesn't verify instance_ids_hash against suite frozen IDs.
        GREEN: must compute expected hash from frozen IDs and compare.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            frozen_ids = ["django__django-001", "django__django-002", "django__django-003"]
            foreign_ids = ["foreign-001", "foreign-002", "foreign-003"]

            suite_yaml = _make_full_suite_yaml(repo, frozen_ids)
            adapter_yaml = _make_adapter_yaml(repo, "swe-frozen-test")

            # Build run with instance_ids_hash matching foreign_ids (not frozen)
            foreign_hash = hashlib.sha256(
                json.dumps(sorted(foreign_ids), sort_keys=True).encode()
            ).hexdigest()
            run_dir = _build_swebench_run_with_frozen(
                repo, frozen_ids,
                model_slug="swe-frozen-test",
                extra_manifest={
                    "item_inventory": {
                        "expected": len(frozen_ids),
                        "submitted": len(frozen_ids),
                        "instance_ids_hash": foreign_hash,  # wrong hash
                    }
                },
            )

            result = validate_campaign.validate(run_dir, suite_yaml, adapter_yaml)
            self.assertFalse(
                result.passed,
                "manifest.item_inventory.instance_ids_hash not matching suite frozen set "
                "must fail. Currently no cross-check = gap. "
                f"errors: {result.errors}"
            )
            hash_errors = [
                e for e in result.errors
                if "hash" in e.lower() or "instance_ids" in e.lower()
                or "frozen" in e.lower()
            ]
            self.assertTrue(
                len(hash_errors) > 0,
                f"Must report instance_ids_hash mismatch error; got: {result.errors}"
            )

    def test_correct_instance_ids_hash_passes(self):
        """Regression: correct instance_ids_hash matching frozen set must pass."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            frozen_ids = ["django__django-001", "django__django-002", "django__django-003"]
            suite_yaml = _make_full_suite_yaml(repo, frozen_ids)
            adapter_yaml = _make_adapter_yaml(repo, "swe-frozen-test")
            run_dir = _build_swebench_run_with_frozen(
                repo, frozen_ids, model_slug="swe-frozen-test"
            )
            result = validate_campaign.validate(run_dir, suite_yaml, adapter_yaml)
            self.assertTrue(
                result.passed,
                f"Run with correct instance_ids_hash must pass; errors: {result.errors}"
            )


class TestT7ManifestExpectedMatchesFrozenCount(unittest.TestCase):
    """T7: manifest.item_inventory.expected must equal suite expected_item_count.

    Current: no cross-check of manifest expected count against suite yaml.
    """

    def test_manifest_expected_not_matching_suite_count_fails(self):
        """manifest.item_inventory.expected != suite expected_item_count must fail.

        RED: validator doesn't cross-check expected against suite yaml.
        GREEN: must reject manifest.item_inventory.expected != suite.expected_item_count.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            frozen_ids = ["django__django-001", "django__django-002", "django__django-003"]

            suite_yaml = _make_full_suite_yaml(repo, frozen_ids)
            adapter_yaml = _make_adapter_yaml(repo, "swe-frozen-test")
            # Build run with wrong expected count (2, not 3)
            run_dir = _build_swebench_run_with_frozen(
                repo, frozen_ids,
                model_slug="swe-frozen-test",
                extra_manifest={
                    "item_inventory": {
                        "expected": 99,  # wrong — suite says 3
                        "submitted": len(frozen_ids),
                        "instance_ids_hash": hashlib.sha256(
                            json.dumps(sorted(frozen_ids), sort_keys=True).encode()
                        ).hexdigest(),
                    }
                },
            )

            result = validate_campaign.validate(run_dir, suite_yaml, adapter_yaml)
            self.assertFalse(
                result.passed,
                "manifest.item_inventory.expected=99 when suite expected_item_count=3 "
                "must fail. Currently no cross-check = gap. "
                f"errors: {result.errors}"
            )
            count_errors = [
                e for e in result.errors
                if "expected" in e.lower() or "count" in e.lower()
                or "item_count" in e.lower()
            ]
            self.assertTrue(
                len(count_errors) > 0,
                f"Must report expected count mismatch; got: {result.errors}"
            )


class TestT8AdapterPathContainment(unittest.TestCase):
    """T8: Adapter path symlink escape must be rejected in _verify_hashes.

    Current _verify_hashes verifies adapter hash but doesn't check
    that the resolved adapter path is inside the repo root.
    """

    def test_adapter_symlink_outside_repo_blocked(self):
        """A symlink at adapter path pointing outside repo must fail.

        RED: _verify_hashes resolves adapter_path directly; no containment check.
        GREEN: must resolve adapter path and verify it stays inside repo.

        Note: This test validates that the adapter_path passed to validate()
        is contained within the repo root when resolved.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            outside_dir = pathlib.Path(tmp) / "outside"
            outside_dir.mkdir()

            frozen_ids = ["django__django-001", "django__django-002"]
            suite_yaml = _make_full_suite_yaml(repo, frozen_ids)

            # Create real adapter outside repo
            real_adapter = outside_dir / "real-adapter.yaml"
            real_adapter.write_text(
                "adapter_schema_version: 1\ncampaign_status: canonical\n"
            )
            # Compute hash from the real adapter
            real_hash = hashlib.sha256(real_adapter.read_bytes()).hexdigest()

            # Create symlink inside repo pointing outside
            adapter_dir = repo / "adapters"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            symlinked_adapter = adapter_dir / "symlinked.yaml"
            try:
                symlinked_adapter.symlink_to(real_adapter)
            except (OSError, NotImplementedError):
                self.skipTest("Symlinks not supported on this filesystem")

            # Build run using hash of the real (outside) file
            run_dir = _build_swebench_run_with_frozen(
                repo, frozen_ids, model_slug="sym-test"
            )
            # Patch manifest adapter_hash to match real file
            manifest = json.loads((run_dir / "manifest.json").read_text())
            manifest["adapter_hash"] = real_hash
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            # Pass the symlinked adapter as adapter_path
            result = validate_campaign.validate(run_dir, suite_yaml, symlinked_adapter)
            # Must not pass: symlink escape outside repo must be blocked
            self.assertFalse(
                result.passed,
                "Adapter path that resolves via symlink outside repo must be blocked. "
                f"errors: {result.errors}"
            )


class TestT9T10SuiteDrivenRequiredEvidence(unittest.TestCase):
    """T9/T10: Suite yaml required_evidence drives mandatory file checks.

    T9: If suite required_evidence lists an evidence name whose corresponding
        file is missing, validation must fail regardless of manifest booleans.
    T10: An unknown required_evidence name in suite YAML must be rejected
         (fail-closed: unrecognized evidence = configuration error).
    """

    def test_missing_required_evidence_from_suite_fails(self):
        """Suite required_evidence file missing from run must fail.

        Build a suite yaml that requires 'grading_results_json' but the file
        is absent. The manifest may claim grading_results_json=False, but the
        suite says it's required — suite is authoritative.

        RED: validator doesn't parse suite required_evidence; manifest booleans govern.
        GREEN: suite required_evidence entries must be checked for each run.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            import yaml  # type: ignore[import]
            frozen_ids = ["django__django-001", "django__django-002"]

            # Create suite yaml that requires grading_results_json
            instances_file = repo / "suite" / "swebench" / "instances.json"
            instances_file.parent.mkdir(parents=True, exist_ok=True)
            instances_file.write_text(json.dumps(frozen_ids))
            inst_hash = hashlib.sha256(instances_file.read_bytes()).hexdigest()

            suite_data = {
                "suite_id": "warpcore-v1",
                "suite_schema_version": 1,
                "benchmarks": {
                    "swebench": {
                        "description": "test",
                        "instance_set_file": "suite/swebench/instances.json",
                        "instances_sha256": inst_hash,
                        "expected_item_count": len(frozen_ids),
                        # grading_results_json is REQUIRED by the suite
                        "required_evidence": [
                            "preds_json", "exit_statuses", "run_log",
                            "command_txt", "manifest_json", "status_json",
                            "done_sentinel", "grading_results_json",
                        ],
                    }
                },
            }
            suite_yaml = repo / "suite" / "warpcore-v1.yaml"
            suite_yaml.parent.mkdir(parents=True, exist_ok=True)
            suite_yaml.write_text(yaml.dump(suite_data, default_flow_style=False))

            adapter_yaml = _make_adapter_yaml(repo, "swe-evidence-test")
            # Build run WITHOUT grading file, manifest claims grading=False
            run_dir = _build_swebench_run_with_frozen(
                repo, frozen_ids,
                include_grading=False,  # no grading file
                model_slug="swe-evidence-test",
            )

            result = validate_campaign.validate(run_dir, suite_yaml, adapter_yaml)
            self.assertFalse(
                result.passed,
                "Run missing grading_results_json (required by suite) must fail, "
                "even though manifest.artifact_inventory.grading_results_json=False. "
                "Suite required_evidence is authoritative over manifest boolean. "
                "Currently the validator doesn't parse suite required_evidence = gap. "
                f"errors: {result.errors}"
            )
            evidence_errors = [
                e for e in result.errors
                if "grading" in e.lower() or "required" in e.lower()
                or "evidence" in e.lower()
            ]
            self.assertTrue(
                len(evidence_errors) > 0,
                f"Must report missing required_evidence error; got: {result.errors}"
            )

    def test_unknown_required_evidence_name_rejected(self):
        """An unrecognized evidence name in suite required_evidence must be rejected.

        RED: validator doesn't parse suite required_evidence at all.
        GREEN: unknown evidence name must fail (fail-closed: configuration error).
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            import yaml  # type: ignore[import]
            frozen_ids = ["django__django-001"]

            instances_file = repo / "suite" / "swebench" / "instances.json"
            instances_file.parent.mkdir(parents=True, exist_ok=True)
            instances_file.write_text(json.dumps(frozen_ids))
            inst_hash = hashlib.sha256(instances_file.read_bytes()).hexdigest()

            suite_data = {
                "suite_id": "warpcore-v1",
                "suite_schema_version": 1,
                "benchmarks": {
                    "swebench": {
                        "description": "test",
                        "instance_set_file": "suite/swebench/instances.json",
                        "instances_sha256": inst_hash,
                        "expected_item_count": 1,
                        "required_evidence": [
                            "preds_json", "totally_unknown_evidence_name_xyz",
                        ],
                    }
                },
            }
            suite_yaml = repo / "suite" / "warpcore-v1.yaml"
            suite_yaml.parent.mkdir(parents=True, exist_ok=True)
            suite_yaml.write_text(yaml.dump(suite_data, default_flow_style=False))

            adapter_yaml = _make_adapter_yaml(repo, "swe-unknown-ev-test")
            run_dir = _build_swebench_run_with_frozen(
                repo, frozen_ids, model_slug="swe-unknown-ev-test"
            )

            result = validate_campaign.validate(run_dir, suite_yaml, adapter_yaml)
            self.assertFalse(
                result.passed,
                "Unknown required_evidence name in suite YAML must fail (fail-closed). "
                "Currently no required_evidence parsing = gap. "
                f"errors: {result.errors}"
            )
            unknown_errors = [
                e for e in result.errors
                if "unknown" in e.lower() or "evidence" in e.lower()
                or "unrecognized" in e.lower()
            ]
            self.assertTrue(
                len(unknown_errors) > 0,
                f"Must report unknown evidence name error; got: {result.errors}"
            )

    def test_grading_within_category_duplicates_fail(self):
        """IDs duplicated WITHIN a single grading category must fail.

        RED: current validator only checks cross-category duplicates.
        GREEN: within-category duplicates must also be detected.

        This supplements the existing A3 test which tests cross-category.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            frozen_ids = ["django__django-001", "django__django-002", "django__django-003"]
            suite_yaml = _make_full_suite_yaml(repo, frozen_ids)
            adapter_yaml = _make_adapter_yaml(repo, "swe-dup-test")

            # Within-category duplicate: django-001 appears twice in unresolved_ids
            grading = {
                "resolved_ids": [],
                "unresolved_ids": [
                    frozen_ids[0], frozen_ids[1], frozen_ids[2],
                    frozen_ids[0],  # duplicate within same category
                ],
                "empty_patch_ids": [],
                "error_ids": [],
                "incomplete_ids": [],
            }
            run_dir = _build_swebench_run_with_frozen(
                repo, frozen_ids,
                include_grading=True,
                grading_override=grading,
                model_slug="swe-dup-test",
            )

            result = validate_campaign.validate(run_dir, suite_yaml, adapter_yaml)
            self.assertFalse(
                result.passed,
                "Within-category duplicate in grading_results must fail. "
                f"errors: {result.errors}"
            )
            dup_errors = [
                e for e in result.errors
                if "duplic" in e.lower() or "unique" in e.lower()
            ]
            self.assertTrue(
                len(dup_errors) > 0,
                f"Must report within-category duplicate; got: {result.errors}"
            )

    def test_grading_uses_frozen_ids_not_preds_as_authority(self):
        """grading_results validation must use frozen set, not preds, as authority.

        Build a run where preds.json has correct frozen IDs but grading categories
        use foreign IDs. Since current validator derives expected set from preds,
        a foreign grading that has same set as preds would be a gap.

        RED: current validator allows grading to reference any ID in preds.
        GREEN: must cross-check grading IDs against suite frozen set.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            frozen_ids = ["django__django-001", "django__django-002", "django__django-003"]
            # All foreign — not in frozen set
            foreign_grading_ids = ["foreign-001", "foreign-002", "foreign-003"]

            suite_yaml = _make_full_suite_yaml(repo, frozen_ids)
            adapter_yaml = _make_adapter_yaml(repo, "swe-grading-auth-test")

            grading = {
                "resolved_ids": [],
                "unresolved_ids": foreign_grading_ids,  # foreign IDs
                "empty_patch_ids": [],
                "error_ids": [],
                "incomplete_ids": [],
            }
            # preds uses foreign IDs to match grading (so preds-based check wouldn't catch it)
            run_dir = _build_swebench_run_with_frozen(
                repo, frozen_ids,
                preds_ids=foreign_grading_ids,  # preds also wrong
                statuses_ids=foreign_grading_ids,
                traj_ids=foreign_grading_ids,
                include_grading=True,
                grading_override=grading,
                model_slug="swe-grading-auth-test",
            )

            result = validate_campaign.validate(run_dir, suite_yaml, adapter_yaml)
            self.assertFalse(
                result.passed,
                "Grading (and preds) using foreign IDs instead of frozen set must fail. "
                "Suite frozen set is the authority, not preds. "
                f"errors: {result.errors}"
            )


if __name__ == "__main__":
    unittest.main()
