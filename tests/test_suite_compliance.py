"""
Task 1 spec-compliance gap tests.

These tests must ALL fail (RED) before the suite/schema fixes are applied,
and ALL pass (GREEN) after.  They cover the 10+1 numbered corrections.

Run:  /usr/bin/python3 -m pytest tests/test_suite_compliance.py -v
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import jsonschema
import pytest
import yaml

from schema_helpers import (
    SCHEMA_ADAPTER,
    SCHEMA_MANIFEST,
    SCHEMA_STATUS,
    SCHEMA_SUITE,
    VALID_MANIFEST,
    validate_doc,
)

REPO = Path(__file__).parent.parent
SUITE_DIR = REPO / "suite"
SUITE_FILE = SUITE_DIR / "warpcore-v1.yaml"
SCHEMAS_DIR = SUITE_DIR / "schemas"
TASKS_DIR = SUITE_DIR / "tasks"
SWEBENCH_DIR = SUITE_DIR / "swebench"

IFEVAL_TASK_FILE = TASKS_DIR / "ifeval_v4.yaml"
INSTANCES_FILE = SWEBENCH_DIR / "instances-seed42-n100.json"


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Known exact revisions from spec
DATASET_REVISIONS = {
    "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "google/IFEval": "966cd89545d6b6acfd7638bc708b98261ca58e84",
    "Idavidrein/gpqa": "633f5ee89ab8ad4522a9f850766b73f62147ffdd",
    "SWE-Bench_Verified": "c104f840cc67f8b6eec6f759ebc8b2693d585d4a",
}

# Known exact harness pins
HARNESS_LM_EVAL_VERSION = "0.4.12"
HARNESS_LM_EVAL_REVISION = "6d642546f4688648fced259eb3302efd36ece5af"
HARNESS_MINI_SWE_AGENT_VERSION = "2.4.6"
HARNESS_SWEBENCH_VERSION = "4.1.0"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_suite() -> dict:
    return yaml.safe_load(SUITE_FILE.read_text())


def load_schema(p: Path) -> dict:
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# Fix #1: Dataset identifiers AND immutable revisions
# ---------------------------------------------------------------------------

class TestDatasetRevisions:
    """Suite §6: dataset identifiers AND immutable revisions for all datasets."""

    def test_gsm8k_has_revision(self):
        suite = load_suite()
        ds = suite["benchmarks"]["gsm8k"]["dataset"]
        assert "revision" in ds, "gsm8k dataset must have a revision"

    def test_gsm8k_revision_correct(self):
        suite = load_suite()
        ds = suite["benchmarks"]["gsm8k"]["dataset"]
        assert ds["revision"] == DATASET_REVISIONS["openai/gsm8k"], (
            f"gsm8k dataset revision mismatch: {ds.get('revision')!r}"
        )

    def test_ifeval_dataset_path_is_google(self):
        suite = load_suite()
        ds = suite["benchmarks"]["ifeval"]["dataset"]
        assert ds["path"] == "google/IFEval", (
            f"IFEval dataset path must be google/IFEval, got: {ds.get('path')!r}"
        )

    def test_ifeval_has_revision(self):
        suite = load_suite()
        ds = suite["benchmarks"]["ifeval"]["dataset"]
        assert "revision" in ds, "ifeval dataset must have a revision"

    def test_ifeval_revision_correct(self):
        suite = load_suite()
        ds = suite["benchmarks"]["ifeval"]["dataset"]
        assert ds["revision"] == DATASET_REVISIONS["google/IFEval"], (
            f"IFEval dataset revision mismatch: {ds.get('revision')!r}"
        )

    def test_gpqa_has_revision(self):
        suite = load_suite()
        ds = suite["benchmarks"]["gpqa_diamond"]["dataset"]
        assert "revision" in ds, "gpqa_diamond dataset must have a revision"

    def test_gpqa_revision_correct(self):
        suite = load_suite()
        ds = suite["benchmarks"]["gpqa_diamond"]["dataset"]
        assert ds["revision"] == DATASET_REVISIONS["Idavidrein/gpqa"], (
            f"gpqa_diamond dataset revision mismatch: {ds.get('revision')!r}"
        )

    def test_swebench_has_revision(self):
        """SWE-bench dataset declaration must have a revision."""
        suite = load_suite()
        swe = suite["benchmarks"]["swebench"]
        # dataset is declared as a dict with revision
        assert isinstance(swe.get("dataset"), dict), "swebench dataset must be a dict with revision"
        assert "revision" in swe["dataset"], "swebench dataset must have a revision"

    def test_swebench_revision_correct(self):
        suite = load_suite()
        swe = suite["benchmarks"]["swebench"]
        assert swe["dataset"]["revision"] == DATASET_REVISIONS["SWE-Bench_Verified"], (
            f"SWE-bench dataset revision mismatch: {swe['dataset'].get('revision')!r}"
        )

    def test_suite_schema_dataset_requires_revision(self):
        """The dataset_ref schema must require 'revision'."""
        schema = load_schema(SCHEMA_SUITE)
        defs = schema.get("$defs", {})
        dataset_ref = defs.get("dataset_ref", {})
        required = dataset_ref.get("required", [])
        assert "revision" in required, (
            "dataset_ref schema must require 'revision'"
        )


# ---------------------------------------------------------------------------
# Fix #2: IFEval expected_item_count + quality_benchmark schema requires it
# ---------------------------------------------------------------------------

class TestIFEvalItemCount:
    def test_ifeval_expected_item_count(self):
        suite = load_suite()
        count = suite["benchmarks"]["ifeval"].get("expected_item_count")
        assert count == 541, f"ifeval expected_item_count must be 541, got {count!r}"

    def test_quality_benchmark_schema_requires_expected_item_count(self):
        """quality_benchmark schema must require expected_item_count."""
        schema = load_schema(SCHEMA_SUITE)
        qb = schema["$defs"]["quality_benchmark"]
        assert "expected_item_count" in qb.get("required", []), (
            "quality_benchmark schema must require expected_item_count"
        )


# ---------------------------------------------------------------------------
# Fix #3: Immutable harness version pins
# ---------------------------------------------------------------------------

class TestHarnessPins:
    def test_lm_eval_exact_version(self):
        suite = load_suite()
        harness = suite["required_harness"]
        assert harness.get("lm_eval_version") == HARNESS_LM_EVAL_VERSION, (
            f"required_harness.lm_eval_version must be {HARNESS_LM_EVAL_VERSION!r}, "
            f"got {harness.get('lm_eval_version')!r}"
        )

    def test_lm_eval_source_revision(self):
        suite = load_suite()
        harness = suite["required_harness"]
        assert harness.get("lm_eval_revision") == HARNESS_LM_EVAL_REVISION, (
            f"required_harness.lm_eval_revision must be exact commit SHA"
        )

    def test_mini_swe_agent_version(self):
        suite = load_suite()
        harness = suite["required_harness"]
        assert harness.get("mini_swe_agent_version") == HARNESS_MINI_SWE_AGENT_VERSION, (
            f"required_harness.mini_swe_agent_version must be {HARNESS_MINI_SWE_AGENT_VERSION!r}"
        )

    def test_swebench_grading_harness_version(self):
        suite = load_suite()
        harness = suite["required_harness"]
        assert harness.get("swebench_version") == HARNESS_SWEBENCH_VERSION, (
            f"required_harness.swebench_version must be {HARNESS_SWEBENCH_VERSION!r}"
        )

    def test_schema_required_harness_requires_pinned_fields(self):
        """Schema must require all four harness pin fields."""
        schema = load_schema(SCHEMA_SUITE)
        rh = schema["properties"]["required_harness"]
        required = rh.get("required", [])
        for field in ("lm_eval_version", "lm_eval_revision", "mini_swe_agent_version", "swebench_version"):
            assert field in required, f"required_harness schema must require '{field}'"


# ---------------------------------------------------------------------------
# Fix #4: ifeval_v4.yaml pinned task file
# ---------------------------------------------------------------------------

class TestIFEvalTaskFile:
    def test_ifeval_task_file_present(self):
        assert IFEVAL_TASK_FILE.exists(), f"Missing: {IFEVAL_TASK_FILE}"

    def test_ifeval_task_file_declares_google_ifeval(self):
        """The pinned task file must reference google/IFEval."""
        content = IFEVAL_TASK_FILE.read_text()
        assert "google/IFEval" in content, (
            "ifeval_v4.yaml must reference dataset_path: google/IFEval"
        )

    def test_ifeval_task_file_task_name_is_ifeval(self):
        """task: field must be 'ifeval' to preserve execution task name.

        The lm-eval task YAML uses !function tags which yaml.safe_load cannot
        parse.  We extract the task: line from raw text instead.
        """
        content = IFEVAL_TASK_FILE.read_text()
        # Find the 'task: <name>' line
        task_name = None
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("task:"):
                task_name = stripped[len("task:"):].strip()
                break
        assert task_name == "ifeval", (
            f"ifeval_v4.yaml task must be 'ifeval', got: {task_name!r}"
        )

    def test_suite_declares_ifeval_task_file(self):
        suite = load_suite()
        task_file = suite["benchmarks"]["ifeval"].get("task_file")
        assert task_file == "suite/tasks/ifeval_v4.yaml", (
            f"ifeval benchmark must declare task_file='suite/tasks/ifeval_v4.yaml', got: {task_file!r}"
        )

    def test_suite_declares_ifeval_task_sha256(self):
        suite = load_suite()
        declared = suite["benchmarks"]["ifeval"].get("task_sha256")
        assert declared is not None, "ifeval benchmark must declare task_sha256"
        actual = sha256_bytes(IFEVAL_TASK_FILE.read_bytes())
        assert declared == actual, (
            f"ifeval task_sha256 mismatch: declared={declared!r} actual={actual!r}"
        )

    def test_ifeval_task_file_content_hash_matches_upstream(self):
        """Hash must match the known lm-eval 0.4.12 builtin ifeval.yaml.

        The lm-eval task YAML uses !function tags which yaml.safe_load cannot
        parse.  We check the file has 'metadata:' and 'version: 4.0' in raw text.
        """
        content = IFEVAL_TASK_FILE.read_text()
        assert "metadata:" in content, (
            "ifeval_v4.yaml must contain a metadata: section (lm-eval 0.4.12 builtin)"
        )
        assert "version: 4.0" in content, (
            "ifeval_v4.yaml must have version: 4.0 in metadata (lm-eval 0.4.12 builtin)"
        )


# ---------------------------------------------------------------------------
# Fix #5: No duplicate lifecycle_states / execution_states in suite
# ---------------------------------------------------------------------------

class TestNoSuiteDuplicateStates:
    def test_suite_has_no_lifecycle_states(self):
        suite = load_suite()
        assert "lifecycle_states" not in suite, (
            "Suite YAML must not duplicate lifecycle_states (owned by result-status schema)"
        )

    def test_suite_has_no_execution_states(self):
        suite = load_suite()
        assert "execution_states" not in suite, (
            "Suite YAML must not duplicate execution_states (owned by result-status schema)"
        )

    def test_suite_schema_has_no_lifecycle_states(self):
        schema = load_schema(SCHEMA_SUITE)
        assert "lifecycle_states" not in schema.get("properties", {}), (
            "suite.schema.json must not define lifecycle_states"
        )

    def test_suite_schema_has_no_execution_states(self):
        schema = load_schema(SCHEMA_SUITE)
        assert "execution_states" not in schema.get("properties", {}), (
            "suite.schema.json must not define execution_states"
        )

    def test_suite_schema_required_no_lifecycle_states(self):
        schema = load_schema(SCHEMA_SUITE)
        required = schema.get("required", [])
        assert "lifecycle_states" not in required
        assert "execution_states" not in required


# ---------------------------------------------------------------------------
# Fix #6: serving_profile_digest in manifest
# ---------------------------------------------------------------------------

class TestServingProfileDigest:

    def test_manifest_missing_serving_profile_digest_rejected(self):
        bad = {k: v for k, v in VALID_MANIFEST.items() if k != "serving_profile_digest"}
        with pytest.raises(jsonschema.ValidationError):
            validate_doc(bad, SCHEMA_MANIFEST)

    def test_manifest_with_serving_profile_digest_passes(self):
        validate_doc(VALID_MANIFEST, SCHEMA_MANIFEST)

    def test_serving_profile_digest_must_match_sha256_pattern(self):
        bad = {**VALID_MANIFEST, "serving_profile_digest": "notadigest"}
        with pytest.raises(jsonschema.ValidationError):
            validate_doc(bad, SCHEMA_MANIFEST)

    def test_manifest_serving_requires_hardware_id(self):
        bad = {
            **VALID_MANIFEST,
            "serving": {k: v for k, v in VALID_MANIFEST["serving"].items() if k != "hardware_id"},
        }
        with pytest.raises(jsonschema.ValidationError):
            validate_doc(bad, SCHEMA_MANIFEST)

    def test_manifest_serving_requires_environment(self):
        bad = {
            **VALID_MANIFEST,
            "serving": {k: v for k, v in VALID_MANIFEST["serving"].items() if k != "environment"},
        }
        with pytest.raises(jsonschema.ValidationError):
            validate_doc(bad, SCHEMA_MANIFEST)


# ---------------------------------------------------------------------------
# Fix #7: suite_input_hashes replaces suite_task_hash
# ---------------------------------------------------------------------------

class TestSuiteInputHashes:

    def test_manifest_rejects_suite_task_hash(self):
        """suite_task_hash is abolished; manifest with it must be rejected."""
        bad = {**VALID_MANIFEST, "suite_task_hash": "b" * 64}
        with pytest.raises(jsonschema.ValidationError):
            validate_doc(bad, SCHEMA_MANIFEST)

    def test_manifest_missing_suite_input_hashes_rejected(self):
        bad = {k: v for k, v in VALID_MANIFEST.items() if k != "suite_input_hashes"}
        with pytest.raises(jsonschema.ValidationError):
            validate_doc(bad, SCHEMA_MANIFEST)

    def test_suite_input_hashes_must_be_nonempty_object(self):
        bad = {**VALID_MANIFEST, "suite_input_hashes": {}}
        with pytest.raises(jsonschema.ValidationError):
            validate_doc(bad, SCHEMA_MANIFEST)

    def test_suite_input_hashes_value_must_be_sha256(self):
        """Values must be 64-char hex strings."""
        bad = {**VALID_MANIFEST, "suite_input_hashes": {"suite/tasks/foo.yaml": "notahash"}}
        with pytest.raises(jsonschema.ValidationError):
            validate_doc(bad, SCHEMA_MANIFEST)

    def test_suite_input_hashes_valid_passes(self):
        validate_doc(VALID_MANIFEST, SCHEMA_MANIFEST)


# ---------------------------------------------------------------------------
# Fix #8: date-time format on completed_utc + FormatChecker
# ---------------------------------------------------------------------------

class TestCompletedUtcFormat:

    def test_invalid_completed_utc_rejected_with_format_checker(self):
        bad = {
            **VALID_MANIFEST,
            "timing": {
                "started_utc": "2026-09-15T00:00:00Z",
                "completed_utc": "not-a-timestamp",
            },
        }
        with pytest.raises(jsonschema.ValidationError):
            validate_doc(bad, SCHEMA_MANIFEST)

    def test_null_completed_utc_passes(self):
        doc = {
            **VALID_MANIFEST,
            "timing": {
                "started_utc": "2026-09-15T00:00:00Z",
                "completed_utc": None,
            },
        }
        validate_doc(doc, SCHEMA_MANIFEST)

    def test_valid_completed_utc_passes(self):
        validate_doc(VALID_MANIFEST, SCHEMA_MANIFEST)


# ---------------------------------------------------------------------------
# Fix #9: Quality evidence fields
# ---------------------------------------------------------------------------

class TestQualityEvidenceFields:
    REQUIRED_EVIDENCE_FIELDS = [
        "aggregate_result",
        "samples_jsonl_gz",
        "per_item_csv",
        "task_yaml",
        "scoring_implementation",  # renamed from scoring_utility (Gap H fix: honest class name)
        "run_log",
        "command_txt",
        "manifest_json",
        "status_json",
        "done_sentinel",
        "completion_token_counts",
        "finish_reasons",
        "full_response_fields",
    ]

    def _check_benchmark_evidence(self, benchmark: str):
        suite = load_suite()
        evidence = suite["benchmarks"][benchmark].get("required_evidence", [])
        missing = [f for f in self.REQUIRED_EVIDENCE_FIELDS if f not in evidence]
        assert not missing, (
            f"{benchmark} required_evidence missing fields: {missing}"
        )

    def test_gsm8k_required_evidence(self):
        self._check_benchmark_evidence("gsm8k")

    def test_ifeval_required_evidence(self):
        self._check_benchmark_evidence("ifeval")

    def test_gpqa_diamond_required_evidence(self):
        self._check_benchmark_evidence("gpqa_diamond")


# ---------------------------------------------------------------------------
# Fix #10: All schemas additionalProperties:false (recursive)
# ---------------------------------------------------------------------------

class TestAllSchemasAdditionalPropertiesFalseRecursive:
    """
    Recursive check: every schema node with type:object or properties
    must have additionalProperties:false.
    Also covers oneOf/anyOf sub-schemas.
    """

    def _violations(self, schema: dict, path: str = "$") -> list[str]:
        found = []
        if schema.get("type") == "object" or "properties" in schema:
            if schema.get("additionalProperties") is not False:
                # patternProperties-only objects are allowed without additionalProperties:false
                # only if they have no 'properties' key
                if "properties" in schema or schema.get("type") == "object":
                    found.append(path)
        for key, sub in schema.get("properties", {}).items():
            found.extend(self._violations(sub, f"{path}.{key}"))
        for sub in schema.get("oneOf", []):
            found.extend(self._violations(sub, f"{path}[oneOf]"))
        for sub in schema.get("anyOf", []):
            found.extend(self._violations(sub, f"{path}[anyOf]"))
        for sub in schema.get("allOf", []):
            found.extend(self._violations(sub, f"{path}[allOf]"))
        if "items" in schema and isinstance(schema["items"], dict):
            found.extend(self._violations(schema["items"], f"{path}[items]"))
        for ns in ("definitions", "$defs"):
            for key, sub in schema.get(ns, {}).items():
                found.extend(self._violations(sub, f"{path}.{ns}.{key}"))
        return found

    def _check(self, schema_path: Path):
        schema = load_schema(schema_path)
        violations = self._violations(schema)
        assert not violations, (
            f"{schema_path.name}: object schemas missing additionalProperties:false:\n"
            + "\n".join(f"  {v}" for v in violations)
        )

    def test_suite_schema_recursive(self):
        self._check(SCHEMA_SUITE)

    def test_adapter_schema_recursive(self):
        self._check(SCHEMA_ADAPTER)

    def test_manifest_schema_recursive(self):
        self._check(SCHEMA_MANIFEST)

    def test_result_status_schema_recursive(self):
        self._check(SCHEMA_STATUS)


# ---------------------------------------------------------------------------
# Additional: retry_policy and timeout_policy required for every benchmark
# ---------------------------------------------------------------------------

class TestBenchmarkPoliciesRequired:
    """Suite plan §6: retry and timeout policy for every benchmark."""

    QUALITY_BENCHMARKS = ["gsm8k", "ifeval", "gpqa_diamond"]

    def _check_has_retry(self, benchmark: str):
        suite = load_suite()
        bm = suite["benchmarks"][benchmark]
        assert "retry_policy" in bm, f"{benchmark} must declare retry_policy"
        assert "max_retries" in bm["retry_policy"]

    def _check_has_timeout(self, benchmark: str):
        suite = load_suite()
        bm = suite["benchmarks"][benchmark]
        assert "timeout_policy" in bm, f"{benchmark} must declare timeout_policy"

    def test_gsm8k_retry_policy(self):
        self._check_has_retry("gsm8k")

    def test_gsm8k_timeout_policy(self):
        self._check_has_timeout("gsm8k")

    def test_ifeval_retry_policy(self):
        self._check_has_retry("ifeval")

    def test_ifeval_timeout_policy(self):
        self._check_has_timeout("ifeval")

    def test_gpqa_diamond_retry_policy(self):
        self._check_has_retry("gpqa_diamond")

    def test_gpqa_diamond_timeout_policy(self):
        self._check_has_timeout("gpqa_diamond")

    def test_swebench_retry_policy(self):
        self._check_has_retry("swebench")

    def test_swebench_timeout_policy(self):
        self._check_has_timeout("swebench")

    def test_throughput_retry_policy(self):
        self._check_has_retry("throughput")

    def test_throughput_timeout_policy(self):
        self._check_has_timeout("throughput")

    def test_schema_swebench_requires_retry_policy(self):
        schema = load_schema(SCHEMA_SUITE)
        swe = schema["$defs"]["swebench_benchmark"]
        assert "retry_policy" in swe.get("required", []), (
            "swebench_benchmark schema must require retry_policy"
        )

    def test_schema_swebench_requires_timeout_policy(self):
        schema = load_schema(SCHEMA_SUITE)
        swe = schema["$defs"]["swebench_benchmark"]
        assert "timeout_policy" in swe.get("required", []), (
            "swebench_benchmark schema must require timeout_policy"
        )

    def test_schema_throughput_requires_retry_policy(self):
        schema = load_schema(SCHEMA_SUITE)
        tp = schema["$defs"]["throughput_benchmark"]
        assert "retry_policy" in tp.get("required", []), (
            "throughput_benchmark schema must require retry_policy"
        )

    def test_schema_throughput_requires_timeout_policy(self):
        schema = load_schema(SCHEMA_SUITE)
        tp = schema["$defs"]["throughput_benchmark"]
        assert "timeout_policy" in tp.get("required", []), (
            "throughput_benchmark schema must require timeout_policy"
        )

    def test_schema_quality_requires_retry_policy(self):
        schema = load_schema(SCHEMA_SUITE)
        qb = schema["$defs"]["quality_benchmark"]
        assert "retry_policy" in qb.get("required", []), (
            "quality_benchmark schema must require retry_policy"
        )

    def test_schema_quality_requires_timeout_policy(self):
        schema = load_schema(SCHEMA_SUITE)
        qb = schema["$defs"]["quality_benchmark"]
        assert "timeout_policy" in qb.get("required", []), (
            "quality_benchmark schema must require timeout_policy"
        )
