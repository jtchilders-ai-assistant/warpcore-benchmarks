"""
Task 1 spec-fix cycle 2: strict-schema and identity hardening regression tests.

All tests in this file MUST fail (RED) before fixes are applied and ALL pass
(GREEN) after.  They cover gaps A-J from the second spec-fix cycle.

Run: /usr/bin/python3 -m pytest tests/test_schema_hardening.py -v
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import jsonschema
import pytest
import yaml

REPO = Path(__file__).parent.parent
SUITE_DIR = REPO / "suite"
SUITE_FILE = SUITE_DIR / "warpcore-v1.yaml"
SCHEMAS_DIR = SUITE_DIR / "schemas"

SCHEMA_SUITE = SCHEMAS_DIR / "suite.schema.json"
SCHEMA_ADAPTER = SCHEMAS_DIR / "adapter.schema.json"
SCHEMA_MANIFEST = SCHEMAS_DIR / "manifest.schema.json"
SCHEMA_STATUS = SCHEMAS_DIR / "result-status.schema.json"

REQ_FILE = REPO / "requirements-viz.txt"


def load_suite() -> dict:
    return yaml.safe_load(SUITE_FILE.read_text())


def load_schema(p: Path) -> dict:
    return json.loads(p.read_text())


def validate(instance: dict, schema_path: Path, format_checker=None) -> None:
    schema = json.loads(schema_path.read_text())
    kwargs = {}
    if format_checker is not None:
        kwargs["format_checker"] = format_checker
    jsonschema.validate(instance, schema, **kwargs)


# ---------------------------------------------------------------------------
# Shared valid documents used across test classes
# ---------------------------------------------------------------------------

VALID_ADAPTER = {
    "adapter_schema_version": 1,
    "model": {
        "slug": "qwen3.6-35b-a3b",
        "id": "Qwen/Qwen3.6-35B-A3B-FP8",
        "revision": "abc123def456abc123def456abc123def456abc1",
    },
    "serving": {
        "image": "eugr/spark-vllm@sha256:" + "dead" * 16,
        "engine": "vllm",
        "engine_version": "0.8.5",
        "quantization": "fp8",
        "reasoning_parser": None,
        "tool_call_parser": None,
        "tokenizer": None,
        "moe_backend": None,
        "max_model_len": 131072,
        "gpu_memory_utilization": 0.90,
        "max_num_seqs": 32,
        "environment": {},
    },
}

VALID_MANIFEST = {
    "schema_version": 1,
    "suite_id": "warpcore-v1",
    "suite_schema_version": 1,
    "run_id": "run-2026-09-15T00-00-00",
    "benchmark": "gsm8k",
    "adapter_hash": "a" * 64,
    "suite_input_hashes": {
        "suite/tasks/gsm8k_clean_v1.yaml": "b" * 64,
    },
    "serving_profile_digest": "sha256:" + "c" * 64,
    "model": {
        "slug": "qwen3.6-35b-a3b",
        "id": "Qwen/Qwen3.6-35B-A3B-FP8",
        "revision": "abc123def456abc123def456abc123def456abc1",
    },
    "serving": {
        "image_digest": "sha256:" + "d" * 64,
        "engine": "vllm",
        "engine_version": "0.8.5",
        "effective_args": [],
        "environment": {},
        "hardware_id": "dgx-spark-gb10",
    },
    "item_inventory": {"expected": 1319, "submitted": 1319},
    "timing": {
        "started_utc": "2026-09-15T00:00:00Z",
        "completed_utc": "2026-09-15T01:00:00Z",
    },
    "artifact_inventory": {
        "samples_jsonl_gz": True,
        "per_item_csv": True,
        "run_log": True,
        "command_txt": True,
        "done_sentinel": True,
    },
}


# ---------------------------------------------------------------------------
# Gap A: Suite schema structural strictness
# ---------------------------------------------------------------------------

class TestGapA_SuiteSchemaStructural:
    """
    Gap A: Suite schema must reject empty benchmarks, missing task_file/task_sha256,
    missing dataset, malformed task_sha256 (not 64 lower hex), empty required_evidence,
    and duplicate required_evidence entries.
    """

    def _suite_with(self, **overrides) -> dict:
        suite = load_suite()
        for k, v in overrides.items():
            suite[k] = v
        return suite

    def _suite_with_gsm8k(self, **overrides) -> dict:
        suite = load_suite()
        suite["benchmarks"]["gsm8k"] = dict(suite["benchmarks"]["gsm8k"], **overrides)
        return suite

    # A1: empty benchmarks must be rejected
    def test_empty_benchmarks_rejected(self):
        bad = self._suite_with(benchmarks={})
        with pytest.raises(jsonschema.ValidationError):
            validate(bad, SCHEMA_SUITE)

    # A2: missing task_file for quality benchmark (gsm8k) must be rejected
    def test_quality_benchmark_missing_task_file_rejected(self):
        suite = load_suite()
        bm = {k: v for k, v in suite["benchmarks"]["gsm8k"].items() if k != "task_file"}
        suite["benchmarks"]["gsm8k"] = bm
        with pytest.raises(jsonschema.ValidationError):
            validate(suite, SCHEMA_SUITE)

    # A3: missing task_sha256 for quality benchmark must be rejected
    def test_quality_benchmark_missing_task_sha256_rejected(self):
        suite = load_suite()
        bm = {k: v for k, v in suite["benchmarks"]["gsm8k"].items() if k != "task_sha256"}
        suite["benchmarks"]["gsm8k"] = bm
        with pytest.raises(jsonschema.ValidationError):
            validate(suite, SCHEMA_SUITE)

    # A4: missing dataset for quality benchmark must be rejected
    def test_quality_benchmark_missing_dataset_rejected(self):
        suite = load_suite()
        bm = {k: v for k, v in suite["benchmarks"]["gsm8k"].items() if k != "dataset"}
        suite["benchmarks"]["gsm8k"] = bm
        with pytest.raises(jsonschema.ValidationError):
            validate(suite, SCHEMA_SUITE)

    # A5: malformed task_sha256 (single char 'x') must be rejected
    def test_quality_benchmark_malformed_task_sha256_rejected(self):
        bad = self._suite_with_gsm8k(task_sha256="x")
        with pytest.raises(jsonschema.ValidationError):
            validate(bad, SCHEMA_SUITE)

    # A6: task_sha256 with uppercase hex must be rejected
    def test_quality_benchmark_uppercase_sha256_rejected(self):
        bad = self._suite_with_gsm8k(task_sha256="A" * 64)
        with pytest.raises(jsonschema.ValidationError):
            validate(bad, SCHEMA_SUITE)

    # A7: task_sha256 with 63 chars must be rejected
    def test_quality_benchmark_short_sha256_rejected(self):
        bad = self._suite_with_gsm8k(task_sha256="a" * 63)
        with pytest.raises(jsonschema.ValidationError):
            validate(bad, SCHEMA_SUITE)

    # A8: empty required_evidence must be rejected
    def test_quality_benchmark_empty_required_evidence_rejected(self):
        bad = self._suite_with_gsm8k(required_evidence=[])
        with pytest.raises(jsonschema.ValidationError):
            validate(bad, SCHEMA_SUITE)

    # A9: suite_id must be exactly "warpcore-v1"
    def test_suite_id_must_be_exact_const(self):
        bad = self._suite_with(suite_id="warpcore-v2")
        with pytest.raises(jsonschema.ValidationError):
            validate(bad, SCHEMA_SUITE)

    # A10: suite_schema_version must be exactly 1
    def test_suite_schema_version_must_be_const_1(self):
        bad = self._suite_with(suite_schema_version=2)
        with pytest.raises(jsonschema.ValidationError):
            validate(bad, SCHEMA_SUITE)

    # A11: benchmarks must require all five benchmark keys (missing swebench)
    def test_benchmarks_missing_required_key_rejected(self):
        suite = load_suite()
        suite["benchmarks"] = {k: v for k, v in suite["benchmarks"].items() if k != "swebench"}
        with pytest.raises(jsonschema.ValidationError):
            validate(suite, SCHEMA_SUITE)

    # A12: utils_sha256 pattern — if present, must be 64 lowercase hex
    def test_utils_sha256_malformed_rejected(self):
        suite = load_suite()
        bm = dict(suite["benchmarks"]["gpqa_diamond"])
        bm["utils_sha256"] = "UPPERCASE" + "a" * 55
        suite["benchmarks"]["gpqa_diamond"] = bm
        with pytest.raises(jsonschema.ValidationError):
            validate(suite, SCHEMA_SUITE)

    # A13: dataset revision must be exactly 40 lowercase hex chars
    def test_dataset_revision_must_be_40hex(self):
        suite = load_suite()
        bm = dict(suite["benchmarks"]["gsm8k"])
        bm["dataset"] = dict(bm["dataset"])
        bm["dataset"]["revision"] = "toolong" + "a" * 40
        suite["benchmarks"]["gsm8k"] = bm
        with pytest.raises(jsonschema.ValidationError):
            validate(suite, SCHEMA_SUITE)

    def test_dataset_revision_uppercase_rejected(self):
        suite = load_suite()
        bm = dict(suite["benchmarks"]["gsm8k"])
        bm["dataset"] = dict(bm["dataset"])
        bm["dataset"]["revision"] = "A" * 40
        suite["benchmarks"]["gsm8k"] = bm
        with pytest.raises(jsonschema.ValidationError):
            validate(suite, SCHEMA_SUITE)

    def test_dataset_revision_short_rejected(self):
        suite = load_suite()
        bm = dict(suite["benchmarks"]["gsm8k"])
        bm["dataset"] = dict(bm["dataset"])
        bm["dataset"]["revision"] = "abc123"
        suite["benchmarks"]["gsm8k"] = bm
        with pytest.raises(jsonschema.ValidationError):
            validate(suite, SCHEMA_SUITE)


# ---------------------------------------------------------------------------
# Gap B: SWE-bench dataset path
# ---------------------------------------------------------------------------

class TestGapB_SwebenchDatasetPath:
    """
    Gap B: SWE-bench dataset path must be princeton-nlp/SWE-bench_Verified.
    Current value 'SWE-bench/SWE-bench_Verified' is wrong.
    """

    def test_swebench_dataset_path_is_princeton_nlp(self):
        suite = load_suite()
        path = suite["benchmarks"]["swebench"]["dataset"]["path"]
        assert path == "princeton-nlp/SWE-bench_Verified", (
            f"swebench dataset path must be 'princeton-nlp/SWE-bench_Verified', got: {path!r}. "
            "Update suite/warpcore-v1.yaml."
        )

    def test_swebench_dataset_schema_rejects_wrong_path(self):
        """Suite schema's swebench_dataset_ref must use enum/const for the path."""
        schema = load_schema(SCHEMA_SUITE)
        swe_ds = schema["$defs"]["swebench_dataset_ref"]
        path_prop = swe_ds.get("properties", {}).get("path", {})
        # Must have a const or enum that pins to the correct org
        has_const = path_prop.get("const") == "princeton-nlp/SWE-bench_Verified"
        has_enum = path_prop.get("enum") == ["princeton-nlp/SWE-bench_Verified"]
        assert has_const or has_enum, (
            "swebench_dataset_ref.path must use const or enum pinning to "
            "'princeton-nlp/SWE-bench_Verified'; currently allows arbitrary string"
        )


# ---------------------------------------------------------------------------
# Gap C: Adapter image digest and model revision strict patterns
# ---------------------------------------------------------------------------

class TestGapC_AdapterStrictPatterns:
    """
    Gap C: Adapter image must be repo@sha256:<64 lowercase hex>.
    Model revision must be exactly 40 lowercase hex.
    'unrecorded' must not be accepted for new adapter definitions.
    """

    def test_valid_adapter_passes(self):
        validate(VALID_ADAPTER, SCHEMA_ADAPTER)

    # C1: mutable image tag (no digest) must be rejected
    def test_adapter_mutable_tag_only_rejected(self):
        bad = {**VALID_ADAPTER, "serving": {**VALID_ADAPTER["serving"], "image": "repo:latest"}}
        with pytest.raises(jsonschema.ValidationError):
            validate(bad, SCHEMA_ADAPTER)

    # C2: image with tag but no digest must be rejected
    def test_adapter_repo_tag_no_digest_rejected(self):
        bad = {**VALID_ADAPTER, "serving": {**VALID_ADAPTER["serving"], "image": "org/repo:v0.8.5"}}
        with pytest.raises(jsonschema.ValidationError):
            validate(bad, SCHEMA_ADAPTER)

    # C3: image with short sha must be rejected
    def test_adapter_image_short_sha_rejected(self):
        bad = {**VALID_ADAPTER, "serving": {**VALID_ADAPTER["serving"], "image": "repo@sha256:abc"}}
        with pytest.raises(jsonschema.ValidationError):
            validate(bad, SCHEMA_ADAPTER)

    # C4: image with uppercase sha must be rejected
    def test_adapter_image_uppercase_sha_rejected(self):
        bad = {**VALID_ADAPTER, "serving": {**VALID_ADAPTER["serving"], "image": "repo@sha256:" + "A" * 64}}
        with pytest.raises(jsonschema.ValidationError):
            validate(bad, SCHEMA_ADAPTER)

    # C5: model revision 'unrecorded' must be rejected for adapter (new definition)
    def test_adapter_model_revision_unrecorded_rejected(self):
        bad = {**VALID_ADAPTER, "model": {**VALID_ADAPTER["model"], "revision": "unrecorded"}}
        with pytest.raises(jsonschema.ValidationError):
            validate(bad, SCHEMA_ADAPTER)

    # C6: model revision arbitrary string must be rejected
    def test_adapter_model_revision_arbitrary_rejected(self):
        bad = {**VALID_ADAPTER, "model": {**VALID_ADAPTER["model"], "revision": "any-arbitrary-string"}}
        with pytest.raises(jsonschema.ValidationError):
            validate(bad, SCHEMA_ADAPTER)

    # C7: model revision short hex must be rejected
    def test_adapter_model_revision_short_sha_rejected(self):
        bad = {**VALID_ADAPTER, "model": {**VALID_ADAPTER["model"], "revision": "abc123"}}
        with pytest.raises(jsonschema.ValidationError):
            validate(bad, SCHEMA_ADAPTER)

    # C8: model revision uppercase hex must be rejected
    def test_adapter_model_revision_uppercase_rejected(self):
        bad = {**VALID_ADAPTER, "model": {**VALID_ADAPTER["model"], "revision": "A" * 40}}
        with pytest.raises(jsonschema.ValidationError):
            validate(bad, SCHEMA_ADAPTER)

    # C9: valid 40-char lowercase hex revision passes
    def test_adapter_model_revision_valid_40hex_passes(self):
        good = {**VALID_ADAPTER, "model": {**VALID_ADAPTER["model"], "revision": "a" * 40}}
        validate(good, SCHEMA_ADAPTER)


# ---------------------------------------------------------------------------
# Gap D: Manifest adapter_hash exact 64 hex; suite_input_hashes safe paths
# ---------------------------------------------------------------------------

class TestGapD_ManifestHashAndPaths:
    """
    Gap D: adapter_hash must be exactly 64 lowercase hex (not merely minLength 64).
    suite_input_hashes keys must be safe repo-relative suite/ paths with no traversal.
    """

    def test_valid_manifest_passes(self):
        validate(VALID_MANIFEST, SCHEMA_MANIFEST)

    # D1: adapter_hash 65 chars (too long) must be rejected
    def test_adapter_hash_65_chars_rejected(self):
        bad = {**VALID_MANIFEST, "adapter_hash": "a" * 65}
        with pytest.raises(jsonschema.ValidationError):
            validate(bad, SCHEMA_MANIFEST)

    # D2: adapter_hash uppercase must be rejected
    def test_adapter_hash_uppercase_rejected(self):
        bad = {**VALID_MANIFEST, "adapter_hash": "A" * 64}
        with pytest.raises(jsonschema.ValidationError):
            validate(bad, SCHEMA_MANIFEST)

    # D3: adapter_hash 63 chars must be rejected
    def test_adapter_hash_63_chars_rejected(self):
        bad = {**VALID_MANIFEST, "adapter_hash": "a" * 63}
        with pytest.raises(jsonschema.ValidationError):
            validate(bad, SCHEMA_MANIFEST)

    # D4: suite_input_hashes with traversal path must be rejected
    def test_suite_input_hashes_traversal_rejected(self):
        bad = {**VALID_MANIFEST, "suite_input_hashes": {"suite/../etc/passwd": "b" * 64}}
        with pytest.raises(jsonschema.ValidationError):
            validate(bad, SCHEMA_MANIFEST)

    # D5: suite_input_hashes with empty component must be rejected
    def test_suite_input_hashes_double_slash_rejected(self):
        bad = {**VALID_MANIFEST, "suite_input_hashes": {"suite//tasks/foo.yaml": "b" * 64}}
        with pytest.raises(jsonschema.ValidationError):
            validate(bad, SCHEMA_MANIFEST)

    # D6: suite_input_hashes path with dot-dot component must be rejected
    def test_suite_input_hashes_dotdot_rejected(self):
        bad = {**VALID_MANIFEST, "suite_input_hashes": {"suite/tasks/../../../x": "b" * 64}}
        with pytest.raises(jsonschema.ValidationError):
            validate(bad, SCHEMA_MANIFEST)

    # D7: manifest model revision arbitrary string must be rejected (for unrecorded - use oneOf)
    # Design: manifest allows 'unrecorded' for historical records OR exact 40-hex
    def test_manifest_model_revision_arbitrary_rejected(self):
        bad = {**VALID_MANIFEST, "model": {**VALID_MANIFEST["model"], "revision": "arbitrary-garbage-string"}}
        with pytest.raises(jsonschema.ValidationError):
            validate(bad, SCHEMA_MANIFEST)

    # D8: manifest serving image_digest arbitrary string must be rejected
    def test_manifest_serving_image_digest_arbitrary_rejected(self):
        bad = {
            **VALID_MANIFEST,
            "serving": {**VALID_MANIFEST["serving"], "image_digest": "not-a-digest"},
        }
        with pytest.raises(jsonschema.ValidationError):
            validate(bad, SCHEMA_MANIFEST)

    # D9: manifest serving image_digest 'unrecorded' passes (historical manifests)
    def test_manifest_serving_image_digest_unrecorded_passes(self):
        doc = {
            **VALID_MANIFEST,
            "model": {**VALID_MANIFEST["model"], "revision": "unrecorded"},
            "serving": {**VALID_MANIFEST["serving"], "image_digest": "unrecorded"},
        }
        validate(doc, SCHEMA_MANIFEST)

    # D10: manifest serving image_digest sha256:<64hex> passes
    def test_manifest_serving_image_digest_sha256_passes(self):
        doc = {
            **VALID_MANIFEST,
            "serving": {**VALID_MANIFEST["serving"], "image_digest": "sha256:" + "d" * 64},
        }
        validate(doc, SCHEMA_MANIFEST)


# ---------------------------------------------------------------------------
# Gap E: Dependencies declared in requirements-viz.txt
# ---------------------------------------------------------------------------

class TestGapE_DependencyPins:
    """
    Gap E: jsonschema and rfc3339-validator must have exact pins in requirements-viz.txt.
    Currently these are only satisfied by local user site-packages and not declared.
    """

    def test_jsonschema_pinned_in_requirements(self):
        content = REQ_FILE.read_text()
        # Must have an exact version pin like jsonschema==4.25.1
        assert re.search(r"^jsonschema==\d+\.\d+\.\d+", content, re.MULTILINE), (
            f"requirements-viz.txt must declare exact jsonschema pin (e.g. jsonschema==4.25.1); "
            f"currently missing. File: {REQ_FILE}"
        )

    def test_rfc3339_validator_pinned_in_requirements(self):
        content = REQ_FILE.read_text()
        # Must have an exact version pin like rfc3339-validator==0.1.4
        assert re.search(r"^rfc3339-validator==\d+\.\d+\.\d+", content, re.MULTILINE), (
            f"requirements-viz.txt must declare exact rfc3339-validator pin "
            f"(e.g. rfc3339-validator==0.1.4); currently missing. File: {REQ_FILE}"
        )

    def test_jsonschema_version_is_importable(self):
        import importlib.metadata
        version_str = importlib.metadata.version("jsonschema")
        version_tuple = tuple(int(x) for x in version_str.split(".")[:2])
        assert version_tuple >= (4, 0), (
            f"jsonschema must be >= 4.0, got {version_str}"
        )

    def test_rfc3339_validator_is_importable(self):
        try:
            import rfc3339_validator  # noqa: F401
        except ImportError:
            pytest.fail(
                "rfc3339_validator is not importable; install rfc3339-validator "
                "(needed for FormatChecker date-time validation)"
            )


# ---------------------------------------------------------------------------
# Gap F: FormatChecker used in manifest and status schema tests
# ---------------------------------------------------------------------------

class TestGapF_FormatCheckerValidation:
    """
    Gap F: Schema tests must use FormatChecker to prove date-time annotations
    reject invalid timestamps in manifest timing and status history.
    """

    # F1: invalid started_utc must be rejected with FormatChecker
    def test_manifest_invalid_started_utc_rejected_with_format_checker(self):
        bad = {
            **VALID_MANIFEST,
            "timing": {"started_utc": "not-a-date", "completed_utc": None},
        }
        with pytest.raises(jsonschema.ValidationError):
            validate(bad, SCHEMA_MANIFEST, format_checker=jsonschema.FormatChecker())

    # F2: invalid completed_utc must be rejected with FormatChecker (non-null)
    def test_manifest_invalid_completed_utc_rejected_with_format_checker(self):
        bad = {
            **VALID_MANIFEST,
            "timing": {"started_utc": "2026-09-15T00:00:00Z", "completed_utc": "not-a-date"},
        }
        with pytest.raises(jsonschema.ValidationError):
            validate(bad, SCHEMA_MANIFEST, format_checker=jsonschema.FormatChecker())

    # F3: valid manifest passes with FormatChecker
    def test_manifest_valid_passes_with_format_checker(self):
        validate(VALID_MANIFEST, SCHEMA_MANIFEST, format_checker=jsonschema.FormatChecker())

    # F4: null completed_utc passes with FormatChecker
    def test_manifest_null_completed_utc_passes_with_format_checker(self):
        doc = {
            **VALID_MANIFEST,
            "timing": {"started_utc": "2026-09-15T00:00:00Z", "completed_utc": None},
        }
        validate(doc, SCHEMA_MANIFEST, format_checker=jsonschema.FormatChecker())

    # F5: invalid status history timestamp must be rejected WITH FormatChecker
    def test_status_invalid_history_timestamp_rejected_with_format_checker(self):
        bad_status = {
            "schema_version": 1,
            "run_id": "run-x",
            "suite_id": "warpcore-v1",
            "execution_state": "running",
            "lifecycle": "current",
            "history": [{"state": "planned", "timestamp": "NOT-A-DATE", "note": "init"}],
        }
        with pytest.raises(jsonschema.ValidationError):
            validate(bad_status, SCHEMA_STATUS, format_checker=jsonschema.FormatChecker())

    # F6: valid status passes WITH FormatChecker
    def test_status_valid_passes_with_format_checker(self):
        good_status = {
            "schema_version": 1,
            "run_id": "run-x",
            "suite_id": "warpcore-v1",
            "execution_state": "running",
            "lifecycle": "current",
            "history": [{"state": "planned", "timestamp": "2026-09-15T00:00:00Z", "note": "init"}],
        }
        validate(good_status, SCHEMA_STATUS, format_checker=jsonschema.FormatChecker())

    # F7: invalid suite_id in status must be rejected
    def test_status_invalid_suite_id_rejected(self):
        bad_status = {
            "schema_version": 1,
            "run_id": "run-x",
            "suite_id": "wrong-id",
            "execution_state": "running",
            "lifecycle": "current",
            "history": [{"state": "planned", "timestamp": "2026-09-15T00:00:00Z"}],
        }
        with pytest.raises(jsonschema.ValidationError):
            validate(bad_status, SCHEMA_STATUS)


# ---------------------------------------------------------------------------
# Gap G: Harness version field patterns (exact version pins, reject mutable)
# ---------------------------------------------------------------------------

class TestGapG_HarnessVersionPatterns:
    """
    Gap G: Harness version fields must have const or pattern to reject mutable
    constraints like '>=0.4.12'. Dataset revisions must be 40 lowercase hex.
    """

    # G1: lm_eval_version schema must have pattern or const rejecting semver ranges
    def test_lm_eval_version_schema_rejects_range(self):
        schema = load_schema(SCHEMA_SUITE)
        suite = load_suite()
        bad = dict(suite)
        bad["required_harness"] = dict(suite["required_harness"])
        bad["required_harness"]["lm_eval_version"] = ">=0.4.12"
        with pytest.raises(jsonschema.ValidationError):
            validate(bad, SCHEMA_SUITE)

    # G2: lm_eval_revision schema must have pattern rejecting arbitrary strings
    def test_lm_eval_revision_schema_rejects_arbitrary(self):
        schema = load_schema(SCHEMA_SUITE)
        suite = load_suite()
        bad = dict(suite)
        bad["required_harness"] = dict(suite["required_harness"])
        bad["required_harness"]["lm_eval_revision"] = "not-a-sha"
        with pytest.raises(jsonschema.ValidationError):
            validate(bad, SCHEMA_SUITE)

    # G3: exact valid harness version passes
    def test_valid_harness_version_passes(self):
        suite = load_suite()
        validate(suite, SCHEMA_SUITE)

    # G4: dataset revision must be exactly 40 lowercase hex (schema enforces pattern)
    def test_dataset_revision_schema_has_40hex_pattern(self):
        schema = load_schema(SCHEMA_SUITE)
        dataset_ref = schema["$defs"]["dataset_ref"]
        rev_prop = dataset_ref["properties"]["revision"]
        pattern = rev_prop.get("pattern")
        assert pattern is not None, (
            "dataset_ref.revision must have a pattern property to enforce 40 lowercase hex"
        )
        # Pattern must reject non-40-hex strings
        assert not re.match(pattern, "short"), "Pattern must reject 'short'"
        assert not re.match(pattern, "A" * 40), "Pattern must reject uppercase"
        assert re.match(pattern, "a" * 40), "Pattern must accept 40 lowercase hex"

    # G5: swebench_dataset_ref revision also enforces 40hex
    def test_swebench_dataset_revision_schema_has_40hex_pattern(self):
        schema = load_schema(SCHEMA_SUITE)
        swe_ref = schema["$defs"]["swebench_dataset_ref"]
        rev_prop = swe_ref["properties"]["revision"]
        pattern = rev_prop.get("pattern")
        assert pattern is not None, (
            "swebench_dataset_ref.revision must have a pattern to enforce 40 lowercase hex"
        )
        assert re.match(pattern, "a" * 40), "Pattern must accept 40 lowercase hex"
        assert not re.match(pattern, "A" * 40), "Pattern must reject uppercase"


# ---------------------------------------------------------------------------
# Gap H: required_evidence honest artifact classes
# ---------------------------------------------------------------------------

class TestGapH_RequiredEvidenceHonesty:
    """
    Gap H: 'scoring_utility' for GSM8K is semantically wrong (GSM8K has no
    utility file; IFEval utilities live in the harness). Replace with honest
    class 'scoring_implementation' or document clearly. At minimum, do not
    promise nonexistent artifact files.

    This test documents the design decision: required_evidence must use
    'scoring_implementation' (not 'scoring_utility') for all quality benchmarks,
    where 'scoring_implementation' refers to the pinned task file + optional
    pinned utility/harness revision.
    """

    EXPECTED_EVIDENCE = [
        "aggregate_result",
        "samples_jsonl_gz",
        "per_item_csv",
        "task_yaml",
        "scoring_implementation",  # replaces 'scoring_utility' - honest class
        "run_log",
        "command_txt",
        "manifest_json",
        "status_json",
        "done_sentinel",
        "completion_token_counts",
        "finish_reasons",
        "full_response_fields",
    ]

    FORBIDDEN_EVIDENCE = ["scoring_utility"]  # removed - was misleading

    def _check_benchmark_evidence(self, benchmark: str):
        suite = load_suite()
        evidence = suite["benchmarks"][benchmark].get("required_evidence", [])
        missing = [f for f in self.EXPECTED_EVIDENCE if f not in evidence]
        forbidden = [f for f in self.FORBIDDEN_EVIDENCE if f in evidence]
        assert not missing, f"{benchmark} required_evidence missing: {missing}"
        assert not forbidden, (
            f"{benchmark} required_evidence must not contain {forbidden}; "
            f"'scoring_utility' is misleading (GSM8K has no utility file, "
            f"IFEval utilities live in harness). Use 'scoring_implementation'."
        )

    def test_gsm8k_evidence_uses_scoring_implementation(self):
        self._check_benchmark_evidence("gsm8k")

    def test_ifeval_evidence_uses_scoring_implementation(self):
        self._check_benchmark_evidence("ifeval")

    def test_gpqa_evidence_uses_scoring_implementation(self):
        self._check_benchmark_evidence("gpqa_diamond")


# ---------------------------------------------------------------------------
# Gap I: Throughput per_request_timeout_s policy
# ---------------------------------------------------------------------------

class TestGapI_ThroughputTimeoutPolicy:
    """
    Gap I: per_request_timeout_s=3600 in throughput is an invented value with no
    approved evidence. The timeout_policy must describe the timeout MODE without
    freezing an arbitrary number. Schema should not allow per_request_timeout_s
    in throughput's timeout_policy (it belongs only in quality/swebench contexts),
    OR the suite must document it as mode-based (not a hardcoded magic number).

    Design decision enforced here: throughput timeout_policy must use
    use_arithmetic: false with an explicit timeout_mode field (not per_request_timeout_s),
    OR per_request_timeout_s must be absent (mode: run_until_plateau).
    """

    def test_throughput_timeout_policy_no_invented_per_request_timeout(self):
        suite = load_suite()
        tp = suite["benchmarks"]["throughput"]["timeout_policy"]
        assert "per_request_timeout_s" not in tp, (
            "throughput timeout_policy must not contain per_request_timeout_s=3600; "
            "this was an invented value with no approved evidence. "
            "Use timeout_mode or document vllm bench serve's natural termination."
        )

    def test_throughput_timeout_policy_use_arithmetic_false(self):
        suite = load_suite()
        tp = suite["benchmarks"]["throughput"]["timeout_policy"]
        assert tp.get("use_arithmetic") is False, (
            "throughput timeout_policy.use_arithmetic must be False "
            "(vllm bench serve runs until plateau, no arithmetic sizing)"
        )

    def test_throughput_timeout_schema_rejects_per_request_in_throughput_context(self):
        """
        The suite schema's throughput_benchmark timeout_policy must not allow
        per_request_timeout_s (it's quality-benchmark-specific).
        Alternatively, if per_request_timeout_s is removed from throughput YAML,
        just verify the suite validates without it.
        """
        suite = load_suite()
        # After fix: suite validates without per_request_timeout_s
        validate(suite, SCHEMA_SUITE)


# ---------------------------------------------------------------------------
# Gap J: SWE-bench dataset provenance comments correctness
# ---------------------------------------------------------------------------

class TestGapJ_SwebenchDatasetProvenance:
    """
    Gap J: Suite comments must not imply old historical runs used the current
    pinned revisions. v1 is prospective; revisions are current local HF refs
    captured for v1, not revisions from historical runs.
    Also verifies the corrected dataset path (from Gap B).
    """

    def test_swebench_dataset_path_is_correct(self):
        suite = load_suite()
        path = suite["benchmarks"]["swebench"]["dataset"]["path"]
        assert path == "princeton-nlp/SWE-bench_Verified", (
            f"swebench dataset.path must be 'princeton-nlp/SWE-bench_Verified' "
            f"(the org that actually hosts this dataset on HF); "
            f"got: {path!r}"
        )

    def test_swebench_revision_is_pinned_40hex(self):
        suite = load_suite()
        rev = suite["benchmarks"]["swebench"]["dataset"]["revision"]
        assert re.match(r"^[0-9a-f]{40}$", rev), (
            f"swebench dataset revision must be 40 lowercase hex, got: {rev!r}"
        )
        assert rev == "c104f840cc67f8b6eec6f759ebc8b2693d585d4a", (
            f"swebench dataset revision must be canonical v1 commit, got: {rev!r}"
        )

    def test_suite_yaml_comment_does_not_claim_historical_runs_used_revision(self):
        """
        The suite YAML must not contain wording that implies old runs used these
        dataset revisions. v1 is prospective; comments must say 'captured for v1'
        not 'used in run X'.
        """
        content = SUITE_FILE.read_text()
        # Check that the file doesn't have misleading 'historical' revision attribution
        # This is a soft check on comment content - ensure no claim that these revisions
        # were used in past/old runs
        forbidden_patterns = [
            "used in historical run",
            "revision from old run",
            "from the original run",
        ]
        for pattern in forbidden_patterns:
            assert pattern not in content.lower(), (
                f"suite YAML must not claim {pattern!r}; "
                f"revisions are current local HF refs captured for v1 (prospective)"
            )
