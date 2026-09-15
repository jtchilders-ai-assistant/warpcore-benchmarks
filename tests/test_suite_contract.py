"""
Task 1 acceptance tests: freeze warpcore-v1 suite contract.

Tests must ALL fail (RED) before suite/ artifacts are created, and ALL pass
(GREEN) after.  Specifically verified before wiring into make ci.

Run with: /usr/bin/python3 -m pytest tests/test_suite_contract.py -q
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).parent.parent
SUITE_DIR = REPO / "suite"
SUITE_FILE = SUITE_DIR / "warpcore-v1.yaml"
SCHEMAS_DIR = SUITE_DIR / "schemas"
TASKS_DIR = SUITE_DIR / "tasks"
SWEBENCH_DIR = SUITE_DIR / "swebench"

INSTANCES_FILE = SWEBENCH_DIR / "instances-seed42-n100.json"
SCAFFOLD_FILE = SWEBENCH_DIR / "scaffold.yaml"
GPQA_TASK = TASKS_DIR / "gpqa_diamond_clean_v3.yaml"
GPQA_UTILS = TASKS_DIR / "gpqa_utils.py"
GSM8K_TASK = TASKS_DIR / "gsm8k_clean_v1.yaml"

SCHEMA_SUITE = SCHEMAS_DIR / "suite.schema.json"
SCHEMA_ADAPTER = SCHEMAS_DIR / "adapter.schema.json"
SCHEMA_MANIFEST = SCHEMAS_DIR / "manifest.schema.json"
SCHEMA_STATUS = SCHEMAS_DIR / "result-status.schema.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_path(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_suite() -> dict:
    return yaml.safe_load(SUITE_FILE.read_text())


def load_instances() -> list:
    return json.loads(INSTANCES_FILE.read_text())


def load_schema(p: Path) -> dict:
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# Suite file existence and top-level identity
# ---------------------------------------------------------------------------

class TestSuiteFileExists:
    def test_suite_file_present(self):
        assert SUITE_FILE.exists(), f"Missing: {SUITE_FILE}"

    def test_suite_id(self):
        suite = load_suite()
        assert suite["suite_id"] == "warpcore-v1"

    def test_suite_schema_version_present(self):
        suite = load_suite()
        assert "suite_schema_version" in suite

    def test_suite_has_benchmarks(self):
        suite = load_suite()
        assert "benchmarks" in suite


# ---------------------------------------------------------------------------
# Generation ceilings — the three numbers that must never drift
# ---------------------------------------------------------------------------

class TestGenerationCeilings:
    def test_gsm8k_ceiling(self):
        suite = load_suite()
        assert suite["benchmarks"]["gsm8k"]["generation_ceiling"] == 8192

    def test_ifeval_ceiling(self):
        suite = load_suite()
        assert suite["benchmarks"]["ifeval"]["generation_ceiling"] == 65536

    def test_gpqa_diamond_ceiling(self):
        suite = load_suite()
        assert suite["benchmarks"]["gpqa_diamond"]["generation_ceiling"] == 65536


# ---------------------------------------------------------------------------
# Sampling — temperature must be 0 across all quality benchmarks
# ---------------------------------------------------------------------------

class TestSampling:
    def test_gsm8k_temperature(self):
        suite = load_suite()
        assert suite["benchmarks"]["gsm8k"]["sampling"]["temperature"] == 0

    def test_ifeval_temperature(self):
        suite = load_suite()
        assert suite["benchmarks"]["ifeval"]["sampling"]["temperature"] == 0

    def test_gpqa_diamond_temperature(self):
        suite = load_suite()
        assert suite["benchmarks"]["gpqa_diamond"]["sampling"]["temperature"] == 0


# ---------------------------------------------------------------------------
# SWE-bench — exactly 100 unique IDs, fixed instance set
# ---------------------------------------------------------------------------

class TestSwebenchInstances:
    def test_instances_file_present(self):
        assert INSTANCES_FILE.exists(), f"Missing: {INSTANCES_FILE}"

    def test_instances_count(self):
        instances = load_instances()
        assert len(instances) == 100

    def test_instances_unique(self):
        instances = load_instances()
        assert len(set(instances)) == 100

    def test_instances_are_strings(self):
        instances = load_instances()
        for item in instances:
            assert isinstance(item, str), f"Expected str, got {type(item)}: {item!r}"

    def test_instances_format(self):
        """Each ID must follow the SWE-bench repo__repo-N pattern."""
        import re
        instances = load_instances()
        pattern = re.compile(r"^[a-zA-Z0-9_-]+__[a-zA-Z0-9_-]+-\d+$")
        bad = [i for i in instances if not pattern.match(i)]
        assert bad == [], f"IDs with unexpected format: {bad[:5]}"


# ---------------------------------------------------------------------------
# Suite hash integrity — declared hashes must match actual files
# ---------------------------------------------------------------------------

class TestSuiteHashes:
    def test_suite_declares_gpqa_task_hash(self):
        suite = load_suite()
        declared = suite["benchmarks"]["gpqa_diamond"]["task_sha256"]
        actual = sha256_path(GPQA_TASK)
        assert declared == actual, (
            f"GPQA task hash mismatch: declared={declared!r} actual={actual!r}"
        )

    def test_suite_declares_gpqa_utils_hash(self):
        suite = load_suite()
        declared = suite["benchmarks"]["gpqa_diamond"]["utils_sha256"]
        actual = sha256_path(GPQA_UTILS)
        assert declared == actual, (
            f"GPQA utils hash mismatch: declared={declared!r} actual={actual!r}"
        )

    def test_suite_declares_gsm8k_task_hash(self):
        suite = load_suite()
        declared = suite["benchmarks"]["gsm8k"]["task_sha256"]
        actual = sha256_path(GSM8K_TASK)
        assert declared == actual, (
            f"GSM8K task hash mismatch: declared={declared!r} actual={actual!r}"
        )

    def test_suite_declares_instances_hash(self):
        suite = load_suite()
        declared = suite["benchmarks"]["swebench"]["instances_sha256"]
        actual = sha256_path(INSTANCES_FILE)
        assert declared == actual, (
            f"Instance set hash mismatch: declared={declared!r} actual={actual!r}"
        )

    def test_suite_declares_scaffold_hash(self):
        suite = load_suite()
        declared = suite["benchmarks"]["swebench"]["scaffold_sha256"]
        actual = sha256_path(SCAFFOLD_FILE)
        assert declared == actual, (
            f"Scaffold hash mismatch: declared={declared!r} actual={actual!r}"
        )


# ---------------------------------------------------------------------------
# Canonical task files present
# ---------------------------------------------------------------------------

class TestTaskFilesPresent:
    def test_gpqa_task_present(self):
        assert GPQA_TASK.exists(), f"Missing: {GPQA_TASK}"

    def test_gpqa_utils_present(self):
        assert GPQA_UTILS.exists(), f"Missing: {GPQA_UTILS}"

    def test_gsm8k_task_present(self):
        assert GSM8K_TASK.exists(), f"Missing: {GSM8K_TASK}"

    def test_scaffold_present(self):
        assert SCAFFOLD_FILE.exists(), f"Missing: {SCAFFOLD_FILE}"


# ---------------------------------------------------------------------------
# Scaffold: model_name must be a runner-injected placeholder, not a real ID
# ---------------------------------------------------------------------------

class TestScaffold:
    def test_scaffold_model_is_placeholder(self):
        scaffold = yaml.safe_load(SCAFFOLD_FILE.read_text())
        model_name = scaffold["model"]["model_name"]
        # Must NOT be hardcoded to any specific model
        assert "{{" in model_name or model_name == "__RUNNER_INJECTED__", (
            f"scaffold.yaml model_name must be a runner-injected placeholder, got: {model_name!r}"
        )

    def test_scaffold_preserves_submit_protocol(self):
        """Robust git-based submit protocol must be present."""
        content = SCAFFOLD_FILE.read_text()
        assert "git add -A" in content and "git diff --cached" in content, (
            "scaffold.yaml must preserve the robust git add -A && git diff --cached submit protocol"
        )

    def test_scaffold_pull_timeout_raised(self):
        scaffold = yaml.safe_load(SCAFFOLD_FILE.read_text())
        pull_timeout = scaffold["environment"]["pull_timeout"]
        assert pull_timeout >= 1800, (
            f"pull_timeout must be >= 1800 (was 120s default that silently zeroed 22/100 instances), got: {pull_timeout}"
        )


# ---------------------------------------------------------------------------
# JSON Schemas present and structurally valid
# ---------------------------------------------------------------------------

class TestSchemasPresent:
    def test_suite_schema_present(self):
        assert SCHEMA_SUITE.exists(), f"Missing: {SCHEMA_SUITE}"

    def test_adapter_schema_present(self):
        assert SCHEMA_ADAPTER.exists(), f"Missing: {SCHEMA_ADAPTER}"

    def test_manifest_schema_present(self):
        assert SCHEMA_MANIFEST.exists(), f"Missing: {SCHEMA_MANIFEST}"

    def test_result_status_schema_present(self):
        assert SCHEMA_STATUS.exists(), f"Missing: {SCHEMA_STATUS}"


# ---------------------------------------------------------------------------
# All schemas use additionalProperties: false
# ---------------------------------------------------------------------------

class TestSchemasAdditionalPropertiesFalse:
    """All object schemas must use additionalProperties:false to reject unknown keys."""

    def _check_no_additional(self, schema: dict, path: str = "$"):
        """Recursively check that any object-type schema has additionalProperties:false."""
        if schema.get("type") == "object" or "properties" in schema:
            assert schema.get("additionalProperties") is False, (
                f"Schema at {path} is an object but missing 'additionalProperties: false'"
            )
        for key, sub in schema.get("properties", {}).items():
            self._check_no_additional(sub, f"{path}.{key}")
        if "definitions" in schema:
            for key, sub in schema["definitions"].items():
                self._check_no_additional(sub, f"{path}.$defs.{key}")
        if "$defs" in schema:
            for key, sub in schema["$defs"].items():
                self._check_no_additional(sub, f"{path}.$defs.{key}")

    def test_suite_schema_additional_false(self):
        self._check_no_additional(load_schema(SCHEMA_SUITE), "suite.schema.json")

    def test_adapter_schema_additional_false(self):
        self._check_no_additional(load_schema(SCHEMA_ADAPTER), "adapter.schema.json")

    def test_manifest_schema_additional_false(self):
        self._check_no_additional(load_schema(SCHEMA_MANIFEST), "manifest.schema.json")

    def test_result_status_schema_additional_false(self):
        self._check_no_additional(load_schema(SCHEMA_STATUS), "result-status.schema.json")


# ---------------------------------------------------------------------------
# Schema validation: valid documents pass, invalid documents are rejected
# ---------------------------------------------------------------------------

import jsonschema  # required; plan tech-stack specifies jsonschema 4


def _validate(instance: dict, schema_path: Path) -> None:
    """Validate instance against schema; raise jsonschema.ValidationError on failure."""
    schema = json.loads(schema_path.read_text())
    jsonschema.validate(instance, schema)


class TestAdapterSchemaValidation:
    """Adapter schema must accept valid documents and reject experiment-level overrides."""

    VALID_ADAPTER = {
        "adapter_schema_version": 1,
        "model": {
            "slug": "qwen3.6-35b-a3b",
            "id": "Qwen/Qwen3.6-35B-A3B-FP8",
            "revision": "abc123def456abc123def456abc123def456abc1",
        },
        "serving": {
            "image": "eugr/spark-vllm@sha256:deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
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

    def test_valid_adapter_passes(self):
        _validate(self.VALID_ADAPTER, SCHEMA_ADAPTER)

    def test_adapter_with_generation_ceiling_rejected(self):
        """An adapter must not be allowed to override generation_ceiling."""
        bad = {**self.VALID_ADAPTER, "generation_ceiling": 99999}
        with pytest.raises(jsonschema.ValidationError):
            _validate(bad, SCHEMA_ADAPTER)

    def test_adapter_with_task_rejected(self):
        bad = {**self.VALID_ADAPTER, "task": "gpqa_diamond_cot_zeroshot_clean"}
        with pytest.raises(jsonschema.ValidationError):
            _validate(bad, SCHEMA_ADAPTER)

    def test_adapter_with_sampling_rejected(self):
        bad = {**self.VALID_ADAPTER, "sampling": {"temperature": 0.5}}
        with pytest.raises(jsonschema.ValidationError):
            _validate(bad, SCHEMA_ADAPTER)

    def test_adapter_with_instance_ids_rejected(self):
        bad = {**self.VALID_ADAPTER, "instance_ids": ["django__django-11299"]}
        with pytest.raises(jsonschema.ValidationError):
            _validate(bad, SCHEMA_ADAPTER)

    def test_adapter_missing_required_model_id_rejected(self):
        bad_model = {k: v for k, v in self.VALID_ADAPTER["model"].items() if k != "id"}
        bad = {**self.VALID_ADAPTER, "model": bad_model}
        with pytest.raises(jsonschema.ValidationError):
            _validate(bad, SCHEMA_ADAPTER)


class TestResultStatusSchemaValidation:
    """result-status schema must separate execution state from lifecycle state."""

    VALID_STATUS = {
        "schema_version": 1,
        "run_id": "run-2026-09-15T00-00-00",
        "suite_id": "warpcore-v1",
        "execution_state": "running",
        "lifecycle": "current",
        "history": [
            {
                "state": "planned",
                "timestamp": "2026-09-15T00:00:00Z",
                "note": "Campaign initialized",
            }
        ],
    }

    def test_valid_status_passes(self):
        _validate(self.VALID_STATUS, SCHEMA_STATUS)

    def test_status_unknown_execution_state_rejected(self):
        bad = {**self.VALID_STATUS, "execution_state": "unknown_state"}
        with pytest.raises(jsonschema.ValidationError):
            _validate(bad, SCHEMA_STATUS)

    def test_status_unknown_lifecycle_rejected(self):
        bad = {**self.VALID_STATUS, "lifecycle": "unknown_lifecycle"}
        with pytest.raises(jsonschema.ValidationError):
            _validate(bad, SCHEMA_STATUS)

    def test_status_extra_field_rejected(self):
        bad = {**self.VALID_STATUS, "extra_key": "surprise"}
        with pytest.raises(jsonschema.ValidationError):
            _validate(bad, SCHEMA_STATUS)


class TestManifestSchemaValidation:
    """Manifest schema must require suite ID, hashes, model identity, etc."""

    VALID_MANIFEST = {
        "schema_version": 1,
        "suite_id": "warpcore-v1",
        "suite_schema_version": 1,
        "run_id": "run-2026-09-15T00-00-00",
        "benchmark": "gsm8k",
        "adapter_hash": "a" * 64,
        "suite_task_hash": "b" * 64,
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
        },
        "item_inventory": {
            "expected": 1319,
            "submitted": 1319,
        },
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

    def test_valid_manifest_passes(self):
        _validate(self.VALID_MANIFEST, SCHEMA_MANIFEST)

    def test_manifest_missing_suite_id_rejected(self):
        bad = {k: v for k, v in self.VALID_MANIFEST.items() if k != "suite_id"}
        with pytest.raises(jsonschema.ValidationError):
            _validate(bad, SCHEMA_MANIFEST)

    def test_manifest_extra_field_rejected(self):
        bad = {**self.VALID_MANIFEST, "mystery_field": "no"}
        with pytest.raises(jsonschema.ValidationError):
            _validate(bad, SCHEMA_MANIFEST)
