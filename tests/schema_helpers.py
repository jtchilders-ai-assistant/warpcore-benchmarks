"""
Shared schema test helpers for warpcore-v1 test suite.

This module is the single source of truth for:
  - Canonical valid document builders (VALID_MANIFEST, VALID_ADAPTER, VALID_STATUS)
  - validate_doc() — jsonschema validate with FormatChecker always enabled
  - Schema file paths (SCHEMA_SUITE, SCHEMA_ADAPTER, SCHEMA_MANIFEST, SCHEMA_STATUS)

All three test modules (test_suite_contract, test_suite_compliance,
test_schema_hardening) import from here so that a schema addition requires
updating exactly one source of truth.

FormatChecker is always active in validate_doc.  Callers must NOT pass it as
an argument — it is not optional.  This prevents the class of bug where a test
exercises the schema without format enforcement and incorrectly accepts invalid
date-time strings.
"""
from __future__ import annotations

import json
from pathlib import Path

import jsonschema

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

_TESTS_DIR = Path(__file__).parent
REPO = _TESTS_DIR.parent
_SUITE_DIR = REPO / "suite"
_SCHEMAS_DIR = _SUITE_DIR / "schemas"

SCHEMA_SUITE = _SCHEMAS_DIR / "suite.schema.json"
SCHEMA_ADAPTER = _SCHEMAS_DIR / "adapter.schema.json"
SCHEMA_MANIFEST = _SCHEMAS_DIR / "manifest.schema.json"
SCHEMA_STATUS = _SCHEMAS_DIR / "result-status.schema.json"


# ---------------------------------------------------------------------------
# Validation helper — FormatChecker always on
# ---------------------------------------------------------------------------

def validate_doc(instance: dict, schema_path: Path) -> None:
    """Validate *instance* against the JSON Schema at *schema_path*.

    FormatChecker is always enabled so date-time format annotations are
    enforced.  Raises jsonschema.ValidationError on failure.
    """
    schema = json.loads(schema_path.read_text())
    jsonschema.validate(instance, schema, format_checker=jsonschema.FormatChecker())


# ---------------------------------------------------------------------------
# Canonical valid documents
# ---------------------------------------------------------------------------

#: A minimal, fully valid adapter document.  Deep-copy before mutating.
#: campaign_status=canonical requires a resolved image digest and 40-hex revision.
VALID_ADAPTER: dict = {
    "adapter_schema_version": 1,
    "campaign_status": "canonical",
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

#: A minimal, fully valid manifest document.  Deep-copy before mutating.
VALID_MANIFEST: dict = {
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

#: A minimal, fully valid result-status document.  Deep-copy before mutating.
VALID_STATUS: dict = {
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
