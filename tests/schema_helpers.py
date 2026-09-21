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


# ---------------------------------------------------------------------------
# SWE-bench qualification fixture
# ---------------------------------------------------------------------------

#: Committed production-path qualification evidence: 20 suite-owned instances,
#: all terminal Submitted, every patch a real diff, every trajectory carrying
#: well-formed tool calls, and an official SWE-bench schema-v2 grader report.
QUALIFICATION_FIXTURE_RUN = _TESTS_DIR / "fixtures" / "swebench_qualification" / "run"


def install_test_qualification(
    *,
    repo,
    adapter_path,
    endpoint: str,
    slug: str | None = None,
    suite_id: str = "warpcore-v1",
    suite_path=None,
    mutate_statuses=None,
    now=None,
):
    """Install a valid SWE-bench qualification under *repo* and return its path.

    Runner tests need a launch-authorizing qualification the same way a real
    campaign does; without one every live-run test would stop at the gate and
    silently stop testing whatever it was written to test.  This copies the
    committed production-path evidence beside the canonical artifact location and
    seals a record bound to *repo*, *adapter_path*, and *endpoint*.

    ``mutate_statuses`` receives the exit-status dict before the record is sealed,
    so a negative test can plant, say, a RepeatedFormatError and keep the digests
    honest.
    """
    import json as _json
    import shutil as _shutil
    import sys as _sys

    _viz = str(REPO / "viz")
    if _viz not in _sys.path:
        _sys.path.insert(0, _viz)
    import swebench_qualification as _sq  # noqa: E402
    import yaml as _yaml  # noqa: E402

    adapter_path = Path(adapter_path)
    if slug is None:
        slug = (_yaml.safe_load(adapter_path.read_text()).get("model") or {}).get("slug", "")
    served_model_id = (_yaml.safe_load(adapter_path.read_text()).get("model") or {}).get("id", "")

    artifact_path = _sq.default_artifact_path(repo, suite_id, slug)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    evidence = artifact_path.parent / "run"
    if evidence.exists():
        _shutil.rmtree(evidence)
    _shutil.copytree(QUALIFICATION_FIXTURE_RUN, evidence)

    if mutate_statuses is not None:
        statuses_path = evidence / "raw" / "exit_statuses.json"
        statuses = _json.loads(statuses_path.read_text())
        mutate_statuses(statuses)
        statuses_path.write_text(_json.dumps(statuses, indent=2) + "\n")

    artifact = _sq.build_qualification_artifact(
        repo=REPO,
        suite_path=suite_path or (REPO / "suite" / "warpcore-v1.yaml"),
        adapter_path=adapter_path,
        endpoint=endpoint,
        served_model_id=served_model_id,
        evidence_run_dir=evidence,
        artifact_path=artifact_path,
        now=now,
    )
    artifact_path.write_text(_json.dumps(artifact, indent=2) + "\n")
    return artifact_path
