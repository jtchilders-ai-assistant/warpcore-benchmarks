"""Regression tests for Task 3 adapter integration and fail-closed readiness."""
from __future__ import annotations

import copy
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[1]
_VIZ = _REPO / "viz"
if str(_VIZ) not in sys.path:
    sys.path.insert(0, str(_VIZ))

from contract import (  # noqa: E402
    validate_adapter_campaign_ready,
    validate_adapters_dir,
    validate_json,
)
from schema_helpers import SCHEMA_ADAPTER, VALID_ADAPTER  # noqa: E402


def _suite() -> dict:
    return yaml.safe_load((_REPO / "suite" / "warpcore-v1.yaml").read_text())


def test_canonical_schema_rejects_unresolved_revision_and_image():
    adapter = copy.deepcopy(VALID_ADAPTER)
    adapter["model"]["revision"] = "unresolved"
    adapter["serving"]["image"] = "unresolved"

    errors = validate_json(adapter, SCHEMA_ADAPTER)

    assert any("revision" in error for error in errors), errors
    assert any("image" in error for error in errors), errors


def test_canonical_schema_rejects_unresolved_capacity_values():
    adapter = copy.deepcopy(VALID_ADAPTER)
    adapter["serving"]["max_model_len"] = "unresolved"
    adapter["serving"]["gpu_memory_utilization"] = "unresolved"
    adapter["serving"]["max_num_seqs"] = "unresolved"

    errors = validate_json(adapter, SCHEMA_ADAPTER)

    assert errors, "canonical adapters must resolve every required serving value"


def test_noncanonical_schema_allows_explicitly_unresolved_capacity_values():
    adapter = copy.deepcopy(VALID_ADAPTER)
    adapter["campaign_status"] = "noncanonical"
    adapter["noncanonical_reason"] = (
        "model_revision_unrecorded; image_digest_unrecorded; "
        "serving_capacity_unrecorded"
    )
    adapter["model"]["revision"] = "unresolved"
    adapter["serving"]["image"] = "unresolved"
    adapter["serving"]["max_model_len"] = "unresolved"
    adapter["serving"]["gpu_memory_utilization"] = "unresolved"
    adapter["serving"]["max_num_seqs"] = "unresolved"

    assert validate_json(adapter, SCHEMA_ADAPTER) == []


def test_campaign_readiness_blocks_without_tokenized_prompt_evidence():
    adapter = copy.deepcopy(VALID_ADAPTER)

    errors = validate_adapter_campaign_ready(adapter, adapter["model"]["slug"], suite=_suite())

    assert any("tokenized prompt" in error.lower() for error in errors), errors


def test_campaign_readiness_rejects_prompt_plus_output_overflow():
    adapter = copy.deepcopy(VALID_ADAPTER)
    adapter["serving"]["max_model_len"] = 65_600
    prompt_token_maxima = {
        "gsm8k": 1_000,
        "ifeval": 100,
        "gpqa_diamond": 100,
    }

    errors = validate_adapter_campaign_ready(
        adapter,
        adapter["model"]["slug"],
        suite=_suite(),
        prompt_token_maxima=prompt_token_maxima,
    )

    assert any("ifeval" in error and "65636" in error for error in errors), errors
    assert any("gpqa_diamond" in error and "65636" in error for error in errors), errors


def test_campaign_readiness_accepts_complete_prompt_plus_output_evidence():
    adapter = copy.deepcopy(VALID_ADAPTER)
    adapter["serving"]["max_model_len"] = 131_072
    prompt_token_maxima = {
        "gsm8k": 1_000,
        "ifeval": 2_000,
        "gpqa_diamond": 2_000,
    }

    errors = validate_adapter_campaign_ready(
        adapter,
        adapter["model"]["slug"],
        suite=_suite(),
        prompt_token_maxima=prompt_token_maxima,
    )

    assert errors == []


def test_directory_validation_rejects_missing_adapter_directory(tmp_path):
    errors = validate_adapters_dir(_REPO, tmp_path / "missing")

    assert any("not found" in error.lower() for error in errors), errors


def test_directory_validation_rejects_invalid_checked_in_adapter(tmp_path):
    adapter = copy.deepcopy(VALID_ADAPTER)
    adapter["adapter_schema_version"] = 999
    (tmp_path / "bad.yaml").write_text(yaml.safe_dump(adapter))

    errors = validate_adapters_dir(_REPO, tmp_path)

    assert any("adapter_schema_version" in error for error in errors), errors


def test_directory_validation_applies_readiness_to_canonical_adapters(tmp_path):
    adapter = copy.deepcopy(VALID_ADAPTER)
    (tmp_path / "canonical.yaml").write_text(yaml.safe_dump(adapter))

    errors = validate_adapters_dir(_REPO, tmp_path, suite=_suite())

    assert any("tokenized prompt evidence" in error.lower() for error in errors), errors


def test_contract_cli_validates_checked_in_adapters(tmp_path):
    repo = tmp_path / "repo"
    (repo / "suite" / "schemas").mkdir(parents=True)
    (repo / "viz").mkdir()
    (repo / "adapters").mkdir()
    shutil.copy2(_REPO / "suite" / "warpcore-v1.yaml", repo / "suite" / "warpcore-v1.yaml")
    shutil.copy2(SCHEMA_ADAPTER, repo / "suite" / "schemas" / "adapter.schema.json")
    adapter = copy.deepcopy(VALID_ADAPTER)
    adapter["adapter_schema_version"] = 999
    (repo / "adapters" / "bad.yaml").write_text(yaml.safe_dump(adapter))

    completed = subprocess.run(
        [
            "/usr/bin/python3",
            str(_REPO / "viz" / "validate_suite.py"),
            str(_REPO / "suite" / "warpcore-v1.yaml"),
            "--repo",
            str(repo),
        ],
        cwd=_REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert "bad.yaml" in completed.stdout + completed.stderr
