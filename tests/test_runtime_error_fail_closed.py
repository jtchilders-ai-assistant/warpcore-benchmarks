"""Regression tests for the all-RuntimeError Qwen SWE-bench campaign."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import yaml

REPO = Path(__file__).parent.parent
VIZ = REPO / "viz"
if str(VIZ) not in sys.path:
    sys.path.insert(0, str(VIZ))

import run_swebench
import validate_campaign


def _write_runtime_error_generation(run_dir: Path, ids: list[str]) -> None:
    raw = run_dir / "raw"
    raw.mkdir(parents=True)
    (raw / "preds.json").write_text(json.dumps({
        iid: {"instance_id": iid, "model_patch": ""} for iid in ids
    }))
    (raw / "exit_statuses.json").write_text(json.dumps({iid: "RuntimeError" for iid in ids}))
    trajectories = raw / "trajectories"
    trajectories.mkdir()
    for iid in ids:
        (trajectories / f"{iid}.traj").write_text(json.dumps({
            "instance_id": iid,
            "info": {"exit_status": "RuntimeError"},
        }))
    (raw / "run.log").write_text("RuntimeError: Error calculating cost for unmapped model\n")


def test_frozen_scaffold_ignores_unmapped_local_model_cost_errors():
    scaffold = yaml.safe_load((REPO / "suite/swebench/scaffold.yaml").read_text())
    assert scaffold["model"]["cost_tracking"] == "ignore_errors"


def test_runner_rejects_complete_inventory_of_runtime_errors(tmp_path):
    ids = ["repo__issue-1", "repo__issue-2"]
    run_dir = tmp_path / "run"
    _write_runtime_error_generation(run_dir, ids)

    errors = run_swebench._verify_generation_evidence(run_dir, ids)

    assert errors
    assert any("RuntimeError" in error and "infrastructure" in error for error in errors)


def test_runner_rejects_even_one_runtime_error_in_a_partial_inventory(tmp_path):
    ids = ["repo__issue-1", "repo__issue-2"]
    run_dir = tmp_path / "run"
    _write_runtime_error_generation(run_dir, ids)
    statuses_path = run_dir / "raw" / "exit_statuses.json"
    statuses_path.write_text(json.dumps({ids[0]: "RuntimeError", ids[1]: "Submitted"}))

    errors = run_swebench._verify_generation_evidence(run_dir, ids)

    assert any("1/2 RuntimeError" in error for error in errors)


def test_invalid_completed_archive_with_done_is_explicitly_ineligible(tmp_path):
    """A historical DONE records what happened; lifecycle controls eligibility."""
    source = (
        REPO / "results/qwen3.6-35b-a3b/runs/warpcore-v1/swebench"
        / "qwen36-swebench-n100-20260917"
    )
    status = json.loads((source / "status.json").read_text())

    assert (source / "DONE").exists()
    assert status["execution_state"] == "completed"
    assert status["lifecycle"] == "invalid"
    result = validate_campaign.validate(
        source,
        REPO / "suite/warpcore-v1.yaml",
        REPO / "adapters/qwen3.6-35b-a3b.yaml",
        for_publication=True,
    )
    assert result.passed is False
    assert result.eligible is False
    assert any("may not be published" in error for error in result.errors)


def test_validator_rejects_runtime_errors_and_zero_graded_verdicts(tmp_path):
    ids = ["repo__issue-1", "repo__issue-2"]
    run_dir = tmp_path / "run"
    _write_runtime_error_generation(run_dir, ids)
    raw = run_dir / "raw"
    (raw / "grading_results.json").write_text(json.dumps({
        "resolved_ids": [],
        "unresolved_ids": [],
        "empty_patch_ids": ids,
        "error_ids": [],
        "incomplete_ids": [],
    }))
    errors: list[str] = []

    validate_campaign._validate_swebench_grading(
        run_dir,
        {"item_inventory": {"expected": len(ids), "submitted": len(ids)}},
        errors,
        for_publication=True,
        frozen_ids=frozenset(ids),
    )

    assert any("RuntimeError" in error and "publication" in error for error in errors)
    assert any("no graded verdict" in error for error in errors)
