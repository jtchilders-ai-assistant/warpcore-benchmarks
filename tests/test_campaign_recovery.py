"""Regression tests for evidence-preserving recovery of failed campaign post-processing."""
from __future__ import annotations

import copy
import pathlib
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
_VIZ = _REPO / "viz"
if str(_VIZ) not in sys.path:
    sys.path.insert(0, str(_VIZ))

import campaign_state
import validate_campaign


def _failed_status() -> dict:
    return {
        "schema_version": 1,
        "run_id": "recovered-run",
        "suite_id": "warpcore-v2",
        "execution_state": "failed",
        "lifecycle": "invalid",
        "history": [
            {"state": "planned", "timestamp": "2026-09-29T00:00:00Z"},
            {"state": "preflight_passed", "timestamp": "2026-09-29T00:00:01Z"},
            {"state": "running", "timestamp": "2026-09-29T00:00:02Z"},
            {
                "state": "failed",
                "timestamp": "2026-09-30T00:00:00Z",
                "note": "grading failed",
            },
        ],
    }


def test_recover_failed_postprocessing_preserves_failure_history():
    original = _failed_status()
    recovered = campaign_state.recover_failed_postprocessing(
        original,
        timestamp="2026-09-30T01:00:00Z",
        note="Recovered official grading from preserved predictions with pinned interpreter.",
    )

    assert original == _failed_status(), "input status must not be mutated"
    assert recovered["execution_state"] == "completed"
    assert recovered["lifecycle"] == "current"
    assert [entry["state"] for entry in recovered["history"]] == [
        "planned",
        "preflight_passed",
        "running",
        "failed",
        "completed",
    ]
    assert recovered["history"][-1]["recovery"] is True
    assert "preserved predictions" in recovered["history"][-1]["note"]
    assert validate_campaign.validate_status_consistency(recovered) == []


def test_recovery_requires_failed_invalid_status():
    status = _failed_status()
    status["lifecycle"] = "diagnostic"
    with pytest.raises(campaign_state.InvalidTransitionError):
        campaign_state.recover_failed_postprocessing(
            status,
            timestamp="2026-09-30T01:00:00Z",
            note="Recovery",
        )


def test_recovery_requires_nonempty_note_and_monotonic_timestamp():
    with pytest.raises(campaign_state.InvalidTransitionError):
        campaign_state.recover_failed_postprocessing(
            _failed_status(),
            timestamp="2026-09-30T01:00:00Z",
            note="",
        )
    with pytest.raises(campaign_state.InvalidTimestampError):
        campaign_state.recover_failed_postprocessing(
            _failed_status(),
            timestamp="2026-09-29T23:59:59Z",
            note="Recovery",
        )


def test_unmarked_failed_to_completed_history_remains_invalid():
    status = copy.deepcopy(_failed_status())
    status["execution_state"] = "completed"
    status["lifecycle"] = "current"
    status["history"].append(
        {"state": "completed", "timestamp": "2026-09-30T01:00:00Z"}
    )

    errors = validate_campaign.validate_status_consistency(status)
    assert any("failed" in error and "completed" in error for error in errors)
