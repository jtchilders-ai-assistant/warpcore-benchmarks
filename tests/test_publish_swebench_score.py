"""Regression tests for SWE-bench canonical publication scores."""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

_TESTS_DIR = pathlib.Path(__file__).parent
_VIZ_DIR = _TESTS_DIR.parent / "viz"
if str(_VIZ_DIR) not in sys.path:
    sys.path.insert(0, str(_VIZ_DIR))

import publish_campaign  # noqa: E402


def test_load_item_scores_uses_full_swebench_denominator(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "grading_results.json").write_text(json.dumps({
        "resolved_ids": ["a", "b"],
        "unresolved_ids": ["c"],
        "empty_patch_ids": ["d"],
        "error_ids": ["e"],
        "incomplete_ids": [],
    }))

    scores = publish_campaign._load_item_scores(tmp_path, benchmark="swebench")

    assert scores == {"a": 1.0, "b": 1.0, "c": 0.0, "d": 0.0, "e": 0.0}


def test_swebench_matrix_entry_has_score_and_all_item_ids(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "grading_results.json").write_text(json.dumps({
        "resolved_ids": ["a", "b"],
        "unresolved_ids": ["c"],
        "empty_patch_ids": ["d"],
        "error_ids": ["e"],
        "incomplete_ids": [],
    }))
    runs = [{
        "manifest": {
            "benchmark": "swebench", "suite_id": "warpcore-v1", "run_id": "run-1",
            "model": {"slug": "model-a"}, "item_inventory": {"expected": 5},
        },
        "status": {"lifecycle": "current", "execution_state": "validated"},
        "run_info": {"run_dir": tmp_path, "model_slug": "model-a"},
    }]

    entry = publish_campaign._build_entries(runs)[0]

    assert entry["score"] == 0.4
    assert entry["item_ids"] == ["a", "b", "c", "d", "e"]


def test_swebench_matrix_entry_rejects_incomplete_score_inventory(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "grading_results.json").write_text(json.dumps({
        "resolved_ids": ["a"],
        "unresolved_ids": ["b"],
        "empty_patch_ids": [],
        "error_ids": [],
        "incomplete_ids": [],
    }))
    runs = [{
        "manifest": {
            "benchmark": "swebench", "suite_id": "warpcore-v1", "run_id": "run-1",
            "model": {"slug": "model-a"}, "item_inventory": {"expected": 3},
        },
        "status": {"lifecycle": "current", "execution_state": "validated"},
        "run_info": {"run_dir": tmp_path, "model_slug": "model-a"},
    }]

    with pytest.raises(ValueError, match="score inventory has 2 items; expected 3"):
        publish_campaign._build_entries(runs)


def test_matrix_entry_rejects_missing_score_artifact(tmp_path):
    runs = [{
        "manifest": {
            "benchmark": "swebench", "suite_id": "warpcore-v1", "run_id": "run-1",
            "model": {"slug": "model-a"}, "item_inventory": {"expected": 3},
        },
        "status": {"lifecycle": "current", "execution_state": "validated"},
        "run_info": {"run_dir": tmp_path, "model_slug": "model-a"},
    }]

    with pytest.raises(ValueError, match="score evidence is missing or malformed"):
        publish_campaign._build_entries(runs)
