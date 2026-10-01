"""Production-shaped recovery and SWE submission-semantics regressions."""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parents[1]
_VIZ = _REPO / "viz"
if str(_VIZ) not in sys.path:
    sys.path.insert(0, str(_VIZ))

import validate_campaign  # noqa: E402

RUN_DIR = (
    _REPO / "results" / "ornith-1.5-35b-a3b" / "runs" / "warpcore-v2"
    / "swebench" / "ornith15-swebench-n100-20260929"
)
SUITE = _REPO / "suite" / "warpcore-v2.yaml"
ADAPTER = _REPO / "adapters" / "ornith-1.5-35b-a3b.yaml"


def test_ornith_recovered_campaign_passes_publication_validation() -> None:
    result = validate_campaign.validate(RUN_DIR, SUITE, ADAPTER, for_publication=True)
    assert result.passed, result.errors


def test_swe_manifest_submitted_means_nonempty_model_patches() -> None:
    manifest = json.loads((RUN_DIR / "manifest.json").read_text())
    predictions = json.loads((RUN_DIR / "raw" / "preds.json").read_text())
    nonempty = sum(
        isinstance(row, dict)
        and isinstance(row.get("model_patch", ""), str)
        and bool(row.get("model_patch", "").strip())
        for row in predictions.values()
    )
    assert len(predictions) == manifest["item_inventory"]["expected"] == 100
    assert manifest["item_inventory"]["submitted"] == nonempty == 69


def test_swe_validator_rejects_manifest_nonempty_submission_mismatch(tmp_path) -> None:
    manifest_path = RUN_DIR / "manifest.json"
    original = json.loads(manifest_path.read_text())
    mutated = copy.deepcopy(original)
    mutated["item_inventory"]["submitted"] = 68

    # Exercise the production validator with a temporary manifest, restoring the
    # repository artifact even if validation raises unexpectedly.
    manifest_path.write_text(json.dumps(mutated, indent=2) + "\n")
    try:
        result = validate_campaign.validate(RUN_DIR, SUITE, ADAPTER, for_publication=True)
    finally:
        manifest_path.write_text(json.dumps(original, indent=2) + "\n")

    assert not result.passed
    assert any("submitted" in error and "nonempty" in error for error in result.errors)


def test_recovery_grading_artifacts_have_identical_partitions() -> None:
    normalized = json.loads((RUN_DIR / "raw" / "grading_results.json").read_text())
    recovery = json.loads((
        RUN_DIR / "raw" /
        "hosted_vllm__ornith-ai__Ornith-1.5-35B-A3B-FP8."
        "ornith15-swebench-n100-20260929-recovery.json"
    ).read_text())
    for key in (
        "resolved_ids", "unresolved_ids", "empty_patch_ids", "error_ids", "incomplete_ids"
    ):
        assert set(normalized[key]) == set(recovery[key]), key


def test_recovery_provenance_binds_unchanged_predictions() -> None:
    recovery = (RUN_DIR / "RECOVERY.md").read_text()
    digest = hashlib.sha256((RUN_DIR / "raw" / "preds.json").read_bytes()).hexdigest()
    assert digest in recovery
    assert "59/100" in recovery
    assert "69/100" in recovery
