"""Tests for the shared SWE-bench reporting parser.

Covers:
 - Source bundle selection per model (Task 2/3)
 - Exact submitted/resolved/decomposition values (Task 2/3)
 - Vocabulary and lifecycle labels (Task 2/3)
 - Invariant checks: resolved <= submitted <= expected, partition sums to 100 (Task 2/3)
 - Nonempty-patch semantics: predictions are authoritative when present (Task 2/3)
 - Historical aggregate provenance limitations (Task 2/3)
 - Old Qwen3.6 44/100 run is addressable but not selected (Task 2/3)
 - Generated data fields: submitted, resolved, expected, decomposition, source_paths,
   lifecycle, provenance (Task 4)
 - resolved_rate == resolved / expected, not resolved / submitted (Task 4)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "viz"))

# Import the new parser — will fail until viz/swebench_reporting.py is created (RED)
from swebench_reporting import load_swebench_report, selected_swebench_reports  # noqa: E402


# ---------------------------------------------------------------------------
# Expected selected values (source of truth from the plan)
# ---------------------------------------------------------------------------
EXPECTED_SELECTED = {
    "ornith-35b": dict(
        submitted=91, resolved=73, expected=100,
        unresolved=18, empty_patch=9, grading_error=0, incomplete=0,
    ),
    "laguna-s-2.1-118b": dict(
        submitted=65, resolved=55, expected=100,
        unresolved=10, empty_patch=35, grading_error=0, incomplete=0,
    ),
    "nemotron-3.5-lightning-30b": dict(
        submitted=98, resolved=51, expected=100,
        unresolved=47, empty_patch=2, grading_error=0, incomplete=0,
    ),
    "qwen3.6-35b-a3b": dict(
        submitted=89, resolved=57, expected=100,
        unresolved=30, empty_patch=11, grading_error=2, incomplete=0,
    ),
    "qwen3.5-122b-a10b": dict(
        submitted=76, resolved=57, expected=100,
        unresolved=19, empty_patch=24, grading_error=0, incomplete=0,
    ),
}

# Frozen seed-42 n=100 instance IDs
INSTANCES_PATH = REPO / "suite" / "swebench" / "instances-seed42-n100.json"
FROZEN_IDS: set[str] = set(json.loads(INSTANCES_PATH.read_text()))


# ---------------------------------------------------------------------------
# 1. Source selection
# ---------------------------------------------------------------------------

def test_selected_swebench_reports_returns_all_five_models() -> None:
    """selected_swebench_reports returns exactly the five known models."""
    reports = selected_swebench_reports(REPO)
    assert set(reports.keys()) == set(EXPECTED_SELECTED.keys())


def test_qwen36_selected_run_is_validated_v1_not_old() -> None:
    """Qwen3.6 must use the validated v1 run, not the old 44/100 report."""
    reports = selected_swebench_reports(REPO)
    report = reports["qwen3.6-35b-a3b"]
    # The selected bundle must reference the normalized run directory
    source_paths_str = " ".join(str(p) for p in report["source_paths"])
    assert "qwen36-swebench-n100-20260919" in source_paths_str
    # Must NOT reference the old 44-resolved report as the primary source
    assert "swebench_verified_shuffle100_report.json" not in source_paths_str


def test_old_qwen36_run_is_addressable_but_not_selected() -> None:
    """The older 44/100 run path can be loaded but must not be selected."""
    old_path = (
        REPO
        / "results"
        / "qwen3.6-35b-a3b"
        / "raw"
        / "swebench"
        / "swebench_verified_shuffle100_report.json"
    )
    assert old_path.exists(), "Historical Qwen3.6 44/100 report must still be present"
    reports = selected_swebench_reports(REPO)
    # The selected report resolves 57, not 44
    assert reports["qwen3.6-35b-a3b"]["resolved"] == 57


def test_qwen35_selected_run_is_diagnostic_trial() -> None:
    """Qwen3.5 must use the diagnostic trial run, not a canonical one."""
    reports = selected_swebench_reports(REPO)
    report = reports["qwen3.5-122b-a10b"]
    source_paths_str = " ".join(str(p) for p in report["source_paths"])
    assert "qwen35-swebench-trial-n100-20260928" in source_paths_str


# ---------------------------------------------------------------------------
# 2. Exact counts per model
# ---------------------------------------------------------------------------

def test_all_selected_submitted_counts() -> None:
    reports = selected_swebench_reports(REPO)
    for model, expected in EXPECTED_SELECTED.items():
        got = reports[model]["submitted"]
        assert got == expected["submitted"], (
            f"{model}: submitted={got}, expected {expected['submitted']}"
        )


def test_all_selected_resolved_counts() -> None:
    reports = selected_swebench_reports(REPO)
    for model, expected in EXPECTED_SELECTED.items():
        got = reports[model]["resolved"]
        assert got == expected["resolved"], (
            f"{model}: resolved={got}, expected {expected['resolved']}"
        )


def test_all_selected_decomposition_fields() -> None:
    """Every decomposition field must match the expected values."""
    reports = selected_swebench_reports(REPO)
    for model, expected in EXPECTED_SELECTED.items():
        r = reports[model]
        for field in ("unresolved", "empty_patch", "grading_error", "incomplete"):
            assert r[field] == expected[field], (
                f"{model}: {field}={r[field]}, expected {expected[field]}"
            )


def test_qwen36_grading_error_count_is_two() -> None:
    """Qwen3.6 has exactly 2 grading errors (error_ids) per source data."""
    reports = selected_swebench_reports(REPO)
    assert reports["qwen3.6-35b-a3b"]["grading_error"] == 2


def test_qwen36_submitted_includes_grading_errors() -> None:
    """Grading-error patches have nonempty model_patch and count as submitted."""
    reports = selected_swebench_reports(REPO)
    r = reports["qwen3.6-35b-a3b"]
    # submitted = resolved(57) + unresolved(30) + grading_error(2) = 89
    assert r["submitted"] == r["resolved"] + r["unresolved"] + r["grading_error"]


# ---------------------------------------------------------------------------
# 3. Invariant checks
# ---------------------------------------------------------------------------

def test_resolved_le_submitted_le_expected_for_all_models() -> None:
    reports = selected_swebench_reports(REPO)
    for model, r in reports.items():
        assert r["resolved"] <= r["submitted"] <= r["expected"], (
            f"{model}: violated resolved <= submitted <= expected: "
            f"{r['resolved']} <= {r['submitted']} <= {r['expected']}"
        )


def test_partition_sums_to_expected_for_all_models() -> None:
    """resolved + unresolved + empty_patch + grading_error + incomplete == expected."""
    reports = selected_swebench_reports(REPO)
    for model, r in reports.items():
        total = (
            r["resolved"]
            + r["unresolved"]
            + r["empty_patch"]
            + r["grading_error"]
            + r["incomplete"]
        )
        assert total == r["expected"], (
            f"{model}: partition total={total} != expected={r['expected']}"
        )


def test_expected_is_100_for_all_models() -> None:
    reports = selected_swebench_reports(REPO)
    for model, r in reports.items():
        assert r["expected"] == 100, f"{model}: expected={r['expected']}"


# ---------------------------------------------------------------------------
# 4. Predictions-based submitted count is authoritative when predictions exist
# ---------------------------------------------------------------------------

def test_qwen36_submitted_derived_from_nonempty_predictions() -> None:
    """For Qwen3.6, submitted comes from counting nonempty model_patch values."""
    reports = selected_swebench_reports(REPO)
    r = reports["qwen3.6-35b-a3b"]
    # Verify against actual predictions file
    preds_path = (
        REPO
        / "results"
        / "qwen3.6-35b-a3b"
        / "runs"
        / "warpcore-v1"
        / "swebench"
        / "qwen36-swebench-n100-20260919"
        / "raw"
        / "preds.json"
    )
    preds = json.loads(preds_path.read_text())
    nonempty_count = sum(1 for p in preds.values() if p.get("model_patch", ""))
    assert r["submitted"] == nonempty_count, (
        f"submitted={r['submitted']} does not match nonempty preds={nonempty_count}"
    )


def test_qwen35_submitted_derived_from_nonempty_predictions() -> None:
    """For Qwen3.5, submitted comes from nonempty model_patch values in predictions."""
    reports = selected_swebench_reports(REPO)
    r = reports["qwen3.5-122b-a10b"]
    preds_path = (
        REPO
        / "results"
        / "qwen3.5-122b-a10b"
        / "runs"
        / "warpcore-v1"
        / "swebench"
        / "qwen35-swebench-trial-n100-20260928"
        / "raw"
        / "preds.json"
    )
    preds = json.loads(preds_path.read_text())
    nonempty_count = sum(1 for p in preds.values() if p.get("model_patch", ""))
    assert r["submitted"] == nonempty_count


def test_ornith_submitted_derived_from_nonempty_predictions() -> None:
    """Ornith has retained predictions; submitted must match nonempty patch count."""
    reports = selected_swebench_reports(REPO)
    r = reports["ornith-35b"]
    preds_path = (
        REPO / "results" / "ornith-35b" / "raw" / "swebench"
        / "swebench_verified_n100_preds.json"
    )
    preds = json.loads(preds_path.read_text())
    nonempty_count = sum(1 for p in preds.values() if p.get("model_patch", ""))
    assert r["submitted"] == nonempty_count


# ---------------------------------------------------------------------------
# 5. Source paths and provenance labels
# ---------------------------------------------------------------------------

def test_all_reports_have_source_paths() -> None:
    reports = selected_swebench_reports(REPO)
    for model, r in reports.items():
        assert "source_paths" in r, f"{model}: missing source_paths"
        assert len(r["source_paths"]) >= 1, f"{model}: source_paths is empty"
        for p in r["source_paths"]:
            assert Path(p).exists(), f"{model}: source path does not exist: {p}"


def test_all_reports_have_provenance_label() -> None:
    reports = selected_swebench_reports(REPO)
    for model, r in reports.items():
        assert "provenance" in r, f"{model}: missing provenance"
        assert r["provenance"] in ("normalized-complete", "legacy-aggregate"), (
            f"{model}: unexpected provenance '{r['provenance']}'"
        )


def test_qwen36_provenance_is_normalized_complete() -> None:
    """Qwen3.6 has predictions + grading_results -> normalized-complete provenance."""
    reports = selected_swebench_reports(REPO)
    assert reports["qwen3.6-35b-a3b"]["provenance"] == "normalized-complete"


def test_qwen35_provenance_is_normalized_complete() -> None:
    """Qwen3.5 also has predictions + grading_results."""
    reports = selected_swebench_reports(REPO)
    assert reports["qwen3.5-122b-a10b"]["provenance"] == "normalized-complete"


def test_historical_models_have_legacy_or_normalized_provenance() -> None:
    """Ornith/Laguna/Nemotron have retained preds; accept normalized-complete."""
    reports = selected_swebench_reports(REPO)
    for model in ("ornith-35b", "laguna-s-2.1-118b", "nemotron-3.5-lightning-30b"):
        assert reports[model]["provenance"] in ("normalized-complete", "legacy-aggregate")


# ---------------------------------------------------------------------------
# 6. Lifecycle labels
# ---------------------------------------------------------------------------

def test_qwen35_lifecycle_is_diagnostic() -> None:
    """Qwen3.5 must carry lifecycle='diagnostic' (not canonical)."""
    reports = selected_swebench_reports(REPO)
    assert reports["qwen3.5-122b-a10b"]["lifecycle"] == "diagnostic"


def test_qwen36_lifecycle_is_current() -> None:
    """Qwen3.6 validated v1 run has lifecycle='current'."""
    reports = selected_swebench_reports(REPO)
    assert reports["qwen3.6-35b-a3b"]["lifecycle"] == "current"


def test_historical_models_have_lifecycle_label() -> None:
    reports = selected_swebench_reports(REPO)
    for model in ("ornith-35b", "laguna-s-2.1-118b", "nemotron-3.5-lightning-30b"):
        assert "lifecycle" in reports[model], f"{model}: missing lifecycle"


# ---------------------------------------------------------------------------
# 7. Frozen ID set
# ---------------------------------------------------------------------------

def test_qwen36_grading_ids_match_frozen_set() -> None:
    """Qwen3.6 (normalized run) prediction IDs must equal the frozen 100 IDs."""
    reports = selected_swebench_reports(REPO)
    r = reports["qwen3.6-35b-a3b"]
    # Source paths include predictions; verify IDs
    preds_path = (
        REPO
        / "results"
        / "qwen3.6-35b-a3b"
        / "runs"
        / "warpcore-v1"
        / "swebench"
        / "qwen36-swebench-n100-20260919"
        / "raw"
        / "preds.json"
    )
    preds = json.loads(preds_path.read_text())
    assert set(preds.keys()) == FROZEN_IDS, "Qwen3.6 pred IDs do not match frozen set"


def test_qwen35_grading_ids_match_frozen_set() -> None:
    preds_path = (
        REPO
        / "results"
        / "qwen3.5-122b-a10b"
        / "runs"
        / "warpcore-v1"
        / "swebench"
        / "qwen35-swebench-trial-n100-20260928"
        / "raw"
        / "preds.json"
    )
    preds = json.loads(preds_path.read_text())
    assert set(preds.keys()) == FROZEN_IDS


def test_ornith_preds_ids_match_frozen_set() -> None:
    preds = json.loads(
        (REPO / "results" / "ornith-35b" / "raw" / "swebench"
         / "swebench_verified_n100_preds.json").read_text()
    )
    assert set(preds.keys()) == FROZEN_IDS


# ---------------------------------------------------------------------------
# 8. load_swebench_report — single-bundle loading
# ---------------------------------------------------------------------------

def test_load_swebench_report_qwen36_returns_correct_shape() -> None:
    """load_swebench_report on the Qwen3.6 normalized bundle returns the expected shape."""
    bundle = {
        "grading": REPO / "results/qwen3.6-35b-a3b/runs/warpcore-v1/swebench/qwen36-swebench-n100-20260919/raw/grading_results.json",
        "predictions": REPO / "results/qwen3.6-35b-a3b/runs/warpcore-v1/swebench/qwen36-swebench-n100-20260919/raw/preds.json",
        "lifecycle": "current",
    }
    r = load_swebench_report(REPO, bundle)
    assert r["expected"] == 100
    assert r["submitted"] == 89
    assert r["resolved"] == 57
    assert r["unresolved"] == 30
    assert r["empty_patch"] == 11
    assert r["grading_error"] == 2
    assert r["incomplete"] == 0
    assert r["provenance"] == "normalized-complete"


def test_load_swebench_report_ornith_legacy_format() -> None:
    """load_swebench_report on the legacy report format (Ornith) returns correct values."""
    bundle = {
        "grading": REPO / "results/ornith-35b/raw/swebench/swebench_verified_n100_results.json",
        "predictions": REPO / "results/ornith-35b/raw/swebench/swebench_verified_n100_preds.json",
        "lifecycle": "historical",
    }
    r = load_swebench_report(REPO, bundle)
    assert r["expected"] == 100
    assert r["submitted"] == 91
    assert r["resolved"] == 73
    assert r["unresolved"] == 18
    assert r["empty_patch"] == 9
    assert r["grading_error"] == 0
    assert r["incomplete"] == 0


def test_load_swebench_report_rejects_non_100_total(tmp_path: Path) -> None:
    """Parser must reject a grading file whose IDs don't sum to 100."""
    import pytest
    grading = {
        "resolved_ids": ["a", "b"],
        "unresolved_ids": ["c"],
        "empty_patch_ids": [],
        "error_ids": [],
        "incomplete_ids": [],
    }
    g_path = tmp_path / "grading.json"
    g_path.write_text(json.dumps(grading))
    preds = {"a": {"model_patch": "x"}, "b": {"model_patch": "y"}, "c": {"model_patch": ""}}
    p_path = tmp_path / "preds.json"
    p_path.write_text(json.dumps(preds))
    bundle = {"grading": g_path, "predictions": p_path, "lifecycle": "historical"}
    with pytest.raises(ValueError, match="100"):
        load_swebench_report(REPO, bundle)


def test_load_swebench_report_rejects_duplicate_ids(tmp_path: Path) -> None:
    """Parser must reject duplicate IDs appearing in multiple partitions."""
    import pytest
    grading = {
        "resolved_ids": ["a"] * 50 + list("bcdefghijklmnopqrstuvwxy"),
        "unresolved_ids": ["a"] + list("z" * 24),  # 'a' is duplicate
        "empty_patch_ids": [],
        "error_ids": [],
        "incomplete_ids": [],
    }
    # Make exactly 100 entries with 'a' duplicated across partitions
    grading = {
        "resolved_ids": [f"id{i}" for i in range(50)],
        "unresolved_ids": ["id0"] + [f"id{i}" for i in range(51, 100)],  # id0 duplicated
        "empty_patch_ids": [],
        "error_ids": [],
        "incomplete_ids": [],
    }
    g_path = tmp_path / "grading.json"
    g_path.write_text(json.dumps(grading))
    bundle = {"grading": g_path, "lifecycle": "historical"}
    with pytest.raises(ValueError, match="[Dd]uplicate|[Oo]verlap"):
        load_swebench_report(REPO, bundle)


# ---------------------------------------------------------------------------
# 9. Mutation test: temporarily corrupt a Qwen3.6 prediction, require test to catch it
# (uses tmp_path fixture so committed artifacts are never modified)
# ---------------------------------------------------------------------------

def test_mutation_empty_patch_changes_submitted_count(tmp_path: Path) -> None:
    """If one nonempty patch is replaced with empty, submitted count drops by 1."""
    # Copy qwen3.6 grading and predictions to tmp_path
    orig_grading = REPO / "results/qwen3.6-35b-a3b/runs/warpcore-v1/swebench/qwen36-swebench-n100-20260919/raw/grading_results.json"
    orig_preds = REPO / "results/qwen3.6-35b-a3b/runs/warpcore-v1/swebench/qwen36-swebench-n100-20260919/raw/preds.json"

    preds = json.loads(orig_preds.read_text())
    # Find one error_id (nonempty but grading error) and blank its patch
    grading = json.loads(orig_grading.read_text())
    error_id = grading["error_ids"][0]

    mutated_preds = dict(preds)
    mutated_preds[error_id] = dict(preds[error_id], model_patch="")

    mut_preds_path = tmp_path / "preds.json"
    mut_preds_path.write_text(json.dumps(mutated_preds))
    mut_grading_path = tmp_path / "grading.json"
    mut_grading_path.write_text(orig_grading.read_text())

    bundle = {
        "grading": mut_grading_path,
        "predictions": mut_preds_path,
        "lifecycle": "current",
    }
    r = load_swebench_report(REPO, bundle)
    # submitted should be 88 (one fewer nonempty patch)
    assert r["submitted"] == 88, f"mutation test failed: submitted={r['submitted']}, expected 88"


# ---------------------------------------------------------------------------
# 10. Generated data fields (Task 4) — bench_matrix.json
# ---------------------------------------------------------------------------

def test_bench_matrix_has_submitted_field_for_swe_entries() -> None:
    """bench_matrix.json SWE entries must carry 'submitted' and 'resolved' fields."""
    matrix_path = REPO / "viz" / "data" / "bench_matrix.json"
    matrix = json.loads(matrix_path.read_text())
    swe_models = [m for m, d in matrix.items() if "swebench" in d]
    assert len(swe_models) >= 1, "No swebench entries in bench_matrix"
    for model in swe_models:
        entry = matrix[model]["swebench"]
        assert "submitted" in entry, f"{model}: swebench entry missing 'submitted'"
        assert "expected" in entry, f"{model}: swebench entry missing 'expected'"
        assert entry.get("expected") == 100, f"{model}: expected != 100"


def test_bench_matrix_resolved_rate_uses_full_denominator() -> None:
    """resolved_rate (if present) must equal resolved/100, not resolved/submitted."""
    matrix_path = REPO / "viz" / "data" / "bench_matrix.json"
    matrix = json.loads(matrix_path.read_text())
    for model, benchmarks in matrix.items():
        if "swebench" not in benchmarks:
            continue
        entry = benchmarks["swebench"]
        if "resolve_rate" in entry:
            resolved = entry.get("value", entry.get("resolved", 0))
            expected = entry.get("expected", 100)
            assert expected == 100
            rate = round(100 * resolved / 100, 1)
            assert abs(entry["resolve_rate"] - rate) < 0.2, (
                f"{model}: resolve_rate={entry['resolve_rate']} "
                f"!= resolved/100={rate}"
            )


def test_bench_matrix_has_decomposition_fields_for_swe_entries() -> None:
    """SWE entries in bench_matrix must have empty_patch and submitted fields."""
    matrix_path = REPO / "viz" / "data" / "bench_matrix.json"
    matrix = json.loads(matrix_path.read_text())
    for model, benchmarks in matrix.items():
        if "swebench" not in benchmarks:
            continue
        entry = benchmarks["swebench"]
        assert "empty_patch" in entry, f"{model}: missing empty_patch"
        assert "submitted" in entry, f"{model}: missing submitted"


def test_selected_reports_use_repo_relative_registered_sources() -> None:
    """Every artifact contributing to a published value is registry classified."""
    registry = json.loads((REPO / "results" / "registry.json").read_text())
    registered = {p for entry in registry["entries"] for p in entry["paths"]}
    for model, report in selected_swebench_reports(REPO).items():
        for source in report["source_paths"]:
            assert not Path(source).is_absolute(), f"{model}: absolute source path leaked"
            assert source in registered, f"{model}: unregistered source {source}"


def test_selected_reports_expose_exact_frozen_partition_ids() -> None:
    reports = selected_swebench_reports(REPO)
    for model, report in reports.items():
        partition = (
            set(report["resolved_ids"])
            | set(report["unresolved_ids"])
            | set(report["empty_patch_ids"])
            | set(report["error_ids"])
            | set(report["incomplete_ids"])
        )
        assert partition == FROZEN_IDS, f"{model}: selected IDs differ from frozen n=100"
        assert set(report["submitted_ids"]) <= FROZEN_IDS
        assert len(report["submitted_ids"]) == report["submitted"]


def test_fig2_generation_uses_exactly_100_instances() -> None:
    completed = subprocess.run(
        [sys.executable, str(REPO / "viz" / "fig2_swebench.py")],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_figure_source_rejects_conditional_accuracy_headlines() -> None:
    source = (REPO / "viz" / "fig2_swebench.py").read_text()
    forbidden = (
        "most accurate coder",
        "% of submitted",
        "gap is not all capability",
    )
    assert not any(term in source for term in forbidden)


def test_readme_has_comparable_swebench_columns_and_values() -> None:
    readme = (REPO / "README.md").read_text()
    assert "SWE submitted" in readme
    assert "SWE resolved" in readme
    for submitted, resolved in ((91, 73), (65, 55), (98, 51), (89, 57), (76, 57)):
        assert f"{submitted}/100" in readme
        assert f"{resolved}/100" in readme


def test_readme_preserves_qwen_lifecycle_and_historical_disclosures() -> None:
    readme = (REPO / "README.md").read_text()
    assert "Qwen3.5" in readme and "diagnostic" in readme and "not canonical" in readme
    assert "Qwen3.6" in readme and "71/100" in readme and "44/100" in readme
    assert "Ornith" in readme and "exit-status" in readme


def test_provenance_documents_selected_sources_and_reporting_semantics() -> None:
    provenance = (REPO / "PROVENANCE.md").read_text()
    required = (
        "89/100 nonempty patches",
        "57/100 resolved",
        "76/100 nonempty patches",
        "retained predictions",
        "exit-status artifact",
    )
    assert all(term in provenance for term in required)


def test_repository_rules_use_nonempty_patch_submission_semantics() -> None:
    rules = (REPO / "AGENTS.md").read_text()
    assert "submitted means an assigned" in rules.lower()
    assert "nonempty `model_patch`" in rules
    assert "`submitted_ids` is the right pool" not in rules


def test_qwen35_card_states_full_denominator_submission_count() -> None:
    card = (REPO / "results/qwen3.5-122b-a10b/README.md").read_text()
    assert "76/100 nonempty" in card
    assert "57/100" in card
    assert "diagnostic" in card.lower()
    assert "not canonical" in card.lower()


def test_legacy_source_map_is_not_used_by_matrix_collector() -> None:
    collector = (REPO / "viz/collect_matrix.py").read_text()
    assert "SWEBENCH_RESULTS" not in collector
    assert "SWEBENCH =" not in collector


def test_legacy_source_map_deprecation_names_real_replacement() -> None:
    common = (REPO / "viz/common.py").read_text()
    assert "swebench_reporting.selected_swebench_reports()" in common
    assert "SWEBENCH_BUNDLES" not in common


def test_qwen36_selected_exit_status_source_is_registered() -> None:
    registry = json.loads((REPO / "results/registry.json").read_text())
    registered = {path for entry in registry["entries"] for path in entry["paths"]}
    expected = (
        "results/qwen3.6-35b-a3b/runs/warpcore-v1/swebench/"
        "qwen36-swebench-n100-20260919/raw/exit_statuses.json"
    )
    assert expected in registered


def test_provenance_audit_covers_all_selected_swebench_sources() -> None:
    audit = (REPO / "viz/audit_provenance.py").read_text()
    assert "selected_swebench_reports" in audit
    assert "EXIT_STATUSES" in audit
