"""Evidence and publication guards for the Qwen3.5 warpcore-v1 campaign."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
MODEL_ROOT = REPO / "results" / "qwen3.5-122b-a10b"
QUAL_ROOT = MODEL_ROOT / "qualification" / "warpcore-v1" / "swebench"
FINAL_RUN = QUAL_ROOT / "run-20260927e"
DECISION = QUAL_ROOT / "qualification_decision.json"
TRIAL_RUN = (
    MODEL_ROOT
    / "runs"
    / "warpcore-v1"
    / "swebench"
    / "qwen35-swebench-trial-n100-20260928"
)


def test_failed_qualification_decision_reconciles_retained_evidence():
    decision = json.loads(DECISION.read_text())
    statuses = json.loads((FINAL_RUN / "raw" / "exit_statuses.json").read_text())
    grading = json.loads((FINAL_RUN / "raw" / "grading_results.json").read_text())

    assert decision["schema_version"] == 1
    assert decision["suite_id"] == "warpcore-v1"
    assert decision["model_id"] == "Intel/Qwen3.5-122B-A10B-int4-AutoRound"
    assert decision["qualification_run"] == "run-20260927e"
    assert decision["qualified"] is False
    assert decision["canonical_n100_launched"] is False
    assert decision["qualification_artifact_written"] is False
    assert decision["required_terminal_status"] == "Submitted"
    assert decision["required_submitted"] == 20
    assert decision["observed_instances"] == 20
    assert decision["generation_dispositions"] == dict(sorted(Counter(statuses.values()).items()))
    assert decision["non_submitted"] == {
        iid: statuses[iid] for iid in sorted(statuses) if statuses[iid] != "Submitted"
    }
    assert decision["official_grading"] == {
        key: len(grading[key])
        for key in (
            "resolved_ids",
            "unresolved_ids",
            "empty_patch_ids",
            "error_ids",
            "incomplete_ids",
        )
    }
    assert decision["publishable_swebench_score"] is False
    assert decision["noncanonical_n100_trial_launched"] is True
    assert decision["noncanonical_n100_trial"]["run_id"] == TRIAL_RUN.name
    assert decision["noncanonical_n100_trial"]["lifecycle"] == "diagnostic"
    assert decision["noncanonical_n100_trial"]["resolved"] == 57
    assert decision["noncanonical_n100_trial"]["unresolved"] == 19
    assert decision["noncanonical_n100_trial"]["empty_patch"] == 24
    assert "qualification" in decision["decision"].lower()
    assert not (QUAL_ROOT / "qualification.json").exists()


def test_quality_registry_entries_are_current_and_bound_to_validated_runs():
    registry = json.loads((REPO / "results" / "registry.json").read_text())
    entries = {entry["id"]: entry for entry in registry["entries"]}
    expected = {
        "qwen3.5-122b-a10b/gsm8k/warpcore-v1/qwen35-gsm8k-20260926":
            "results/qwen3.5-122b-a10b/runs/warpcore-v1/gsm8k/qwen35-gsm8k-20260926",
        "qwen3.5-122b-a10b/ifeval/warpcore-v1/qwen35-ifeval-20260925b":
            "results/qwen3.5-122b-a10b/runs/warpcore-v1/ifeval/qwen35-ifeval-20260925b",
        "qwen3.5-122b-a10b/gpqa-diamond/warpcore-v1/qwen35-gpqa-20260924":
            "results/qwen3.5-122b-a10b/runs/warpcore-v1/gpqa_diamond/qwen35-gpqa-20260924",
    }
    for entry_id, run_dir in expected.items():
        entry = entries[entry_id]
        assert entry["status"] == "current"
        assert entry["v1_validated"] is True
        assert entry["run_dir"] == run_dir
        status = json.loads((REPO / run_dir / "status.json").read_text())
        assert status["execution_state"] == "validated"
        assert status["lifecycle"] == "current"


def test_failed_qualification_is_registered_diagnostic_not_current():
    registry = json.loads((REPO / "results" / "registry.json").read_text())
    entries = {entry["id"]: entry for entry in registry["entries"]}
    entry = entries["qwen3.5-122b-a10b/swebench/warpcore-v1/qualification-20260927e"]
    assert entry["status"] == "diagnostic"
    assert entry.get("v1_validated") is not True
    assert "qualification_decision.json" in "\n".join(entry["paths"])
    assert "qualification.json" not in "\n".join(entry["paths"])


def test_noncanonical_n100_trial_reconciles_and_stays_diagnostic():
    manifest = json.loads((TRIAL_RUN / "manifest.json").read_text())
    status = json.loads((TRIAL_RUN / "status.json").read_text())
    statuses = json.loads((TRIAL_RUN / "raw" / "exit_statuses.json").read_text())
    grading = json.loads((TRIAL_RUN / "raw" / "grading_results.json").read_text())
    predictions = json.loads((TRIAL_RUN / "raw" / "preds.json").read_text())

    assert (TRIAL_RUN / "DONE").read_text().strip() == "completed"
    assert status["execution_state"] == "completed"
    assert status["lifecycle"] == "diagnostic"
    assert manifest["item_inventory"]["expected"] == 100
    assert manifest["item_inventory"]["submitted"] == 100
    assert len(statuses) == len(predictions) == 100
    assert Counter(statuses.values()) == {
        "Submitted": 76,
        "LimitsExceeded": 21,
        "Timeout": 2,
        "ContextWindowExceededError": 1,
    }
    assert {key: len(grading[key]) for key in grading} == {
        "resolved_ids": 57,
        "unresolved_ids": 19,
        "empty_patch_ids": 24,
        "error_ids": 0,
        "incomplete_ids": 0,
    }
    non_submitted = {iid for iid, disposition in statuses.items() if disposition != "Submitted"}
    assert non_submitted == set(grading["empty_patch_ids"])
    assert all(predictions[iid]["model_patch"] == "" for iid in non_submitted)
    assert all(predictions[iid]["model_patch"] for iid in statuses if iid not in non_submitted)
    assert len(list((TRIAL_RUN / "raw" / "trajectories").glob("*.traj"))) == 100
    required_provenance = (
        "command.txt",
        "launch_swebench_trial_n100_20260928.sh",
        "suite_input_snapshots/adapters/qwen3.5-122b-a10b.yaml",
        "suite_input_snapshots/suite/warpcore-v1.yaml",
        "suite_input_snapshots/suite/swebench/instances-seed42-n100.json",
        "suite_input_snapshots/suite/swebench/scaffold.yaml",
    )
    assert all((TRIAL_RUN / rel).is_file() for rel in required_provenance)
    assert not (QUAL_ROOT / "qualification.json").exists()


def test_noncanonical_n100_trial_is_registered_but_not_publishable():
    registry = json.loads((REPO / "results" / "registry.json").read_text())
    entries = {entry["id"]: entry for entry in registry["entries"]}
    entry = entries[
        "qwen3.5-122b-a10b/swebench/warpcore-v1/"
        "qwen35-swebench-trial-n100-20260928"
    ]
    assert entry["status"] == "diagnostic"
    assert entry.get("v1_validated") is not True
    assert entry["run_dir"] == str(TRIAL_RUN.relative_to(REPO))


def test_publication_surfaces_label_n100_result_diagnostic():
    root = (REPO / "README.md").read_text()
    card = (MODEL_ROOT / "README.md").read_text()
    for text in (root, card):
        assert "57/100" in text
        assert "diagnostic" in text.lower()
        assert "76" in text
        assert "24" in text
        assert "not canonical" in text.lower()


def test_all_retained_qualification_attempts_are_registry_classified():
    registry = json.loads((REPO / "results" / "registry.json").read_text())
    entries = {entry["id"]: entry for entry in registry["entries"]}
    expected = {
        "run": ("qualification-run", "diagnostic"),
        "run-20260927b": ("qualification-run-20260927b", "diagnostic"),
        "run-20260927c": ("qualification-run-20260927c", "invalid"),
        "run-20260927d": ("qualification-run-20260927d", "diagnostic"),
        "run-20260927e": ("qualification-20260927e", "diagnostic"),
    }
    for run_name, (id_suffix, lifecycle) in expected.items():
        entry_id = f"qwen3.5-122b-a10b/swebench/warpcore-v1/{id_suffix}"
        assert entries[entry_id]["status"] == lifecycle
        assert entries[entry_id].get("v1_validated") is not True
        assert (QUAL_ROOT / run_name).is_dir()


def test_provenance_audit_accepts_normalized_json_exit_statuses():
    import sys

    viz = str(REPO / "viz")
    if viz not in sys.path:
        sys.path.insert(0, viz)
    audit_provenance = __import__("audit_provenance")

    row = audit_provenance.audit_model(MODEL_ROOT)
    assert row["swebench_run"] is True
    assert row["swe_exit_statuses"] is True
    assert row["swe_preds"] is True
    assert row["swe_results"] is True
    assert row["swe_trajectories"] is True


def test_provenance_audit_does_not_combine_partial_normalized_runs(tmp_path, monkeypatch):
    import sys

    viz = str(REPO / "viz")
    if viz not in sys.path:
        sys.path.insert(0, viz)
    audit_provenance = __import__("audit_provenance")

    model = tmp_path / "results" / "model"
    model.mkdir(parents=True)
    (model / "README.md").write_text("# model\nSWE-bench 1/1\n")
    first = tmp_path / "first"
    second = tmp_path / "second"
    for run_dir in (first, second):
        (run_dir / "raw").mkdir(parents=True)
        (run_dir / "manifest.json").write_text('{"benchmark": "swebench"}\n')
    (first / "raw" / "preds.json").write_text("{}\n")
    (first / "raw" / "grading_results.json").write_text("{}\n")
    (second / "raw" / "exit_statuses.json").write_text("{}\n")
    (second / "raw" / "trajectories").mkdir()
    (second / "raw" / "trajectories" / "one.traj").write_text("{}\n")

    monkeypatch.setattr(
        audit_provenance,
        "discover_runs",
        lambda _repo: [
            {
                "layout": "normalized",
                "model_slug": "model",
                "manifest_path": str(first / "manifest.json"),
            },
            {
                "layout": "normalized",
                "model_slug": "model",
                "manifest_path": str(second / "manifest.json"),
            },
        ],
    )
    row = audit_provenance.audit_model(model)
    assert not all(
        row.get(key, False)
        for key in ("swe_exit_statuses", "swe_preds", "swe_results", "swe_trajectories")
    )


def test_provenance_baseline_drops_repaired_qwen_manifest_gap():
    baseline = json.loads((REPO / "viz" / "data" / "provenance_baseline.json").read_text())
    assert "qwen3.5-122b-a10b: NO manifest.json anywhere under raw/" not in baseline["accepted_gaps"]


def test_model_card_discloses_quality_budget_exhaustions():
    card = (MODEL_ROOT / "README.md").read_text()
    gsm8k = card.split("- **GSM8K:**", 1)[1].split("- **IFEval:**", 1)[0]
    ifeval = card.split("- **IFEval:**", 1)[1].split("- **GPQA-Diamond:**", 1)[0]
    gpqa = card.split("- **GPQA-Diamond:**", 1)[1].split("- **SWE-bench", 1)[0]
    assert "Fourteen empty responses" in gsm8k
    assert "8,192-token output ceiling" in gsm8k
    assert "Thirty-eight responses exhausted" in ifeval
    assert "65,536-token" in ifeval
    assert "Four responses exhausted" in gpqa
    assert "65,536-token" in gpqa
