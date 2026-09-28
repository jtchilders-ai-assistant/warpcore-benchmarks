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
