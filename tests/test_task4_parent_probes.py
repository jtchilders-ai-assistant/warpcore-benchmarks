"""Parent-review regression probes for Task 4 hardening."""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

_TESTS = pathlib.Path(__file__).parent
_VIZ = _TESTS.parent / "viz"
for _p in (str(_TESTS), str(_VIZ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import campaign_state
import create_campaign
from test_task4_hardening import _create, _setup_real_suite_repo


def _maxima() -> dict[str, int]:
    return {"gsm8k": 1, "ifeval": 1, "gpqa_diamond": 1}


def _create_ready(repo, suite, adapter, **kwargs):
    return create_campaign.create_campaign(
        repo=repo,
        suite_path=suite,
        adapter_path=adapter,
        benchmark="gsm8k",
        run_id="run-parent-probe",
        prompt_token_maxima=_maxima(),
        **kwargs,
    )


def test_context_evidence_is_required(tmp_path):
    repo, suite, adapter = _setup_real_suite_repo(tmp_path)
    with pytest.raises(create_campaign.NoncanonicalAdapterError, match="tokenized prompt evidence"):
        create_campaign.create_campaign(
            repo=repo,
            suite_path=suite,
            adapter_path=adapter,
            benchmark="gsm8k",
            run_id="run-no-context",
        )


def test_resume_rejects_extra_suite_hash(tmp_path):
    repo, suite, adapter = _setup_real_suite_repo(tmp_path)
    run_dir = _create_ready(repo, suite, adapter)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["suite_input_hashes"]["suite/extra.yaml"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(create_campaign.ResumeIdentityMismatchError):
        _create_ready(repo, suite, adapter, resume=True)


def test_resume_rejects_missing_status(tmp_path):
    repo, suite, adapter = _setup_real_suite_repo(tmp_path)
    run_dir = _create_ready(repo, suite, adapter)
    (run_dir / "status.json").unlink()
    with pytest.raises(create_campaign.ResumeIdentityMismatchError):
        _create_ready(repo, suite, adapter, resume=True)


def test_resume_rejects_status_identity_mismatch(tmp_path):
    repo, suite, adapter = _setup_real_suite_repo(tmp_path)
    run_dir = _create_ready(repo, suite, adapter)
    status_path = run_dir / "status.json"
    status = json.loads(status_path.read_text())
    status["run_id"] = "different-run"
    status_path.write_text(json.dumps(status))
    with pytest.raises(create_campaign.ResumeIdentityMismatchError):
        _create_ready(repo, suite, adapter, resume=True)


def test_suite_path_outside_repo_rejected(tmp_path):
    repo, suite, adapter = _setup_real_suite_repo(tmp_path / "repo")
    outside = tmp_path / "outside-suite.yaml"
    outside.write_bytes(suite.read_bytes())
    with pytest.raises(create_campaign.SuiteValidationError):
        create_campaign.create_campaign(
            repo=repo,
            suite_path=outside,
            adapter_path=adapter,
            benchmark="gsm8k",
            run_id="run-outside-suite",
            prompt_token_maxima=_maxima(),
        )


def test_entire_history_must_be_legal_and_timestamp_valid():
    skipped = {
        "schema_version": 1,
        "run_id": "run-test",
        "suite_id": "warpcore-v1",
        "execution_state": "running",
        "lifecycle": "current",
        "history": [
            {"state": "planned", "timestamp": "2026-09-15T00:00:00Z"},
            {"state": "running", "timestamp": "2026-09-15T01:00:00Z"},
        ],
    }
    with pytest.raises(campaign_state.InvalidTransitionError, match="history"):
        campaign_state.apply_transition(skipped, "completed", "2026-09-15T02:00:00Z")

    malformed = {
        "schema_version": 1,
        "run_id": "run-test",
        "suite_id": "warpcore-v1",
        "execution_state": "planned",
        "lifecycle": "current",
        "history": [{"state": "planned", "timestamp": "not-a-time"}],
    }
    with pytest.raises(campaign_state.InvalidTimestampError):
        campaign_state.apply_transition(
            malformed, "preflight_passed", "2026-09-15T02:00:00Z"
        )
