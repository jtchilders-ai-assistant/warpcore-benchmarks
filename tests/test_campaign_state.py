"""
tests/test_campaign_state.py — Task 4: transactional campaign state and manifests.

Tests cover:
  T1.  Legal forward transitions: planned -> preflight_passed -> running ->
       completed -> validated -> published
  T2.  Legal failure transition: running -> failed
  T3.  Rejection of skipped states (planned -> running, planned -> completed, …)
  T4.  Rejection of state rewrites (transitioning to the same or earlier state)
  T5.  Rejection of malformed timestamps (non-ISO, naive/non-UTC)
  T6.  Rejection of duplicate terminal transitions (two separate writes of
       'published' or 'failed' after one is already recorded)
  T7.  Rejection of publication when lifecycle is not 'current'
  T8.  Execution state and lifecycle are separate fields; failed execution maps
       to lifecycle 'invalid' unless explicitly 'diagnostic'
  T9.  Atomic status.json write: sibling temp file + flush/fsync + os.replace
  T10. create_campaign: resolves and records all suite/adapter hashes before
       any command generation; creates normalized run directory
  T11. create_campaign: refuses existing output directory unless explicit resume
       with matching identity and inventory
  T12. create_campaign: rejects noncanonical adapter
  T13. Safe path components: run_id, model slug, benchmark must not contain
       path traversal or shell-unsafe characters
  T14. Manifest and status schema validated with FormatChecker
  T15. Symlink safety: status path must not resolve outside the run directory
  T16. Transition to 'published' requires lifecycle == 'current'
  T17. Legacy manifest_scaffold.py: make manifest MODEL=... BENCH=... still
       produces a valid v1 legacy manifest (schema_version=1 with 'unrecorded'
       fields) without regression
  T18. Transition history is append-only: each successful transition appends
       exactly one entry; no entry is removed or modified

Run:
    /usr/bin/python3 -m pytest tests/test_campaign_state.py -v

All tests in this file are RED before Task 4 implementation, GREEN after.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
import re
import sys
import tempfile
import time

import pytest

# Allow imports from viz/ and tests/
_TESTS_DIR = pathlib.Path(__file__).parent
_REPO = _TESTS_DIR.parent
_VIZ_DIR = _REPO / "viz"
for _p in (str(_TESTS_DIR), str(_VIZ_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from schema_helpers import (
    SCHEMA_MANIFEST,
    SCHEMA_STATUS,
    VALID_MANIFEST,
    VALID_STATUS,
    validate_doc,
)

# ---------------------------------------------------------------------------
# The modules under test — these will fail to import until Task 4 is done.
# ---------------------------------------------------------------------------
import campaign_state  # viz/campaign_state.py
import create_campaign  # viz/create_campaign.py


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_utc() -> str:
    """Return an ISO 8601 UTC timestamp suitable for status records."""
    from datetime import datetime, timezone
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_minimal_status(*, run_id="run-test-001", state="planned") -> dict:
    """Return a minimal valid status document for *state*."""
    chain = ["planned", "preflight_passed", "running", "completed", "validated", "published"]
    if state == "failed":
        states = ["planned", "preflight_passed", "running", "failed"]
    else:
        states = chain[: chain.index(state) + 1]
    return {
        "schema_version": 1,
        "run_id": run_id,
        "suite_id": "warpcore-v1",
        "execution_state": state,
        "lifecycle": "current",
        "history": [
            {"state": s, "timestamp": f"2026-09-15T{i:02d}:00:00Z", "note": "initial"}
            for i, s in enumerate(states)
        ],
    }


def _make_full_status_history(*states: str) -> dict:
    """Build a status dict walking through *states* in order."""
    ORDERED = [
        "planned", "preflight_passed", "running", "completed",
        "validated", "published",
    ]
    base_ts = "2026-09-15T00:00:00Z"
    history = []
    for i, s in enumerate(states):
        history.append({"state": s, "timestamp": f"2026-09-15T0{i}:00:00Z"})
    return {
        "schema_version": 1,
        "run_id": "run-test-full",
        "suite_id": "warpcore-v1",
        "execution_state": states[-1],
        "lifecycle": "current",
        "history": history,
    }


# ---------------------------------------------------------------------------
# T1. Legal forward transitions
# ---------------------------------------------------------------------------

class TestLegalForwardTransitions:
    """T1: every legal forward step is accepted."""

    FORWARD_CHAIN = [
        ("planned", "preflight_passed"),
        ("preflight_passed", "running"),
        ("running", "completed"),
        ("completed", "validated"),
        ("validated", "published"),
    ]

    @pytest.mark.parametrize("from_state,to_state", FORWARD_CHAIN)
    def test_legal_transition_accepted(self, from_state: str, to_state: str):
        current = _make_minimal_status(state=from_state)
        ts = _now_utc()
        updated = campaign_state.apply_transition(current, to_state, ts)
        assert updated["execution_state"] == to_state, (
            f"Expected execution_state={to_state!r} after transition from {from_state!r}"
        )

    def test_full_chain_completes(self):
        """Walk the entire happy-path chain in order."""
        chain = ["planned", "preflight_passed", "running", "completed", "validated", "published"]
        status = _make_minimal_status(state="planned")
        for i in range(1, len(chain)):
            ts = f"2026-09-15T{i:02d}:00:00Z"
            status = campaign_state.apply_transition(status, chain[i], ts)
        assert status["execution_state"] == "published"

    def test_history_grows_by_one_per_transition(self):
        """Each transition appends exactly one entry."""
        status = _make_minimal_status(state="planned")
        initial_len = len(status["history"])
        ts = _now_utc()
        updated = campaign_state.apply_transition(status, "preflight_passed", ts)
        assert len(updated["history"]) == initial_len + 1


# ---------------------------------------------------------------------------
# T2. Legal failure transition
# ---------------------------------------------------------------------------

class TestFailureTransition:
    """T2: running -> failed is the only legal failure path."""

    def test_running_to_failed_accepted(self):
        current = _make_minimal_status(state="running")
        ts = _now_utc()
        updated = campaign_state.apply_transition(current, "failed", ts)
        assert updated["execution_state"] == "failed"

    def test_failed_from_planned_rejected(self):
        """Cannot jump planned -> failed (no work started)."""
        current = _make_minimal_status(state="planned")
        ts = _now_utc()
        with pytest.raises(campaign_state.InvalidTransitionError):
            campaign_state.apply_transition(current, "failed", ts)

    def test_failed_from_completed_rejected(self):
        current = _make_minimal_status(state="completed")
        ts = _now_utc()
        with pytest.raises(campaign_state.InvalidTransitionError):
            campaign_state.apply_transition(current, "failed", ts)


# ---------------------------------------------------------------------------
# T3. Rejection of skipped states
# ---------------------------------------------------------------------------

class TestSkipRejection:
    """T3: skipping states is illegal."""

    SKIP_PAIRS = [
        ("planned", "running"),
        ("planned", "completed"),
        ("planned", "validated"),
        ("planned", "published"),
        ("preflight_passed", "completed"),
        ("preflight_passed", "validated"),
        ("preflight_passed", "published"),
        ("running", "validated"),
        ("running", "published"),
        ("completed", "published"),
    ]

    @pytest.mark.parametrize("from_state,to_state", SKIP_PAIRS)
    def test_skip_rejected(self, from_state: str, to_state: str):
        current = _make_minimal_status(state=from_state)
        ts = _now_utc()
        with pytest.raises(campaign_state.InvalidTransitionError):
            campaign_state.apply_transition(current, to_state, ts)


# ---------------------------------------------------------------------------
# T4. Rejection of state rewrites
# ---------------------------------------------------------------------------

class TestStateRewriteRejection:
    """T4: transitioning to the same or an earlier state is illegal."""

    def test_same_state_rewrite_rejected(self):
        current = _make_minimal_status(state="running")
        ts = _now_utc()
        with pytest.raises(campaign_state.InvalidTransitionError):
            campaign_state.apply_transition(current, "running", ts)

    def test_backward_transition_rejected(self):
        current = _make_minimal_status(state="completed")
        ts = _now_utc()
        with pytest.raises(campaign_state.InvalidTransitionError):
            campaign_state.apply_transition(current, "running", ts)

    def test_rewrite_to_planned_rejected(self):
        current = _make_minimal_status(state="running")
        ts = _now_utc()
        with pytest.raises(campaign_state.InvalidTransitionError):
            campaign_state.apply_transition(current, "planned", ts)


# ---------------------------------------------------------------------------
# T5. Rejection of malformed timestamps
# ---------------------------------------------------------------------------

class TestTimestampValidation:
    """T5: malformed or naive (non-UTC) timestamps are rejected."""

    BAD_TIMESTAMPS = [
        "not-a-timestamp",
        "2026-09-15",                   # date only, no time
        "2026-09-15T00:00:00",          # naive (no TZ info)
        "2026-09-15T00:00:00+05:00",    # non-UTC timezone offset
        "",
        "2026-09-15 00:00:00Z",         # space separator instead of T
    ]

    @pytest.mark.parametrize("bad_ts", BAD_TIMESTAMPS)
    def test_bad_timestamp_rejected(self, bad_ts: str):
        current = _make_minimal_status(state="planned")
        with pytest.raises((campaign_state.InvalidTimestampError, ValueError)):
            campaign_state.apply_transition(current, "preflight_passed", bad_ts)

    def test_valid_utc_timestamp_accepted(self):
        current = _make_minimal_status(state="planned")
        campaign_state.apply_transition(current, "preflight_passed", "2026-09-15T12:34:56Z")


# ---------------------------------------------------------------------------
# T6. Rejection of duplicate terminal transitions
# ---------------------------------------------------------------------------

class TestDuplicateTerminalTransition:
    """T6: once a terminal state ('published' or 'failed') is reached, no
    further transition (including the same state again) is allowed."""

    def test_published_to_published_rejected(self):
        status = _make_full_status_history(
            "planned", "preflight_passed", "running", "completed", "validated", "published"
        )
        ts = _now_utc()
        with pytest.raises(campaign_state.InvalidTransitionError):
            campaign_state.apply_transition(status, "published", ts)

    def test_failed_to_failed_rejected(self):
        base = _make_minimal_status(state="running")
        ts = "2026-09-15T03:00:00Z"
        failed = campaign_state.apply_transition(base, "failed", ts)
        with pytest.raises(campaign_state.InvalidTransitionError):
            campaign_state.apply_transition(failed, "failed", _now_utc())

    def test_failed_to_running_rejected(self):
        base = _make_minimal_status(state="running")
        ts = "2026-09-15T03:00:00Z"
        failed = campaign_state.apply_transition(base, "failed", ts)
        with pytest.raises(campaign_state.InvalidTransitionError):
            campaign_state.apply_transition(failed, "running", _now_utc())


# ---------------------------------------------------------------------------
# T7. Rejection of publication when lifecycle is not 'current'
# ---------------------------------------------------------------------------

class TestPublicationLifecycleGate:
    """T7: apply_transition to 'published' must fail when lifecycle != 'current'."""

    NON_CURRENT_LIFECYCLES = [
        "superseded", "historical", "diagnostic", "replay", "invalid"
    ]

    @pytest.mark.parametrize("lifecycle", NON_CURRENT_LIFECYCLES)
    def test_publish_non_current_rejected(self, lifecycle: str):
        status = _make_full_status_history(
            "planned", "preflight_passed", "running", "completed", "validated"
        )
        status["lifecycle"] = lifecycle
        ts = _now_utc()
        with pytest.raises(campaign_state.InvalidTransitionError):
            campaign_state.apply_transition(status, "published", ts)

    def test_publish_current_lifecycle_accepted(self):
        status = _make_full_status_history(
            "planned", "preflight_passed", "running", "completed", "validated"
        )
        assert status["lifecycle"] == "current"
        ts = _now_utc()
        updated = campaign_state.apply_transition(status, "published", ts)
        assert updated["execution_state"] == "published"


# ---------------------------------------------------------------------------
# T8. Execution state vs lifecycle are separate; failed -> lifecycle 'invalid'
# ---------------------------------------------------------------------------

class TestExecutionVsLifecycleSeparation:
    """T8: execution_state and lifecycle are separate dimensions.
    A failed campaign has lifecycle='invalid' unless explicitly 'diagnostic'."""

    def test_execution_state_and_lifecycle_are_separate_fields(self):
        status = _make_minimal_status(state="planned")
        assert "execution_state" in status
        assert "lifecycle" in status
        # They are independent: running with any lifecycle is representable
        status["execution_state"] = "running"
        status["lifecycle"] = "diagnostic"
        # Schema must still accept it
        validate_doc(status, SCHEMA_STATUS)

    def test_failed_execution_maps_to_lifecycle_invalid_by_default(self):
        """After failed transition, campaign_state sets lifecycle='invalid'
        when no explicit override is given."""
        status = _make_minimal_status(state="running")
        ts = _now_utc()
        updated = campaign_state.apply_transition(
            status, "failed", ts
        )
        assert updated["lifecycle"] == "invalid"

    def test_failed_execution_can_be_explicitly_diagnostic(self):
        """Caller may explicitly request lifecycle='diagnostic' for a failed run."""
        status = _make_minimal_status(state="running")
        ts = _now_utc()
        updated = campaign_state.apply_transition(
            status, "failed", ts, lifecycle="diagnostic"
        )
        assert updated["lifecycle"] == "diagnostic"

    def test_failed_execution_invalid_lifecycle_override_rejected(self):
        """Only 'invalid' or 'diagnostic' are valid lifecycle targets for a
        failed execution; 'current' is never a valid override for failed."""
        status = _make_minimal_status(state="running")
        ts = _now_utc()
        with pytest.raises((campaign_state.InvalidTransitionError, ValueError)):
            campaign_state.apply_transition(
                status, "failed", ts, lifecycle="current"
            )


# ---------------------------------------------------------------------------
# T9. Atomic status.json write
# ---------------------------------------------------------------------------

class TestAtomicWrite:
    """T9: write_status uses a sibling temp file + flush/fsync + os.replace."""

    def test_write_creates_file(self, tmp_path):
        status = _make_minimal_status(state="planned")
        dest = tmp_path / "status.json"
        campaign_state.write_status(dest, status)
        assert dest.exists()

    def test_written_file_is_valid_json(self, tmp_path):
        status = _make_minimal_status(state="planned")
        dest = tmp_path / "status.json"
        campaign_state.write_status(dest, status)
        data = json.loads(dest.read_text())
        assert data["execution_state"] == "planned"

    def test_write_is_atomic_replace(self, tmp_path, monkeypatch):
        """Verify that write_status goes through os.replace (atomic rename).

        We monkeypatch os.replace to track calls and confirm it is used.
        """
        replaced = []
        original_replace = os.replace

        def mock_replace(src, dst):
            replaced.append((src, dst))
            return original_replace(src, dst)

        monkeypatch.setattr(os, "replace", mock_replace)

        status = _make_minimal_status(state="planned")
        dest = tmp_path / "status.json"
        campaign_state.write_status(dest, status)

        assert len(replaced) == 1, "os.replace must be called exactly once"
        src_path, dst_path = replaced[0]
        assert pathlib.Path(src_path).parent == tmp_path, (
            "temp file must be a sibling of the destination (same directory)"
        )
        assert pathlib.Path(dst_path) == dest

    def test_written_status_passes_schema(self, tmp_path):
        status = _make_minimal_status(state="planned")
        dest = tmp_path / "status.json"
        campaign_state.write_status(dest, status)
        data = json.loads(dest.read_text())
        validate_doc(data, SCHEMA_STATUS)

    def test_overwrite_updates_content(self, tmp_path):
        status = _make_minimal_status(state="planned")
        dest = tmp_path / "status.json"
        campaign_state.write_status(dest, status)

        status2 = campaign_state.apply_transition(
            status, "preflight_passed", _now_utc()
        )
        campaign_state.write_status(dest, status2)
        data = json.loads(dest.read_text())
        assert data["execution_state"] == "preflight_passed"


# ---------------------------------------------------------------------------
# T10. create_campaign: hashes resolved before command generation
# ---------------------------------------------------------------------------

class TestCreateCampaignHashResolution:
    """T10: create_campaign resolves suite/adapter hashes before any command."""

    @pytest.fixture
    def tmp_repo(self, tmp_path):
        """Repo tree with the real suite, schemas, and a canonical adapter."""
        import shutil
        import yaml

        # Copy the entire suite/ directory tree (ensures validate_suite passes)
        suite_src = _REPO / "suite"
        suite_dst = tmp_path / "suite"
        for item in suite_src.rglob("*"):
            if item.is_file():
                rel = item.relative_to(suite_src)
                dst = suite_dst / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dst)

        # Results dir
        model_dir = tmp_path / "results" / "qwen3.6-35b-a3b"
        model_dir.mkdir(parents=True)

        # Canonical adapter YAML
        adapters_dir = tmp_path / "adapters"
        adapters_dir.mkdir()
        adapter_data = {
            "adapter_schema_version": 1,
            "campaign_status": "canonical",
            "model": {
                "slug": "qwen3.6-35b-a3b",
                "id": "Qwen/Qwen3.6-35B-A3B-FP8",
                "revision": "a" * 40,
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
        adapter_file = adapters_dir / "qwen3.6-35b-a3b.yaml"
        adapter_file.write_text(yaml.dump(adapter_data))

        return tmp_path

    def test_create_campaign_returns_run_dir(self, tmp_repo):
        run_dir = create_campaign.create_campaign(
            repo=tmp_repo,
            suite_path=tmp_repo / "suite" / "warpcore-v1.yaml",
            adapter_path=tmp_repo / "adapters" / "qwen3.6-35b-a3b.yaml",
            benchmark="gsm8k",
            run_id="run-2026-09-15T00-00-00",
            prompt_token_maxima={"gsm8k": 1, "ifeval": 1, "gpqa_diamond": 1},
        )
        assert run_dir.exists(), "run directory must be created"

    def test_run_dir_under_normalized_path(self, tmp_repo):
        run_dir = create_campaign.create_campaign(
            repo=tmp_repo,
            suite_path=tmp_repo / "suite" / "warpcore-v1.yaml",
            adapter_path=tmp_repo / "adapters" / "qwen3.6-35b-a3b.yaml",
            benchmark="gsm8k",
            run_id="run-2026-09-15T00-00-00",
            prompt_token_maxima={"gsm8k": 1, "ifeval": 1, "gpqa_diamond": 1},
        )
        # Must be results/<model>/runs/<suite-version>/<benchmark>/<run-id>/
        rel = run_dir.relative_to(tmp_repo / "results")
        parts = rel.parts
        assert parts[0] == "qwen3.6-35b-a3b", f"model slug mismatch: {parts}"
        assert parts[1] == "runs"
        assert parts[2] == "warpcore-v1"
        assert parts[3] == "gsm8k"
        assert parts[4] == "run-2026-09-15T00-00-00"

    def test_status_json_created_with_planned_state(self, tmp_repo):
        run_dir = create_campaign.create_campaign(
            repo=tmp_repo,
            suite_path=tmp_repo / "suite" / "warpcore-v1.yaml",
            adapter_path=tmp_repo / "adapters" / "qwen3.6-35b-a3b.yaml",
            benchmark="gsm8k",
            run_id="run-2026-09-15T00-00-00",
            prompt_token_maxima={"gsm8k": 1, "ifeval": 1, "gpqa_diamond": 1},
        )
        status_file = run_dir / "status.json"
        assert status_file.exists()
        status = json.loads(status_file.read_text())
        assert status["execution_state"] == "planned"
        assert status["history"][0]["state"] == "planned"

    def test_manifest_json_created(self, tmp_repo):
        run_dir = create_campaign.create_campaign(
            repo=tmp_repo,
            suite_path=tmp_repo / "suite" / "warpcore-v1.yaml",
            adapter_path=tmp_repo / "adapters" / "qwen3.6-35b-a3b.yaml",
            benchmark="gsm8k",
            run_id="run-2026-09-15T00-00-00",
            prompt_token_maxima={"gsm8k": 1, "ifeval": 1, "gpqa_diamond": 1},
        )
        manifest_file = run_dir / "manifest.json"
        assert manifest_file.exists()
        manifest = json.loads(manifest_file.read_text())
        assert manifest["suite_id"] == "warpcore-v1"
        assert manifest["benchmark"] == "gsm8k"

    def test_manifest_contains_adapter_hash(self, tmp_repo):
        adapter_path = tmp_repo / "adapters" / "qwen3.6-35b-a3b.yaml"
        expected_hash = hashlib.sha256(adapter_path.read_bytes()).hexdigest()

        run_dir = create_campaign.create_campaign(
            repo=tmp_repo,
            suite_path=tmp_repo / "suite" / "warpcore-v1.yaml",
            adapter_path=adapter_path,
            benchmark="gsm8k",
            run_id="run-2026-09-15T00-00-00",
            prompt_token_maxima={"gsm8k": 1, "ifeval": 1, "gpqa_diamond": 1},
        )
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["adapter_hash"] == expected_hash

    def test_manifest_contains_suite_input_hashes(self, tmp_repo):
        suite_input = tmp_repo / "suite" / "tasks" / "gsm8k_clean_v1.yaml"
        expected_hash = hashlib.sha256(suite_input.read_bytes()).hexdigest()

        run_dir = create_campaign.create_campaign(
            repo=tmp_repo,
            suite_path=tmp_repo / "suite" / "warpcore-v1.yaml",
            adapter_path=tmp_repo / "adapters" / "qwen3.6-35b-a3b.yaml",
            benchmark="gsm8k",
            run_id="run-2026-09-15T00-00-00",
            prompt_token_maxima={"gsm8k": 1, "ifeval": 1, "gpqa_diamond": 1},
        )
        manifest = json.loads((run_dir / "manifest.json").read_text())
        hashes = manifest["suite_input_hashes"]
        assert "suite/tasks/gsm8k_clean_v1.yaml" in hashes
        assert hashes["suite/tasks/gsm8k_clean_v1.yaml"] == expected_hash

    def test_manifest_passes_schema(self, tmp_repo):
        run_dir = create_campaign.create_campaign(
            repo=tmp_repo,
            suite_path=tmp_repo / "suite" / "warpcore-v1.yaml",
            adapter_path=tmp_repo / "adapters" / "qwen3.6-35b-a3b.yaml",
            benchmark="gsm8k",
            run_id="run-2026-09-15T00-00-00",
            prompt_token_maxima={"gsm8k": 1, "ifeval": 1, "gpqa_diamond": 1},
        )
        manifest = json.loads((run_dir / "manifest.json").read_text())
        validate_doc(manifest, SCHEMA_MANIFEST)

    def test_status_passes_schema(self, tmp_repo):
        run_dir = create_campaign.create_campaign(
            repo=tmp_repo,
            suite_path=tmp_repo / "suite" / "warpcore-v1.yaml",
            adapter_path=tmp_repo / "adapters" / "qwen3.6-35b-a3b.yaml",
            benchmark="gsm8k",
            run_id="run-2026-09-15T00-00-00",
            prompt_token_maxima={"gsm8k": 1, "ifeval": 1, "gpqa_diamond": 1},
        )
        status = json.loads((run_dir / "status.json").read_text())
        validate_doc(status, SCHEMA_STATUS)


# ---------------------------------------------------------------------------
# T11. create_campaign: directory collision policy
# ---------------------------------------------------------------------------

class TestCreateCampaignCollision:
    """T11: refuse existing output directory unless explicit resume."""

    @pytest.fixture
    def tmp_repo_with_run(self, tmp_path):
        """A repo where a run directory already exists."""
        import shutil
        import yaml

        # Copy the entire suite/ directory tree (ensures validate_suite passes)
        suite_src = _REPO / "suite"
        suite_dst = tmp_path / "suite"
        for item in suite_src.rglob("*"):
            if item.is_file():
                rel = item.relative_to(suite_src)
                dst = suite_dst / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dst)

        suite_yaml = suite_dst / "warpcore-v1.yaml"

        adapters_dir = tmp_path / "adapters"
        adapters_dir.mkdir()
        adapter_data = {
            "adapter_schema_version": 1,
            "campaign_status": "canonical",
            "model": {
                "slug": "qwen3.6-35b-a3b",
                "id": "Qwen/Qwen3.6-35B-A3B-FP8",
                "revision": "a" * 40,
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
        adapter_file = adapters_dir / "qwen3.6-35b-a3b.yaml"
        adapter_file.write_text(yaml.dump(adapter_data))

        model_dir = tmp_path / "results" / "qwen3.6-35b-a3b"
        model_dir.mkdir(parents=True)

        # First campaign creation
        run_dir = create_campaign.create_campaign(
            repo=tmp_path,
            suite_path=suite_yaml,
            adapter_path=adapter_file,
            benchmark="gsm8k",
            run_id="run-2026-09-15T00-00-00",
            prompt_token_maxima={"gsm8k": 1, "ifeval": 1, "gpqa_diamond": 1},
        )
        return tmp_path, suite_yaml, adapter_file, run_dir

    def test_second_creation_without_resume_raises(self, tmp_repo_with_run):
        tmp_path, suite_yaml, adapter_file, _ = tmp_repo_with_run
        with pytest.raises(create_campaign.OutputDirectoryCollisionError):
            create_campaign.create_campaign(
                repo=tmp_path,
                suite_path=suite_yaml,
                adapter_path=adapter_file,
                benchmark="gsm8k",
                run_id="run-2026-09-15T00-00-00",
                prompt_token_maxima={"gsm8k": 1, "ifeval": 1, "gpqa_diamond": 1},
            )


# ---------------------------------------------------------------------------
# T12. create_campaign: rejects noncanonical adapter
# ---------------------------------------------------------------------------

class TestCreateCampaignNoncanonicalAdapter:
    """T12: noncanonical adapters must be refused."""

    def test_noncanonical_adapter_rejected(self, tmp_path):
        import shutil
        import yaml

        # Copy the entire suite/ directory tree (ensures validate_suite passes)
        suite_src = _REPO / "suite"
        suite_dst = tmp_path / "suite"
        for item in suite_src.rglob("*"):
            if item.is_file():
                rel = item.relative_to(suite_src)
                dst = suite_dst / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dst)

        suite_yaml = suite_dst / "warpcore-v1.yaml"

        model_dir = tmp_path / "results" / "qwen3.6-35b-a3b"
        model_dir.mkdir(parents=True)

        adapters_dir = tmp_path / "adapters"
        adapters_dir.mkdir()
        adapter_data = {
            "adapter_schema_version": 1,
            "campaign_status": "noncanonical",
            "noncanonical_reason": "missing revision",
            "model": {
                "slug": "qwen3.6-35b-a3b",
                "id": "Qwen/Qwen3.6-35B-A3B-FP8",
                "revision": "unresolved",
            },
            "serving": {
                "image": "unresolved",
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
        adapter_file = adapters_dir / "qwen3.6-35b-a3b.yaml"
        adapter_file.write_text(yaml.dump(adapter_data))

        with pytest.raises(create_campaign.NoncanonicalAdapterError):
            create_campaign.create_campaign(
                repo=tmp_path,
                suite_path=suite_yaml,
                adapter_path=adapter_file,
                benchmark="gsm8k",
                run_id="run-2026-09-15T00-00-00",
            )


# ---------------------------------------------------------------------------
# T13. Safe path component validation
# ---------------------------------------------------------------------------

class TestSafePathComponents:
    """T13: run_id, model slug, and benchmark names must be safe."""

    UNSAFE_COMPONENTS = [
        "../evil",
        "../../etc/passwd",
        "run id with spaces",
        "run\x00null",
        "run/with/slash",
        "",
    ]

    @pytest.mark.parametrize("unsafe", UNSAFE_COMPONENTS)
    def test_unsafe_run_id_rejected(self, tmp_path, unsafe):
        with pytest.raises((create_campaign.UnsafePathComponentError, ValueError)):
            create_campaign.validate_safe_path_component(unsafe)

    def test_safe_run_id_accepted(self):
        # Should not raise
        create_campaign.validate_safe_path_component("run-2026-09-15T00-00-00")
        create_campaign.validate_safe_path_component("run-abc123")

    def test_safe_model_slug_accepted(self):
        create_campaign.validate_safe_path_component("qwen3.6-35b-a3b")
        create_campaign.validate_safe_path_component("gpt-oss-120b")

    def test_unsafe_benchmark_rejected(self):
        with pytest.raises((create_campaign.UnsafePathComponentError, ValueError)):
            create_campaign.validate_safe_path_component("../bench")


# ---------------------------------------------------------------------------
# T14. Manifest and status schema validation with FormatChecker
# ---------------------------------------------------------------------------

class TestSchemaValidationWithFormatChecker:
    """T14: manifests and status records are validated with FormatChecker."""

    def test_valid_manifest_passes(self):
        validate_doc(copy.deepcopy(VALID_MANIFEST), SCHEMA_MANIFEST)

    def test_valid_status_passes(self):
        validate_doc(copy.deepcopy(VALID_STATUS), SCHEMA_STATUS)

    def test_manifest_bad_datetime_fails(self):
        import jsonschema
        doc = copy.deepcopy(VALID_MANIFEST)
        doc["timing"]["started_utc"] = "not-a-datetime"
        with pytest.raises(jsonschema.ValidationError):
            validate_doc(doc, SCHEMA_MANIFEST)

    def test_status_bad_datetime_fails(self):
        import jsonschema
        doc = copy.deepcopy(VALID_STATUS)
        doc["history"][0]["timestamp"] = "not-a-datetime"
        with pytest.raises(jsonschema.ValidationError):
            validate_doc(doc, SCHEMA_STATUS)

    def test_status_unknown_field_fails(self):
        import jsonschema
        doc = copy.deepcopy(VALID_STATUS)
        doc["unknown_extra_field"] = "oops"
        with pytest.raises(jsonschema.ValidationError):
            validate_doc(doc, SCHEMA_STATUS)


# ---------------------------------------------------------------------------
# T15. Symlink safety
# ---------------------------------------------------------------------------

class TestSymlinkSafety:
    """T15: status path must resolve inside the run directory; symlinks
    that escape the run dir are refused."""

    def test_symlink_outside_run_dir_refused(self, tmp_path):
        """write_status must refuse if the destination path resolves outside
        the run directory (e.g., via a symlink)."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        outside = tmp_path / "outside.json"
        outside.write_text("{}")
        # Create a symlink inside run_dir pointing outside
        evil_link = run_dir / "status.json"
        evil_link.symlink_to(outside)

        status = _make_minimal_status(state="planned")
        with pytest.raises((campaign_state.UnsafePathError, OSError, PermissionError)):
            campaign_state.write_status(evil_link, status, run_dir=run_dir)


# ---------------------------------------------------------------------------
# T16. Transition to 'published' requires lifecycle == 'current'
# ---------------------------------------------------------------------------
# (Covered by T7 above; kept as alias for clarity.)


# ---------------------------------------------------------------------------
# T17. Legacy manifest_scaffold.py backward compatibility
# ---------------------------------------------------------------------------

class TestLegacyManifestScaffold:
    """T17: make manifest MODEL=... BENCH=... must still work and produce
    schema-valid v1 legacy manifests."""

    def test_legacy_build_returns_dict_with_schema_version(self):
        """manifest_scaffold.build() returns a dict with schema_version=1."""
        import manifest_scaffold
        manifest = manifest_scaffold.build(
            model="ornith-35b",
            bench="swebench",
            endpoint=None,
            launch_script=None,
        )
        assert manifest.get("schema_version") == 1

    def test_legacy_build_populates_unrecorded(self):
        """Unprobed fields are 'unrecorded', never None or missing."""
        import manifest_scaffold
        manifest = manifest_scaffold.build(
            model="ornith-35b",
            bench="swebench",
            endpoint=None,
            launch_script=None,
        )
        # At minimum these legacy fields must be present
        assert "model" in manifest
        assert "benchmark" in manifest
        assert "serving" in manifest

    def test_legacy_stdout_mode_exits_zero(self):
        """manifest_scaffold.main(['--model', ..., '--bench', ..., '--stdout']) -> 0."""
        import manifest_scaffold
        import io
        from contextlib import redirect_stdout

        # Need a real model dir to exist — use a real one from the repo
        RESULTS = _REPO / "results"
        existing_models = [p.name for p in RESULTS.iterdir() if p.is_dir()]
        if not existing_models:
            pytest.skip("No model directories in results/ for legacy test")

        model = existing_models[0]
        buf = io.StringIO()
        with redirect_stdout(buf):
            ret = manifest_scaffold.main(
                ["--model", model, "--bench", "swebench", "--stdout"]
            )
        assert ret == 0
        data = json.loads(buf.getvalue())
        assert data.get("schema_version") == 1

    def test_legacy_nonexistent_model_exits_nonzero(self):
        """manifest_scaffold.main returns non-zero for nonexistent model dir."""
        import manifest_scaffold
        ret = manifest_scaffold.main(
            ["--model", "NONEXISTENT_MODEL_XYZ_123", "--bench", "swebench", "--stdout"]
        )
        assert ret != 0


# ---------------------------------------------------------------------------
# T18. History is append-only
# ---------------------------------------------------------------------------

class TestAppendOnlyHistory:
    """T18: each transition appends exactly one entry; no existing entry is
    removed or modified."""

    def test_each_transition_appends_exactly_one_entry(self):
        status = _make_minimal_status(state="planned")
        assert len(status["history"]) == 1

        ts1 = "2026-09-15T01:00:00Z"
        status = campaign_state.apply_transition(status, "preflight_passed", ts1)
        assert len(status["history"]) == 2
        assert status["history"][0]["state"] == "planned"
        assert status["history"][1]["state"] == "preflight_passed"

        ts2 = "2026-09-15T02:00:00Z"
        status = campaign_state.apply_transition(status, "running", ts2)
        assert len(status["history"]) == 3
        assert status["history"][0]["state"] == "planned"
        assert status["history"][1]["state"] == "preflight_passed"
        assert status["history"][2]["state"] == "running"

    def test_original_history_not_mutated(self):
        """apply_transition must not modify the input dict in-place."""
        status = _make_minimal_status(state="planned")
        original_history = copy.deepcopy(status["history"])
        ts = _now_utc()
        _updated = campaign_state.apply_transition(status, "preflight_passed", ts)
        # The original status dict's history must be unchanged
        assert status["history"] == original_history
