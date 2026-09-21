"""Phase A evidence closure — the failed gpt-oss-120b warpcore-v1 SWE-bench campaign.

The 2026-09-21 gpt-oss-120b SWE-bench campaign was stopped by the operator after
the vLLM `openai` tool-call parser corrupted tool-call JSON arguments on almost
every instance. It produced 51 terminal trajectories (49 RepeatedFormatError,
2 Submitted), no grading, and no score. The retained evidence is committed as
immutable diagnostic material under an `invalid` lifecycle.

These tests enforce three things:

  1. The retained evidence is present, complete, and unaltered — the creation-time
     manifest keeps its `submitted: 0` and all-false artifact flags, because it is
     raw evidence of what the runner actually wrote, not a record to be repaired.
  2. `viz/derive_diagnostic.py` derives the diagnostic summary deterministically
     from that evidence and fails closed on every count / ID / hash inconsistency.
  3. The run can never populate canonical publication.

Fail-closed behaviour is exercised against a synthetic miniature run rather than
the real 328 MB evidence tree, so the mutation matrix stays cheap.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import pathlib
import shutil
import sys
import tarfile

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
VIZ = REPO / "viz"
if str(VIZ) not in sys.path:
    sys.path.insert(0, str(VIZ))

RUN_REL = (
    "results/gpt-oss-120b/runs/warpcore-v1/swebench/gptoss-swebench-n100-20260921"
)
RUN_DIR = REPO / RUN_REL
SUITE = REPO / "suite" / "warpcore-v1.yaml"
ADAPTER = REPO / "adapters" / "gpt-oss-120b.yaml"

EXPECTED_ITEMS = 100
TERMINAL_OBSERVED = 51
UNOBSERVED = 49
EXIT_STATUS_COUNTS = {"RepeatedFormatError": 49, "Submitted": 2}

#: Every physical file that carries the diagnostic entry.
EVIDENCE_FILES = [
    "TERMINATION.json",
    "command.txt",
    "diagnostic_summary.json",
    "manifest.json",
    "raw/exit_statuses_1789996651.9608052.yaml",
    "raw/minisweagent.log",
    "raw/preds.json",
    "raw/run.log",
    "raw/trajectories.tar.gz",
    "status.json",
    "suite_input_snapshots/adapters/gpt-oss-120b.yaml",
    "suite_input_snapshots/suite/swebench/instances-seed42-n100.json",
    "suite_input_snapshots/suite/swebench/scaffold.yaml",
    "suite_input_snapshots/suite/warpcore-v1.yaml",
]


def _import_tool():
    import derive_diagnostic  # noqa: PLC0415

    return derive_diagnostic


# ---------------------------------------------------------------------------
# Synthetic miniature campaign — the mutation substrate
# ---------------------------------------------------------------------------

_FAKE_IDS = ["aaa__aaa-1", "bbb__bbb-2", "ccc__ccc-3"]


_FAKE_PATCH = "diff --git a b\n"


def _fake_traj(instance_id: str, exit_status: str) -> bytes:
    """A miniature mini-swe-agent trajectory with the real terminal-status shape."""
    submission = _FAKE_PATCH if exit_status == "Submitted" else ""
    doc = {
        "info": {
            "model_stats": {"instance_cost": 0.0, "api_calls": 3},
            "mini_version": "2.4.6",
            "exit_status": exit_status,
            "submission": submission,
        },
        "instance_id": instance_id,
        "trajectory_format": "mini-swe-agent-1.1",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": None, "extra": {"response": {"id": "x"}}},
            {"role": "user", "content": "Tool call error:\n\n<error>\nError parsing "
                                        "tool call arguments: Expecting ',' delimiter\n</error>"},
            {"role": "exit", "content": submission or exit_status,
             "extra": {"exit_status": exit_status, "submission": submission}},
        ],
    }
    return json.dumps(doc, indent=2).encode()


def _write_archive(path: pathlib.Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tar:
        for name in sorted(members):
            data = members[name]
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0
            tar.addfile(info, io.BytesIO(data))


def build_mini_run(root: pathlib.Path) -> pathlib.Path:
    """Build a 3-instance analogue of the real failed campaign under *root*.

    Returns the repo root; the run directory is
    ``results/fake-model/runs/warpcore-v1/swebench/fake-run``.
    """
    instances = sorted(_FAKE_IDS + ["ddd__ddd-4", "eee__eee-5"])
    instances_file = root / "suite" / "swebench" / "instances-fake.json"
    instances_file.parent.mkdir(parents=True, exist_ok=True)
    instances_payload = json.dumps(instances, indent=2) + "\n"
    instances_file.write_text(instances_payload)

    suite_path = root / "suite" / "warpcore-v1.yaml"
    suite_path.write_text(
        "suite_id: warpcore-v1\n"
        "benchmarks:\n"
        "  swebench:\n"
        '    instance_set_file: "suite/swebench/instances-fake.json"\n'
        f'    instances_sha256: "{hashlib.sha256(instances_payload.encode()).hexdigest()}"\n'
        "    expected_item_count: 5\n"
    )

    adapter_path = root / "adapters" / "fake-model.yaml"
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    adapter_path.write_text("adapter_schema_version: 1\nmodel:\n  slug: fake-model\n")

    run_dir = root / "results/fake-model/runs/warpcore-v1/swebench/fake-run"
    (run_dir / "raw").mkdir(parents=True)

    statuses = {_FAKE_IDS[0]: "RepeatedFormatError",
                _FAKE_IDS[1]: "RepeatedFormatError",
                _FAKE_IDS[2]: "Submitted"}

    (run_dir / "raw" / "exit_statuses_1.yaml").write_text(
        "instances_by_exit_status:\n"
        "    RepeatedFormatError:\n"
        f"    - {_FAKE_IDS[0]}\n"
        f"    - {_FAKE_IDS[1]}\n"
        "    Submitted:\n"
        f"    - {_FAKE_IDS[2]}\n"
    )
    (run_dir / "raw" / "preds.json").write_text(json.dumps({
        iid: {
            "model_name_or_path": "fake",
            "instance_id": iid,
            "model_patch": _FAKE_PATCH if statuses[iid] == "Submitted" else "",
        }
        for iid in _FAKE_IDS
    }, indent=2))
    (run_dir / "raw" / "run.log").write_text("runner log\n")
    (run_dir / "raw" / "minisweagent.log").write_text("agent log\n")
    _write_archive(
        run_dir / "raw" / "trajectories.tar.gz",
        {f"{iid}/{iid}.traj.json": _fake_traj(iid, statuses[iid]) for iid in _FAKE_IDS},
    )
    (run_dir / "command.txt").write_text("python3 viz/run_swebench.py --run-id fake-run\n")

    manifest = {
        "schema_version": 1,
        "suite_id": "warpcore-v1",
        "run_id": "fake-run",
        "benchmark": "swebench",
        "adapter_hash": hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
        "suite_input_hashes": {
            "suite/swebench/instances-fake.json":
                hashlib.sha256(instances_payload.encode()).hexdigest(),
        },
        "model": {"slug": "fake-model", "id": "fake/model"},
        "item_inventory": {
            "expected": 5,
            "submitted": 0,
            "instance_ids_hash": hashlib.sha256(
                json.dumps(sorted(instances), sort_keys=True).encode()
            ).hexdigest(),
        },
        "timing": {"started_utc": "2026-09-21T13:17:25Z", "completed_utc": None},
        "artifact_inventory": {
            "samples_jsonl_gz": False, "per_item_csv": False, "run_log": False,
            "command_txt": False, "done_sentinel": False,
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (run_dir / "status.json").write_text(json.dumps({
        "schema_version": 1,
        "run_id": "fake-run",
        "suite_id": "warpcore-v1",
        "execution_state": "failed",
        "lifecycle": "invalid",
        "history": [
            {"state": "planned", "timestamp": "2026-09-21T13:17:25Z"},
            {"state": "preflight_passed", "timestamp": "2026-09-21T13:17:28Z"},
            {"state": "running", "timestamp": "2026-09-21T13:17:28Z"},
            {"state": "failed", "timestamp": "2026-09-21T14:37:44Z"},
        ],
    }, indent=2))
    (run_dir / "TERMINATION.json").write_text(json.dumps({
        "schema_version": 1,
        "run_id": "fake-run",
        "mode": "operator_initiated",
        "attested_by": "operator",
        "attested_utc": "2026-09-21T14:37:44Z",
        "reason": "systemic tool-call parse failure observed; campaign stopped",
        "score_claimed": False,
        "grading_performed": False,
        "corroborating_evidence": ["raw/run.log ends with no traceback"],
    }, indent=2))
    return root


@pytest.fixture()
def mini_repo(tmp_path):
    return build_mini_run(tmp_path / "repo")


MINI_RUN_REL = "results/fake-model/runs/warpcore-v1/swebench/fake-run"


def _mini_run_dir(repo: pathlib.Path) -> pathlib.Path:
    return repo / MINI_RUN_REL


def _emit_mini(repo: pathlib.Path):
    tool = _import_tool()
    run_dir = _mini_run_dir(repo)
    summary = tool.derive(run_dir, repo=repo)
    tool.write_summary(run_dir, summary)
    return summary


# ---------------------------------------------------------------------------
# 1. Retained evidence is present, complete, and unaltered
# ---------------------------------------------------------------------------

class TestRetainedEvidence:
    def test_every_evidence_file_is_committed(self):
        missing = [rel for rel in EVIDENCE_FILES if not (RUN_DIR / rel).is_file()]
        assert not missing, f"Missing retained evidence: {missing}"

    def test_no_expanded_trajectory_copies(self):
        """Trajectories live only in the archive — no duplicate expanded tree."""
        expanded = sorted(p.relative_to(REPO) for p in RUN_DIR.rglob("*.traj.json"))
        assert not expanded, (
            "Trajectories must be retained only as raw/trajectories.tar.gz; "
            f"found expanded copies: {expanded[:5]}"
        )

    def test_run_directory_holds_only_registered_evidence(self):
        present = sorted(
            str(p.relative_to(RUN_DIR)) for p in RUN_DIR.rglob("*")
            if p.is_file() and p.name != ".DS_Store"
        )
        assert present == sorted(EVIDENCE_FILES)

    def test_creation_time_manifest_is_preserved_unrepaired(self):
        """The manifest is raw evidence: its wrong counters must NOT be corrected."""
        manifest = json.loads((RUN_DIR / "manifest.json").read_text())
        assert manifest["item_inventory"]["submitted"] == 0
        assert manifest["item_inventory"]["expected"] == EXPECTED_ITEMS
        assert manifest["timing"]["completed_utc"] is None
        assert set(manifest["artifact_inventory"].values()) == {False}

    def test_status_history_is_preserved(self):
        status = json.loads((RUN_DIR / "status.json").read_text())
        assert status["execution_state"] == "failed"
        assert status["lifecycle"] == "invalid"
        assert [h["state"] for h in status["history"]] == [
            "planned", "preflight_passed", "running", "failed",
        ]

    def test_no_done_sentinel_and_no_grading(self):
        assert not (RUN_DIR / "DONE").exists()
        assert not (RUN_DIR / "raw" / "grading_results.json").exists()
        assert not (RUN_DIR / "per_item.csv").exists()

    def test_archive_holds_exactly_the_terminal_trajectories(self):
        with tarfile.open(RUN_DIR / "raw" / "trajectories.tar.gz") as tar:
            names = [m.name for m in tar.getmembers() if m.isfile()]
        assert len(names) == TERMINAL_OBSERVED
        for name in names:
            iid, _, leaf = name.partition("/")
            assert leaf == f"{iid}.traj.json", name


# ---------------------------------------------------------------------------
# 2. Deterministic derivation over the real evidence
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_summary():
    tool = _import_tool()
    return tool.derive(RUN_DIR, repo=REPO)


class TestDerivedSummary:
    def test_committed_summary_matches_rederivation(self, real_summary):
        committed = json.loads((RUN_DIR / "diagnostic_summary.json").read_text())
        assert committed == real_summary

    def test_verify_passes_on_committed_evidence(self):
        _import_tool().verify(RUN_DIR, repo=REPO)

    def test_records_operator_initiated_termination(self, real_summary):
        assert real_summary["termination"]["mode"] == "operator_initiated"
        assert real_summary["termination"]["attested_utc"]
        assert real_summary["termination"]["reason"]

    def test_records_unobserved_instance_ids(self, real_summary):
        inv = real_summary["item_inventory"]
        assert inv["expected"] == EXPECTED_ITEMS
        assert inv["terminal_observed"] == TERMINAL_OBSERVED
        assert inv["unobserved"] == UNOBSERVED
        assert len(inv["unobserved_instance_ids"]) == UNOBSERVED
        assert len(inv["terminal_instance_ids"]) == TERMINAL_OBSERVED
        assert not set(inv["unobserved_instance_ids"]) & set(inv["terminal_instance_ids"])

    def test_records_terminal_exit_status_counts(self, real_summary):
        assert real_summary["terminal_exit_statuses"] == EXIT_STATUS_COUNTS

    def test_records_no_score_and_no_grading(self, real_summary):
        assert real_summary["score"] is None
        assert real_summary["grading"]["present"] is False
        assert real_summary["grading"]["graded_instances"] == 0

    def test_records_nonpublishability(self, real_summary):
        assert real_summary["publishable"] is False
        assert real_summary["lifecycle"] == "invalid"
        assert real_summary["nonpublishable_reasons"]

    def test_records_trajectory_inventory_and_digests(self, real_summary):
        trajs = real_summary["trajectories"]
        assert trajs["count"] == TERMINAL_OBSERVED
        assert trajs["archive"] == "raw/trajectories.tar.gz"
        assert len(real_summary["trajectory_digests"]) == TERMINAL_OBSERVED

    def test_records_artifact_digests_that_match_disk(self, real_summary):
        for rel, rec in real_summary["artifacts"].items():
            blob = (RUN_DIR / rel).read_bytes()
            assert rec["bytes"] == len(blob), rel
            assert rec["sha256"] == hashlib.sha256(blob).hexdigest(), rel

    def test_derivation_is_deterministic(self, real_summary):
        assert _import_tool().derive(RUN_DIR, repo=REPO) == real_summary


# ---------------------------------------------------------------------------
# 3. Fail-closed behaviour on malformed / incomplete diagnostics
# ---------------------------------------------------------------------------

class TestFailClosed:
    def test_mini_run_is_a_valid_baseline(self, mini_repo):
        """Negative-control anchor: the unmutated fixture must derive cleanly."""
        summary = _emit_mini(mini_repo)
        assert summary["item_inventory"]["terminal_observed"] == 3
        assert summary["item_inventory"]["unobserved"] == 2
        assert summary["score"] is None
        _import_tool().verify(_mini_run_dir(mini_repo), repo=mini_repo)

    def _expect_failure(self, repo, match):
        tool = _import_tool()
        with pytest.raises(tool.DiagnosticEvidenceError, match=match):
            tool.derive(_mini_run_dir(repo), repo=repo)

    def test_trajectory_missing_from_archive_fails(self, mini_repo):
        run_dir = _mini_run_dir(mini_repo)
        statuses = {_FAKE_IDS[0]: "RepeatedFormatError", _FAKE_IDS[1]: "RepeatedFormatError"}
        _write_archive(
            run_dir / "raw" / "trajectories.tar.gz",
            {f"{i}/{i}.traj.json": _fake_traj(i, s) for i, s in statuses.items()},
        )
        self._expect_failure(mini_repo, r"missing trajectories \['ccc__ccc-3'\]")

    def test_foreign_instance_id_in_preds_fails(self, mini_repo):
        run_dir = _mini_run_dir(mini_repo)
        preds = json.loads((run_dir / "raw" / "preds.json").read_text())
        preds["zzz__zzz-9"] = {"instance_id": "zzz__zzz-9", "model_patch": ""}
        (run_dir / "raw" / "preds.json").write_text(json.dumps(preds, indent=2))
        self._expect_failure(mini_repo, r"only in preds \['zzz__zzz-9'\]")

    def test_exit_status_id_set_mismatch_fails(self, mini_repo):
        run_dir = _mini_run_dir(mini_repo)
        (run_dir / "raw" / "exit_statuses_1.yaml").write_text(
            "instances_by_exit_status:\n"
            "    RepeatedFormatError:\n"
            f"    - {_FAKE_IDS[0]}\n"
            "    Submitted:\n"
            f"    - {_FAKE_IDS[2]}\n"
        )
        self._expect_failure(mini_repo, r"only in preds \['bbb__bbb-2'\]")

    def test_duplicate_instance_id_across_exit_buckets_fails(self, mini_repo):
        run_dir = _mini_run_dir(mini_repo)
        (run_dir / "raw" / "exit_statuses_1.yaml").write_text(
            "instances_by_exit_status:\n"
            "    RepeatedFormatError:\n"
            f"    - {_FAKE_IDS[0]}\n"
            f"    - {_FAKE_IDS[1]}\n"
            f"    - {_FAKE_IDS[2]}\n"
            "    Submitted:\n"
            f"    - {_FAKE_IDS[2]}\n"
        )
        self._expect_failure(mini_repo, "duplicate instance ID")

    def test_trajectory_exit_status_disagreeing_with_yaml_fails(self, mini_repo):
        run_dir = _mini_run_dir(mini_repo)
        statuses = {_FAKE_IDS[0]: "Submitted", _FAKE_IDS[1]: "RepeatedFormatError",
                    _FAKE_IDS[2]: "Submitted"}
        _write_archive(
            run_dir / "raw" / "trajectories.tar.gz",
            {f"{i}/{i}.traj.json": _fake_traj(i, s) for i, s in statuses.items()},
        )
        self._expect_failure(
            mini_repo,
            "ends in exit status 'Submitted' but the exit status inventory records",
        )

    def test_instance_ids_hash_mismatch_fails(self, mini_repo):
        run_dir = _mini_run_dir(mini_repo)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        manifest["item_inventory"]["instance_ids_hash"] = "f" * 64
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        self._expect_failure(mini_repo, "item_inventory.instance_ids_hash")

    def test_expected_count_disagreeing_with_frozen_set_fails(self, mini_repo):
        run_dir = _mini_run_dir(mini_repo)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        manifest["item_inventory"]["expected"] = 4
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        self._expect_failure(mini_repo, "item_inventory.expected=4")

    def test_suite_input_hash_mismatch_fails(self, mini_repo):
        run_dir = _mini_run_dir(mini_repo)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        manifest["suite_input_hashes"]["suite/swebench/instances-fake.json"] = "a" * 64
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        self._expect_failure(mini_repo, "suite_input_hashes hash mismatch")

    # --- Regression: snapshot-aware G8 verification -------------------------

    def _write_full_mini_snapshots(self, run_dir, mini_repo, tool):
        """Write snapshots for all suite_input_hashes entries + the adapter.

        In snapshot mode every suite_input_hashes key and the adapter must have
        a run-owned snapshot.  Call this helper after updating the manifest with
        any additional suite_input_hashes entries, before calling derive().
        """
        manifest = json.loads((run_dir / "manifest.json").read_text())
        snapshots_dir = run_dir / tool.SNAPSHOTS_DIR

        for rel in manifest.get("suite_input_hashes", {}):
            src = mini_repo / rel
            dst = snapshots_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())

        # Adapter snapshot.
        model_slug = (manifest.get("model") or {}).get("slug", "")
        adapter_src = mini_repo / "adapters" / f"{model_slug}.yaml"
        adapter_dst = snapshots_dir / "adapters" / f"{model_slug}.yaml"
        adapter_dst.parent.mkdir(parents=True, exist_ok=True)
        adapter_dst.write_bytes(adapter_src.read_bytes())

    def test_evolved_suite_input_verified_via_snapshot(self, mini_repo):
        """G8 regression: a suite input that changed after the run must not
        falsely invalidate historical evidence when a run-owned snapshot exists.

        The snapshot mirrors the full repo-relative path under suite_input_snapshots/
        so there is no basename collision risk.  Mutating the canonical repo file
        must not cause derive() to fail if the snapshot still matches the declared hash.
        """
        run_dir = _mini_run_dir(mini_repo)
        tool = _import_tool()
        # Add a warpcore-v1.yaml entry to suite_input_hashes.
        suite_yaml_path = mini_repo / "suite" / "warpcore-v1.yaml"
        original_bytes = suite_yaml_path.read_bytes()
        original_hash = hashlib.sha256(original_bytes).hexdigest()

        manifest = json.loads((run_dir / "manifest.json").read_text())
        manifest["suite_input_hashes"]["suite/warpcore-v1.yaml"] = original_hash
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        # Write all required snapshots (every suite_input_hashes entry + adapter).
        self._write_full_mini_snapshots(run_dir, mini_repo, tool)

        # Now evolve the canonical file — simulates a qualification gate update.
        suite_yaml_path.write_text(
            suite_yaml_path.read_text() + "  # post-run qualification change\n"
        )
        assert hashlib.sha256(suite_yaml_path.read_bytes()).hexdigest() != original_hash

        # derive() must succeed: snapshot matches declared hash.
        summary = tool.derive(run_dir, repo=mini_repo)
        assert summary["suite_input_hashes"]["suite/warpcore-v1.yaml"] == original_hash

    def test_no_basename_collision_with_full_path_snapshots(self, mini_repo):
        """G8: two suite inputs sharing a basename but different directories
        must not collide in the snapshot store — full repo-relative paths are used.
        """
        run_dir = _mini_run_dir(mini_repo)
        tool = _import_tool()

        # Create two files that share a basename but live in different directories.
        path_a = mini_repo / "suite" / "warpcore-v1.yaml"
        alt_dir = mini_repo / "suite" / "swebench"
        alt_dir.mkdir(parents=True, exist_ok=True)
        path_b = alt_dir / "warpcore-v1.yaml"
        path_b.write_text("# alternate warpcore-v1\n")

        bytes_a = path_a.read_bytes()
        bytes_b = path_b.read_bytes()
        hash_a = hashlib.sha256(bytes_a).hexdigest()
        hash_b = hashlib.sha256(bytes_b).hexdigest()
        assert hash_a != hash_b, "test requires distinct content for the two files"

        manifest = json.loads((run_dir / "manifest.json").read_text())
        manifest["suite_input_hashes"]["suite/warpcore-v1.yaml"] = hash_a
        manifest["suite_input_hashes"]["suite/swebench/warpcore-v1.yaml"] = hash_b
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        # Write all snapshots (including instances-fake.json and adapter).
        self._write_full_mini_snapshots(run_dir, mini_repo, tool)

        # Both canonical files evolve — neither snapshot-backed hash should fail.
        path_a.write_text(path_a.read_text() + "# evolved\n")
        path_b.write_text(path_b.read_text() + "# evolved\n")

        summary = tool.derive(run_dir, repo=mini_repo)
        assert summary["suite_input_hashes"]["suite/warpcore-v1.yaml"] == hash_a
        assert summary["suite_input_hashes"]["suite/swebench/warpcore-v1.yaml"] == hash_b

    def test_tampered_snapshot_fails_closed(self, mini_repo):
        """G8 fail-closed: a snapshot that does not match its declared hash
        must cause derive() to raise DiagnosticEvidenceError.
        """
        run_dir = _mini_run_dir(mini_repo)
        tool = _import_tool()
        suite_yaml_path = mini_repo / "suite" / "warpcore-v1.yaml"
        original_bytes = suite_yaml_path.read_bytes()
        original_hash = hashlib.sha256(original_bytes).hexdigest()

        manifest = json.loads((run_dir / "manifest.json").read_text())
        manifest["suite_input_hashes"]["suite/warpcore-v1.yaml"] = original_hash
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        # Write all required snapshots first, then tamper with one.
        self._write_full_mini_snapshots(run_dir, mini_repo, tool)
        # Overwrite the warpcore-v1.yaml snapshot with tampered bytes.
        (run_dir / tool.SNAPSHOTS_DIR / "suite" / "warpcore-v1.yaml").write_bytes(
            b"tampered content\n"
        )

        with pytest.raises(tool.DiagnosticEvidenceError, match=tool.SNAPSHOTS_DIR):
            tool.derive(run_dir, repo=mini_repo)

    def test_grading_artifact_present_fails(self, mini_repo):
        run_dir = _mini_run_dir(mini_repo)
        (run_dir / "raw" / "grading_results.json").write_text(
            json.dumps({"resolved_ids": [_FAKE_IDS[2]], "unresolved_ids": []})
        )
        self._expect_failure(mini_repo, "raw/grading_results.json")

    def test_done_sentinel_present_fails(self, mini_repo):
        (_mini_run_dir(mini_repo) / "DONE").write_text("")
        self._expect_failure(mini_repo, r"\['DONE'\]")

    def test_non_failed_execution_state_fails(self, mini_repo):
        run_dir = _mini_run_dir(mini_repo)
        status = json.loads((run_dir / "status.json").read_text())
        status["execution_state"] = "completed"
        status["history"][-1] = {"state": "completed", "timestamp": "2026-09-21T14:37:44Z"}
        (run_dir / "status.json").write_text(json.dumps(status, indent=2))
        self._expect_failure(mini_repo, "execution_state is 'completed'")

    def test_publishable_lifecycle_fails(self, mini_repo):
        run_dir = _mini_run_dir(mini_repo)
        status = json.loads((run_dir / "status.json").read_text())
        status["lifecycle"] = "current"
        (run_dir / "status.json").write_text(json.dumps(status, indent=2))
        self._expect_failure(mini_repo, "lifecycle is 'current'")

    def test_missing_termination_attestation_fails(self, mini_repo):
        (_mini_run_dir(mini_repo) / "TERMINATION.json").unlink()
        self._expect_failure(mini_repo, "TERMINATION.json not found")

    def test_termination_claiming_a_score_fails(self, mini_repo):
        run_dir = _mini_run_dir(mini_repo)
        att = json.loads((run_dir / "TERMINATION.json").read_text())
        att["score_claimed"] = True
        (run_dir / "TERMINATION.json").write_text(json.dumps(att, indent=2))
        self._expect_failure(mini_repo, "score_claimed=false")

    def test_termination_mode_must_be_declared_from_evidence(self, mini_repo):
        run_dir = _mini_run_dir(mini_repo)
        att = json.loads((run_dir / "TERMINATION.json").read_text())
        att["mode"] = "completed_normally"
        (run_dir / "TERMINATION.json").write_text(json.dumps(att, indent=2))
        self._expect_failure(mini_repo, "declares termination mode")

    def test_termination_run_id_mismatch_fails(self, mini_repo):
        run_dir = _mini_run_dir(mini_repo)
        att = json.loads((run_dir / "TERMINATION.json").read_text())
        att["run_id"] = "some-other-run"
        (run_dir / "TERMINATION.json").write_text(json.dumps(att, indent=2))
        self._expect_failure(mini_repo, "attests run_id")

    def test_missing_exit_status_file_fails(self, mini_repo):
        (_mini_run_dir(mini_repo) / "raw" / "exit_statuses_1.yaml").unlink()
        self._expect_failure(mini_repo, r"exactly one raw/exit_statuses_\*\.yaml.*found 0")

    def test_ambiguous_multiple_exit_status_files_fail(self, mini_repo):
        raw = _mini_run_dir(mini_repo) / "raw"
        shutil.copyfile(raw / "exit_statuses_1.yaml", raw / "exit_statuses_2.yaml")
        self._expect_failure(mini_repo, r"exactly one raw/exit_statuses_\*\.yaml.*found 2")

    def test_submitted_bucket_disagreeing_with_patches_fails(self, mini_repo):
        run_dir = _mini_run_dir(mini_repo)
        preds = json.loads((run_dir / "raw" / "preds.json").read_text())
        preds[_FAKE_IDS[0]]["model_patch"] = "diff --git a b\n"
        (run_dir / "raw" / "preds.json").write_text(json.dumps(preds, indent=2))
        self._expect_failure(
            mini_repo, "does not match the 'Submitted' exit status bucket"
        )

    # --- Adversarial security: path traversal in snapshot reads --------------

    def test_path_traversal_in_suite_input_hash_key_fails(self, mini_repo):
        """G8: a manifest suite_input_hashes key with '..' components must be
        rejected before any filesystem read occurs — path traversal is a blocker.
        """
        run_dir = _mini_run_dir(mini_repo)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        # Attempt to escape the repo via a traversal key.
        manifest["suite_input_hashes"]["../../etc/passwd"] = "a" * 64
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        self._expect_failure(mini_repo, r"unsafe.*path|path.*traversal|\.\.|\.\.")

    def test_absolute_path_in_suite_input_hash_key_fails(self, mini_repo):
        """G8: an absolute path key in suite_input_hashes must be rejected."""
        run_dir = _mini_run_dir(mini_repo)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        manifest["suite_input_hashes"]["/etc/passwd"] = "a" * 64
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        self._expect_failure(mini_repo, r"unsafe.*path|path.*traversal|absolute")

    # --- Adversarial coverage: partial snapshot mode must not silently skip ---

    def test_partial_snapshots_missing_one_suite_input_fails(self, mini_repo):
        """Once suite_input_snapshots/ exists, ALL suite_input_hashes entries
        must have a snapshot — partial coverage is a blocker, not a fallback.
        """
        run_dir = _mini_run_dir(mini_repo)
        tool = _import_tool()

        # Establish a full snapshot set for the mini run's two inputs.
        suite_yaml = mini_repo / "suite" / "warpcore-v1.yaml"
        instances_json = mini_repo / "suite" / "swebench" / "instances-fake.json"

        suite_bytes = suite_yaml.read_bytes()
        instances_bytes = instances_json.read_bytes()

        manifest = json.loads((run_dir / "manifest.json").read_text())
        manifest["suite_input_hashes"]["suite/warpcore-v1.yaml"] = (
            hashlib.sha256(suite_bytes).hexdigest()
        )
        # instances-fake.json is already in suite_input_hashes; update hash.
        manifest["suite_input_hashes"]["suite/swebench/instances-fake.json"] = (
            hashlib.sha256(instances_bytes).hexdigest()
        )
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        # Create snapshots dir with only one of the two entries — partial coverage.
        snapshots_dir = run_dir / tool.SNAPSHOTS_DIR
        (snapshots_dir / "suite").mkdir(parents=True, exist_ok=True)
        (snapshots_dir / "suite" / "warpcore-v1.yaml").write_bytes(suite_bytes)
        # Intentionally omit "suite/swebench/instances-fake.json" snapshot.

        self._expect_failure(
            mini_repo,
            r"snapshot.*missing|missing.*snapshot|suite_input_snapshots.*instances-fake|partial",
        )

    def test_partial_snapshots_missing_adapter_snapshot_fails(self, mini_repo):
        """Once suite_input_snapshots/ exists, the adapter snapshot must also
        be present (under suite_input_snapshots/adapters/<slug>.yaml).
        """
        run_dir = _mini_run_dir(mini_repo)
        tool = _import_tool()

        adapter_path = mini_repo / "adapters" / "fake-model.yaml"
        adapter_bytes = adapter_path.read_bytes()

        # Create snapshots dir with suite input but NOT the adapter snapshot.
        suite_yaml = mini_repo / "suite" / "warpcore-v1.yaml"
        suite_bytes = suite_yaml.read_bytes()

        manifest = json.loads((run_dir / "manifest.json").read_text())
        manifest["suite_input_hashes"]["suite/warpcore-v1.yaml"] = (
            hashlib.sha256(suite_bytes).hexdigest()
        )
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        snapshots_dir = run_dir / tool.SNAPSHOTS_DIR
        (snapshots_dir / "suite").mkdir(parents=True, exist_ok=True)
        (snapshots_dir / "suite" / "warpcore-v1.yaml").write_bytes(suite_bytes)
        (snapshots_dir / "suite" / "swebench").mkdir(parents=True, exist_ok=True)
        (snapshots_dir / "suite" / "swebench" / "instances-fake.json").write_bytes(
            (mini_repo / "suite" / "swebench" / "instances-fake.json").read_bytes()
        )
        # Intentionally omit the adapter snapshot.

        self._expect_failure(
            mini_repo,
            r"adapter.*snapshot.*missing|missing.*adapter.*snapshot|suite_input_snapshots.*adapter",
        )

    # --- Adversarial coverage: adapter snapshot hash verification ------------

    def test_evolved_adapter_verified_via_snapshot(self, mini_repo):
        """Adapter regression: if an adapter snapshot exists it must be used
        for G8 verification even if the canonical adapter file evolved.
        """
        run_dir = _mini_run_dir(mini_repo)
        tool = _import_tool()

        adapter_path = mini_repo / "adapters" / "fake-model.yaml"
        original_bytes = adapter_path.read_bytes()
        original_hash = hashlib.sha256(original_bytes).hexdigest()

        # Confirm the manifest already records this hash.
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["adapter_hash"] == original_hash

        # Write all suite input snapshots + the adapter snapshot.
        suite_bytes = (mini_repo / "suite" / "warpcore-v1.yaml").read_bytes()
        instances_bytes = (mini_repo / "suite" / "swebench" / "instances-fake.json").read_bytes()

        manifest["suite_input_hashes"]["suite/warpcore-v1.yaml"] = (
            hashlib.sha256(suite_bytes).hexdigest()
        )
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        snapshots_dir = run_dir / tool.SNAPSHOTS_DIR
        (snapshots_dir / "suite").mkdir(parents=True, exist_ok=True)
        (snapshots_dir / "suite" / "warpcore-v1.yaml").write_bytes(suite_bytes)
        (snapshots_dir / "suite" / "swebench").mkdir(parents=True, exist_ok=True)
        (snapshots_dir / "suite" / "swebench" / "instances-fake.json").write_bytes(instances_bytes)
        (snapshots_dir / "adapters").mkdir(parents=True, exist_ok=True)
        (snapshots_dir / "adapters" / "fake-model.yaml").write_bytes(original_bytes)

        # Now evolve the canonical adapter — simulates post-run adapter update.
        adapter_path.write_text(adapter_path.read_text() + "# post-run change\n")
        assert hashlib.sha256(adapter_path.read_bytes()).hexdigest() != original_hash

        # derive() must succeed: adapter snapshot matches declared hash.
        summary = tool.derive(run_dir, repo=mini_repo)
        assert summary["adapter_hash"] == original_hash
        assert "adapter_snapshot" in summary

    def test_tampered_adapter_snapshot_fails_closed(self, mini_repo):
        """G8 fail-closed: a tampered adapter snapshot must cause derive() to fail."""
        run_dir = _mini_run_dir(mini_repo)
        tool = _import_tool()

        adapter_path = mini_repo / "adapters" / "fake-model.yaml"
        original_bytes = adapter_path.read_bytes()
        original_hash = hashlib.sha256(original_bytes).hexdigest()

        suite_bytes = (mini_repo / "suite" / "warpcore-v1.yaml").read_bytes()
        instances_bytes = (mini_repo / "suite" / "swebench" / "instances-fake.json").read_bytes()

        manifest = json.loads((run_dir / "manifest.json").read_text())
        manifest["suite_input_hashes"]["suite/warpcore-v1.yaml"] = (
            hashlib.sha256(suite_bytes).hexdigest()
        )
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        snapshots_dir = run_dir / tool.SNAPSHOTS_DIR
        (snapshots_dir / "suite").mkdir(parents=True, exist_ok=True)
        (snapshots_dir / "suite" / "warpcore-v1.yaml").write_bytes(suite_bytes)
        (snapshots_dir / "suite" / "swebench").mkdir(parents=True, exist_ok=True)
        (snapshots_dir / "suite" / "swebench" / "instances-fake.json").write_bytes(instances_bytes)
        (snapshots_dir / "adapters").mkdir(parents=True, exist_ok=True)
        # Write a tampered adapter snapshot.
        (snapshots_dir / "adapters" / "fake-model.yaml").write_bytes(b"tampered adapter\n")

        with pytest.raises(tool.DiagnosticEvidenceError, match=tool.SNAPSHOTS_DIR):
            tool.derive(run_dir, repo=mini_repo)

    def test_adapter_snapshot_containment_escape_fails(self, mini_repo):
        """Adapter snapshot path must be contained within suite_input_snapshots/ —
        a model slug that resolves outside the snapshot root must be rejected.
        """
        run_dir = _mini_run_dir(mini_repo)
        # Use a manifest with a model slug containing '..' path components.
        # The tool must reject this before attempting any read.
        manifest = json.loads((run_dir / "manifest.json").read_text())
        manifest["model"]["slug"] = "../../evil"
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        self._expect_failure(mini_repo, r"unsafe.*path|path.*traversal|slug|adapter.*path")


class TestVerifyFailsClosedOnTamperedSummary:
    def test_verify_rejects_edited_summary(self, mini_repo):
        tool = _import_tool()
        run_dir = _mini_run_dir(mini_repo)
        summary = _emit_mini(mini_repo)
        tampered = copy.deepcopy(summary)
        tampered["item_inventory"]["unobserved"] = 0
        (run_dir / "diagnostic_summary.json").write_text(json.dumps(tampered, indent=2))
        with pytest.raises(tool.DiagnosticEvidenceError, match="does not match"):
            tool.verify(run_dir, repo=mini_repo)

    def test_verify_rejects_injected_score(self, mini_repo):
        tool = _import_tool()
        run_dir = _mini_run_dir(mini_repo)
        summary = _emit_mini(mini_repo)
        summary["score"] = 0.02
        (run_dir / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2))
        with pytest.raises(tool.DiagnosticEvidenceError):
            tool.verify(run_dir, repo=mini_repo)

    def test_verify_rejects_missing_summary(self, mini_repo):
        tool = _import_tool()
        run_dir = _mini_run_dir(mini_repo)
        with pytest.raises(tool.DiagnosticEvidenceError, match="diagnostic_summary"):
            tool.verify(run_dir, repo=mini_repo)

    def test_cli_verify_exits_nonzero_on_tampered_summary(self, mini_repo):
        tool = _import_tool()
        run_dir = _mini_run_dir(mini_repo)
        summary = _emit_mini(mini_repo)
        summary["publishable"] = True
        (run_dir / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2))
        rc = tool.main(["--run-dir", str(run_dir), "--repo", str(mini_repo), "--verify"])
        assert rc == 1

    def test_cli_verify_exits_zero_on_committed_evidence(self):
        tool = _import_tool()
        rc = tool.main(["--run-dir", str(RUN_DIR), "--repo", str(REPO), "--verify"])
        assert rc == 0


# ---------------------------------------------------------------------------
# 4. The run cannot populate canonical publication
# ---------------------------------------------------------------------------

class TestCannotBecomeCanonical:
    def test_validator_refuses_to_certify_the_run(self):
        import validate_campaign

        result = validate_campaign.validate(RUN_DIR, SUITE, ADAPTER)
        assert result.passed is False
        assert result.eligible is False
        assert any("invalid" in e for e in result.errors), result.errors

    def test_validator_refuses_publication(self):
        import validate_campaign

        result = validate_campaign.validate(
            RUN_DIR, SUITE, ADAPTER, for_publication=True
        )
        assert result.passed is False

    def test_publish_rejects_the_run_and_publishes_nothing(self, tmp_path):
        import publish_campaign

        repo = tmp_path / "repo"
        (repo / "results").mkdir(parents=True)
        shutil.copytree(RUN_DIR, repo / RUN_REL)
        shutil.copytree(REPO / "suite", repo / "suite")
        shutil.copytree(REPO / "adapters", repo / "adapters")

        result = publish_campaign.publish(
            repo, output_path=tmp_path / "canonical_matrix.json"
        )
        assert result.entries == []
        rejected = [r for r in result.rejected if r["run_id"] == RUN_DIR.name]
        assert len(rejected) == 1
        assert rejected[0]["reason"] == "not_validated"

    def test_no_canonical_matrix_entry_names_this_run(self):
        matrix_path = REPO / "viz" / "data" / "canonical_matrix.json"
        if not matrix_path.exists():
            pytest.skip("canonical_matrix.json not generated in this checkout")
        matrix = json.loads(matrix_path.read_text())
        assert RUN_DIR.name not in json.dumps(matrix)

    def test_run_is_not_a_published_swebench_source(self):
        import common

        assert not any(
            RUN_DIR.name in path for path in common.SWEBENCH_RESULTS.values()
        )


# ---------------------------------------------------------------------------
# 5. Registry classification
# ---------------------------------------------------------------------------

REGISTRY_ID = "gpt-oss-120b/swebench/warpcore-v1/gptoss-swebench-n100-20260921"


def _registry_entry() -> dict:
    registry = json.loads((REPO / "results" / "registry.json").read_text())
    matches = [e for e in registry["entries"] if e["id"] == REGISTRY_ID]
    assert len(matches) == 1, f"expected exactly one {REGISTRY_ID!r} entry"
    return matches[0]


class TestRegistryClassification:
    def test_entry_exists_with_nonpublishable_lifecycle(self):
        entry = _registry_entry()
        assert entry["status"] in {"invalid", "diagnostic"}
        assert entry["status"] == "invalid", (
            "The committed status.json records lifecycle 'invalid'; the registry "
            "must not contradict immutable status evidence."
        )
        assert not entry.get("v1_validated")

    def test_entry_registers_every_physical_artifact(self):
        entry = _registry_entry()
        assert sorted(entry["paths"]) == sorted(f"{RUN_REL}/{rel}" for rel in EVIDENCE_FILES)

    def test_entry_explains_the_classification(self):
        entry = _registry_entry()
        assert entry.get("note")
        assert "invalid" in entry["note"].lower()

    def test_registry_still_validates_fail_closed(self):
        import viz_registry

        viz_registry.reload()
        viz_registry.validate_registry(viz_registry.load_registry(), check_paths=True)
        for rel in EVIDENCE_FILES:
            assert viz_registry.status_for_path(f"{RUN_REL}/{rel}") == "invalid"
        assert viz_registry.is_current(f"{RUN_REL}/manifest.json") is False
        viz_registry.reload()


# ---------------------------------------------------------------------------
# 6. Repository prose no longer claims the serving bug is gone
# ---------------------------------------------------------------------------

class TestStaleProseUpdated:
    def test_todo_does_not_claim_the_bug_no_longer_applies(self):
        todo = (REPO / "TODO.md").read_text()
        assert "serving bug that no longer applies" not in todo

    def test_todo_records_the_2026_09_21_reproduction(self):
        todo = (REPO / "TODO.md").read_text()
        assert "gptoss-swebench-n100-20260921" in todo
