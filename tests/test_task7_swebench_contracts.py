"""tests/test_task7_swebench_contracts.py — Task 7: SWE-bench runner/validator
production-contract compatibility tests.

TDD: tests written BEFORE implementation. Must fail (RED) until run_swebench.py
and validate_campaign.py are updated.

Covers the SWE-bench-specific production gaps identified in the audit:

BLOCKER 5 (runner side) — _normalize_generation_artifacts must treat missing
  trajectories as infrastructure_failure dispositions rather than normalization
  failures. Per AGENTS.md: trajectories are absent for pre-agent infrastructure
  failures; this is explicitly normal. The runner must not fail closed when a
  trajectory is missing — it must record that instance as infrastructure_failure
  in exit_statuses.json and proceed.

BLOCKER 5 (validator side) — validate_campaign._check_required_files for
  swebench must verify that raw/trajectories/ directory exists (runner guarantees
  it after normalization). The validator currently checks only preds.json and
  exit_statuses.json; it does not check trajectories/.

BLOCKER 3+4 (SWE-bench) — SwebenchRunner._transition_completed_atomic must
  update manifest.json before writing DONE:
    - timing.completed_utc set to non-null timestamp
    - item_inventory.submitted set to actual resolved+unresolved+etc. count
    - artifact_inventory.preds_json = True (if raw/preds.json exists)
    - artifact_inventory.exit_statuses_json = True (if raw/exit_statuses.json exists)
    - artifact_inventory.grading_results_json = True (if raw/grading_results.json exists)
    - artifact_inventory.run_log = True (if run.log exists)
    - artifact_inventory.command_txt = True (if command.txt exists)
    - artifact_inventory.done_sentinel = True (always — written immediately after)

Suite required_evidence for swebench (from warpcore-v1.yaml):
  preds_json, exit_statuses, run_log, command_txt, manifest_json, status_json,
  done_sentinel. Trajectories are required by the runner contract even though the
  suite YAML lists them under disposition_categories, not required_evidence.

FROZEN IDs: the suite declares exactly 100 instance IDs from
  suite/swebench/instances-seed42-n100.json. All runner and validator checks
  operate against this exact frozen set.

EXIT STATUSES normalized by the runner into exit_statuses.json must map each
  frozen ID to exactly one terminal disposition string. When trajectories are
  missing for an ID (pre-agent infra failure), the runner must NOT fail — it must
  record that ID's exit status as whatever the exit_statuses_*.yaml said (which
  will be an infra-failure code), and omit the trajectory for that ID rather than
  failing the whole normalization.

DISJOINT/EXHAUSTIVE: grading_results.json must cover all 100 IDs across disjoint
  categories. Missing IDs → normalization failure.

TIMING: after successful run, manifest.json timing.completed_utc is a valid ISO
  8601 timestamp, not null.

SUBMITTED COUNT: item_inventory.submitted equals the total count across all
  grading disposition categories.

ARTIFACT INVENTORY: all artifact_inventory booleans set True for every artifact
  that actually exists on disk.

DONE ONLY AFTER: manifest update is durably written before DONE sentinel.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

_TESTS_DIR = pathlib.Path(__file__).parent
_REPO = _TESTS_DIR.parent
_VIZ_DIR = _REPO / "viz"
for _p in (str(_TESTS_DIR), str(_VIZ_DIR), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_swebench  # noqa: E402
from schema_helpers import install_test_qualification  # noqa: E402
import validate_campaign  # noqa: E402

_REAL_SUITE = _REPO / "suite" / "warpcore-v1.yaml"
_SCHEMAS_DIR = _REPO / "suite" / "schemas"
_REAL_INSTANCES = _REPO / "suite" / "swebench" / "instances-seed42-n100.json"

# Minimal frozen IDs for unit tests (subset of real 100)
_FAKE_IDS = ["django__django-001", "django__django-002", "django__django-003"]

_CANONICAL_ADAPTER = {
    "adapter_schema_version": 1,
    "campaign_status": "canonical",
    "model": {
        "slug": "test-canonical-swe",
        "id": "testorg/TestCanonicalSWE",
        "revision": "a" * 40,
    },
    "serving": {
        "image": "testregistry.example.com/test@sha256:" + "b" * 64,
        "engine": "vllm",
        "engine_version": "0.6.6",
        "quantization": "fp8",
        "reasoning_parser": None,
        "tool_call_parser": None,
        "tokenizer": None,
        "moe_backend": None,
        "max_model_len": 300000,
        "gpu_memory_utilization": 0.95,
        "max_num_seqs": 256,
        "environment": {},
    },
}

_PROMPT_TOKEN_MAXIMA = {
    "gsm8k": 500,
    "ifeval": 2000,
    "gpqa_diamond": 1000,
}


def _write_canonical_adapter(path: pathlib.Path) -> None:
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(_CANONICAL_ADAPTER, default_flow_style=False))


def _make_swe_run_dir(
    tmp: pathlib.Path,
    instance_ids: list,
    *,
    run_id: str = "run-test",
    suite_id: str = "warpcore-v1",
) -> pathlib.Path:
    """Create a minimal planned run directory for SWE-bench."""
    slug = _CANONICAL_ADAPTER["model"]["slug"]
    run_dir = tmp / "results" / slug / "runs" / suite_id / "swebench" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "schema_version": 1,
        "run_id": run_id,
        "suite_id": suite_id,
        "execution_state": "planned",
        "lifecycle": "current",
        "history": [{"state": "planned", "timestamp": "2026-09-15T12:00:00Z"}],
    }
    manifest = {
        "suite_id": suite_id,
        "run_id": run_id,
        "benchmark": "swebench",
        "model": {
            "slug": slug,
            "id": "testorg/TestCanonicalSWE",
            "revision": "a" * 40,
        },
        "item_inventory": {"expected": len(instance_ids), "submitted": 0},
        "timing": {"started_utc": "2026-09-15T12:00:00Z", "completed_utc": None},
        "artifact_inventory": {
            "preds_json": False,
            "exit_statuses_json": False,
            "grading_results_json": False,
            "run_log": False,
            "command_txt": False,
            "done_sentinel": False,
        },
    }
    (run_dir / "status.json").write_text(json.dumps(status))
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    return run_dir


def _make_generation_artifacts(
    run_dir: pathlib.Path,
    instance_ids: list,
    *,
    missing_traj_ids: list = None,
) -> None:
    """Write stable post-normalization generation artifacts under raw/.

    Simulates what the normalization step produces:
      - raw/preds.json: dict keyed by instance_id
      - raw/exit_statuses.json: dict instance_id -> exit_status
      - raw/trajectories/<id>.traj: one per id (unless in missing_traj_ids)
      - raw/run.log: nonempty log
    """
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # preds.json
    preds = {iid: {"model_patch": f"diff for {iid}"} for iid in instance_ids}
    (raw_dir / "preds.json").write_text(json.dumps(preds))

    # exit_statuses.json
    statuses = {}
    for iid in instance_ids:
        if missing_traj_ids and iid in missing_traj_ids:
            statuses[iid] = "infrastructure_error"
        else:
            statuses[iid] = "submitted"
    (raw_dir / "exit_statuses.json").write_text(json.dumps(statuses))

    # trajectories/
    traj_dir = raw_dir / "trajectories"
    traj_dir.mkdir(exist_ok=True)
    for iid in instance_ids:
        if missing_traj_ids and iid in missing_traj_ids:
            continue  # intentionally absent for infra failures
        (traj_dir / f"{iid}.traj").write_text(
            json.dumps({"instance_id": iid, "steps": []})
        )

    # run.log
    (raw_dir / "run.log").write_text("mock generation log\n")


def _make_grading_artifacts(
    run_dir: pathlib.Path,
    instance_ids: list,
    *,
    resolved: list = None,
    unresolved: list = None,
    empty_patch: list = None,
    error: list = None,
    incomplete: list = None,
) -> None:
    """Write grading_results.json with the given disposition categories."""
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    grading = {
        "resolved_ids": resolved or [],
        "unresolved_ids": unresolved or instance_ids,
        "empty_patch_ids": empty_patch or [],
        "error_ids": error or [],
        "incomplete_ids": incomplete or [],
    }
    (raw_dir / "grading_results.json").write_text(json.dumps(grading))


# ---------------------------------------------------------------------------
# BLOCKER 5 (runner): _normalize_generation_artifacts must treat missing
# trajectories as infrastructure_failure, not a normalization error
# ---------------------------------------------------------------------------


class TestNormalizeInfraFailureTraj(unittest.TestCase):
    """Missing trajectories must become infrastructure_failure, not normalization failure."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_exit_statuses_yaml(
        self,
        raw_dir: pathlib.Path,
        by_status: dict,
    ) -> pathlib.Path:
        """Write a real mini-swe-agent-style exit_statuses_<ts>.yaml."""
        import yaml
        path = raw_dir / "exit_statuses_1726999999.yaml"
        path.write_text(yaml.dump({"instances_by_exit_status": by_status}))
        return path

    def test_missing_trajectory_treated_as_infra_failure_not_normalization_error(self):
        """When an instance has no trajectory (pre-agent infra failure), normalization
        must succeed and the instance gets its existing exit_status, not an error.

        The runner must NOT fail closed on missing trajectories — infra failures are
        explicitly normal per AGENTS.md and the suite disposition_categories.
        """
        raw_dir = self.tmp / "raw"
        raw_dir.mkdir()

        ids = ["repo__repo-001", "repo__repo-002", "repo__repo-003"]

        # Write preds.json covering all IDs
        preds = {iid: {"model_patch": ""} for iid in ids}
        (raw_dir / "preds.json").write_text(json.dumps(preds))

        # Only repo-001 and repo-002 got to run (have trajectories)
        # repo-003 had an infra failure (docker pull failed) — no trajectory
        self._make_exit_statuses_yaml(raw_dir, {
            "submitted": [ids[0], ids[1]],
            "infrastructure_error": [ids[2]],
        })

        # Write trajectory for ids[0] and ids[1] only
        for iid in ids[:2]:
            d = raw_dir / iid
            d.mkdir()
            (d / f"{iid}.traj.json").write_text(json.dumps({"steps": []}))

        # ids[2] has no trajectory dir at all (pre-agent infra failure)

        errors = run_swebench._normalize_generation_artifacts(raw_dir, ids)

        # KEY ASSERTION: must succeed (no errors) even though ids[2] has no trajectory
        self.assertEqual(
            errors, [],
            f"normalize must not fail when a trajectory is missing for infra-failure "
            f"instance. Got errors: {errors}"
        )

    def test_missing_trajectory_infra_id_gets_infrastructure_failure_exit_status(self):
        """After normalization with infra-failure IDs, exit_statuses.json must cover
        the infra-failure instance with its exit status from the YAML."""
        import yaml

        raw_dir = self.tmp / "raw"
        raw_dir.mkdir()

        ids = ["repo__repo-001", "repo__repo-002"]

        preds = {iid: {"model_patch": ""} for iid in ids}
        (raw_dir / "preds.json").write_text(json.dumps(preds))

        self._make_exit_statuses_yaml(raw_dir, {
            "submitted": [ids[0]],
            "infrastructure_error": [ids[1]],
        })

        # ids[0] has trajectory, ids[1] does not (infra failure)
        d = raw_dir / ids[0]
        d.mkdir()
        (d / f"{ids[0]}.traj.json").write_text(json.dumps({"steps": []}))

        errors = run_swebench._normalize_generation_artifacts(raw_dir, ids)
        self.assertEqual(errors, [], f"normalization should not fail: {errors}")

        # exit_statuses.json must include ids[1] with its YAML status
        es_path = raw_dir / "exit_statuses.json"
        self.assertTrue(es_path.exists(), "exit_statuses.json must exist after normalization")
        es = json.loads(es_path.read_text())
        self.assertIn(ids[1], es,
                      f"infra-failure instance {ids[1]} must appear in exit_statuses.json")
        self.assertEqual(
            es[ids[1]], "infrastructure_error",
            f"infra-failure instance must have its exit status from the YAML; got {es[ids[1]]!r}"
        )

    def test_missing_trajectory_infra_id_not_in_trajectories_dir(self):
        """Infra-failure instances (no trajectory) must NOT appear in trajectories/ dir.

        The runner must not fabricate trajectories for infra-failure instances.
        """
        raw_dir = self.tmp / "raw"
        raw_dir.mkdir()

        ids = ["repo__repo-001", "repo__repo-002"]

        preds = {iid: {"model_patch": ""} for iid in ids}
        (raw_dir / "preds.json").write_text(json.dumps(preds))

        self._make_exit_statuses_yaml(raw_dir, {
            "submitted": [ids[0]],
            "infrastructure_error": [ids[1]],
        })

        d = raw_dir / ids[0]
        d.mkdir()
        (d / f"{ids[0]}.traj.json").write_text(json.dumps({"steps": []}))

        errors = run_swebench._normalize_generation_artifacts(raw_dir, ids)
        self.assertEqual(errors, [], f"normalization should succeed: {errors}")

        traj_dir = raw_dir / "trajectories"
        infra_traj = traj_dir / f"{ids[1]}.traj"
        self.assertFalse(
            infra_traj.exists(),
            f"infra-failure instance must not have a fabricated trajectory at {infra_traj}"
        )

    def test_normalize_still_fails_when_non_infra_trajectory_missing(self):
        """If an ID with a non-infra exit status has no trajectory, that IS a failure.

        Only IDs whose exit_status indicates pre-agent infra failure may lack a trajectory.
        IDs with 'submitted' exit status MUST have a trajectory.
        """
        raw_dir = self.tmp / "raw"
        raw_dir.mkdir()

        ids = ["repo__repo-001", "repo__repo-002"]

        preds = {iid: {"model_patch": ""} for iid in ids}
        (raw_dir / "preds.json").write_text(json.dumps(preds))

        # Both IDs show as 'submitted' in the YAML — but no trajectory for ids[1]
        self._make_exit_statuses_yaml(raw_dir, {
            "submitted": ids,  # both supposedly submitted
        })

        # Only ids[0] has a trajectory
        d = raw_dir / ids[0]
        d.mkdir()
        (d / f"{ids[0]}.traj.json").write_text(json.dumps({"steps": []}))

        # ids[1] has no trajectory but it has exit_status="submitted" (not infra failure)
        errors = run_swebench._normalize_generation_artifacts(raw_dir, ids)
        self.assertNotEqual(
            errors, [],
            "normalization must fail when a non-infra-failure instance has no trajectory"
        )

    def test_normalize_all_trajectories_present_succeeds(self):
        """When all IDs have trajectories (no infra failures), normalization succeeds."""
        raw_dir = self.tmp / "raw"
        raw_dir.mkdir()

        ids = ["repo__repo-001", "repo__repo-002"]

        preds = {iid: {"model_patch": ""} for iid in ids}
        (raw_dir / "preds.json").write_text(json.dumps(preds))

        self._make_exit_statuses_yaml(raw_dir, {
            "submitted": ids,
        })

        for iid in ids:
            d = raw_dir / iid
            d.mkdir()
            (d / f"{iid}.traj.json").write_text(json.dumps({"steps": []}))

        errors = run_swebench._normalize_generation_artifacts(raw_dir, ids)
        self.assertEqual(errors, [], f"all trajectories present must succeed: {errors}")


# ---------------------------------------------------------------------------
# BLOCKER 5 (validator): validate_campaign must check trajectories/ presence
# ---------------------------------------------------------------------------


class TestValidatorTrajectoryCheck(unittest.TestCase):
    """validate_campaign must require raw/trajectories/ for swebench runs."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_swe_run(
        self,
        run_dir: pathlib.Path,
        *,
        with_trajectories: bool = True,
        instance_ids: list = None,
    ) -> pathlib.Path:
        """Build a minimal completed SWE-bench run directory for validator testing."""
        ids = instance_ids or _FAKE_IDS
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # required artifacts
        (run_dir / "command.txt").write_text("mock command\n")
        (run_dir / "run.log").write_text("mock log\n")
        (run_dir / "DONE").write_text("completed\n")
        preds = {iid: {"model_patch": f"diff {iid}"} for iid in ids}
        (raw_dir / "preds.json").write_text(json.dumps(preds))
        statuses = {iid: "submitted" for iid in ids}
        (raw_dir / "exit_statuses.json").write_text(json.dumps(statuses))
        grading = {
            "resolved_ids": [],
            "unresolved_ids": ids,
            "empty_patch_ids": [],
            "error_ids": [],
            "incomplete_ids": [],
        }
        (raw_dir / "grading_results.json").write_text(json.dumps(grading))

        if with_trajectories:
            traj_dir = raw_dir / "trajectories"
            traj_dir.mkdir()
            for iid in ids:
                (traj_dir / f"{iid}.traj").write_text(json.dumps({"steps": []}))

        return run_dir

    def test_validator_requires_trajectories_dir_for_swebench(self):
        """validate_campaign._check_required_files must fail when trajectories/ is absent."""
        run_dir = self.tmp / "run-no-traj"
        self._make_swe_run(run_dir, with_trajectories=False)

        artifact_inv = {
            "preds_json": True,
            "exit_statuses_json": True,
            "grading_results_json": True,
            "run_log": True,
            "command_txt": True,
            "done_sentinel": True,
        }
        errors = []
        validate_campaign._check_required_files(
            run_dir, artifact_inv, errors, benchmark="swebench"
        )
        self.assertTrue(
            any("trajectories" in e for e in errors),
            f"_check_required_files must report missing trajectories/; got: {errors}"
        )

    def test_validator_passes_when_trajectories_dir_present(self):
        """validate_campaign._check_required_files must not error on present trajectories/."""
        run_dir = self.tmp / "run-with-traj"
        self._make_swe_run(run_dir, with_trajectories=True)

        artifact_inv = {
            "preds_json": True,
            "exit_statuses_json": True,
            "grading_results_json": True,
            "run_log": True,
            "command_txt": True,
            "done_sentinel": True,
        }
        errors = []
        validate_campaign._check_required_files(
            run_dir, artifact_inv, errors, benchmark="swebench"
        )
        traj_errors = [e for e in errors if "trajectories" in e]
        self.assertEqual(
            traj_errors, [],
            f"trajectories/ present must not cause errors; got: {traj_errors}"
        )

    def test_validator_trajectory_check_not_applied_to_quality_benchmarks(self):
        """trajectories/ check must only apply to swebench, not quality benchmarks."""
        run_dir = self.tmp / "run-quality"
        run_dir.mkdir(parents=True)
        raw_dir = run_dir / "raw"
        raw_dir.mkdir()

        # Quality benchmark artifact layout (no trajectories)
        (run_dir / "command.txt").write_text("lm-eval command\n")
        (run_dir / "run.log").write_text("lm-eval log\n")
        (run_dir / "DONE").write_text("completed\n")
        (run_dir / "per_item.csv").write_text("item_id,score\n0,1.0\n")
        # Write a fake samples file
        import gzip
        with gzip.open(raw_dir / "samples_gsm8k_cot_2026-01-01.jsonl.gz", "wt") as f:
            f.write('{"doc_id": 0}\n')

        artifact_inv = {
            "samples_jsonl_gz": True,
            "per_item_csv": True,
            "run_log": True,
            "command_txt": True,
            "done_sentinel": True,
        }
        errors = []
        validate_campaign._check_required_files(
            run_dir, artifact_inv, errors, benchmark="gsm8k"
        )
        traj_errors = [e for e in errors if "trajectories" in e]
        self.assertEqual(
            traj_errors, [],
            f"trajectories/ check must not fire for quality benchmarks; got: {traj_errors}"
        )


# ---------------------------------------------------------------------------
# BLOCKER 3+4 (SWE-bench): SwebenchRunner must update manifest on completion
# ---------------------------------------------------------------------------


class TestSwebenchRunnerManifestUpdate(unittest.TestCase):
    """SwebenchRunner must durably update manifest.json before writing DONE."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-swe.yaml"
        _write_canonical_adapter(self.adapter_path)
        # A live launch is gated on a SWE-bench qualification record; install a
        # valid one so these manifest-ordering tests reach the code they test.
        install_test_qualification(
            repo=self.tmp,
            adapter_path=self.adapter_path,
            endpoint="http://fake:8000/v1",
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_runner_with_artifacts(
        self,
        *,
        resolved: list = None,
        unresolved: list = None,
        empty_patch: list = None,
        error_ids: list = None,
        incomplete: list = None,
        run_id: str = "run-swe-test",
    ):
        """Build a SwebenchRunner with injected runners and pre-placed artifacts."""
        ids = _REAL_INSTANCES.read_text()
        instance_ids = json.loads(ids)

        run_dir = _make_swe_run_dir(self.tmp, instance_ids, run_id=run_id)

        # Write command.txt (normally written before generation)
        (run_dir / "command.txt").write_text("mock swe command\n")

        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # preds.json
        preds = {iid: {"model_patch": f"diff for {iid}"} for iid in instance_ids}
        (raw_dir / "preds.json").write_text(json.dumps(preds))

        # exit_statuses.json
        statuses = {iid: "submitted" for iid in instance_ids}
        (raw_dir / "exit_statuses.json").write_text(json.dumps(statuses))

        # trajectories/
        traj_dir = raw_dir / "trajectories"
        traj_dir.mkdir(exist_ok=True)
        for iid in instance_ids:
            (traj_dir / f"{iid}.traj").write_text(json.dumps({"steps": []}))

        # grading_results.json
        _resolved = resolved or []
        _unresolved = unresolved or instance_ids
        _empty_patch = empty_patch or []
        _error_ids = error_ids or []
        _incomplete = incomplete or []
        grading = {
            "resolved_ids": _resolved,
            "unresolved_ids": _unresolved,
            "empty_patch_ids": _empty_patch,
            "error_ids": _error_ids,
            "incomplete_ids": _incomplete,
        }
        (raw_dir / "grading_results.json").write_text(json.dumps(grading))

        # run.log (written by the real generation subprocess; mock it)
        (raw_dir / "run.log").write_text("mock generation log\n")

        def generation_runner(scaffold_config, run_dir_arg):
            # artifacts already placed; generation "succeeded"
            return 0

        def grading_runner(preds_path, run_dir_arg):
            # grading artifacts already placed
            return 0

        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://fake:8000/v1",
            run_dir=run_dir,
            repo=self.tmp,
            allow_no_screen=True,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            preflight_runner=MagicMock(return_value=0),
            generation_runner=generation_runner,
            grading_runner=grading_runner,
        )
        return runner, run_dir

    def test_manifest_completed_utc_set_after_swebench_success(self):
        """timing.completed_utc must be set (non-null) after a successful SWE-bench run."""
        runner, run_dir = self._make_runner_with_artifacts()
        rc = runner.run()
        self.assertEqual(rc, 0, f"Expected exit 0, got {rc}")

        manifest = json.loads((run_dir / "manifest.json").read_text())
        completed_utc = manifest.get("timing", {}).get("completed_utc")
        self.assertIsNotNone(
            completed_utc,
            "timing.completed_utc must be set (non-null) after a successful SWE-bench run"
        )
        self.assertNotEqual(completed_utc, "",
                            "timing.completed_utc must not be empty string")

    def test_manifest_completed_utc_is_valid_datetime(self):
        """timing.completed_utc must be a valid ISO 8601 date-time string."""
        from datetime import datetime
        runner, run_dir = self._make_runner_with_artifacts()
        rc = runner.run()
        self.assertEqual(rc, 0)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        completed_utc = manifest["timing"]["completed_utc"]
        try:
            datetime.fromisoformat(completed_utc.replace("Z", "+00:00"))
        except (ValueError, AttributeError) as e:
            self.fail(
                f"timing.completed_utc {completed_utc!r} is not valid ISO 8601: {e}"
            )

    def test_manifest_submitted_count_updated_after_swebench_success(self):
        """item_inventory.submitted must be updated from 0 to total dispositioned count."""
        instance_ids = json.loads(_REAL_INSTANCES.read_text())
        # 90 unresolved, 10 resolved = 100 total
        resolved = instance_ids[:10]
        unresolved = instance_ids[10:]

        runner, run_dir = self._make_runner_with_artifacts(
            resolved=resolved,
            unresolved=unresolved,
        )
        rc = runner.run()
        self.assertEqual(rc, 0)

        manifest = json.loads((run_dir / "manifest.json").read_text())
        submitted = manifest.get("item_inventory", {}).get("submitted", 0)
        total_dispositioned = len(resolved) + len(unresolved)
        self.assertGreater(submitted, 0,
                           "item_inventory.submitted must be > 0 after successful run")
        self.assertEqual(
            submitted, total_dispositioned,
            f"item_inventory.submitted must equal total dispositioned count "
            f"({total_dispositioned}), got {submitted}"
        )

    def test_manifest_submitted_not_updated_on_grading_failure(self):
        """item_inventory.submitted must remain 0 when grading fails."""
        instance_ids = json.loads(_REAL_INSTANCES.read_text())
        run_dir = _make_swe_run_dir(self.tmp, instance_ids, run_id="run-fail-test")
        (run_dir / "command.txt").write_text("mock command\n")

        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        preds = {iid: {"model_patch": ""} for iid in instance_ids}
        (raw_dir / "preds.json").write_text(json.dumps(preds))
        statuses = {iid: "submitted" for iid in instance_ids}
        (raw_dir / "exit_statuses.json").write_text(json.dumps(statuses))
        traj_dir = raw_dir / "trajectories"
        traj_dir.mkdir()
        for iid in instance_ids:
            (traj_dir / f"{iid}.traj").write_text(json.dumps({"steps": []}))
        (raw_dir / "run.log").write_text("mock log\n")

        def gen_runner(cfg, rd):
            return 0

        def grad_runner(preds_path, rd):
            return 1  # grading fails

        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://fake:8000/v1",
            run_dir=run_dir,
            repo=self.tmp,
            allow_no_screen=True,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            preflight_runner=MagicMock(return_value=0),
            generation_runner=gen_runner,
            grading_runner=grad_runner,
        )
        rc = runner.run()
        self.assertNotEqual(rc, 0)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        submitted = manifest.get("item_inventory", {}).get("submitted", 0)
        self.assertEqual(submitted, 0,
                         "item_inventory.submitted must remain 0 when grading fails")

    def test_manifest_artifact_inventory_updated_on_success(self):
        """artifact_inventory must reflect actual artifacts after successful SWE-bench run."""
        runner, run_dir = self._make_runner_with_artifacts()
        rc = runner.run()
        self.assertEqual(rc, 0)

        manifest = json.loads((run_dir / "manifest.json").read_text())
        inv = manifest.get("artifact_inventory", {})

        raw_dir = run_dir / "raw"
        # Check that artifacts declared True actually exist
        for key, path in [
            ("preds_json", raw_dir / "preds.json"),
            ("exit_statuses_json", raw_dir / "exit_statuses.json"),
            ("grading_results_json", raw_dir / "grading_results.json"),
            ("run_log", raw_dir / "run.log"),
            ("command_txt", run_dir / "command.txt"),
            ("done_sentinel", run_dir / "DONE"),
        ]:
            if path.exists():
                self.assertTrue(
                    inv.get(key),
                    f"artifact_inventory.{key} must be True after successful run "
                    f"(file exists at {path}); got {inv.get(key)!r}"
                )

    def test_manifest_done_sentinel_true_after_success(self):
        """artifact_inventory.done_sentinel must be True after successful run."""
        runner, run_dir = self._make_runner_with_artifacts()
        rc = runner.run()
        self.assertEqual(rc, 0)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        self.assertTrue(
            manifest.get("artifact_inventory", {}).get("done_sentinel"),
            "artifact_inventory.done_sentinel must be True after successful SWE-bench run"
        )

    def test_manifest_written_before_done_sentinel(self):
        """manifest.json must be written (durably) before DONE is written."""
        writes = []
        runner, run_dir = self._make_runner_with_artifacts()

        original_write = pathlib.Path.write_text

        def recording_write(self_path, text, **kw):
            name = self_path.name
            if name in ("manifest.json", "DONE"):
                writes.append(name)
            return original_write(self_path, text, **kw)

        import unittest.mock
        with unittest.mock.patch.object(pathlib.Path, "write_text", recording_write):
            rc = runner.run()

        self.assertEqual(rc, 0)
        if "manifest.json" in writes and "DONE" in writes:
            mi = writes.index("manifest.json")
            di = writes.index("DONE")
            self.assertLess(
                mi, di,
                f"manifest.json must be written before DONE; order was: {writes}"
            )


# ---------------------------------------------------------------------------
# BLOCKER 5 (runner): _verify_generation_evidence trajectory check
# ---------------------------------------------------------------------------


class TestVerifyGenerationEvidenceTrajectories(unittest.TestCase):
    """_verify_generation_evidence must handle infra-failure IDs without trajectories."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_verify_passes_when_trajectories_dir_exists_with_all_non_infra_ids(self):
        """_verify_generation_evidence passes when trajectories/ has files for all
        non-infra-failure IDs. Infra-failure IDs in exit_statuses.json are exempt."""
        ids = ["repo__repo-001", "repo__repo-002"]
        infra_id = "repo__repo-003"
        all_ids = ids + [infra_id]

        run_dir = self.tmp / "run"
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True)

        preds = {iid: {} for iid in all_ids}
        (raw_dir / "preds.json").write_text(json.dumps(preds))

        statuses = {ids[0]: "submitted", ids[1]: "submitted",
                    infra_id: "infrastructure_error"}
        (raw_dir / "exit_statuses.json").write_text(json.dumps(statuses))

        traj_dir = raw_dir / "trajectories"
        traj_dir.mkdir()
        for iid in ids:  # only non-infra IDs get trajectories
            (traj_dir / f"{iid}.traj").write_text(json.dumps({}))

        (raw_dir / "run.log").write_text("log\n")

        errors = run_swebench._verify_generation_evidence(
            run_dir, expected_instance_ids=all_ids
        )
        # Should NOT fail because infra_id is exempt (it's in exit_statuses with infra status)
        # If the current implementation requires all IDs to have trajectories, this will fail RED
        traj_errors = [e for e in errors if "trajectories" in e.lower() or "missing" in e.lower()]
        # This is the RED test: currently it will report missing trajectory for infra_id
        # After fix: must return [] (no errors)
        self.assertEqual(
            errors, [],
            f"_verify_generation_evidence must not require trajectory for infra-failure IDs. "
            f"Got: {errors}"
        )

    def test_verify_fails_when_non_infra_id_missing_trajectory(self):
        """_verify_generation_evidence must still fail if a non-infra-failure ID
        is missing its trajectory."""
        ids = ["repo__repo-001", "repo__repo-002"]
        run_dir = self.tmp / "run2"
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True)

        preds = {iid: {} for iid in ids}
        (raw_dir / "preds.json").write_text(json.dumps(preds))
        statuses = {iid: "submitted" for iid in ids}
        (raw_dir / "exit_statuses.json").write_text(json.dumps(statuses))

        traj_dir = raw_dir / "trajectories"
        traj_dir.mkdir()
        # Only ids[0] has trajectory; ids[1] (non-infra) is missing
        (traj_dir / f"{ids[0]}.traj").write_text(json.dumps({}))

        (raw_dir / "run.log").write_text("log\n")

        errors = run_swebench._verify_generation_evidence(
            run_dir, expected_instance_ids=ids
        )
        self.assertNotEqual(
            errors, [],
            "must fail when non-infra-failure ID is missing trajectory"
        )


# ---------------------------------------------------------------------------
# Frozen IDs contract: preds.json must use exact frozen instance IDs
# ---------------------------------------------------------------------------


class TestFrozenInstanceIDs(unittest.TestCase):
    """Ensure all SWE-bench artifacts use the exact frozen instance IDs."""

    def test_real_instance_ids_file_exists_and_has_100_entries(self):
        """The frozen instance set file must exist and contain exactly 100 IDs."""
        self.assertTrue(
            _REAL_INSTANCES.exists(),
            f"Frozen instances file must exist at {_REAL_INSTANCES}"
        )
        ids = json.loads(_REAL_INSTANCES.read_text())
        self.assertEqual(
            len(ids), 100,
            f"Frozen instance set must contain exactly 100 IDs; got {len(ids)}"
        )
        # All IDs must be strings
        for iid in ids:
            self.assertIsInstance(iid, str,
                                  f"All instance IDs must be strings; got {type(iid).__name__}")

    def test_instance_ids_are_unique(self):
        """All 100 frozen instance IDs must be unique."""
        ids = json.loads(_REAL_INSTANCES.read_text())
        seen = set()
        duplicates = []
        for iid in ids:
            if iid in seen:
                duplicates.append(iid)
            seen.add(iid)
        self.assertEqual(duplicates, [],
                         f"Frozen instance IDs must be unique; duplicates: {duplicates}")


# ---------------------------------------------------------------------------
# Exit status normalization: disjoint and exhaustive
# ---------------------------------------------------------------------------


class TestExitStatusNormalization(unittest.TestCase):
    """exit_statuses.json from normalization must be disjoint and exhaustive."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_yaml(self, raw_dir, by_status):
        import yaml
        path = raw_dir / "exit_statuses_1726999999.yaml"
        path.write_text(yaml.dump({"instances_by_exit_status": by_status}))
        return path

    def test_all_ids_appear_in_exit_statuses_json_after_normalization(self):
        """exit_statuses.json must contain every expected ID exactly once."""
        ids = ["a__a-001", "a__a-002", "a__a-003"]
        raw_dir = self.tmp / "raw"
        raw_dir.mkdir()

        preds = {iid: {} for iid in ids}
        (raw_dir / "preds.json").write_text(json.dumps(preds))

        self._make_yaml(raw_dir, {
            "submitted": [ids[0], ids[1]],
            "infrastructure_error": [ids[2]],
        })

        # trajectory for submitted IDs only
        for iid in ids[:2]:
            d = raw_dir / iid
            d.mkdir()
            (d / f"{iid}.traj.json").write_text("{}")

        errors = run_swebench._normalize_generation_artifacts(raw_dir, ids)
        self.assertEqual(errors, [], f"normalization must succeed: {errors}")

        es = json.loads((raw_dir / "exit_statuses.json").read_text())
        for iid in ids:
            self.assertIn(iid, es,
                          f"all IDs must appear in exit_statuses.json; missing {iid}")

    def test_duplicate_ids_in_yaml_cause_error(self):
        """If the same ID appears in two exit_status buckets, normalization must fail."""
        ids = ["a__a-001", "a__a-002"]
        raw_dir = self.tmp / "raw2"
        raw_dir.mkdir()

        preds = {iid: {} for iid in ids}
        (raw_dir / "preds.json").write_text(json.dumps(preds))

        # ids[0] appears in two buckets (duplicate)
        self._make_yaml(raw_dir, {
            "submitted": ids,
            "infrastructure_error": [ids[0]],  # duplicate!
        })

        errors = run_swebench._normalize_generation_artifacts(raw_dir, ids)
        self.assertNotEqual(errors, [],
                            "duplicate ID in YAML must cause normalization failure")


if __name__ == "__main__":
    unittest.main()
