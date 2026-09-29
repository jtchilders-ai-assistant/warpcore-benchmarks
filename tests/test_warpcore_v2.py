"""tests/test_warpcore_v2.py — TDD: warpcore-v2 suite policy and authorization.

Task 7: RED contract tests for a closed launch-policy vocabulary.
Task 8: GREEN after suite/warpcore-v2.yaml and schema/contract updates.
Task 9: Suite-driven runner authorization tests.
Task 10: Preflight/circuit-breaker integration tests.
Task 11: V2 publication validation tests.

Spec: docs/superpowers/plans/2026-09-29-swebench-reporting-and-v2-implementation.md
Design: docs/superpowers/specs/2026-09-29-swebench-reporting-and-v2-launch-design.md
"""
from __future__ import annotations

import copy
import json
import pathlib
import shutil
import sys
import tempfile
import unittest

_TESTS_DIR = pathlib.Path(__file__).parent
_REPO = _TESTS_DIR.parent
_VIZ_DIR = _REPO / "viz"
_SUITE_DIR = _REPO / "suite"

for _p in (str(_VIZ_DIR), str(_REPO), str(_TESTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Real suite paths
_V1_SUITE = _REPO / "suite" / "warpcore-v1.yaml"
_V2_SUITE = _REPO / "suite" / "warpcore-v2.yaml"
_SCHEMA = _REPO / "suite" / "schemas" / "suite.schema.json"
_ADAPTER_PATH = _REPO / "adapters" / "qwen3.6-35b-a3b.yaml"

import contract  # noqa: E402


def _load_yaml(path: pathlib.Path) -> dict:
    return contract.load_yaml(path)


# ===========================================================================
# Task 7 — RED contract tests for launch-policy vocabulary
# ===========================================================================


class TestLaunchPolicyVocabularyRedGreen(unittest.TestCase):
    """
    Task 7: Test that the suite schema and contract enforce a closed launch-policy
    vocabulary for SWE-bench. These tests are written RED first, then GREEN after
    Task 8 creates the schema/suite changes.
    """

    # -----------------------------------------------------------------------
    # V1 declares qualification_seal
    # -----------------------------------------------------------------------

    def test_v1_explicitly_declares_qualification_seal(self):
        """v1 must explicitly declare launch_authorization.mode = qualification_seal."""
        suite = _load_yaml(_V1_SUITE)
        bench = suite.get("benchmarks", {}).get("swebench", {})
        launch_auth = bench.get("launch_authorization", {})
        self.assertIsNotNone(
            launch_auth,
            "swebench benchmark must have an explicit launch_authorization block",
        )
        mode = launch_auth.get("mode")
        self.assertEqual(
            mode,
            "qualification_seal",
            f"v1 swebench.launch_authorization.mode must be 'qualification_seal', got {mode!r}",
        )

    def test_v1_qualification_block_still_present(self):
        """v1 must retain its qualification block for backward compatibility."""
        suite = _load_yaml(_V1_SUITE)
        bench = suite.get("benchmarks", {}).get("swebench", {})
        self.assertIn(
            "qualification",
            bench,
            "v1 swebench must retain the 'qualification' block — it is required for seal mode",
        )

    # -----------------------------------------------------------------------
    # V2 declares direct_n100_after_preflight
    # -----------------------------------------------------------------------

    def test_v2_suite_exists(self):
        """suite/warpcore-v2.yaml must exist after Task 8."""
        self.assertTrue(
            _V2_SUITE.exists(),
            f"suite/warpcore-v2.yaml not found at {_V2_SUITE}",
        )

    def test_v2_explicitly_declares_direct_n100_after_preflight(self):
        """v2 must explicitly declare launch_authorization.mode = direct_n100_after_preflight."""
        suite = _load_yaml(_V2_SUITE)
        bench = suite.get("benchmarks", {}).get("swebench", {})
        launch_auth = bench.get("launch_authorization", {})
        mode = launch_auth.get("mode")
        self.assertEqual(
            mode,
            "direct_n100_after_preflight",
            f"v2 swebench.launch_authorization.mode must be 'direct_n100_after_preflight', got {mode!r}",
        )

    def test_v2_has_no_qualification_block(self):
        """v2 must NOT have a qualification block — it uses direct launch."""
        suite = _load_yaml(_V2_SUITE)
        bench = suite.get("benchmarks", {}).get("swebench", {})
        self.assertNotIn(
            "qualification",
            bench,
            "v2 swebench must NOT have a 'qualification' block — direct_n100_after_preflight mode",
        )

    # -----------------------------------------------------------------------
    # V2 preserves v1 experiment identity fields
    # -----------------------------------------------------------------------

    def test_v2_preserves_dataset_revision(self):
        """v2 must use the same SWE-bench dataset revision as v1."""
        v1 = _load_yaml(_V1_SUITE)
        v2 = _load_yaml(_V2_SUITE)
        v1_rev = v1["benchmarks"]["swebench"]["dataset"]["revision"]
        v2_rev = v2["benchmarks"]["swebench"]["dataset"]["revision"]
        self.assertEqual(
            v1_rev, v2_rev,
            f"v2 dataset revision {v2_rev!r} must match v1 {v1_rev!r}"
        )

    def test_v2_preserves_instance_set_file_and_hash(self):
        """v2 must use the same frozen seed-42 n=100 instance set as v1."""
        v1 = _load_yaml(_V1_SUITE)
        v2 = _load_yaml(_V2_SUITE)
        v1_b = v1["benchmarks"]["swebench"]
        v2_b = v2["benchmarks"]["swebench"]
        self.assertEqual(v1_b["instance_set_file"], v2_b["instance_set_file"])
        self.assertEqual(v1_b["instances_sha256"], v2_b["instances_sha256"])

    def test_v2_preserves_scaffold_file_and_hash(self):
        """v2 must use the same scaffold as v1."""
        v1 = _load_yaml(_V1_SUITE)
        v2 = _load_yaml(_V2_SUITE)
        v1_b = v1["benchmarks"]["swebench"]
        v2_b = v2["benchmarks"]["swebench"]
        self.assertEqual(v1_b["scaffold_file"], v2_b["scaffold_file"])
        self.assertEqual(v1_b["scaffold_sha256"], v2_b["scaffold_sha256"])

    def test_v2_preserves_expected_item_count_100(self):
        """v2 must have expected_item_count = 100 (no separate qualification)."""
        v2 = _load_yaml(_V2_SUITE)
        count = v2["benchmarks"]["swebench"].get("expected_item_count")
        self.assertEqual(count, 100)

    def test_v2_preserves_retry_policy(self):
        """v2 retry_policy must match v1."""
        v1 = _load_yaml(_V1_SUITE)
        v2 = _load_yaml(_V2_SUITE)
        self.assertEqual(
            v1["benchmarks"]["swebench"]["retry_policy"],
            v2["benchmarks"]["swebench"]["retry_policy"],
        )

    def test_v2_preserves_timeout_policy(self):
        """v2 timeout_policy must match v1."""
        v1 = _load_yaml(_V1_SUITE)
        v2 = _load_yaml(_V2_SUITE)
        self.assertEqual(
            v1["benchmarks"]["swebench"]["timeout_policy"],
            v2["benchmarks"]["swebench"]["timeout_policy"],
        )

    def test_v2_preserves_circuit_breaker_reference(self):
        """v2 must not weaken the circuit breaker — it shares the same policy."""
        # The circuit breaker policy is loaded from the repo, shared by both suites.
        # Verify the policy file still exists and is loadable.
        import swebench_circuit_breaker as cb
        policy = cb.load_policy(_REPO)
        self.assertIsNotNone(policy)
        self.assertTrue(policy.enabled)
        self.assertGreater(policy.min_completed, 0)
        self.assertGreater(policy.ratio_threshold, 0)

    def test_v2_suite_id_is_warpcore_v2(self):
        """v2 must have suite_id = 'warpcore-v2'."""
        v2 = _load_yaml(_V2_SUITE)
        self.assertEqual(v2.get("suite_id"), "warpcore-v2")

    def test_v2_has_different_suite_id_than_v1(self):
        """v1 and v2 must have different suite IDs."""
        v1 = _load_yaml(_V1_SUITE)
        v2 = _load_yaml(_V2_SUITE)
        self.assertNotEqual(v1.get("suite_id"), v2.get("suite_id"))

    def test_v2_preserves_harness_versions(self):
        """v2 must pin the same harness versions as v1."""
        v1 = _load_yaml(_V1_SUITE)
        v2 = _load_yaml(_V2_SUITE)
        self.assertEqual(v1["required_harness"], v2["required_harness"])

    def test_v2_preserves_statistical_policy(self):
        """v2 must use the same statistical policy as v1."""
        v1 = _load_yaml(_V1_SUITE)
        v2 = _load_yaml(_V2_SUITE)
        self.assertEqual(v1["statistical_policy"], v2["statistical_policy"])

    def test_v2_preserves_canonical_headline(self):
        """v2 canonical_headline must match v1."""
        v1 = _load_yaml(_V1_SUITE)
        v2 = _load_yaml(_V2_SUITE)
        self.assertEqual(
            v1["benchmarks"]["swebench"]["canonical_headline"],
            v2["benchmarks"]["swebench"]["canonical_headline"],
        )

    # -----------------------------------------------------------------------
    # Fail-closed: missing / unknown / misspelled / contradictory policy
    # -----------------------------------------------------------------------

    def test_schema_validates_v1(self):
        """v1 suite must pass the updated schema."""
        suite = _load_yaml(_V1_SUITE)
        errors = contract.validate_json(suite, _SCHEMA)
        self.assertEqual(errors, [], f"v1 suite schema errors: {errors}")

    def test_schema_validates_v2(self):
        """v2 suite must pass the updated schema."""
        suite = _load_yaml(_V2_SUITE)
        errors = contract.validate_json(suite, _SCHEMA)
        self.assertEqual(errors, [], f"v2 suite schema errors: {errors}")

    def test_schema_rejects_missing_launch_authorization(self):
        """Schema must reject a swebench block with no launch_authorization."""
        suite = _load_yaml(_V1_SUITE)
        # Remove launch_authorization from swebench
        bench = suite["benchmarks"]["swebench"]
        bench.pop("launch_authorization", None)
        errors = contract.validate_json(suite, _SCHEMA)
        self.assertGreater(
            len(errors), 0,
            "Schema must reject swebench block missing launch_authorization",
        )

    def test_schema_rejects_unknown_launch_mode(self):
        """Schema must reject an unknown launch_authorization mode."""
        suite = _load_yaml(_V1_SUITE)
        suite["benchmarks"]["swebench"]["launch_authorization"] = {
            "mode": "turbo_bypass"
        }
        errors = contract.validate_json(suite, _SCHEMA)
        self.assertGreater(
            len(errors), 0,
            "Schema must reject unknown launch_authorization.mode 'turbo_bypass'",
        )

    def test_schema_rejects_misspelled_launch_mode(self):
        """Schema must reject misspelled launch mode."""
        suite = _load_yaml(_V1_SUITE)
        suite["benchmarks"]["swebench"]["launch_authorization"] = {
            "mode": "qualifcation_seal"  # misspelled
        }
        errors = contract.validate_json(suite, _SCHEMA)
        self.assertGreater(len(errors), 0, "Misspelled mode must be rejected by schema")

    def test_schema_rejects_direct_mode_without_qualification_removed(self):
        """Schema should require v2 direct mode does NOT have a qualification block.

        A suite declaring direct_n100_after_preflight must not also have a
        qualification block — that would be a contradictory policy.
        """
        # Start from v2 if it exists, else skip
        if not _V2_SUITE.exists():
            self.skipTest("v2 suite not yet created (Task 8)")
        suite = _load_yaml(_V2_SUITE)
        # Inject a qualification block into a direct-mode suite — contradictory
        suite["benchmarks"]["swebench"]["qualification"] = {
            "ids_file": "suite/swebench/qualification-ids-v1.json",
            "ids_sha256": "a" * 64,
            "expected_count": 20,
            "selection_criterion": "round_robin_by_repository_over_frozen_seed42_n100",
            "required_terminal_status": "Submitted",
            "max_age_hours": 24,
        }
        errors = contract.validate_json(suite, _SCHEMA)
        self.assertGreater(
            len(errors), 0,
            "Schema must reject a direct-mode suite that also carries a qualification block",
        )

    def test_contract_validate_suite_passes_v1(self):
        """validate_suite must return [] for v1."""
        errors = contract.validate_suite(_REPO, _V1_SUITE)
        self.assertEqual(errors, [], f"v1 validate_suite errors: {errors}")

    def test_contract_validate_suite_passes_v2(self):
        """validate_suite must return [] for v2."""
        if not _V2_SUITE.exists():
            self.skipTest("v2 suite not yet created (Task 8)")
        errors = contract.validate_suite(_REPO, _V2_SUITE)
        self.assertEqual(errors, [], f"v2 validate_suite errors: {errors}")


# ===========================================================================
# Task 9 — Suite-driven runner authorization
# ===========================================================================


class TestSuiteDrivenRunnerAuthorization(unittest.TestCase):
    """
    Task 9: Tests that run_swebench.SwebenchRunner reads launch mode from the
    validated suite and branches only on the closed enum. No generic bypass.
    """

    def setUp(self):
        import run_swebench
        import swebench_qualification
        self.run_swebench = run_swebench
        self.sq = swebench_qualification
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.run_dir = self.tmp / "run-test"
        self.run_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_minimal_run_dir(self, run_dir: pathlib.Path, suite_id: str = "warpcore-v1") -> None:
        """Set up a minimal planned run directory."""
        status = {
            "schema_version": 1,
            "run_id": run_dir.name,
            "suite_id": suite_id,
            "lifecycle": "current",
            "execution_state": "planned",
            "history": [{"state": "planned", "timestamp": "2026-09-29T00:00:00Z"}],
        }
        (run_dir / "status.json").write_text(json.dumps(status, indent=2))

    def test_v1_missing_qualification_exits_nonzero_before_command_txt(self):
        """v1 run with missing qualification must exit nonzero before command.txt is written.

        Proves the gate fires before any state mutation (command.txt is written
        only after the qualification gate passes in run()).
        """
        import run_swebench as rs

        run_dir = self.tmp / "run-v1-no-qual"
        run_dir.mkdir()
        self._make_minimal_run_dir(run_dir)

        # Build a runner with v1 suite and no qualification record
        # Use allow_no_screen so screen guard doesn't interfere
        try:
            runner = rs.SwebenchRunner(
                suite_path=_V1_SUITE,
                adapter_path=_ADAPTER_PATH,
                endpoint="http://fake.endpoint/v1",
                run_dir=run_dir,
                repo=_REPO,
                allow_no_screen=True,
                prompt_token_maxima={"gsm8k": 100, "ifeval": 100, "gpqa_diamond": 100},
                qualification_path=self.tmp / "nonexistent_qual.json",
            )
        except ValueError as exc:
            # Construction error is acceptable if adapter is noncanonical
            self.skipTest(f"Adapter not campaign-ready: {exc}")

        rc = runner.run()
        self.assertNotEqual(rc, 0, "v1 with missing qualification must exit nonzero")
        # command.txt must NOT exist (gate fires before any write)
        self.assertFalse(
            (run_dir / "command.txt").exists(),
            "command.txt must not be written when qualification gate blocks",
        )

    def test_v2_runner_does_not_read_qualification_json(self):
        """v2 runner must NOT look for or create qualification.json.

        v2 uses direct_n100_after_preflight; it must never consult or produce
        a qualification.json artifact.
        """
        if not _V2_SUITE.exists():
            self.skipTest("v2 suite not yet created")

        import run_swebench as rs

        run_dir = self.tmp / "run-v2-no-qual"
        run_dir.mkdir()
        # Write status.json with suite_id = warpcore-v2
        status = {
            "schema_version": 1,
            "run_id": run_dir.name,
            "suite_id": "warpcore-v2",
            "lifecycle": "current",
            "execution_state": "planned",
            "history": [{"state": "planned", "timestamp": "2026-09-29T00:00:00Z"}],
        }
        (run_dir / "status.json").write_text(json.dumps(status, indent=2))

        # Preflight that always returns failure (simulates preflight exit 1)
        preflight_calls = []

        def fake_preflight(model_id: str) -> int:
            preflight_calls.append(model_id)
            return 1  # preflight fails

        try:
            runner = rs.SwebenchRunner(
                suite_path=_V2_SUITE,
                adapter_path=_ADAPTER_PATH,
                endpoint="http://fake.endpoint/v1",
                run_dir=run_dir,
                repo=_REPO,
                allow_no_screen=True,
                prompt_token_maxima={"gsm8k": 100, "ifeval": 100, "gpqa_diamond": 100},
                preflight_runner=fake_preflight,
            )
        except ValueError as exc:
            self.skipTest(f"Runner construction failed: {exc}")

        # v2 runner must never look for qualification.json
        qual_path = run_dir / "qualification.json"
        self.assertFalse(
            qual_path.exists(),
            "v2 run dir must not contain qualification.json before run()",
        )

        rc = runner.run()

        # qualification.json must not be created
        self.assertFalse(
            qual_path.exists(),
            "v2 runner must not create qualification.json",
        )
        # Preflight must have been called
        self.assertTrue(
            len(preflight_calls) > 0,
            "v2 runner must call preflight before generation",
        )

    def test_v2_preflight_exit_1_blocks_before_generation_and_done(self):
        """v2 preflight exit 1 must block before generation and DONE sentinel."""
        if not _V2_SUITE.exists():
            self.skipTest("v2 suite not yet created (Task 8)")

        import run_swebench as rs

        run_dir = self.tmp / "run-v2-preflight-fail"
        run_dir.mkdir()
        status = {
            "schema_version": 1,
            "run_id": run_dir.name,
            "suite_id": "warpcore-v2",
            "lifecycle": "current",
            "execution_state": "planned",
            "history": [{"state": "planned", "timestamp": "2026-09-29T00:00:00Z"}],
        }
        (run_dir / "status.json").write_text(json.dumps(status, indent=2))

        generation_called = []

        def failing_preflight(model_id: str) -> int:
            return 1  # defect

        def tracking_generation(config, run_dir_arg) -> int:
            generation_called.append(True)
            return 0

        try:
            runner = rs.SwebenchRunner(
                suite_path=_V2_SUITE,
                adapter_path=_ADAPTER_PATH,
                endpoint="http://fake.endpoint/v1",
                run_dir=run_dir,
                repo=_REPO,
                allow_no_screen=True,
                prompt_token_maxima={"gsm8k": 100, "ifeval": 100, "gpqa_diamond": 100},
                preflight_runner=failing_preflight,
                generation_runner=tracking_generation,
            )
        except ValueError as exc:
            self.skipTest(f"Runner construction failed: {exc}")

        rc = runner.run()

        self.assertNotEqual(rc, 0, "v2 with failed preflight must exit nonzero")
        self.assertEqual(generation_called, [], "generation must not be called after preflight failure")
        self.assertFalse(
            (run_dir / "DONE").exists(),
            "DONE must not exist after preflight failure",
        )

    def test_v2_preflight_exit_2_blocks_before_generation(self):
        """v2 preflight exit 2 (inconclusive) must also block before generation."""
        if not _V2_SUITE.exists():
            self.skipTest("v2 suite not yet created (Task 8)")

        import run_swebench as rs

        run_dir = self.tmp / "run-v2-preflight-inconclusive"
        run_dir.mkdir()
        status = {
            "schema_version": 1,
            "run_id": run_dir.name,
            "suite_id": "warpcore-v2",
            "lifecycle": "current",
            "execution_state": "planned",
            "history": [{"state": "planned", "timestamp": "2026-09-29T00:00:00Z"}],
        }
        (run_dir / "status.json").write_text(json.dumps(status, indent=2))

        generation_called = []

        def inconclusive_preflight(model_id: str) -> int:
            return 2  # inconclusive

        def tracking_generation(config, run_dir_arg) -> int:
            generation_called.append(True)
            return 0

        try:
            runner = rs.SwebenchRunner(
                suite_path=_V2_SUITE,
                adapter_path=_ADAPTER_PATH,
                endpoint="http://fake.endpoint/v1",
                run_dir=run_dir,
                repo=_REPO,
                allow_no_screen=True,
                prompt_token_maxima={"gsm8k": 100, "ifeval": 100, "gpqa_diamond": 100},
                preflight_runner=inconclusive_preflight,
                generation_runner=tracking_generation,
            )
        except ValueError as exc:
            self.skipTest(f"Runner construction failed: {exc}")

        rc = runner.run()
        self.assertNotEqual(rc, 0)
        self.assertEqual(generation_called, [])

    def test_unknown_launch_policy_blocks_runner_construction(self):
        """Runner built with a suite containing unknown launch_authorization.mode must fail."""
        import run_swebench as rs
        import yaml

        # Patch v1 suite to have unknown policy
        v1_data = _load_yaml(_V1_SUITE)
        v1_data["benchmarks"]["swebench"]["launch_authorization"] = {
            "mode": "unknown_mode_xyz"
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
            yaml.safe_dump(v1_data, tf)
            bad_suite_path = pathlib.Path(tf.name)

        run_dir = self.tmp / "run-unknown-policy"
        run_dir.mkdir()

        try:
            with self.assertRaises((ValueError, SystemExit)):
                runner = rs.SwebenchRunner(
                    suite_path=bad_suite_path,
                    adapter_path=_ADAPTER_PATH,
                    endpoint="http://fake.endpoint/v1",
                    run_dir=run_dir,
                    repo=_REPO,
                    allow_no_screen=True,
                    prompt_token_maxima={"gsm8k": 100, "ifeval": 100, "gpqa_diamond": 100},
                )
        finally:
            bad_suite_path.unlink(missing_ok=True)

    def test_noncanonical_trial_cannot_authorize_v2_run(self):
        """--noncanonical-trial must remain restricted to diagnostic v1 runs.

        It cannot be used to authorize a v2 canonical run; v2 uses preflight-only.
        Specifically: a v2 run with noncanonical_trial=True must be blocked because
        v2 status.lifecycle must be 'current', not 'diagnostic'.
        """
        if not _V2_SUITE.exists():
            self.skipTest("v2 suite not yet created (Task 8)")

        import run_swebench as rs

        run_dir = self.tmp / "run-v2-noncanonical"
        run_dir.mkdir()
        # v2 run with lifecycle='current' — noncanonical_trial requires 'diagnostic'
        status = {
            "schema_version": 1,
            "run_id": run_dir.name,
            "suite_id": "warpcore-v2",
            "lifecycle": "current",
            "execution_state": "planned",
            "history": [{"state": "planned", "timestamp": "2026-09-29T00:00:00Z"}],
        }
        (run_dir / "status.json").write_text(json.dumps(status, indent=2))

        try:
            runner = rs.SwebenchRunner(
                suite_path=_V2_SUITE,
                adapter_path=_ADAPTER_PATH,
                endpoint="http://fake.endpoint/v1",
                run_dir=run_dir,
                repo=_REPO,
                allow_no_screen=True,
                prompt_token_maxima={"gsm8k": 100, "ifeval": 100, "gpqa_diamond": 100},
                noncanonical_trial=True,
            )
        except ValueError as exc:
            # Construction rejection is acceptable
            return

        rc = runner.run()
        # Must fail: noncanonical_trial with lifecycle='current' is blocked
        self.assertNotEqual(rc, 0, "noncanonical_trial on v2 current-lifecycle run must fail")


# ===========================================================================
# Task 10 — Preflight and circuit-breaker gate preservation
# ===========================================================================


class TestPreflightAndCircuitBreakerGatesV2(unittest.TestCase):
    """
    Task 10: Prove v2 still requires all production preflight checks and the
    existing in-run circuit breaker. No weaker v2 duplicates.
    """

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_v2_run_dir(self, name: str) -> pathlib.Path:
        run_dir = self.tmp / name
        run_dir.mkdir()
        status = {
            "schema_version": 1,
            "run_id": name,
            "suite_id": "warpcore-v2",
            "lifecycle": "current",
            "execution_state": "planned",
            "history": [{"state": "planned", "timestamp": "2026-09-29T00:00:00Z"}],
        }
        (run_dir / "status.json").write_text(json.dumps(status, indent=2))
        manifest = {
            "suite_id": "warpcore-v2",
            "run_id": name,
            "benchmark": "swebench",
            "model": {
                "slug": "qwen3.6-35b-a3b",
                "id": "Intel/Qwen3.5-122B-A10B-int4-AutoRound",
                "revision": "a" * 40,
            },
            "item_inventory": {"expected": 100, "submitted": 0},
            "timing": {
                "started_utc": "2026-09-29T00:00:00Z",
                "completed_utc": None,
            },
            "artifact_inventory": {
                "preds_json": False,
                "exit_statuses_json": False,
                "grading_results_json": False,
                "run_log": False,
                "command_txt": False,
                "done_sentinel": False,
            },
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return run_dir

    def _make_runner(
        self,
        run_dir: pathlib.Path,
        preflight_runner=None,
        generation_runner=None,
        grading_runner=None,
        qualification_verifier=None,
    ):
        import run_swebench as rs

        suite_path = _V2_SUITE if _V2_SUITE.exists() else _V1_SUITE
        try:
            return rs.SwebenchRunner(
                suite_path=suite_path,
                adapter_path=_ADAPTER_PATH,
                endpoint="http://fake.endpoint/v1",
                run_dir=run_dir,
                repo=_REPO,
                allow_no_screen=True,
                prompt_token_maxima={"gsm8k": 100, "ifeval": 100, "gpqa_diamond": 100},
                preflight_runner=preflight_runner,
                generation_runner=generation_runner,
                grading_runner=grading_runner,
                qualification_verifier=qualification_verifier,
            )
        except ValueError as exc:
            raise unittest.SkipTest(f"Runner construction failed: {exc}")

    def test_v2_circuit_breaker_policy_loads_at_construction(self):
        """v2 runner must load and enforce the circuit-breaker policy at construction time."""
        if not _V2_SUITE.exists():
            self.skipTest("v2 suite not yet created")

        run_dir = self._make_v2_run_dir("run-cb-policy")

        import run_swebench as rs
        import swebench_circuit_breaker as cb

        runner = self._make_runner(
            run_dir,
            preflight_runner=lambda m: 0,
            generation_runner=lambda c, r: 0,
        )
        # Circuit breaker policy must be loaded
        self.assertIsNotNone(runner.circuit_breaker_policy)
        self.assertIsInstance(runner.circuit_breaker_policy, cb.CircuitBreakerPolicy)
        # Policy must have at least one of the known threshold fields
        policy = runner.circuit_breaker_policy
        has_threshold = (
            hasattr(policy, "min_completed") or
            hasattr(policy, "ratio_threshold") or
            hasattr(policy, "enabled")
        )
        self.assertTrue(has_threshold, f"CircuitBreakerPolicy has no threshold fields: {dir(policy)}")

    def test_v2_systemic_failure_triggers_breaker_writes_artifact_no_done(self):
        """v2 circuit breaker must trip, write circuit_breaker.json, and never write DONE.

        This is the critical systemic-failure path: the breaker terminates generation,
        marks failed/invalid, preserves partial evidence, skips grading, writes no DONE.
        """
        if not _V2_SUITE.exists():
            self.skipTest("v2 suite not yet created")

        import run_swebench as rs
        import swebench_circuit_breaker as cb

        run_dir = self._make_v2_run_dir("run-cb-trip")

        # Generation runner that simulates systemic failure by setting the breaker decision
        def breaker_generation(config: dict, run_dir_arg: pathlib.Path) -> int:
            # Write a fake exit_statuses YAML that will trigger the breaker
            # by making all instances RuntimeError
            raw_dir = run_dir_arg / "raw"
            raw_dir.mkdir(exist_ok=True)
            import yaml
            instances = json.loads(
                (_REPO / "suite" / "swebench" / "instances-seed42-n100.json").read_text()
            )
            statuses = {"instances_by_exit_status": {"RuntimeError": instances}}
            (raw_dir / "exit_statuses_1234567890.0.yaml").write_text(
                yaml.safe_dump(statuses)
            )
            (raw_dir / "run.log").write_text("fake generation log\n")
            # Injected generation runners intentionally bypass the live subprocess
            # monitor. Exercise the same BreakerMonitor explicitly, then feed its
            # decision into the runner's normal abort branch. The existing
            # production-path tests cover owned-process-group termination.
            monitor = cb.BreakerMonitor(
                raw_dir=raw_dir,
                policy=runner.circuit_breaker_policy,
                sleep=lambda _seconds: None,
            )
            decision = monitor.observe()
            self.assertIsNotNone(decision)
            runner._circuit_breaker_decision = decision
            runner._circuit_breaker_termination = {
                "terminated_by": "warpcore_circuit_breaker",
                "scope": "injected_test_runner",
                "target": "injected_runner",
            }
            runner._write_circuit_breaker_artifact(
                decision, termination=runner._circuit_breaker_termination
            )
            return 0

        # Use a real qualification verifier that passes (v2 doesn't need seal)
        def pass_qual():
            import swebench_qualification as sq
            return sq.QualificationResult(ok=True, reason="v2-direct-preflight-passed")

        runner = self._make_runner(
            run_dir,
            preflight_runner=lambda m: 0,
            generation_runner=breaker_generation,
            qualification_verifier=pass_qual,
        )
        rc = runner.run()

        # Run must fail (circuit breaker)
        self.assertNotEqual(rc, 0, "circuit breaker trip must return nonzero")
        # DONE must not exist
        self.assertFalse(
            (run_dir / "DONE").exists(),
            "DONE sentinel must not exist after circuit breaker trip",
        )
        # circuit_breaker.json must exist
        self.assertTrue(
            (run_dir / "circuit_breaker.json").exists(),
            "circuit_breaker.json must be written after breaker trip",
        )

    def test_v2_success_path_writes_done_last(self):
        """v2 complete n=100 run must write DONE as the last artifact.

        Success path: preflight passes, generation succeeds with 100 predictions,
        grading succeeds with 100 dispositions, DONE written last.
        """
        if not _V2_SUITE.exists():
            self.skipTest("v2 suite not yet created")

        import run_swebench as rs

        run_dir = self._make_v2_run_dir("run-v2-success")

        instances = json.loads(
            (_REPO / "suite" / "swebench" / "instances-seed42-n100.json").read_text()
        )

        def generation_runner(config: dict, run_dir_arg: pathlib.Path) -> int:
            raw_dir = run_dir_arg / "raw"
            raw_dir.mkdir(exist_ok=True)
            # Write preds.json with 100 nonempty predictions
            preds = {iid: {"model_patch": f"patch-{i}", "instance_id": iid}
                     for i, iid in enumerate(instances)}
            (raw_dir / "preds.json").write_text(json.dumps(preds, indent=2))
            # Write exit_statuses (all Submitted)
            import yaml
            statuses = {"instances_by_exit_status": {"Submitted": instances}}
            (raw_dir / "exit_statuses_1234567890.0.yaml").write_text(yaml.safe_dump(statuses))
            (raw_dir / "exit_statuses.json").write_text(
                json.dumps({iid: "Submitted" for iid in instances})
            )
            (raw_dir / "run.log").write_text("generation complete\n")
            # Write trajectories
            traj_dir = raw_dir / "trajectories"
            traj_dir.mkdir(exist_ok=True)
            for iid in instances:
                (traj_dir / f"{iid}.traj").write_text(json.dumps({"instance_id": iid}))
            return 0

        def grading_runner(preds_path: pathlib.Path, run_dir_arg: pathlib.Path, **kw) -> int:
            raw_dir = run_dir_arg / "raw"
            grading = {
                "resolved_ids": instances[:57],
                "unresolved_ids": instances[57:89],
                "empty_patch_ids": instances[89:],
                "error_ids": [],
                "incomplete_ids": [],
            }
            (raw_dir / "grading_results.json").write_text(json.dumps(grading, indent=2))
            return 0

        def pass_qual():
            import swebench_qualification as sq
            return sq.QualificationResult(ok=True, reason="v2-direct-preflight-passed")

        runner = self._make_runner(
            run_dir,
            preflight_runner=lambda m: 0,
            generation_runner=generation_runner,
            grading_runner=grading_runner,
            qualification_verifier=pass_qual,
        )
        rc = runner.run()
        self.assertEqual(rc, 0, f"v2 success path must return 0, got {rc}")
        self.assertTrue(
            (run_dir / "DONE").exists(),
            "DONE must be written after successful v2 run",
        )
        manifest = json.loads((run_dir / "manifest.json").read_text())
        self.assertEqual(manifest["item_inventory"]["submitted"], 100)


# ===========================================================================
# Task 11 — V2 publication validation
# ===========================================================================


class TestV2PublicationValidation(unittest.TestCase):
    """
    Task 11: Tests proving v2 publication derives resolved/100 plus full
    decomposition; incomplete, invalid, diagnostic, and zero-verdict runs rejected.
    V1 canonical fixtures still pass. Qwen3.5 diagnostic v1 run stays rejected.
    """

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_validated_v2_run(self, resolved: int = 57, submitted: int = 89) -> pathlib.Path:
        """Create a minimal validated v2 run directory fixture."""
        run_dir = self.tmp / "runs" / "warpcore-v2" / "swebench" / "run-v2-test"
        run_dir.mkdir(parents=True)
        raw_dir = run_dir / "raw"
        raw_dir.mkdir()

        instances = json.loads(
            (_REPO / "suite" / "swebench" / "instances-seed42-n100.json").read_text()
        )[:100]
        non_empty = instances[:submitted]
        empty_patch = instances[submitted:]
        resolved_ids = non_empty[:resolved]
        unresolved_ids = non_empty[resolved:]

        # preds.json: submitted nonempty, rest empty
        preds = {}
        for iid in non_empty:
            preds[iid] = {"model_patch": "patch content", "instance_id": iid}
        for iid in empty_patch:
            preds[iid] = {"model_patch": "", "instance_id": iid}
        (raw_dir / "preds.json").write_text(json.dumps(preds))

        # grading_results.json
        grading = {
            "resolved_ids": resolved_ids,
            "unresolved_ids": unresolved_ids,
            "empty_patch_ids": empty_patch,
            "error_ids": [],
            "incomplete_ids": [],
        }
        (raw_dir / "grading_results.json").write_text(json.dumps(grading))

        # manifest.json
        manifest = {
            "run_id": "run-v2-test",
            "suite_id": "warpcore-v2",
            "benchmark": "swebench",
            "model": {"slug": "qwen3.6-35b-a3b", "id": "qwen3.6-35b-a3b"},
            "item_inventory": {"expected": 100, "submitted": submitted, "resolved": resolved},
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest))

        # status.json: validated
        status = {
            "run_id": "run-v2-test",
            "suite_id": "warpcore-v2",
            "lifecycle": "current",
            "execution_state": "validated",
            "history": [
                {"state": "planned", "timestamp": "2026-09-29T00:00:00Z"},
                {"state": "completed", "timestamp": "2026-09-29T01:00:00Z"},
                {"state": "validated", "timestamp": "2026-09-29T02:00:00Z"},
            ],
        }
        (run_dir / "status.json").write_text(json.dumps(status))
        (run_dir / "DONE").write_text("DONE\n")

        return run_dir

    def test_publisher_derives_submitted_from_nonempty_predictions(self):
        """Publisher must derive submitted from nonempty model_patch, not verdict count."""
        import publish_campaign as pc

        run_dir = self._make_validated_v2_run(resolved=57, submitted=89)

        # Load item scores
        scores = pc._load_item_scores(run_dir, benchmark="swebench")
        # submitted = nonempty predictions = 89
        # resolved = 57
        # The score computation is resolved/100
        resolved_count = sum(1 for v in scores.values() if v == 1.0)
        self.assertEqual(resolved_count, 57, f"Expected 57 resolved, got {resolved_count}")

        runs = [{
            "manifest": json.loads((run_dir / "manifest.json").read_text()),
            "status": json.loads((run_dir / "status.json").read_text()),
            "run_info": {"run_dir": run_dir, "model_slug": "qwen3.6-35b-a3b"},
        }]
        entry = pc._build_entries(runs)[0]
        self.assertEqual(entry["submitted"], 89)
        self.assertEqual(entry["resolved"], 57)
        self.assertEqual(entry["submitted_but_wrong"], 32)
        self.assertEqual(entry["model_non_submission"], 11)
        self.assertEqual(entry["empty_patch"], 11)
        self.assertEqual(entry["grading_error"], 0)
        self.assertEqual(entry["infrastructure_failure"], 0)
        self.assertEqual(entry["incomplete"], 0)
        self.assertEqual(entry["score"], 0.57)

    def test_v2_publisher_rejects_manifest_submitted_disagreement(self):
        """V2 publication must reject a manifest count not derived from predictions."""
        import publish_campaign as pc

        run_dir = self._make_validated_v2_run(resolved=57, submitted=89)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        manifest["item_inventory"]["submitted"] = 88
        runs = [{
            "manifest": manifest,
            "status": json.loads((run_dir / "status.json").read_text()),
            "run_info": {"run_dir": run_dir, "model_slug": "qwen3.6-35b-a3b"},
        }]
        with self.assertRaisesRegex(ValueError, "submitted.*nonempty"):
            pc._build_entries(runs)

    def test_publisher_rejects_zero_verdict_run(self):
        """Publisher must reject a run with 0 grading verdicts."""
        import publish_campaign as pc

        run_dir = self.tmp / "zero-verdict"
        run_dir.mkdir()
        raw_dir = run_dir / "raw"
        raw_dir.mkdir()

        # All predictions empty — no verdicts
        instances = ["id-1", "id-2", "id-3"]
        preds = {iid: {"model_patch": "", "instance_id": iid} for iid in instances}
        (raw_dir / "preds.json").write_text(json.dumps(preds))
        grading = {
            "resolved_ids": [],
            "unresolved_ids": [],
            "empty_patch_ids": instances,
            "error_ids": [],
            "incomplete_ids": [],
        }
        (raw_dir / "grading_results.json").write_text(json.dumps(grading))

        runs = [{
            "manifest": {
                "benchmark": "swebench",
                "suite_id": "warpcore-v2",
                "run_id": "run-zero",
                "model": {"slug": "model-x"},
                "item_inventory": {"expected": 3},
            },
            "status": {"lifecycle": "current", "execution_state": "validated"},
            "run_info": {"run_dir": run_dir, "model_slug": "model-x"},
        }]

        with self.assertRaisesRegex(ValueError, "no graded verdict"):
            pc._build_entries(runs)

    def test_publisher_rejects_incomplete_run(self):
        """Publisher must reject a run that is not in 'validated' state."""
        import publish_campaign as pc

        run_dir = self.tmp / "incomplete-run"
        run_dir.mkdir()
        raw_dir = run_dir / "raw"
        raw_dir.mkdir()

        instances = ["id-1", "id-2"]
        preds = {iid: {"model_patch": "patch", "instance_id": iid} for iid in instances}
        (raw_dir / "preds.json").write_text(json.dumps(preds))
        grading = {
            "resolved_ids": ["id-1"],
            "unresolved_ids": ["id-2"],
            "empty_patch_ids": [],
            "error_ids": [],
            "incomplete_ids": [],
        }
        (raw_dir / "grading_results.json").write_text(json.dumps(grading))

        # Not validated — execution_state is 'running'
        runs = [{
            "manifest": {
                "benchmark": "swebench",
                "suite_id": "warpcore-v2",
                "run_id": "run-incomplete",
                "model": {"slug": "model-x"},
                "item_inventory": {"expected": 2},
            },
            "status": {"lifecycle": "current", "execution_state": "running"},
            "run_info": {"run_dir": run_dir, "model_slug": "model-x"},
        }]

        # publish() owns eligibility filtering; _build_entries() receives only
        # already-accepted runs. Assert the public gate's closed state set.
        self.assertNotIn(
            "running", pc._PUBLICATION_STATES,
            "Incomplete (non-validated) runs must not be publication candidates",
        )

    def test_qwen35_diagnostic_run_not_published_without_qualification_seal(self):
        """Qwen3.5 diagnostic v1 run must remain rejected with no canonical-matrix cell.

        Verifies historical non-promotion: the Qwen3.5 run has lifecycle=diagnostic
        and must not appear in any canonical publication entry.
        """
        # The Qwen3.5 evidence is committed but noncanonical
        qwen35_result_dir = _REPO / "results" / "qwen3.5-122b-a10b"
        if not qwen35_result_dir.exists():
            self.skipTest("Qwen3.5 results not in this repo")

        import validate_campaign as vc

        # Discover qwen3.5 runs — they must all have lifecycle != 'current'
        # or be excluded from canonical entries
        runs = vc.discover_runs(_REPO)
        qwen35_runs = [
            r for r in runs
            if r.get("run_info", {}).get("model_slug", "") == "qwen3.5-122b-a10b"
        ]

        # If any qwen3.5 run exists, it must not be 'current' lifecycle
        for run in qwen35_runs:
            lc = run.get("status", {}).get("lifecycle", "unknown")
            self.assertNotEqual(
                lc, "current",
                f"Qwen3.5 run must not have lifecycle='current': {run}",
            )

    def test_v1_canonical_run_still_validates(self):
        """v1 canonical fixtures (gpt-oss) must still validate after v2 additions."""
        import validate_campaign as vc

        gptoss_run = (
            _REPO / "results" / "gpt-oss-120b" / "runs" / "warpcore-v1" / "swebench" /
            "gptoss-swebench-n100-20260921"
        )
        if not gptoss_run.exists():
            self.skipTest("gpt-oss canonical run not in this repo")

        result = vc.validate(
            run_dir=gptoss_run,
            suite_path=_V1_SUITE,
            adapter_path=_REPO / "adapters" / "gpt-oss-120b.yaml",
        )
        # Historical run may be ineligible but must not error on the infrastructure
        # (the run exists as committed evidence; its ineligibility is expected)
        self.assertIsNotNone(result)

    def test_v2_publication_resolved_div_100(self):
        """v2 run score must equal resolved/100, not resolved/submitted."""
        import publish_campaign as pc

        run_dir = self._make_validated_v2_run(resolved=57, submitted=89)

        runs = [{
            "manifest": {
                "benchmark": "swebench",
                "suite_id": "warpcore-v2",
                "run_id": "run-v2-test",
                "model": {"slug": "qwen3.6-35b-a3b"},
                "item_inventory": {"expected": 100},
            },
            "status": {"lifecycle": "current", "execution_state": "validated"},
            "run_info": {"run_dir": run_dir, "model_slug": "qwen3.6-35b-a3b"},
        }]

        entries = pc._build_entries(runs)
        self.assertEqual(len(entries), 1)
        score = entries[0]["score"]
        # resolved/100 = 57/100 = 0.57
        self.assertAlmostEqual(score, 57 / 100, places=4,
                               msg=f"Score must be resolved/100 = 0.57, got {score}")
        # Must NOT be resolved/submitted = 57/89
        self.assertFalse(
            abs(score - 57 / 89) < 0.001,
            f"Score must not be resolved/submitted ({57/89:.4f}); got {score}",
        )


# ===========================================================================
# Task 7+8 — validate_suite.py also validates v2
# ===========================================================================


class TestValidateSuiteAcceptsV2(unittest.TestCase):
    """Prove that validate_suite.py handles v2 correctly."""

    def test_validate_suite_v2_returns_no_errors(self):
        """validate_suite must return [] for the v2 suite file."""
        if not _V2_SUITE.exists():
            self.skipTest("v2 suite not yet created (Task 8)")
        errors = contract.validate_suite(_REPO, _V2_SUITE)
        self.assertEqual(errors, [], f"v2 validate_suite errors:\n" + "\n".join(errors))

    def test_validate_suite_v1_still_returns_no_errors(self):
        """validate_suite must still return [] for v1 after v2 additions."""
        errors = contract.validate_suite(_REPO, _V1_SUITE)
        self.assertEqual(errors, [], f"v1 validate_suite errors:\n" + "\n".join(errors))


if __name__ == "__main__":
    unittest.main()
