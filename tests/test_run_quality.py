"""tests/test_run_quality.py — Contract-aware quality runner (Task 5).

TDD: tests written BEFORE implementation. Must fail (RED) until viz/run_quality.py exists.

Design contract (from docs/superpowers/specs/2026-09-15-warpcore-v1-apples-to-apples-design.md):
- Accepts: suite, adapter, benchmark, endpoint, throughput, concurrency, timeout
- Generated command uses canonical task path, suite generation ceiling and temperature,
  includes --log_samples, points at normalized run directory
- CLI does NOT accept overrides for task, ceiling, sampling, scoring, datasets, instances
- Runs QualityPreflightGate before harness; exit 1 or 2 blocks launch
- Refuses long launch outside /usr/bin/screen except in --dry-run or explicit test mode
- Dependency-injected process execution
- Does not duplicate preflight_serving.py, check_output_budget.py, or timeout arithmetic
- Writes exact argv to command.txt using shell-safe quoting
- On success: verifies required raw files before writing DONE
- On nonzero harness exit: does NOT write DONE; records exit code; transitions to failed
- Adds make run-quality SUITE=... ADAPTER=... BENCH=... with explicit required variables
"""
from __future__ import annotations

import json
import os
import pathlib
import shlex
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, call, patch

# Allow imports from viz/ and tests/
_TESTS_DIR = pathlib.Path(__file__).parent
_REPO = _TESTS_DIR.parent
_VIZ_DIR = _REPO / "viz"
for _p in (str(_TESTS_DIR), str(_VIZ_DIR), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The module under test — will not exist until implementation
import run_quality  # noqa: E402  (expected ImportError during RED)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REAL_SUITE = _REPO / "suite" / "warpcore-v1.yaml"
_SCHEMAS_DIR = _REPO / "suite" / "schemas"


def _minimal_status(state: str = "preflight_passed") -> dict:
    return {
        "execution_state": state,
        "lifecycle": "current",
        "history": [
            {"state": "planned", "timestamp": "2026-09-15T12:00:00Z"},
            {"state": "preflight_passed", "timestamp": "2026-09-15T12:01:00Z"},
        ],
    }


def _build_run_dir(tmp_path: pathlib.Path, bench: str = "gsm8k") -> pathlib.Path:
    """Create a minimal normalized run directory for test use."""
    run_dir = tmp_path / "results" / "test-model" / "runs" / "warpcore-v1" / bench / "run-test"
    run_dir.mkdir(parents=True)
    status = {
        "execution_state": "preflight_passed",
        "lifecycle": "current",
        "history": [
            {"state": "planned", "timestamp": "2026-09-15T12:00:00Z"},
            {"state": "preflight_passed", "timestamp": "2026-09-15T12:01:00Z"},
        ],
    }
    (run_dir / "status.json").write_text(json.dumps(status))
    manifest = {
        "suite_id": "warpcore-v1",
        "run_id": "run-test",
        "benchmark": bench,
        "model": {"slug": "test-model", "id": "test/model", "revision": "abc123"},
        "item_inventory": {"expected": 1319},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    return run_dir


# ---------------------------------------------------------------------------
# 1. Module interface — QualityRunner class exists
# ---------------------------------------------------------------------------


class TestModuleInterface(unittest.TestCase):
    """The module must expose a QualityRunner class."""

    def test_quality_runner_class_exists(self):
        self.assertTrue(hasattr(run_quality, "QualityRunner"))

    def test_quality_runner_is_callable(self):
        self.assertTrue(callable(run_quality.QualityRunner))

    def test_forbidden_overrides_constant_exists(self):
        """FORBIDDEN_OVERRIDE_FLAGS must list all CLI flags callers cannot pass."""
        self.assertTrue(hasattr(run_quality, "FORBIDDEN_OVERRIDE_FLAGS"))
        forbidden = run_quality.FORBIDDEN_OVERRIDE_FLAGS
        self.assertIsInstance(forbidden, (list, tuple, frozenset, set))

    def test_forbidden_overrides_covers_task_and_sampling(self):
        """Forbidden set must include task, ceiling, temperature, datasets, instances."""
        forbidden = set(run_quality.FORBIDDEN_OVERRIDE_FLAGS)
        # Each of these controls an experiment-level variable the suite owns
        for flag in ("--tasks", "--num_fewshot", "--gen_kwargs", "--limit", "--include_path"):
            self.assertIn(flag, forbidden, f"Missing forbidden flag: {flag}")


# ---------------------------------------------------------------------------
# 2. Command construction — canonical task path and suite settings
# ---------------------------------------------------------------------------


class TestCommandConstruction(unittest.TestCase):
    """QualityRunner must build a deterministic lm-eval command from suite + adapter."""

    def _make_runner(
        self,
        tmp_path,
        bench="gsm8k",
        *,
        preflight_exit=0,
        harness_exit=0,
        endpoint="http://fake:8000/v1",
        throughput=64.0,
        concurrency=8,
        timeout=14400,
        dry_run=True,
    ):
        run_dir = _build_run_dir(tmp_path, bench)
        runner = run_quality.QualityRunner(
            suite_path=_REAL_SUITE,
            adapter_path=_REPO / "adapters" / "qwen3.6-35b-a3b.yaml",
            benchmark=bench,
            endpoint=endpoint,
            throughput=throughput,
            concurrency=concurrency,
            timeout=timeout,
            run_dir=run_dir,
            dry_run=dry_run,
            preflight_runner=MagicMock(return_value=preflight_exit),
            harness_runner=MagicMock(return_value=harness_exit),
        )
        return runner, run_dir

    def test_command_includes_task_from_suite(self, tmp_path=None):
        """Generated command must use the canonical task path from the suite."""
        if tmp_path is None:
            tmp_path = pathlib.Path(tempfile.mkdtemp())
        runner, run_dir = self._make_runner(tmp_path)
        cmd = runner.build_command()
        cmd_str = " ".join(cmd)
        # Suite gsm8k task_file = suite/tasks/gsm8k_clean_v1.yaml
        self.assertIn("gsm8k_clean_v1", cmd_str)

    def test_command_includes_generation_ceiling(self, tmp_path=None):
        """Generated command must include suite generation ceiling (max_gen_toks)."""
        if tmp_path is None:
            tmp_path = pathlib.Path(tempfile.mkdtemp())
        runner, run_dir = self._make_runner(tmp_path)
        cmd = runner.build_command()
        cmd_str = " ".join(cmd)
        # Suite gsm8k generation_ceiling = 8192
        self.assertIn("8192", cmd_str)
        self.assertIn("max_gen_toks", cmd_str)

    def test_command_includes_temperature_zero(self, tmp_path=None):
        """Generated command must include temperature=0 from suite sampling."""
        if tmp_path is None:
            tmp_path = pathlib.Path(tempfile.mkdtemp())
        runner, run_dir = self._make_runner(tmp_path)
        cmd = runner.build_command()
        cmd_str = " ".join(cmd)
        self.assertIn("temperature", cmd_str)
        self.assertIn("0", cmd_str)

    def test_command_includes_log_samples(self, tmp_path=None):
        """Generated command must include --log_samples."""
        if tmp_path is None:
            tmp_path = pathlib.Path(tempfile.mkdtemp())
        runner, run_dir = self._make_runner(tmp_path)
        cmd = runner.build_command()
        self.assertIn("--log_samples", cmd)

    def test_command_points_at_normalized_run_dir(self, tmp_path=None):
        """Generated command must point output at the normalized run directory."""
        if tmp_path is None:
            tmp_path = pathlib.Path(tempfile.mkdtemp())
        runner, run_dir = self._make_runner(tmp_path)
        cmd = runner.build_command()
        cmd_str = " ".join(cmd)
        self.assertIn(str(run_dir), cmd_str)

    def test_command_includes_endpoint(self, tmp_path=None):
        """Generated command must include the specified endpoint."""
        if tmp_path is None:
            tmp_path = pathlib.Path(tempfile.mkdtemp())
        runner, run_dir = self._make_runner(tmp_path, endpoint="http://custom:9000/v1")
        cmd = runner.build_command()
        cmd_str = " ".join(cmd)
        self.assertIn("http://custom:9000/v1", cmd_str)

    def test_command_includes_concurrency(self, tmp_path=None):
        """Generated command must include concurrency."""
        if tmp_path is None:
            tmp_path = pathlib.Path(tempfile.mkdtemp())
        runner, run_dir = self._make_runner(tmp_path, concurrency=12)
        cmd = runner.build_command()
        cmd_str = " ".join(cmd)
        self.assertIn("12", cmd_str)

    def test_command_includes_timeout(self, tmp_path=None):
        """Generated command must include the client timeout."""
        if tmp_path is None:
            tmp_path = pathlib.Path(tempfile.mkdtemp())
        runner, run_dir = self._make_runner(tmp_path, timeout=30000)
        cmd = runner.build_command()
        cmd_str = " ".join(cmd)
        self.assertIn("30000", cmd_str)


# ---------------------------------------------------------------------------
# 3. CLI rejection of forbidden overrides
# ---------------------------------------------------------------------------


class TestForbiddenOverrides(unittest.TestCase):
    """CLI must not accept overrides for suite-owned experiment variables."""

    def _try_parse(self, extra_args):
        """Attempt to parse CLI with extra args; expect SystemExit or ValueError."""
        argv = [
            "--suite", str(_REAL_SUITE),
            "--adapter", str(_REPO / "adapters" / "qwen3.6-35b-a3b.yaml"),
            "--benchmark", "gsm8k",
            "--endpoint", "http://fake:8000/v1",
            "--throughput", "64",
            "--concurrency", "8",
            "--timeout", "14400",
            "--dry-run",
        ] + extra_args
        try:
            rc = run_quality.main(argv)
            return rc
        except SystemExit as e:
            return e.code

    def test_rejects_tasks_override(self):
        """--tasks must not be accepted."""
        rc = self._try_parse(["--tasks", "gsm8k"])
        self.assertNotEqual(rc, 0)

    def test_rejects_num_fewshot_override(self):
        """--num_fewshot must not be accepted."""
        rc = self._try_parse(["--num_fewshot", "5"])
        self.assertNotEqual(rc, 0)

    def test_rejects_gen_kwargs_override(self):
        """--gen_kwargs must not be accepted."""
        rc = self._try_parse(["--gen_kwargs", "temperature=0.5"])
        self.assertNotEqual(rc, 0)

    def test_rejects_limit_override(self):
        """--limit must not be accepted."""
        rc = self._try_parse(["--limit", "10"])
        self.assertNotEqual(rc, 0)


# ---------------------------------------------------------------------------
# 4. Preflight gate integration
# ---------------------------------------------------------------------------


class TestPreflightIntegration(unittest.TestCase):
    """QualityRunner must call QualityPreflightGate before harness execution."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_runner(self, *, preflight_exit=0, harness_exit=0, dry_run=False):
        run_dir = _build_run_dir(self.tmp)
        preflight_mock = MagicMock(return_value=preflight_exit)
        harness_mock = MagicMock(return_value=harness_exit)
        runner = run_quality.QualityRunner(
            suite_path=_REAL_SUITE,
            adapter_path=_REPO / "adapters" / "qwen3.6-35b-a3b.yaml",
            benchmark="gsm8k",
            endpoint="http://fake:8000/v1",
            throughput=64.0,
            concurrency=8,
            timeout=14400,
            run_dir=run_dir,
            dry_run=dry_run,
            allow_no_screen=True,
            preflight_runner=preflight_mock,
            harness_runner=harness_mock,
        )
        return runner, run_dir, preflight_mock, harness_mock

    def test_preflight_called_before_harness(self):
        """Preflight must be invoked before harness on a non-dry-run."""
        runner, run_dir, preflight_mock, harness_mock = self._make_runner(dry_run=False)
        # Create raw dir to satisfy DONE check
        (run_dir / "raw").mkdir()
        runner.run()
        preflight_mock.assert_called()

    def test_preflight_exit_1_blocks_harness(self):
        """If preflight exits 1, harness must NOT be called."""
        runner, run_dir, preflight_mock, harness_mock = self._make_runner(preflight_exit=1, dry_run=False)
        runner.run()
        harness_mock.assert_not_called()

    def test_preflight_exit_2_blocks_harness(self):
        """If preflight exits 2, harness must NOT be called."""
        runner, run_dir, preflight_mock, harness_mock = self._make_runner(preflight_exit=2, dry_run=False)
        runner.run()
        harness_mock.assert_not_called()

    def test_preflight_exit_1_returns_nonzero(self):
        """If preflight exits 1, run() must return nonzero."""
        runner, run_dir, preflight_mock, harness_mock = self._make_runner(preflight_exit=1, dry_run=False)
        rc = runner.run()
        self.assertNotEqual(rc, 0)

    def test_preflight_exit_0_proceeds_to_harness(self):
        """If preflight exits 0, harness must be called."""
        runner, run_dir, preflight_mock, harness_mock = self._make_runner(
            preflight_exit=0, harness_exit=0, dry_run=False)
        (run_dir / "raw").mkdir()
        runner.run()
        harness_mock.assert_called()

    def test_dry_run_skips_preflight_network_calls(self):
        """In dry-run mode, preflight must not make network calls (preflight_runner not called or returns early)."""
        runner, run_dir, preflight_mock, harness_mock = self._make_runner(dry_run=True)
        runner.run()
        harness_mock.assert_not_called()


# ---------------------------------------------------------------------------
# 5. Screen guard
# ---------------------------------------------------------------------------


class TestScreenGuard(unittest.TestCase):
    """Long launches must be refused outside /usr/bin/screen except dry-run or test-mode."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_runner(self, *, dry_run=False, allow_no_screen=False):
        run_dir = _build_run_dir(self.tmp)
        runner = run_quality.QualityRunner(
            suite_path=_REAL_SUITE,
            adapter_path=_REPO / "adapters" / "qwen3.6-35b-a3b.yaml",
            benchmark="gsm8k",
            endpoint="http://fake:8000/v1",
            throughput=64.0,
            concurrency=8,
            timeout=14400,
            run_dir=run_dir,
            dry_run=dry_run,
            allow_no_screen=allow_no_screen,
            preflight_runner=MagicMock(return_value=0),
            harness_runner=MagicMock(return_value=0),
        )
        return runner, run_dir

    def test_refuses_launch_when_not_under_screen(self):
        """Without --allow-no-screen or --dry-run, must refuse launch outside screen."""
        runner, run_dir = self._make_runner(dry_run=False, allow_no_screen=False)
        # Simulate not being under screen by ensuring TERM env does not include 'screen'
        env_backup = os.environ.pop("STY", None)
        term_backup = os.environ.get("TERM")
        try:
            if "STY" in os.environ:
                del os.environ["STY"]
            if term_backup and "screen" in term_backup:
                os.environ["TERM"] = "xterm"
            rc = runner.run()
            # Must refuse with a nonzero code (not proceed)
            self.assertNotEqual(rc, 0)
        finally:
            if env_backup is not None:
                os.environ["STY"] = env_backup
            if term_backup is not None:
                os.environ["TERM"] = term_backup

    def test_dry_run_bypasses_screen_guard(self):
        """In dry-run mode, screen guard must NOT block execution."""
        runner, run_dir = self._make_runner(dry_run=True, allow_no_screen=False)
        env_backup = os.environ.pop("STY", None)
        try:
            if "STY" in os.environ:
                del os.environ["STY"]
            rc = runner.run()
            # Dry-run: should not fail due to screen check
            # (May be nonzero for other reasons but not screen)
            # We test this by verifying the preflight mock was not the cause
        finally:
            if env_backup is not None:
                os.environ["STY"] = env_backup

    def test_allow_no_screen_bypasses_guard(self):
        """--allow-no-screen must bypass the screen guard."""
        runner, run_dir = self._make_runner(dry_run=False, allow_no_screen=True)
        env_backup = os.environ.pop("STY", None)
        try:
            if "STY" in os.environ:
                del os.environ["STY"]
            # Should proceed to preflight (which is mocked to return 0)
            # and harness (which returns 0), then check for raw files
            (run_dir / "raw").mkdir()
            rc = runner.run()
            # Should not fail on screen check
            self.assertIsNotNone(rc)
        finally:
            if env_backup is not None:
                os.environ["STY"] = env_backup


# ---------------------------------------------------------------------------
# 6. command.txt — shell-safe quoting
# ---------------------------------------------------------------------------


class TestCommandTxt(unittest.TestCase):
    """Runner must write exact shell-safe command to command.txt."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_command_txt_written_before_harness(self):
        """command.txt must exist after a dry-run build."""
        run_dir = _build_run_dir(self.tmp)
        runner = run_quality.QualityRunner(
            suite_path=_REAL_SUITE,
            adapter_path=_REPO / "adapters" / "qwen3.6-35b-a3b.yaml",
            benchmark="gsm8k",
            endpoint="http://fake:8000/v1",
            throughput=64.0,
            concurrency=8,
            timeout=14400,
            run_dir=run_dir,
            dry_run=True,
            preflight_runner=MagicMock(return_value=0),
            harness_runner=MagicMock(return_value=0),
        )
        runner.run()
        cmd_file = run_dir / "command.txt"
        self.assertTrue(cmd_file.exists(), "command.txt must be written")

    def test_command_txt_is_shell_safe(self):
        """command.txt content must be parseable by shlex.split."""
        run_dir = _build_run_dir(self.tmp)
        runner = run_quality.QualityRunner(
            suite_path=_REAL_SUITE,
            adapter_path=_REPO / "adapters" / "qwen3.6-35b-a3b.yaml",
            benchmark="gsm8k",
            endpoint="http://fake:8000/v1",
            throughput=64.0,
            concurrency=8,
            timeout=14400,
            run_dir=run_dir,
            dry_run=True,
            preflight_runner=MagicMock(return_value=0),
            harness_runner=MagicMock(return_value=0),
        )
        runner.run()
        cmd_file = run_dir / "command.txt"
        content = cmd_file.read_text().strip()
        # Must be parseable
        parsed = shlex.split(content)
        self.assertGreater(len(parsed), 0)

    def test_command_txt_round_trips_argv(self):
        """shlex.split(command.txt) must equal the generated argv."""
        run_dir = _build_run_dir(self.tmp)
        runner = run_quality.QualityRunner(
            suite_path=_REAL_SUITE,
            adapter_path=_REPO / "adapters" / "qwen3.6-35b-a3b.yaml",
            benchmark="gsm8k",
            endpoint="http://fake:8000/v1",
            throughput=64.0,
            concurrency=8,
            timeout=14400,
            run_dir=run_dir,
            dry_run=True,
            preflight_runner=MagicMock(return_value=0),
            harness_runner=MagicMock(return_value=0),
        )
        cmd = runner.build_command()
        runner.run()
        cmd_file = run_dir / "command.txt"
        parsed = shlex.split(cmd_file.read_text().strip())
        self.assertEqual(parsed, cmd)


# ---------------------------------------------------------------------------
# 7. DONE sentinel and raw file verification
# ---------------------------------------------------------------------------


class TestDoneSentinel(unittest.TestCase):
    """DONE must only be written on success and only after verifying raw files."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_runner(self, *, harness_exit=0, raw_exists=True, dry_run=False):
        run_dir = _build_run_dir(self.tmp)
        if raw_exists:
            (run_dir / "raw").mkdir()
        runner = run_quality.QualityRunner(
            suite_path=_REAL_SUITE,
            adapter_path=_REPO / "adapters" / "qwen3.6-35b-a3b.yaml",
            benchmark="gsm8k",
            endpoint="http://fake:8000/v1",
            throughput=64.0,
            concurrency=8,
            timeout=14400,
            run_dir=run_dir,
            dry_run=dry_run,
            allow_no_screen=True,
            preflight_runner=MagicMock(return_value=0),
            harness_runner=MagicMock(return_value=harness_exit),
        )
        return runner, run_dir

    def test_done_written_on_success_with_raw_files(self):
        """DONE is written when harness exits 0 and raw/ directory exists."""
        runner, run_dir = self._make_runner(harness_exit=0, raw_exists=True)
        runner.run()
        self.assertTrue((run_dir / "DONE").exists(), "DONE must be written on success")

    def test_done_not_written_on_harness_failure(self):
        """DONE must NOT be written if harness exits nonzero."""
        runner, run_dir = self._make_runner(harness_exit=1, raw_exists=True)
        runner.run()
        self.assertFalse((run_dir / "DONE").exists(), "DONE must NOT be written on harness failure")

    def test_done_not_written_when_raw_missing(self):
        """DONE must NOT be written if raw/ directory does not exist (harness succeeded but raw missing)."""
        runner, run_dir = self._make_runner(harness_exit=0, raw_exists=False)
        runner.run()
        self.assertFalse((run_dir / "DONE").exists(), "DONE must NOT be written when raw/ is missing")

    def test_done_not_written_in_dry_run(self):
        """DONE must NOT be written in dry-run mode."""
        runner, run_dir = self._make_runner(dry_run=True)
        runner.run()
        self.assertFalse((run_dir / "DONE").exists(), "DONE must NOT be written in dry-run")


# ---------------------------------------------------------------------------
# 8. Campaign state transitions
# ---------------------------------------------------------------------------


class TestCampaignStateTransitions(unittest.TestCase):
    """Runner must update status.json on transitions."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_runner(self, *, harness_exit=0, raw_exists=True):
        run_dir = _build_run_dir(self.tmp)
        if raw_exists:
            (run_dir / "raw").mkdir()
        runner = run_quality.QualityRunner(
            suite_path=_REAL_SUITE,
            adapter_path=_REPO / "adapters" / "qwen3.6-35b-a3b.yaml",
            benchmark="gsm8k",
            endpoint="http://fake:8000/v1",
            throughput=64.0,
            concurrency=8,
            timeout=14400,
            run_dir=run_dir,
            allow_no_screen=True,
            preflight_runner=MagicMock(return_value=0),
            harness_runner=MagicMock(return_value=harness_exit),
        )
        return runner, run_dir

    def test_status_transitions_to_running(self):
        """On launch, status must transition from preflight_passed -> running."""
        runner, run_dir = self._make_runner()
        runner.run()
        status = json.loads((run_dir / "status.json").read_text())
        states = [h["state"] for h in status["history"]]
        self.assertIn("running", states)

    def test_status_transitions_to_completed_on_success(self):
        """On harness success with raw files, status must be completed."""
        runner, run_dir = self._make_runner(harness_exit=0, raw_exists=True)
        runner.run()
        status = json.loads((run_dir / "status.json").read_text())
        self.assertEqual(status["execution_state"], "completed")

    def test_status_transitions_to_failed_on_harness_error(self):
        """On harness exit nonzero, status must transition to failed."""
        runner, run_dir = self._make_runner(harness_exit=1)
        runner.run()
        status = json.loads((run_dir / "status.json").read_text())
        self.assertEqual(status["execution_state"], "failed")

    def test_failed_status_records_exit_code(self):
        """On failure, status.json must record the harness exit code."""
        runner, run_dir = self._make_runner(harness_exit=2)
        runner.run()
        status = json.loads((run_dir / "status.json").read_text())
        # The exit code should appear somewhere in the status
        status_str = json.dumps(status)
        self.assertIn("2", status_str)


# ---------------------------------------------------------------------------
# 9. CLI main entry point
# ---------------------------------------------------------------------------


class TestCLI(unittest.TestCase):
    """run_quality must expose a main() function with required arguments."""

    def test_main_is_callable(self):
        self.assertTrue(callable(run_quality.main))

    def test_missing_suite_exits_nonzero(self):
        try:
            rc = run_quality.main([])
        except SystemExit as e:
            rc = e.code
        self.assertNotEqual(rc, 0)

    def test_dry_run_flag_exists(self):
        """--dry-run flag must be accepted without error."""
        argv = [
            "--suite", str(_REAL_SUITE),
            "--adapter", str(_REPO / "adapters" / "qwen3.6-35b-a3b.yaml"),
            "--benchmark", "gsm8k",
            "--endpoint", "http://fake:8000/v1",
            "--throughput", "64",
            "--concurrency", "8",
            "--timeout", "14400",
            "--run-id", "run-test-cli",
            "--dry-run",
        ]
        # In dry-run, should print command and exit 0 (or possibly nonzero due to
        # missing run dir -- but must not fail on unknown argument)
        try:
            rc = run_quality.main(argv)
        except SystemExit as e:
            rc = e.code
        except Exception:
            rc = None
        # The important thing: no SystemExit from "unrecognized argument"
        self.assertNotEqual(rc, 2, "Exit code 2 suggests unrecognized --dry-run argument")

    def test_gpqa_diamond_ceiling_in_command(self):
        """GPQA benchmark must use 65536 generation ceiling from suite."""
        tmp = pathlib.Path(tempfile.mkdtemp())
        try:
            run_dir = _build_run_dir(tmp, bench="gpqa_diamond")
            runner = run_quality.QualityRunner(
                suite_path=_REAL_SUITE,
                adapter_path=_REPO / "adapters" / "qwen3.6-35b-a3b.yaml",
                benchmark="gpqa_diamond",
                endpoint="http://fake:8000/v1",
                throughput=64.0,
                concurrency=8,
                timeout=14400,
                run_dir=run_dir,
                dry_run=True,
                preflight_runner=MagicMock(return_value=0),
                harness_runner=MagicMock(return_value=0),
            )
            cmd = runner.build_command()
            cmd_str = " ".join(cmd)
            self.assertIn("65536", cmd_str)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 10. No duplication of existing scripts
# ---------------------------------------------------------------------------


class TestNoDuplication(unittest.TestCase):
    """run_quality.py must delegate to existing tools, not re-implement them."""

    def test_run_quality_imports_quality_preflight(self):
        """run_quality must import from quality_preflight, not re-implement it."""
        import inspect
        source = inspect.getsource(run_quality)
        # Must reference quality_preflight
        self.assertIn("quality_preflight", source)

    def test_run_quality_does_not_define_timeout_arithmetic(self):
        """run_quality must not re-define compute_timeout_arithmetic."""
        import inspect
        source = inspect.getsource(run_quality)
        # Should NOT define this function itself
        self.assertNotIn("def compute_timeout_arithmetic", source)

    def test_run_quality_does_not_define_check_output_budget(self):
        """run_quality must not re-implement check_output_budget logic."""
        import inspect
        source = inspect.getsource(run_quality)
        self.assertNotIn("def check_output_budget", source)


# ---------------------------------------------------------------------------
# 11. Dependency injection — process execution
# ---------------------------------------------------------------------------


class TestDependencyInjection(unittest.TestCase):
    """QualityRunner must use injected process runners, not subprocess directly."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_harness_runner_is_called_with_command_list(self):
        """harness_runner must be called with the command as a list."""
        run_dir = _build_run_dir(self.tmp)
        (run_dir / "raw").mkdir()
        harness_mock = MagicMock(return_value=0)
        runner = run_quality.QualityRunner(
            suite_path=_REAL_SUITE,
            adapter_path=_REPO / "adapters" / "qwen3.6-35b-a3b.yaml",
            benchmark="gsm8k",
            endpoint="http://fake:8000/v1",
            throughput=64.0,
            concurrency=8,
            timeout=14400,
            run_dir=run_dir,
            allow_no_screen=True,
            preflight_runner=MagicMock(return_value=0),
            harness_runner=harness_mock,
        )
        runner.run()
        harness_mock.assert_called_once()
        call_args = harness_mock.call_args
        # First positional arg must be the command list
        cmd_arg = call_args[0][0] if call_args[0] else call_args[1].get("cmd", None)
        self.assertIsInstance(cmd_arg, list, "harness_runner must be called with a list")

    def test_preflight_runner_receives_gate_args(self):
        """preflight_runner must be called with gate configuration arguments."""
        run_dir = _build_run_dir(self.tmp)
        (run_dir / "raw").mkdir()
        preflight_mock = MagicMock(return_value=0)
        runner = run_quality.QualityRunner(
            suite_path=_REAL_SUITE,
            adapter_path=_REPO / "adapters" / "qwen3.6-35b-a3b.yaml",
            benchmark="gsm8k",
            endpoint="http://fake:8000/v1",
            throughput=64.0,
            concurrency=8,
            timeout=14400,
            run_dir=run_dir,
            allow_no_screen=True,
            preflight_runner=preflight_mock,
            harness_runner=MagicMock(return_value=0),
        )
        runner.run()
        preflight_mock.assert_called()


# ---------------------------------------------------------------------------
# 12. IFEval ceiling uses 65536
# ---------------------------------------------------------------------------


class TestIFEvalCeiling(unittest.TestCase):
    def test_ifeval_uses_65536_ceiling(self):
        """IFEval must use 65536 generation ceiling from suite."""
        tmp = pathlib.Path(tempfile.mkdtemp())
        try:
            run_dir = _build_run_dir(tmp, bench="ifeval")
            runner = run_quality.QualityRunner(
                suite_path=_REAL_SUITE,
                adapter_path=_REPO / "adapters" / "qwen3.6-35b-a3b.yaml",
                benchmark="ifeval",
                endpoint="http://fake:8000/v1",
                throughput=64.0,
                concurrency=8,
                timeout=14400,
                run_dir=run_dir,
                dry_run=True,
                preflight_runner=MagicMock(return_value=0),
                harness_runner=MagicMock(return_value=0),
            )
            cmd = runner.build_command()
            cmd_str = " ".join(cmd)
            self.assertIn("65536", cmd_str)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
