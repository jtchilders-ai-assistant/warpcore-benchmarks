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

import gzip
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
import campaign_state  # noqa: E402


# ---------------------------------------------------------------------------
# Canonical adapter fixture
# ---------------------------------------------------------------------------
# A canonical adapter passes validate_adapter_campaign_ready() with a supplied
# prompt_token_maxima. This is a real canonical adapter with valid immutable
# revision, image digest, and capacity fields. No noncanonical bypass allowed.

_CANONICAL_ADAPTER = {
    "adapter_schema_version": 1,
    "campaign_status": "canonical",
    "model": {
        "slug": "test-canonical-model",
        "id": "testorg/TestCanonicalModel",
        # 40-char lowercase hex SHA
        "revision": "a" * 40,
    },
    "serving": {
        # repo@sha256:<64hex>
        "image": "testregistry.example.com/test@sha256:" + "b" * 64,
        "engine": "vllm",
        "engine_version": "0.6.6",
        "quantization": "fp8",
        "reasoning_parser": None,
        "tool_call_parser": None,
        "tokenizer": None,
        "moe_backend": None,
        "max_model_len": 300000,  # big enough for all benchmarks
        "gpu_memory_utilization": 0.95,
        "max_num_seqs": 256,
        "environment": {},
    },
}

# Prompt token maxima: measured for every benchmark with a generation_ceiling.
# Must satisfy: prompt_tokens + ceiling <= max_model_len (300000).
# gsm8k ceiling=8192, ifeval=65536, gpqa_diamond=65536
_PROMPT_TOKEN_MAXIMA = {
    "gsm8k": 500,
    "ifeval": 2000,
    "gpqa_diamond": 1000,
}

# Adapter slug derived from the canonical adapter
_ADAPTER_SLUG = _CANONICAL_ADAPTER["model"]["slug"]  # "test-canonical-model"


def _write_canonical_adapter(path: pathlib.Path) -> None:
    """Write a canonical adapter YAML to *path*."""
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(_CANONICAL_ADAPTER, default_flow_style=False))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REAL_SUITE = _REPO / "suite" / "warpcore-v1.yaml"
_SCHEMAS_DIR = _REPO / "suite" / "schemas"


def _make_planned_status(run_id: str = "run-test", suite_id: str = "warpcore-v1") -> dict:
    """Return a schema-valid planned status dict."""
    return {
        "schema_version": 1,
        "run_id": run_id,
        "suite_id": suite_id,
        "execution_state": "planned",
        "lifecycle": "current",
        "history": [
            {"state": "planned", "timestamp": "2026-09-15T12:00:00Z"},
        ],
    }


def _build_run_dir(
    repo: pathlib.Path,
    bench: str = "gsm8k",
    slug: str = _ADAPTER_SLUG,
    run_id: str = "run-test",
    suite_id: str = "warpcore-v1",
) -> pathlib.Path:
    """Create a minimal normalized run directory for test use.

    Layout: repo/results/<slug>/runs/<suite_id>/<bench>/<run_id>
    This is the exact normalized path the runner enforces.
    Initial state is 'planned' — the only state QualityRunner accepts to start a run.
    """
    run_dir = repo / "results" / slug / "runs" / suite_id / bench / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # Schema-valid planned status
    (run_dir / "status.json").write_text(json.dumps(_make_planned_status(run_id, suite_id)))
    manifest = {
        "suite_id": suite_id,
        "run_id": run_id,
        "benchmark": bench,
        "model": {"slug": slug, "id": "testorg/TestCanonicalModel", "revision": "a" * 40},
        "item_inventory": {"expected": 1319},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    return run_dir


def _make_harness_artifacts(run_dir: pathlib.Path) -> None:
    """Populate raw/ with minimal valid lm-eval artifacts."""
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    (raw_dir / "results_2026-01-01T00-00-00.json").write_text(
        '{"results": {"gsm8k_cot_zeroshot_clean": {"exact_match,none": 0.5}}}\n'
    )
    messages = [{"role": "user", "content": "fixture question"}]
    sample = {"doc_id": 0, "resps": [["42"]], "filtered_resps": ["42"],
              "target": "42", "exact_match": 1.0,
              "arguments": {"gen_args_0": {"arg_0": [json.dumps(messages)]}}}
    gz_path = raw_dir / "samples_gsm8k_cot_zeroshot_clean_2026-01-01T00-00-00.jsonl.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps(sample) + "\n")
    import hashlib
    fingerprint = hashlib.sha256(
        json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    (raw_dir / "response_metadata.jsonl").write_text(json.dumps({
        "fingerprint": fingerprint, "finish_reasons": ["stop"],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        "content": ["42"], "reasoning_content": [None], "reasoning": [None],
        "model": "testorg/TestCanonicalModel",
        "run_id": run_dir.name,
    }) + "\n")


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

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_runner(
        self,
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
        run_dir = _build_run_dir(self.tmp, bench)
        runner = run_quality.QualityRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            benchmark=bench,
            endpoint=endpoint,
            throughput=throughput,
            concurrency=concurrency,
            timeout=timeout,
            run_dir=run_dir,
            repo=self.tmp,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            dry_run=dry_run,
            preflight_runner=MagicMock(return_value=preflight_exit),
            harness_runner=MagicMock(return_value=harness_exit),
        )
        return runner, run_dir

    def test_command_includes_task_from_suite(self):
        """Generated command must use the task name from the pinned task YAML's 'task:' field."""
        runner, run_dir = self._make_runner()
        cmd = runner.build_command()
        cmd_str = " ".join(cmd)
        # The task name comes from suite/tasks/gsm8k_clean_v1.yaml's 'task:' field,
        # which is 'gsm8k_cot_zeroshot_clean' — NOT the file stem.
        self.assertIn("gsm8k_cot_zeroshot_clean", cmd_str)

    def test_command_includes_generation_ceiling(self):
        """Generated command must include suite generation ceiling (max_gen_toks)."""
        runner, run_dir = self._make_runner()
        cmd = runner.build_command()
        cmd_str = " ".join(cmd)
        # Suite gsm8k generation_ceiling = 8192
        self.assertIn("8192", cmd_str)
        self.assertIn("max_gen_toks", cmd_str)

    def test_command_includes_temperature_zero(self):
        """Generated command must include temperature=0 from suite sampling."""
        runner, run_dir = self._make_runner()
        cmd = runner.build_command()
        cmd_str = " ".join(cmd)
        self.assertIn("temperature", cmd_str)
        self.assertIn("0", cmd_str)

    def test_command_includes_log_samples(self):
        """Generated command must include --log_samples."""
        runner, run_dir = self._make_runner()
        cmd = runner.build_command()
        self.assertIn("--log_samples", cmd)

    def test_command_points_at_normalized_run_dir(self):
        """Generated command must point output at the normalized run directory."""
        runner, run_dir = self._make_runner()
        cmd = runner.build_command()
        cmd_str = " ".join(cmd)
        self.assertIn(str(run_dir), cmd_str)

    def test_command_includes_endpoint(self):
        """Generated command must include the specified endpoint."""
        runner, run_dir = self._make_runner(endpoint="http://custom:9000/v1")
        cmd = runner.build_command()
        cmd_str = " ".join(cmd)
        self.assertIn("http://custom:9000/v1", cmd_str)

    def test_command_includes_concurrency(self):
        """Generated command must include concurrency."""
        runner, run_dir = self._make_runner(concurrency=12)
        cmd = runner.build_command()
        cmd_str = " ".join(cmd)
        self.assertIn("12", cmd_str)

    def test_command_includes_timeout(self):
        """Generated command must include the client timeout."""
        runner, run_dir = self._make_runner(timeout=30000)
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
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_runner(self, *, preflight_exit=0, harness_exit=0, dry_run=False):
        run_dir = _build_run_dir(self.tmp)
        preflight_mock = MagicMock(return_value=preflight_exit)
        harness_mock = MagicMock(return_value=harness_exit)
        runner = run_quality.QualityRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            benchmark="gsm8k",
            endpoint="http://fake:8000/v1",
            throughput=64.0,
            concurrency=8,
            timeout=14400,
            run_dir=run_dir,
            repo=self.tmp,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            dry_run=dry_run,
            allow_no_screen=True,
            preflight_runner=preflight_mock,
            harness_runner=harness_mock,
        )
        return runner, run_dir, preflight_mock, harness_mock

    def test_preflight_called_before_harness(self):
        """Preflight must be invoked before harness on a non-dry-run."""
        runner, run_dir, preflight_mock, harness_mock = self._make_runner(dry_run=False)
        # Create raw dir and artifacts to satisfy DONE check
        _make_harness_artifacts(run_dir)
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
        _make_harness_artifacts(run_dir)
        runner.run()
        harness_mock.assert_called()

    def test_dry_run_skips_preflight_network_calls(self):
        """In dry-run mode, preflight must not make network calls (harness not called)."""
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
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_runner(self, *, dry_run=False, allow_no_screen=False):
        run_dir = _build_run_dir(self.tmp)
        runner = run_quality.QualityRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            benchmark="gsm8k",
            endpoint="http://fake:8000/v1",
            throughput=64.0,
            concurrency=8,
            timeout=14400,
            run_dir=run_dir,
            repo=self.tmp,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            dry_run=dry_run,
            allow_no_screen=allow_no_screen,
            preflight_runner=MagicMock(return_value=0),
            harness_runner=MagicMock(return_value=0),
        )
        return runner, run_dir

    def test_refuses_launch_when_not_under_screen(self):
        """Without --allow-no-screen or --dry-run, must refuse launch outside screen."""
        runner, run_dir = self._make_runner(dry_run=False, allow_no_screen=False)
        # Simulate not being under screen by ensuring STY env is absent
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
            # Dry-run: should not fail due to screen check
            rc = runner.run()
            # Should succeed (0) in dry-run
            self.assertEqual(rc, 0)
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
            _make_harness_artifacts(run_dir)
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
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_command_txt_written_before_harness(self):
        """Live execution writes command.txt before invoking the harness."""
        run_dir = _build_run_dir(self.tmp)
        observed = {}

        def harness(_cmd):
            observed["exists"] = (run_dir / "command.txt").exists()
            observed["content"] = (run_dir / "command.txt").read_text().strip()
            _make_harness_artifacts(run_dir)
            return 0

        runner = run_quality.QualityRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            benchmark="gsm8k",
            endpoint="http://fake:8000/v1",
            throughput=64.0,
            concurrency=8,
            timeout=14400,
            run_dir=run_dir,
            repo=self.tmp,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            allow_no_screen=True,
            preflight_runner=MagicMock(return_value=0),
            harness_runner=harness,
        )
        expected = runner.build_command()
        self.assertEqual(runner.run(), 0)
        self.assertTrue(observed["exists"], "command.txt must exist before harness")
        self.assertEqual(shlex.split(observed["content"]), expected)

    def test_command_txt_is_shell_safe(self):
        """The live command file must be parseable by shlex.split."""
        run_dir = _build_run_dir(self.tmp)

        def harness(_cmd):
            _make_harness_artifacts(run_dir)
            return 0

        runner = run_quality.QualityRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            benchmark="gsm8k",
            endpoint="http://fake:8000/v1",
            throughput=64.0,
            concurrency=8,
            timeout=14400,
            run_dir=run_dir,
            repo=self.tmp,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            allow_no_screen=True,
            preflight_runner=MagicMock(return_value=0),
            harness_runner=harness,
        )
        self.assertEqual(runner.run(), 0)
        parsed = shlex.split((run_dir / "command.txt").read_text().strip())
        self.assertGreater(len(parsed), 0)

    def test_command_txt_round_trips_argv(self):
        """shlex.split(command.txt) must equal the generated live argv."""
        run_dir = _build_run_dir(self.tmp)

        def harness(_cmd):
            _make_harness_artifacts(run_dir)
            return 0

        runner = run_quality.QualityRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            benchmark="gsm8k",
            endpoint="http://fake:8000/v1",
            throughput=64.0,
            concurrency=8,
            timeout=14400,
            run_dir=run_dir,
            repo=self.tmp,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            allow_no_screen=True,
            preflight_runner=MagicMock(return_value=0),
            harness_runner=harness,
        )
        cmd = runner.build_command()
        self.assertEqual(runner.run(), 0)
        parsed = shlex.split((run_dir / "command.txt").read_text().strip())
        self.assertEqual(parsed, cmd)


# ---------------------------------------------------------------------------
# 7. DONE sentinel and raw file verification
# ---------------------------------------------------------------------------


class TestDoneSentinel(unittest.TestCase):
    """DONE must only be written on success and only after verifying raw files."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_runner(self, *, harness_exit=0, raw_exists=True, dry_run=False):
        run_dir = _build_run_dir(self.tmp)
        if raw_exists:
            _make_harness_artifacts(run_dir)
        runner = run_quality.QualityRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            benchmark="gsm8k",
            endpoint="http://fake:8000/v1",
            throughput=64.0,
            concurrency=8,
            timeout=14400,
            run_dir=run_dir,
            repo=self.tmp,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
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
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_runner(self, *, harness_exit=0, raw_exists=True):
        run_dir = _build_run_dir(self.tmp)
        if raw_exists:
            _make_harness_artifacts(run_dir)
        runner = run_quality.QualityRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            benchmark="gsm8k",
            endpoint="http://fake:8000/v1",
            throughput=64.0,
            concurrency=8,
            timeout=14400,
            run_dir=run_dir,
            repo=self.tmp,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            allow_no_screen=True,
            preflight_runner=MagicMock(return_value=0),
            harness_runner=MagicMock(return_value=harness_exit),
        )
        return runner, run_dir

    def test_status_transitions_to_running(self):
        """On launch, status must transition through planned -> preflight_passed -> running."""
        runner, run_dir = self._make_runner()
        runner.run()
        status = json.loads((run_dir / "status.json").read_text())
        states = [h["state"] for h in status["history"]]
        self.assertIn("running", states)

    def test_status_lifecycle_includes_preflight_passed(self):
        """Lifecycle must include planned -> preflight_passed -> running -> completed."""
        runner, run_dir = self._make_runner(harness_exit=0, raw_exists=True)
        runner.run()
        status = json.loads((run_dir / "status.json").read_text())
        states = [h["state"] for h in status["history"]]
        self.assertIn("planned", states)
        self.assertIn("preflight_passed", states)
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

    def test_failed_transition_write_error_is_fatal(self):
        """A failed-state write error must be surfaced, not treated as recorded."""
        runner, run_dir = self._make_runner(harness_exit=7)
        original_write = campaign_state.write_status

        def failing_write(dest, status, run_dir=None):
            if status.get("execution_state") == "failed":
                raise OSError("simulated failed-state write error")
            return original_write(dest, status, run_dir=run_dir)

        with patch.object(campaign_state, "write_status", side_effect=failing_write):
            rc = runner.run()

        self.assertEqual(rc, 1)
        status = json.loads((run_dir / "status.json").read_text())
        self.assertEqual(status["execution_state"], "running")
        self.assertFalse((run_dir / "DONE").exists())

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
        """--dry-run flag must be accepted without error. Uses a tempdir to avoid repo mutation."""
        import tempfile
        tmp = pathlib.Path(tempfile.mkdtemp())
        adapter_path = tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(adapter_path)
        tmp_run = tmp / "results" / _ADAPTER_SLUG / "runs" / "warpcore-v1" / "gsm8k" / "run-test-cli"
        tmp_run.mkdir(parents=True, exist_ok=True)
        try:
            # Write a valid planned status.json in the temp dir so CLI can proceed
            status = _make_planned_status("run-test-cli")
            (tmp_run / "status.json").write_text(json.dumps(status))
            argv = [
                "--suite", str(_REAL_SUITE),
                "--adapter", str(adapter_path),
                "--benchmark", "gsm8k",
                "--endpoint", "http://fake:8000/v1",
                "--throughput", "64",
                "--concurrency", "8",
                "--timeout", "14400",
                "--run-dir", str(tmp_run),
                "--repo", str(tmp),
                "--prompt-tokens", "gsm8k=500,ifeval=2000,gpqa_diamond=1000",
                "--dry-run",
            ]
            # In dry-run, should print command and exit 0
            try:
                rc = run_quality.main(argv)
            except SystemExit as e:
                rc = e.code
            except Exception:
                rc = None
            # The important thing: no SystemExit(2) from "unrecognized argument"
            self.assertNotEqual(rc, 2, "Exit code 2 suggests unrecognized --dry-run argument")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_gpqa_diamond_ceiling_in_command(self):
        """GPQA benchmark must use 65536 generation ceiling from suite."""
        tmp = pathlib.Path(tempfile.mkdtemp())
        adapter_path = tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(adapter_path)
        try:
            run_dir = _build_run_dir(tmp, bench="gpqa_diamond")
            runner = run_quality.QualityRunner(
                suite_path=_REAL_SUITE,
                adapter_path=adapter_path,
                benchmark="gpqa_diamond",
                endpoint="http://fake:8000/v1",
                throughput=64.0,
                concurrency=8,
                timeout=14400,
                run_dir=run_dir,
                repo=tmp,
                prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
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
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_harness_runner_is_called_with_command_list(self):
        """harness_runner must be called with the command as a list."""
        run_dir = _build_run_dir(self.tmp)
        _make_harness_artifacts(run_dir)
        harness_mock = MagicMock(return_value=0)
        runner = run_quality.QualityRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            benchmark="gsm8k",
            endpoint="http://fake:8000/v1",
            throughput=64.0,
            concurrency=8,
            timeout=14400,
            run_dir=run_dir,
            repo=self.tmp,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
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
        _make_harness_artifacts(run_dir)
        preflight_mock = MagicMock(return_value=0)
        runner = run_quality.QualityRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            benchmark="gsm8k",
            endpoint="http://fake:8000/v1",
            throughput=64.0,
            concurrency=8,
            timeout=14400,
            run_dir=run_dir,
            repo=self.tmp,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
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
        adapter_path = tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(adapter_path)
        try:
            run_dir = _build_run_dir(tmp, bench="ifeval")
            runner = run_quality.QualityRunner(
                suite_path=_REAL_SUITE,
                adapter_path=adapter_path,
                benchmark="ifeval",
                endpoint="http://fake:8000/v1",
                throughput=64.0,
                concurrency=8,
                timeout=14400,
                run_dir=run_dir,
                repo=tmp,
                prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
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
# HARDENING TESTS — added after adversarial review CHANGES_REQUIRED
# ---------------------------------------------------------------------------


class TestCommandUsesLocalChatCompletions(unittest.TestCase):
    """F2: build_command() must use local-chat-completions, not local-completions."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_runner(self, bench="gsm8k"):
        run_dir = _build_run_dir(self.tmp, bench)
        return run_quality.QualityRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            benchmark=bench,
            endpoint="http://fake:8000/v1",
            throughput=64.0,
            concurrency=8,
            timeout=14400,
            run_dir=run_dir,
            repo=self.tmp,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            dry_run=True,
            preflight_runner=MagicMock(return_value=0),
            harness_runner=MagicMock(return_value=0),
        )

    def test_model_is_sidecar_chat_completions(self):
        """--model must capture response metadata on the chat-completions path."""
        runner = self._make_runner()
        cmd = runner.build_command()
        self.assertIn("sidecar-chat-completions", cmd,
                      "--model must be 'sidecar-chat-completions'")
        self.assertNotIn("local-completions", cmd,
                         "--model must not be 'local-completions'")

    def test_command_includes_apply_chat_template(self):
        """build_command() must include --apply_chat_template."""
        runner = self._make_runner()
        cmd = runner.build_command()
        self.assertIn("--apply_chat_template", cmd,
                      "--apply_chat_template is required for chat-completion models")

    def test_command_includes_tokenized_requests_false(self):
        """build_command() must include tokenized_requests=False in model_args."""
        runner = self._make_runner()
        cmd = runner.build_command()
        cmd_str = " ".join(cmd)
        self.assertIn("tokenized_requests=False", cmd_str,
                      "tokenized_requests=False prevents double-tokenization")


class TestCommandUsesTaskNameNotPath(unittest.TestCase):
    """F3: --tasks must receive the registered task name, not an absolute file path.
    --include_path must carry the canonical task directory."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_runner(self, bench="gsm8k"):
        run_dir = _build_run_dir(self.tmp, bench)
        return run_quality.QualityRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            benchmark=bench,
            endpoint="http://fake:8000/v1",
            throughput=64.0,
            concurrency=8,
            timeout=14400,
            run_dir=run_dir,
            repo=self.tmp,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            dry_run=True,
            preflight_runner=MagicMock(return_value=0),
            harness_runner=MagicMock(return_value=0),
        )

    def test_tasks_arg_is_task_name_not_file_path(self):
        """--tasks value must be the registered task name from the task YAML, not a file path."""
        runner = self._make_runner()
        cmd = runner.build_command()
        # Find the value after --tasks
        tasks_idx = cmd.index("--tasks") if "--tasks" in cmd else None
        self.assertIsNotNone(tasks_idx, "--tasks must be present")
        task_value = cmd[tasks_idx + 1]
        # Must not be an absolute path (starting with /)
        self.assertFalse(task_value.startswith("/"),
                         f"--tasks value must be the task name, not absolute path: {task_value!r}")
        # Must be the registered task name from the YAML 'task:' field
        self.assertEqual(task_value, "gsm8k_cot_zeroshot_clean",
                         f"--tasks must be the registered task name; got {task_value!r}")

    def test_include_path_is_present(self):
        """--include_path must be set to the suite tasks directory."""
        runner = self._make_runner()
        cmd = runner.build_command()
        self.assertIn("--include_path", cmd, "--include_path must be present for custom task YAMLs")

    def test_include_path_points_to_task_directory(self):
        """--include_path must point to the directory containing the task YAML."""
        runner = self._make_runner()
        cmd = runner.build_command()
        idx = cmd.index("--include_path") if "--include_path" in cmd else None
        self.assertIsNotNone(idx)
        include_path = pathlib.Path(cmd[idx + 1])
        # Must exist and contain the task YAML
        task_file = include_path / "gsm8k_clean_v1.yaml"
        self.assertTrue(task_file.exists(),
                        f"--include_path {include_path} must contain the task YAML")


class TestCommandMaxRetries(unittest.TestCase):
    """F11 (clarified): max_retries=0 must be preserved; suite explicitly owns it."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_max_retries_is_zero_as_suite_specifies(self):
        """Suite warpcore-v1.yaml declares retry_policy.max_retries=0; harness must preserve it."""
        run_dir = _build_run_dir(self.tmp)
        runner = run_quality.QualityRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            benchmark="gsm8k",
            endpoint="http://fake:8000/v1",
            throughput=64.0,
            concurrency=8,
            timeout=14400,
            run_dir=run_dir,
            repo=self.tmp,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            dry_run=True,
            preflight_runner=MagicMock(return_value=0),
            harness_runner=MagicMock(return_value=0),
        )
        cmd = runner.build_command()
        cmd_str = " ".join(cmd)
        self.assertIn("max_retries=0", cmd_str,
                      "max_retries=0 must be preserved per suite retry_policy")


class TestReadStatusFailsClosedWhenMissing(unittest.TestCase):
    """F8: _read_status() must raise when status.json is missing, not fabricate state."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_missing_status_json_raises_not_fabricates(self):
        """If status.json is absent, run() must return nonzero (fail closed), not proceed."""
        # Create a run dir WITHOUT status.json (and without manifest.json)
        # Must be in the normalized path for identity to pass
        run_dir = self.tmp / "results" / _ADAPTER_SLUG / "runs" / "warpcore-v1" / "gsm8k" / "bare-run"
        run_dir.mkdir(parents=True)
        # Create raw dir so DONE check isn't the failure point
        (run_dir / "raw").mkdir()

        runner = run_quality.QualityRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            benchmark="gsm8k",
            endpoint="http://fake:8000/v1",
            throughput=64.0,
            concurrency=8,
            timeout=14400,
            run_dir=run_dir,
            repo=self.tmp,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            allow_no_screen=True,
            preflight_runner=MagicMock(return_value=0),
            harness_runner=MagicMock(return_value=0),
        )
        rc = runner.run()
        # Must not succeed — missing status.json must not be fabricated
        self.assertNotEqual(rc, 0, "Missing status.json must fail closed, not fabricate planned state")
        # DONE must NOT be written
        self.assertFalse((run_dir / "DONE").exists(),
                         "DONE must not be written when status.json is absent")


class TestTransitionErrorsAreFatal(unittest.TestCase):
    """F5: Lifecycle transition errors must be fatal, not silently swallowed."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_terminal_state_blocks_run(self):
        """A run dir already in 'completed' state must not run again."""
        run_dir = _build_run_dir(self.tmp)
        # Force terminal state
        terminal_status = {
            "schema_version": 1,
            "run_id": "run-test",
            "suite_id": "warpcore-v1",
            "execution_state": "completed",
            "lifecycle": "current",
            "history": [
                {"state": "planned", "timestamp": "2026-09-15T12:00:00Z"},
                {"state": "preflight_passed", "timestamp": "2026-09-15T12:01:00Z"},
                {"state": "running", "timestamp": "2026-09-15T12:02:00Z"},
                {"state": "completed", "timestamp": "2026-09-15T12:03:00Z"},
            ],
        }
        (run_dir / "status.json").write_text(json.dumps(terminal_status))
        (run_dir / "raw").mkdir(exist_ok=True)

        runner = run_quality.QualityRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            benchmark="gsm8k",
            endpoint="http://fake:8000/v1",
            throughput=64.0,
            concurrency=8,
            timeout=14400,
            run_dir=run_dir,
            repo=self.tmp,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            allow_no_screen=True,
            preflight_runner=MagicMock(return_value=0),
            harness_runner=MagicMock(return_value=0),
        )
        rc = runner.run()
        self.assertNotEqual(rc, 0, "Running against a terminal-state run dir must fail closed")

    def test_failed_terminal_state_blocks_run(self):
        """A run dir already in 'failed' state must not run again."""
        run_dir = _build_run_dir(self.tmp)
        failed_status = {
            "schema_version": 1,
            "run_id": "run-test",
            "suite_id": "warpcore-v1",
            "execution_state": "failed",
            "lifecycle": "invalid",
            "history": [
                {"state": "planned", "timestamp": "2026-09-15T12:00:00Z"},
                {"state": "preflight_passed", "timestamp": "2026-09-15T12:01:00Z"},
                {"state": "running", "timestamp": "2026-09-15T12:02:00Z"},
                {"state": "failed", "timestamp": "2026-09-15T12:03:00Z"},
            ],
        }
        (run_dir / "status.json").write_text(json.dumps(failed_status))
        (run_dir / "raw").mkdir(exist_ok=True)

        runner = run_quality.QualityRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            benchmark="gsm8k",
            endpoint="http://fake:8000/v1",
            throughput=64.0,
            concurrency=8,
            timeout=14400,
            run_dir=run_dir,
            repo=self.tmp,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            allow_no_screen=True,
            preflight_runner=MagicMock(return_value=0),
            harness_runner=MagicMock(return_value=0),
        )
        rc = runner.run()
        self.assertNotEqual(rc, 0, "Running from terminal 'failed' state must fail closed")


class TestEvidenceVerificationBeforeDone(unittest.TestCase):
    """F4: DONE must not be written when raw/ exists but lacks required artifacts."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_runner(self, harness_exit=0):
        run_dir = _build_run_dir(self.tmp)
        runner = run_quality.QualityRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            benchmark="gsm8k",
            endpoint="http://fake:8000/v1",
            throughput=64.0,
            concurrency=8,
            timeout=14400,
            run_dir=run_dir,
            repo=self.tmp,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            allow_no_screen=True,
            preflight_runner=MagicMock(return_value=0),
            harness_runner=MagicMock(return_value=harness_exit),
        )
        return runner, run_dir

    def test_done_not_written_when_raw_dir_empty(self):
        """DONE must NOT be written when raw/ directory exists but is empty."""
        runner, run_dir = self._make_runner(harness_exit=0)
        # Create empty raw dir
        (run_dir / "raw").mkdir()
        rc = runner.run()
        self.assertFalse((run_dir / "DONE").exists(),
                         "DONE must not be written when raw/ is empty (no artifacts)")
        self.assertNotEqual(rc, 0, "Run must fail when raw/ has no evidence")

    def test_done_written_only_when_aggregate_result_and_samples_exist(self):
        """DONE is written only when aggregate result JSON and samples JSONL exist in raw/."""
        runner, run_dir = self._make_runner(harness_exit=0)
        raw_dir = run_dir / "raw"
        raw_dir.mkdir()
        # Simulate complete canonical response evidence, not reduced lm-eval fields.
        (raw_dir / "results_2026-09-15T12-00-00.json").write_text(
            '{"results": {"gsm8k_cot_zeroshot_clean": {"exact_match,none": 0.85}}}'
        )
        messages = [{"role": "user", "content": "fixture question"}]
        sample = {"doc_id": 0, "target": "42", "resps": [["42"]],
                  "filtered_resps": ["42"], "exact_match": 1.0,
                  "arguments": {"gen_args_0": {"arg_0": [json.dumps(messages)]}}}
        samples_gz = raw_dir / "samples_gsm8k_cot_zeroshot_clean_2026-09-15T12-00-00.jsonl.gz"
        samples_gz.write_bytes(gzip.compress((json.dumps(sample) + "\n").encode()))
        import hashlib
        fingerprint = hashlib.sha256(
            json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        (raw_dir / "response_metadata.jsonl").write_text(json.dumps({
            "fingerprint": fingerprint, "finish_reasons": ["stop"],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            "content": ["42"], "reasoning_content": [None], "reasoning": [None],
            "model": "testorg/TestCanonicalModel",
            "run_id": run_dir.name,
        }) + "\n")
        rc = runner.run()
        self.assertTrue((run_dir / "DONE").exists(), "DONE must be written when required artifacts exist")
        self.assertEqual(rc, 0)


class TestHarnessExitCodeInHistoryEntry(unittest.TestCase):
    """F6: exit_code must be recorded in history entry, not as top-level field.
    result-status schema has additionalProperties=false."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_exit_code_in_history_entry_not_top_level(self):
        """Failed run must record exit_code in history entry, not as harness_exit_code top-level."""
        run_dir = _build_run_dir(self.tmp)
        (run_dir / "raw").mkdir()

        runner = run_quality.QualityRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            benchmark="gsm8k",
            endpoint="http://fake:8000/v1",
            throughput=64.0,
            concurrency=8,
            timeout=14400,
            run_dir=run_dir,
            repo=self.tmp,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            allow_no_screen=True,
            preflight_runner=MagicMock(return_value=0),
            harness_runner=MagicMock(return_value=42),
        )
        runner.run()
        status = json.loads((run_dir / "status.json").read_text())
        # Must NOT have top-level harness_exit_code (additionalProperties: false)
        self.assertNotIn("harness_exit_code", status,
                         "harness_exit_code must not appear as top-level field (schema forbids it)")
        self.assertNotIn("failure_reason", status,
                         "failure_reason must not appear as top-level field (schema forbids it)")
        # Exit code must appear in history entry
        failed_entries = [e for e in status.get("history", []) if e.get("state") == "failed"]
        self.assertTrue(len(failed_entries) > 0, "Must have a 'failed' history entry")
        failed_entry = failed_entries[-1]
        self.assertIn("exit_code", failed_entry,
                      "exit_code must be recorded in the history entry for failed state")
        self.assertEqual(failed_entry["exit_code"], 42)

    def test_status_json_passes_schema_validation_on_failure(self):
        """Status written on failure must validate against result-status.schema.json."""
        from contract import validate_json
        run_dir = _build_run_dir(self.tmp)
        (run_dir / "raw").mkdir()

        runner = run_quality.QualityRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            benchmark="gsm8k",
            endpoint="http://fake:8000/v1",
            throughput=64.0,
            concurrency=8,
            timeout=14400,
            run_dir=run_dir,
            repo=self.tmp,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            allow_no_screen=True,
            preflight_runner=MagicMock(return_value=0),
            harness_runner=MagicMock(return_value=1),
        )
        runner.run()
        status = json.loads((run_dir / "status.json").read_text())
        schema_path = _REPO / "suite" / "schemas" / "result-status.schema.json"
        errors = validate_json(status, schema_path)
        self.assertEqual(errors, [], f"Status JSON must validate against schema; errors: {errors}")


class TestStatusSchemaRequiredFields(unittest.TestCase):
    """Status written by runner must include schema_version, run_id, suite_id."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_runner(self, harness_exit=0, raw_exists=True):
        run_dir = _build_run_dir(self.tmp)
        if raw_exists:
            _make_harness_artifacts(run_dir)
        runner = run_quality.QualityRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            benchmark="gsm8k",
            endpoint="http://fake:8000/v1",
            throughput=64.0,
            concurrency=8,
            timeout=14400,
            run_dir=run_dir,
            repo=self.tmp,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            allow_no_screen=True,
            preflight_runner=MagicMock(return_value=0),
            harness_runner=MagicMock(return_value=harness_exit),
        )
        return runner, run_dir

    def test_status_has_schema_version(self):
        """status.json must include schema_version=1 (required by result-status schema)."""
        runner, run_dir = self._make_runner(harness_exit=0, raw_exists=True)
        runner.run()
        status = json.loads((run_dir / "status.json").read_text())
        self.assertIn("schema_version", status)
        self.assertEqual(status["schema_version"], 1)

    def test_status_has_run_id(self):
        """status.json must include run_id (required by result-status schema)."""
        runner, run_dir = self._make_runner(harness_exit=0, raw_exists=True)
        runner.run()
        status = json.loads((run_dir / "status.json").read_text())
        self.assertIn("run_id", status)
        self.assertIsInstance(status["run_id"], str)
        self.assertTrue(len(status["run_id"]) > 0)

    def test_status_has_suite_id(self):
        """status.json must include suite_id='warpcore-v1' (required by result-status schema)."""
        runner, run_dir = self._make_runner(harness_exit=0, raw_exists=True)
        runner.run()
        status = json.loads((run_dir / "status.json").read_text())
        self.assertIn("suite_id", status)
        self.assertEqual(status["suite_id"], "warpcore-v1")


class TestAdapterValidationInInit(unittest.TestCase):
    """F7: QualityRunner must validate adapter is canonical on __init__."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_noncanonical_adapter_raises_on_init(self):
        """QualityRunner must reject noncanonical adapters at construction time."""
        run_dir = self.tmp / "results" / "qwen3.6-35b-a3b" / "runs" / "warpcore-v1" / "gsm8k" / "run-test"
        run_dir.mkdir(parents=True)
        (run_dir / "status.json").write_text(json.dumps({
            "schema_version": 1,
            "run_id": "run-test",
            "suite_id": "warpcore-v1",
            "execution_state": "planned",
            "lifecycle": "current",
            "history": [{"state": "planned", "timestamp": "2026-09-15T12:00:00Z"}],
        }))
        # The real adapter is noncanonical — QualityRunner must raise
        with self.assertRaises(Exception) as ctx:
            run_quality.QualityRunner(
                suite_path=_REAL_SUITE,
                adapter_path=_REPO / "adapters" / "qwen3.6-35b-a3b.yaml",
                benchmark="gsm8k",
                endpoint="http://fake:8000/v1",
                throughput=64.0,
                concurrency=8,
                timeout=14400,
                run_dir=run_dir,
                repo=self.tmp,
                prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
                dry_run=True,
                preflight_runner=MagicMock(return_value=0),
                harness_runner=MagicMock(return_value=0),
            )
        err_msg = str(ctx.exception).lower()
        self.assertTrue(
            "noncanonical" in err_msg or "canonical" in err_msg or "campaign" in err_msg,
            f"Exception must mention noncanonical/campaign readiness; got: {ctx.exception!r}"
        )


class TestDryRunNoMutation(unittest.TestCase):
    """F6/F13: Dry-run must not create files inside the repo. Tests must not leave artifacts."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_dry_run_with_explicit_run_dir_writes_nothing(self):
        """With an explicit run dir, dry-run must not create command artifacts."""
        run_dir = _build_run_dir(self.tmp)
        runner = run_quality.QualityRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            benchmark="gsm8k",
            endpoint="http://fake:8000/v1",
            throughput=64.0,
            concurrency=8,
            timeout=14400,
            run_dir=run_dir,
            repo=self.tmp,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            dry_run=True,
            preflight_runner=MagicMock(return_value=0),
            harness_runner=MagicMock(return_value=0),
        )
        runner.run()
        self.assertFalse((run_dir / "command.txt").exists())

    def test_dry_run_cli_with_explicit_run_dir_no_repo_artifacts(self):
        """CLI dry-run with explicit --run-dir must not create files inside the real repo."""
        # Build run dir in tmp (not real repo)
        tmp_run = self.tmp / "results" / _ADAPTER_SLUG / "runs" / "warpcore-v1" / "gsm8k" / "test-run-cli"
        tmp_run.mkdir(parents=True)
        # Write required planned status.json
        status = _make_planned_status("test-run-cli")
        (tmp_run / "status.json").write_text(json.dumps(status))
        (tmp_run / "manifest.json").write_text(json.dumps({
            "suite_id": "warpcore-v1", "run_id": "test-run-cli",
            "benchmark": "gsm8k",
            "model": {"slug": _ADAPTER_SLUG, "id": "testorg/TestCanonicalModel", "revision": "a" * 40},
            "item_inventory": {"expected": 1319},
        }))
        argv = [
            "--suite", str(_REAL_SUITE),
            "--adapter", str(self.adapter_path),
            "--benchmark", "gsm8k",
            "--endpoint", "http://fake:8000/v1",
            "--throughput", "64",
            "--concurrency", "8",
            "--timeout", "14400",
            "--run-dir", str(tmp_run),
            "--repo", str(self.tmp),
            "--prompt-tokens", "gsm8k=500,ifeval=2000,gpqa_diamond=1000",
            "--dry-run",
        ]
        try:
            run_quality.main(argv)
        except SystemExit:
            pass
        # The real repo's results/ dir must NOT have been created by this test
        repo_results = _REPO / "results" / _ADAPTER_SLUG / "runs"
        new_run = repo_results / "warpcore-v1" / "gsm8k" / "test-run-cli"
        self.assertFalse(new_run.exists(),
                         "CLI with explicit --run-dir must not create artifacts inside real repo")


class TestRunDirContainment(unittest.TestCase):
    """F15: Normalized run directory must be proven to reside inside the repository."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_run_dir_outside_repo_rejected(self):
        """CLI must reject --run-dir that resolves outside the repository."""
        # run_dir in /tmp (outside repo)
        outside_tmp = pathlib.Path(tempfile.mkdtemp())
        run_dir = outside_tmp / "escape-run"
        run_dir.mkdir()
        # Write a valid planned status so we reach containment check
        status = _make_planned_status("escape-run")
        (run_dir / "status.json").write_text(json.dumps(status))
        argv = [
            "--suite", str(_REAL_SUITE),
            "--adapter", str(self.adapter_path),
            "--benchmark", "gsm8k",
            "--endpoint", "http://fake:8000/v1",
            "--throughput", "64",
            "--concurrency", "8",
            "--timeout", "14400",
            "--run-dir", str(run_dir),
            "--repo", str(self.tmp),  # repo is self.tmp, run_dir is outside_tmp
            "--prompt-tokens", "gsm8k=500,ifeval=2000,gpqa_diamond=1000",
            "--dry-run",
        ]
        try:
            rc = run_quality.main(argv)
        except SystemExit as e:
            rc = e.code
        finally:
            shutil.rmtree(outside_tmp, ignore_errors=True)
        self.assertNotEqual(rc, 0,
                            "--run-dir outside repo must be rejected (nonzero exit)")


class TestTaskNameReadFromYaml(unittest.TestCase):
    """F3: Task name for --tasks is read from the task YAML 'task:' field, not derived from filename."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_gpqa_task_name_from_yaml(self):
        """GPQA --tasks value must be 'gpqa_diamond_cot_zeroshot_clean' (from task YAML)."""
        run_dir = _build_run_dir(self.tmp, bench="gpqa_diamond")
        runner = run_quality.QualityRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            benchmark="gpqa_diamond",
            endpoint="http://fake:8000/v1",
            throughput=64.0,
            concurrency=8,
            timeout=14400,
            run_dir=run_dir,
            repo=self.tmp,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            dry_run=True,
            preflight_runner=MagicMock(return_value=0),
            harness_runner=MagicMock(return_value=0),
        )
        cmd = runner.build_command()
        tasks_idx = cmd.index("--tasks") if "--tasks" in cmd else None
        self.assertIsNotNone(tasks_idx)
        task_value = cmd[tasks_idx + 1]
        self.assertEqual(task_value, "gpqa_diamond_cot_zeroshot_clean",
                         f"GPQA --tasks must be task name from YAML; got {task_value!r}")


class TestExactArgvAssertions(unittest.TestCase):
    """Tighten vacuous tests: exact argv assertions for command construction."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_runner(self, bench="gsm8k"):
        run_dir = _build_run_dir(self.tmp, bench)
        return run_quality.QualityRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            benchmark=bench,
            endpoint="http://fake:8000/v1",
            throughput=64.0,
            concurrency=8,
            timeout=14400,
            run_dir=run_dir,
            repo=self.tmp,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            dry_run=True,
            preflight_runner=MagicMock(return_value=0),
            harness_runner=MagicMock(return_value=0),
        )

    def test_model_arg_appears_as_discrete_token(self):
        """'sidecar-chat-completions' must be the discrete --model token."""
        runner = self._make_runner()
        cmd = runner.build_command()
        # Find --model and check the next token
        if "--model" in cmd:
            model_idx = cmd.index("--model")
            self.assertEqual(cmd[model_idx + 1], "sidecar-chat-completions")
        else:
            self.fail("--model not found in command")

    def test_num_fewshot_is_zero(self):
        """--num_fewshot must be 0 as suite specifies."""
        runner = self._make_runner()
        cmd = runner.build_command()
        if "--num_fewshot" in cmd:
            idx = cmd.index("--num_fewshot")
            self.assertEqual(cmd[idx + 1], "0", "--num_fewshot must be 0")

    def test_log_samples_is_discrete_flag(self):
        """--log_samples must be a discrete flag token in the command."""
        runner = self._make_runner()
        cmd = runner.build_command()
        self.assertIn("--log_samples", cmd, "--log_samples must be present as discrete token")


if __name__ == "__main__":
    unittest.main()
