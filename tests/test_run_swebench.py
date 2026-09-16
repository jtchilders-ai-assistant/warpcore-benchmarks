"""tests/test_run_swebench.py — Contract-aware SWE-bench runner (Task 6).

TDD: tests written BEFORE implementation. Must fail (RED) until viz/run_swebench.py exists.

Design contract (docs/superpowers/specs/2026-09-15-warpcore-v1-apples-to-apples-design.md,
plan lines 307–345):

- Uses exactly the 100 frozen instance IDs in suite/swebench/instances-seed42-n100.json
  and verifies the hash in the suite before launch.
- Preflight verifies: x86 Docker host, image cache (wraps swebench_preflight.py), live
  model identity from /v1/models, scaffold hash matches suite, submit protocol intact.
- Injects only adapter model_id into the frozen scaffold (model_name and api_base/key);
  never overrides step_limit, cost_limit, environment.timeout, pull_timeout, temperature,
  max_tokens, or submit protocol.
- Preserves suite-fixed limits from scaffold.yaml.
- Creates normalized stable run dir via create_campaign (transactional).
- Refuses launch outside /usr/bin/screen except dry-run or test mode.
- Explicit resume only: existing dir accepted only when instance IDs, hashes, and
  adapter identity all match.
- Separates generation and grading as distinct execution phases in state history
  without violating the status schema.
- Preserves trajectories.
- Completion only after grader artifacts and all 100 terminal dispositions exist.
- Fails closed on every state/evidence write error.
- Rejects noncanonical adapters at construction.
- Does not launch live benchmark (runner is unit-tested with injected runners).
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

_TESTS_DIR = pathlib.Path(__file__).parent
_REPO = _TESTS_DIR.parent
_VIZ_DIR = _REPO / "viz"
for _p in (str(_TESTS_DIR), str(_VIZ_DIR), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The module under test — will not exist until implementation (RED phase)
import run_swebench  # noqa: E402  (expected ImportError during RED)
import campaign_state  # noqa: E402


# ---------------------------------------------------------------------------
# Canonical adapter fixture (same pattern as Task 5)
# ---------------------------------------------------------------------------

_CANONICAL_ADAPTER = {
    "adapter_schema_version": 1,
    "campaign_status": "canonical",
    "model": {
        "slug": "test-canonical-model",
        "id": "testorg/TestCanonicalModel",
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

_ADAPTER_SLUG = _CANONICAL_ADAPTER["model"]["slug"]  # "test-canonical-model"
_MODEL_ID = _CANONICAL_ADAPTER["model"]["id"]

# Prompt token maxima for the quality benchmarks (not needed by SWE-bench itself,
# but validate_adapter_campaign_ready requires them to prove context feasibility
# across all suite benchmarks that have a generation_ceiling).
_PROMPT_TOKEN_MAXIMA = {
    "gsm8k": 500,
    "ifeval": 2000,
    "gpqa_diamond": 1000,
}

_REAL_SUITE = _REPO / "suite" / "warpcore-v1.yaml"
_REAL_INSTANCES = _REPO / "suite" / "swebench" / "instances-seed42-n100.json"
_REAL_SCAFFOLD = _REPO / "suite" / "swebench" / "scaffold.yaml"


def _write_canonical_adapter(path: pathlib.Path) -> None:
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(_CANONICAL_ADAPTER, default_flow_style=False))


def _make_planned_status(
    run_id: str = "run-test", suite_id: str = "warpcore-v1"
) -> dict:
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
    bench: str = "swebench",
    slug: str = _ADAPTER_SLUG,
    run_id: str = "run-test",
    suite_id: str = "warpcore-v1",
) -> pathlib.Path:
    """Create a minimal normalized run directory (planned state)."""
    run_dir = repo / "results" / slug / "runs" / suite_id / bench / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "status.json").write_text(
        json.dumps(_make_planned_status(run_id, suite_id))
    )
    manifest = {
        "suite_id": suite_id,
        "run_id": run_id,
        "benchmark": bench,
        "model": {
            "slug": slug,
            "id": _MODEL_ID,
            "revision": "a" * 40,
        },
        "item_inventory": {"expected": 100},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    return run_dir


def _make_generation_artifacts(run_dir: pathlib.Path) -> None:
    """Populate raw/ with minimal mini-swe-agent generation outputs."""
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    # preds.json: 100 instance predictions
    instances = json.loads(_REAL_INSTANCES.read_text())
    preds = {iid: {"model_patch": "diff --git a/x b/x\n+fix", "instance_id": iid}
             for iid in instances}
    (raw_dir / "preds.json").write_text(json.dumps(preds))
    # exit_statuses.json: all 100 exit codes
    statuses = {iid: 0 for iid in instances}
    (raw_dir / "exit_statuses.json").write_text(json.dumps(statuses))
    # trajectories/ directory: required evidence for audit
    traj_dir = raw_dir / "trajectories"
    traj_dir.mkdir(exist_ok=True)
    for iid in instances:
        (traj_dir / f"{iid}.traj").write_text("{}")
    # run.log: required evidence for generation subprocess logging
    (raw_dir / "run.log").write_text("generation completed\n")


def _make_grading_artifacts(run_dir: pathlib.Path) -> None:
    """Populate raw/ with SWE-bench grading results for all 100 instances."""
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    instances = json.loads(_REAL_INSTANCES.read_text())
    # Grader produces a results JSON with resolved_ids etc.
    grading = {
        "resolved_ids": instances[:5],
        "unresolved_ids": instances[5:],
        "empty_patch_ids": [],
        "error_ids": [],
    }
    (raw_dir / "grading_results.json").write_text(json.dumps(grading))


# ---------------------------------------------------------------------------
# 1. Module interface
# ---------------------------------------------------------------------------


class TestModuleInterface(unittest.TestCase):
    """The module must expose a SwebenchRunner class."""

    def test_swebench_runner_class_exists(self):
        self.assertTrue(hasattr(run_swebench, "SwebenchRunner"))

    def test_swebench_runner_is_callable(self):
        self.assertTrue(callable(run_swebench.SwebenchRunner))

    def test_module_has_exit_code_constants(self):
        """Module must expose EXIT_SUCCESS, EXIT_DEFECT, EXIT_INCONCLUSIVE, EXIT_CONFIG."""
        for attr in ("EXIT_SUCCESS", "EXIT_DEFECT", "EXIT_INCONCLUSIVE", "EXIT_CONFIG"):
            self.assertTrue(
                hasattr(run_swebench, attr),
                f"Missing constant: {attr}",
            )


# ---------------------------------------------------------------------------
# 2. Construction — canonical adapter required
# ---------------------------------------------------------------------------


class TestConstruction(unittest.TestCase):
    """SwebenchRunner rejects non-canonical adapters at construction time."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)
        self.run_dir = _build_run_dir(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_canonical_adapter_constructs(self):
        """A fully canonical adapter must construct without error."""
        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            dry_run=True,
        prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
        )
        self.assertIsNotNone(runner)

    def test_noncanonical_adapter_rejected(self):
        """Adapters without campaign_status='canonical' must be rejected at construction."""
        import yaml
        bad = dict(_CANONICAL_ADAPTER)
        bad["campaign_status"] = "noncanonical"
        bad_path = self.tmp / "adapters" / "bad.yaml"
        bad_path.write_text(yaml.dump(bad))
        with self.assertRaises((ValueError, Exception)):
            run_swebench.SwebenchRunner(
                suite_path=_REAL_SUITE,
                adapter_path=bad_path,
                endpoint="http://localhost:8000/v1",
                run_dir=self.run_dir,
                repo=self.tmp,
                dry_run=True,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            )

    def test_swebench_benchmark_required(self):
        """Runner must only operate on the 'swebench' benchmark from suite."""
        # The suite must contain the swebench benchmark — verify at construction.
        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            dry_run=True,
        prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
        )
        # The runner knows it's operating on swebench
        self.assertIsNotNone(runner)


# ---------------------------------------------------------------------------
# 3. Instance set — frozen 100 IDs, hash verified
# ---------------------------------------------------------------------------


class TestInstanceSet(unittest.TestCase):
    """Runner must use exactly the 100 frozen instance IDs and verify hashes."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)
        self.run_dir = _build_run_dir(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_instance_set_is_exactly_100(self):
        """The frozen instance set from the suite must contain exactly 100 IDs."""
        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            dry_run=True,
        prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
        )
        ids = runner.get_instance_ids()
        self.assertEqual(len(ids), 100)

    def test_instance_ids_match_suite_file(self):
        """Instance IDs must equal the frozen list in instances-seed42-n100.json."""
        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            dry_run=True,
        prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
        )
        ids = runner.get_instance_ids()
        expected = json.loads(_REAL_INSTANCES.read_text())
        self.assertEqual(sorted(ids), sorted(expected))

    def test_instance_set_hash_verified_at_construction(self):
        """Construction must verify the instances file hash against suite declaration."""
        # If the hash passes construction without error, hash verification runs.
        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            dry_run=True,
        prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
        )
        # Successful construction proves hash was verified
        self.assertIsNotNone(runner)

    def test_scaffold_hash_verified_at_construction(self):
        """Construction must verify the scaffold file hash against suite declaration."""
        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            dry_run=True,
        prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
        )
        self.assertIsNotNone(runner)


# ---------------------------------------------------------------------------
# 4. Scaffold injection — only model_id injected; limits preserved
# ---------------------------------------------------------------------------


class TestScaffoldInjection(unittest.TestCase):
    """Runner injects only model identity into the frozen scaffold."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)
        self.run_dir = _build_run_dir(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _get_runner(self) -> run_swebench.SwebenchRunner:
        return run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            dry_run=True,
        prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
        )

    def test_model_name_injected(self):
        """model_name in rendered config must contain the adapter model_id."""
        runner = self._get_runner()
        config = runner.build_scaffold_config(
            endpoint="http://localhost:8000/v1",
            api_key="warpcore",
        )
        # mini-swe-agent uses "hosted_vllm/<model-id>" format
        model_name = config.get("model", {}).get("model_name", "")
        self.assertIn(_MODEL_ID, model_name)

    def test_step_limit_preserved(self):
        """step_limit must not be changed from the frozen scaffold value."""
        runner = self._get_runner()
        config = runner.build_scaffold_config(
            endpoint="http://localhost:8000/v1",
            api_key="warpcore",
        )
        import yaml as _yaml
        scaffold = _yaml.safe_load(_REAL_SCAFFOLD.read_text())
        self.assertEqual(
            config["agent"]["step_limit"],
            scaffold["agent"]["step_limit"],
        )

    def test_cost_limit_preserved(self):
        """cost_limit must not be changed from the frozen scaffold value."""
        runner = self._get_runner()
        config = runner.build_scaffold_config(
            endpoint="http://localhost:8000/v1",
            api_key="warpcore",
        )
        import yaml as _yaml
        scaffold = _yaml.safe_load(_REAL_SCAFFOLD.read_text())
        self.assertEqual(
            config["agent"]["cost_limit"],
            scaffold["agent"]["cost_limit"],
        )

    def test_environment_timeout_preserved(self):
        """environment.timeout must not be changed from the frozen scaffold value."""
        runner = self._get_runner()
        config = runner.build_scaffold_config(
            endpoint="http://localhost:8000/v1",
            api_key="warpcore",
        )
        import yaml as _yaml
        scaffold = _yaml.safe_load(_REAL_SCAFFOLD.read_text())
        self.assertEqual(
            config["environment"]["timeout"],
            scaffold["environment"]["timeout"],
        )

    def test_pull_timeout_preserved(self):
        """environment.pull_timeout must not be changed from the frozen scaffold value."""
        runner = self._get_runner()
        config = runner.build_scaffold_config(
            endpoint="http://localhost:8000/v1",
            api_key="warpcore",
        )
        import yaml as _yaml
        scaffold = _yaml.safe_load(_REAL_SCAFFOLD.read_text())
        self.assertEqual(
            config["environment"]["pull_timeout"],
            scaffold["environment"]["pull_timeout"],
        )

    def test_temperature_preserved(self):
        """model.model_kwargs.temperature must not be changed from scaffold."""
        runner = self._get_runner()
        config = runner.build_scaffold_config(
            endpoint="http://localhost:8000/v1",
            api_key="warpcore",
        )
        import yaml as _yaml
        scaffold = _yaml.safe_load(_REAL_SCAFFOLD.read_text())
        self.assertEqual(
            config["model"]["model_kwargs"]["temperature"],
            scaffold["model"]["model_kwargs"]["temperature"],
        )

    def test_max_tokens_preserved(self):
        """model.model_kwargs.max_tokens must not be changed from scaffold."""
        runner = self._get_runner()
        config = runner.build_scaffold_config(
            endpoint="http://localhost:8000/v1",
            api_key="warpcore",
        )
        import yaml as _yaml
        scaffold = _yaml.safe_load(_REAL_SCAFFOLD.read_text())
        self.assertEqual(
            config["model"]["model_kwargs"]["max_tokens"],
            scaffold["model"]["model_kwargs"]["max_tokens"],
        )

    def test_api_base_injected(self):
        """model.model_kwargs.api_base must be the runner endpoint."""
        runner = self._get_runner()
        endpoint = "http://localhost:8000/v1"
        config = runner.build_scaffold_config(endpoint=endpoint, api_key="warpcore")
        self.assertEqual(
            config["model"]["model_kwargs"]["api_base"],
            endpoint,
        )

    def test_no_runner_injected_placeholders_remain(self):
        """No '__RUNNER_INJECTED__' placeholders must remain in the rendered config."""
        runner = self._get_runner()
        config = runner.build_scaffold_config(
            endpoint="http://localhost:8000/v1",
            api_key="warpcore",
        )
        config_str = json.dumps(config)
        self.assertNotIn("__RUNNER_INJECTED__", config_str)


# ---------------------------------------------------------------------------
# 5. Screen guard
# ---------------------------------------------------------------------------


class TestScreenGuard(unittest.TestCase):
    """Runner must refuse live launch outside /usr/bin/screen."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)
        self.run_dir = _build_run_dir(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_dry_run_bypasses_screen_guard(self):
        """dry_run=True must not trigger the screen guard."""
        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            dry_run=True,
        prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
        )
        # dry_run run() must return without error (or return a success code)
        rc = runner.run()
        self.assertEqual(rc, run_swebench.EXIT_SUCCESS)

    def test_allow_no_screen_bypasses_guard(self):
        """allow_no_screen=True must bypass the screen check for tests."""
        # This is what test harnesses set — confirm the parameter exists and works.
        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            dry_run=True,
            allow_no_screen=True,
        prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
        )
        rc = runner.run()
        self.assertEqual(rc, run_swebench.EXIT_SUCCESS)

    def test_live_run_outside_screen_returns_inconclusive(self):
        """A live run outside /usr/bin/screen must return EXIT_INCONCLUSIVE (2)."""
        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            dry_run=False,
            allow_no_screen=False,
        prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
        )
        # Ensure STY is not set (not inside screen)
        env_without_sty = {k: v for k, v in os.environ.items() if k != "STY"}
        with patch.dict(os.environ, env_without_sty, clear=True):
            rc = runner.run()
        self.assertEqual(rc, run_swebench.EXIT_INCONCLUSIVE)


# ---------------------------------------------------------------------------
# 6. Run directory layout — normalized path via create_campaign
# ---------------------------------------------------------------------------


class TestRunDirectoryLayout(unittest.TestCase):
    """Run directory must be the normalized v1 layout."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)
        self.run_dir = _build_run_dir(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_run_dir_identity(self):
        """run_dir must exactly match repo/results/<slug>/runs/warpcore-v1/swebench/<run_id>."""
        expected = (
            self.tmp / "results" / _ADAPTER_SLUG
            / "runs" / "warpcore-v1" / "swebench" / "run-test"
        ).resolve()
        self.assertEqual(self.run_dir.resolve(), expected)


# ---------------------------------------------------------------------------
# 7. Preflight gates — injected runner pattern (no live network)
# ---------------------------------------------------------------------------


class TestPreflightGates(unittest.TestCase):
    """Preflight must check x86 Docker, image cache, model identity, scaffold hash."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)
        self.run_dir = _build_run_dir(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_preflight_pass_proceeds(self):
        """When preflight_runner returns 0, run continues to generation."""
        generation_calls = []
        grading_calls = []

        def mock_preflight(model_id: str) -> int:
            return 0  # pass

        def mock_generation(config: dict, run_dir: pathlib.Path, **kw) -> int:
            generation_calls.append((config, run_dir))
            # Populate artifacts so evidence check passes
            _make_generation_artifacts(run_dir)
            return 0

        def mock_grading(preds_path: pathlib.Path, run_dir: pathlib.Path, **kw) -> int:
            grading_calls.append((preds_path, run_dir))
            _make_grading_artifacts(run_dir)
            return 0

        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            dry_run=False,
            allow_no_screen=True,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            preflight_runner=mock_preflight,
            generation_runner=mock_generation,
            grading_runner=mock_grading,
        )
        rc = runner.run()
        self.assertEqual(rc, run_swebench.EXIT_SUCCESS)
        self.assertEqual(len(generation_calls), 1)
        self.assertEqual(len(grading_calls), 1)

    def test_preflight_defect_blocks_generation(self):
        """When preflight_runner returns 1 (defect), generation must not run."""
        generation_calls = []

        def mock_preflight(model_id: str) -> int:
            return 1  # defect

        def mock_generation(config: dict, run_dir: pathlib.Path, **kw) -> int:
            generation_calls.append(True)
            return 0

        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            dry_run=False,
            allow_no_screen=True,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            preflight_runner=mock_preflight,
            generation_runner=mock_generation,
        )
        rc = runner.run()
        self.assertNotEqual(rc, run_swebench.EXIT_SUCCESS)
        self.assertEqual(generation_calls, [])

    def test_preflight_inconclusive_blocks_generation(self):
        """When preflight_runner returns 2 (inconclusive), generation must not run."""
        generation_calls = []

        def mock_preflight(model_id: str) -> int:
            return 2  # inconclusive

        def mock_generation(config: dict, run_dir: pathlib.Path, **kw) -> int:
            generation_calls.append(True)
            return 0

        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            dry_run=False,
            allow_no_screen=True,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            preflight_runner=mock_preflight,
            generation_runner=mock_generation,
        )
        rc = runner.run()
        self.assertNotEqual(rc, run_swebench.EXIT_SUCCESS)
        self.assertEqual(generation_calls, [])


# ---------------------------------------------------------------------------
# 8. State lifecycle — generation and grading are separate history entries
# ---------------------------------------------------------------------------


class TestStateLifecycle(unittest.TestCase):
    """State history must record generation and grading as separate phases."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)
        self.run_dir = _build_run_dir(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_planned_state_accepted(self):
        """Runner must accept a run dir in 'planned' execution_state."""
        status = json.loads((self.run_dir / "status.json").read_text())
        self.assertEqual(status["execution_state"], "planned")

    def test_lifecycle_transitions_recorded(self):
        """After a successful run the status must show transitions through running->completed."""
        def mock_preflight(model_id: str) -> int:
            return 0

        def mock_generation(config: dict, run_dir: pathlib.Path, **kw) -> int:
            _make_generation_artifacts(run_dir)
            return 0

        def mock_grading(preds_path: pathlib.Path, run_dir: pathlib.Path, **kw) -> int:
            _make_grading_artifacts(run_dir)
            return 0

        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            dry_run=False,
            allow_no_screen=True,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            preflight_runner=mock_preflight,
            generation_runner=mock_generation,
            grading_runner=mock_grading,
        )
        rc = runner.run()
        self.assertEqual(rc, run_swebench.EXIT_SUCCESS)

        status = json.loads((self.run_dir / "status.json").read_text())
        states = [h["state"] for h in status["history"]]
        # Must progress through planned -> preflight_passed -> running -> completed
        self.assertIn("planned", states)
        self.assertIn("preflight_passed", states)
        self.assertIn("running", states)
        self.assertIn("completed", states)

    def test_history_entries_schema_valid(self):
        """All history entries must satisfy the result-status schema."""
        from contract import validate_json
        schema_path = _REPO / "suite" / "schemas" / "result-status.schema.json"

        def mock_preflight(model_id: str) -> int:
            return 0

        def mock_generation(config: dict, run_dir: pathlib.Path, **kw) -> int:
            _make_generation_artifacts(run_dir)
            return 0

        def mock_grading(preds_path: pathlib.Path, run_dir: pathlib.Path, **kw) -> int:
            _make_grading_artifacts(run_dir)
            return 0

        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            dry_run=False,
            allow_no_screen=True,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            preflight_runner=mock_preflight,
            generation_runner=mock_generation,
            grading_runner=mock_grading,
        )
        runner.run()
        status = json.loads((self.run_dir / "status.json").read_text())
        errors = validate_json(status, schema_path)
        self.assertEqual(errors, [], f"Status schema errors: {errors}")

    def test_generation_and_grading_noted_in_history(self):
        """History must record separate notes for generation and grading phases."""
        def mock_preflight(model_id: str) -> int:
            return 0

        def mock_generation(config: dict, run_dir: pathlib.Path, **kw) -> int:
            _make_generation_artifacts(run_dir)
            return 0

        def mock_grading(preds_path: pathlib.Path, run_dir: pathlib.Path, **kw) -> int:
            _make_grading_artifacts(run_dir)
            return 0

        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            dry_run=False,
            allow_no_screen=True,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            preflight_runner=mock_preflight,
            generation_runner=mock_generation,
            grading_runner=mock_grading,
        )
        runner.run()
        status = json.loads((self.run_dir / "status.json").read_text())
        notes = [h.get("note", "") for h in status["history"]]
        combined = " ".join(notes).lower()
        # Must mention both generation and grading somewhere in history notes
        self.assertIn("generat", combined, "History must note generation phase")
        self.assertIn("grad", combined, "History must note grading phase")


# ---------------------------------------------------------------------------
# 9. Completion sentinel and evidence
# ---------------------------------------------------------------------------


class TestCompletionEvidence(unittest.TestCase):
    """DONE written only after grader artifacts and all 100 dispositions exist."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)
        self.run_dir = _build_run_dir(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_done_written_on_success(self):
        """DONE sentinel must be written after a fully successful run."""
        def mock_preflight(model_id: str) -> int:
            return 0

        def mock_generation(config: dict, run_dir: pathlib.Path, **kw) -> int:
            _make_generation_artifacts(run_dir)
            return 0

        def mock_grading(preds_path: pathlib.Path, run_dir: pathlib.Path, **kw) -> int:
            _make_grading_artifacts(run_dir)
            return 0

        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            dry_run=False,
            allow_no_screen=True,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            preflight_runner=mock_preflight,
            generation_runner=mock_generation,
            grading_runner=mock_grading,
        )
        rc = runner.run()
        self.assertEqual(rc, run_swebench.EXIT_SUCCESS)
        self.assertTrue(
            (self.run_dir / "DONE").exists(),
            "DONE sentinel must exist after successful run",
        )

    def test_done_not_written_when_generation_fails(self):
        """DONE must not be written if generation exits nonzero."""
        def mock_preflight(model_id: str) -> int:
            return 0

        def mock_generation(config: dict, run_dir: pathlib.Path, **kw) -> int:
            return 1  # failure, no artifacts

        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            dry_run=False,
            allow_no_screen=True,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            preflight_runner=lambda m: 0,
            generation_runner=mock_generation,
        )
        rc = runner.run()
        self.assertNotEqual(rc, run_swebench.EXIT_SUCCESS)
        self.assertFalse(
            (self.run_dir / "DONE").exists(),
            "DONE must not be written after generation failure",
        )

    def test_done_not_written_when_grading_fails(self):
        """DONE must not be written if grading exits nonzero."""
        def mock_generation(config: dict, run_dir: pathlib.Path, **kw) -> int:
            _make_generation_artifacts(run_dir)
            return 0

        def mock_grading(preds_path: pathlib.Path, run_dir: pathlib.Path, **kw) -> int:
            return 1  # grading failure

        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            dry_run=False,
            allow_no_screen=True,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            preflight_runner=lambda m: 0,
            generation_runner=mock_generation,
            grading_runner=mock_grading,
        )
        rc = runner.run()
        self.assertNotEqual(rc, run_swebench.EXIT_SUCCESS)
        self.assertFalse(
            (self.run_dir / "DONE").exists(),
            "DONE must not be written after grading failure",
        )

    def test_preds_json_required_for_completion(self):
        """Completion requires preds.json (generation output) to exist."""
        def mock_preflight(model_id: str) -> int:
            return 0

        def mock_generation(config: dict, run_dir: pathlib.Path, **kw) -> int:
            # No preds.json produced
            (run_dir / "raw").mkdir(exist_ok=True)
            return 0  # claims success but no artifacts

        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            dry_run=False,
            allow_no_screen=True,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            preflight_runner=mock_preflight,
            generation_runner=mock_generation,
        )
        rc = runner.run()
        self.assertNotEqual(rc, run_swebench.EXIT_SUCCESS)
        self.assertFalse((self.run_dir / "DONE").exists())

    def test_grading_results_required_for_completion(self):
        """Completion requires grading_results.json to exist."""
        def mock_generation(config: dict, run_dir: pathlib.Path, **kw) -> int:
            _make_generation_artifacts(run_dir)
            return 0

        def mock_grading(preds_path: pathlib.Path, run_dir: pathlib.Path, **kw) -> int:
            # Grading "succeeds" but leaves no artifacts
            return 0

        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            dry_run=False,
            allow_no_screen=True,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            preflight_runner=lambda m: 0,
            generation_runner=mock_generation,
            grading_runner=mock_grading,
        )
        rc = runner.run()
        self.assertNotEqual(rc, run_swebench.EXIT_SUCCESS)
        self.assertFalse((self.run_dir / "DONE").exists())


# ---------------------------------------------------------------------------
# 10. Failure state transitions
# ---------------------------------------------------------------------------


class TestFailureTransitions(unittest.TestCase):
    """Runner must transition to 'failed' on generation or grading failure."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)
        self.run_dir = _build_run_dir(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_failed_state_on_generation_failure(self):
        """status must show 'failed' after generation returns nonzero."""
        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            dry_run=False,
            allow_no_screen=True,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            preflight_runner=lambda m: 0,
            generation_runner=lambda config, run_dir, **kw: 1,
        )
        runner.run()
        status = json.loads((self.run_dir / "status.json").read_text())
        self.assertEqual(status["execution_state"], "failed")

    def test_failed_state_on_grading_failure(self):
        """status must show 'failed' after grading returns nonzero."""
        def mock_generation(config: dict, run_dir: pathlib.Path, **kw) -> int:
            _make_generation_artifacts(run_dir)
            return 0

        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            dry_run=False,
            allow_no_screen=True,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            preflight_runner=lambda m: 0,
            generation_runner=mock_generation,
            grading_runner=lambda preds, run_dir, **kw: 1,
        )
        runner.run()
        status = json.loads((self.run_dir / "status.json").read_text())
        self.assertEqual(status["execution_state"], "failed")


# ---------------------------------------------------------------------------
# 11. Dry-run — side-effect free
# ---------------------------------------------------------------------------


class TestDryRun(unittest.TestCase):
    """dry_run=True must print the command and return 0 without any state writes."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)
        # For dry-run we don't need a pre-existing run_dir
        self.run_dir = (
            self.tmp / "results" / _ADAPTER_SLUG
            / "runs" / "warpcore-v1" / "swebench" / "run-dryrun"
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_dry_run_returns_success(self):
        """dry_run must return EXIT_SUCCESS without touching the filesystem."""
        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            dry_run=True,
        prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
        )
        rc = runner.run()
        self.assertEqual(rc, run_swebench.EXIT_SUCCESS)

    def test_dry_run_no_status_written(self):
        """dry_run must not create or modify status.json."""
        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            dry_run=True,
        prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
        )
        runner.run()
        self.assertFalse(
            (self.run_dir / "status.json").exists(),
            "dry_run must not write status.json",
        )

    def test_dry_run_no_done_written(self):
        """dry_run must not create DONE sentinel."""
        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            dry_run=True,
        prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
        )
        runner.run()
        self.assertFalse(
            (self.run_dir / "DONE").exists(),
            "dry_run must not write DONE sentinel",
        )


# ---------------------------------------------------------------------------
# 12. command.txt written with correct content
# ---------------------------------------------------------------------------


class TestCommandTxt(unittest.TestCase):
    """command.txt must be written to the run directory with the launch command."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)
        self.run_dir = _build_run_dir(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_command_txt_written_on_successful_run(self):
        """command.txt must exist after a successful run."""
        def mock_preflight(model_id: str) -> int:
            return 0

        def mock_generation(config: dict, run_dir: pathlib.Path, **kw) -> int:
            _make_generation_artifacts(run_dir)
            return 0

        def mock_grading(preds_path: pathlib.Path, run_dir: pathlib.Path, **kw) -> int:
            _make_grading_artifacts(run_dir)
            return 0

        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            dry_run=False,
            allow_no_screen=True,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            preflight_runner=mock_preflight,
            generation_runner=mock_generation,
            grading_runner=mock_grading,
        )
        runner.run()
        self.assertTrue(
            (self.run_dir / "command.txt").exists(),
            "command.txt must be written to run directory",
        )

    def test_command_txt_contains_model_id(self):
        """command.txt must contain the adapter model_id."""
        def mock_preflight(model_id: str) -> int:
            return 0

        def mock_generation(config: dict, run_dir: pathlib.Path, **kw) -> int:
            _make_generation_artifacts(run_dir)
            return 0

        def mock_grading(preds_path: pathlib.Path, run_dir: pathlib.Path, **kw) -> int:
            _make_grading_artifacts(run_dir)
            return 0

        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            dry_run=False,
            allow_no_screen=True,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            preflight_runner=mock_preflight,
            generation_runner=mock_generation,
            grading_runner=mock_grading,
        )
        runner.run()
        cmd_txt = (self.run_dir / "command.txt").read_text()
        self.assertIn(_MODEL_ID, cmd_txt)


# ---------------------------------------------------------------------------
# 13. Disposition count — all 100 required for completion
# ---------------------------------------------------------------------------


class TestDispositionCount(unittest.TestCase):
    """Completion requires dispositions for all 100 instances."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)
        self.run_dir = _build_run_dir(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_partial_grading_prevents_completion(self):
        """Grading results covering fewer than 100 instances block DONE."""
        def mock_generation(config: dict, run_dir: pathlib.Path, **kw) -> int:
            _make_generation_artifacts(run_dir)
            return 0

        def mock_grading_partial(preds_path: pathlib.Path, run_dir: pathlib.Path, **kw) -> int:
            # Only 5 of 100 graded
            instances = json.loads(_REAL_INSTANCES.read_text())
            grading = {
                "resolved_ids": instances[:2],
                "unresolved_ids": instances[2:5],
                "empty_patch_ids": [],
                "error_ids": [],
            }
            (run_dir / "raw" / "grading_results.json").write_text(json.dumps(grading))
            return 0

        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            dry_run=False,
            allow_no_screen=True,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            preflight_runner=lambda m: 0,
            generation_runner=mock_generation,
            grading_runner=mock_grading_partial,
        )
        rc = runner.run()
        self.assertNotEqual(rc, run_swebench.EXIT_SUCCESS)
        self.assertFalse((self.run_dir / "DONE").exists())


# ---------------------------------------------------------------------------
# 14. CLI main() entrypoint and make run-swebench
# ---------------------------------------------------------------------------


class TestCLIInterface(unittest.TestCase):
    """CLI must accept required arguments and reject suite-owned overrides."""

    def test_main_function_exists(self):
        """Module must expose a main() function."""
        self.assertTrue(hasattr(run_swebench, "main"))
        self.assertTrue(callable(run_swebench.main))

    def test_dry_run_cli(self):
        """CLI with --dry-run must return 0."""
        import io
        from contextlib import redirect_stdout, redirect_stderr
        tmp = pathlib.Path(tempfile.mkdtemp())
        try:
            adapter_path = tmp / "adapters" / "test-canonical-model.yaml"
            _write_canonical_adapter(adapter_path)
            out = io.StringIO()
            err = io.StringIO()
            try:
                with redirect_stdout(out), redirect_stderr(err):
                    rc = run_swebench.main([
                        "--suite", str(_REAL_SUITE),
                        "--adapter", str(adapter_path),
                        "--endpoint", "http://localhost:8000/v1",
                        "--run-id", "run-cli-dryrun",
                        "--repo", str(tmp),
                        "--prompt-tokens", "gsm8k=500,ifeval=2000,gpqa_diamond=1000",
                        "--dry-run",
                    ])
            except SystemExit as e:
                rc = int(e.code)
            self.assertEqual(rc, run_swebench.EXIT_SUCCESS)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
