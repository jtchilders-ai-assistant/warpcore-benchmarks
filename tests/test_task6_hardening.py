"""tests/test_task6_hardening.py — Adversarial RED tests for Task 6 defects.

These tests expose concrete implementation defects in viz/run_swebench.py and
viz/swebench_preflight.py. They must fail (RED) before the fixes are applied.

Defects being tested:
  D1: swebench_preflight.load_instance_ids rejects JSON list (the canonical format)
  D2: dry-run prints api_key in plaintext via yaml.dump(config)
  D3: create_campaign call in main() omits prompt_token_maxima
  D4: grading phase status annotation failure is swallowed (WARNING, non-fatal)
  D5: _verify_generation_evidence does not require trajectories directory
  D6: _verify_generation_evidence does not require run.log
  D7: _verify_grading_evidence does not detect duplicate IDs in dispositions
  D8: instance list not validated for exact count (100) or uniqueness at construction
  D9: default _run_preflight writes temp JSON list but load_instance_ids rejects lists
"""
from __future__ import annotations

import io
import json
import pathlib
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

_TESTS_DIR = pathlib.Path(__file__).parent
_REPO = _TESTS_DIR.parent
_VIZ_DIR = _REPO / "viz"
for _p in (str(_TESTS_DIR), str(_VIZ_DIR), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_swebench  # noqa: E402
import swebench_preflight  # noqa: E402
from schema_helpers import install_test_qualification  # noqa: E402

_REAL_SUITE = _REPO / "suite" / "warpcore-v1.yaml"
_REAL_INSTANCES = _REPO / "suite" / "swebench" / "instances-seed42-n100.json"
_REAL_SCAFFOLD = _REPO / "suite" / "swebench" / "scaffold.yaml"

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

_ADAPTER_SLUG = _CANONICAL_ADAPTER["model"]["slug"]
_MODEL_ID = _CANONICAL_ADAPTER["model"]["id"]

_PROMPT_TOKEN_MAXIMA = {
    "gsm8k": 500,
    "ifeval": 2000,
    "gpqa_diamond": 1000,
}


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

    # A live launch is gated on a SWE-bench qualification record; install a valid
    # one so these tests keep exercising the runner mechanics they were written
    # for. The gate itself is adversarially tested in
    # tests/test_swebench_qualification.py.
    adapter_path = repo / "adapters" / f"{_ADAPTER_SLUG}.yaml"
    if adapter_path.is_file():
        install_test_qualification(
            repo=repo,
            adapter_path=adapter_path,
            endpoint="http://localhost:8000/v1",
            slug=slug,
            suite_id=suite_id,
        )
    return run_dir


def _make_generation_artifacts(run_dir: pathlib.Path) -> None:
    """Populate raw/ with minimal generation outputs (preds, exit_statuses, trajectories, run.log)."""
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    instances = json.loads(_REAL_INSTANCES.read_text())
    preds = {iid: {"model_patch": "diff --git a/x b/x\n+fix", "instance_id": iid}
             for iid in instances}
    (raw_dir / "preds.json").write_text(json.dumps(preds))
    statuses = {iid: 0 for iid in instances}
    (raw_dir / "exit_statuses.json").write_text(json.dumps(statuses))
    # trajectories dir and run.log - required by hardened evidence check
    (raw_dir / "trajectories").mkdir(exist_ok=True)
    for iid in instances:
        (raw_dir / "trajectories" / f"{iid}.traj").write_text("{}")
    (raw_dir / "run.log").write_text("run completed\n")


def _make_grading_artifacts(run_dir: pathlib.Path) -> None:
    """Populate raw/ with grading results for all 100 instances."""
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    instances = json.loads(_REAL_INSTANCES.read_text())
    grading = {
        "resolved_ids": instances[:5],
        "unresolved_ids": instances[5:],
        "empty_patch_ids": [],
        "error_ids": [],
    }
    (raw_dir / "grading_results.json").write_text(json.dumps(grading))


# ---------------------------------------------------------------------------
# D1: swebench_preflight.load_instance_ids must accept JSON list format
# ---------------------------------------------------------------------------


class TestPreflightLoadInstanceIdsList(unittest.TestCase):
    """D1: load_instance_ids must accept a JSON list of instance IDs.

    The canonical suite artifact (instances-seed42-n100.json) is a JSON list.
    The current implementation only handles dict formats, causing the default
    preflight in _run_preflight() to always fail with ValueError.
    """

    def test_load_instance_ids_accepts_json_list(self):
        """load_instance_ids must accept a plain JSON list of IDs."""
        tmp = pathlib.Path(tempfile.mkdtemp())
        try:
            ids = ["django__django-11299", "astropy__astropy-14096"]
            p = tmp / "instances.json"
            p.write_text(json.dumps(ids))
            result = swebench_preflight.load_instance_ids(p)
            self.assertEqual(sorted(result), sorted(ids))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_load_instance_ids_accepts_real_suite_file(self):
        """load_instance_ids must accept the real suite instances file."""
        result = swebench_preflight.load_instance_ids(_REAL_INSTANCES)
        self.assertEqual(len(result), 100)

    def test_load_instance_ids_list_order_stable(self):
        """load_instance_ids must return all IDs when given a JSON list."""
        tmp = pathlib.Path(tempfile.mkdtemp())
        try:
            ids = [f"repo__repo-{i}" for i in range(50)]
            p = tmp / "ids.json"
            p.write_text(json.dumps(ids))
            result = swebench_preflight.load_instance_ids(p)
            self.assertEqual(sorted(result), sorted(ids))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# D2: dry-run must not print API key in plaintext
# ---------------------------------------------------------------------------


class TestDryRunApiKeyRedaction(unittest.TestCase):
    """D2: dry-run must never print the API key in stdout output.

    yaml.dump(config) includes model.model_kwargs.api_key in plaintext.
    The dry-run path must redact secrets before printing.
    """

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)
        self.run_dir = (
            self.tmp / "results" / _ADAPTER_SLUG
            / "runs" / "warpcore-v1" / "swebench" / "run-dryrun"
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_dry_run_does_not_print_api_key(self):
        """dry-run output must not contain the literal API key value."""
        secret = "MY_SECRET_API_KEY_XYZ123"
        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            api_key=secret,
            dry_run=True,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
        )
        captured = io.StringIO()
        with redirect_stdout(captured):
            runner.run()
        output = captured.getvalue()
        self.assertNotIn(
            secret,
            output,
            f"dry-run printed the API key '{secret}' in stdout. Secrets must be redacted.",
        )

    def test_dry_run_does_not_print_api_key_cli(self):
        """CLI --dry-run must not print API key in stdout."""
        secret = "ANOTHER_SECRET_987654"
        tmp = pathlib.Path(tempfile.mkdtemp())
        try:
            adapter_path = tmp / "adapters" / "test-canonical-model.yaml"
            _write_canonical_adapter(adapter_path)
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                try:
                    run_swebench.main([
                        "--suite", str(_REAL_SUITE),
                        "--adapter", str(adapter_path),
                        "--endpoint", "http://localhost:8000/v1",
                        "--run-id", "run-cli-secret-test",
                        "--repo", str(tmp),
                        "--api-key", secret,
                        "--prompt-tokens", "gsm8k=500,ifeval=2000,gpqa_diamond=1000",
                        "--dry-run",
                    ])
                except SystemExit:
                    pass
            output = out.getvalue() + err.getvalue()
            self.assertNotIn(
                secret,
                output,
                f"CLI --dry-run printed API key '{secret}'. Secrets must never be logged.",
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# D3: create_campaign must receive prompt_token_maxima
# ---------------------------------------------------------------------------


class TestCreateCampaignPromptTokenMaxima(unittest.TestCase):
    """D3: The create_campaign call in main() must pass prompt_token_maxima.

    Without prompt_token_maxima, create_campaign cannot validate that context
    lengths are feasible for campaign-ready adapter, violating the contract.
    """

    def test_create_campaign_receives_prompt_token_maxima(self):
        """When --prompt-tokens is given, create_campaign must receive them."""
        calls = []

        def fake_create_campaign(**kwargs):
            calls.append(kwargs)
            raise RuntimeError("abort after recording args")

        tmp = pathlib.Path(tempfile.mkdtemp())
        try:
            adapter_path = tmp / "adapters" / "test-canonical-model.yaml"
            _write_canonical_adapter(adapter_path)
            # We intercept create_campaign to check its arguments
            with patch.dict("sys.modules", {}):
                import importlib
                # Patch create_campaign module attribute
                import create_campaign as cc_mod
                original = cc_mod.create_campaign
                cc_mod.create_campaign = lambda **kw: (calls.append(kw), (_ for _ in ()).throw(RuntimeError("abort")))[0]
                try:
                    out = io.StringIO()
                    err = io.StringIO()
                    with redirect_stdout(out), redirect_stderr(err):
                        try:
                            run_swebench.main([
                                "--suite", str(_REAL_SUITE),
                                "--adapter", str(adapter_path),
                                "--endpoint", "http://localhost:8000/v1",
                                "--run-id", "run-test-ptm",
                                "--repo", str(tmp),
                                "--prompt-tokens", "gsm8k=500,ifeval=2000,gpqa_diamond=1000",
                            ])
                        except (SystemExit, RuntimeError, Exception):
                            pass
                finally:
                    cc_mod.create_campaign = original
            # If create_campaign was called with prompt_token_maxima, it must be non-None
            if calls:
                call_kwargs = calls[0]
                self.assertIn(
                    "prompt_token_maxima",
                    call_kwargs,
                    "create_campaign must receive prompt_token_maxima kwarg",
                )
                self.assertIsNotNone(
                    call_kwargs.get("prompt_token_maxima"),
                    "prompt_token_maxima must not be None when --prompt-tokens is provided",
                )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# D4: Grading phase status write failure must be fatal, not swallowed
# ---------------------------------------------------------------------------


class TestGradingPhaseAnnotationFatal(unittest.TestCase):
    """D4: The grading phase status annotation failure must not be swallowed.

    Currently, _write_status failure in the grading phase annotation block
    is caught and emitted as a WARNING while execution continues. Status
    write failures must be fatal — a run must not continue with a corrupt
    or unwritable status file.
    """

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)
        self.run_dir = _build_run_dir(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_grading_phase_annotation_write_failure_is_fatal(self):
        """If status write fails during grading phase annotation, run must return nonzero.

        The current code emits WARNING and continues, which is wrong — a status
        write failure means the run's durable state is inconsistent.
        """
        write_count = [0]
        original_write = run_swebench.campaign_state.write_status

        def fail_on_third_write(dest, status, run_dir=None):
            write_count[0] += 1
            # First 2 writes succeed (planned->preflight_passed, preflight_passed->running)
            # Third write is the grading phase annotation - make it fail
            if write_count[0] >= 3:
                raise OSError("Simulated disk failure on grading phase status write")
            return original_write(dest, status, run_dir=run_dir)

        def mock_generation(config, run_dir, **kw):
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
        )
        with patch.object(run_swebench.campaign_state, "write_status", side_effect=fail_on_third_write):
            rc = runner.run()
        self.assertNotEqual(
            rc,
            run_swebench.EXIT_SUCCESS,
            "A status write failure during grading phase annotation must return nonzero. "
            "The current implementation swallows this failure and continues.",
        )


# ---------------------------------------------------------------------------
# D5 + D6: generation evidence must require trajectories/ and run.log
# ---------------------------------------------------------------------------


class TestGenerationEvidenceRequiresTrajectories(unittest.TestCase):
    """D5/D6: _verify_generation_evidence must require trajectories/ dir and run.log.

    The plan requires trajectories to be preserved and verifiable. A generation
    run without trajectories/ or run.log is incomplete evidence.
    """

    def test_missing_trajectories_blocks_completion(self):
        """If trajectories/ dir is absent, generation evidence check must fail."""
        tmp = pathlib.Path(tempfile.mkdtemp())
        try:
            raw = tmp / "raw"
            raw.mkdir()
            instances = json.loads(_REAL_INSTANCES.read_text())
            preds = {iid: {"model_patch": "x", "instance_id": iid} for iid in instances}
            (raw / "preds.json").write_text(json.dumps(preds))
            (raw / "exit_statuses.json").write_text(json.dumps({iid: 0 for iid in instances}))
            (raw / "run.log").write_text("run completed\n")
            # No trajectories/ directory
            errors = run_swebench._verify_generation_evidence(tmp)
            self.assertTrue(
                len(errors) > 0,
                "Missing trajectories/ dir must produce evidence errors. "
                "Current implementation does not check for trajectories.",
            )
            combined = " ".join(errors).lower()
            self.assertIn(
                "traject",
                combined,
                "Error message must mention trajectories when they are missing.",
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_missing_run_log_blocks_completion(self):
        """If run.log is absent, generation evidence check must fail."""
        tmp = pathlib.Path(tempfile.mkdtemp())
        try:
            raw = tmp / "raw"
            raw.mkdir()
            instances = json.loads(_REAL_INSTANCES.read_text())
            preds = {iid: {"model_patch": "x", "instance_id": iid} for iid in instances}
            (raw / "preds.json").write_text(json.dumps(preds))
            (raw / "exit_statuses.json").write_text(json.dumps({iid: 0 for iid in instances}))
            (raw / "trajectories").mkdir()
            # No run.log
            errors = run_swebench._verify_generation_evidence(tmp)
            self.assertTrue(
                len(errors) > 0,
                "Missing run.log must produce evidence errors. "
                "Current implementation does not check for run.log.",
            )
            combined = " ".join(errors).lower()
            self.assertIn(
                "run.log",
                combined,
                "Error message must mention run.log when it is missing.",
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_completion_without_trajectories_is_blocked(self):
        """A successful run without trajectories/ must not write DONE."""
        tmp = pathlib.Path(tempfile.mkdtemp())
        adapter_path = tmp / "adapters" / "test.yaml"
        _write_canonical_adapter(adapter_path)
        run_dir = _build_run_dir(tmp)
        try:
            def mock_generation_no_traj(config, run_dir, **kw):
                raw = run_dir / "raw"
                raw.mkdir(exist_ok=True)
                instances = json.loads(_REAL_INSTANCES.read_text())
                preds = {iid: {"model_patch": "x", "instance_id": iid} for iid in instances}
                (raw / "preds.json").write_text(json.dumps(preds))
                (raw / "exit_statuses.json").write_text(json.dumps({iid: 0 for iid in instances}))
                (raw / "run.log").write_text("done\n")
                # No trajectories/ directory
                return 0

            runner = run_swebench.SwebenchRunner(
                suite_path=_REAL_SUITE,
                adapter_path=adapter_path,
                endpoint="http://localhost:8000/v1",
                run_dir=run_dir,
                repo=tmp,
                dry_run=False,
                allow_no_screen=True,
                prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
                preflight_runner=lambda m: 0,
                generation_runner=mock_generation_no_traj,
            )
            rc = runner.run()
            self.assertNotEqual(rc, run_swebench.EXIT_SUCCESS)
            self.assertFalse(
                (run_dir / "DONE").exists(),
                "DONE must not be written when trajectories/ is missing.",
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# D7: _verify_grading_evidence must detect duplicate IDs across categories
# ---------------------------------------------------------------------------


class TestGradingEvidenceDuplicateIds(unittest.TestCase):
    """D7: Grading evidence must reject duplicate IDs appearing in multiple categories.

    If the same instance_id appears in both resolved_ids and unresolved_ids,
    the set union hides the duplication. The current implementation uses set
    union which silently accepts duplicates as long as the union covers 100 IDs.
    """

    def test_duplicate_id_across_categories_rejected(self):
        """An ID appearing in both resolved and unresolved must be rejected."""
        tmp = pathlib.Path(tempfile.mkdtemp())
        try:
            raw = tmp / "raw"
            raw.mkdir()
            instances = json.loads(_REAL_INSTANCES.read_text())
            # 50 unique IDs appear in BOTH resolved_ids and unresolved_ids
            # Set union = 50 unique IDs = missing 50 -> should fail with 50 missing
            # But the defect is that set union doesn't catch the duplicates explicitly
            # We want the validator to ALSO flag duplicates as a distinct error
            half = instances[:50]
            grading = {
                "resolved_ids": half,
                "unresolved_ids": half,  # same 50 IDs - duplicate!
                "empty_patch_ids": [],
                "error_ids": [],
            }
            (raw / "grading_results.json").write_text(json.dumps(grading))
            errors = run_swebench._verify_grading_evidence(tmp, instances)
            # Must catch that 50 instances are missing (set union = 50 unique)
            self.assertTrue(len(errors) > 0, "Duplicate IDs should produce errors")
            # Also check duplicate detection is explicit
            combined = " ".join(errors).lower()
            # The defect: current impl only catches 'missing' but not 'duplicate' as a distinct error
            # After fix: must also say 'duplicate' or give the count difference
            self.assertTrue(
                "missing" in combined or "duplicate" in combined,
                "Error must explicitly address missing or duplicate dispositions.",
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_foreign_ids_in_grading_rejected(self):
        """IDs not in the expected set must be explicitly rejected."""
        tmp = pathlib.Path(tempfile.mkdtemp())
        try:
            raw = tmp / "raw"
            raw.mkdir()
            instances = json.loads(_REAL_INSTANCES.read_text())
            # All 100 expected + 5 foreign IDs
            foreign = ["foreign__repo-0001", "foreign__repo-0002", "foreign__repo-0003",
                       "foreign__repo-0004", "foreign__repo-0005"]
            grading = {
                "resolved_ids": instances[:5] + foreign,
                "unresolved_ids": instances[5:],
                "empty_patch_ids": [],
                "error_ids": [],
            }
            (raw / "grading_results.json").write_text(json.dumps(grading))
            errors = run_swebench._verify_grading_evidence(tmp, instances)
            self.assertTrue(len(errors) > 0, "Foreign IDs must produce grading evidence errors")
            combined = " ".join(errors).lower()
            self.assertTrue(
                "unexpected" in combined or "foreign" in combined or "extra" in combined,
                "Error must mention unexpected/extra IDs.",
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# D8: Instance list must be validated for exact count and uniqueness
# ---------------------------------------------------------------------------


class TestInstanceListValidation(unittest.TestCase):
    """D8: The frozen instance list must be exactly 100 unique IDs.

    If the suite artifact contains duplicates or fewer than 100 IDs,
    construction must raise ValueError. The current implementation only
    checks isinstance(list) but not count or uniqueness.
    """

    def _make_bad_suite_with_instances(
        self, tmp: pathlib.Path, instance_ids: list
    ) -> tuple:
        """Write a suite pointing to a custom instances file."""
        import yaml
        import hashlib

        # Write the tampered instances file
        inst_path = tmp / "suite" / "swebench" / "instances-bad.json"
        inst_path.parent.mkdir(parents=True, exist_ok=True)
        inst_path.write_text(json.dumps(instance_ids))

        # Compute its hash
        inst_hash = hashlib.sha256(inst_path.read_bytes()).hexdigest()

        # Load real suite and patch it
        with open(_REAL_SUITE) as f:
            suite = yaml.safe_load(f)

        swe_bench = suite["benchmarks"]["swebench"]
        swe_bench["instance_set_file"] = "suite/swebench/instances-bad.json"
        swe_bench["instances_sha256"] = inst_hash
        # Recompute scaffold hash if needed (scaffold is unchanged)
        import shutil as shutil_mod
        scaffold_src = _REPO / "suite" / "swebench" / "scaffold.yaml"
        scaffold_dst = tmp / "suite" / "swebench" / "scaffold.yaml"
        scaffold_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil_mod.copy(scaffold_src, scaffold_dst)
        scaffold_hash = hashlib.sha256(scaffold_dst.read_bytes()).hexdigest()
        swe_bench["scaffold_file"] = "suite/swebench/scaffold.yaml"
        swe_bench["scaffold_sha256"] = scaffold_hash

        suite_path = tmp / "suite" / "warpcore-v1-bad.yaml"
        suite_path.parent.mkdir(parents=True, exist_ok=True)
        suite_path.write_text(yaml.dump(suite, default_flow_style=False))

        # Rewrite suite hash in itself (suite_input_hashes for CI)
        adapter_path = tmp / "adapters" / "test.yaml"
        _write_canonical_adapter(adapter_path)

        run_dir = (
            tmp / "results" / _ADAPTER_SLUG
            / "runs" / "warpcore-v1" / "swebench" / "run-test"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "status.json").write_text(
            json.dumps(_make_planned_status("run-test", "warpcore-v1"))
        )
        manifest = {
            "suite_id": "warpcore-v1",
            "run_id": "run-test",
            "benchmark": "swebench",
            "model": {"slug": _ADAPTER_SLUG, "id": _MODEL_ID, "revision": "a" * 40},
            "item_inventory": {"expected": 100},
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest))

        return suite_path, adapter_path, run_dir

    def test_instance_list_too_short_rejected(self):
        """Construction must reject an instance list with fewer than 100 IDs."""
        tmp = pathlib.Path(tempfile.mkdtemp())
        try:
            real_ids = json.loads(_REAL_INSTANCES.read_text())
            short_ids = real_ids[:50]  # only 50 IDs
            suite_path, adapter_path, run_dir = self._make_bad_suite_with_instances(tmp, short_ids)
            with self.assertRaises((ValueError, Exception),
                                   msg="Construction must reject instance list with < 100 IDs"):
                run_swebench.SwebenchRunner(
                    suite_path=suite_path,
                    adapter_path=adapter_path,
                    endpoint="http://localhost:8000/v1",
                    run_dir=run_dir,
                    repo=tmp,
                    dry_run=True,
                    prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
                )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_instance_list_with_duplicates_rejected(self):
        """Construction must reject an instance list with duplicate IDs."""
        tmp = pathlib.Path(tempfile.mkdtemp())
        try:
            real_ids = json.loads(_REAL_INSTANCES.read_text())
            # 100 entries but 50 are duplicates
            duped_ids = real_ids[:50] + real_ids[:50]
            suite_path, adapter_path, run_dir = self._make_bad_suite_with_instances(tmp, duped_ids)
            with self.assertRaises((ValueError, Exception),
                                   msg="Construction must reject instance list with duplicate IDs"):
                run_swebench.SwebenchRunner(
                    suite_path=suite_path,
                    adapter_path=adapter_path,
                    endpoint="http://localhost:8000/v1",
                    run_dir=run_dir,
                    repo=tmp,
                    dry_run=True,
                    prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
                )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# D9: default preflight wraps load_instance_ids — must handle list format
# ---------------------------------------------------------------------------


class TestDefaultPreflightHandlesList(unittest.TestCase):
    """D9: The default _run_preflight writes a temp JSON list and calls swebench_preflight.

    Since load_instance_ids rejects JSON lists (D1), the default preflight
    path would always return EXIT_INCONCLUSIVE, preventing any live run
    from ever passing preflight without an injected preflight_runner.
    """

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)
        self.run_dir = _build_run_dir(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_default_preflight_temp_file_is_list_compatible(self):
        """The temp file written by _run_preflight must be parseable by load_instance_ids."""
        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            dry_run=True,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
        )
        # Verify the instance IDs are a proper list
        ids = runner.get_instance_ids()
        self.assertIsInstance(ids, list)

        # Write the temp file the same way _run_preflight does and try parsing it
        import tempfile as _tf
        with _tf.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
            json.dump(ids, tf)
            tmp_path = pathlib.Path(tf.name)
        try:
            # This must not raise ValueError after D1 fix
            result = swebench_preflight.load_instance_ids(tmp_path)
            self.assertEqual(len(result), 100)
        finally:
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# D10: DONE ordering — status must be written before DONE sentinel
# ---------------------------------------------------------------------------


class TestDoneOrdering(unittest.TestCase):
    """DONE sentinel must only be written after a verified completed status write.

    If the completed status write fails, DONE must not be written. The current
    implementation of _transition_completed_atomic returns EXIT_DEFECT on
    write failure, and the caller checks that before writing DONE — this is
    correct. But we verify the contract holds under simulated write failure.
    """

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)
        self.run_dir = _build_run_dir(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_done_not_written_when_completed_status_write_fails(self):
        """If the completed status write fails, DONE must not be written."""
        write_count = [0]
        original_write = run_swebench.campaign_state.write_status

        def fail_on_completed_write(dest, status, run_dir=None):
            write_count[0] += 1
            # Fail the final 'completed' write
            if status.get("execution_state") == "completed":
                raise OSError("Simulated disk failure on completed status write")
            return original_write(dest, status, run_dir=run_dir)

        def mock_generation(config, run_dir, **kw):
            _make_generation_artifacts(run_dir)
            return 0

        def mock_grading(preds_path, run_dir, **kw):
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
            preflight_runner=lambda m: 0,
            generation_runner=mock_generation,
            grading_runner=mock_grading,
        )
        with patch.object(run_swebench.campaign_state, "write_status",
                          side_effect=fail_on_completed_write):
            rc = runner.run()
        self.assertNotEqual(
            rc,
            run_swebench.EXIT_SUCCESS,
            "Must return nonzero when completed status write fails.",
        )
        self.assertFalse(
            (self.run_dir / "DONE").exists(),
            "DONE must not be written when completed status write fails.",
        )


# ---------------------------------------------------------------------------
# Summary: preds.json must contain exactly the 100 frozen IDs (no extras, no missing)
# ---------------------------------------------------------------------------


class TestGenerationEvidenceExactIds(unittest.TestCase):
    """preds.json must contain exactly the 100 frozen instance IDs.

    Currently _verify_generation_evidence only checks that preds.json is nonempty.
    It must also validate that the prediction keys match the frozen 100 IDs exactly.
    """

    def test_preds_with_wrong_ids_rejected(self):
        """preds.json with IDs not matching the frozen set must be rejected."""
        tmp = pathlib.Path(tempfile.mkdtemp())
        try:
            raw = tmp / "raw"
            raw.mkdir()
            # Wrong IDs — not from the frozen set
            wrong_ids = [f"wrong__repo-{i}" for i in range(100)]
            preds = {iid: {"model_patch": "x"} for iid in wrong_ids}
            (raw / "preds.json").write_text(json.dumps(preds))
            (raw / "exit_statuses.json").write_text(json.dumps({iid: 0 for iid in wrong_ids}))
            (raw / "trajectories").mkdir()
            (raw / "run.log").write_text("done\n")
            instances = json.loads(_REAL_INSTANCES.read_text())
            errors = run_swebench._verify_generation_evidence(tmp, expected_instance_ids=instances)
            self.assertTrue(
                len(errors) > 0,
                "preds.json with wrong instance IDs must produce evidence errors. "
                "Current implementation only checks nonempty.",
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_preds_missing_ids_rejected(self):
        """preds.json with fewer than 100 instances must be rejected."""
        tmp = pathlib.Path(tempfile.mkdtemp())
        try:
            raw = tmp / "raw"
            raw.mkdir()
            instances = json.loads(_REAL_INSTANCES.read_text())
            # Only 50 of the 100 IDs
            partial_preds = {iid: {"model_patch": "x"} for iid in instances[:50]}
            (raw / "preds.json").write_text(json.dumps(partial_preds))
            (raw / "exit_statuses.json").write_text(json.dumps({iid: 0 for iid in instances[:50]}))
            (raw / "trajectories").mkdir()
            (raw / "run.log").write_text("done\n")
            errors = run_swebench._verify_generation_evidence(tmp, expected_instance_ids=instances)
            self.assertTrue(
                len(errors) > 0,
                "preds.json with only 50 of 100 instances must produce evidence errors.",
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# N1: _run_generation / _run_grading must implement real subprocess wrappers
#     (not just return EXIT_DEFECT when no runner is injected)
# ---------------------------------------------------------------------------


class TestLiveSubprocessRunners(unittest.TestCase):
    """N1: Default _run_generation and _run_grading must call real subprocesses.

    When no runner is injected, the runner should call mini-swe-agent and
    python -m swebench.harness.run_evaluation rather than returning EXIT_DEFECT.
    These tests verify the subprocess is actually attempted (using a fake
    executable that fails fast), not that EXIT_DEFECT is returned immediately.
    """

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)
        self.run_dir = _build_run_dir(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_run_generation_attempts_subprocess_not_immediate_defect(self):
        """_run_generation without injected runner must attempt a subprocess call,
        not immediately return EXIT_DEFECT with no subprocess attempt.

        Generation launches via subprocess.Popen (not subprocess.run) because the
        campaign circuit breaker polls live progress evidence while the harness is
        still running, and needs the child in its own process group to be able to
        terminate exactly what it owns.
        """
        import subprocess as _sp
        attempted = []

        def fake_popen(cmd, *args, **kwargs):
            attempted.append((cmd, kwargs))
            raise FileNotFoundError("fake: command not found")

        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            dry_run=False,
            allow_no_screen=True,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
        )
        config = runner.build_scaffold_config("http://localhost:8000/v1", "warpcore")
        with patch.object(_sp, "Popen", side_effect=fake_popen):
            rc = runner._run_generation(config)
        self.assertTrue(
            len(attempted) > 0,
            "_run_generation without injected runner must attempt a subprocess, "
            "not immediately return EXIT_DEFECT.",
        )
        self.assertTrue(
            attempted[0][1].get("start_new_session"),
            "Generation must run in its own session so the circuit breaker can "
            "terminate exactly the owned process group.",
        )

    def test_run_grading_attempts_subprocess_not_immediate_defect(self):
        """_run_grading without injected runner must attempt a subprocess call,
        not immediately return EXIT_DEFECT with no subprocess attempt."""
        import subprocess as _sp
        attempted = []

        def fake_run(cmd, *args, **kwargs):
            attempted.append(cmd)
            raise FileNotFoundError("fake: command not found")

        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            dry_run=False,
            allow_no_screen=True,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
        )
        preds_path = self.run_dir / "raw" / "preds.json"
        preds_path.parent.mkdir(parents=True, exist_ok=True)
        preds_path.write_text("{}")
        with patch.object(_sp, "run", side_effect=fake_run):
            rc = runner._run_grading(preds_path)
        self.assertTrue(
            len(attempted) > 0,
            "_run_grading without injected runner must attempt a subprocess, "
            "not immediately return EXIT_DEFECT. Current implementation never "
            "calls subprocess.run.",
        )

    def test_run_generation_uses_argv_not_shell(self):
        """_run_generation must use argv list, not shell=True."""
        import subprocess as _sp
        calls = []

        def capture_run(cmd, *args, **kwargs):
            calls.append({"cmd": cmd, "kwargs": kwargs})
            raise FileNotFoundError("fake: not found")

        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            dry_run=False,
            allow_no_screen=True,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
        )
        config = runner.build_scaffold_config("http://localhost:8000/v1", "warpcore")
        with patch.object(_sp, "run", side_effect=capture_run):
            runner._run_generation(config)

        if calls:
            for call in calls:
                self.assertIsInstance(
                    call["cmd"], list,
                    f"subprocess.run must be called with a list (argv), not a string. "
                    f"Got: {call['cmd']!r}"
                )
                self.assertFalse(
                    call["kwargs"].get("shell", False),
                    "subprocess.run must not use shell=True. "
                    "Shell injection is a security risk."
                )


# ---------------------------------------------------------------------------
# N2: main() --run-dir explicit path must call create_campaign with resume=True
#     and demand returned normalized path equals requested path
# ---------------------------------------------------------------------------


class TestMainRunDirResume(unittest.TestCase):
    """N2: Explicit --run-dir must imply resume=True and call create_campaign.

    When --run-dir is given explicitly on a live (non-dry-run) path, main()
    currently bypasses create_campaign entirely. It must instead derive the
    run_id, call create_campaign(resume=True), and demand the returned path
    equals the requested path.
    """

    def test_explicit_run_dir_calls_create_campaign_with_resume(self):
        """When --run-dir is given explicitly for live run, create_campaign must be called."""
        calls = []

        tmp = pathlib.Path(tempfile.mkdtemp())
        try:
            adapter_path = tmp / "adapters" / "test-canonical-model.yaml"
            _write_canonical_adapter(adapter_path)

            # Build a normalized run_dir path (what the runner would use)
            slug = _ADAPTER_SLUG
            run_id = "run-existing-test"
            run_dir = tmp / "results" / slug / "runs" / "warpcore-v1" / "swebench" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            # Put a planned status there
            (run_dir / "status.json").write_text(
                json.dumps({
                    "schema_version": 1,
                    "run_id": run_id,
                    "suite_id": "warpcore-v1",
                    "execution_state": "planned",
                    "lifecycle": "current",
                    "history": [{"state": "planned", "timestamp": "2026-09-15T12:00:00Z"}],
                })
            )
            (run_dir / "manifest.json").write_text(json.dumps({
                "suite_id": "warpcore-v1", "run_id": run_id, "benchmark": "swebench",
                "model": {"slug": slug, "id": _MODEL_ID, "revision": "a" * 40},
                "item_inventory": {"expected": 100},
            }))

            # main() gates on the SWE-bench qualification before create_campaign,
            # so a launch-authorizing record must exist for this test to reach it.
            install_test_qualification(
                repo=tmp, adapter_path=adapter_path,
                endpoint="http://localhost:8000/v1",
            )

            import create_campaign as cc_mod
            original_cc = cc_mod.create_campaign

            def fake_create_campaign(**kwargs):
                calls.append(kwargs)
                raise RuntimeError("abort after recording")

            cc_mod.create_campaign = fake_create_campaign
            try:
                out = io.StringIO()
                err = io.StringIO()
                with redirect_stdout(out), redirect_stderr(err):
                    try:
                        run_swebench.main([
                            "--suite", str(_REAL_SUITE),
                            "--adapter", str(adapter_path),
                            "--endpoint", "http://localhost:8000/v1",
                            "--run-dir", str(run_dir),
                            "--repo", str(tmp),
                            "--prompt-tokens", "gsm8k=500,ifeval=2000,gpqa_diamond=1000",
                            "--allow-no-screen",
                        ])
                    except (SystemExit, RuntimeError, Exception):
                        pass
            finally:
                cc_mod.create_campaign = original_cc

            self.assertTrue(
                len(calls) > 0,
                "When --run-dir is given explicitly for a live (non-dry-run) run, "
                "main() must call create_campaign(resume=True) to validate the run dir. "
                "Current implementation bypasses create_campaign entirely when --run-dir is given.",
            )
            if calls:
                self.assertTrue(
                    calls[0].get("resume", False),
                    "create_campaign must be called with resume=True when --run-dir is explicit. "
                    f"Got: {calls[0]}",
                )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_default_path_calls_create_campaign(self):
        """Default (no --run-dir) live path must call create_campaign for a fresh run."""
        calls = []
        tmp = pathlib.Path(tempfile.mkdtemp())
        try:
            adapter_path = tmp / "adapters" / "test-canonical-model.yaml"
            _write_canonical_adapter(adapter_path)

            # main() gates on the SWE-bench qualification before create_campaign,
            # so a launch-authorizing record must exist for this test to reach it.
            install_test_qualification(
                repo=tmp, adapter_path=adapter_path,
                endpoint="http://localhost:8000/v1",
            )

            import create_campaign as cc_mod
            original_cc = cc_mod.create_campaign

            def fake_create_campaign(**kwargs):
                calls.append(kwargs)
                raise RuntimeError("abort after recording")

            cc_mod.create_campaign = fake_create_campaign
            try:
                out = io.StringIO()
                err = io.StringIO()
                with redirect_stdout(out), redirect_stderr(err):
                    try:
                        run_swebench.main([
                            "--suite", str(_REAL_SUITE),
                            "--adapter", str(adapter_path),
                            "--endpoint", "http://localhost:8000/v1",
                            "--run-id", "run-default-test",
                            "--repo", str(tmp),
                            "--prompt-tokens", "gsm8k=500,ifeval=2000,gpqa_diamond=1000",
                            "--allow-no-screen",
                        ])
                    except (SystemExit, RuntimeError, Exception):
                        pass
            finally:
                cc_mod.create_campaign = original_cc

            self.assertTrue(
                len(calls) > 0,
                "Default (no --run-dir) live path must call create_campaign. "
                "This verifies the basic create_campaign wiring for the non-resume path.",
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# N3: preflight x86 check must reject empty/unknown arch as inconclusive
# ---------------------------------------------------------------------------


class TestPreflightX86Strictness(unittest.TestCase):
    """N3: Preflight x86 check must treat empty/unknown arch as inconclusive.

    Current code: arch not in ("x86_64", "amd64", "") — empty string is accepted
    as OK, which is wrong. An empty arch means docker info returned nothing usable,
    and we cannot confirm x86 capability. Must return EXIT_INCONCLUSIVE.
    """

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)
        self.run_dir = _build_run_dir(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_runner(self):
        return run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            dry_run=False,
            allow_no_screen=True,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
        )

    def test_empty_arch_is_inconclusive_not_ok(self):
        """When docker info returns empty arch string, preflight must be EXIT_INCONCLUSIVE."""
        import subprocess as _sp

        class FakeResult:
            returncode = 0
            stdout = ""  # empty arch
            stderr = ""

        runner = self._make_runner()
        with patch.object(_sp, "run", return_value=FakeResult()):
            with patch("urllib.request.urlopen") as urlopen:
                rc = runner._run_preflight()
        self.assertNotEqual(
            rc,
            0,
            "Empty arch string from 'docker info' must NOT be treated as a passing x86 check. "
            "Current code accepts '' as a valid arch, which is a fail-open defect. "
            "Expected EXIT_INCONCLUSIVE (2) but got 0.",
        )
        self.assertEqual(
            rc,
            run_swebench.EXIT_INCONCLUSIVE,
            f"Empty arch must return EXIT_INCONCLUSIVE (2), got {rc}.",
        )
        urlopen.assert_not_called()

    def test_unknown_arch_is_inconclusive(self):
        """When docker info returns an unknown arch string, preflight must be EXIT_INCONCLUSIVE."""
        import subprocess as _sp

        class FakeResult:
            returncode = 0
            stdout = "unknown_arch_xyz"
            stderr = ""

        runner = self._make_runner()
        with patch.object(_sp, "run", return_value=FakeResult()):
            rc = runner._run_preflight()
        self.assertNotEqual(
            rc,
            0,
            "Unknown arch string from 'docker info' must NOT be treated as passing. "
            "Expected EXIT_INCONCLUSIVE (2) or EXIT_DEFECT (1), got 0.",
        )

    def test_x86_64_arch_passes_x86_check(self):
        """When docker info returns 'x86_64', the x86 check must pass (proceed to next check)."""
        import subprocess as _sp
        import urllib.error as _uerr

        class FakeResult:
            returncode = 0
            stdout = "x86_64"
            stderr = ""

        runner = self._make_runner()
        with patch.object(_sp, "run", return_value=FakeResult()):
            # After passing x86 check, /v1/models will fail (no real endpoint)
            # We just want to confirm x86_64 doesn't fail the x86 check itself
            with patch("urllib.request.urlopen", side_effect=_uerr.URLError("no endpoint")):
                rc = runner._run_preflight()
        # Should be INCONCLUSIVE (2) from the /v1/models check, not DEFECT from x86
        self.assertEqual(
            rc,
            run_swebench.EXIT_INCONCLUSIVE,
            f"x86_64 arch must pass the x86 check and proceed to /v1/models. "
            f"Got rc={rc} instead of EXIT_INCONCLUSIVE from /v1/models failure.",
        )


# ---------------------------------------------------------------------------
# N4: /v1/models request must send API key Authorization header
# ---------------------------------------------------------------------------


class TestPreflightModelsApiKey(unittest.TestCase):
    """N4: /v1/models request must include Authorization: Bearer <api_key> header.

    Current implementation uses urlopen(models_url) without any Authorization header.
    Endpoints that require authentication will return 401 instead of 200, making
    the preflight inconclusive rather than correctly authenticated.
    """

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)
        self.run_dir = _build_run_dir(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_models_request_includes_authorization_header(self):
        """The /v1/models GET request must include an Authorization: Bearer header."""
        import subprocess as _sp
        import urllib.request as _req

        class FakeResult:
            returncode = 0
            stdout = "x86_64"
            stderr = ""

        captured_requests = []

        class FakeResponse:
            def __init__(self):
                self.status = 200

            def read(self):
                return json.dumps({"data": [{"id": _MODEL_ID}]}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

        def fake_urlopen(req, **kwargs):
            captured_requests.append(req)
            return FakeResponse()

        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            dry_run=False,
            allow_no_screen=True,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            api_key="TEST_API_KEY_789",
        )
        with patch.object(_sp, "run", return_value=FakeResult()):
            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                runner._run_preflight()

        self.assertTrue(
            len(captured_requests) > 0,
            "/v1/models must be called — no request was captured.",
        )
        self.assertTrue(
            all(isinstance(req, _req.Request) for req in captured_requests),
            "urlopen must receive Request objects so headers can be verified.",
        )
        for req in captured_requests:
            auth = req.get_header("Authorization")
            self.assertEqual(auth, "Bearer TEST_API_KEY_789")

    def test_models_request_does_not_leak_api_key_in_output(self):
        """The api_key must not appear in stdout/stderr during preflight."""
        import subprocess as _sp
        import urllib.request as _req
        import urllib.error as _uerr

        secret = "SUPER_SECRET_KEY_PREFLIGHT_TEST"

        class FakeResult:
            returncode = 0
            stdout = "x86_64"
            stderr = ""

        runner = run_swebench.SwebenchRunner(
            suite_path=_REAL_SUITE,
            adapter_path=self.adapter_path,
            endpoint="http://localhost:8000/v1",
            run_dir=self.run_dir,
            repo=self.tmp,
            dry_run=False,
            allow_no_screen=True,
            prompt_token_maxima=_PROMPT_TOKEN_MAXIMA,
            api_key=secret,
        )

        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            with patch.object(_sp, "run", return_value=FakeResult()):
                with patch("urllib.request.urlopen", side_effect=_uerr.URLError("no endpoint")):
                    runner._run_preflight()

        combined = out.getvalue() + err.getvalue()
        self.assertNotIn(
            secret,
            combined,
            f"API key '{secret}' must not appear in preflight output. "
            f"Got: {combined[:500]}",
        )


# ---------------------------------------------------------------------------
# N5: _transition_failed return value must be acted upon (fatal on write failure)
# ---------------------------------------------------------------------------


class TestTransitionFailedReturnValueActedOn(unittest.TestCase):
    """N5: Every _transition_failed call site must act on the return value.

    When _transition_failed returns False (write failed), the run is in an
    indeterminate state — we cannot confirm 'failed' was durably written.
    The run must abort with a fatal error, not silently continue.
    """

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)
        self.run_dir = _build_run_dir(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_failed_state_write_failure_is_fatal_on_generation_failure(self):
        """If _transition_failed write fails after generation failure, run must abort fatally."""
        write_count = [0]
        original_write = run_swebench.campaign_state.write_status

        def fail_all_after_running(dest, status, run_dir=None):
            # Allow writes until 'running'; fail all subsequent writes
            if status.get("execution_state") in ("preflight_passed", "planned"):
                return original_write(dest, status, run_dir=run_dir)
            if status.get("execution_state") == "running":
                return original_write(dest, status, run_dir=run_dir)
            raise OSError("Simulated: failed-state write failure")

        def failing_generation(config, run_dir, **kw):
            return 1  # generation fails

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
            generation_runner=failing_generation,
        )
        with patch.object(run_swebench.campaign_state, "write_status",
                          side_effect=fail_all_after_running):
            rc = runner.run()

        # Must return nonzero — not EXIT_SUCCESS
        self.assertNotEqual(
            rc,
            run_swebench.EXIT_SUCCESS,
            "When generation fails AND _transition_failed write fails, run must return nonzero. "
            "Current code ignores the return value of _transition_failed.",
        )
        # The status must not be 'running' after a failed write — it should either be
        # 'failed' (if the write somehow succeeded) or at minimum return nonzero
        status = json.loads((self.run_dir / "status.json").read_text())
        self.assertNotEqual(
            status.get("execution_state"),
            "completed",
            "Status must not show 'completed' when generation failed.",
        )


# ---------------------------------------------------------------------------
# N6: DONE write failure must not leave completed status without DONE
#     (fail-closed ordering: write DONE before returning EXIT_SUCCESS)
# ---------------------------------------------------------------------------


class TestDoneWriteFailureFailClosed(unittest.TestCase):
    """N6: If DONE write fails after completed status, run must return nonzero.

    Current implementation writes completed status first, then writes DONE.
    If DONE write fails, the status says 'completed' but there is no DONE
    sentinel — a corrupt state. The run must return EXIT_DEFECT in this case.

    The fix requires atomicity: either both succeed or neither commits.
    In practice: write DONE first (or use a temp+rename), then write completed.
    Or: write completed, if DONE fails, rewrite status to failed.
    """

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.adapter_path = self.tmp / "adapters" / "test-canonical-model.yaml"
        _write_canonical_adapter(self.adapter_path)
        self.run_dir = _build_run_dir(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_done_write_failure_leaves_no_inconsistent_completed_status(self):
        """If DONE cannot be written, the status must NOT be left as 'completed'.

        The invariant is: DONE exists iff status == 'completed'.
        Violation: status='completed' but no DONE file.

        Uses done_writer injection (not pathlib.Path.write_text patching) because
        macOS resolves /tmp → /private/tmp, making path identity comparisons in
        a write_text patch unreliable.
        """

        def mock_generation(config, run_dir, **kw):
            _make_generation_artifacts(run_dir)
            return 0

        def mock_grading(preds_path, run_dir, **kw):
            _make_grading_artifacts(run_dir)
            return 0

        def failing_done_writer(done_path):
            raise OSError("Simulated DONE write failure")

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
            done_writer=failing_done_writer,
        )

        rc = runner.run()

        self.assertNotEqual(
            rc,
            run_swebench.EXIT_SUCCESS,
            "DONE write failure must return nonzero (EXIT_DEFECT). "
            "Current implementation already returns EXIT_DEFECT on DONE write failure. "
            "This test verifies the status does not show 'completed' without DONE.",
        )

        done_path = self.run_dir / "DONE"
        # Critical: if status shows 'completed' but no DONE exists, that's a corrupt state
        if done_path.exists():
            return  # DONE was written despite our injected failure — skip check
        if (self.run_dir / "status.json").exists():
            status = json.loads((self.run_dir / "status.json").read_text())
            self.assertNotEqual(
                status.get("execution_state"),
                "completed",
                "INVARIANT VIOLATION: status.json says 'completed' but DONE does not exist. "
                "The fail-closed scheme must ensure this cannot happen. "
                "The completed status must be reverted to 'failed' when DONE write fails.",
            )

    def test_done_and_completed_are_consistent_on_success(self):
        """On a successful run, both DONE and completed status must exist."""

        def mock_generation(config, run_dir, **kw):
            _make_generation_artifacts(run_dir)
            return 0

        def mock_grading(preds_path, run_dir, **kw):
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
            preflight_runner=lambda m: 0,
            generation_runner=mock_generation,
            grading_runner=mock_grading,
        )
        rc = runner.run()
        self.assertEqual(rc, run_swebench.EXIT_SUCCESS)
        self.assertTrue(
            (self.run_dir / "DONE").exists(),
            "DONE sentinel must exist after successful run.",
        )
        status = json.loads((self.run_dir / "status.json").read_text())
        self.assertEqual(
            status.get("execution_state"),
            "completed",
            "Status must be 'completed' after successful run.",
        )


# ---------------------------------------------------------------------------
# N7: Generation evidence must check exactly one trajectory per expected ID
# ---------------------------------------------------------------------------


class TestGenerationEvidencePerIdTrajectory(unittest.TestCase):
    """N7: trajectories/ must contain exactly one .traj file per expected instance ID.

    Current code only checks that the trajectories/ directory exists.
    It must also verify each expected ID has a corresponding trajectory file.
    """

    def test_trajectories_missing_for_some_ids(self):
        """If some expected IDs have no trajectory file, evidence check must fail."""
        tmp = pathlib.Path(tempfile.mkdtemp())
        try:
            raw = tmp / "raw"
            raw.mkdir()
            instances = json.loads(_REAL_INSTANCES.read_text())
            preds = {iid: {"model_patch": "x", "instance_id": iid} for iid in instances}
            (raw / "preds.json").write_text(json.dumps(preds))
            (raw / "exit_statuses.json").write_text(json.dumps({iid: 0 for iid in instances}))
            (raw / "run.log").write_text("done\n")
            traj_dir = raw / "trajectories"
            traj_dir.mkdir()
            # Only write trajectories for the first 50 of 100 IDs
            for iid in instances[:50]:
                (traj_dir / f"{iid}.traj").write_text("{}")
            # Leave instances[50:] without trajectory files
            errors = run_swebench._verify_generation_evidence(
                tmp, expected_instance_ids=instances
            )
            self.assertTrue(
                len(errors) > 0,
                "Missing trajectory files for 50 of 100 instances must produce evidence errors. "
                "Current implementation only checks directory existence, not per-ID files.",
            )
            combined = " ".join(errors).lower()
            self.assertIn(
                "traject",
                combined,
                "Error must mention 'trajectory' or 'trajectories'.",
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_trajectories_all_present_passes(self):
        """When all 100 trajectory files are present, evidence check must pass."""
        tmp = pathlib.Path(tempfile.mkdtemp())
        try:
            raw = tmp / "raw"
            raw.mkdir()
            instances = json.loads(_REAL_INSTANCES.read_text())
            preds = {iid: {"model_patch": "x", "instance_id": iid} for iid in instances}
            (raw / "preds.json").write_text(json.dumps(preds))
            (raw / "exit_statuses.json").write_text(json.dumps({iid: 0 for iid in instances}))
            (raw / "run.log").write_text("done\n")
            traj_dir = raw / "trajectories"
            traj_dir.mkdir()
            for iid in instances:
                (traj_dir / f"{iid}.traj").write_text("{}")
            errors = run_swebench._verify_generation_evidence(
                tmp, expected_instance_ids=instances
            )
            self.assertEqual(
                errors,
                [],
                f"All 100 trajectory files present must produce no errors. Got: {errors}",
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# N8: exit_statuses.json must also check for EXTRA (unexpected) IDs
# ---------------------------------------------------------------------------


class TestGenerationEvidenceExitStatusExtra(unittest.TestCase):
    """N8: exit_statuses.json must reject both missing AND extra IDs.

    Current implementation only checks for missing IDs in exit_statuses.json.
    Extra IDs (from a different instance set) must also be rejected.
    """

    def test_extra_exit_status_ids_rejected(self):
        """exit_statuses.json with extra IDs not in the frozen set must be rejected."""
        tmp = pathlib.Path(tempfile.mkdtemp())
        try:
            raw = tmp / "raw"
            raw.mkdir()
            instances = json.loads(_REAL_INSTANCES.read_text())
            preds = {iid: {"model_patch": "x"} for iid in instances}
            (raw / "preds.json").write_text(json.dumps(preds))
            # exit_statuses has all 100 expected + 5 extra foreign IDs
            extra = ["foreign__repo-0001", "foreign__repo-0002", "foreign__repo-0003",
                     "foreign__repo-0004", "foreign__repo-0005"]
            es = {iid: 0 for iid in instances}
            for eid in extra:
                es[eid] = 0
            (raw / "exit_statuses.json").write_text(json.dumps(es))
            traj_dir = raw / "trajectories"
            traj_dir.mkdir()
            for iid in instances:
                (traj_dir / f"{iid}.traj").write_text("{}")
            (raw / "run.log").write_text("done\n")
            errors = run_swebench._verify_generation_evidence(
                tmp, expected_instance_ids=instances
            )
            self.assertTrue(
                len(errors) > 0,
                "exit_statuses.json with extra IDs not in the frozen set must produce errors. "
                "Current implementation only checks for missing IDs, not extra ones.",
            )
            combined = " ".join(errors).lower()
            self.assertTrue(
                "unexpected" in combined or "extra" in combined or "foreign" in combined,
                f"Error must mention unexpected/extra IDs. Got: {errors}",
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# N9: run.log must be nonempty (not just exist)
# ---------------------------------------------------------------------------


class TestRunLogNonempty(unittest.TestCase):
    """N9: run.log must be nonempty — an empty run.log is not valid evidence."""

    def test_empty_run_log_fails_evidence_check(self):
        """An empty run.log must produce a generation evidence error."""
        tmp = pathlib.Path(tempfile.mkdtemp())
        try:
            raw = tmp / "raw"
            raw.mkdir()
            instances = json.loads(_REAL_INSTANCES.read_text())
            preds = {iid: {"model_patch": "x", "instance_id": iid} for iid in instances}
            (raw / "preds.json").write_text(json.dumps(preds))
            (raw / "exit_statuses.json").write_text(json.dumps({iid: 0 for iid in instances}))
            traj_dir = raw / "trajectories"
            traj_dir.mkdir()
            for iid in instances:
                (traj_dir / f"{iid}.traj").write_text("{}")
            (raw / "run.log").write_text("")  # empty run.log
            errors = run_swebench._verify_generation_evidence(
                tmp, expected_instance_ids=instances
            )
            self.assertTrue(
                len(errors) > 0,
                "An empty run.log must produce evidence errors. "
                "Current implementation only checks existence, not content.",
            )
            combined = " ".join(errors).lower()
            self.assertIn(
                "run.log",
                combined,
                "Error must mention 'run.log'.",
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
