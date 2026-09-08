"""Tests for viz/quality_preflight.py — quality-run preflight and timeout feasibility gate.

TDD: these tests are written BEFORE the implementation. They must fail (RED) until
quality_preflight.py is implemented, then pass (GREEN).

Design constraints:
- All arithmetic tests are offline (no network, no GPU).
- Subprocess/network calls are dependency-injected or mocked at the boundary.
- The gate fails CLOSED unless ALL three checks pass:
    (a) live endpoint/model preflight succeeds (via preflight_serving behavior)
    (b) output budget check succeeds (via check_output_budget behavior)
    (c) timeout arithmetic closes (worst-case per-request timeout fits client timeout)
"""
from __future__ import annotations

import sys
import types
import unittest
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, call

# Add repo root so we can import viz modules
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from viz.quality_preflight import (
    TimeoutArithmetic,
    compute_timeout_arithmetic,
    QualityPreflightGate,
    ExitCode,
)


# ---------------------------------------------------------------------------
# 1. Arithmetic: basic happy path
# ---------------------------------------------------------------------------

class TestTimeoutArithmeticBasic(unittest.TestCase):

    def test_per_request_throughput_aggregate_division(self):
        """aggregate throughput divided by concurrency gives per-request tok/s."""
        result = compute_timeout_arithmetic(
            max_gen_toks=4096,
            aggregate_tok_s=160.0,  # 160 tok/s total across 10 workers
            concurrency=10,
            client_timeout_s=3600,
            safety_factor=2.0,
        )
        # per_request = 160/10 = 16 tok/s
        self.assertAlmostEqual(result.per_request_tok_s, 16.0)

    def test_worst_case_time_formula(self):
        """worst_case_s = max_gen_toks / per_request_tok_s."""
        result = compute_timeout_arithmetic(
            max_gen_toks=4096,
            aggregate_tok_s=160.0,
            concurrency=10,
            client_timeout_s=3600,
            safety_factor=2.0,
        )
        # worst_case = 4096 / 16 = 256 s
        self.assertAlmostEqual(result.worst_case_s, 256.0)

    def test_required_timeout_formula(self):
        """required_timeout = safety_factor * worst_case_s."""
        result = compute_timeout_arithmetic(
            max_gen_toks=4096,
            aggregate_tok_s=160.0,
            concurrency=10,
            client_timeout_s=3600,
            safety_factor=2.0,
        )
        # required = 2.0 * 256 = 512 s
        self.assertAlmostEqual(result.required_timeout_s, 512.0)

    def test_feasible_when_client_timeout_exceeds_required(self):
        """timeout is feasible when client_timeout >= required_timeout."""
        result = compute_timeout_arithmetic(
            max_gen_toks=4096,
            aggregate_tok_s=160.0,
            concurrency=10,
            client_timeout_s=3600,  # 3600 > 512
            safety_factor=2.0,
        )
        self.assertTrue(result.feasible)

    def test_infeasible_when_client_timeout_too_small(self):
        """timeout is infeasible when client_timeout < required_timeout."""
        result = compute_timeout_arithmetic(
            max_gen_toks=32768,
            aggregate_tok_s=16.0,   # 16 tok/s total / 16 concurrency = 1 tok/s per req
            concurrency=16,
            client_timeout_s=3600,  # 3600 < 2*32768 = 65536
            safety_factor=2.0,
        )
        self.assertFalse(result.feasible)

    def test_safety_factor_respected(self):
        """safety_factor=1.0 gives exactly worst_case as required; 3.0 triples it."""
        r1 = compute_timeout_arithmetic(1000, 100.0, 1, 9999, safety_factor=1.0)
        r3 = compute_timeout_arithmetic(1000, 100.0, 1, 9999, safety_factor=3.0)
        self.assertAlmostEqual(r1.required_timeout_s, 10.0)
        self.assertAlmostEqual(r3.required_timeout_s, 30.0)

    def test_default_safety_factor_is_2(self):
        """Default safety factor should be 2.0."""
        r = compute_timeout_arithmetic(1000, 100.0, 1, 9999)
        # per_req = 100/1 = 100 tok/s; worst = 1000/100 = 10s; required = 2*10 = 20s
        self.assertAlmostEqual(r.required_timeout_s, 20.0)

    def test_result_is_named_tuple_or_dataclass(self):
        """Result must expose per_request_tok_s, worst_case_s, required_timeout_s, feasible."""
        r = compute_timeout_arithmetic(1000, 100.0, 1, 9999)
        self.assertTrue(hasattr(r, "per_request_tok_s"))
        self.assertTrue(hasattr(r, "worst_case_s"))
        self.assertTrue(hasattr(r, "required_timeout_s"))
        self.assertTrue(hasattr(r, "feasible"))

    def test_margin_field(self):
        """Result should expose a margin (client_timeout - required_timeout)."""
        r = compute_timeout_arithmetic(1000, 100.0, 1, 9999, safety_factor=2.0)
        # required = 20; margin = 9999 - 20 = 9979
        self.assertAlmostEqual(r.margin_s, 9999 - 20.0)

    def test_boundary_exactly_equal_is_feasible(self):
        """When client_timeout == required_timeout the gate is feasible (edge: >= not >)."""
        # per_req=100, worst=10, required=20; set client to exactly 20
        r = compute_timeout_arithmetic(1000, 100.0, 1, 20.0, safety_factor=2.0)
        self.assertTrue(r.feasible)


# ---------------------------------------------------------------------------
# 2. Arithmetic: guard rails – refuse nonpositive values
# ---------------------------------------------------------------------------

class TestTimeoutArithmeticGuardRails(unittest.TestCase):

    def _call(self, **overrides):
        defaults = dict(max_gen_toks=1000, aggregate_tok_s=100.0,
                        concurrency=1, client_timeout_s=9999, safety_factor=2.0)
        defaults.update(overrides)
        return compute_timeout_arithmetic(**defaults)

    def test_refuses_zero_max_gen_toks(self):
        with self.assertRaises(ValueError):
            self._call(max_gen_toks=0)

    def test_refuses_negative_max_gen_toks(self):
        with self.assertRaises(ValueError):
            self._call(max_gen_toks=-1)

    def test_refuses_zero_aggregate_tok_s(self):
        with self.assertRaises(ValueError):
            self._call(aggregate_tok_s=0.0)

    def test_refuses_negative_aggregate_tok_s(self):
        with self.assertRaises(ValueError):
            self._call(aggregate_tok_s=-5.0)

    def test_refuses_zero_concurrency(self):
        with self.assertRaises(ValueError):
            self._call(concurrency=0)

    def test_refuses_negative_concurrency(self):
        with self.assertRaises(ValueError):
            self._call(concurrency=-2)

    def test_refuses_zero_client_timeout(self):
        with self.assertRaises(ValueError):
            self._call(client_timeout_s=0)

    def test_refuses_negative_client_timeout(self):
        with self.assertRaises(ValueError):
            self._call(client_timeout_s=-1)

    def test_refuses_zero_safety_factor(self):
        with self.assertRaises(ValueError):
            self._call(safety_factor=0.0)

    def test_refuses_negative_safety_factor(self):
        with self.assertRaises(ValueError):
            self._call(safety_factor=-1.0)


# ---------------------------------------------------------------------------
# 3. TimeoutArithmetic class (alternate entry point)
# ---------------------------------------------------------------------------

class TestTimeoutArithmeticClass(unittest.TestCase):
    """TimeoutArithmetic may be a class or a namespace; test it wraps the same logic."""

    def test_can_be_constructed_with_same_args(self):
        ta = TimeoutArithmetic(
            max_gen_toks=4096,
            aggregate_tok_s=160.0,
            concurrency=10,
            client_timeout_s=3600,
            safety_factor=2.0,
        )
        self.assertAlmostEqual(ta.per_request_tok_s, 16.0)
        self.assertAlmostEqual(ta.required_timeout_s, 512.0)
        self.assertTrue(ta.feasible)

    def test_repr_contains_key_fields(self):
        ta = TimeoutArithmetic(1000, 100.0, 1, 9999)
        r = repr(ta)
        # repr should mention whether it's feasible and required timeout
        self.assertIn("feasible", r.lower())


# ---------------------------------------------------------------------------
# 4. ExitCode constants
# ---------------------------------------------------------------------------

class TestExitCodes(unittest.TestCase):

    def test_ok_is_zero(self):
        self.assertEqual(ExitCode.OK, 0)

    def test_defect_is_one(self):
        self.assertEqual(ExitCode.DEFECT, 1)

    def test_inconclusive_is_two(self):
        self.assertEqual(ExitCode.INCONCLUSIVE, 2)


# ---------------------------------------------------------------------------
# 5. QualityPreflightGate — command sequencing and fail-closed behavior
# ---------------------------------------------------------------------------

class TestQualityPreflightGate(unittest.TestCase):
    """Gate integrates three checks; uses injected subprocess runners so no network."""

    def _make_gate(self, *, serving_exit=0, budget_exit=0, timeout_feasible=True):
        """Build a gate with fully mocked sub-checkers.

        The injected mocks accept (model, endpoint, int) -> int, so tests can
        inspect call_args and verify the exact model ID was forwarded.
        """
        gate = QualityPreflightGate(
            endpoint="http://fake:8000/v1",
            model="fake/model",
            max_gen_toks=4096,
            aggregate_tok_s=160.0,
            concurrency=10,
            client_timeout_s=3600,
            safety_factor=2.0,
        )
        # Inject mock runners — signature: (model, endpoint, int) -> int
        gate._run_serving_preflight = MagicMock(return_value=serving_exit)
        gate._run_budget_check = MagicMock(return_value=budget_exit)
        # Override arithmetic result
        arith = compute_timeout_arithmetic(4096, 160.0, 10, 3600, 2.0)
        if not timeout_feasible:
            # Manufacture an infeasible arithmetic result
            arith = compute_timeout_arithmetic(4096, 1.0, 1, 1, 2.0)
        gate._arithmetic = arith
        return gate

    def test_all_pass_returns_ok(self):
        gate = self._make_gate(serving_exit=0, budget_exit=0, timeout_feasible=True)
        self.assertEqual(gate.run(), ExitCode.OK)

    def test_serving_defect_returns_defect(self):
        """If serving preflight exits 1, the whole gate must exit DEFECT."""
        gate = self._make_gate(serving_exit=1, budget_exit=0, timeout_feasible=True)
        self.assertEqual(gate.run(), ExitCode.DEFECT)

    def test_serving_inconclusive_returns_inconclusive(self):
        """If serving preflight exits 2, gate must exit INCONCLUSIVE (never OK)."""
        gate = self._make_gate(serving_exit=2, budget_exit=0, timeout_feasible=True)
        self.assertEqual(gate.run(), ExitCode.INCONCLUSIVE)

    def test_budget_defect_returns_defect(self):
        gate = self._make_gate(serving_exit=0, budget_exit=1, timeout_feasible=True)
        self.assertEqual(gate.run(), ExitCode.DEFECT)

    def test_budget_inconclusive_returns_inconclusive(self):
        gate = self._make_gate(serving_exit=0, budget_exit=2, timeout_feasible=True)
        self.assertEqual(gate.run(), ExitCode.INCONCLUSIVE)

    def test_timeout_infeasible_returns_defect(self):
        gate = self._make_gate(serving_exit=0, budget_exit=0, timeout_feasible=False)
        self.assertEqual(gate.run(), ExitCode.DEFECT)

    def test_fails_closed_on_serving_error(self):
        """If serving preflight raises an exception, gate must NOT return OK."""
        gate = self._make_gate(serving_exit=0, budget_exit=0, timeout_feasible=True)
        gate._run_serving_preflight = MagicMock(side_effect=RuntimeError("network down"))
        result = gate.run()
        self.assertNotEqual(result, ExitCode.OK)

    def test_fails_closed_on_budget_error(self):
        """If budget check raises an exception, gate must NOT return OK."""
        gate = self._make_gate(serving_exit=0, budget_exit=0, timeout_feasible=True)
        gate._run_budget_check = MagicMock(side_effect=RuntimeError("network down"))
        result = gate.run()
        self.assertNotEqual(result, ExitCode.OK)

    def test_serving_check_called_with_model(self):
        """Gate must pass the exact model ID to serving preflight."""
        gate = self._make_gate()
        gate.run()
        gate._run_serving_preflight.assert_called_once()
        # The model argument should appear somewhere in the call
        args, kwargs = gate._run_serving_preflight.call_args
        all_args = list(args) + list(kwargs.values())
        self.assertTrue(
            any("fake/model" in str(a) for a in all_args),
            f"model not passed to serving preflight; call_args={gate._run_serving_preflight.call_args}"
        )

    def test_budget_check_called_with_model(self):
        """Gate must pass the exact model ID to budget check."""
        gate = self._make_gate()
        gate.run()
        gate._run_budget_check.assert_called_once()
        args, kwargs = gate._run_budget_check.call_args
        all_args = list(args) + list(kwargs.values())
        self.assertTrue(
            any("fake/model" in str(a) for a in all_args),
            f"model not passed to budget check; call_args={gate._run_budget_check.call_args}"
        )

    def test_runner_receives_script_specific_endpoint_forms(self):
        """Serving receives /v1; budget receives the host root expected by its CLI."""
        runner = MagicMock(side_effect=[0, 0])
        gate = QualityPreflightGate(
            endpoint="http://fake:8000/v1/",
            model="fake/model",
            max_gen_toks=4096,
            aggregate_tok_s=160.0,
            concurrency=10,
            client_timeout_s=3600,
            runner=runner,
        )
        self.assertEqual(gate.run(verbose=False), ExitCode.OK)
        serving_cmd, budget_cmd = [c.args[0] for c in runner.call_args_list]
        self.assertEqual(serving_cmd[serving_cmd.index("--endpoint") + 1], "http://fake:8000/v1/")
        self.assertEqual(budget_cmd[budget_cmd.index("--base-url") + 1], "http://fake:8000")

    def test_short_circuits_after_serving_failure(self):
        """If serving preflight fails, budget check should NOT be called."""
        gate = self._make_gate(serving_exit=1)
        gate.run()
        gate._run_budget_check.assert_not_called()


# ---------------------------------------------------------------------------
# 6. CLI interface
# ---------------------------------------------------------------------------

class TestCLI(unittest.TestCase):
    """Test that the CLI entry point exists and respects --help / --self-test."""

    def test_module_has_main(self):
        from viz.quality_preflight import main
        self.assertTrue(callable(main))

    def test_self_test_mode_offline(self):
        """--self-test must not touch the network; runs arithmetic self-tests."""
        from viz.quality_preflight import main
        # Should not raise; exit code must be 0 for self-test
        try:
            rc = main(["--self-test"])
        except SystemExit as e:
            rc = e.code
        self.assertEqual(rc, 0)

    def test_missing_required_args_exits_nonzero(self):
        """Calling with incomplete args should exit nonzero (not 0)."""
        from viz.quality_preflight import main
        try:
            rc = main([])
        except SystemExit as e:
            rc = e.code
        except Exception:
            rc = 1
        self.assertNotEqual(rc, 0)

    def test_arithmetic_check_only_flag(self):
        """--check-timeout-only should run arithmetic offline and exit based on feasibility."""
        from viz.quality_preflight import main
        # Feasible: 1000 toks / (100 tok/s agg / 1 conc) = 10s worst case; *2 = 20s; client=9999
        try:
            rc = main([
                "--check-timeout-only",
                "--max-gen-toks", "1000",
                "--aggregate-tok-s", "100",
                "--concurrency", "1",
                "--client-timeout", "9999",
            ])
        except SystemExit as e:
            rc = e.code
        self.assertEqual(rc, 0)

    def test_arithmetic_check_only_infeasible(self):
        """--check-timeout-only exits 1 when timeout is infeasible."""
        from viz.quality_preflight import main
        # 32768 toks / (1 tok/s) = 32768s worst; *2 = 65536s; client=3600 -> INFEASIBLE
        try:
            rc = main([
                "--check-timeout-only",
                "--max-gen-toks", "32768",
                "--aggregate-tok-s", "1",
                "--concurrency", "1",
                "--client-timeout", "3600",
            ])
        except SystemExit as e:
            rc = e.code
        self.assertEqual(rc, 1)

    def test_make_arithmetic_only_mode(self):
        """Make target adds --check-timeout-only when endpoint and model are absent."""
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                "make", "quality-preflight", "MODE=arithmetic",
                "MAX_GEN_TOKS=1000", "AGGREGATE_TOK_S=100",
                "CONCURRENCY=1", "CLIENT_TIMEOUT=9999",
            ],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("[timeout-arithmetic]", result.stdout)
        self.assertNotIn("QUALITY PREFLIGHT GATE", result.stdout)

    def test_make_rejects_partial_live_configuration(self):
        """MODEL without ENDPOINT must not silently downgrade to arithmetic-only."""
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                "make", "quality-preflight", "MODEL=fake/model",
                "MAX_GEN_TOKS=1000", "AGGREGATE_TOK_S=100",
                "CONCURRENCY=1", "CLIENT_TIMEOUT=9999",
            ],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ENDPOINT and MODEL must be set together", result.stdout + result.stderr)


# ---------------------------------------------------------------------------
# 7. Regression: the GPQA 13-hour abort scenario
# ---------------------------------------------------------------------------

class TestRegressionGPQAAbort(unittest.TestCase):
    """Encodes the exact numbers from the 13-hour GPQA abort (AGENTS.md §Size the time budget).

    At c=16 and ~4 tok/s per request, max_gen_toks~=32768 (long reasoning),
    client_timeout=3600 (1 h). This MUST be caught as infeasible.
    """

    def test_gpqa_abort_scenario_is_infeasible(self):
        # 4 tok/s per request * 16 concurrency = 64 tok/s aggregate
        result = compute_timeout_arithmetic(
            max_gen_toks=32768,
            aggregate_tok_s=64.0,
            concurrency=16,
            client_timeout_s=3600,
            safety_factor=2.0,
        )
        # per_req = 64/16 = 4 tok/s; worst = 32768/4 = 8192s; required = 2*8192 = 16384s
        self.assertAlmostEqual(result.per_request_tok_s, 4.0, places=1)
        self.assertAlmostEqual(result.worst_case_s, 8192.0, places=0)
        self.assertAlmostEqual(result.required_timeout_s, 16384.0, places=0)
        self.assertFalse(result.feasible)

    def test_gpqa_abort_gate_blocks_launch(self):
        gate = QualityPreflightGate(
            endpoint="http://fake:8000/v1",
            model="fake/model",
            max_gen_toks=32768,
            aggregate_tok_s=64.0,
            concurrency=16,
            client_timeout_s=3600,
            safety_factor=2.0,
        )
        gate._run_serving_preflight = MagicMock(return_value=0)
        gate._run_budget_check = MagicMock(return_value=0)
        # Even if endpoint is healthy, arithmetic must block launch
        result = gate.run()
        self.assertEqual(result, ExitCode.DEFECT)


if __name__ == "__main__":
    unittest.main()
