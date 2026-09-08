#!/usr/bin/env python3
"""Mandatory quality-run preflight and timeout feasibility gate.

Combines three independent checks before a quality benchmark run is allowed to launch:

  (a) SERVING PREFLIGHT  — live endpoint/model probe (via preflight_serving.py logic)
  (b) OUTPUT BUDGET      — server-side output-token ceiling check (via check_output_budget.py)
  (c) TIMEOUT ARITHMETIC — worst-case per-request time vs client timeout

ALL THREE must pass. Fail-closed: any uncertainty returns INCONCLUSIVE (exit 2), not OK.

USAGE (CLI)
-----------
    # Full preflight (checks a, b, c):
    python3 viz/quality_preflight.py \\
        --endpoint http://csi370295.alcf.anl.gov:8000/v1 \\
        --model my/model \\
        --max-gen-toks 32768 \\
        --aggregate-tok-s 64 \\
        --concurrency 16 \\
        --client-timeout 14400

    # Arithmetic only (offline, no network):
    python3 viz/quality_preflight.py \\
        --check-timeout-only \\
        --max-gen-toks 32768 --aggregate-tok-s 64 --concurrency 16 --client-timeout 14400

    # Fixture self-test (CI, no GPU, no network):
    python3 viz/quality_preflight.py --self-test

EXIT CODES
----------
    0  All checks passed. Safe to launch.
    1  DEFECT — a check found a problem. Do NOT launch.
    2  INCONCLUSIVE — could not determine safety. Do NOT launch.

TIMEOUT ARITHMETIC
------------------
Inputs:
    max_gen_toks          — maximum output tokens per request
    aggregate_tok_s       — measured total throughput across all concurrent workers (tok/s)
    concurrency           — number of parallel lm-eval workers
    client_timeout_s      — the --timeout value passed to lm-eval
    safety_factor         — multiplier on worst-case time; default 2.0

Derivation (when aggregate throughput supplied):
    per_request_tok_s  = aggregate_tok_s / concurrency
    worst_case_s       = max_gen_toks / per_request_tok_s
    required_timeout_s = safety_factor * worst_case_s
    feasible           = client_timeout_s >= required_timeout_s

All inputs must be strictly positive. Zero or negative values raise ValueError.

INTEGRATION WITH EXISTING SCRIPTS
----------------------------------
This gate delegates to:
    viz/preflight_serving.py    --endpoint URL --model NAME --max-tokens N
    viz/check_output_budget.py  --model NAME --base-url URL --require-budget N

Both are called as subprocesses with dependency-injected runners for testability.
"""
from __future__ import annotations

import argparse
import dataclasses
import enum
import os
import subprocess
import sys
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------

class ExitCode(int, enum.Enum):
    OK           = 0
    DEFECT       = 1
    INCONCLUSIVE = 2


# ---------------------------------------------------------------------------
# Timeout arithmetic
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TimeoutArithmetic:
    """Compute and hold worst-case timeout feasibility for a benchmark run.

    Parameters
    ----------
    max_gen_toks : int
        Maximum output tokens lm-eval will request per item.
    aggregate_tok_s : float
        Total throughput across all concurrent workers (tok/s), as measured
        by a sweep or quick probe. Divided by *concurrency* to get per-request rate.
    concurrency : int
        Number of parallel lm-eval worker threads/processes.
    client_timeout_s : float
        The --timeout value (seconds) that will be passed to lm-eval.
    safety_factor : float
        Multiplier applied to worst_case_s. Default 2.0.

    Computed attributes (set post-init)
    ----------
    per_request_tok_s  : float
    worst_case_s       : float
    required_timeout_s : float
    margin_s           : float
    feasible           : bool
    """
    max_gen_toks: int
    aggregate_tok_s: float
    concurrency: int
    client_timeout_s: float
    safety_factor: float = 2.0

    # computed (set in __post_init__)
    per_request_tok_s: float = dataclasses.field(init=False)
    worst_case_s: float = dataclasses.field(init=False)
    required_timeout_s: float = dataclasses.field(init=False)
    margin_s: float = dataclasses.field(init=False)
    feasible: bool = dataclasses.field(init=False)

    def __post_init__(self):
        _validate_positive("max_gen_toks", self.max_gen_toks)
        _validate_positive("aggregate_tok_s", self.aggregate_tok_s)
        _validate_positive("concurrency", self.concurrency)
        _validate_positive("client_timeout_s", self.client_timeout_s)
        _validate_positive("safety_factor", self.safety_factor)

        self.per_request_tok_s  = self.aggregate_tok_s / self.concurrency
        self.worst_case_s       = self.max_gen_toks / self.per_request_tok_s
        self.required_timeout_s = self.safety_factor * self.worst_case_s
        self.margin_s           = self.client_timeout_s - self.required_timeout_s
        self.feasible           = self.client_timeout_s >= self.required_timeout_s

    def __repr__(self) -> str:
        return (
            f"TimeoutArithmetic("
            f"per_request_tok_s={self.per_request_tok_s:.2f}, "
            f"worst_case_s={self.worst_case_s:.1f}, "
            f"required_timeout_s={self.required_timeout_s:.1f}, "
            f"client_timeout_s={self.client_timeout_s:.1f}, "
            f"margin_s={self.margin_s:.1f}, "
            f"feasible={self.feasible})"
        )

    def print_report(self, label: str = "timeout-arithmetic") -> None:
        status = "OK" if self.feasible else "INFEASIBLE"
        print(f"\n[{label}]")
        print(f"  per-request throughput : {self.per_request_tok_s:.2f} tok/s"
              f"  ({self.aggregate_tok_s:.1f} agg / {self.concurrency} workers)")
        print(f"  worst-case time        : {self.worst_case_s:.1f} s"
              f"  ({self.max_gen_toks} toks / {self.per_request_tok_s:.2f} tok/s)")
        print(f"  required timeout       : {self.required_timeout_s:.1f} s"
              f"  (safety={self.safety_factor}x * worst_case)")
        print(f"  client timeout         : {self.client_timeout_s:.1f} s")
        print(f"  margin                 : {self.margin_s:+.1f} s")
        print(f"  verdict                : {status}")
        if not self.feasible:
            deficit = -self.margin_s
            print(f"\n  WARNING: client timeout is {deficit:.0f} s SHORT of the required budget.")
            print(f"  Long items will time out, retry, and hit the same wall repeatedly.")
            print(f"  Increase --timeout to at least {self.required_timeout_s:.0f} s before launching.")


def _validate_positive(name: str, value) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")


def compute_timeout_arithmetic(
    max_gen_toks: int,
    aggregate_tok_s: float,
    concurrency: int,
    client_timeout_s: float,
    safety_factor: float = 2.0,
) -> TimeoutArithmetic:
    """Functional entry point; delegates to TimeoutArithmetic dataclass."""
    return TimeoutArithmetic(
        max_gen_toks=max_gen_toks,
        aggregate_tok_s=aggregate_tok_s,
        concurrency=concurrency,
        client_timeout_s=client_timeout_s,
        safety_factor=safety_factor,
    )


# ---------------------------------------------------------------------------
# Quality preflight gate
# ---------------------------------------------------------------------------

def _default_subprocess_runner(cmd: list[str]) -> int:
    """Run a command and return its exit code. Network/GPU calls go here."""
    result = subprocess.run(cmd, check=False)
    return result.returncode


class QualityPreflightGate:
    """Integrated quality-run gate: serving preflight + budget check + timeout arithmetic.

    Dependency injection: replace _run_serving_preflight and _run_budget_check
    attributes with mocks in tests to avoid network/GPU calls.
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        max_gen_toks: int,
        aggregate_tok_s: float,
        concurrency: int,
        client_timeout_s: float,
        safety_factor: float = 2.0,
        max_tokens_probe: int = 1024,
        runner: Optional[Callable[[list[str]], int]] = None,
    ):
        self.endpoint        = endpoint
        self.model           = model
        self.max_gen_toks    = max_gen_toks
        self.client_timeout_s = client_timeout_s
        self.max_tokens_probe = max_tokens_probe

        # Pre-compute arithmetic (offline, always runs)
        self._arithmetic = compute_timeout_arithmetic(
            max_gen_toks=max_gen_toks,
            aggregate_tok_s=aggregate_tok_s,
            concurrency=concurrency,
            client_timeout_s=client_timeout_s,
            safety_factor=safety_factor,
        )

        _runner = runner or _default_subprocess_runner

        # Public injection points: callables(model, endpoint, int) -> int.
        # Tests replace these attributes with MagicMock and inspect call_args to verify
        # the exact model ID is forwarded — do NOT collapse into zero-arg lambdas here.
        def _serving_impl(model: str, endpoint: str, max_tokens: int) -> int:
            return _runner([
                sys.executable,
                _script_path("preflight_serving.py"),
                "--endpoint", endpoint,
                "--model", model,
                "--max-tokens", str(max_tokens),
            ])

        def _budget_impl(model: str, endpoint: str, require_budget: int) -> int:
            # The two existing CLIs deliberately use different URL contracts:
            # preflight_serving.py takes the OpenAI base ending in /v1, while
            # check_output_budget.py takes the host root and appends /v1/models.
            base = endpoint.rstrip("/")
            if base.endswith("/v1"):
                base = base[:-3]
            return _runner([
                sys.executable,
                _script_path("check_output_budget.py"),
                "--model", model,
                "--base-url", base,
                "--require-budget", str(require_budget),
            ])

        self._run_serving_preflight = _serving_impl
        self._run_budget_check      = _budget_impl

    def run(self, verbose: bool = True) -> ExitCode:
        """Execute all three checks in sequence. Fail-closed.

        Returns ExitCode.OK only when all pass. Any failure short-circuits.
        """
        if verbose:
            print("=" * 60)
            print("QUALITY PREFLIGHT GATE")
            print(f"  endpoint  : {self.endpoint}")
            print(f"  model     : {self.model}")
            print(f"  max_gen_toks: {self.max_gen_toks}")
            print("=" * 60)

        # --- (a) Serving preflight ---
        try:
            serving_rc = self._run_serving_preflight(
                self.model, self.endpoint, self.max_tokens_probe)
        except Exception as exc:
            if verbose:
                print(f"\n[serving-preflight] INCONCLUSIVE: runner raised {exc}")
            return ExitCode.INCONCLUSIVE

        if verbose:
            tag = {0: "OK", 1: "DEFECT", 2: "INCONCLUSIVE"}.get(serving_rc, f"exit={serving_rc}")
            print(f"\n[serving-preflight] {tag}")

        if serving_rc == 1:
            if verbose:
                print("  Endpoint has a defect. Do NOT launch.")
            return ExitCode.DEFECT
        if serving_rc != 0:
            if verbose:
                print("  Could not probe endpoint. Do NOT launch.")
            return ExitCode.INCONCLUSIVE

        # --- (b) Output budget check ---
        try:
            budget_rc = self._run_budget_check(
                self.model, self.endpoint, self.max_gen_toks)
        except Exception as exc:
            if verbose:
                print(f"\n[budget-check] INCONCLUSIVE: runner raised {exc}")
            return ExitCode.INCONCLUSIVE

        if verbose:
            tag = {0: "OK", 1: "DEFECT", 2: "INCONCLUSIVE"}.get(budget_rc, f"exit={budget_rc}")
            print(f"\n[budget-check] {tag}")

        if budget_rc == 1:
            if verbose:
                print("  Output budget is insufficient. Do NOT launch.")
            return ExitCode.DEFECT
        if budget_rc != 0:
            if verbose:
                print("  Could not verify budget. Do NOT launch.")
            return ExitCode.INCONCLUSIVE

        # --- (c) Timeout arithmetic ---
        arith = self._arithmetic
        if verbose:
            arith.print_report()

        if not arith.feasible:
            return ExitCode.DEFECT

        if verbose:
            print("\nOK: all preflight checks passed. Safe to launch.\n")
        return ExitCode.OK


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _script_path(name: str) -> str:
    """Return path to a sibling viz/ script."""
    return os.path.join(os.path.dirname(__file__), name)


# ---------------------------------------------------------------------------
# Self-test (arithmetic fixtures, no network)
# ---------------------------------------------------------------------------

_ARITH_FIXTURES = [
    # (label, kwargs, expected_feasible, expected_per_req, expected_required)
    ("basic_feasible",
     dict(max_gen_toks=4096, aggregate_tok_s=160.0, concurrency=10,
          client_timeout_s=3600, safety_factor=2.0),
     True, 16.0, 512.0),
    ("gpqa_abort_infeasible",
     dict(max_gen_toks=32768, aggregate_tok_s=64.0, concurrency=16,
          client_timeout_s=3600, safety_factor=2.0),
     False, 4.0, 16384.0),
    ("safety_factor_1",
     dict(max_gen_toks=1000, aggregate_tok_s=100.0, concurrency=1,
          client_timeout_s=9999, safety_factor=1.0),
     True, 100.0, 10.0),
]


def run_self_test() -> int:
    failures = 0
    print("Running arithmetic self-tests (offline)...\n")
    for label, kwargs, exp_feasible, exp_per_req, exp_required in _ARITH_FIXTURES:
        try:
            r = compute_timeout_arithmetic(**kwargs)
            ok = (
                r.feasible == exp_feasible
                and abs(r.per_request_tok_s - exp_per_req) < 0.01
                and abs(r.required_timeout_s - exp_required) < 0.1
            )
        except Exception as e:
            ok = False
            print(f"  [FAIL] {label}: raised {e}")
            failures += 1
            continue
        failures += not ok
        status = "pass" if ok else "FAIL"
        print(f"  [{status}] {label:35s}  feasible={r.feasible} "
              f"per_req={r.per_request_tok_s:.1f} required={r.required_timeout_s:.1f}")

    print()
    if failures:
        print(f"SELF-TEST FAILED: {failures} fixture(s) wrong.")
        return 1
    print(f"OK: {len(_ARITH_FIXTURES)}/{len(_ARITH_FIXTURES)} arithmetic fixtures passed.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Quality-run preflight gate: serving + budget + timeout arithmetic.")

    ap.add_argument("--self-test", action="store_true",
                    help="Run offline arithmetic fixture tests (CI, no GPU).")
    ap.add_argument("--check-timeout-only", action="store_true",
                    help="Run arithmetic check only (offline). Requires arithmetic flags.")

    # Serving / model args
    ap.add_argument("--endpoint", default="http://csi370295.alcf.anl.gov:8000/v1",
                    help="OpenAI-compatible base URL ending in /v1")
    ap.add_argument("--model", default=None,
                    help="Exact model ID that must be served (required unless --check-timeout-only)")
    ap.add_argument("--max-tokens-probe", type=int, default=1024,
                    help="max_tokens for serving preflight probes")

    # Arithmetic args
    ap.add_argument("--max-gen-toks", type=int, default=None,
                    help="Max output tokens per request for the benchmark")
    ap.add_argument("--aggregate-tok-s", type=float, default=None,
                    help="Total aggregate throughput across all workers (tok/s)")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="Number of parallel lm-eval workers")
    ap.add_argument("--client-timeout", type=float, default=None,
                    help="lm-eval --timeout value in seconds")
    ap.add_argument("--safety-factor", type=float, default=2.0,
                    help="Multiplier on worst-case time (default 2.0)")

    args = ap.parse_args(argv)

    if args.self_test:
        return run_self_test()

    if args.check_timeout_only:
        missing = [f for f in ("max_gen_toks", "aggregate_tok_s", "concurrency", "client_timeout")
                   if getattr(args, f.replace("-", "_"), None) is None]
        if missing:
            ap.error(f"--check-timeout-only requires: {', '.join('--' + m.replace('_','-') for m in missing)}")

        try:
            arith = compute_timeout_arithmetic(
                max_gen_toks=int(args.max_gen_toks),
                aggregate_tok_s=float(args.aggregate_tok_s),
                concurrency=int(args.concurrency),
                client_timeout_s=float(args.client_timeout),
                safety_factor=args.safety_factor,
            )
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return ExitCode.DEFECT

        arith.print_report()
        return ExitCode.OK if arith.feasible else ExitCode.DEFECT

    # Full gate: all args required
    missing_full = []
    if not args.model:
        missing_full.append("--model")
    if args.max_gen_toks is None:
        missing_full.append("--max-gen-toks")
    if args.aggregate_tok_s is None:
        missing_full.append("--aggregate-tok-s")
    if args.concurrency is None:
        missing_full.append("--concurrency")
    if args.client_timeout is None:
        missing_full.append("--client-timeout")

    if missing_full:
        ap.error(f"Full preflight requires: {', '.join(missing_full)}")

    try:
        # Narrowing asserts: the missing_full guard above ensures these are set.
        assert args.model is not None
        assert args.max_gen_toks is not None
        assert args.aggregate_tok_s is not None
        assert args.concurrency is not None
        assert args.client_timeout is not None
        gate = QualityPreflightGate(
            endpoint=args.endpoint,
            model=args.model,
            max_gen_toks=args.max_gen_toks,
            aggregate_tok_s=args.aggregate_tok_s,
            concurrency=args.concurrency,
            client_timeout_s=args.client_timeout,
            safety_factor=args.safety_factor,
            max_tokens_probe=args.max_tokens_probe,
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return ExitCode.DEFECT

    return gate.run()


if __name__ == "__main__":
    sys.exit(main())
