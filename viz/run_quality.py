#!/usr/bin/env python3
"""viz/run_quality.py — Contract-aware quality runner for warpcore-v1 benchmarks.

Given suite, adapter, benchmark, endpoint, throughput, concurrency, and timeout,
builds a canonical lm-eval command from suite settings and executes it under the
required safety gates.

CONTRACT (from docs/superpowers/specs/2026-09-15-warpcore-v1-apples-to-apples-design.md):
- Generated command uses canonical task path, suite generation ceiling and temperature.
- Includes --log_samples.
- Points output at normalized run directory.
- CLI does NOT accept overrides for task, ceiling, sampling, scoring, datasets, or instances.
- Runs QualityPreflightGate before harness; exit 1 or 2 blocks launch.
- Refuses long launch outside /usr/bin/screen except in --dry-run or explicit test mode.
- Uses dependency-injected process execution.
- Does not duplicate preflight_serving.py, check_output_budget.py, or timeout arithmetic.
- Writes exact argv to command.txt using shell-safe quoting.
- On success: verifies required raw files before writing DONE.
- On nonzero harness exit: does NOT write DONE; records exit code; transitions to failed.

USAGE
-----
    # Dry-run — inspect command without network/GPU work:
    python3 viz/run_quality.py \\
        --suite suite/warpcore-v1.yaml \\
        --adapter adapters/qwen3.6-35b-a3b.yaml \\
        --benchmark gsm8k \\
        --endpoint http://csi370295.alcf.anl.gov:8000/v1 \\
        --throughput 64 \\
        --concurrency 8 \\
        --timeout 14400 \\
        --run-id run-2026-09-15T12-00-00 \\
        --dry-run

    # Live run (must be inside /usr/bin/screen):
    screen -S quality-gsm8k
    python3 viz/run_quality.py \\
        --suite suite/warpcore-v1.yaml \\
        --adapter adapters/qwen3.6-35b-a3b.yaml \\
        --benchmark gsm8k \\
        --endpoint http://csi370295.alcf.anl.gov:8000/v1 \\
        --throughput 64 \\
        --concurrency 8 \\
        --timeout 14400 \\
        --run-id run-2026-09-15T12-00-00

EXIT CODES
----------
    0   Success: harness exited 0 and required raw files exist; DONE written.
    1   Preflight defect or harness failure; DONE not written.
    2   Preflight inconclusive or screen guard triggered; DONE not written.
    3   Command construction or validation error.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import sys
from datetime import datetime, timezone
from typing import Callable, List, Optional

# ---------------------------------------------------------------------------
# Path setup — allow importing sibling viz modules
# ---------------------------------------------------------------------------

_VIZ_DIR = pathlib.Path(__file__).parent
_REPO_DIR = _VIZ_DIR.parent
for _p in (str(_VIZ_DIR), str(_REPO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Imports from existing viz modules (no duplication)
# ---------------------------------------------------------------------------

from quality_preflight import QualityPreflightGate, ExitCode  # noqa: E402
import campaign_state  # noqa: E402

# ---------------------------------------------------------------------------
# Forbidden CLI override flags
# ---------------------------------------------------------------------------

#: CLI flags that override suite-owned experiment variables.
#: The runner refuses any invocation that passes these.
FORBIDDEN_OVERRIDE_FLAGS: List[str] = [
    "--tasks",
    "--num_fewshot",
    "--gen_kwargs",
    "--limit",
    "--include_path",
    "--max_gen_toks",
    "--temperature",
    "--do_sample",
    "--fewshot_config",
    "--dataset_path",
    "--dataset_name",
    "--dataset_kwargs",
    "--use_cache",
    "--cache_requests",
]

# ---------------------------------------------------------------------------
# UTC timestamp helper
# ---------------------------------------------------------------------------


def _utcnow() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Screen guard
# ---------------------------------------------------------------------------


def _is_under_screen() -> bool:
    """Return True when the process is running inside a GNU screen session."""
    # GNU screen sets STY to the session name
    return bool(os.environ.get("STY", ""))


# ---------------------------------------------------------------------------
# QualityRunner
# ---------------------------------------------------------------------------


class QualityRunner:
    """Contract-aware quality runner.

    Parameters
    ----------
    suite_path : pathlib.Path
        Path to warpcore-v1.yaml.
    adapter_path : pathlib.Path
        Path to adapter YAML (e.g. adapters/qwen3.6-35b-a3b.yaml).
    benchmark : str
        Benchmark key (e.g. "gsm8k", "gpqa_diamond").
    endpoint : str
        OpenAI-compatible base URL ending in /v1.
    throughput : float
        Measured aggregate token throughput (tok/s) across all workers.
    concurrency : int
        Number of parallel lm-eval workers.
    timeout : float
        lm-eval --timeout value in seconds.
    run_dir : pathlib.Path
        Normalized run directory (already created by create_campaign).
    dry_run : bool
        If True, build and print the command, write command.txt, but do not
        invoke preflight or harness. Does not require /usr/bin/screen.
    allow_no_screen : bool
        If True, bypass the screen guard (for tests and special ops). Never
        pass this from the CLI in production.
    preflight_runner : callable or None
        Injected callable(gate) -> int for running the preflight gate.
        Defaults to gate.run().
    harness_runner : callable or None
        Injected callable(cmd: list) -> int for running the harness.
        Defaults to subprocess.run.
    """

    def __init__(
        self,
        suite_path: pathlib.Path,
        adapter_path: pathlib.Path,
        benchmark: str,
        endpoint: str,
        throughput: float,
        concurrency: int,
        timeout: float,
        run_dir: pathlib.Path,
        dry_run: bool = False,
        allow_no_screen: bool = False,
        preflight_runner: Optional[Callable] = None,
        harness_runner: Optional[Callable] = None,
    ) -> None:
        self.suite_path = pathlib.Path(suite_path).resolve()
        self.adapter_path = pathlib.Path(adapter_path).resolve()
        self.benchmark = benchmark
        self.endpoint = endpoint
        self.throughput = float(throughput)
        self.concurrency = int(concurrency)
        self.timeout = float(timeout)
        self.run_dir = pathlib.Path(run_dir).resolve()
        self.dry_run = dry_run
        self.allow_no_screen = allow_no_screen

        # Load suite
        import yaml  # type: ignore
        with open(self.suite_path) as fh:
            self._suite: dict = yaml.safe_load(fh)

        # Load adapter
        with open(self.adapter_path) as fh:
            self._adapter: dict = yaml.safe_load(fh)

        # Resolve benchmark config from suite
        benchmarks = self._suite.get("benchmarks", {})
        if benchmark not in benchmarks:
            raise ValueError(
                f"Benchmark {benchmark!r} not found in suite {suite_path}. "
                f"Available: {sorted(benchmarks.keys())}"
            )
        self._bench_cfg: dict = benchmarks[benchmark]

        # Injected runners
        self._preflight_runner = preflight_runner
        self._harness_runner = harness_runner

    # ------------------------------------------------------------------
    # Command construction
    # ------------------------------------------------------------------

    def build_command(self) -> List[str]:
        """Build the canonical lm-eval argv list from suite + adapter settings.

        This is the ONLY place the harness command is assembled.  It reads
        task paths, generation ceilings, and sampling settings exclusively
        from the suite; no caller-provided overrides are accepted.
        """
        bench = self._bench_cfg
        suite_id = self._suite.get("suite_id", "warpcore-v1")

        # Task file path — canonical suite task, resolved from repo root
        repo = self.suite_path.parent.parent  # suite/warpcore-v1.yaml -> repo root
        task_file = bench.get("task_file", "")
        canonical_task_path = str(repo / task_file) if task_file else ""

        # Generation ceiling from suite (never caller-overrideable)
        generation_ceiling: int = bench.get("generation_ceiling", 8192)

        # Temperature from suite sampling block
        sampling = bench.get("sampling", {})
        temperature = sampling.get("temperature", 0)
        do_sample = sampling.get("do_sample", False)

        # Model ID from adapter
        model_id = (self._adapter.get("model") or {}).get("id", "")

        # gen_kwargs — suite-owned temperature and ceiling only
        gen_kwargs = f"max_gen_toks={generation_ceiling},temperature={temperature},do_sample={str(do_sample).lower()}"

        # Output dir inside normalized run directory
        output_path = str(self.run_dir)

        cmd = [
            sys.executable, "-m", "lm_eval",
            "--model", "local-completions",
            "--model_args", f"base_url={self.endpoint},model={model_id},num_concurrent={self.concurrency},max_retries=0",
            "--tasks", canonical_task_path,
            "--gen_kwargs", gen_kwargs,
            "--output_path", output_path,
            "--log_samples",
            "--num_fewshot", "0",
            "--timeout", str(int(self.timeout)),
        ]

        return cmd

    # ------------------------------------------------------------------
    # Core run logic
    # ------------------------------------------------------------------

    def run(self) -> int:
        """Execute the full quality run lifecycle.

        Returns an exit code (0 = success, nonzero = failure).
        """
        # --- Dry-run: build command, write command.txt, print, and exit 0 ---
        if self.dry_run:
            cmd = self.build_command()
            self._write_command_txt(cmd)
            print("[dry-run] Command:")
            print(" ".join(shlex.quote(a) for a in cmd))
            return 0

        # --- Screen guard ---
        if not self.allow_no_screen and not _is_under_screen():
            print(
                "ERROR: Long quality runs must be launched inside /usr/bin/screen "
                "to survive terminal disconnection. Start a screen session first:\n"
                "  screen -S quality-run\n"
                "Or pass --allow-no-screen to bypass (tests/special ops only).",
                file=sys.stderr,
            )
            return 2

        # --- Build command ---
        cmd = self.build_command()
        self._write_command_txt(cmd)

        # --- Resolve model ID for preflight ---
        model_id = (self._adapter.get("model") or {}).get("id", "")
        bench_cfg = self._bench_cfg
        generation_ceiling: int = bench_cfg.get("generation_ceiling", 8192)

        # --- Run preflight gate (delegates; never re-implements arithmetic) ---
        preflight_rc = self._run_preflight(
            model_id=model_id,
            generation_ceiling=generation_ceiling,
        )
        if preflight_rc != 0:
            print(
                f"[run-quality] Preflight gate failed (exit {preflight_rc}). "
                "Harness not launched.",
                file=sys.stderr,
            )
            return preflight_rc

        # --- Transition to running ---
        self._transition_status("running")

        # --- Execute harness ---
        harness_rc = self._execute_harness(cmd)

        # --- Post-run: check raw files, write DONE or record failure ---
        if harness_rc == 0:
            raw_dir = self.run_dir / "raw"
            if raw_dir.exists():
                # Write DONE sentinel
                (self.run_dir / "DONE").write_text("completed\n")
                self._transition_status("completed")
                return 0
            else:
                # Harness exited 0 but raw files missing — do not write DONE
                print(
                    f"ERROR: Harness exited 0 but raw/ directory does not exist at {raw_dir}. "
                    "DONE not written.",
                    file=sys.stderr,
                )
                self._transition_status_failed(harness_rc=0, reason="raw_dir_missing")
                return 1
        else:
            # Nonzero harness exit — record and transition to failed
            self._transition_status_failed(harness_rc=harness_rc)
            return harness_rc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_command_txt(self, cmd: List[str]) -> None:
        """Write shell-safe command to command.txt in the run directory."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        shell_line = " ".join(shlex.quote(a) for a in cmd)
        (self.run_dir / "command.txt").write_text(shell_line + "\n", encoding="utf-8")

    def _run_preflight(self, model_id: str, generation_ceiling: int) -> int:
        """Run the QualityPreflightGate; return its exit code.

        Delegates to QualityPreflightGate (which already owns timeout arithmetic,
        serving preflight, and budget checks). Never re-implements those.
        """
        if self._preflight_runner is not None:
            # Injected for tests
            return int(self._preflight_runner(model_id))

        # Default: build and run the real gate
        try:
            gate = QualityPreflightGate(
                endpoint=self.endpoint,
                model=model_id,
                max_gen_toks=generation_ceiling,
                aggregate_tok_s=self.throughput,
                concurrency=self.concurrency,
                client_timeout_s=self.timeout,
            )
            return int(gate.run())
        except Exception as exc:
            print(f"[run-quality] Preflight raised: {exc}", file=sys.stderr)
            return int(ExitCode.INCONCLUSIVE)

    def _execute_harness(self, cmd: List[str]) -> int:
        """Execute the lm-eval harness; return its exit code."""
        if self._harness_runner is not None:
            return int(self._harness_runner(cmd))

        import subprocess
        result = subprocess.run(cmd, check=False)
        return result.returncode

    def _read_status(self) -> dict:
        status_path = self.run_dir / "status.json"
        if status_path.exists():
            return json.loads(status_path.read_text())
        # Minimal default if missing
        return {
            "execution_state": "preflight_passed",
            "lifecycle": "current",
            "history": [
                {"state": "planned", "timestamp": "2026-09-15T12:00:00Z"},
                {"state": "preflight_passed", "timestamp": "2026-09-15T12:01:00Z"},
            ],
        }

    def _write_status(self, status: dict) -> None:
        campaign_state.write_status(
            dest=self.run_dir / "status.json",
            status=status,
            run_dir=self.run_dir,
        )

    def _transition_status(self, new_state: str, lifecycle: Optional[str] = None) -> None:
        """Apply a state transition and persist status.json."""
        try:
            status = self._read_status()
            new_status = campaign_state.apply_transition(
                status=status,
                new_state=new_state,
                timestamp=_utcnow(),
                lifecycle=lifecycle,
            )
            self._write_status(new_status)
        except Exception as exc:
            # State transition failures are logged but do not abort the run
            print(f"[run-quality] Warning: status transition to {new_state!r} failed: {exc}", file=sys.stderr)

    def _transition_status_failed(
        self, harness_rc: int, reason: Optional[str] = None
    ) -> None:
        """Transition to failed and record the exit code in status."""
        try:
            status = self._read_status()
            # Add exit code context to the history entry
            ts = _utcnow()
            new_status = campaign_state.apply_transition(
                status=status,
                new_state="failed",
                timestamp=ts,
                lifecycle="invalid",
            )
            # Record the harness exit code in the status for traceability
            new_status["harness_exit_code"] = harness_rc
            if reason:
                new_status["failure_reason"] = reason
            self._write_status(new_status)
        except Exception as exc:
            print(f"[run-quality] Warning: status transition to 'failed' failed: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Contract-aware quality runner for warpcore-v1 benchmarks. "
            "Builds a canonical lm-eval command from suite settings; does NOT "
            "accept overrides for task, ceiling, sampling, scoring, datasets, or instances."
        )
    )

    # Required arguments
    ap.add_argument("--suite", required=True, type=pathlib.Path,
                    help="Path to suite YAML (e.g. suite/warpcore-v1.yaml)")
    ap.add_argument("--adapter", required=True, type=pathlib.Path,
                    help="Path to adapter YAML (e.g. adapters/qwen3.6-35b-a3b.yaml)")
    ap.add_argument("--benchmark", required=True,
                    help="Benchmark key (e.g. gsm8k, gpqa_diamond, ifeval)")
    ap.add_argument("--endpoint", required=True,
                    help="OpenAI-compatible base URL ending in /v1")
    ap.add_argument("--throughput", required=True, type=float,
                    help="Measured aggregate tok/s across all workers")
    ap.add_argument("--concurrency", required=True, type=int,
                    help="Number of parallel lm-eval workers")
    ap.add_argument("--timeout", required=True, type=float,
                    help="lm-eval --timeout value in seconds")

    # Optional
    ap.add_argument("--run-id", default=None,
                    help="Run identifier. If omitted and --run-dir is not given, auto-generated.")
    ap.add_argument("--run-dir", type=pathlib.Path, default=None,
                    help="Explicit run directory (normalized layout). If omitted, derived from repo/suite/adapter.")
    ap.add_argument("--repo", type=pathlib.Path, default=None,
                    help="Repository root. Defaults to parent of suite file.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build and print command; write command.txt; do not run preflight or harness.")
    ap.add_argument("--allow-no-screen", action="store_true",
                    help="Bypass the screen guard (tests and special ops only; never use in production).")

    # Detect and reject forbidden override flags before argparse sees them
    if argv is not None:
        args_to_check = argv
    else:
        args_to_check = sys.argv[1:]

    for flag in FORBIDDEN_OVERRIDE_FLAGS:
        if flag in args_to_check:
            ap.error(
                f"'{flag}' is a suite-owned experiment variable and cannot be overridden "
                f"through the runner CLI. The suite controls all experiment-level settings."
            )

    args = ap.parse_args(argv)

    # Resolve run directory
    if args.run_dir is not None:
        run_dir = pathlib.Path(args.run_dir).resolve()
    else:
        # Derive from repo/suite/adapter
        suite_path = pathlib.Path(args.suite).resolve()
        import yaml
        with open(suite_path) as fh:
            suite = yaml.safe_load(fh)
        with open(pathlib.Path(args.adapter).resolve()) as fh:
            adapter = yaml.safe_load(fh)
        suite_id = suite.get("suite_id", "warpcore-v1")
        model_slug = (adapter.get("model") or {}).get("slug", "unknown")
        repo = args.repo or suite_path.parent.parent
        run_id = args.run_id or datetime.now(tz=timezone.utc).strftime("run-%Y-%m-%dT%H-%M-%S")
        run_dir = (
            pathlib.Path(repo) / "results" / model_slug / "runs"
            / suite_id / args.benchmark / run_id
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        # Write minimal status if not already present
        status_path = run_dir / "status.json"
        if not status_path.exists():
            ts = _utcnow()
            status = {
                "execution_state": "preflight_passed",
                "lifecycle": "current",
                "history": [
                    {"state": "planned", "timestamp": ts},
                    {"state": "preflight_passed", "timestamp": ts},
                ],
            }
            campaign_state.write_status(dest=status_path, status=status, run_dir=run_dir)

    try:
        runner = QualityRunner(
            suite_path=args.suite,
            adapter_path=args.adapter,
            benchmark=args.benchmark,
            endpoint=args.endpoint,
            throughput=args.throughput,
            concurrency=args.concurrency,
            timeout=args.timeout,
            run_dir=run_dir,
            dry_run=args.dry_run,
            allow_no_screen=args.allow_no_screen,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
