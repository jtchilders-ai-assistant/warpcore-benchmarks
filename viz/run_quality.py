#!/usr/bin/env python3
"""viz/run_quality.py — Contract-aware quality runner for warpcore-v1 benchmarks.

Given suite, adapter, benchmark, endpoint, throughput, concurrency, and timeout,
builds a canonical lm-eval command from suite settings and executes it under the
required safety gates.

CONTRACT (from docs/superpowers/specs/2026-09-15-warpcore-v1-apples-to-apples-design.md):
- Generated command uses local-chat-completions (routes to /v1/chat/completions).
- Includes --apply_chat_template and tokenized_requests=False.
- Uses --include_path <task_dir> and --tasks <task_name_from_yaml>.
- Uses canonical task name from suite task_file YAML 'task:' field, not the file path.
- Includes --log_samples.
- Points output at normalized run directory.
- CLI does NOT accept overrides for task, ceiling, sampling, scoring, datasets, or instances.
- Runs QualityPreflightGate before harness; exit 1 or 2 blocks launch.
- Refuses long launch outside /usr/bin/screen except in --dry-run or explicit test mode.
- Uses dependency-injected process execution.
- Does not duplicate preflight_serving.py, check_output_budget.py, or timeout arithmetic.
- Writes exact argv to command.txt using shell-safe quoting.
- On success: verifies required raw files (aggregate result + nonempty samples) before writing DONE.
- On nonzero harness exit: does NOT write DONE; records exit_code in history entry;
  transitions to failed. Does NOT add top-level harness_exit_code (schema forbids it).
- State transitions are fatal: missing/invalid status.json or illegal transition aborts run.
- Noncanonical adapters are rejected at construction time.
- Run directory must resolve inside the repository root (containment check).

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
    1   Preflight defect, harness failure, or lifecycle/evidence error; DONE not written.
    2   Preflight inconclusive or screen guard triggered; DONE not written.
    3   Command construction, validation, or containment error.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
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
from contract import (  # noqa: E402
    load_yaml,
    validate_json,
    validate_adapter,
    validate_adapter_campaign_ready,
)

# ---------------------------------------------------------------------------
# Schemas dir
# ---------------------------------------------------------------------------

_SCHEMAS_DIR = _REPO_DIR / "suite" / "schemas"
_STATUS_SCHEMA = _SCHEMAS_DIR / "result-status.schema.json"

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
# Evidence verification
# ---------------------------------------------------------------------------

# Patterns matching lm-eval 0.4.12 output file naming conventions.
# These are the minimum required evidence: aggregate result + nonempty samples.
_AGGREGATE_RESULT_RE = re.compile(r"^results_.*\.json$")
_SAMPLES_JSONL_GZ_RE = re.compile(r"^samples_.*\.jsonl\.gz$")


def _verify_required_evidence(run_dir: pathlib.Path) -> List[str]:
    """Return a list of error strings if required harness artifacts are missing.

    Required minimum (subset of full required_evidence from suite):
      - At least one aggregate result JSON (results_*.json) in raw/
      - At least one nonempty samples JSONL.GZ (samples_*.jsonl.gz) in raw/

    Returns [] on success, list of errors on failure.
    """
    raw_dir = run_dir / "raw"
    errors: List[str] = []

    if not raw_dir.exists():
        errors.append(f"raw/ directory does not exist at {raw_dir}")
        return errors

    files = list(raw_dir.iterdir())
    if not files:
        errors.append(f"raw/ directory is empty at {raw_dir} — no harness artifacts produced")
        return errors

    aggregate_results = [f for f in files if _AGGREGATE_RESULT_RE.match(f.name)]
    if not aggregate_results:
        errors.append(
            f"No aggregate result file (results_*.json) found in {raw_dir}. "
            "lm-eval must produce at least one results JSON."
        )

    samples_gz = [f for f in files if _SAMPLES_JSONL_GZ_RE.match(f.name)]
    if not samples_gz:
        errors.append(
            f"No samples file (samples_*.jsonl.gz) found in {raw_dir}. "
            "lm-eval must produce compressed samples (--log_samples required)."
        )
    else:
        # Verify at least one sample file is nonempty (gzip header at minimum)
        nonempty = [f for f in samples_gz if f.stat().st_size > 20]
        if not nonempty:
            errors.append(
                f"All samples_*.jsonl.gz files in {raw_dir} are empty. "
                "Evidence is required to be nonempty."
            )

    return errors


# ---------------------------------------------------------------------------
# Task name extraction
# ---------------------------------------------------------------------------


def _read_task_name_from_yaml(task_file_path: pathlib.Path) -> str:
    """Read the registered 'task:' name from a lm-eval task YAML file.

    lm-eval 0.4.12 registers tasks by the 'task:' field in the YAML, not by
    filename. --tasks must receive this registered name, not the file path.

    lm-eval task YAMLs may use custom tags (e.g. ``!function``) that PyYAML's
    SafeLoader rejects.  We only need the scalar ``task:`` field, so we install
    a permissive multi-tag constructor that yields ``None`` for any unknown tag
    — sufficient for key extraction without executing any callables.

    Raises ValueError if the 'task:' field is absent or empty.
    """
    import yaml as _yaml

    class _PermissiveLoader(_yaml.SafeLoader):
        pass

    # Accept any !tag by returning None for unknown constructors
    _PermissiveLoader.add_multi_constructor(
        "",
        lambda loader, tag_suffix, node: None,
    )

    raw = pathlib.Path(task_file_path).read_text(encoding="utf-8")
    data = _yaml.load(raw, Loader=_PermissiveLoader)  # noqa: S506 — not untrusted, local repo file
    if not isinstance(data, dict):
        raise ValueError(
            f"Task YAML {task_file_path} did not parse to a mapping; got {type(data).__name__}."
        )
    task_name = data.get("task", "") or ""
    if not task_name:
        raise ValueError(
            f"Task YAML {task_file_path} does not define a 'task:' field. "
            "lm-eval registers tasks by name; the 'task:' field is required."
        )
    return str(task_name)


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
    allow_noncanonical_adapter : bool
        If True, skip the noncanonical adapter check (for test fixtures only).
        Never pass this from the CLI in production.
    preflight_runner : callable or None
        Injected callable(model_id) -> int for running the preflight gate.
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
        allow_noncanonical_adapter: bool = False,
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

        # -- Adapter validation (fail closed on noncanonical) --
        if not allow_noncanonical_adapter:
            repo = self.suite_path.parent.parent
            adapter_errors = validate_adapter(repo, self.adapter_path)
            if adapter_errors:
                raise ValueError(
                    f"Adapter schema validation failed for {self.adapter_path}:\n"
                    + "\n".join(adapter_errors)
                )
            model_slug = (self._adapter.get("model") or {}).get("slug", "")
            readiness_errors = validate_adapter_campaign_ready(
                self._adapter,
                model_slug,
                suite=self._suite,
                prompt_token_maxima=None,
            )
            if readiness_errors:
                raise ValueError(
                    f"Adapter {self.adapter_path} is not campaign-ready (noncanonical):\n"
                    + "\n".join(readiness_errors)
                )

        # Resolve benchmark config from suite
        benchmarks = self._suite.get("benchmarks", {})
        if benchmark not in benchmarks:
            raise ValueError(
                f"Benchmark {benchmark!r} not found in suite {suite_path}. "
                f"Available: {sorted(benchmarks.keys())}"
            )
        self._bench_cfg: dict = benchmarks[benchmark]

        # Suite-level IDs for status documents
        self._suite_id: str = self._suite.get("suite_id", "warpcore-v1")
        self._run_id: str = self.run_dir.name  # last path component

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

        Uses:
          - local-chat-completions (routes to /v1/chat/completions, required for reasoning models)
          - --apply_chat_template (required for chat-completion endpoint)
          - tokenized_requests=False (prevents double-tokenization)
          - --include_path <task_dir> + --tasks <task_name> (not file path)
          - max_retries=0 (suite owns retry_policy)
        """
        bench = self._bench_cfg

        # Task file path — canonical suite task, resolved from repo root
        repo = self.suite_path.parent.parent  # suite/warpcore-v1.yaml -> repo root
        task_file = bench.get("task_file", "")
        if task_file:
            canonical_task_path = (repo / task_file).resolve()
            task_dir = str(canonical_task_path.parent)
            # Read the registered task name from the YAML 'task:' field
            task_name = _read_task_name_from_yaml(canonical_task_path)
        else:
            # Benchmark without a task file (e.g. swebench) — no lm-eval task
            task_dir = ""
            task_name = bench.get("task_name", self.benchmark)

        # Generation ceiling from suite (never caller-overrideable)
        generation_ceiling: int = bench.get("generation_ceiling", 8192)

        # Suite retry policy — max_retries=0 is suite-owned
        retry_policy = bench.get("retry_policy", {})
        max_retries: int = retry_policy.get("max_retries", 0)

        # Temperature from suite sampling block
        sampling = bench.get("sampling", {})
        temperature = sampling.get("temperature", 0)
        do_sample = sampling.get("do_sample", False)

        # Model ID from adapter
        model_id = (self._adapter.get("model") or {}).get("id", "")

        # gen_kwargs — suite-owned temperature and ceiling only
        gen_kwargs = (
            f"max_gen_toks={generation_ceiling},"
            f"temperature={temperature},"
            f"do_sample={str(do_sample).lower()}"
        )

        # model_args — tokenized_requests=False prevents double-tokenization on
        # local-chat-completions endpoint (proven endpoint path for reasoning models)
        model_args = (
            f"base_url={self.endpoint},"
            f"model={model_id},"
            f"num_concurrent={self.concurrency},"
            f"max_retries={max_retries},"
            f"tokenized_requests=False"
        )

        # Output dir inside normalized run directory
        output_path = str(self.run_dir)

        cmd = [
            sys.executable, "-m", "lm_eval",
            "--model", "local-chat-completions",
            "--model_args", model_args,
            "--apply_chat_template",
            "--tasks", task_name,
            "--gen_kwargs", gen_kwargs,
            "--output_path", output_path,
            "--log_samples",
            "--num_fewshot", "0",
            "--timeout", str(int(self.timeout)),
        ]

        # Add --include_path for benchmarks with a custom task YAML
        if task_dir:
            cmd.extend(["--include_path", task_dir])

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

        # --- Read and validate status.json (fail closed if missing or invalid) ---
        try:
            status = self._read_status_strict()
        except Exception as exc:
            print(
                f"[run-quality] FATAL: Cannot read or validate status.json: {exc}. "
                "Refusing to run against an uninitialized or corrupt run directory.",
                file=sys.stderr,
            )
            return 1

        # --- Check that the current state allows launching (must be preflight_passed) ---
        current_state = status.get("execution_state", "")
        if current_state != "preflight_passed":
            print(
                f"[run-quality] FATAL: Run directory is in state {current_state!r}; "
                "expected 'preflight_passed'. Cannot launch a new run from this state. "
                "Only a run directory that has completed preflight and not yet run may be launched.",
                file=sys.stderr,
            )
            return 1

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

        # --- Transition to running (fatal if this fails) ---
        try:
            status = self._read_status_strict()
            new_status = campaign_state.apply_transition(
                status=status,
                new_state="running",
                timestamp=_utcnow(),
            )
            self._write_status(new_status)
        except Exception as exc:
            print(
                f"[run-quality] FATAL: Lifecycle transition to 'running' failed: {exc}. "
                "Aborting — run directory state machine refused the transition.",
                file=sys.stderr,
            )
            return 1

        # --- Execute harness ---
        harness_rc = self._execute_harness(cmd)

        # --- Post-run: verify required artifacts, write DONE or record failure ---
        if harness_rc == 0:
            evidence_errors = _verify_required_evidence(self.run_dir)
            if not evidence_errors:
                # Write DONE sentinel
                (self.run_dir / "DONE").write_text("completed\n")
                self._transition_status_completed()
                return 0
            else:
                # Harness exited 0 but required artifacts missing — do not write DONE
                print(
                    f"ERROR: Harness exited 0 but required evidence is missing:\n"
                    + "\n".join(f"  {e}" for e in evidence_errors)
                    + "\nDONE not written.",
                    file=sys.stderr,
                )
                self._transition_status_failed(harness_rc=0, reason="evidence_missing")
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

    def _read_status_strict(self) -> dict:
        """Read and return status.json; raise if missing or unreadable.

        This is the fail-closed variant. It never fabricates state.
        Raises FileNotFoundError if status.json does not exist.
        Raises ValueError if status.json cannot be parsed.
        """
        status_path = self.run_dir / "status.json"
        if not status_path.exists():
            raise FileNotFoundError(
                f"status.json not found in run directory {self.run_dir}. "
                "Run directory must be initialized by create_campaign before launching."
            )
        try:
            return json.loads(status_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"status.json in {self.run_dir} is not valid JSON: {exc}"
            ) from exc

    def _write_status(self, status: dict) -> None:
        campaign_state.write_status(
            dest=self.run_dir / "status.json",
            status=status,
            run_dir=self.run_dir,
        )

    def _build_status_with_identity(self, status: dict) -> dict:
        """Enrich a status dict with required schema fields (schema_version, run_id, suite_id).

        The result-status.schema.json requires schema_version, run_id, suite_id.
        These may be absent from old/test status dicts; we inject them here.
        """
        enriched = dict(status)
        if "schema_version" not in enriched:
            enriched["schema_version"] = 1
        if "run_id" not in enriched:
            enriched["run_id"] = self._run_id
        if "suite_id" not in enriched:
            enriched["suite_id"] = self._suite_id
        return enriched

    def _transition_status_completed(self) -> None:
        """Apply completed transition and persist status.json."""
        try:
            status = self._read_status_strict()
            new_status = campaign_state.apply_transition(
                status=status,
                new_state="completed",
                timestamp=_utcnow(),
            )
            enriched = self._build_status_with_identity(new_status)
            self._write_status(enriched)
        except Exception as exc:
            # Completed transition failure is logged but run was already successful
            print(f"[run-quality] Warning: status transition to 'completed' failed: {exc}", file=sys.stderr)

    def _transition_status_failed(
        self, harness_rc: int, reason: Optional[str] = None
    ) -> None:
        """Transition to failed and record the exit code in the history entry."""
        try:
            status = self._read_status_strict()
            ts = _utcnow()
            new_status = campaign_state.apply_transition(
                status=status,
                new_state="failed",
                timestamp=ts,
                lifecycle="invalid",
            )
            # Record the harness exit code in the LAST history entry (the 'failed' one).
            # The result-status schema allows exit_code and note in history entries.
            # It does NOT allow top-level harness_exit_code (additionalProperties: false).
            if new_status["history"]:
                last_entry = new_status["history"][-1]
                last_entry["exit_code"] = harness_rc
                if reason:
                    last_entry["note"] = reason
            enriched = self._build_status_with_identity(new_status)
            self._write_status(enriched)
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

    # Resolve repository root
    suite_path = pathlib.Path(args.suite).resolve()
    repo = pathlib.Path(args.repo).resolve() if args.repo else suite_path.parent.parent

    # Resolve run directory
    if args.run_dir is not None:
        run_dir = pathlib.Path(args.run_dir).resolve()

        # -- Containment check: run_dir must reside inside the repository --
        try:
            run_dir.relative_to(repo)
        except ValueError:
            print(
                f"ERROR: --run-dir {run_dir} resolves outside repository {repo}. "
                "Run directories must reside inside the repository root for safety.",
                file=sys.stderr,
            )
            return 3

    else:
        # Derive from repo/suite/adapter — use create_campaign for transactional setup
        # In dry-run mode, use a temporary directory to avoid mutating the repo
        import yaml
        with open(suite_path) as fh:
            suite = yaml.safe_load(fh)
        with open(pathlib.Path(args.adapter).resolve()) as fh:
            adapter = yaml.safe_load(fh)

        suite_id = suite.get("suite_id", "warpcore-v1")
        model_slug = (adapter.get("model") or {}).get("slug", "unknown")
        run_id = args.run_id or datetime.now(tz=timezone.utc).strftime("run-%Y-%m-%dT%H-%M-%S")

        run_dir = (
            repo / "results" / model_slug / "runs"
            / suite_id / args.benchmark / run_id
        )

        # -- Containment check (always, even for derived dirs) --
        try:
            run_dir.relative_to(repo)
        except ValueError:
            print(
                f"ERROR: Derived run directory {run_dir} would be outside repository {repo}. "
                "Check that --repo and --suite are consistent.",
                file=sys.stderr,
            )
            return 3

        if args.dry_run:
            # Dry-run without explicit run dir: use a temporary directory so
            # no artifacts are created inside the repository.
            import tempfile
            _tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="run-quality-dryrun-"))
            run_dir = _tmp_dir
            # Write a minimal valid status for dry-run command generation
            status = {
                "schema_version": 1,
                "run_id": run_id,
                "suite_id": suite_id,
                "execution_state": "preflight_passed",
                "lifecycle": "current",
                "history": [
                    {"state": "planned", "timestamp": _utcnow()},
                    {"state": "preflight_passed", "timestamp": _utcnow()},
                ],
            }
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "status.json").write_text(json.dumps(status))
        else:
            # Live run: run directory must already exist (created by create_campaign)
            if not run_dir.exists():
                print(
                    f"ERROR: Run directory {run_dir} does not exist. "
                    "Create it first with create_campaign, then re-run with --run-dir.",
                    file=sys.stderr,
                )
                return 3

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
