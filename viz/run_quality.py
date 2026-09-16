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
- Points output at normalized run directory's raw/ subdirectory.
- CLI does NOT accept overrides for task, ceiling, sampling, scoring, datasets, or instances.
- Lifecycle order: planned -> QualityPreflightGate -> preflight_passed -> running -> harness.
- Refuses long launch outside /usr/bin/screen except in --dry-run or explicit test mode.
- Uses dependency-injected process execution.
- Does not duplicate preflight_serving.py, check_output_budget.py, or timeout arithmetic.
- Writes exact argv to command.txt using shell-safe quoting.
- Captures harness stdout+stderr in run.log (run_dir/run.log).
- On success: verifies required raw files (aggregate result + nonempty samples) before writing DONE.
  DONE is written only after completed status is schema-validated and written successfully.
- On nonzero harness exit: does NOT write DONE; records exit_code in history entry;
  transitions to failed. Does NOT add top-level harness_exit_code (schema forbids it).
- State transitions are fatal: missing/invalid status.json or illegal transition aborts run.
  _read_status_strict validates against JSON schema, verifies run_id==run_dir.name,
  and verifies suite_id matches loaded suite.
- Noncanonical adapters are rejected at construction time (no allow_noncanonical_adapter bypass).
- Run directory must be exactly repo/results/<adapter-slug>/runs/<suite_id>/<bench>/<run_id>.
- Completion is atomic: schema-validate completed status, write status, then write DONE.
  If status write fails, return nonzero — never return success with running status.
- Evidence validator recursively finds artifacts under raw/ subdirectories.

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
        --prompt-tokens gsm8k=500 \\
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
        --run-id run-2026-09-15T12-00-00 \\
        --prompt-tokens gsm8k=500

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
import subprocess
import sys
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

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
    validate_adapter_campaign_ready,
)

# ---------------------------------------------------------------------------
# Schemas dir — always the real repo schemas, regardless of run-dir layout repo
# ---------------------------------------------------------------------------

_SCHEMAS_DIR = _REPO_DIR / "suite" / "schemas"
_STATUS_SCHEMA = _SCHEMAS_DIR / "result-status.schema.json"
_ADAPTER_SCHEMA = _SCHEMAS_DIR / "adapter.schema.json"

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
    return bool(os.environ.get("STY", ""))


# ---------------------------------------------------------------------------
# Evidence verification (recursive)
# ---------------------------------------------------------------------------

_AGGREGATE_RESULT_RE = re.compile(r"^results_.*\.json$")
_SAMPLES_JSONL_GZ_RE = re.compile(r"^samples_.*\.jsonl\.gz$")


def _verify_required_evidence(run_dir: pathlib.Path) -> List[str]:
    """Return a list of error strings if required harness artifacts are missing.

    Required minimum (subset of full required_evidence from suite):
      - At least one aggregate result JSON (results_*.json) that is nonempty
      - At least one nonempty valid gzip samples file (samples_*.jsonl.gz)

    Searches recursively under raw/ to find artifacts in subdirectories
    (lm-eval 0.4.12 writes into named subdirs under the output path).

    Returns [] on success, list of errors on failure.
    """
    import gzip as _gzip

    raw_dir = run_dir / "raw"
    errors: List[str] = []

    if not raw_dir.exists():
        errors.append(f"raw/ directory does not exist at {raw_dir}")
        return errors

    # Recursive search for artifacts under raw/
    all_files = list(raw_dir.rglob("*"))
    if not any(f.is_file() for f in all_files):
        errors.append(f"raw/ directory is empty at {raw_dir} — no harness artifacts produced")
        return errors

    # Find aggregate result JSONs (nonempty)
    aggregate_results = [
        f for f in all_files
        if f.is_file() and _AGGREGATE_RESULT_RE.match(f.name)
    ]
    nonempty_results = [f for f in aggregate_results if f.stat().st_size > 0]
    if not nonempty_results:
        if aggregate_results:
            errors.append(
                f"All results_*.json files in {raw_dir} (searched recursively) are empty. "
                "lm-eval must produce a nonempty aggregate result JSON."
            )
        else:
            errors.append(
                f"No aggregate result file (results_*.json) found under {raw_dir} "
                "(searched recursively). lm-eval must produce at least one results JSON."
            )

    # Find samples JSONL.GZ files — must be valid nonempty gzip
    samples_gz = [
        f for f in all_files
        if f.is_file() and _SAMPLES_JSONL_GZ_RE.match(f.name)
    ]
    if not samples_gz:
        errors.append(
            f"No samples file (samples_*.jsonl.gz) found under {raw_dir} "
            "(searched recursively). lm-eval must produce compressed samples (--log_samples required)."
        )
    else:
        valid_nonempty = []
        for gz_path in samples_gz:
            try:
                with _gzip.open(gz_path, "rb") as fh:
                    content = fh.read(1)
                if len(content) > 0:
                    valid_nonempty.append(gz_path)
                else:
                    errors.append(
                        f"samples file {gz_path.name} is valid gzip but contains no data. "
                        "Evidence must be a nonempty archive."
                    )
            except Exception:
                errors.append(
                    f"samples file {gz_path.name} is corrupt or not valid gzip. "
                    "Evidence must be a valid nonempty gzip archive."
                )

    return errors


# ---------------------------------------------------------------------------
# Task name extraction
# ---------------------------------------------------------------------------


def _read_task_name_from_yaml(task_file_path: pathlib.Path) -> str:
    """Read the registered 'task:' name from a lm-eval task YAML file."""
    import yaml as _yaml

    class _PermissiveLoader(_yaml.SafeLoader):
        pass

    _PermissiveLoader.add_multi_constructor(
        "",
        lambda loader, tag_suffix, node: None,
    )

    raw = pathlib.Path(task_file_path).read_text(encoding="utf-8")
    data = _yaml.load(raw, Loader=_PermissiveLoader)  # noqa: S506
    if not isinstance(data, dict):
        raise ValueError(
            f"Task YAML {task_file_path} did not parse to a mapping; got {type(data).__name__}."
        )
    task_name = data.get("task", "") or ""
    if not task_name:
        raise ValueError(
            f"Task YAML {task_file_path} does not define a 'task:' field."
        )
    return str(task_name)


# ---------------------------------------------------------------------------
# Normalized run directory validation
# ---------------------------------------------------------------------------


def _validate_run_dir_identity(
    run_dir: pathlib.Path,
    repo: pathlib.Path,
    adapter_slug: str,
    suite_id: str,
    benchmark: str,
    run_id: str,
) -> List[str]:
    """Validate that run_dir exactly equals repo/results/<slug>/runs/<suite_id>/<bench>/<run_id>.

    Returns [] on success, list of error strings on failure.
    """
    expected = (
        repo / "results" / adapter_slug / "runs" / suite_id / benchmark / run_id
    ).resolve()
    actual = run_dir.resolve()
    if actual != expected:
        return [
            f"Run directory identity mismatch: "
            f"expected {expected}, got {actual}. "
            f"Run directories must match the exact normalized layout: "
            f"<repo>/results/<adapter-slug>/runs/<suite-id>/<benchmark>/<run-id>."
        ]
    return []


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
        Path to adapter YAML.
    benchmark : str
        Benchmark key (e.g. "gsm8k").
    endpoint : str
        OpenAI-compatible base URL ending in /v1.
    throughput : float
        Measured aggregate token throughput (tok/s).
    concurrency : int
        Number of parallel lm-eval workers.
    timeout : float
        lm-eval --timeout value in seconds.
    run_dir : pathlib.Path
        Normalized run directory.
    repo : pathlib.Path or None
        Repository root for run directory layout. Defaults to parent of suite file.
    prompt_token_maxima : dict or None
        Measured tokenized prompt maxima keyed by benchmark.
    dry_run : bool
        If True, build and print command without invoking preflight or harness.
    allow_no_screen : bool
        Bypass the screen guard (tests and special ops only).
    preflight_runner : callable or None
        Injected callable(model_id) -> int.
    harness_runner : callable or None
        Injected callable(cmd: list, **kw) -> int.
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
        repo: Optional[pathlib.Path] = None,
        prompt_token_maxima: Optional[Dict[str, int]] = None,
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
        self.prompt_token_maxima = prompt_token_maxima

        # _repo: used ONLY for run directory layout (not for schema lookup)
        if repo is not None:
            self._repo = pathlib.Path(repo).resolve()
        else:
            self._repo = self.suite_path.parent.parent

        # Load suite
        import yaml
        with open(self.suite_path) as fh:
            self._suite: dict = yaml.safe_load(fh)

        # Load adapter
        with open(self.adapter_path) as fh:
            self._adapter: dict = yaml.safe_load(fh)

        # -- Adapter schema validation --
        # Always uses _ADAPTER_SCHEMA from the real repo (module-level constant),
        # not from self._repo (which may be a temp dir in tests).
        if _ADAPTER_SCHEMA.exists():
            schema_errors = validate_json(self._adapter, _ADAPTER_SCHEMA)
            if schema_errors:
                raise ValueError(
                    f"Adapter schema validation failed for {self.adapter_path}:\n"
                    + "\n".join(schema_errors)
                )
        else:
            raise ValueError(
                f"Adapter schema not found at {_ADAPTER_SCHEMA}. "
                "Cannot validate adapter without schema."
            )

        # -- Campaign-readiness validation (noncanonical adapters blocked) --
        model_slug = (self._adapter.get("model") or {}).get("slug", "")
        readiness_errors = validate_adapter_campaign_ready(
            self._adapter,
            model_slug,
            suite=self._suite,
            prompt_token_maxima=self.prompt_token_maxima,
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

        # Suite-level IDs
        self._suite_id: str = self._suite.get("suite_id", "warpcore-v1")
        self._run_id: str = self.run_dir.name
        self._model_slug: str = model_slug

        # Injected runners
        self._preflight_runner = preflight_runner
        self._harness_runner = harness_runner

    # ------------------------------------------------------------------
    # Command construction
    # ------------------------------------------------------------------

    def build_command(self) -> List[str]:
        """Build the canonical lm-eval argv list from suite + adapter settings."""
        bench = self._bench_cfg

        task_file = bench.get("task_file", "")
        if task_file:
            # Resolve from the actual task file location (always in real repo)
            canonical_task_path = (_REPO_DIR / task_file).resolve()
            task_dir = str(canonical_task_path.parent)
            task_name = _read_task_name_from_yaml(canonical_task_path)
        else:
            task_dir = ""
            task_name = bench.get("task_name", self.benchmark)

        generation_ceiling: int = bench.get("generation_ceiling", 8192)
        retry_policy = bench.get("retry_policy", {})
        max_retries: int = retry_policy.get("max_retries", 0)
        sampling = bench.get("sampling", {})
        temperature = sampling.get("temperature", 0)
        do_sample = sampling.get("do_sample", False)

        model_id = (self._adapter.get("model") or {}).get("id", "")

        gen_kwargs = (
            f"max_gen_toks={generation_ceiling},"
            f"temperature={temperature},"
            f"do_sample={str(do_sample).lower()}"
        )
        model_args = (
            f"base_url={self.endpoint},"
            f"model={model_id},"
            f"num_concurrent={self.concurrency},"
            f"max_retries={max_retries},"
            f"tokenized_requests=False"
        )

        output_path = str(self.run_dir / "raw")

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

        if task_dir:
            cmd.extend(["--include_path", task_dir])

        return cmd

    # ------------------------------------------------------------------
    # Core run logic
    # ------------------------------------------------------------------

    def run(self) -> int:
        """Execute the full quality run lifecycle.

        Lifecycle order (for live runs):
          1. Read and strict-validate status.json (schema + run_id + suite_id).
          2. Validate normalized run directory identity.
          3. Accept only 'planned' state.
          4. Build command, write command.txt.
          5. Run QualityPreflightGate (while state is 'planned').
          6. On preflight pass: transition planned -> preflight_passed.
          7. Transition preflight_passed -> running.
          8. Execute harness (stdout+stderr -> run.log).
          9. On success: verify evidence, schema-validate completed status, write status,
             then write DONE. If any step fails, return nonzero.
         10. On failure: transition to failed (best-effort), return nonzero.

        Returns an exit code (0 = success, nonzero = failure).
        """
        # --- Dry-run: validate/build/print only; persist nothing ---
        if self.dry_run:
            cmd = self.build_command()
            print("[dry-run] Command:")
            print(" ".join(shlex.quote(a) for a in cmd))
            return 0

        # --- Screen guard ---
        if not self.allow_no_screen and not _is_under_screen():
            print(
                "ERROR: Long quality runs must be launched inside /usr/bin/screen. "
                "Start a screen session first:\n  screen -S quality-run\n"
                "Or pass --allow-no-screen to bypass (tests/special ops only).",
                file=sys.stderr,
            )
            return 2

        # --- Read and strict-validate status.json ---
        try:
            status = self._read_status_strict()
        except Exception as exc:
            print(
                f"[run-quality] FATAL: Cannot read or validate status.json: {exc}",
                file=sys.stderr,
            )
            return 1

        # --- Validate normalized run directory identity ---
        identity_errors = _validate_run_dir_identity(
            run_dir=self.run_dir,
            repo=self._repo,
            adapter_slug=self._model_slug,
            suite_id=self._suite_id,
            benchmark=self.benchmark,
            run_id=self._run_id,
        )
        if identity_errors:
            for err in identity_errors:
                print(f"[run-quality] FATAL: {err}", file=sys.stderr)
            return 1

        # --- Accept only 'planned' state ---
        current_state = status.get("execution_state", "")
        if current_state != "planned":
            print(
                f"[run-quality] FATAL: Run directory is in state {current_state!r}; "
                "expected 'planned'.",
                file=sys.stderr,
            )
            return 1

        # --- Build command and write command.txt ---
        cmd = self.build_command()
        self._write_command_txt(cmd)

        # --- Run preflight gate while status is 'planned' ---
        model_id = (self._adapter.get("model") or {}).get("id", "")
        generation_ceiling: int = self._bench_cfg.get("generation_ceiling", 8192)

        preflight_rc = self._run_preflight(
            model_id=model_id,
            generation_ceiling=generation_ceiling,
        )
        if preflight_rc != 0:
            print(
                f"[run-quality] Preflight gate failed (exit {preflight_rc}).",
                file=sys.stderr,
            )
            return preflight_rc

        # --- Transition planned -> preflight_passed ---
        try:
            status = self._read_status_strict()
            new_status = campaign_state.apply_transition(
                status=status,
                new_state="preflight_passed",
                timestamp=_utcnow(),
            )
            self._write_status(new_status)
        except Exception as exc:
            print(
                f"[run-quality] FATAL: Lifecycle transition to 'preflight_passed' failed: {exc}",
                file=sys.stderr,
            )
            return 1

        # --- Transition preflight_passed -> running ---
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
                f"[run-quality] FATAL: Lifecycle transition to 'running' failed: {exc}",
                file=sys.stderr,
            )
            return 1

        # --- Execute harness (stdout+stderr captured to run.log) ---
        harness_rc = self._execute_harness(cmd)

        # --- Post-run: verify evidence, write DONE or record failure ---
        if harness_rc == 0:
            evidence_errors = _verify_required_evidence(self.run_dir)
            if not evidence_errors:
                # Completion atomicity: schema-validate completed status, write it,
                # then write DONE.
                rc = self._transition_status_completed_atomic()
                if rc != 0:
                    return rc
                try:
                    (self.run_dir / "DONE").write_text("completed\n")
                except OSError as exc:
                    print(
                        f"[run-quality] FATAL: Cannot write DONE sentinel: {exc}",
                        file=sys.stderr,
                    )
                    return 1
                return 0
            else:
                print(
                    "ERROR: Harness exited 0 but required evidence is missing:\n"
                    + "\n".join(f"  {e}" for e in evidence_errors)
                    + "\nDONE not written.",
                    file=sys.stderr,
                )
                if not self._transition_status_failed(harness_rc=0, reason="evidence_missing"):
                    return 1
                return 1
        else:
            if not self._transition_status_failed(harness_rc=harness_rc):
                return 1
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
        """Run the QualityPreflightGate; return its exit code."""
        if self._preflight_runner is not None:
            return int(self._preflight_runner(model_id))

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
        """Execute the lm-eval harness; return its exit code.

        Default: captures stdout+stderr to run.log in the run directory.
        Injected harness_runner receives (cmd, **kw) and must return an int exit code.
        """
        if self._harness_runner is not None:
            return int(self._harness_runner(cmd))

        # Default subprocess execution: redirect stdout+stderr to run.log
        run_log = self.run_dir / "run.log"
        with open(run_log, "w", encoding="utf-8") as log_fh:
            result = subprocess.run(cmd, stdout=log_fh, stderr=log_fh, check=False)
        return result.returncode

    def _read_status_strict(self) -> dict:
        """Read and return status.json; raise if missing, unreadable, or invalid.

        Validates:
          1. status.json exists and is valid JSON.
          2. The document passes the result-status.schema.json JSON schema.
          3. status.run_id == run_dir.name (last path component).
          4. status.suite_id == the loaded suite's suite_id.

        Raises FileNotFoundError if status.json does not exist.
        Raises ValueError if any validation fails.
        Never adds missing identity fields — fails closed.
        """
        status_path = self.run_dir / "status.json"
        if not status_path.exists():
            raise FileNotFoundError(
                f"status.json not found in run directory {self.run_dir}. "
                "Run directory must be initialized by create_campaign before launching."
            )
        try:
            status = json.loads(status_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"status.json in {self.run_dir} is not valid JSON: {exc}"
            ) from exc

        # Schema validation (uses real repo schema always)
        if _STATUS_SCHEMA.exists():
            schema_errors = validate_json(status, _STATUS_SCHEMA)
            if schema_errors:
                raise ValueError(
                    f"status.json in {self.run_dir} fails schema validation:\n"
                    + "\n".join(schema_errors)
                )
        else:
            raise ValueError(
                f"result-status schema not found at {_STATUS_SCHEMA}"
            )

        # run_id must match run_dir.name — fail closed, never add missing identity
        expected_run_id = self.run_dir.name
        actual_run_id = status.get("run_id", "")
        if actual_run_id != expected_run_id:
            raise ValueError(
                f"status.json run_id={actual_run_id!r} does not match "
                f"run_dir.name={expected_run_id!r}."
            )

        # suite_id must match loaded suite
        expected_suite_id = self._suite_id
        actual_suite_id = status.get("suite_id", "")
        if actual_suite_id != expected_suite_id:
            raise ValueError(
                f"status.json suite_id={actual_suite_id!r} does not match "
                f"loaded suite suite_id={expected_suite_id!r}."
            )

        return status

    def _write_status(self, status: dict) -> None:
        campaign_state.write_status(
            dest=self.run_dir / "status.json",
            status=status,
            run_dir=self.run_dir,
        )

    def _build_completed_status(self, status: dict) -> dict:
        """Build and schema-validate a completed status dict.

        Raises ValueError if schema validation fails (atomicity guard).
        """
        new_status = campaign_state.apply_transition(
            status=status,
            new_state="completed",
            timestamp=_utcnow(),
        )

        # Schema-validate before writing (atomicity)
        if _STATUS_SCHEMA.exists():
            schema_errors = validate_json(new_status, _STATUS_SCHEMA)
            if schema_errors:
                raise ValueError(
                    f"Completed status failed schema validation:\n"
                    + "\n".join(schema_errors)
                )
        return new_status

    def _transition_status_completed_atomic(self) -> int:
        """Apply completed transition atomically: validate schema then write.

        Returns 0 on success, 1 on failure.
        Never returns 0 if the status write fails.
        """
        try:
            status = self._read_status_strict()
            completed_status = self._build_completed_status(status)
            self._write_status(completed_status)
            return 0
        except Exception as exc:
            print(
                f"[run-quality] FATAL: Completed status write failed: {exc}. "
                "DONE not written — returning nonzero.",
                file=sys.stderr,
            )
            return 1

    def _transition_status_failed(
        self, harness_rc: int, reason: Optional[str] = None
    ) -> bool:
        """Transition to failed; return whether the durable state write succeeded."""
        try:
            status = self._read_status_strict()
            ts = _utcnow()
            new_status = campaign_state.apply_transition(
                status=status,
                new_state="failed",
                timestamp=ts,
                lifecycle="invalid",
            )
            # Record the harness exit code in the last history entry.
            # Schema allows exit_code and note in history entries.
            # Does NOT add top-level harness_exit_code (schema forbids it).
            if new_status["history"]:
                last_entry = new_status["history"][-1]
                last_entry["exit_code"] = harness_rc
                if reason:
                    last_entry["note"] = reason
            self._write_status(new_status)
            return True
        except Exception as exc:
            print(
                f"[run-quality] FATAL: status transition to 'failed' failed: {exc}",
                file=sys.stderr,
            )
            return False


# ---------------------------------------------------------------------------
# CLI prompt-tokens parser
# ---------------------------------------------------------------------------


def _parse_prompt_tokens(value: str) -> Dict[str, int]:
    """Parse --prompt-tokens: 'gsm8k=500,ifeval=2000' -> {'gsm8k': 500, 'ifeval': 2000}."""
    result: Dict[str, int] = {}
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise argparse.ArgumentTypeError(
                f"Invalid --prompt-tokens format {part!r}. "
                "Expected: benchmark=tokens (e.g. gsm8k=500,ifeval=2000)"
            )
        bench, _, tok_str = part.partition("=")
        bench = bench.strip()
        tok_str = tok_str.strip()
        try:
            tokens = int(tok_str)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid token count {tok_str!r} for benchmark {bench!r}."
            )
        if tokens < 0:
            raise argparse.ArgumentTypeError(
                f"Token count must be non-negative for benchmark {bench!r}; got {tokens}."
            )
        result[bench] = tokens
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Contract-aware quality runner for warpcore-v1 benchmarks."
        )
    )

    ap.add_argument("--suite", required=True, type=pathlib.Path)
    ap.add_argument("--adapter", required=True, type=pathlib.Path)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--throughput", required=True, type=float)
    ap.add_argument("--concurrency", required=True, type=int)
    ap.add_argument("--timeout", required=True, type=float)
    ap.add_argument("--prompt-tokens", type=str, default=None,
                    help="Measured prompt maxima: 'bench=N,...' (e.g. gsm8k=500,ifeval=2000).")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--run-dir", type=pathlib.Path, default=None)
    ap.add_argument("--repo", type=pathlib.Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-no-screen", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="Resume an existing campaign run directory.")

    # Detect and reject forbidden override flags
    if argv is not None:
        args_to_check = argv
    else:
        args_to_check = sys.argv[1:]

    for flag in FORBIDDEN_OVERRIDE_FLAGS:
        if flag in args_to_check:
            ap.error(
                f"'{flag}' is a suite-owned experiment variable and cannot be overridden."
            )

    args = ap.parse_args(argv)

    # Parse prompt_token_maxima
    prompt_token_maxima: Optional[Dict[str, int]] = None
    if args.prompt_tokens is not None:
        try:
            prompt_token_maxima = _parse_prompt_tokens(args.prompt_tokens)
        except argparse.ArgumentTypeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 3

    # Resolve repository root
    suite_path = pathlib.Path(args.suite).resolve()
    repo = pathlib.Path(args.repo).resolve() if args.repo else suite_path.parent.parent

    # Resolve run directory. Live execution always passes through
    # create_campaign(..., resume=...), including an explicit --run-dir; this
    # prevents hand-built or stale manifests from bypassing exact identity
    # validation. Dry-run only derives/validates a path and performs no writes.
    if args.run_dir is not None and args.dry_run:
        run_dir = pathlib.Path(args.run_dir).resolve()
        try:
            run_dir.relative_to(repo)
        except ValueError:
            print(
                f"ERROR: --run-dir {run_dir} resolves outside repository {repo}.",
                file=sys.stderr,
            )
            return 3
    else:
        import yaml
        with open(suite_path) as fh:
            suite = yaml.safe_load(fh)
        with open(pathlib.Path(args.adapter).resolve()) as fh:
            adapter = yaml.safe_load(fh)

        suite_id = suite.get("suite_id", "warpcore-v1")
        model_slug = (adapter.get("model") or {}).get("slug", "unknown")
        if args.run_dir is not None:
            requested_run_dir = pathlib.Path(args.run_dir).resolve()
            try:
                requested_run_dir.relative_to(repo)
            except ValueError:
                print(
                    f"ERROR: --run-dir {requested_run_dir} resolves outside repository {repo}.",
                    file=sys.stderr,
                )
                return 3
            run_id = requested_run_dir.name
            expected = (
                repo / "results" / model_slug / "runs"
                / suite_id / args.benchmark / run_id
            ).resolve()
            if requested_run_dir != expected:
                print(
                    f"ERROR: --run-dir must equal normalized campaign path {expected}; "
                    f"got {requested_run_dir}.",
                    file=sys.stderr,
                )
                return 3
        else:
            run_id = args.run_id or datetime.now(tz=timezone.utc).strftime("run-%Y-%m-%dT%H-%M-%S")

        if args.dry_run:
            # Dry-run derives the path only. It must not create campaign state,
            # command files, or even parent directories.
            run_dir = (
                repo / "results" / model_slug / "runs"
                / suite_id / args.benchmark / run_id
            )
            try:
                run_dir.relative_to(repo)
            except ValueError:
                print(
                    f"ERROR: Derived run directory {run_dir} would be outside repository {repo}.",
                    file=sys.stderr,
                )
                return 3
        else:
            try:
                import create_campaign as cc_mod
                run_dir = cc_mod.create_campaign(
                    repo=repo,
                    suite_path=suite_path,
                    adapter_path=pathlib.Path(args.adapter).resolve(),
                    benchmark=args.benchmark,
                    run_id=run_id,
                    resume=(args.resume or args.run_dir is not None),
                    prompt_token_maxima=prompt_token_maxima,
                )
            except Exception as exc:
                print(f"ERROR: create_campaign failed: {exc}", file=sys.stderr)
                return 3

        # Containment check
        try:
            run_dir.relative_to(repo)
        except ValueError:
            print(
                f"ERROR: Run directory {run_dir} resolves outside repository {repo}.",
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
            repo=repo,
            prompt_token_maxima=prompt_token_maxima,
            dry_run=args.dry_run,
            allow_no_screen=args.allow_no_screen,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
