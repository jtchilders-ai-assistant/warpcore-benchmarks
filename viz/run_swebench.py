#!/usr/bin/env python3
"""viz/run_swebench.py — Contract-aware SWE-bench runner for warpcore-v1.

Implements the full generation+grading lifecycle for the SWE-bench Verified
benchmark under the warpcore-v1 measurement contract.

CONTRACT (docs/superpowers/specs/2026-09-15-warpcore-v1-apples-to-apples-design.md):
- Uses exactly the 100 frozen instance IDs in suite/swebench/instances-seed42-n100.json
  and verifies their SHA-256 hash from the suite at construction.
- Refuses to launch a canonical campaign without a valid SWE-bench qualification
  (design §4.4.1, viz/swebench_qualification.py). The gate runs before any campaign
  state is created, in both main() and SwebenchRunner.run(); a dry run reports the
  verdict and stays side-effect-free. --qualification-run executes the 20 suite-owned
  qualification instances instead, producing the evidence a record is sealed from:
  that mode is not gated and writes no campaign state.
- Preflight gates (all must pass before generation):
    1. x86 Docker host check (SWE-bench test containers are x86).
    2. Image cache check via swebench_preflight.py wrapping.
    3. Live model identity from /v1/models (exact match).
    4. Scaffold hash matches suite declaration.
    5. Submit protocol intact (git add -A && git diff --cached present in scaffold).
- Injects ONLY adapter model_id (as "hosted_vllm/<model_id>"), endpoint, and api_key
  into the frozen scaffold; never overrides step_limit, cost_limit, environment.timeout,
  pull_timeout, model.model_kwargs.temperature, model.model_kwargs.max_tokens, or submit
  protocol.
- Creates normalized stable run dir via create_campaign (transactional); run_dir must be
  repo/results/<adapter-slug>/runs/<suite_id>/swebench/<run_id>.
- Refuses live launch outside /usr/bin/screen (exit 2) except in dry_run or allow_no_screen.
- Explicit resume only with exact IDs/hashes/identity via create_campaign(resume=True).
- Separates generation and grading as distinct execution phases in status history
  (history notes "generation" and "grading") without violating the result-status schema.
- Preserves trajectories (preds.json is never deleted or modified).
- Completion only after grader artifacts (grading_results.json) and all 100 terminal
  dispositions exist across all disposition categories.
- Fails closed on every state/evidence write error (never returns EXIT_SUCCESS with
  running/planned status or missing evidence).
- Rejects noncanonical adapters at construction.
- Does NOT launch live benchmark (generation_runner and grading_runner are injected).

CLI WRAPPERS (live defaults when no runner is injected)
-------------------------------------------------------
Generation: minisweagent.run.benchmarks.swebench (mini-swe-agent 2.4.6)
    --subset princeton-nlp/SWE-bench_Verified  HuggingFace dataset path
    --split test
    --filter <anchored-regex>   anchored OR-regex matching all 100 frozen IDs exactly
    -c <scaffold-config-yaml>
    -w <workers>
    -o <raw_dir>

Real mini-swe-agent 2.4.6 output layout (under -o <raw_dir>):
    preds.json                            predictions dict keyed by instance_id
    <instance_id>/<instance_id>.traj.json trajectory (only when agent was created;
                                          absent on pre-agent infrastructure failures)
    exit_statuses_<timestamp>.yaml        progress YAML: {instances_by_exit_status: {status: [ids]}}

Post-generation normalization (run_swebench.py after generation returns 0):
    Reads exit_statuses_<timestamp>.yaml (latest by mtime when multiple exist).
    Copies all produced <id>/<id>.traj.json to raw/trajectories/<id>.traj
    (source files are preserved; trajectories/ is additive, never deletes source).
    Derives exit_statuses.json from the YAML (flat dict: instance_id -> exit_status string).
    Rejects missing trajectories, missing/extra/duplicate IDs, and never invents evidence.

Grading: swebench.harness.run_evaluation
    -d princeton-nlp/SWE-bench_Verified
    -s test
    -i <instance_id> ...        all frozen IDs (space-separated)
    -p <preds_path>
    -id <run_id>
    --report_dir <raw_dir>

The harness emits <model>.<run_id>.json under --report_dir (default: CWD).
After grading, this script locates the exact report file, validates it against
official schema-v2 keys (resolved_ids, unresolved_ids, empty_patch_ids, error_ids,
incomplete_ids) with strict overlap and foreign-ID checks, and normalizes it to
grading_results.json with disjoint complete categories. Never assigns absent IDs.

EXIT CODES
----------
    EXIT_SUCCESS (0)    Success: DONE written, 100 dispositions present.
    EXIT_DEFECT (1)     Preflight defect, generation/grading failure, or evidence error.
    EXIT_INCONCLUSIVE (2)  Screen guard or preflight inconclusive.
    EXIT_CONFIG (3)     Construction, validation, or configuration error.

USAGE
-----
    # Dry-run — inspect scaffold config and instance set without any side effects:
    python3 viz/run_swebench.py \\
        --suite suite/warpcore-v1.yaml \\
        --adapter adapters/qwen3.6-35b-a3b.yaml \\
        --endpoint http://csi370295.alcf.anl.gov:8000/v1 \\
        --run-id run-2026-09-15T12-00-00 \\
        --dry-run

    # Live run (must be inside /usr/bin/screen on Mac mini):
    screen -S swebench-run
    python3 viz/run_swebench.py \\
        --suite suite/warpcore-v1.yaml \\
        --adapter adapters/qwen3.6-35b-a3b.yaml \\
        --endpoint http://csi370295.alcf.anl.gov:8000/v1 \\
        --run-id run-2026-09-15T12-00-00

make run-swebench SUITE=suite/warpcore-v1.yaml ADAPTER=adapters/qwen3.6-35b-a3b.yaml \\
    ENDPOINT=http://host:8000/v1 [RUN_ID=run-xxx] [DRY_RUN=1] [ALLOW_NO_SCREEN=1]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
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

import campaign_state  # noqa: E402
import swebench_qualification  # noqa: E402
from contract import (  # noqa: E402
    load_yaml,
    sha256_file,
    validate_json,
    validate_adapter_campaign_ready,
)

# ---------------------------------------------------------------------------
# Schemas dir — always the real repo schemas
# ---------------------------------------------------------------------------

_SCHEMAS_DIR = _REPO_DIR / "suite" / "schemas"
_STATUS_SCHEMA = _SCHEMAS_DIR / "result-status.schema.json"
_ADAPTER_SCHEMA = _SCHEMAS_DIR / "adapter.schema.json"

# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------

EXIT_SUCCESS = 0
EXIT_DEFECT = 1
EXIT_INCONCLUSIVE = 2
EXIT_CONFIG = 3

# SWE-bench benchmark key in suite
_SWEBENCH_BENCH = "swebench"


# ---------------------------------------------------------------------------
# UTC timestamp helper
# ---------------------------------------------------------------------------


def _utcnow() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Screen guard
# ---------------------------------------------------------------------------


def _is_under_screen() -> bool:
    """Return True when running inside a GNU screen session."""
    return bool(os.environ.get("STY", ""))


# ---------------------------------------------------------------------------
# Evidence verification helpers
# ---------------------------------------------------------------------------


def _normalize_generation_artifacts(
    raw_dir: pathlib.Path,
    expected_instance_ids: List[str],
) -> List[str]:
    """Normalize real mini-swe-agent 2.4.6 output into the stable raw/ layout.

    Real mini-swe-agent 2.4.6 emits:
      raw_dir/preds.json                            (always)
      raw_dir/<id>/<id>.traj.json                   (only when agent ran; absent for pre-agent failures)
      raw_dir/exit_statuses_<timestamp>.yaml        (progress YAML, possibly multiple files)

    Normalization (additive — never deletes source files):
      1. Reads the latest exit_statuses_<timestamp>.yaml by mtime.
      2. Inverts {status: [ids]} to a flat {id: status} dict.
      3. Verifies all expected_instance_ids appear in the YAML (fail closed if any missing or extra).
      4. Copies each <id>/<id>.traj.json to raw_dir/trajectories/<id>.traj.
         Source files are preserved. Missing per-instance trajectories fail closed.
      5. Writes raw_dir/exit_statuses.json (stable path for downstream tools).

    Returns [] on success, list of error strings on failure.
    Fails closed: returns errors without writing any normalized file if inputs are incomplete.
    """
    errors: List[str] = []

    # --- 1. Find the latest exit_statuses_<timestamp>.yaml ---
    yaml_files = sorted(raw_dir.glob("exit_statuses_*.yaml"), key=lambda p: p.stat().st_mtime)
    if not yaml_files:
        errors.append(
            f"No exit_statuses_<timestamp>.yaml found in {raw_dir}. "
            "mini-swe-agent 2.4.6 must produce this file. "
            "Expected: exit_statuses_<float-timestamp>.yaml"
        )
        return errors

    latest_yaml = yaml_files[-1]
    try:
        import yaml as _yaml
        yaml_data = _yaml.safe_load(latest_yaml.read_text())
    except Exception as exc:
        errors.append(f"Could not parse {latest_yaml.name}: {exc}")
        return errors

    if not isinstance(yaml_data, dict):
        errors.append(
            f"{latest_yaml.name} must be a YAML dict; got {type(yaml_data).__name__}."
        )
        return errors

    # instances_by_exit_status: {exit_status_str: [instance_id, ...]}
    instances_by_exit_status = yaml_data.get("instances_by_exit_status") or {}
    if not isinstance(instances_by_exit_status, dict):
        errors.append(
            f"{latest_yaml.name}: 'instances_by_exit_status' must be a dict; "
            f"got {type(instances_by_exit_status).__name__}."
        )
        return errors

    # --- 2. Invert to flat {id: status} ---
    id_to_status: dict = {}
    duplicate_ids: list = []
    for status_str, id_list in instances_by_exit_status.items():
        if not isinstance(id_list, list):
            errors.append(
                f"{latest_yaml.name}: instances_by_exit_status[{status_str!r}] "
                f"must be a list; got {type(id_list).__name__}."
            )
            return errors
        for iid in id_list:
            if iid in id_to_status:
                duplicate_ids.append(iid)
            id_to_status[iid] = str(status_str)

    if duplicate_ids:
        errors.append(
            f"{latest_yaml.name}: {len(duplicate_ids)} instance IDs appear in multiple "
            f"exit_status buckets (duplicate dispositions): "
            f"{sorted(duplicate_ids)[:5]}{'...' if len(duplicate_ids) > 5 else ''}. "
            "Each instance must appear in exactly one status bucket."
        )
        return errors

    # --- 3. Verify all expected IDs appear in YAML (exactly) ---
    expected_set = set(expected_instance_ids)
    yaml_set = set(id_to_status.keys())
    missing_from_yaml = expected_set - yaml_set
    extra_in_yaml = yaml_set - expected_set

    if missing_from_yaml:
        errors.append(
            f"{latest_yaml.name} is missing {len(missing_from_yaml)} of "
            f"{len(expected_instance_ids)} expected instance IDs. "
            f"Missing: {sorted(missing_from_yaml)[:5]}"
            f"{'...' if len(missing_from_yaml) > 5 else ''}. "
            "The YAML must contain exactly the frozen 100 IDs."
        )
    if extra_in_yaml:
        errors.append(
            f"{latest_yaml.name} contains {len(extra_in_yaml)} unexpected instance IDs "
            f"not in the frozen suite: "
            f"{sorted(extra_in_yaml)[:5]}"
            f"{'...' if len(extra_in_yaml) > 5 else ''}."
        )
    if errors:
        return errors

    # --- 4. Copy every trajectory into stable raw/trajectories/ ---
    # Trajectories are absent for pre-agent infrastructure failures (e.g. docker
    # pull timeout, vLLM 5xx). These IDs have exit statuses that indicate the
    # agent was never invoked; their absence is expected and must not fail
    # normalization. Only IDs with agent-level exit statuses (Submitted,
    # RepeatedFormatError, ContextWindowExceededError, LimitsExceeded, etc.)
    # are required to have trajectories.
    #
    # Infrastructure-failure exit statuses (no agent was ever created):
    _INFRA_FAILURE_STATUSES = frozenset({
        "TimeoutExpired",       # docker pull/test timeout, model never invoked
        "InternalServerError",  # vLLM 5xx (e.g. GB10 long-context wedge)
        # Generic test-mode names used in tests:
        "infrastructure_error",
        "infra_failure",
    })

    trajectories_dir = raw_dir / "trajectories"

    import shutil as _shutil
    traj_errors: list = []
    for iid in expected_instance_ids:
        src = raw_dir / iid / f"{iid}.traj.json"
        # Check if this ID had a pre-agent infrastructure failure
        exit_status = id_to_status.get(iid, "")
        if exit_status in _INFRA_FAILURE_STATUSES:
            # No trajectory expected — skip without error
            continue
        if not src.is_file() or src.stat().st_size == 0:
            traj_errors.append(f"Missing or empty trajectory for {iid}: {src}")

    if traj_errors:
        errors.extend(traj_errors)
        return errors

    try:
        trajectories_dir.mkdir(exist_ok=True)
        for iid in expected_instance_ids:
            # Skip infra-failure IDs — they have no trajectory to copy
            exit_status = id_to_status.get(iid, "")
            if exit_status in _INFRA_FAILURE_STATUSES:
                continue
            src = raw_dir / iid / f"{iid}.traj.json"
            dst = trajectories_dir / f"{iid}.traj"
            _shutil.copy2(str(src), str(dst))
    except OSError as exc:
        errors.append(f"Could not normalize trajectories: {exc}")
        return errors

    # --- 5. Write stable exit_statuses.json ---
    exit_statuses_json = raw_dir / "exit_statuses.json"
    try:
        exit_statuses_json.write_text(json.dumps(id_to_status, indent=2))
    except OSError as exc:
        errors.append(f"Could not write exit_statuses.json: {exc}")

    return errors


def _verify_generation_evidence(
    run_dir: pathlib.Path,
    expected_instance_ids: Optional[List[str]] = None,
) -> List[str]:
    """Verify generation phase artifacts exist and are complete.

    Checks (post-normalization stable layout):
    - raw/ directory exists
    - preds.json is nonempty and keys match expected_instance_ids (if given)
    - exit_statuses.json exists (written by _normalize_generation_artifacts)
      and keys match expected_instance_ids (if given)
    - trajectories/ directory exists (written by _normalize_generation_artifacts)
    - every expected ID has a nonempty raw/trajectories/<id>.traj
    - run.log exists and is nonempty

    Returns [] on success, list of error strings on failure.
    """
    errors: List[str] = []
    raw_dir = run_dir / "raw"

    if not raw_dir.exists():
        errors.append(f"raw/ directory does not exist at {raw_dir}")
        return errors

    preds = raw_dir / "preds.json"
    if not preds.exists() or preds.stat().st_size == 0:
        errors.append(
            f"preds.json missing or empty in {raw_dir}. "
            "Generation must produce a nonempty preds.json."
        )
    elif expected_instance_ids is not None:
        # Validate preds.json keys match the frozen instance set
        try:
            preds_data = json.loads(preds.read_text())
            if not isinstance(preds_data, dict):
                errors.append(
                    f"preds.json must be a dict keyed by instance_id; "
                    f"got {type(preds_data).__name__}."
                )
            else:
                preds_set = set(preds_data.keys())
                expected_set = set(expected_instance_ids)
                missing_preds = expected_set - preds_set
                extra_preds = preds_set - expected_set
                if missing_preds:
                    errors.append(
                        f"preds.json is missing {len(missing_preds)} of "
                        f"{len(expected_instance_ids)} expected instance IDs. "
                        f"Missing: {sorted(missing_preds)[:5]}"
                        f"{'...' if len(missing_preds) > 5 else ''}."
                    )
                if extra_preds:
                    errors.append(
                        f"preds.json contains {len(extra_preds)} unexpected instance IDs "
                        f"not in the frozen suite: "
                        f"{sorted(extra_preds)[:5]}"
                        f"{'...' if len(extra_preds) > 5 else ''}."
                    )
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"preds.json is not valid JSON: {exc}")

    exit_statuses = raw_dir / "exit_statuses.json"
    statuses: dict = {}
    if not exit_statuses.exists():
        errors.append(
            f"exit_statuses.json missing in {raw_dir}. "
            "Post-generation normalization must produce exit_statuses.json "
            "derived from the real mini-swe-agent exit_statuses_<timestamp>.yaml."
        )
    elif expected_instance_ids is not None:
        # Validate exit_statuses.json keys match the frozen instance set
        try:
            es_data = json.loads(exit_statuses.read_text())
            if not isinstance(es_data, dict):
                errors.append(
                    f"exit_statuses.json must be a dict keyed by instance_id; "
                    f"got {type(es_data).__name__}."
                )
            else:
                statuses = es_data
                es_set = set(es_data.keys())
                expected_set = set(expected_instance_ids)
                missing_es = expected_set - es_set
                extra_es = es_set - expected_set
                if missing_es:
                    errors.append(
                        f"exit_statuses.json is missing {len(missing_es)} of "
                        f"{len(expected_instance_ids)} expected instance IDs. "
                        f"Missing: {sorted(missing_es)[:5]}"
                        f"{'...' if len(missing_es) > 5 else ''}."
                    )
                if extra_es:
                    errors.append(
                        f"exit_statuses.json contains {len(extra_es)} unexpected instance IDs "
                        f"not in the frozen suite: "
                        f"{sorted(extra_es)[:5]}"
                        f"{'...' if len(extra_es) > 5 else ''}."
                    )
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"exit_statuses.json is not valid JSON: {exc}")

    # Require trajectories/ directory (written by normalization)
    trajectories_dir = raw_dir / "trajectories"
    if not trajectories_dir.exists() or not trajectories_dir.is_dir():
        errors.append(
            f"trajectories/ directory missing in {raw_dir}. "
            "Post-generation normalization must copy produced trajectory files "
            "from <id>/<id>.traj.json into trajectories/<id>.traj.json. "
            "Generation must preserve trajectories for evidence and audit."
        )
    elif expected_instance_ids is not None:
        # Determine which IDs are infra-failure (no trajectory required).
        # Read exit_statuses.json if present to identify exempt IDs.
        _INFRA_FAILURE_STATUSES = frozenset({
            "TimeoutExpired",       # docker pull/test timeout, model never invoked
            "InternalServerError",  # vLLM 5xx
            "infrastructure_error",
            "infra_failure",
        })
        infra_exempt: set = set()
        if exit_statuses.exists():
            try:
                es_data_for_exempt = json.loads(exit_statuses.read_text())
                if isinstance(es_data_for_exempt, dict):
                    for iid, status in es_data_for_exempt.items():
                        if str(status) in _INFRA_FAILURE_STATUSES:
                            infra_exempt.add(iid)
            except (json.JSONDecodeError, OSError):
                pass  # best-effort; if unreadable, exempt nothing

        # Verify exactly one stable trajectory artifact per expected ID,
        # excluding infra-failure IDs (no trajectory produced for them).
        missing_trajs: list = []
        for iid in expected_instance_ids:
            if iid in infra_exempt:
                continue  # pre-agent infra failure — no trajectory expected
            dst_traj = trajectories_dir / f"{iid}.traj"
            if not dst_traj.is_file() or dst_traj.stat().st_size == 0:
                missing_trajs.append(iid)
        if missing_trajs:
            errors.append(
                f"trajectories/ directory is missing {len(missing_trajs)} of "
                f"{len(expected_instance_ids)} required trajectory files. "
                f"Missing or empty: {sorted(missing_trajs)[:5]}"
                f"{'...' if len(missing_trajs) > 5 else ''}. "
                "Every frozen instance requires auditable trajectory evidence."
            )

    # Require run.log (must be nonempty — empty means generation never ran)
    run_log = raw_dir / "run.log"
    if not run_log.exists():
        errors.append(
            f"run.log missing in {raw_dir}. "
            "Generation must produce run.log capturing subprocess output."
        )
    elif run_log.stat().st_size == 0:
        errors.append(
            f"run.log in {raw_dir} is empty. "
            "A nonempty run.log is required as evidence that generation ran. "
            "An empty run.log suggests the subprocess was never invoked or wrote nothing."
        )

    # --- Reject campaign-wide runtime/infrastructure failure before grading ---
    # mini-swe-agent can return process exit 0 while every instance records
    # RuntimeError (for example, unmapped LiteLLM cost metadata). Complete file/ID
    # coverage is not scientific success in that case.
    runtime_error_ids = [
        iid for iid, disposition in statuses.items()
        if isinstance(disposition, str) and disposition.lower() == "runtimeerror"
    ]
    if runtime_error_ids:
        errors.append(
            f"Generation contains {len(runtime_error_ids)}/{len(statuses)} RuntimeError "
            "infrastructure dispositions; refusing to grade or complete the campaign."
        )

    return errors


def _verify_grading_evidence(
    run_dir: pathlib.Path,
    expected_instance_ids: List[str],
) -> List[str]:
    """Verify grading phase artifacts cover all 100 expected instances.

    Checks grading_results.json contains disposition entries for every expected
    instance ID across all four disposition categories.

    Returns [] on success, list of error strings on failure.
    """
    errors: List[str] = []
    raw_dir = run_dir / "raw"
    grading_path = raw_dir / "grading_results.json"

    if not grading_path.exists() or grading_path.stat().st_size == 0:
        errors.append(
            f"grading_results.json missing or empty in {raw_dir}. "
            "Grading must produce a nonempty grading_results.json."
        )
        return errors

    try:
        grading = json.loads(grading_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        errors.append(f"grading_results.json is not valid JSON: {exc}")
        return errors

    # Collect all dispositioned instance IDs — check for duplicates
    dispositioned: list = []
    for key in ("resolved_ids", "unresolved_ids", "empty_patch_ids", "error_ids", "incomplete_ids"):
        ids = grading.get(key) or []
        dispositioned.extend(ids)

    # Check for duplicate IDs across categories (a disposition integrity violation)
    dispositioned_counter: dict = {}
    for iid in dispositioned:
        dispositioned_counter[iid] = dispositioned_counter.get(iid, 0) + 1
    duplicates = {iid for iid, cnt in dispositioned_counter.items() if cnt > 1}
    if duplicates:
        errors.append(
            f"{len(duplicates)} instance IDs appear in multiple disposition categories "
            f"(duplicate dispositions are a grading integrity violation): "
            f"{sorted(duplicates)[:5]}{'...' if len(duplicates) > 5 else ''}."
        )

    dispositioned_set = set(dispositioned)
    expected_set = set(expected_instance_ids)
    missing = expected_set - dispositioned_set
    extra = dispositioned_set - expected_set

    if missing:
        errors.append(
            f"{len(missing)} of {len(expected_instance_ids)} expected instances have no "
            f"grading disposition. Missing: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}. "
            "All 100 instances must have a terminal disposition."
        )
    if extra:
        errors.append(
            f"{len(extra)} unexpected instance IDs in grading results: "
            f"{sorted(extra)[:5]}{'...' if len(extra) > 5 else ''}."
        )

    return errors


# ---------------------------------------------------------------------------
# Grading report normalization
# ---------------------------------------------------------------------------


def _normalize_grading_report(
    raw_report: dict,
    expected_instance_ids: List[str],
) -> Optional[dict]:
    """Normalize a raw SWE-bench harness report to our grading_results.json schema.

    The harness schema-v2 (make_run_report in swebench.harness.reporting) emits:
      resolved_ids, unresolved_ids, empty_patch_ids, error_ids, incomplete_ids
      (plus completed_ids, submitted_ids, and count fields — ignored here)

    Schema-v2 disposition categories (disjoint):
      resolved_ids     — instances whose patch resolved the issue
      unresolved_ids   — instances whose patch did not resolve
      empty_patch_ids  — instances with empty/None patch (not graded)
      error_ids        — instances that errored during grading
      incomplete_ids   — instances with no prediction or not attempted

    STRICT rules enforced:
      - Only IDs that appear in the report are assigned (never invented).
      - IDs present in multiple categories → normalization fails (returns None).
      - IDs not in expected_instance_ids → normalization fails (foreign ID rejected).
      - Expected IDs not in ANY category → assigned to incomplete_ids from the report
        or left in the normalization output; the caller (_verify_grading_evidence) checks
        that all 100 are covered.

    Format A (per-instance dict): {instance_id: {"resolved": bool, ...}}
    Format B (schema-v2 top-level keys): {"resolved_ids": [...], ...}

    Returns a dict with disjoint categories, or None on parse failure.
    """
    if not isinstance(raw_report, dict):
        return None

    expected_set = set(expected_instance_ids)

    # --- Detect Format B: schema-v2 top-level list keys ---
    _SCHEMA_V2_KEYS = {"resolved_ids", "unresolved_ids", "empty_patch_ids", "error_ids", "incomplete_ids"}
    _ALLOWED_FOREIGN_KEYS = {
        # Schema-v2 count/metadata fields that are not ID lists — tolerated but not mapped
        "total_instances", "submitted_instances", "completed_instances",
        "resolved_instances", "unresolved_instances", "empty_patch_instances",
        "error_instances", "incomplete_instances", "schema_version",
        "completed_ids", "submitted_ids",
        "unstopped_instances", "unstopped_containers", "unremoved_images",
    }
    has_v2_key = any(k in raw_report for k in _SCHEMA_V2_KEYS)
    has_legacy_key = any(k in raw_report for k in ("resolved", "unresolved"))

    if has_v2_key or has_legacy_key:
        # Format B: schema-v2 or legacy top-level lists
        resolved = list(raw_report.get("resolved_ids") or raw_report.get("resolved") or [])
        unresolved = list(raw_report.get("unresolved_ids") or raw_report.get("unresolved") or [])
        empty_patch = list(raw_report.get("empty_patch_ids") or raw_report.get("empty_patch") or [])
        error = list(raw_report.get("error_ids") or raw_report.get("error") or [])
        incomplete = list(raw_report.get("incomplete_ids") or [])

        # --- Strict: reject any foreign key that is not schema-v2 or tolerated metadata ---
        unknown_keys = set(raw_report.keys()) - _SCHEMA_V2_KEYS - _ALLOWED_FOREIGN_KEYS
        # Also tolerate old-style aliases used in detection above
        unknown_keys -= {"resolved", "unresolved", "empty_patch", "error"}
        if unknown_keys:
            # Not fatal for normalization — log concern but continue
            pass  # foreign keys from future schema versions are tolerated

        # Collect all assigned IDs for overlap + foreign checks
        all_lists = [resolved, unresolved, empty_patch, error, incomplete]
        all_ids: list = []
        for lst in all_lists:
            all_ids.extend(lst)

        # Reject foreign IDs (not in expected_instance_ids)
        foreign_ids = [iid for iid in all_ids if iid not in expected_set]
        if foreign_ids:
            # Return None to signal normalization failure — caller logs error
            return None

        # Reject duplicate IDs across categories
        seen: dict = {}
        for iid in all_ids:
            seen[iid] = seen.get(iid, 0) + 1
        duplicates = [iid for iid, cnt in seen.items() if cnt > 1]
        if duplicates:
            return None

        return {
            "resolved_ids": resolved,
            "unresolved_ids": unresolved,
            "empty_patch_ids": empty_patch,
            "error_ids": error,
            "incomplete_ids": incomplete,
        }

    # --- Detect Format A: per-instance dict keyed by instance_id ---
    # Values may be dicts with "resolved" key (bool), or just booleans
    resolved_ids: list = []
    unresolved_ids: list = []
    error_ids: list = []

    for iid, val in raw_report.items():
        if iid not in expected_set:
            continue  # skip unexpected IDs (foreign — tolerated in Format A)
        if isinstance(val, dict):
            if val.get("resolved", False):
                resolved_ids.append(iid)
            elif val.get("error"):
                error_ids.append(iid)
            else:
                unresolved_ids.append(iid)
        elif isinstance(val, bool):
            if val:
                resolved_ids.append(iid)
            else:
                unresolved_ids.append(iid)
        else:
            unresolved_ids.append(iid)

    return {
        "resolved_ids": resolved_ids,
        "unresolved_ids": unresolved_ids,
        "empty_patch_ids": [],
        "error_ids": error_ids,
        "incomplete_ids": [],
    }


# ---------------------------------------------------------------------------
# SwebenchRunner
# ---------------------------------------------------------------------------


class SwebenchRunner:
    """Contract-aware SWE-bench runner.

    Parameters
    ----------
    suite_path : pathlib.Path
        Path to warpcore-v1.yaml.
    adapter_path : pathlib.Path
        Path to adapter YAML.
    endpoint : str
        OpenAI-compatible base URL ending in /v1.
    run_dir : pathlib.Path
        Normalized run directory (repo/results/<slug>/runs/<suite_id>/swebench/<run_id>).
    repo : pathlib.Path or None
        Repository root. Defaults to parent of suite file.
    api_key : str
        API key for the serving endpoint. Defaults to "warpcore".
    workers : int
        Number of parallel mini-swe-agent workers.
    dry_run : bool
        If True, print scaffold config and return 0 without any side effects.
    allow_no_screen : bool
        Bypass the /usr/bin/screen guard (tests and special ops only).
    preflight_runner : callable or None
        Injected callable(model_id: str) -> int. Defaults to SwebenchPreflightGate.
    generation_runner : callable or None
        Injected callable(config: dict, run_dir: Path, **kw) -> int.
    grading_runner : callable or None
        Injected callable(preds_path: Path, run_dir: Path, **kw) -> int.
    """

    def __init__(
        self,
        suite_path: pathlib.Path,
        adapter_path: pathlib.Path,
        endpoint: str,
        run_dir: pathlib.Path,
        repo: Optional[pathlib.Path] = None,
        api_key: str = "warpcore",
        workers: int = 4,
        dry_run: bool = False,
        allow_no_screen: bool = False,
        prompt_token_maxima: Optional[Dict[str, int]] = None,
        qualification_path: Optional[pathlib.Path] = None,
        repo_sha: Optional[str] = None,
        qualification_run: bool = False,
        preflight_runner: Optional[Callable] = None,
        generation_runner: Optional[Callable] = None,
        grading_runner: Optional[Callable] = None,
        done_writer: Optional[Callable] = None,
    ) -> None:
        self.suite_path = pathlib.Path(suite_path).resolve()
        self.adapter_path = pathlib.Path(adapter_path).resolve()
        self.endpoint = endpoint
        self.run_dir = pathlib.Path(run_dir).resolve()
        self.api_key = api_key
        self.workers = int(workers)
        self.dry_run = dry_run
        self.allow_no_screen = allow_no_screen
        self._prompt_token_maxima = prompt_token_maxima
        self._qualification_path = (
            pathlib.Path(qualification_path).resolve() if qualification_path else None
        )
        self._repo_sha = repo_sha
        self.qualification_run = qualification_run

        # Resolve repo
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

        # -- Campaign-readiness validation (rejects noncanonical) --
        model_slug = (self._adapter.get("model") or {}).get("slug", "")
        readiness_errors = validate_adapter_campaign_ready(
            self._adapter,
            model_slug,
            suite=self._suite,
            prompt_token_maxima=self._prompt_token_maxima,
        )
        if readiness_errors:
            raise ValueError(
                f"Adapter {self.adapter_path} is not campaign-ready (noncanonical):\n"
                + "\n".join(readiness_errors)
            )

        # -- Resolve benchmark config --
        benchmarks = self._suite.get("benchmarks", {})
        if _SWEBENCH_BENCH not in benchmarks:
            raise ValueError(
                f"Benchmark {_SWEBENCH_BENCH!r} not found in suite {suite_path}."
            )
        self._bench_cfg: dict = benchmarks[_SWEBENCH_BENCH]

        # Suite-level IDs
        self._suite_id: str = self._suite.get("suite_id", "warpcore-v1")
        self._run_id: str = self.run_dir.name
        self._model_slug: str = model_slug
        self._model_id: str = (self._adapter.get("model") or {}).get("id", "")

        # -- Load and verify instance set --
        instance_set_file = self._bench_cfg.get("instance_set_file", "")
        if not instance_set_file:
            raise ValueError(
                "suite/swebench benchmark missing 'instance_set_file' field."
            )
        self._instances_path = (_REPO_DIR / instance_set_file).resolve()
        if not self._instances_path.exists():
            raise ValueError(
                f"Instance set file not found: {self._instances_path}"
            )

        # Verify instance set hash
        declared_hash = self._bench_cfg.get("instances_sha256", "")
        if declared_hash:
            actual_hash = sha256_file(self._instances_path)
            if actual_hash != declared_hash:
                raise ValueError(
                    f"Instance set hash mismatch: suite declares {declared_hash!r} "
                    f"but actual is {actual_hash!r}. Suite input is corrupt or stale."
                )

        self._instance_ids: List[str] = json.loads(
            self._instances_path.read_text()
        )
        if not isinstance(self._instance_ids, list):
            raise ValueError(
                f"Instance set file must contain a JSON list; got {type(self._instance_ids).__name__}."
            )

        # Validate exact count
        expected_count = self._bench_cfg.get("expected_item_count", 100)
        if len(self._instance_ids) != expected_count:
            raise ValueError(
                f"Instance set contains {len(self._instance_ids)} IDs but suite requires exactly "
                f"{expected_count}. The instance set file is corrupt or stale."
            )

        # Validate uniqueness
        seen: set = set()
        duplicates: list = []
        for iid in self._instance_ids:
            if iid in seen:
                duplicates.append(iid)
            seen.add(iid)
        if duplicates:
            raise ValueError(
                f"Instance set contains {len(duplicates)} duplicate instance IDs: "
                f"{sorted(set(duplicates))[:5]}{'...' if len(duplicates) > 5 else ''}. "
                "Instance IDs must be unique."
            )

        # -- Qualification mode: narrow to the suite-owned qualification set --
        # The frozen n=100 set above is still loaded and hash-verified first, so a
        # corrupt suite fails here rather than producing a qualification against
        # drifted inputs. Only then do we substitute the 20 suite-owned IDs, which
        # must be a hash-verified subset of that frozen set.
        if self.qualification_run:
            qual_cfg = self._bench_cfg.get("qualification") or {}
            qual_ids_file = qual_cfg.get("ids_file", "")
            if not qual_ids_file:
                raise ValueError(
                    "suite/swebench benchmark missing 'qualification.ids_file'; "
                    "a qualification run needs a suite-owned instance set."
                )
            self._qual_ids_path = (_REPO_DIR / qual_ids_file).resolve()
            if not self._qual_ids_path.exists():
                raise ValueError(
                    f"Qualification instance set file not found: {self._qual_ids_path}"
                )
            declared_qual_hash = qual_cfg.get("ids_sha256", "")
            if declared_qual_hash:
                actual_qual_hash = sha256_file(self._qual_ids_path)
                if actual_qual_hash != declared_qual_hash:
                    raise ValueError(
                        f"Qualification instance set hash mismatch: suite declares "
                        f"{declared_qual_hash!r} but actual is {actual_qual_hash!r}."
                    )
            qual_ids = json.loads(self._qual_ids_path.read_text())
            if not isinstance(qual_ids, list):
                raise ValueError(
                    "Qualification instance set file must contain a JSON list; "
                    f"got {type(qual_ids).__name__}."
                )
            required = swebench_qualification.REQUIRED_QUALIFICATION_COUNT
            if len(qual_ids) != required or len(set(qual_ids)) != required:
                raise ValueError(
                    f"Qualification instance set must hold exactly {required} unique IDs; "
                    f"got {len(qual_ids)} ({len(set(qual_ids))} unique)."
                )
            foreign = [i for i in qual_ids if i not in seen]
            if foreign:
                raise ValueError(
                    f"Qualification instance set contains {len(foreign)} ID(s) that are not "
                    f"in the frozen instance set: {sorted(foreign)[:5]}."
                )
            self._instance_ids = qual_ids
            self._instances_path = self._qual_ids_path

        # -- Load and verify scaffold --
        scaffold_file = self._bench_cfg.get("scaffold_file", "")
        if not scaffold_file:
            raise ValueError(
                "suite/swebench benchmark missing 'scaffold_file' field."
            )
        self._scaffold_path = (_REPO_DIR / scaffold_file).resolve()
        if not self._scaffold_path.exists():
            raise ValueError(
                f"Scaffold file not found: {self._scaffold_path}"
            )

        # Verify scaffold hash
        declared_scaffold_hash = self._bench_cfg.get("scaffold_sha256", "")
        if declared_scaffold_hash:
            actual_scaffold_hash = sha256_file(self._scaffold_path)
            if actual_scaffold_hash != declared_scaffold_hash:
                raise ValueError(
                    f"Scaffold hash mismatch: suite declares {declared_scaffold_hash!r} "
                    f"but actual is {actual_scaffold_hash!r}. "
                    "Scaffold is corrupt or stale — suite must be updated."
                )

        import yaml
        with open(self._scaffold_path) as fh:
            self._scaffold: dict = yaml.safe_load(fh)

        # Verify submit protocol is intact
        self._verify_submit_protocol()

        # Injected runners
        self._preflight_runner = preflight_runner
        self._generation_runner = generation_runner
        self._grading_runner = grading_runner
        self._done_writer = done_writer  # callable(done_path: Path) -> None; default: atomic rename

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_instance_ids(self) -> List[str]:
        """Return the frozen 100 instance IDs."""
        return list(self._instance_ids)

    def build_scaffold_config(
        self,
        endpoint: str,
        api_key: str = "warpcore",
    ) -> dict:
        """Return the scaffold config with only the adapter model identity injected.

        Delegates to swebench_qualification.build_production_scaffold_config, the
        single definition of the production config builder.  A qualification
        records a digest of exactly this output, so a campaign cannot launch on a
        config that differs from the one that qualified.
        """
        return swebench_qualification.build_production_scaffold_config(
            scaffold=self._scaffold,
            model_id=self._model_id,
            endpoint=endpoint,
            api_key=api_key,
        )

    # ------------------------------------------------------------------
    # Qualification gate
    # ------------------------------------------------------------------

    def qualification_artifact_path(self) -> pathlib.Path:
        """Return the qualification record this run must be authorized by."""
        if self._qualification_path is not None:
            return self._qualification_path
        return swebench_qualification.default_artifact_path(
            self._repo, self._suite_id, self._model_slug
        )

    def check_qualification(self) -> "swebench_qualification.QualificationResult":
        """Evaluate the authoritative launch gate.  Read-only; never writes.

        Suite, scaffold, qualification-ID, and repo-SHA identity are resolved
        against the repository that owns the contract code (_REPO_DIR), not the
        results root: the qualification binds to the policy that produced it.
        """
        return swebench_qualification.verify_qualification_for_launch(
            repo=_REPO_DIR,
            suite_path=self.suite_path,
            adapter_path=self.adapter_path,
            endpoint=self.endpoint,
            artifact_path=self.qualification_artifact_path(),
            api_key=self.api_key,
            repo_sha=self._repo_sha,
        )

    # ------------------------------------------------------------------
    # Qualification execution mode
    # ------------------------------------------------------------------

    def run_qualification(self) -> int:
        """Execute the 20-instance qualification run that produces a launch record.

        This is the production runner in a narrow mode, not a second runner: same
        preflight, same production config builder, same generation and grading
        code paths, same artifact normalization.  Only the instance set differs,
        and it is the suite-owned qualification set.

        Two deliberate differences from :meth:`run`:

        * It is **not** gated on an existing qualification.  A qualification run
          is what produces the authorization; requiring one here would make the
          gate unreachable.
        * It writes **no campaign state**.  A qualification is not a campaign: no
          create_campaign, no status.json, no manifest.json, no DONE sentinel.
          The evidence it leaves is read by the qualification validator, and the
          record sealed from it is what a campaign is later checked against.

        It still refuses to produce evidence from a bad run: the same policy the
        gate enforces is applied here, so a RepeatedFormatError qualification run
        fails before any record can be sealed from it.

        Returns 0 on success, or one of the module EXIT_* codes.
        """
        if not self.qualification_run:
            print(
                "[run-swebench] FATAL: run_qualification() requires "
                "qualification_run=True; a campaign runner must not produce a "
                "qualification.",
                file=sys.stderr,
            )
            return EXIT_CONFIG

        if not self.allow_no_screen and not _is_under_screen():
            print(
                "ERROR: SWE-bench qualification runs must be launched inside "
                "/usr/bin/screen on the Mac mini. Start a screen session first:\n"
                "  screen -S swebench-qualify\n"
                "Or pass --allow-no-screen to bypass (tests/special ops only).",
                file=sys.stderr,
            )
            return EXIT_INCONCLUSIVE

        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "command.txt").write_text(
            f"qualification run: {len(self._instance_ids)} suite-owned instances from "
            f"{self._instances_path.name}\n",
            encoding="utf-8",
        )

        preflight_rc = self._run_preflight()
        if preflight_rc != 0:
            print(
                f"[run-swebench] Qualification preflight failed (exit {preflight_rc}).",
                file=sys.stderr,
            )
            return EXIT_DEFECT if preflight_rc == 1 else EXIT_INCONCLUSIVE

        scaffold_config = self.build_scaffold_config(
            endpoint=self.endpoint, api_key=self.api_key
        )
        generation_rc = self._run_generation(scaffold_config)
        if generation_rc != 0:
            print(
                f"[run-swebench] Qualification generation failed (exit {generation_rc}).",
                file=sys.stderr,
            )
            return EXIT_DEFECT

        if self._generation_runner is None:
            norm_errors = _normalize_generation_artifacts(
                self.run_dir / "raw", expected_instance_ids=self._instance_ids
            )
            if norm_errors:
                print(
                    "ERROR: Qualification artifact normalization failed:\n"
                    + "\n".join(f"  {e}" for e in norm_errors),
                    file=sys.stderr,
                )
                return EXIT_DEFECT

        gen_errors = _verify_generation_evidence(
            self.run_dir, expected_instance_ids=self._instance_ids
        )
        if gen_errors:
            print(
                "ERROR: Qualification generation evidence is incomplete:\n"
                + "\n".join(f"  {e}" for e in gen_errors),
                file=sys.stderr,
            )
            return EXIT_DEFECT

        grading_rc = self._run_grading(self.run_dir / "raw" / "preds.json")
        if grading_rc != 0:
            print(
                f"[run-swebench] Qualification grading failed (exit {grading_rc}).",
                file=sys.stderr,
            )
            return EXIT_DEFECT

        # Apply the gate's own policy to the evidence now, so a bad qualification
        # run reports its defect here rather than at `make qualify-swebench`.
        policy_errors = swebench_qualification.check_raw_evidence_policy(
            self.run_dir / "raw", self._instance_ids
        )
        if policy_errors:
            print(
                "ERROR: Qualification run completed but does not qualify:\n"
                + "\n".join(f"  {e}" for e in policy_errors)
                + "\nRe-qualify. Do not add a bridge, parser repair, retry, output "
                "sanitizer, or scaffold change to make this pass.",
                file=sys.stderr,
            )
            return EXIT_DEFECT

        print(
            f"[run-swebench] Qualification run complete: {len(self._instance_ids)} "
            f"suite-owned instances, evidence under {self.run_dir}. "
            "Seal it with `make qualify-swebench`.",
            file=sys.stderr,
        )
        return EXIT_SUCCESS

    # ------------------------------------------------------------------
    # Core run logic
    # ------------------------------------------------------------------

    def run(self) -> int:
        """Execute the full SWE-bench run lifecycle.

        Lifecycle (for live runs):
          1. Screen guard check.
          2. Read and strict-validate status.json (schema + run_id + suite_id).
          3. Accept only 'planned' state.
          4. Write command.txt.
          5. Run preflight gate.
          6. Transition: planned -> preflight_passed -> running (generation note).
          7. Execute generation runner.
          8. Verify generation evidence (preds.json, exit_statuses.json).
          9. Transition history note: grading phase started.
          10. Execute grading runner.
          11. Verify grading evidence (all 100 dispositions in grading_results.json).
          12. Atomic completed transition + DONE sentinel.

        Returns exit code (0 = success, nonzero = failure).
        """
        # --- Dry-run: no side effects ---
        if self.dry_run:
            config = self.build_scaffold_config(
                endpoint=self.endpoint,
                api_key=self.api_key,
            )
            # Redact API key before printing — never log secrets
            import copy as _copy
            display_config = _copy.deepcopy(config)
            if isinstance(display_config.get("model"), dict):
                kwargs = display_config["model"].get("model_kwargs", {})
                if "api_key" in kwargs:
                    kwargs["api_key"] = "***REDACTED***"
            print("[dry-run] Scaffold config (injected):")
            import yaml
            print(yaml.dump(display_config, default_flow_style=False))
            print(f"[dry-run] Instance set: {len(self._instance_ids)} instances from "
                  f"{self._instances_path.name}")
            if self.qualification_run:
                print(
                    "[dry-run] QUALIFICATION RUN: this mode produces a qualification "
                    "record and is deliberately not gated on one. It writes no "
                    "campaign state."
                )
                return EXIT_SUCCESS
            # A dry run inspects; it never authorizes. Report the gate verdict
            # plainly so an operator cannot read "dry-run OK" as "cleared to launch".
            verdict = self.check_qualification()
            print(f"[dry-run] {verdict.render()}")
            print(f"[dry-run] Qualification record: {self.qualification_artifact_path()}")
            if not verdict.ok:
                print(
                    "[dry-run] A live launch would be refused until this qualification "
                    "is fresh, complete, and bound to this exact launch context."
                )
            return EXIT_SUCCESS

        # --- Screen guard ---
        if not self.allow_no_screen and not _is_under_screen():
            print(
                "ERROR: SWE-bench runs must be launched inside /usr/bin/screen "
                "on the Mac mini. Start a screen session first:\n  screen -S swebench-run\n"
                "Or pass --allow-no-screen to bypass (tests/special ops only).",
                file=sys.stderr,
            )
            return EXIT_INCONCLUSIVE

        # --- Authoritative qualification gate (before any write) ---
        # This is the gate that external cron/shell logic did not have: a
        # RepeatedFormatError smoke, a stale record, or a record sealed for a
        # different SHA/suite/adapter/profile/model/scaffold stops the launch here,
        # with no campaign state mutated.
        qualification = self.check_qualification()
        if not qualification.ok:
            print(qualification.render(), file=sys.stderr)
            print(
                f"[run-swebench] FATAL: refusing to launch a canonical SWE-bench campaign "
                f"without a valid qualification "
                f"({self.qualification_artifact_path()}).",
                file=sys.stderr,
            )
            return EXIT_DEFECT
        print(f"[run-swebench] {qualification.render()}", file=sys.stderr)

        # --- Read and strict-validate status.json ---
        try:
            status = self._read_status_strict()
        except Exception as exc:
            print(
                f"[run-swebench] FATAL: Cannot read or validate status.json: {exc}",
                file=sys.stderr,
            )
            return EXIT_DEFECT

        # --- Accept only 'planned' state ---
        current_state = status.get("execution_state", "")
        if current_state != "planned":
            print(
                f"[run-swebench] FATAL: Run directory is in state {current_state!r}; "
                "expected 'planned'.",
                file=sys.stderr,
            )
            return EXIT_DEFECT

        # --- Write command.txt ---
        self._write_command_txt()

        # --- Run preflight gate ---
        preflight_rc = self._run_preflight()
        if preflight_rc != 0:
            print(
                f"[run-swebench] Preflight gate failed (exit {preflight_rc}).",
                file=sys.stderr,
            )
            return EXIT_DEFECT if preflight_rc == 1 else EXIT_INCONCLUSIVE

        # --- Transition planned -> preflight_passed ---
        try:
            status = self._read_status_strict()
            status = campaign_state.apply_transition(
                status=status,
                new_state="preflight_passed",
                timestamp=_utcnow(),
            )
            self._write_status(status)
        except Exception as exc:
            print(
                f"[run-swebench] FATAL: Transition to 'preflight_passed' failed: {exc}",
                file=sys.stderr,
            )
            return EXIT_DEFECT

        # --- Transition preflight_passed -> running (generation phase) ---
        try:
            status = self._read_status_strict()
            # Build running entry with generation note
            new_status = campaign_state.apply_transition(
                status=status,
                new_state="running",
                timestamp=_utcnow(),
            )
            # Annotate last history entry with generation note
            if new_status.get("history"):
                new_status["history"][-1]["note"] = "generation phase started"
            self._write_status(new_status)
        except Exception as exc:
            print(
                f"[run-swebench] FATAL: Transition to 'running' (generation) failed: {exc}",
                file=sys.stderr,
            )
            return EXIT_DEFECT

        # --- Execute generation ---
        scaffold_config = self.build_scaffold_config(
            endpoint=self.endpoint,
            api_key=self.api_key,
        )
        generation_rc = self._run_generation(scaffold_config)

        if generation_rc != 0:
            print(
                f"[run-swebench] Generation failed (exit {generation_rc}).",
                file=sys.stderr,
            )
            if not self._transition_failed(note="generation failed"):
                print(
                    "[run-swebench] FATAL: _transition_failed write failed after generation failure. "
                    "Run is in an indeterminate state.",
                    file=sys.stderr,
                )
            return EXIT_DEFECT

        # --- Normalize real mini-swe-agent 2.4.6 artifacts into stable raw/ layout ---
        # The built-in runner emits the upstream layout. Injected runners implement
        # the existing stable-artifact contract directly and must not be forced to
        # synthesize upstream-only YAML as well.
        if self._generation_runner is None:
            norm_errors = _normalize_generation_artifacts(
                self.run_dir / "raw",
                expected_instance_ids=self._instance_ids,
            )
            if norm_errors:
                print(
                    "ERROR: Post-generation normalization failed:\n"
                    + "\n".join(f"  {e}" for e in norm_errors)
                    + "\nDONE not written.",
                    file=sys.stderr,
                )
                if not self._transition_failed(note="generation artifact normalization failed"):
                    print(
                        "[run-swebench] FATAL: _transition_failed write failed after normalization failure. "
                        "Run is in an indeterminate state.",
                        file=sys.stderr,
                    )
                return EXIT_DEFECT

        # --- Verify generation evidence ---
        gen_errors = _verify_generation_evidence(self.run_dir, expected_instance_ids=self._instance_ids)
        if gen_errors:
            print(
                "ERROR: Generation reported success but required evidence is missing:\n"
                + "\n".join(f"  {e}" for e in gen_errors)
                + "\nDONE not written.",
                file=sys.stderr,
            )
            if not self._transition_failed(note="generation evidence missing"):
                print(
                    "[run-swebench] FATAL: _transition_failed write failed after evidence check failure. "
                    "Run is in an indeterminate state.",
                    file=sys.stderr,
                )
            return EXIT_DEFECT

        # --- Record grading phase start in history ---
        try:
            status = self._read_status_strict()
            # Append a note about grading phase to the last history entry's note field
            # We do this by adding a new history annotation — but the schema only allows
            # transitions via apply_transition. We add the grading note inline to the
            # running state's last entry (note field is allowed by schema).
            if status.get("history"):
                # Find the running entry and update its note
                for entry in reversed(status["history"]):
                    if entry.get("state") == "running":
                        existing_note = entry.get("note", "")
                        entry["note"] = (existing_note + "; grading phase started").lstrip("; ")
                        break
            self._write_status(status)
        except Exception as exc:
            print(
                f"[run-swebench] FATAL: Could not annotate grading phase in status: {exc}",
                file=sys.stderr,
            )
            # Status write failure is fatal — a run must not continue with inconsistent state.
            if not self._transition_failed(note="grading phase annotation write failed"):
                print(
                    "[run-swebench] FATAL: _transition_failed write failed after grading annotation failure. "
                    "Run is in an indeterminate state.",
                    file=sys.stderr,
                )
            return EXIT_DEFECT

        # --- Execute grading ---
        preds_path = self.run_dir / "raw" / "preds.json"
        grading_rc = self._run_grading(preds_path)

        if grading_rc != 0:
            print(
                f"[run-swebench] Grading failed (exit {grading_rc}).",
                file=sys.stderr,
            )
            if not self._transition_failed(note="grading failed"):
                print(
                    "[run-swebench] FATAL: _transition_failed write failed after grading failure. "
                    "Run is in an indeterminate state.",
                    file=sys.stderr,
                )
            return EXIT_DEFECT

        # --- Verify grading evidence (all 100 dispositions) ---
        grading_errors = _verify_grading_evidence(self.run_dir, self._instance_ids)
        if grading_errors:
            print(
                "ERROR: Grading reported success but evidence is missing or incomplete:\n"
                + "\n".join(f"  {e}" for e in grading_errors)
                + "\nDONE not written.",
                file=sys.stderr,
            )
            if not self._transition_failed(note="grading evidence incomplete"):
                print(
                    "[run-swebench] FATAL: _transition_failed write failed after grading evidence check. "
                    "Run is in an indeterminate state.",
                    file=sys.stderr,
                )
            return EXIT_DEFECT

        # --- Atomic completed transition + DONE ---
        # Fail-closed ordering: write DONE sentinel first (before committing
        # completed status). If DONE write fails, status stays as 'running'
        # and we transition to 'failed' — no inconsistent completed-without-DONE state.
        # If DONE write succeeds but completed status write fails, we attempt
        # to remove DONE and transition to failed so no inconsistency persists.
        return self._transition_completed_atomic()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _verify_submit_protocol(self) -> None:
        """Verify the submit protocol is intact in the scaffold.

        The canonical submit protocol is:
          git add -A && git diff --cached

        Raises ValueError if not found.
        """
        # Check instance_template for the submit command
        agent = self._scaffold.get("agent", {})
        instance_template = agent.get("instance_template", "")
        if "git add -A" not in instance_template or "git diff --cached" not in instance_template:
            raise ValueError(
                "Scaffold submit protocol is missing or corrupt. "
                "Expected 'git add -A && git diff --cached' in agent.instance_template. "
                "The scaffold must not be modified without updating the suite hash."
            )

    def _write_command_txt(self) -> None:
        """Write a human-readable record of the runner command to command.txt."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        cmd_parts = [
            "python3", "viz/run_swebench.py",
            "--suite", str(self.suite_path.relative_to(_REPO_DIR) if self.suite_path.is_relative_to(_REPO_DIR) else self.suite_path),
            "--adapter", str(self.adapter_path.relative_to(_REPO_DIR) if self.adapter_path.is_relative_to(_REPO_DIR) else self.adapter_path),
            "--endpoint", self.endpoint,
            "--run-id", self._run_id,
            "--model", self._model_id,
        ]
        cmd_line = " ".join(shlex.quote(a) for a in cmd_parts)
        (self.run_dir / "command.txt").write_text(cmd_line + "\n", encoding="utf-8")

    def _run_preflight(self) -> int:
        """Run the SWE-bench preflight gate; return exit code (0=pass, 1=defect, 2=inconclusive).

        When preflight_runner is injected, delegates to it (for tests).
        Default live preflight checks:
          1. x86 Docker host (SWE-bench containers require x86).
          2. Live model identity from /v1/models (exact match against adapter model_id).
          3. Image cache via swebench_preflight.py wrapper.
        """
        if self._preflight_runner is not None:
            return int(self._preflight_runner(self._model_id))

        # --- Default live preflight ---

        # 1. x86 Docker host check
        import subprocess as _sp
        try:
            docker_info_result = _sp.run(
                ["docker", "info", "--format", "{{.Architecture}}"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if docker_info_result.returncode != 0:
                print(
                    "[run-swebench] PREFLIGHT FAIL: docker info failed. "
                    "Docker must be running.",
                    file=sys.stderr,
                )
                return EXIT_INCONCLUSIVE
            arch = docker_info_result.stdout.strip().lower()
            # SWE-bench containers are x86_64; unknown/empty cannot establish readiness.
            if arch not in ("x86_64", "amd64"):
                if "arm" in arch or "aarch" in arch:
                    print(
                        f"[run-swebench] PREFLIGHT FAIL: Docker host architecture is {arch!r}. "
                        "SWE-bench containers require x86_64. "
                        "This host cannot run SWE-bench test containers natively.",
                        file=sys.stderr,
                    )
                    return EXIT_DEFECT
                print(
                    f"[run-swebench] PREFLIGHT INCONCLUSIVE: Docker returned unknown architecture {arch!r}; "
                    "cannot confirm the required x86_64 host.",
                    file=sys.stderr,
                )
                return EXIT_INCONCLUSIVE
        except Exception as exc:
            print(f"[run-swebench] PREFLIGHT INCONCLUSIVE: x86 check raised: {exc}", file=sys.stderr)
            return EXIT_INCONCLUSIVE

        # 2. Live model identity check — /v1/models must include the adapter model_id
        import urllib.request as _req
        import urllib.error as _uerr
        models_url = self.endpoint.rstrip("/") + "/models"
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        request = _req.Request(models_url, headers=headers, method="GET")
        try:
            with _req.urlopen(request, timeout=30) as resp:  # noqa: S310
                models_data = json.loads(resp.read())
            model_ids = [m.get("id", "") for m in (models_data.get("data") or [])]
            if self._model_id not in model_ids:
                print(
                    f"[run-swebench] PREFLIGHT FAIL: Model {self._model_id!r} not found at "
                    f"{models_url}. Served models: {model_ids}. "
                    "Adapter model_id must exactly match the served model.",
                    file=sys.stderr,
                )
                return EXIT_DEFECT
        except (_uerr.URLError, OSError) as exc:
            print(
                f"[run-swebench] PREFLIGHT INCONCLUSIVE: Cannot probe {models_url}: {exc}",
                file=sys.stderr,
            )
            return EXIT_INCONCLUSIVE

        # 3. Image cache check via swebench_preflight.py
        preflight_script = _VIZ_DIR / "swebench_preflight.py"
        if not preflight_script.exists():
            print(
                f"[run-swebench] PREFLIGHT INCONCLUSIVE: swebench_preflight.py not found at "
                f"{preflight_script}",
                file=sys.stderr,
            )
            return EXIT_INCONCLUSIVE

        # Write a temporary instances file for the preflight (JSON list — now supported)
        import tempfile as _tf
        with _tf.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tf:
            json.dump(self._instance_ids, tf)
            tmp_instances = tf.name

        try:
            result = _sp.run(
                [sys.executable, str(preflight_script), "--instances", tmp_instances],
                capture_output=True,
                text=True,
                check=False,
            )
            return result.returncode
        except Exception as exc:
            print(f"[run-swebench] Preflight raised: {exc}", file=sys.stderr)
            return EXIT_INCONCLUSIVE
        finally:
            try:
                os.unlink(tmp_instances)
            except OSError:
                pass

    def _run_generation(self, scaffold_config: dict) -> int:
        """Execute mini-swe-agent generation; return exit code.

        When generation_runner is injected (tests), delegates to it.
        Default live path calls mini-swe-agent 2.x as a subprocess (argv list, no shell)
        using module minisweagent.run.benchmarks.swebench with the injected scaffold
        config written to a temp YAML file. Captures subprocess output to raw/run.log
        for evidence. Writes preds.json, exit_statuses.json, and trajectories/ to raw/.

        Instance selection: builds an anchored OR-regex from the frozen 100 IDs so that
        --filter matches exactly the frozen set without relying on a local dataset file.
        """
        if self._generation_runner is not None:
            return int(self._generation_runner(scaffold_config, self.run_dir))

        # --- Live default: invoke mini-swe-agent 2.x as subprocess ---
        import re
        import subprocess as _sp
        import tempfile as _tf
        import yaml

        raw_dir = self.run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        run_log = raw_dir / "run.log"

        # Write the injected scaffold config to a temp file
        with _tf.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix="swe_config_", delete=False
        ) as tf:
            yaml.dump(scaffold_config, tf, default_flow_style=False)
            config_path = tf.name

        # Build an anchored exact-ID OR-regex for --filter
        # Each ID is anchor-escaped: ^ ID $ with | between them
        # re.escape handles any special chars in IDs (e.g. __ is safe but be defensive)
        anchored_ids = [f"^{re.escape(iid)}$" for iid in self._instance_ids]
        filter_regex = "|".join(anchored_ids)

        try:
            # mini-swe-agent 2.x invocation (argv only, no shell)
            # Outputs land in raw/ as preds.json + exit_statuses.json + trajectories/
            cmd = [
                sys.executable, "-m", "minisweagent.run.benchmarks.swebench",
                "--subset", "princeton-nlp/SWE-bench_Verified",
                "--split", "test",
                "--filter", filter_regex,
                "-c", config_path,
                "-w", str(self.workers),
                "-o", str(raw_dir),
            ]
            print(
                f"[run-swebench] Generation: {' '.join(shlex.quote(a) for a in cmd[:4])} "
                f"--filter <anchored-{len(self._instance_ids)}-id-regex> "
                f"-c {shlex.quote(config_path)} "
                f"-w {self.workers} -o {shlex.quote(str(raw_dir))}",
                file=sys.stderr,
            )
            with open(run_log, "w") as log_fh:
                result = _sp.run(
                    cmd,
                    stdout=log_fh,
                    stderr=_sp.STDOUT,
                    check=False,
                )
            return result.returncode
        except Exception as exc:
            print(
                f"[run-swebench] Generation subprocess failed: {exc}",
                file=sys.stderr,
            )
            # Write the error to run.log for evidence
            try:
                with open(run_log, "a") as log_fh:
                    log_fh.write(f"\nGeneration subprocess error: {exc}\n")
            except OSError:
                pass
            return EXIT_DEFECT
        finally:
            try:
                os.unlink(config_path)
            except OSError:
                pass

    def _run_grading(self, preds_path: pathlib.Path) -> int:
        """Execute SWE-bench grading; return exit code.

        When grading_runner is injected (tests), delegates to it.
        Default live path calls python -m swebench.harness.run_evaluation as a
        subprocess (argv only, no shell). Uses the frozen 100 instance IDs from
        the suite.

        The harness emits <model>.<run_id>.json under --report_dir. After a
        successful run, this method locates the exact report file, validates it,
        and normalizes it to grading_results.json with disjoint categories.
        """
        if self._grading_runner is not None:
            return int(self._grading_runner(preds_path, self.run_dir))

        # --- Live default: invoke swebench harness as subprocess ---
        import subprocess as _sp

        raw_dir = self.run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        grading_out = raw_dir / "grading_results.json"

        # Build argv list (no shell)
        # Official flags: -d dataset, -s split, -i instance_ids..., -p preds_path,
        #                 -id run_id, --report_dir output_dir
        cmd = [
            sys.executable, "-m", "swebench.harness.run_evaluation",
            "-d", "princeton-nlp/SWE-bench_Verified",
            "-s", "test",
            "-i", *self._instance_ids,
            "-p", str(preds_path),
            "-id", self._run_id,
            "--report_dir", str(raw_dir),
        ]
        print(
            f"[run-swebench] Grading: {' '.join(shlex.quote(a) for a in cmd[:6])} "
            f"[... {len(self._instance_ids)} instance IDs ...] "
            f"-p {shlex.quote(str(preds_path))} -id {shlex.quote(self._run_id)} "
            f"--report_dir {shlex.quote(str(raw_dir))}",
            file=sys.stderr,
        )
        try:
            result = _sp.run(
                cmd,
                capture_output=True,
                check=False,
                cwd=str(raw_dir),
            )
            if result.returncode != 0:
                return result.returncode

            # Locate the report file emitted by the harness.
            # The harness writes <model_slug>.<run_id>.json under --report_dir.
            # Scan for any JSON file matching *.<run_id>.json in raw_dir.
            report_files = sorted(raw_dir.glob(f"*.{self._run_id}.json"))
            if not report_files:
                # Fallback: scan for any .json that isn't our known files
                known = {"grading_results.json", "preds.json", "exit_statuses.json"}
                report_files = [
                    f for f in raw_dir.glob("*.json")
                    if f.name not in known
                ]
            if not report_files:
                print(
                    f"[run-swebench] Grading succeeded but no report JSON found in {raw_dir}. "
                    "grading_results.json will be missing.",
                    file=sys.stderr,
                )
                return EXIT_DEFECT

            # Use the first/only report file found
            report_path = report_files[0]
            if len(report_files) > 1:
                print(
                    f"[run-swebench] Multiple report files found: {[f.name for f in report_files]}. "
                    f"Using {report_path.name}.",
                    file=sys.stderr,
                )

            # Parse and normalize to grading_results.json
            try:
                raw_report = json.loads(report_path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                print(
                    f"[run-swebench] Could not parse grading report {report_path}: {exc}",
                    file=sys.stderr,
                )
                return EXIT_DEFECT

            # Normalize: extract disjoint ID sets from the harness report.
            # The harness report format: {instance_id: {"resolved": bool, ...}}
            # or top-level keys "resolved", "unresolved", "empty_patch", "error"
            normalized = _normalize_grading_report(raw_report, self._instance_ids)
            if normalized is None:
                print(
                    f"[run-swebench] Could not normalize grading report from {report_path.name}. "
                    "Unexpected format.",
                    file=sys.stderr,
                )
                return EXIT_DEFECT

            try:
                grading_out.write_text(json.dumps(normalized, indent=2))
            except OSError as exc:
                print(
                    f"[run-swebench] Could not write grading_results.json: {exc}",
                    file=sys.stderr,
                )
                return EXIT_DEFECT

            return 0
        except Exception as exc:
            print(
                f"[run-swebench] Grading subprocess failed: {exc}",
                file=sys.stderr,
            )
            return EXIT_DEFECT

    def _read_status_strict(self) -> dict:
        """Read and return status.json, raising on any validation failure."""
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

        # Schema validation
        if _STATUS_SCHEMA.exists():
            schema_errors = validate_json(status, _STATUS_SCHEMA)
            if schema_errors:
                raise ValueError(
                    f"status.json in {self.run_dir} fails schema validation:\n"
                    + "\n".join(schema_errors)
                )
        else:
            raise ValueError(f"result-status schema not found at {_STATUS_SCHEMA}")

        # run_id must match
        expected_run_id = self.run_dir.name
        actual_run_id = status.get("run_id", "")
        if actual_run_id != expected_run_id:
            raise ValueError(
                f"status.json run_id={actual_run_id!r} does not match "
                f"run_dir.name={expected_run_id!r}."
            )

        # suite_id must match
        actual_suite_id = status.get("suite_id", "")
        if actual_suite_id != self._suite_id:
            raise ValueError(
                f"status.json suite_id={actual_suite_id!r} does not match "
                f"loaded suite suite_id={self._suite_id!r}."
            )

        return status

    def _write_status(self, status: dict) -> None:
        campaign_state.write_status(
            dest=self.run_dir / "status.json",
            status=status,
            run_dir=self.run_dir,
        )

    def _write_manifest_atomic(self, manifest: dict) -> None:
        """Write manifest.json atomically via sibling-temp + fsync + os.replace.

        Uses the same durable-write pattern as campaign_state.write_status:
          1. Write to a sibling temp file in the same directory (same filesystem).
          2. fsync to flush to durable storage.
          3. os.replace (atomic on POSIX) to swap in the new file.

        Raises OSError on any failure; the destination is never half-written.
        """
        import tempfile as _tf
        manifest_path = self.run_dir / "manifest.json"
        payload = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
        fd, tmp_path = _tf.mkstemp(
            dir=str(self.run_dir),
            prefix=".tmp_manifest_",
            suffix=".json",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, str(manifest_path))
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _update_manifest_on_completion(self) -> None:
        """Update manifest.json with completion metadata BEFORE writing DONE.

        Sets:
          - timing.completed_utc = UTC timestamp
          - item_inventory.submitted = total count across all grading disposition categories
          - artifact_inventory.preds_json = True (if raw/preds.json exists)
          - artifact_inventory.exit_statuses_json = True (if raw/exit_statuses.json exists)
          - artifact_inventory.grading_results_json = True (if raw/grading_results.json exists)
          - artifact_inventory.run_log = True (if raw/run.log exists)
          - artifact_inventory.command_txt = True (if command.txt exists)
          - artifact_inventory.done_sentinel = False  ← DONE not yet written; must not claim True

        done_sentinel is set to True only AFTER DONE is successfully written, by a
        separate call to _mark_manifest_done_sentinel_true().

        Raises OSError if manifest cannot be read or written.
        Raises json.JSONDecodeError if manifest is corrupt.
        Raises ValueError if grading_results.json is malformed or missing required keys.
        """
        manifest_path = self.run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        # Update timing
        timing = manifest.setdefault("timing", {})
        timing["completed_utc"] = _utcnow()

        # Compute submitted count from grading_results.json.
        # Grading has already been verified by _verify_grading_evidence before this
        # method is called — any malformed or missing grading file would have caused
        # an earlier EXIT_DEFECT.  We still parse strictly here (fail-closed, not
        # best-effort) because the submitted count must equal the actual disposition
        # union and silent submitted=0 would be a correctness violation.
        raw_dir = self.run_dir / "raw"
        grading_path = raw_dir / "grading_results.json"
        if not grading_path.exists():
            raise ValueError(
                f"grading_results.json not found at {grading_path}. "
                "Cannot compute submitted count for manifest — grading evidence is required."
            )
        try:
            grading = json.loads(grading_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(
                f"grading_results.json at {grading_path} is not valid JSON: {exc}. "
                "Manifest submitted count cannot be computed from malformed grading data."
            ) from exc

        if not isinstance(grading, dict):
            raise ValueError(
                f"grading_results.json must be a JSON object; got {type(grading).__name__}. "
                "Cannot compute submitted count."
            )

        _DISPOSITION_KEYS = (
            "resolved_ids", "unresolved_ids", "empty_patch_ids",
            "error_ids", "incomplete_ids",
        )
        submitted_count = 0
        seen_ids: set = set()
        for key in _DISPOSITION_KEYS:
            ids = grading.get(key) or []
            if not isinstance(ids, list):
                raise ValueError(
                    f"grading_results.json[{key!r}] must be a list; "
                    f"got {type(ids).__name__}."
                )
            for iid in ids:
                if iid in seen_ids:
                    raise ValueError(
                        f"Instance ID {iid!r} appears in multiple grading disposition "
                        "categories. Disposition union must be disjoint."
                    )
                seen_ids.add(iid)
            submitted_count += len(ids)

        # submitted must equal the number of expected instances
        expected_count = manifest.get("item_inventory", {}).get("expected", 0)
        if expected_count and submitted_count != expected_count:
            raise ValueError(
                f"Grading disposition union covers {submitted_count} instances but "
                f"manifest.item_inventory.expected = {expected_count}. "
                "submitted must equal expected — all instances must have a terminal disposition."
            )

        inventory = manifest.setdefault("item_inventory", {})
        inventory["submitted"] = submitted_count

        # Update artifact_inventory — done_sentinel stays False here.
        # It will be set to True only after DONE is durably written.
        art = manifest.setdefault("artifact_inventory", {})
        art["preds_json"] = (raw_dir / "preds.json").exists()
        art["exit_statuses_json"] = (raw_dir / "exit_statuses.json").exists()
        art["grading_results_json"] = grading_path.exists()
        art["run_log"] = (raw_dir / "run.log").exists()
        art["command_txt"] = (self.run_dir / "command.txt").exists()
        # IMPORTANT: done_sentinel MUST be False here — DONE has not been written yet.
        # Setting it True before DONE exists would create an inconsistent manifest state.
        art["done_sentinel"] = False

        # Atomic write: sibling-temp + fsync + os.replace
        self._write_manifest_atomic(manifest)

    def _mark_manifest_done_sentinel_true(self) -> None:
        """Update manifest.json to set artifact_inventory.done_sentinel = True.

        Called AFTER DONE is successfully written on disk.  Uses the same atomic
        write mechanism as _update_manifest_on_completion.

        Raises OSError or json.JSONDecodeError on failure.
        """
        manifest_path = self.run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        art = manifest.setdefault("artifact_inventory", {})
        art["done_sentinel"] = True
        self._write_manifest_atomic(manifest)

    def _transition_completed_atomic(self) -> int:
        """Update manifest, write DONE, mark done_sentinel, write completed status. Returns 0 or EXIT_DEFECT.

        Lifecycle atomicity (fail-closed):
          0. Update manifest.json with completion metadata (timing, submitted count,
             artifact_inventory booleans). done_sentinel stays False — DONE not yet written.
             Fail closed if this fails — no DONE written.
          1. Write DONE via _write_done (default: atomic staged temp + os.replace).
             If this fails, status remains at 'running' → transition to failed.
             No inconsistency: no DONE, no completed status, done_sentinel still False.
          2. Mark manifest done_sentinel = True (atomic write, DONE now on disk).
             If this fails: DONE exists but done_sentinel is False — inconsistent.
             Revert: remove DONE, transition running → failed, persist failed manifest.
          3. Write completed status to disk (DONE + done_sentinel both durable).
             If this fails: DONE exists but status is 'running' (inconsistent).
             Revert: attempt to delete DONE, re-write manifest done_sentinel=False,
             then transition running → failed.

        This ordering ensures: if status == 'completed', DONE definitely exists and
        manifest.done_sentinel is True.
        """
        done_path = self.run_dir / "DONE"

        # Step 0: Update manifest.json before DONE (done_sentinel=False, fail-closed)
        try:
            self._update_manifest_on_completion()
        except Exception as exc:
            print(
                f"[run-swebench] FATAL: Cannot update manifest.json: {exc}. "
                "DONE not written.",
                file=sys.stderr,
            )
            if not self._transition_failed(note="manifest update failed before DONE"):
                print(
                    "[run-swebench] FATAL: _transition_failed write failed after manifest update failure. "
                    "Run is in an indeterminate state.",
                    file=sys.stderr,
                )
            return EXIT_DEFECT

        # Step 1: Write DONE sentinel atomically (status stays at 'running')
        try:
            self._write_done(done_path)
        except OSError as exc:
            print(
                f"[run-swebench] FATAL: Cannot write DONE sentinel: {exc}. "
                "Transitioning to failed — completed status NOT written.",
                file=sys.stderr,
            )
            if not self._transition_failed(note="DONE sentinel write failed"):
                print(
                    "[run-swebench] FATAL: _transition_failed write failed after DONE sentinel failure. "
                    "Run is in an indeterminate state.",
                    file=sys.stderr,
                )
            return EXIT_DEFECT

        # Step 2: Mark manifest done_sentinel = True (DONE now on disk).
        # If this fails, revert DONE so no inconsistency persists.
        try:
            self._mark_manifest_done_sentinel_true()
        except Exception as exc:
            print(
                f"[run-swebench] FATAL: Cannot mark manifest done_sentinel=True after DONE write: {exc}. "
                "Reverting DONE and transitioning to failed.",
                file=sys.stderr,
            )
            try:
                done_path.unlink(missing_ok=True)
            except OSError:
                pass
            if not self._transition_failed(note="manifest done_sentinel update failed; DONE reverted"):
                print(
                    "[run-swebench] FATAL: _transition_failed write failed after done_sentinel update failure. "
                    "Run is in an indeterminate state — DONE has been reverted.",
                    file=sys.stderr,
                )
            return EXIT_DEFECT

        # Step 3: Write completed status (DONE + done_sentinel both durable).
        try:
            status = self._read_status_strict()
            new_status = campaign_state.apply_transition(
                status=status,
                new_state="completed",
                timestamp=_utcnow(),
            )
            # Add grading note to completed entry
            if new_status.get("history"):
                new_status["history"][-1]["note"] = "grading completed; all dispositions verified"
            # Schema-validate before writing
            if _STATUS_SCHEMA.exists():
                schema_errors = validate_json(new_status, _STATUS_SCHEMA)
                if schema_errors:
                    raise ValueError(
                        f"Completed status failed schema validation:\n"
                        + "\n".join(schema_errors)
                    )
            self._write_status(new_status)
            return EXIT_SUCCESS
        except Exception as exc:
            print(
                f"[run-swebench] FATAL: Completed status write failed: {exc}. "
                "DONE sentinel was already written. Reverting DONE and transitioning to failed.",
                file=sys.stderr,
            )
            # Revert DONE so invariant (DONE ↔ completed) is preserved
            try:
                done_path.unlink(missing_ok=True)
            except OSError:
                pass
            # Re-write manifest to clear done_sentinel=True (best-effort, don't re-raise)
            try:
                manifest_path = self.run_dir / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                art = manifest.setdefault("artifact_inventory", {})
                art["done_sentinel"] = False
                self._write_manifest_atomic(manifest)
            except Exception as manifest_exc:
                print(
                    f"[run-swebench] WARNING: Could not revert manifest done_sentinel to False "
                    f"after status write failure: {manifest_exc}. "
                    "Manifest may be inconsistent with DONE state.",
                    file=sys.stderr,
                )
            # Transition to failed (still in 'running' state, so this is legal)
            if not self._transition_failed(note="completed status write failed; DONE reverted"):
                print(
                    "[run-swebench] FATAL: _transition_failed write failed after completed status failure. "
                    "Run is in an indeterminate state — DONE has been reverted.",
                    file=sys.stderr,
                )
            return EXIT_DEFECT

    def _write_done(self, done_path: pathlib.Path) -> None:
        """Write the DONE sentinel atomically via a staged temp file + os.replace.

        Uses a sibling temp file in the same directory so os.replace is atomic
        on POSIX (same filesystem). Raises OSError on any failure.

        Tests may override this by passing done_writer= to the constructor.
        """
        if self._done_writer is not None:
            self._done_writer(done_path)
            return

        import tempfile as _tf
        # Write to a sibling temp file first, then atomically rename
        done_dir = done_path.parent
        fd, tmp_path = _tf.mkstemp(dir=str(done_dir), prefix=".DONE_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write("completed\n")
            os.replace(tmp_path, str(done_path))
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _transition_failed(self, note: str = "") -> bool:
        """Transition to failed; return whether the durable state write succeeded."""
        try:
            status = self._read_status_strict()
            new_status = campaign_state.apply_transition(
                status=status,
                new_state="failed",
                timestamp=_utcnow(),
                lifecycle="invalid",
            )
            if note and new_status.get("history"):
                new_status["history"][-1]["note"] = note
            self._write_status(new_status)
            return True
        except Exception as exc:
            print(
                f"[run-swebench] FATAL: status transition to 'failed' failed: {exc}",
                file=sys.stderr,
            )
            return False


# ---------------------------------------------------------------------------
# Prompt tokens parser (same pattern as run_quality.py)
# ---------------------------------------------------------------------------


def _parse_prompt_tokens(value: str) -> Dict[str, int]:
    """Parse --prompt-tokens: 'gsm8k=500,ifeval=2000' -> {'gsm8k': 500, 'ifeval': 2000}."""
    result: Dict[str, int] = {}
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                f"Invalid --prompt-tokens format {part!r}. "
                "Expected: benchmark=tokens (e.g. gsm8k=500,ifeval=2000)"
            )
        bench, _, tok_str = part.partition("=")
        bench = bench.strip()
        tok_str = tok_str.strip()
        try:
            tokens = int(tok_str)
        except ValueError:
            raise ValueError(
                f"Invalid token count {tok_str!r} for benchmark {bench!r}."
            )
        if tokens < 0:
            raise ValueError(
                f"Token count must be non-negative for benchmark {bench!r}; got {tokens}."
            )
        result[bench] = tokens
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Contract-aware SWE-bench runner for warpcore-v1."
    )
    ap.add_argument("--suite", required=True, type=pathlib.Path)
    ap.add_argument("--adapter", required=True, type=pathlib.Path)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--run-dir", type=pathlib.Path, default=None)
    ap.add_argument("--repo", type=pathlib.Path, default=None)
    ap.add_argument("--api-key", default="warpcore")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--prompt-tokens", type=str, default=None,
                    help="Measured prompt maxima for quality benchmarks: 'bench=N,...' "
                         "(e.g. gsm8k=500,ifeval=2000,gpqa_diamond=1000). Required for "
                         "adapter campaign-readiness validation.")
    ap.add_argument("--qualification", type=pathlib.Path, default=None,
                    help="Qualification record authorizing this launch. Defaults to "
                         "<repo>/results/<slug>/qualification/<suite-id>/swebench/"
                         "qualification.json.")
    ap.add_argument("--qualification-run", action="store_true",
                    help="Run the 20 suite-owned qualification instances instead of the "
                         "frozen 100. Produces the evidence a qualification record is "
                         "sealed from; writes no campaign state and is not itself gated.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-no-screen", action="store_true")
    ap.add_argument("--resume", action="store_true")
    # model arg for command.txt record only (derived from adapter in runner)
    ap.add_argument("--model", default=None, help=argparse.SUPPRESS)

    args = ap.parse_args(argv)

    # Parse prompt_token_maxima
    prompt_token_maxima: Optional[Dict[str, int]] = None
    if args.prompt_tokens is not None:
        try:
            prompt_token_maxima = _parse_prompt_tokens(args.prompt_tokens)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return EXIT_CONFIG

    suite_path = pathlib.Path(args.suite).resolve()
    repo = pathlib.Path(args.repo).resolve() if args.repo else suite_path.parent.parent

    # --- Qualification run: the mode that produces the record ---
    # Not gated (it is what authorizes later launches) and deliberately outside
    # create_campaign: a qualification is not a campaign.
    if args.qualification_run:
        import yaml as _yaml_q
        with open(pathlib.Path(args.adapter).resolve()) as fh:
            _adapter_q = _yaml_q.safe_load(fh)
        with open(suite_path) as fh:
            _suite_q = _yaml_q.safe_load(fh)
        _slug_q = (_adapter_q.get("model") or {}).get("slug", "unknown")
        _suite_id_q = _suite_q.get("suite_id", "warpcore-v1")
        qual_run_dir = (
            pathlib.Path(args.run_dir).resolve() if args.run_dir is not None
            else swebench_qualification.default_artifact_path(
                repo, _suite_id_q, _slug_q
            ).parent / "run"
        )
        try:
            runner = SwebenchRunner(
                suite_path=args.suite,
                adapter_path=args.adapter,
                endpoint=args.endpoint,
                run_dir=qual_run_dir,
                repo=repo,
                api_key=args.api_key,
                workers=args.workers,
                dry_run=args.dry_run,
                allow_no_screen=args.allow_no_screen,
                prompt_token_maxima=prompt_token_maxima,
                qualification_run=True,
            )
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return EXIT_CONFIG
        return runner.run() if args.dry_run else runner.run_qualification()

    # --- Qualification gate, before create_campaign touches the filesystem ---
    # A live launch must fail closed with no campaign directory, no status.json,
    # and no manifest when it is not authorized. A dry run skips this check here
    # because the runner reports the same verdict without side effects.
    if not args.dry_run:
        try:
            import yaml as _yaml_gate
            _suite_gate = _yaml_gate.safe_load(suite_path.read_text())
            _adapter_gate = _yaml_gate.safe_load(
                pathlib.Path(args.adapter).resolve().read_text()
            )
            _suite_id_gate = _suite_gate.get("suite_id", "warpcore-v1")
            _slug_gate = (_adapter_gate.get("model") or {}).get("slug", "unknown")
        except Exception as exc:
            print(f"ERROR: cannot read suite or adapter for qualification: {exc}", file=sys.stderr)
            return EXIT_CONFIG
        artifact_path = args.qualification or swebench_qualification.default_artifact_path(
            repo, _suite_id_gate, _slug_gate
        )
        verdict = swebench_qualification.verify_qualification_for_launch(
            repo=_REPO_DIR,
            suite_path=suite_path,
            adapter_path=pathlib.Path(args.adapter).resolve(),
            endpoint=args.endpoint,
            artifact_path=artifact_path,
            api_key=args.api_key,
        )
        if not verdict.ok:
            print(verdict.render(), file=sys.stderr)
            print(
                f"ERROR: refusing to create or launch a canonical SWE-bench campaign "
                f"without a valid qualification ({artifact_path}).",
                file=sys.stderr,
            )
            return EXIT_DEFECT

    # Resolve run directory
    if args.run_dir is not None:
        explicit_run_dir = pathlib.Path(args.run_dir).resolve()
        if args.dry_run:
            # Dry-run: use the explicit path directly (no side effects)
            run_dir = explicit_run_dir
        else:
            # Live run: explicit --run-dir implies resume=True.
            # Derive run_id from the path and call create_campaign(resume=True)
            # to validate the run dir and demand the normalized path matches.
            run_id = explicit_run_dir.name
            import yaml as _yaml
            with open(suite_path) as fh:
                suite = _yaml.safe_load(fh)
            suite_id = suite.get("suite_id", "warpcore-v1")
            try:
                import create_campaign as cc_mod
                normalized_run_dir = cc_mod.create_campaign(
                    repo=repo,
                    suite_path=suite_path,
                    adapter_path=pathlib.Path(args.adapter).resolve(),
                    benchmark=_SWEBENCH_BENCH,
                    run_id=run_id,
                    resume=True,
                    prompt_token_maxima=prompt_token_maxima,
                )
                if pathlib.Path(normalized_run_dir).resolve() != explicit_run_dir:
                    print(
                        f"ERROR: Explicit --run-dir {explicit_run_dir} does not match "
                        f"create_campaign normalized path {normalized_run_dir}. "
                        "The run directory must be at the canonical location.",
                        file=sys.stderr,
                    )
                    return EXIT_CONFIG
                run_dir = pathlib.Path(normalized_run_dir).resolve()
            except Exception as exc:
                print(f"ERROR: create_campaign (resume) failed: {exc}", file=sys.stderr)
                return EXIT_CONFIG
    else:
        import yaml
        with open(suite_path) as fh:
            suite = yaml.safe_load(fh)
        with open(pathlib.Path(args.adapter).resolve()) as fh:
            adapter = yaml.safe_load(fh)

        suite_id = suite.get("suite_id", "warpcore-v1")
        model_slug = (adapter.get("model") or {}).get("slug", "unknown")
        run_id = args.run_id or datetime.now(tz=timezone.utc).strftime("run-%Y-%m-%dT%H-%M-%S")

        if args.dry_run:
            run_dir = (
                repo / "results" / model_slug / "runs"
                / suite_id / _SWEBENCH_BENCH / run_id
            )
        else:
            try:
                import create_campaign as cc_mod
                run_dir = cc_mod.create_campaign(
                    repo=repo,
                    suite_path=suite_path,
                    adapter_path=pathlib.Path(args.adapter).resolve(),
                    benchmark=_SWEBENCH_BENCH,
                    run_id=run_id,
                    resume=args.resume,
                    prompt_token_maxima=prompt_token_maxima,
                )
            except Exception as exc:
                print(f"ERROR: create_campaign failed: {exc}", file=sys.stderr)
                return EXIT_CONFIG

    try:
        runner = SwebenchRunner(
            suite_path=args.suite,
            adapter_path=args.adapter,
            endpoint=args.endpoint,
            run_dir=run_dir,
            repo=repo,
            api_key=args.api_key,
            workers=args.workers,
            dry_run=args.dry_run,
            allow_no_screen=args.allow_no_screen,
            prompt_token_maxima=prompt_token_maxima,
            qualification_path=args.qualification,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
