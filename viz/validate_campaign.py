"""viz/validate_campaign.py — Offline campaign validator (fail-closed).

Validates a completed campaign run before it can be transitioned to 'validated'
or published as canonical. Current v1 runs are gated hard; historical debt
remains in the ratchet but cannot become canonical and must not be certified.

Public API
----------
validate(run_dir, suite_path, adapter_path, *, for_publication=False) -> ValidationResult
discover_runs(repo) -> list[dict]
validate_status_consistency(status) -> list[str]

ValidationResult has:
    .passed  (bool)  — True only when all gates pass and run is eligible
    .errors  (list[str])
    .eligible (bool)  — False when run is historical/cannot be certified

Gates enforced (current-lifecycle runs only, unless noted):
  V1  No missing or duplicate item IDs in per-item evidence.
  V2  Raw samples (*.jsonl.gz) and all required manifest fields present.
  V3  Aggregate score reconciles with per-item evidence (within tolerance).
       For publication runs, missing aggregate file is a hard failure.
  V4  Every item has a classified disposition from the allowlist (unknown fails).
  V5  Suite and adapter hashes match the actual files on disk.
  V6  effective_args must not contain suite-owned field overrides.
  V7  lifecycle must be 'current' when for_publication=True.
  V8  Secret-scan: no credential patterns in staged artifacts.
  V9  DONE sentinel: must not exist for failed runs; when present for
      completed/validated runs, all artifact_inventory=True items must exist.
  V10 Historical runs: eligible=False, passed=False — not certified by this validator.
  V11 JSON schema validation for manifest.json and status.json.
  V12 Identity agreement: run_id matches among dir name, manifest, status;
      suite_id matches between manifest and status.

Layout support
--------------
Normalized layout  : results/<model>/runs/<suite>/<bench>/<run_id>/
Historical layout  : results/<model>/raw/  (or results/<model>/raw/<bench>/)
Both layouts share the same discovery path via discover_runs().
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import pathlib
import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import yaml as _yaml  # type: ignore[import]
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

# Allow import from viz/ when run directly
_VIZ_DIR = pathlib.Path(__file__).parent
if str(_VIZ_DIR) not in sys.path:
    sys.path.insert(0, str(_VIZ_DIR))

# ---------------------------------------------------------------------------
# Forbidden effective_args patterns (suite-owned fields)
# ---------------------------------------------------------------------------
# An adapter or command-line must never override these suite-owned variables.

# Sentinel object for distinguishing "key absent from dict" from "key present with None value"
_ABSENT = object()
_FORBIDDEN_ARG_PATTERNS: list[str] = [
    r"--tasks?=",
    r"--tasks?\s",
    r"--num_fewshot",
    r"--temperature",
    r"--do_sample",
    r"--max_gen_toks?=",
    r"--max_new_tokens?=",
    r"--max_length=",
    r"--system_instruction",
    r"--prompt",
    r"--dataset",
    r"--filter",
    r"--scorer",
]

# ---------------------------------------------------------------------------
# Secret-scan patterns (heuristics over known credential shapes)
# ---------------------------------------------------------------------------
_SECRET_PATTERNS: list[re.Pattern] = [
    re.compile(r"sk-[A-Za-z0-9]{20,}", re.I),         # OpenAI-style keys
    re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),          # GitHub token shapes
    re.compile(r"api[_-]?key\s*=\s*['\"]?\S{8,}", re.I),
    re.compile(r"Authorization:\s*Bearer\s+\S{8,}", re.I),
    re.compile(r"OPENAI_API_KEY\s*=\s*\S+"),
    re.compile(r"--api[_-]?key\s+\S{8,}", re.I),
    re.compile(r"--api[_-]?key=\S{8,}", re.I),
]

# A full canonical SWE-bench campaign can exceed 175 MiB.  On the reference
# Intel Mac mini, gitleaks needs about 45 seconds for that tree; retain a bounded
# timeout while leaving conservative headroom for filesystem variance.
_GITLEAKS_SCAN_TIMEOUT_SECONDS = 120

# Files to scan for secrets
_SECRET_SCAN_FILES = ["command.txt", "run.log"]

# Well-known install locations for gitleaks on Intel and Apple Silicon macOS,
# checked in order when PATH lookup and GITLEAKS_BIN env override both fail.
_GITLEAKS_WELL_KNOWN: list[str] = [
    "/usr/local/bin/gitleaks",      # Homebrew Intel Mac
    "/opt/homebrew/bin/gitleaks",   # Homebrew Apple Silicon
    "/usr/bin/gitleaks",            # System / Linux
    "/usr/local/sbin/gitleaks",
]


def _resolve_gitleaks_bin() -> "Optional[str]":
    """Locate the gitleaks executable via env override, PATH, or well-known paths.

    Resolution order (fail-closed):
    1. ``GITLEAKS_BIN`` env var — must point to an executable file; if set but
       invalid the resolver returns ``None`` (callers treat that as missing).
    2. ``shutil.which("gitleaks")`` — searches the process PATH.
    3. Well-known install locations for Intel and Apple Silicon macOS.

    Returns the absolute path string when a usable binary is found, else None.
    """
    import os
    import shutil

    # 1. Explicit env override
    env_bin = os.environ.get("GITLEAKS_BIN", "").strip()
    if env_bin:
        if os.path.isfile(env_bin) and os.access(env_bin, os.X_OK):
            return env_bin
        # Override set but invalid — do NOT fall through to PATH/well-known;
        # an explicit override that doesn't work is a configuration error.
        return None

    # 2. PATH lookup
    path_bin = shutil.which("gitleaks")
    if path_bin:
        return path_bin

    # 3. Well-known locations (covers stripped CI/make PATH)
    for candidate in _GITLEAKS_WELL_KNOWN:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    return None

# Valid disposition values for classified items (fail-closed allowlist)
_VALID_DISPOSITIONS = frozenset({
    "correct",
    "wrong",
    "empty",           # empty generation (not parser defect)
    "budget",          # finish_reason=length (budget truncated)
    "parser",          # ISSUES #15: answer in reasoning field
    "timeout",         # request timed out
    "transport",       # transport/server error
    "infrastructure",  # SWE-bench infrastructure failure
    "swebench_fail",   # SWE-bench graded failure
    "submitted_wrong", # submitted but wrong
    "non_submission",  # model did not submit
    "harness_fail",    # harness/scaffold defect
    "blocked",         # explicitly blocked with DIAGNOSIS
})

# Aggregate score tolerance (allow floating-point slop up to this threshold)
_AGGREGATE_TOLERANCE = 0.005  # 0.5pp

# Schema paths
_SCHEMA_DIR = _VIZ_DIR.parent / "suite" / "schemas"
_MANIFEST_SCHEMA = _SCHEMA_DIR / "manifest.schema.json"
_STATUS_SCHEMA = _SCHEMA_DIR / "result-status.schema.json"

# ---------------------------------------------------------------------------
# Suite-driven evidence: canonical mapping from evidence name → file check
# ---------------------------------------------------------------------------
# Each key is the evidence name used in suite required_evidence lists.
# Value is a tuple: (relative_path_from_run_dir_or_raw_dir, is_in_raw_subdir)
# Special value None means the evidence is validated structurally elsewhere.
_EVIDENCE_NAME_MAP: dict[str, tuple[str, bool]] = {
    # SWE-bench evidence names
    "preds_json":              ("preds.json",           True),   # raw/preds.json
    "exit_statuses":           ("exit_statuses.json",   True),   # raw/exit_statuses.json
    "grading_results_json":    ("grading_results.json", True),   # raw/grading_results.json
    "trajectories":            ("trajectories",         True),   # raw/trajectories/ dir
    # Quality benchmark evidence names
    "aggregate_result":        ("results_*.json",       True),   # raw/results_*.json (glob)
    "samples_jsonl_gz":        ("samples_*.jsonl.gz",   True),   # raw/samples_*.jsonl.gz (glob)
    "per_item_csv":            ("per_item.csv",         False),  # per_item.csv at run root
    "task_yaml":               (None,                   False),  # structural (task file hash)
    "scoring_implementation":  (None,                   False),  # structural (utils hash)
    "completion_token_counts": (None,                   False),  # embedded in samples
    "finish_reasons":          (None,                   False),  # embedded in samples
    "full_response_fields":    (None,                   False),  # embedded in samples
    # Universal evidence names (all benchmarks)
    "run_log":       ("run.log",    False),
    "command_txt":   ("command.txt",False),
    "manifest_json": ("manifest.json", False),
    "status_json":   ("status.json", False),
    "done_sentinel": ("DONE",       False),
}
# Frozenset of known evidence names (for unknown-name rejection)
_KNOWN_EVIDENCE_NAMES: frozenset[str] = frozenset(_EVIDENCE_NAME_MAP.keys())


# ---------------------------------------------------------------------------
# Suite YAML loading helpers
# ---------------------------------------------------------------------------

def _load_suite_yaml(suite_path: pathlib.Path, errors: list) -> Optional[dict]:
    """Load and parse the suite YAML file. Returns None on failure (error appended)."""
    if not _YAML_AVAILABLE:
        errors.append(
            "PyYAML is not installed; suite YAML cannot be parsed. "
            "Install pyyaml to enable suite-driven validation gates."
        )
        return None
    try:
        with suite_path.open(encoding="utf-8") as fh:
            data = _yaml.safe_load(fh)
        if not isinstance(data, dict):
            errors.append(
                f"suite YAML at {suite_path} did not parse as a dict "
                f"(got {type(data).__name__}). Suite validation cannot proceed."
            )
            return None
        return data
    except Exception as exc:
        errors.append(f"Failed to parse suite YAML {suite_path}: {exc}")
        return None


def _get_suite_benchmark_config(
    suite_data: dict, benchmark: str
) -> Optional[dict]:
    """Return the benchmark config dict from suite_data, or None if not present."""
    benchmarks = suite_data.get("benchmarks", {})
    if not isinstance(benchmarks, dict):
        return None
    cfg = benchmarks.get(benchmark)
    if not isinstance(cfg, dict):
        return None
    return cfg


def _load_frozen_instance_ids(
    suite_data: dict,
    suite_path: pathlib.Path,
    errors: list,
) -> Optional[frozenset]:
    """Load and verify the frozen SWE-bench instance ID set from suite YAML.

    Resolves instance_set_file relative to the suite YAML's parent directory
    (i.e., the repo root).  Verifies:
      1. Path containment: resolved path must be inside the repo root.
      2. File exists on disk.
      3. SHA-256 matches suite instances_sha256.
      4. File parses as a JSON list of unique strings.
      5. Length matches suite expected_item_count.

    Returns the frozenset of IDs on success, None on any failure.
    """
    repo_root = suite_path.parent.parent  # suite_path = .../suite/warpcore-v1.yaml

    swe_cfg = _get_suite_benchmark_config(suite_data, "swebench")
    if swe_cfg is None:
        # Suite doesn't define a swebench benchmark — skip frozen ID loading
        return None

    instance_set_file = swe_cfg.get("instance_set_file")
    instances_sha256 = swe_cfg.get("instances_sha256")
    expected_item_count = swe_cfg.get("expected_item_count")

    if not instance_set_file:
        errors.append(
            "Suite swebench benchmark config is missing 'instance_set_file'. "
            "A frozen instance set is required for authoritative SWE-bench validation."
        )
        return None

    if not instances_sha256:
        errors.append(
            "Suite swebench benchmark config is missing 'instances_sha256'. "
            "The SHA-256 hash of the instance file is required to verify integrity."
        )
        return None

    # Resolve path with containment check
    try:
        abs_path = (repo_root / instance_set_file).resolve()
        repo_resolved = repo_root.resolve()
        try:
            abs_path.relative_to(repo_resolved)
        except ValueError:
            errors.append(
                f"Suite instance_set_file {instance_set_file!r} resolves outside "
                f"the repo root ({repo_resolved}). Path traversal/symlink escape is not allowed."
            )
            return None
    except OSError as exc:
        errors.append(
            f"Could not resolve suite instance_set_file path {instance_set_file!r}: {exc}"
        )
        return None

    if not abs_path.exists():
        errors.append(
            f"Suite instance_set_file not found: {instance_set_file!r} "
            f"(expected at {abs_path}). The frozen instance file must exist."
        )
        return None

    # Hash verification
    actual_sha256 = hashlib.sha256(abs_path.read_bytes()).hexdigest()
    if actual_sha256 != instances_sha256:
        errors.append(
            f"Suite instances_sha256 mismatch for {instance_set_file!r}: "
            f"suite records {instances_sha256[:16]}... but actual is {actual_sha256[:16]}... "
            "The instance set file has been modified or the suite hash is wrong."
        )
        return None

    # Parse and validate the instance list
    try:
        raw = json.loads(abs_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        errors.append(
            f"Suite instance_set_file {instance_set_file!r} is not valid JSON: {exc}"
        )
        return None

    if not isinstance(raw, list):
        errors.append(
            f"Suite instance_set_file {instance_set_file!r} must be a JSON array "
            f"of instance ID strings; got {type(raw).__name__}."
        )
        return None

    for idx, item in enumerate(raw):
        if not isinstance(item, str) or not item.strip():
            errors.append(
                f"Suite instance_set_file {instance_set_file!r} item [{idx}] is not "
                f"a non-empty string: {item!r}. All entries must be unique non-empty strings."
            )
            return None

    # Uniqueness check
    seen: set[str] = set()
    duplicates: list[str] = []
    for item in raw:
        if item in seen:
            duplicates.append(item)
        seen.add(item)
    if duplicates:
        errors.append(
            f"Suite instance_set_file {instance_set_file!r} contains duplicate IDs: "
            f"{sorted(set(duplicates))[:10]}. Instance IDs must be unique."
        )
        return None

    frozen = frozenset(raw)

    # Count check
    if expected_item_count is not None and len(frozen) != expected_item_count:
        errors.append(
            f"Suite instance_set_file contains {len(frozen)} unique IDs but "
            f"suite expected_item_count={expected_item_count}. These must agree."
        )
        return None

    return frozen


# ---------------------------------------------------------------------------
# Suite-driven evidence gate
# ---------------------------------------------------------------------------

def _check_sample_field_evidence(
    run_dir: pathlib.Path,
    ev_name: str,
    errors: list,
) -> None:
    """Check quality response evidence in the reconciled metadata sidecar.

    Used for: completion_token_counts, finish_reasons, full_response_fields.

    The production runner captures these fields before lm-eval discards them.
    Validate the complete sample/sidecar inventory, never an existential field
    in one sample row.
    """
    import gzip as _gzip_mod
    raw_dir = run_dir / "raw"
    sample_files = sorted(raw_dir.rglob("samples_*.jsonl.gz")) if raw_dir.exists() else []
    if not sample_files:
        errors.append(
            f"Suite required_evidence '{ev_name}' requires inspection of retained sample "
            f"records in samples_*.jsonl.gz under {raw_dir}, but no samples files were found. "
            "Evidence cannot be verified without retained samples (fail-closed)."
        )
        return

    metadata_path = raw_dir / "response_metadata.jsonl"
    if not metadata_path.is_file():
        errors.append(
            f"Suite required_evidence '{ev_name}' requires {metadata_path}, but the "
            "response metadata sidecar is absent."
        )
        return

    try:
        from lmeval_sidecar.reconcile import reconcile_inventory
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        expected_model = (manifest.get("model") or {}).get("id")
        report = reconcile_inventory(
            sample_files, metadata_path,
            expected_run_id=run_dir.name,
            expected_model=expected_model,
        )
    except Exception as exc:
        errors.append(
            f"Suite required_evidence '{ev_name}' could not reconcile response "
            f"metadata against retained samples: {exc}"
        )
        return

    if report.get("exit_code") != 0:
        errors.append(
            f"Suite required_evidence '{ev_name}' failed whole-inventory response "
            "metadata reconciliation: " + "; ".join(report.get("errors", []))
        )


def _check_suite_required_evidence(
    run_dir: pathlib.Path,
    suite_data: dict,
    benchmark: str,
    errors: list,
) -> None:
    """Check that all required_evidence named in suite YAML are present on disk.

    This gate is authoritative over manifest.artifact_inventory booleans:
    if the suite says an evidence file is required, it must be present,
    regardless of what the manifest claims.

    Unknown evidence names in required_evidence are fail-closed configuration errors.

    Benchmark-aware overrides:
      run_log: SWE-bench uses raw/run.log; quality benchmarks use run_dir/run.log.
      task_yaml: verifies task_file hash is recorded in manifest.suite_input_hashes.
      scoring_implementation: verifies utils_file hash is in manifest.suite_input_hashes.
      completion_token_counts: inspects samples for completion_tokens/token_count field.
      finish_reasons: inspects samples for finish_reason field.
      full_response_fields: inspects samples for resps/filtered_resps fields.
    """
    cfg = _get_suite_benchmark_config(suite_data, benchmark)
    if cfg is None:
        return  # Suite doesn't define this benchmark — no suite-driven check

    required_evidence = cfg.get("required_evidence", [])
    if not isinstance(required_evidence, list):
        errors.append(
            f"Suite benchmark '{benchmark}'.required_evidence must be a list; "
            f"got {type(required_evidence).__name__}."
        )
        return

    raw_dir = run_dir / "raw"
    _SWE_BENCHMARKS = frozenset({"swebench"})

    # Load the manifest once for hash-based checks (task_yaml, scoring_implementation)
    manifest_path = run_dir / "manifest.json"
    manifest_suite_input_hashes: dict = {}
    if manifest_path.exists():
        try:
            manifest_suite_input_hashes = (
                json.loads(manifest_path.read_text(encoding="utf-8"))
                .get("suite_input_hashes", {})
            )
        except (json.JSONDecodeError, OSError):
            pass  # hash checks will fail below when manifest is unreadable

    for ev_name in required_evidence:
        if not isinstance(ev_name, str):
            errors.append(
                f"Suite required_evidence entry {ev_name!r} is not a string. "
                "All required_evidence entries must be string evidence names."
            )
            continue

        if ev_name not in _KNOWN_EVIDENCE_NAMES:
            errors.append(
                f"Suite required_evidence contains unknown evidence name {ev_name!r}. "
                f"Known names: {sorted(_KNOWN_EVIDENCE_NAMES)}. "
                "Unknown evidence names are rejected (fail-closed configuration error)."
            )
            continue

        # ---------------------------------------------------------------------------
        # Benchmark-aware structural evidence checks (no longer silent no-ops)
        # ---------------------------------------------------------------------------

        if ev_name == "run_log":
            # SWE-bench: run.log lives under raw/; quality: at run_dir root.
            if benchmark in _SWE_BENCHMARKS:
                p = raw_dir / "run.log"
                if not p.exists():
                    errors.append(
                        f"Suite required_evidence 'run_log' requires 'run.log' "
                        f"at {p} (SWE-bench run.log is under raw/) but the file is absent. "
                        "Suite required_evidence is authoritative over manifest booleans."
                    )
            else:
                p = run_dir / "run.log"
                if not p.exists():
                    errors.append(
                        f"Suite required_evidence 'run_log' requires 'run.log' "
                        f"at {p} but the file is absent. "
                        "Suite required_evidence is authoritative over manifest booleans."
                    )
            continue

        if ev_name == "task_yaml":
            # task_yaml is evidenced by the task file hash in manifest.suite_input_hashes.
            task_file = cfg.get("task_file", "")
            if not task_file:
                errors.append(
                    "Suite required_evidence 'task_yaml' requires 'task_file' to be "
                    f"declared in the benchmark config for '{benchmark}', but it is absent. "
                    "task_yaml evidence cannot be verified without a declared task_file."
                )
            elif task_file not in manifest_suite_input_hashes:
                errors.append(
                    f"Suite required_evidence 'task_yaml' requires the task file hash for "
                    f"{task_file!r} to be recorded in manifest.suite_input_hashes, "
                    f"but it is absent. "
                    "The task file hash must be committed in the manifest for task_yaml evidence."
                )
            continue

        if ev_name == "scoring_implementation":
            # scoring_implementation provenance is benchmark-aware:
            #   1. If utils_file is declared (e.g. GPQA), its exact path must be in
            #      manifest.suite_input_hashes — the utils file is the authoritative scorer.
            #   2. If no utils_file but task_file is declared (e.g. GSM8K, IFEval), the
            #      task_file hash in manifest.suite_input_hashes satisfies scorer provenance
            #      because scoring is embedded in the task YAML.
            #   3. If neither file is declared, require suite.required_harness.lm_eval_revision
            #      so the harness pin provides an immutable provenance anchor.
            utils_file = cfg.get("utils_file", "")
            task_file = cfg.get("task_file", "")
            if utils_file:
                # Case 1: utils_file is the authoritative scorer — must be hashed in manifest.
                if utils_file not in manifest_suite_input_hashes:
                    errors.append(
                        f"Suite required_evidence 'scoring_implementation' requires the utils file "
                        f"hash for {utils_file!r} to be recorded in manifest.suite_input_hashes, "
                        f"but it is absent. "
                        "The scoring implementation file hash must be committed in the manifest."
                    )
            elif task_file:
                # Case 2: no utils_file; scoring is embedded in task YAML — task_file hash suffices.
                if task_file not in manifest_suite_input_hashes:
                    errors.append(
                        f"Suite required_evidence 'scoring_implementation' requires the task file "
                        f"hash for {task_file!r} to be recorded in manifest.suite_input_hashes, "
                        f"but it is absent. "
                        "The task file embeds the scoring implementation; its hash must be committed."
                    )
            else:
                # Case 3: neither file; require pinned harness revision for provenance.
                required_harness = suite_data.get("required_harness", {})
                lm_eval_revision = required_harness.get("lm_eval_revision", "")
                if not lm_eval_revision:
                    errors.append(
                        f"Suite required_evidence 'scoring_implementation' for benchmark "
                        f"'{benchmark}' declares neither 'utils_file' nor 'task_file'. "
                        "When scoring is embedded in the harness, "
                        "suite.required_harness.lm_eval_revision must be declared "
                        "to provide an immutable provenance anchor for the scoring implementation. "
                        "Add a pinned lm_eval_revision to the suite's required_harness block."
                    )
            continue

        if ev_name in ("completion_token_counts", "finish_reasons", "full_response_fields"):
            # These must be verified from actual retained sample records.
            _check_sample_field_evidence(run_dir, ev_name, errors)
            continue

        file_hint, is_raw = _EVIDENCE_NAME_MAP[ev_name]
        if file_hint is None:
            # Any remaining None-mapped evidence name is a fail-closed configuration error.
            errors.append(
                f"Suite required_evidence '{ev_name}' has no file-presence check defined "
                "and no structural check implemented. This evidence name cannot be verified "
                "and is rejected (fail-closed configuration error). "
                "Add a verification implementation before using this evidence name."
            )
            continue

        base_dir = raw_dir if is_raw else run_dir

        if "*" in file_hint:
            # Glob pattern
            found = list(base_dir.glob(file_hint)) if base_dir.exists() else []
            if not found:
                errors.append(
                    f"Suite required_evidence '{ev_name}' requires {file_hint!r} "
                    f"under {base_dir} but no matching file was found. "
                    "Suite required_evidence is authoritative over manifest booleans."
                )
        else:
            p = base_dir / file_hint
            if not p.exists():
                errors.append(
                    f"Suite required_evidence '{ev_name}' requires {file_hint!r} "
                    f"at {p} but the file is absent. "
                    "Suite required_evidence is authoritative over manifest booleans."
                )


# ---------------------------------------------------------------------------
# Frozen SWE ID gates
# ---------------------------------------------------------------------------

def _validate_swebench_frozen_ids(
    run_dir: pathlib.Path,
    manifest: dict,
    frozen_ids: frozenset,
    errors: list,
) -> None:
    """Validate preds.json, exit_statuses.json, trajectories, and manifest counts
    against the authoritative frozen instance ID set loaded from the suite.

    Gates enforced:
      - preds.json must be a dict with EXACT key equality to frozen_ids
      - exit_statuses.json must be a dict with EXACT key equality to frozen_ids
      - Every frozen ID must have a trajectory file in raw/trajectories/<id>.traj
      - manifest.item_inventory.expected must equal len(frozen_ids)
      - manifest.item_inventory.instance_ids_hash must match SHA-256(sorted frozen_ids)
    """
    raw_dir = run_dir / "raw"
    item_inv = manifest.get("item_inventory", {})

    # --- preds.json: shape + exact key equality ---
    preds_path = raw_dir / "preds.json"
    if preds_path.exists():
        try:
            preds = json.loads(preds_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"preds.json could not be parsed: {exc}")
            preds = None

        if preds is not None:
            if not isinstance(preds, dict):
                errors.append(
                    f"preds.json must be a JSON object (dict); got {type(preds).__name__}. "
                    "preds.json must be keyed by instance_id."
                )
            else:
                preds_ids = frozenset(preds.keys())
                foreign = preds_ids - frozen_ids
                missing = frozen_ids - preds_ids
                if foreign:
                    fsorted = sorted(foreign)[:5]
                    more = len(foreign) - len(fsorted)
                    suffix = f" ... and {more} more" if more else ""
                    errors.append(
                        f"preds.json contains {len(foreign)} instance ID(s) not in the "
                        f"suite frozen set: {fsorted}{suffix}. "
                        "Only frozen instance IDs may appear in preds.json."
                    )
                if missing:
                    msorted = sorted(missing)[:5]
                    more = len(missing) - len(msorted)
                    suffix = f" ... and {more} more" if more else ""
                    errors.append(
                        f"preds.json is missing {len(missing)} frozen instance ID(s): "
                        f"{msorted}{suffix}. "
                        "Every frozen instance ID must appear in preds.json."
                    )

    # --- exit_statuses.json: shape + exact key equality ---
    statuses_path = raw_dir / "exit_statuses.json"
    if statuses_path.exists():
        try:
            statuses = json.loads(statuses_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"exit_statuses.json could not be parsed: {exc}")
            statuses = None

        if statuses is not None:
            if not isinstance(statuses, dict):
                errors.append(
                    f"exit_statuses.json must be a JSON object (dict); "
                    f"got {type(statuses).__name__}. "
                    "exit_statuses.json must be keyed by instance_id."
                )
            else:
                statuses_ids = frozenset(statuses.keys())
                foreign = statuses_ids - frozen_ids
                missing = frozen_ids - statuses_ids
                if foreign:
                    fsorted = sorted(foreign)[:5]
                    more = len(foreign) - len(fsorted)
                    suffix = f" ... and {more} more" if more else ""
                    errors.append(
                        f"exit_statuses.json contains {len(foreign)} instance ID(s) not in "
                        f"the suite frozen set: {fsorted}{suffix}. "
                        "Only frozen instance IDs may appear in exit_statuses.json."
                    )
                if missing:
                    msorted = sorted(missing)[:5]
                    more = len(missing) - len(msorted)
                    suffix = f" ... and {more} more" if more else ""
                    errors.append(
                        f"exit_statuses.json is missing {len(missing)} frozen instance ID(s): "
                        f"{msorted}{suffix}. "
                        "Every frozen instance ID must appear in exit_statuses.json."
                    )

    # --- Trajectory coverage: every frozen ID must have a .traj file ---
    traj_dir = raw_dir / "trajectories"
    if traj_dir.exists() and traj_dir.is_dir():
        present_trajs: set[str] = set()
        for traj_file in traj_dir.iterdir():
            if traj_file.suffix == ".traj":
                present_trajs.add(traj_file.stem)
        missing_trajs = frozen_ids - present_trajs
        if missing_trajs:
            msorted = sorted(missing_trajs)[:5]
            more = len(missing_trajs) - len(msorted)
            suffix = f" ... and {more} more" if more else ""
            errors.append(
                f"raw/trajectories/ is missing trajectory files for {len(missing_trajs)} "
                f"frozen instance ID(s): {msorted}{suffix}. "
                "Every frozen instance ID must have a corresponding .traj file."
            )

    # --- manifest.item_inventory.expected vs suite frozen count ---
    expected = item_inv.get("expected")
    if expected is not None and expected != len(frozen_ids):
        errors.append(
            f"manifest.item_inventory.expected={expected} does not match "
            f"the suite frozen instance count {len(frozen_ids)}. "
            "The manifest expected count must equal the suite expected_item_count."
        )

    # --- manifest.item_inventory.submitted — must exist and equal len(frozen_ids) ---
    # submitted must be a non-bool integer equal to expected and len(frozen_ids).
    # Missing, bool, non-int, lower, and higher values all fail.
    if "submitted" not in item_inv:
        errors.append(
            "manifest.item_inventory.submitted is absent. "
            "SWE-bench manifests must record the count of submitted instances "
            f"in item_inventory.submitted (must equal {len(frozen_ids)})."
        )
    else:
        submitted_raw = item_inv["submitted"]
        if isinstance(submitted_raw, bool):
            errors.append(
                f"manifest.item_inventory.submitted={submitted_raw!r} is a boolean, not an integer. "
                "submitted must be a non-bool integer equal to the frozen instance count."
            )
        elif not isinstance(submitted_raw, int):
            errors.append(
                f"manifest.item_inventory.submitted={submitted_raw!r} is not an integer "
                f"(got {type(submitted_raw).__name__}). "
                "submitted must be a non-bool integer equal to the frozen instance count."
            )
        elif submitted_raw != len(frozen_ids):
            errors.append(
                f"manifest.item_inventory.submitted={submitted_raw} does not match "
                f"the suite frozen instance count {len(frozen_ids)}. "
                "submitted must equal expected_item_count and the number of frozen IDs."
            )

    # --- manifest.item_inventory.instance_ids_hash vs frozen set digest ---
    recorded_hash = item_inv.get("instance_ids_hash", "")
    expected_hash = hashlib.sha256(
        json.dumps(sorted(frozen_ids), sort_keys=True).encode()
    ).hexdigest()
    if not recorded_hash:
        errors.append(
            "manifest.item_inventory.instance_ids_hash is absent. "
            "SWE-bench manifests must record the canonical SHA-256 of the sorted frozen "
            "instance ID set in item_inventory.instance_ids_hash. "
            f"Expected hash (from frozen set): {expected_hash[:16]}..."
        )
    elif recorded_hash != expected_hash:
            errors.append(
                f"manifest.item_inventory.instance_ids_hash does not match the "
                f"suite-frozen ID set digest. "
                f"Recorded: {recorded_hash[:16]}... "
                f"Expected (from frozen set): {expected_hash[:16]}... "
                "The instance_ids_hash must be the canonical SHA-256 of the sorted frozen IDs."
            )


@dataclass
class ValidationResult:
    """Result of validate()."""
    passed: bool
    errors: List[str] = field(default_factory=list)
    eligible: bool = True  # False for historical runs that cannot be certified


# ---------------------------------------------------------------------------
# Discovery — single shared path used by all callers
# ---------------------------------------------------------------------------

def discover_runs(repo: pathlib.Path) -> list[dict]:
    """Discover manifest-bearing campaign runs in both repository layouts.

    This is the authoritative campaign discovery path used by publication.
    Artifact-oriented historical auditors additionally discover loose sample and
    card files because those predate manifests and are intentionally ineligible
    for canonical publication.

    Returns a list of dicts with keys:
        model_slug, run_id, layout, run_dir, manifest_path, status_path
    """
    repo = pathlib.Path(repo).resolve()
    results_dir = repo / "results"
    runs: list[dict] = []

    if not results_dir.exists():
        return runs

    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        slug = model_dir.name

        # --- Normalized layout: results/<model>/runs/<suite>/<bench>/<run_id>/ ---
        runs_root = model_dir / "runs"
        if runs_root.is_dir():
            for suite_dir in sorted(runs_root.iterdir()):
                if not suite_dir.is_dir():
                    continue
                for bench_dir in sorted(suite_dir.iterdir()):
                    if not bench_dir.is_dir():
                        continue
                    for run_dir in sorted(bench_dir.iterdir()):
                        if not run_dir.is_dir():
                            continue
                        manifest_p = run_dir / "manifest.json"
                        status_p = run_dir / "status.json"
                        if manifest_p.exists() or status_p.exists():
                            runs.append({
                                "model_slug": slug,
                                "run_id": run_dir.name,
                                "layout": "normalized",
                                "run_dir": run_dir,
                                "manifest_path": manifest_p,
                                "status_path": status_p,
                            })

        # --- Historical layout: results/<model>/raw/ (or results/<model>/raw/<bench>/) ---
        raw_root = model_dir / "raw"
        if raw_root.is_dir():
            # Check if manifest.json is directly under raw/
            manifest_p = raw_root / "manifest.json"
            status_p = raw_root / "status.json"
            if manifest_p.exists() or status_p.exists():
                runs.append({
                    "model_slug": slug,
                    "run_id": raw_root.name,
                    "layout": "historical",
                    "run_dir": raw_root,
                    "manifest_path": manifest_p,
                    "status_path": status_p,
                })
            # Also check subdirectories of raw/
            for sub_dir in sorted(raw_root.iterdir()):
                if not sub_dir.is_dir():
                    continue
                manifest_p2 = sub_dir / "manifest.json"
                status_p2 = sub_dir / "status.json"
                if manifest_p2.exists() or status_p2.exists():
                    runs.append({
                        "model_slug": slug,
                        "run_id": sub_dir.name,
                        "layout": "historical",
                        "run_dir": sub_dir,
                        "manifest_path": manifest_p2,
                        "status_path": status_p2,
                    })

    return runs


# ---------------------------------------------------------------------------
# Status consistency validation (public for use by publisher)
# ---------------------------------------------------------------------------

def validate_status_consistency(status: dict) -> list[str]:
    """Validate the internal consistency of a status document.

    Checks:
    - History is a non-empty list of valid transition entries
    - execution_state matches history tail
    - lifecycle is a valid value
    - Timestamps are monotonically non-decreasing

    Returns list of error strings (empty = consistent).
    """
    errors: list[str] = []
    execution_state = status.get("execution_state", "")
    history = status.get("history", [])

    if not isinstance(history, list) or len(history) == 0:
        errors.append("status.history must be a non-empty list")
        return errors

    _VALID_STATES = frozenset({
        "planned", "preflight_passed", "running", "completed",
        "validated", "published", "failed"
    })
    _VALID_LIFECYCLES = frozenset({
        "current", "historical", "superseded", "diagnostic", "replay", "invalid"
    })
    _LEGAL_TRANSITIONS = {
        "planned": {"preflight_passed"},
        "preflight_passed": {"running"},
        "running": {"completed", "failed"},
        "completed": {"validated"},
        "validated": {"published"},
    }

    prev_state = None
    prev_ts = None
    for idx, entry in enumerate(history):
        if not isinstance(entry, dict):
            errors.append(f"status.history[{idx}] is not a dict")
            continue
        state = entry.get("state", "")
        ts = entry.get("timestamp", "")
        if state not in _VALID_STATES:
            errors.append(f"status.history[{idx}].state={state!r} is not a valid state")
        if idx == 0 and state != "planned":
            errors.append(f"status.history first entry must be 'planned', got {state!r}")
        if prev_state is not None and state in _VALID_STATES:
            legal_next = _LEGAL_TRANSITIONS.get(prev_state, set())
            if state not in legal_next and not (prev_state == "running" and state == "failed"):
                errors.append(
                    f"status.history invalid transition at [{idx}]: "
                    f"{prev_state!r} -> {state!r}"
                )
        if prev_ts is not None and isinstance(ts, str) and isinstance(prev_ts, str):
            if ts < prev_ts:
                errors.append(
                    f"status.history non-monotonic timestamp at [{idx}]: "
                    f"{ts!r} < {prev_ts!r}"
                )
        prev_state = state
        prev_ts = ts

    # execution_state must match history tail
    if history and isinstance(history[-1], dict):
        tail_state = history[-1].get("state", "")
        if tail_state != execution_state:
            errors.append(
                f"status.execution_state={execution_state!r} does not match "
                f"history tail state={tail_state!r}"
            )

    lifecycle = status.get("lifecycle", "")
    if lifecycle not in _VALID_LIFECYCLES:
        errors.append(f"status.lifecycle={lifecycle!r} is not a valid lifecycle value")

    return errors


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------

def validate(
    run_dir: pathlib.Path,
    suite_path: pathlib.Path,
    adapter_path: pathlib.Path,
    *,
    for_publication: bool = False,
) -> ValidationResult:
    """Validate a campaign run directory.

    Parameters
    ----------
    run_dir:
        The run directory to validate.
    suite_path:
        Path to the suite YAML file.
    adapter_path:
        Path to the adapter YAML file.
    for_publication:
        When True, also enforce the publication gate (lifecycle must be 'current',
        aggregate file must be present).

    Returns
    -------
    ValidationResult
        .passed is True only when all active gates pass.
        .eligible is False when the run is historical (cannot be certified).
        .errors lists every failure.
    """
    run_dir = pathlib.Path(run_dir).resolve()
    suite_path = pathlib.Path(suite_path).resolve()
    adapter_path = pathlib.Path(adapter_path).resolve()
    errors: list[str] = []

    # --- Load suite YAML (authoritative for evidence requirements and frozen IDs) ---
    suite_data = _load_suite_yaml(suite_path, errors)
    # Note: suite_data may be None if YAML is unavailable or parse fails, but
    # we continue — errors from suite loading are appended and will fail validation.
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        errors.append(f"manifest.json not found in {run_dir}")
        return ValidationResult(passed=False, errors=errors)

    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        errors.append(f"manifest.json is invalid JSON: {exc}")
        return ValidationResult(passed=False, errors=errors)

    # --- Load status ---
    status_path = run_dir / "status.json"
    status: dict = {}
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f"status.json is invalid JSON: {exc}")

    # --- V11: JSON schema validation ---
    _validate_manifest_schema(manifest, errors)
    if status:
        _validate_status_schema(status, errors)

    if errors:
        return ValidationResult(passed=False, errors=errors, eligible=True)

    lifecycle = status.get("lifecycle", "current")
    execution_state = status.get("execution_state", "")

    # --- V10: Historical runs are not certifiable ---
    if lifecycle in ("historical", "superseded", "diagnostic", "replay", "invalid"):
        reason = (
            f"Run lifecycle is {lifecycle!r}. Historical/superseded/diagnostic/replay/"
            f"invalid runs are not certifiable by the v1 validator and may not be published. "
            f"They remain discoverable for audit only."
        )
        errors.append(reason)
        return ValidationResult(passed=False, errors=errors, eligible=False)

    # --- V12: Identity agreement ---
    _check_identity_agreement(run_dir, manifest, status, errors)

    # --- V7: Lifecycle gate (for_publication only) ---
    if for_publication and lifecycle != "current":
        errors.append(
            f"lifecycle is {lifecycle!r} but must be 'current' for publication. "
            "Only validated current runs may be published."
        )

    # --- Status consistency ---
    consistency_errors = validate_status_consistency(status)
    errors.extend(consistency_errors)

    # --- V9: DONE sentinel consistency ---
    done_path = run_dir / "DONE"
    done_exists = done_path.exists()

    if execution_state in ("failed",):
        if done_exists:
            errors.append(
                f"DONE sentinel present but execution_state is {execution_state!r}. "
                "DONE must only be written after successful harness exit."
            )

    # --- V2: Required manifest fields ---
    required_manifest_fields = [
        "schema_version", "suite_id", "suite_schema_version", "run_id",
        "benchmark", "adapter_hash", "suite_input_hashes", "serving_profile_digest",
        "model", "serving", "item_inventory", "timing", "artifact_inventory",
    ]
    for fld in required_manifest_fields:
        if fld not in manifest:
            errors.append(f"manifest.json missing required field: {fld!r}")

    if errors:
        # If manifest is structurally broken, skip further checks
        return ValidationResult(passed=False, errors=errors)

    artifact_inv = manifest.get("artifact_inventory", {})
    item_inv = manifest.get("item_inventory", {})
    n_expected = item_inv.get("expected", 0)
    benchmark = manifest.get("benchmark", "")

    # --- V2: Required files ---
    _check_required_files(run_dir, artifact_inv, errors, benchmark=benchmark)

    # --- SWE-bench: grading_results content validation ---
    if benchmark == "swebench":
        # Load frozen IDs for grading authority (preferred over preds.json)
        _swe_frozen_errors: list[str] = []
        _swe_frozen_ids: Optional[frozenset] = None
        if suite_data is not None:
            _swe_frozen_ids = _load_frozen_instance_ids(
                suite_data, suite_path, _swe_frozen_errors
            )
        # Pass frozen_ids to grading validator (None = fall back to preds-based check)
        _validate_swebench_grading(
            run_dir, manifest, errors,
            for_publication=for_publication,
            frozen_ids=_swe_frozen_ids if not _swe_frozen_errors else None,
        )

    # --- Suite-driven required_evidence gate (authoritative over manifest booleans) ---
    if suite_data is not None:
        _check_suite_required_evidence(run_dir, suite_data, benchmark, errors)

    # --- SWE-bench: frozen ID authority gate ---
    # Load frozen IDs from suite and cross-check preds/statuses/trajectories/manifest.
    if benchmark == "swebench" and suite_data is not None:
        _frozen_errors: list[str] = []
        frozen_ids = _load_frozen_instance_ids(suite_data, suite_path, _frozen_errors)
        if _frozen_errors:
            errors.extend(_frozen_errors)
        elif frozen_ids is not None:
            _validate_swebench_frozen_ids(run_dir, manifest, frozen_ids, errors)

    # --- V8: Secret scan ---
    _secret_scan(run_dir, errors)

    # --- V5: Hash verification ---
    _verify_hashes(run_dir, manifest, suite_path, adapter_path, errors)

    # --- V6: Forbidden overrides ---
    _check_forbidden_overrides(manifest, errors)

    # --- V9: DONE + complete artifacts ---
    if done_exists:
        _check_done_artifacts(run_dir, artifact_inv, manifest, errors)

    # --- V1, V3, V4: Per-item evidence checks ---
    # Per-item evidence is only required for quality benchmarks (gsm8k, ifeval, gpqa_diamond).
    # SWE-bench uses preds.json + grading_results.json instead.
    _QUALITY_BENCHMARKS = frozenset({"gsm8k", "ifeval", "gpqa_diamond"})
    per_item_path = run_dir / "per_item.csv"
    if per_item_path.exists():
        items = _load_per_item(per_item_path, errors)
        if items is not None:
            _check_item_ids(
                items,
                n_expected,
                errors,
                expected_ids_hash=item_inv.get("instance_ids_hash", ""),
                submitted=item_inv.get("submitted", _ABSENT),
            )
            _check_raw_sample_ids(run_dir, items, errors)
            _check_dispositions(items, errors)
            _check_aggregate_reconciliation(
                run_dir, items, benchmark, errors,
                fail_on_missing=for_publication,
            )
    elif n_expected > 0 and benchmark in _QUALITY_BENCHMARKS:
        # per_item.csv missing for a quality benchmark — hard failure for current runs
        if lifecycle == "current":
            errors.append(
                f"per_item.csv not found in {run_dir} but item_inventory.expected={n_expected}. "
                "Per-item evidence is required for current lifecycle quality benchmark runs."
            )

    return ValidationResult(passed=len(errors) == 0, errors=errors)


# ---------------------------------------------------------------------------
# Schema validation gates
# ---------------------------------------------------------------------------

def _validate_manifest_schema(manifest: dict, errors: list) -> None:
    """V11: Validate manifest against JSON schema — fail-closed.

    Both absent schema files and un-importable contract module are hard failures.
    Silently skipping schema validation is not acceptable: it creates a window where
    malformed manifests pass all downstream gates unchecked.
    """
    if not _MANIFEST_SCHEMA.exists():
        errors.append(
            f"manifest JSON schema not found at {_MANIFEST_SCHEMA}. "
            "Schema validation is required; run `make schemas` or restore the schema file."
        )
        return
    try:
        from contract import validate_json  # type: ignore[import]
    except ImportError as exc:
        errors.append(
            f"contract module not importable ({exc}); JSON schema validation cannot proceed. "
            "Ensure the contract module is installed or available on PYTHONPATH."
        )
        return
    schema_errors = validate_json(manifest, _MANIFEST_SCHEMA)
    for e in schema_errors:
        errors.append(f"manifest.json schema violation: {e}")


def _validate_status_schema(status: dict, errors: list) -> None:
    """V11: Validate status against JSON schema — fail-closed.

    Both absent schema files and un-importable contract module are hard failures.
    """
    if not _STATUS_SCHEMA.exists():
        errors.append(
            f"status JSON schema not found at {_STATUS_SCHEMA}. "
            "Schema validation is required; run `make schemas` or restore the schema file."
        )
        return
    try:
        from contract import validate_json  # type: ignore[import]
    except ImportError as exc:
        errors.append(
            f"contract module not importable ({exc}); status.json schema validation cannot proceed. "
            "Ensure the contract module is installed or available on PYTHONPATH."
        )
        return
    schema_errors = validate_json(status, _STATUS_SCHEMA)
    for e in schema_errors:
        errors.append(f"status.json schema violation: {e}")


# ---------------------------------------------------------------------------
# Identity agreement
# ---------------------------------------------------------------------------

def _check_identity_agreement(
    run_dir: pathlib.Path,
    manifest: dict,
    status: dict,
    errors: list,
) -> None:
    """V12: Verify identity agreement among dir name, manifest, and status."""
    dir_name = run_dir.name
    manifest_run_id = manifest.get("run_id", "")
    status_run_id = status.get("run_id", "")
    manifest_suite_id = manifest.get("suite_id", "")
    status_suite_id = status.get("suite_id", "")

    # run_id: dir name vs manifest
    if manifest_run_id and dir_name != manifest_run_id:
        errors.append(
            f"Identity mismatch: run directory name {dir_name!r} != "
            f"manifest.run_id {manifest_run_id!r}. "
            "The run directory must be named after its run_id."
        )

    # run_id: manifest vs status
    if status_run_id and manifest_run_id and status_run_id != manifest_run_id:
        errors.append(
            f"Identity mismatch: manifest.run_id={manifest_run_id!r} != "
            f"status.run_id={status_run_id!r}. "
            "Both must reference the same run."
        )

    # suite_id: manifest vs status
    if status_suite_id and manifest_suite_id and status_suite_id != manifest_suite_id:
        errors.append(
            f"Identity mismatch: manifest.suite_id={manifest_suite_id!r} != "
            f"status.suite_id={status_suite_id!r}. "
            "Both must reference the same suite."
        )


# ---------------------------------------------------------------------------
# Gate implementations
# ---------------------------------------------------------------------------

def _validate_swebench_grading(
    run_dir: pathlib.Path,
    manifest: dict,
    errors: list,
    *,
    for_publication: bool = False,
    frozen_ids: Optional[frozenset] = None,
) -> None:
    """Validate grading_results.json content for SWE-bench runs.

    Enforces:
      - grading_results.json is always required for SWE-bench when for_publication=True,
        regardless of manifest.artifact_inventory.grading_results_json claim.
      - grading_results.json, when present, must be valid JSON with the expected keys.
      - All IDs across grading categories must be disjoint (no duplicates across categories).
      - Within each category, IDs must be unique (no within-category duplicates).
      - All expected instance IDs must appear in exactly one category (exhaustive coverage).
      - No foreign IDs may appear (only IDs from the expected set are allowed).

    The expected ID set is derived from the suite frozen instance set (when available)
    via the frozen_ids parameter.  Falls back to preds.json keys when frozen_ids is None.
    """
    raw_dir = run_dir / "raw"
    grading_path = raw_dir / "grading_results.json"
    _GRADING_KEYS = ("resolved_ids", "unresolved_ids", "empty_patch_ids",
                     "error_ids", "incomplete_ids")

    # For publication: require grading unconditionally, regardless of manifest claim
    if for_publication and not grading_path.exists():
        errors.append(
            "SWE-bench for_publication=True requires grading_results.json "
            f"in {raw_dir}, but the file is absent. "
            "Grading evidence is mandatory for publication regardless of "
            "manifest.artifact_inventory.grading_results_json value."
        )
        return

    if not grading_path.exists():
        return  # not a publication run and not claimed; skip

    # Parse grading_results.json — malformed JSON is a hard failure
    try:
        grading = json.loads(grading_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(
            f"grading_results.json is not valid JSON: {exc}. "
            "The SWE-bench grading report must be a well-formed JSON object."
        )
        return
    except OSError as exc:
        errors.append(f"grading_results.json could not be read: {exc}")
        return

    if not isinstance(grading, dict):
        errors.append(
            f"grading_results.json must be a JSON object; got {type(grading).__name__}."
        )
        return

    # Collect all IDs from all categories, checking both within-category and cross-category dups
    seen_ids: dict[str, str] = {}  # id -> first category it appeared in
    all_graded: set[str] = set()
    dup_errors: list[str] = []
    for key in _GRADING_KEYS:
        cat_ids = grading.get(key, [])
        if not isinstance(cat_ids, list):
            errors.append(
                f"grading_results.json['{key}'] must be a list; "
                f"got {type(cat_ids).__name__}."
            )
            continue
        # Within-category uniqueness check
        within_seen: set[str] = set()
        for iid in cat_ids:
            if iid in within_seen:
                dup_errors.append(
                    f"Duplicate instance ID {iid!r} within grading_results.json['{key}']. "
                    "Each ID must appear at most once within a single category."
                )
            within_seen.add(iid)
            # Cross-category uniqueness check
            if iid in seen_ids:
                dup_errors.append(
                    f"Duplicate instance ID {iid!r} in grading_results.json: "
                    f"appears in both '{seen_ids[iid]}' and '{key}'. "
                    "Grading categories must be disjoint — each ID must appear in exactly one."
                )
            else:
                seen_ids[iid] = key
            all_graded.add(iid)
    errors.extend(dup_errors)

    # A complete inventory of harness RuntimeError dispositions is an
    # infrastructure failure, not a model score. Inspect generation evidence as
    # well as grading buckets because the grader may classify empty predictions
    # without producing any resolved/unresolved verdicts.
    statuses_path = raw_dir / "exit_statuses.json"
    if statuses_path.exists():
        try:
            statuses = json.loads(statuses_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"exit_statuses.json could not be parsed for runtime-error gating: {exc}")
            statuses = None
        if isinstance(statuses, dict):
            runtime_error_ids = {
                iid for iid, disposition in statuses.items()
                if isinstance(disposition, str) and disposition.lower() == "runtimeerror"
            }
            if runtime_error_ids:
                errors.append(
                    f"SWE-bench publication rejected: {len(runtime_error_ids)}/{len(statuses)} "
                    "instances have RuntimeError infrastructure dispositions."
                )

    graded_verdict_count = len(grading.get("resolved_ids", [])) + len(
        grading.get("unresolved_ids", [])
    )
    if for_publication and graded_verdict_count == 0:
        errors.append(
            "SWE-bench publication rejected: no graded verdict exists in resolved_ids "
            "or unresolved_ids; empty/error/incomplete dispositions alone are not a "
            "scientifically valid capability result."
        )

    # Determine expected ID set: prefer suite frozen_ids, fall back to preds.json
    if frozen_ids is not None:
        expected_ids: set[str] = set(frozen_ids)
    else:
        # Fall back: derive expected ID set from preds.json keys
        preds_path = raw_dir / "preds.json"
        if not preds_path.exists():
            # preds.json absence is caught by _check_required_files; skip ID reconciliation
            return
        try:
            preds = json.loads(preds_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"preds.json could not be parsed for grading reconciliation: {exc}")
            return
        if not isinstance(preds, dict):
            return  # shape error already caught elsewhere
        expected_ids = set(preds.keys())

    # Foreign IDs: in grading but not in expected
    foreign = all_graded - expected_ids
    if foreign:
        foreign_sorted = sorted(foreign)[:5]  # show up to 5 for brevity
        more = len(foreign) - len(foreign_sorted)
        suffix = f" ... and {more} more" if more > 0 else ""
        errors.append(
            f"grading_results.json contains {len(foreign)} foreign instance ID(s) "
            f"not present in the expected set: "
            f"{foreign_sorted}{suffix}. "
            "Only IDs from the frozen benchmark set may appear in grading categories."
        )

    # Missing IDs: in expected but not covered by any grading category
    missing = expected_ids - all_graded
    if missing:
        missing_sorted = sorted(missing)[:5]
        more = len(missing) - len(missing_sorted)
        suffix = f" ... and {more} more" if more > 0 else ""
        errors.append(
            f"grading_results.json missing disposition for {len(missing)} instance ID(s): "
            f"{missing_sorted}{suffix}. "
            "Every expected instance must appear in exactly one grading category."
        )


def _check_required_files(
    run_dir: pathlib.Path,
    artifact_inv: dict,
    errors: list,
    *,
    benchmark: str = "",
) -> None:
    """V2: Check that required artifact files are present.

    For quality benchmarks (gsm8k, ifeval, gpqa_diamond): requires samples_jsonl_gz
    and per_item_csv in addition to run_log, command_txt, done_sentinel.

    For SWE-bench: requires preds_json, exit_statuses_json in raw/ instead of
    samples_jsonl_gz and per_item_csv. grading_results_json is also required
    when artifact_inventory declares it.

    Universal: run_log, command_txt, DONE sentinel required for all benchmarks.
    """
    _QUALITY_BENCHMARKS = frozenset({"gsm8k", "ifeval", "gpqa_diamond"})
    _SWEBENCH_BENCHMARKS = frozenset({"swebench"})

    # Check files listed as True in artifact_inventory
    file_map = {
        "samples_jsonl_gz": "raw/samples_*.jsonl.gz",
        "per_item_csv": "per_item.csv",
        "run_log": "run.log",
        "command_txt": "command.txt",
        "done_sentinel": "DONE",
        "preds_json": "raw/preds.json",
        "exit_statuses_json": "raw/exit_statuses.json",
        "grading_results_json": "raw/grading_results.json",
    }
    for key, hint in file_map.items():
        if not artifact_inv.get(key, False):
            continue  # not claimed to exist, skip
        if key == "samples_jsonl_gz":
            raw_dir = run_dir / "raw"
            found = list(raw_dir.rglob("*.jsonl.gz")) if raw_dir.exists() else []
            if not found:
                errors.append(
                    f"artifact_inventory.samples_jsonl_gz=True but no *.jsonl.gz found under {raw_dir}"
                )
        elif key == "done_sentinel":
            if not (run_dir / "DONE").exists():
                errors.append(
                    f"artifact_inventory.done_sentinel=True but DONE not found in {run_dir}"
                )
        elif key == "run_log":
            # SWE-bench run.log lives in raw/; quality benchmarks at run_dir root.
            if benchmark in _SWEBENCH_BENCHMARKS:
                p = run_dir / "raw" / "run.log"
            else:
                p = run_dir / "run.log"
            if not p.exists():
                errors.append(
                    f"artifact_inventory.{key}=True but {hint} not found in {run_dir}"
                )
        elif key in ("preds_json", "exit_statuses_json", "grading_results_json"):
            # SWE-bench specific artifacts live in raw/
            raw_dir = run_dir / "raw"
            filename = hint.split("/", 1)[-1]  # strip "raw/" prefix
            p = raw_dir / filename
            if not p.exists():
                errors.append(
                    f"artifact_inventory.{key}=True but {hint} not found in {run_dir}"
                )
        else:
            p = run_dir / hint
            if not p.exists():
                errors.append(
                    f"artifact_inventory.{key}=True but {hint} not found in {run_dir}"
                )

    # Always require run.log and command.txt for any run.
    # SWE-bench run.log is under raw/; quality at run_dir root.
    if benchmark in _SWEBENCH_BENCHMARKS:
        run_log_path = run_dir / "raw" / "run.log"
    else:
        run_log_path = run_dir / "run.log"
    if not run_log_path.exists():
        errors.append(f"run.log not found in {run_dir}")
    if not (run_dir / "command.txt").exists():
        errors.append(f"command.txt not found in {run_dir}")
    if not (run_dir / "DONE").exists():
        errors.append(
            f"DONE sentinel missing from {run_dir}. "
            "DONE must be written only after successful harness exit."
        )

    # Benchmark-specific evidence requirements
    if benchmark in _QUALITY_BENCHMARKS:
        # Quality benchmarks require samples + per_item
        raw_dir = run_dir / "raw"
        found_samples = list(raw_dir.rglob("*.jsonl.gz")) if raw_dir.exists() else []
        if not found_samples:
            errors.append(
                f"No raw samples (*.jsonl.gz) found under {run_dir / 'raw'}. "
                f"Raw samples are required for quality benchmark '{benchmark}' validation."
            )
        if not (run_dir / "per_item.csv").exists():
            errors.append(
                f"per_item.csv not found in {run_dir}. "
                f"Per-item evidence is required for quality benchmark '{benchmark}' validation."
            )
    elif benchmark in _SWEBENCH_BENCHMARKS:
        # SWE-bench requires preds.json and exit_statuses.json in raw/
        raw_dir = run_dir / "raw"
        if not (raw_dir / "preds.json").exists():
            errors.append(
                f"preds.json not found under {raw_dir}. "
                "SWE-bench validation requires preds.json (predictions dict keyed by instance_id)."
            )
        if not (raw_dir / "exit_statuses.json").exists():
            errors.append(
                f"exit_statuses.json not found under {raw_dir}. "
                "SWE-bench validation requires exit_statuses.json (instance_id -> exit_status mapping)."
            )
        # SWE-bench requires trajectories/ directory (runner normalization guarantees it)
        traj_dir = raw_dir / "trajectories"
        if not traj_dir.exists() or not traj_dir.is_dir():
            errors.append(
                f"trajectories/ directory not found under {raw_dir}. "
                "SWE-bench validation requires raw/trajectories/ (runner normalization "
                "copies trajectory files here for evidence and audit)."
            )
    else:
        # Unknown/throughput benchmark: fall back to checking samples if claimed
        raw_dir = run_dir / "raw"
        found_samples = list(raw_dir.rglob("*.jsonl.gz")) if raw_dir.exists() else []
        if artifact_inv.get("samples_jsonl_gz", False) and not found_samples:
            errors.append(
                f"artifact_inventory.samples_jsonl_gz=True but no *.jsonl.gz found under {raw_dir}"
            )


def _secret_scan(run_dir: pathlib.Path, errors: list) -> None:
    """V8: Scan staged artifacts for credential patterns.

    Uses gitleaks (when installed) for comprehensive detection over all files in
    run_dir. Falls back to regex-only scan over command.txt and run.log when
    gitleaks is unavailable.
    """
    import subprocess as _subprocess

    gitleaks_bin = _resolve_gitleaks_bin()
    if not gitleaks_bin:
        errors.append(
            "gitleaks is not installed; required artifact secret scanning cannot be performed."
        )
        return

    # gitleaks v8 scans arbitrary directory trees with the `dir` command.  Any
    # result other than a clean exit is blocking: exit 1 means a finding, while
    # other codes mean the required scan itself did not complete.
    try:
        result = _subprocess.run(
            [gitleaks_bin, "dir", "--redact", "--exit-code", "1", str(run_dir)],
            capture_output=True,
            text=True,
            timeout=_GITLEAKS_SCAN_TIMEOUT_SECONDS,
        )
    except (OSError, _subprocess.TimeoutExpired) as exc:
        errors.append(f"gitleaks artifact scan could not be completed: {exc}")
        return

    if result.returncode == 1:
        errors.append(
            f"gitleaks detected credential(s) in run artifacts under {run_dir}. "
            "Remove credentials before committing artifacts."
        )
        return
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown gitleaks error").strip()
        errors.append(
            f"gitleaks artifact scan failed with exit {result.returncode}: {detail}"
        )
        return

    # Keep targeted checks as defense in depth for credential-bearing command
    # forms that may not meet a provider rule's entropy threshold.
    for fpath in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        try:
            content = fpath.read_text(errors="replace")
        except OSError as exc:
            errors.append(f"Could not read artifact during secret scan: {fpath}: {exc}")
            continue
        for pat in _SECRET_PATTERNS:
            if pat.search(content):
                errors.append(
                    f"Secret/credential pattern detected in {fpath.relative_to(run_dir)}: "
                    f"matches {pat.pattern!r}. Remove credentials before committing artifacts."
                )
                break


def _verify_hashes(
    run_dir: pathlib.Path,
    manifest: dict,
    suite_path: pathlib.Path,
    adapter_path: pathlib.Path,
    errors: list,
) -> None:
    """V5: Verify that recorded hashes match actual files."""
    suite_input_hashes = manifest.get("suite_input_hashes", {})

    # Locate the repo root (parent of suite/ directory)
    repo = suite_path.parent.parent
    repo_resolved = repo.resolve()

    for rel_path, recorded_hash in suite_input_hashes.items():
        abs_path = repo / rel_path
        if not abs_path.exists():
            errors.append(
                f"Suite input file not found: {rel_path} (expected at {abs_path}). "
                "Hash cannot be verified."
            )
            continue
        # Containment check: resolve symlinks and confirm inside repo root.
        try:
            resolved = abs_path.resolve()
            try:
                resolved.relative_to(repo_resolved)
            except ValueError:
                errors.append(
                    f"Suite input file {rel_path!r} resolves to {resolved}, "
                    f"which is outside the repo root ({repo_resolved}). "
                    "Symlink escape outside the repo root is not allowed, "
                    "even when the external file's hash matches the recorded hash."
                )
                continue  # Do not hash-check an escaped path
        except OSError as exc:
            errors.append(
                f"Could not resolve suite input path {rel_path!r}: {exc}"
            )
            continue
        actual_hash = hashlib.sha256(abs_path.read_bytes()).hexdigest()
        if actual_hash != recorded_hash:
            errors.append(
                f"Suite input hash mismatch for {rel_path}: "
                f"recorded={recorded_hash[:16]}... actual={actual_hash[:16]}... "
                "The file has changed since the manifest was created (stale hash)."
            )

    # Verify adapter hash
    recorded_adapter_hash = manifest.get("adapter_hash", "")
    if adapter_path.exists():
        # Containment check: adapter must resolve inside the repo root
        repo_resolved = repo.resolve()
        try:
            adapter_resolved = adapter_path.resolve()
            try:
                adapter_resolved.relative_to(repo_resolved)
            except ValueError:
                errors.append(
                    f"Adapter file {adapter_path} resolves to {adapter_resolved}, "
                    f"which is outside the repo root ({repo_resolved}). "
                    "Adapter symlink escape outside repo is not allowed."
                )
                return  # Cannot safely continue hash check
        except OSError as exc:
            errors.append(f"Could not resolve adapter path {adapter_path}: {exc}")
            return

        actual_adapter_hash = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
        if actual_adapter_hash != recorded_adapter_hash:
            errors.append(
                f"Adapter hash mismatch: "
                f"recorded={recorded_adapter_hash[:16]}... actual={actual_adapter_hash[:16]}... "
                "The adapter file has changed since the manifest was created (stale adapter hash)."
            )
    else:
        errors.append(f"Adapter file not found at {adapter_path}. Cannot verify adapter hash.")


def _check_forbidden_overrides(manifest: dict, errors: list) -> None:
    """V6: Reject effective_args containing suite-owned field overrides."""
    effective_args = manifest.get("serving", {}).get("effective_args", [])
    if not isinstance(effective_args, list):
        return
    for arg in effective_args:
        for pattern in _FORBIDDEN_ARG_PATTERNS:
            if re.search(pattern, arg, re.I):
                errors.append(
                    f"Forbidden suite-owned override in effective_args: {arg!r}. "
                    f"Adapter/command-line must not override suite-owned fields "
                    f"(tasks, prompts, datasets, ceilings, scoring, sampling, denominators)."
                )
                break


def _check_done_artifacts(
    run_dir: pathlib.Path,
    artifact_inv: dict,
    manifest: dict,
    errors: list,
) -> None:
    """V9: When DONE is present, all artifact_inventory=True items must exist on disk."""
    benchmark = manifest.get("benchmark", "")
    _SWEBENCH_BENCHMARKS = frozenset({"swebench"})

    # Artifact checks for every key declared True
    # run_log is benchmark-aware: SWE-bench in raw/, others at run_dir root.
    if artifact_inv.get("per_item_csv", False):
        if not (run_dir / "per_item.csv").exists():
            errors.append(
                f"DONE is present but artifact_inventory.per_item_csv=True and "
                f"per_item.csv is missing from {run_dir}. "
                "All claimed artifacts must exist when DONE is written."
            )
    if artifact_inv.get("run_log", False):
        if benchmark in _SWEBENCH_BENCHMARKS:
            run_log_path = run_dir / "raw" / "run.log"
        else:
            run_log_path = run_dir / "run.log"
        if not run_log_path.exists():
            errors.append(
                f"DONE is present but artifact_inventory.run_log=True and "
                f"run.log is missing from {run_dir}. "
                "All claimed artifacts must exist when DONE is written."
            )
    if artifact_inv.get("command_txt", False):
        if not (run_dir / "command.txt").exists():
            errors.append(
                f"DONE is present but artifact_inventory.command_txt=True and "
                f"command.txt is missing from {run_dir}. "
                "All claimed artifacts must exist when DONE is written."
            )

    if artifact_inv.get("samples_jsonl_gz", False):
        raw_dir = run_dir / "raw"
        found = list(raw_dir.rglob("*.jsonl.gz")) if raw_dir.exists() else []
        if not found:
            errors.append(
                f"DONE is present but artifact_inventory.samples_jsonl_gz=True "
                f"and no *.jsonl.gz found under {run_dir / 'raw'}."
            )


def _load_per_item(
    per_item_path: pathlib.Path,
    errors: list,
) -> Optional[list]:
    """Load per_item.csv and return records, or None on failure."""
    try:
        records = []
        with per_item_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                records.append(dict(row))
        return records
    except Exception as exc:
        errors.append(f"Failed to read per_item.csv: {exc}")
        return None


def _check_item_ids(
    items: list,
    n_expected: int,
    errors: list,
    expected_ids_hash: str = "",
    submitted: object = _ABSENT,
) -> None:
    """V1: Check for missing/duplicate IDs and frozen-set drift.

    Also validates manifest.item_inventory.submitted for quality benchmarks:
    submitted must be a non-bool integer equal to both n_expected and the
    unique row count in per_item.csv.

    submitted=_ABSENT means the key was absent from item_inventory (missing field).
    submitted=<value> means the key was present with that value.
    """
    ids = [r.get("item_id", "") for r in items]
    seen: dict = {}
    for i, iid in enumerate(ids):
        if iid in seen:
            errors.append(
                f"Duplicate item ID {iid!r} at row {i} (also at row {seen[iid]}). "
                "Item IDs must be unique."
            )
        else:
            seen[iid] = i

    # Check count vs expected
    unique_count = len(seen)
    if unique_count < n_expected:
        errors.append(
            f"Item count mismatch: expected {n_expected} items but found {unique_count} unique "
            f"item IDs in per_item.csv. {n_expected - unique_count} item(s) are missing."
        )
    elif unique_count > n_expected:
        errors.append(
            f"Item count mismatch: expected {n_expected} items but found {unique_count} unique "
            f"item IDs in per_item.csv. {unique_count - n_expected} unexpected item(s)."
        )

    # --- submitted reconciliation for quality benchmarks ---
    # manifest.item_inventory.submitted (passed as submitted argument) must be a non-bool
    # int equal to unique_count (and implicitly n_expected when both agree).
    # _ABSENT means the caller found the key missing from item_inventory.
    if submitted is _ABSENT:
        errors.append(
            "manifest.item_inventory.submitted is absent. "
            f"Quality benchmark manifests must record submitted count "
            f"(must equal {unique_count} unique item rows and expected={n_expected})."
        )
    elif isinstance(submitted, bool):
        errors.append(
            f"manifest.item_inventory.submitted={submitted!r} is a boolean, not an integer. "
            "submitted must be a non-bool integer equal to the unique per-item row count."
        )
    elif not isinstance(submitted, int):
        errors.append(
            f"manifest.item_inventory.submitted={submitted!r} is not an integer "
            f"(got {type(submitted).__name__}). "
            "submitted must be a non-bool integer equal to the unique per-item row count."
        )
    elif submitted != unique_count:
        errors.append(
            f"manifest.item_inventory.submitted={submitted} does not match "
            f"unique per_item.csv row count={unique_count}. "
            "submitted must equal the number of unique item rows in per_item.csv."
        )

    # A count alone cannot distinguish the frozen set from a same-size foreign
    # set.  New manifests record the digest of the sorted exact IDs.
    if expected_ids_hash and unique_count == len(ids):
        actual_ids_hash = hashlib.sha256(
            json.dumps(sorted(ids), sort_keys=True).encode()
        ).hexdigest()
        if actual_ids_hash != expected_ids_hash:
            errors.append(
                "Instance ID set hash mismatch: per_item.csv does not contain the "
                "frozen item set recorded in manifest.item_inventory.instance_ids_hash."
            )


def _check_raw_sample_ids(
    run_dir: pathlib.Path,
    items: list,
    errors: list,
) -> None:
    """Verify derived per-item IDs agree with retained raw sample IDs.

    Raw samples are primary evidence.  One lm-eval document may produce several
    filter rows, so compare sets rather than row counts.  If a sample object has
    neither ``item_id`` nor ``doc_id``, identity cannot be proved and validation
    fails closed.
    """
    raw_dir = run_dir / "raw"
    sample_paths = sorted(raw_dir.rglob("samples_*.jsonl.gz")) if raw_dir.exists() else []
    if not sample_paths:
        return  # The required-file gate reports the absence.

    raw_ids: set[str] = set()
    for sample_path in sample_paths:
        try:
            with gzip.open(sample_path, "rt") as fh:
                for line_number, line in enumerate(fh, start=1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        errors.append(
                            f"Raw sample {sample_path.name}:{line_number} is not a JSON object."
                        )
                        continue
                    item_id = record.get("item_id", record.get("doc_id"))
                    if item_id is None or not str(item_id).strip():
                        errors.append(
                            f"Raw sample {sample_path.name}:{line_number} has no non-empty "
                            "item_id or doc_id; item identity cannot be verified."
                        )
                        continue
                    raw_ids.add(str(item_id))
        except (OSError, EOFError, json.JSONDecodeError) as exc:
            errors.append(f"Could not parse raw sample evidence {sample_path}: {exc}")

    csv_ids = {str(row.get("item_id", "")) for row in items if str(row.get("item_id", "")).strip()}
    if raw_ids and csv_ids != raw_ids:
        missing = sorted(raw_ids - csv_ids)
        foreign = sorted(csv_ids - raw_ids)
        errors.append(
            "Per-item ID set does not match retained raw samples: "
            f"missing={missing[:10]}, foreign={foreign[:10]}."
        )


def _check_dispositions(items: list, errors: list) -> None:
    """V4: Verify every item has a valid classified disposition (allowlist enforced)."""
    for row in items:
        disposition = row.get("disposition", "").strip()
        if not disposition or disposition.lower() == "unclassified":
            errors.append(
                f"Item {row.get('item_id', '?')!r} has unclassified disposition: "
                f"{disposition!r}. Every item must have an explicit disposition "
                f"(correct, wrong, empty, budget, parser, timeout, transport, etc.)."
            )
        elif disposition not in _VALID_DISPOSITIONS:
            errors.append(
                f"Item {row.get('item_id', '?')!r} has unknown disposition: "
                f"{disposition!r}. Allowed values: {sorted(_VALID_DISPOSITIONS)}. "
                "Unknown dispositions fail closed — add to the allowlist if genuinely needed."
            )


def _check_aggregate_reconciliation(
    run_dir: pathlib.Path,
    items: list,
    benchmark: str,
    errors: list,
    *,
    fail_on_missing: bool = False,
) -> None:
    """V3: Verify aggregate score reconciles with per-item evidence.

    When fail_on_missing=True (for_publication=True), the absence of an
    aggregate result file is a hard failure rather than a silent skip.

    G8 (non-finite scores): Any per-item score that is missing (empty string)
    or non-finite (NaN, Inf) is a data integrity error and causes validation
    failure, not a silent skip.
    """
    # G8: Check all per-item scores for missing or non-finite values
    missing_scores = []
    nonfinite_scores = []
    valid_scores = []
    for row in items:
        score_str = row.get("score", "")
        item_id = row.get("item_id", "?")
        if score_str in ("", None):
            missing_scores.append(item_id)
        else:
            try:
                score_val = float(score_str)
                import math
                if not math.isfinite(score_val):
                    nonfinite_scores.append((item_id, score_str))
                else:
                    valid_scores.append(score_val)
            except (ValueError, TypeError):
                missing_scores.append(item_id)

    if missing_scores:
        errors.append(
            f"Per-item evidence integrity error: {len(missing_scores)} item(s) have missing "
            f"or unparseable scores: {missing_scores[:10]}. "
            "Every scored item must have a valid numeric score for aggregate reconciliation."
        )
    if nonfinite_scores:
        errors.append(
            f"Per-item evidence integrity error: {len(nonfinite_scores)} item(s) have "
            f"non-finite scores (NaN/Inf): {nonfinite_scores[:10]}. "
            "Non-finite scores cannot be reconciled with aggregate evidence."
        )

    # If any per-item score is invalid, skip aggregate reconciliation (errors already logged)
    if missing_scores or nonfinite_scores:
        return

    raw_dir = run_dir / "raw"
    if not raw_dir.exists():
        if fail_on_missing:
            errors.append(
                f"raw/ directory not found in {run_dir}; aggregate result cannot be verified. "
                "An aggregate result file is required for publication."
            )
        return

    # Find aggregate result file
    result_files = list(raw_dir.glob("results_*.json")) + list(raw_dir.glob("*results*.json"))
    if not result_files:
        if fail_on_missing:
            errors.append(
                f"No aggregate result file (results_*.json) found under {raw_dir}. "
                "An aggregate result is required for publication."
            )
        return  # Not fail_on_missing: skip silently

    try:
        agg_data = json.loads(result_files[0].read_text())
    except (json.JSONDecodeError, OSError):
        errors.append(
            f"Could not parse aggregate result file {result_files[0].name}. "
            "Aggregate reconciliation cannot be verified."
        )
        return

    if not valid_scores:
        return

    per_item_mean = sum(valid_scores) / len(valid_scores)

    # Extract aggregate score from result file
    results_block = agg_data.get("results", {})
    agg_score = None
    for task_data in results_block.values():
        if isinstance(task_data, dict):
            for key in ("exact_match,flexible-fallback", "exact_match,flexible-extract",
                        "prompt_level_strict_acc,none", "exact_match,answer-line"):
                if key in task_data:
                    raw_val = task_data[key]
                    if raw_val is None:
                        # None aggregate score cannot be reconciled — fail-closed for publication
                        if fail_on_missing:
                            errors.append(
                                f"Aggregate score for key {key!r} in {result_files[0].name} "
                                f"is null/None — cannot reconcile with per-item evidence. "
                                "A valid numeric aggregate score is required for publication."
                            )
                            return
                        # For non-publication validation, skip silently
                        return
                    try:
                        agg_score = float(raw_val)
                        import math
                        if not math.isfinite(agg_score):
                            errors.append(
                                f"Aggregate score for key {key!r} is non-finite "
                                f"({raw_val!r}). Cannot reconcile with per-item evidence."
                            )
                            return
                    except (TypeError, ValueError):
                        pass
                    break
        if agg_score is not None:
            break

    if agg_score is None:
        return  # Cannot extract aggregate score, skip

    diff = abs(agg_score - per_item_mean)
    if diff > _AGGREGATE_TOLERANCE:
        errors.append(
            f"Aggregate score mismatch: aggregate={agg_score:.4f} vs "
            f"per-item mean={per_item_mean:.4f} (diff={diff:.4f}, "
            f"tolerance={_AGGREGATE_TOLERANCE}). "
            "Aggregate scores must reconcile with per-item evidence."
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Validate a campaign run directory before publication."
    )
    ap.add_argument("run_dir", type=pathlib.Path, help="Run directory to validate")
    ap.add_argument("--suite", type=pathlib.Path, required=True,
                    help="Path to suite YAML file")
    ap.add_argument("--adapter", type=pathlib.Path, required=True,
                    help="Path to adapter YAML file")
    ap.add_argument("--for-publication", action="store_true",
                    help="Also enforce the publication lifecycle gate")
    args = ap.parse_args(argv)

    result = validate(
        args.run_dir,
        args.suite,
        args.adapter,
        for_publication=args.for_publication,
    )

    if result.passed:
        print(f"OK: {args.run_dir} passed all validation gates.")
        return 0
    else:
        if not result.eligible:
            print(f"INELIGIBLE: {args.run_dir} is not certifiable by this validator:")
        else:
            print(f"FAIL: {args.run_dir} failed validation:")
        for err in result.errors:
            print(f"  - {err}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
