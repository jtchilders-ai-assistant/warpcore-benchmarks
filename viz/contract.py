"""
viz/contract.py — warpcore-v1 contract validation library.

Public API:
    load_yaml(path: Path) -> dict
    sha256_file(path: Path) -> str
    validate_json(instance: dict, schema_path: Path) -> list[str]
    validate_suite(repo: Path, suite_path: Path) -> list[str]
    validate_adapter(repo: Path, adapter_path: Path) -> list[str]

validate_json enforces JSON Schema format annotations via FormatChecker.
validate_suite and validate_adapter return [] on success, list[str] of
human-readable error messages on failure.

Exit codes (for CLI callers):
    0 — valid
    1 — diagnosed defect
    2 — unreadable / inconclusive input
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import jsonschema
import yaml


# ---------------------------------------------------------------------------
# Primitive helpers
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> dict:
    """Load a YAML file and return the parsed dict.

    Raises OSError / yaml.YAMLError if the file cannot be read or parsed.
    """
    return yaml.safe_load(Path(path).read_text())


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of the file at *path* (64 lowercase hex chars).

    Raises OSError if the file cannot be read.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_json(instance: dict, schema_path: Path) -> list[str]:
    """Validate *instance* against the JSON Schema at *schema_path*.

    FormatChecker is always enabled so date-time format annotations are
    enforced.

    Returns [] on success, or a list of human-readable error strings.
    """
    schema = json.loads(Path(schema_path).read_text())
    validator_cls = jsonschema.validators.validator_for(schema)
    validator = validator_cls(schema, format_checker=jsonschema.FormatChecker())
    errors = list(validator.iter_errors(instance))
    if not errors:
        return []
    return [_format_error(e) for e in errors]


def _format_error(err: jsonschema.ValidationError) -> str:
    path = " -> ".join(str(p) for p in err.absolute_path) if err.absolute_path else "<root>"
    return f"{path}: {err.message}"


# ---------------------------------------------------------------------------
# Suite validator
# ---------------------------------------------------------------------------

def validate_suite(repo: Path, suite_path: Path) -> list[str]:
    """Validate the suite YAML at *suite_path* against the repo at *repo*.

    Checks performed:
    1. JSON Schema validation against suite/schemas/suite.schema.json
    2. All canonical task/utils file hashes match actual files
    3. SWE-bench instance set: hash, count, uniqueness, expected_item_count
    4. All referenced canonical file paths resolve inside the repo (no escape)

    Returns [] on success, list[str] of error messages on failure.
    """
    repo = Path(repo).resolve()
    suite_path = Path(suite_path).resolve()
    errors: list[str] = []

    # -- 1. Schema validation -------------------------------------------------
    schema_path = repo / "suite" / "schemas" / "suite.schema.json"
    try:
        suite = load_yaml(suite_path)
    except (OSError, IOError) as exc:
        # Unreadable file: caller (CLI) should treat as exit 2 (inconclusive)
        raise
    except yaml.YAMLError as exc:
        # Syntactically unparseable YAML: unreadable/inconclusive → re-raise
        # so the CLI can map it to exit 2, not exit 1.
        raise

    # Structurally non-mapping YAML (null, scalar, list) is syntactically
    # valid but is a diagnosed contract defect → return list[str], exit 1.
    if not isinstance(suite, dict):
        kind = type(suite).__name__ if suite is not None else "null"
        return [
            f"Suite YAML must be a mapping (got {kind}): "
            f"the file parsed successfully but its top-level value is not a dict"
        ]

    if schema_path.exists():
        schema_errors = validate_json(suite, schema_path)
        errors.extend(schema_errors)
    else:
        errors.append(f"Suite schema not found: {schema_path}")

    # -- 2. Canonical file hash integrity -------------------------------------
    benchmarks = suite.get("benchmarks", {})
    for bench_name, bench in benchmarks.items():
        if not isinstance(bench, dict):
            continue

        # task_file / task_sha256
        task_file = bench.get("task_file")
        task_sha256 = bench.get("task_sha256")
        if task_file and task_sha256:
            task_path = _resolve_repo_path(repo, task_file)
            if task_path is None:
                errors.append(
                    f"{bench_name}: task_file path escapes repo root: {task_file!r}"
                )
            elif not task_path.exists():
                errors.append(
                    f"{bench_name}: task_file not found: {task_path}"
                )
            else:
                actual = sha256_file(task_path)
                if actual != task_sha256:
                    errors.append(
                        f"{bench_name}: task_sha256 mismatch — "
                        f"declared {task_sha256!r}, actual {actual!r} ({task_file})"
                    )

        # utils_file / utils_sha256
        utils_file = bench.get("utils_file")
        utils_sha256 = bench.get("utils_sha256")
        if utils_file and utils_sha256:
            utils_path = _resolve_repo_path(repo, utils_file)
            if utils_path is None:
                errors.append(
                    f"{bench_name}: utils_file path escapes repo root: {utils_file!r}"
                )
            elif not utils_path.exists():
                errors.append(
                    f"{bench_name}: utils_file not found: {utils_path}"
                )
            else:
                actual = sha256_file(utils_path)
                if actual != utils_sha256:
                    errors.append(
                        f"{bench_name}: utils_sha256 mismatch — "
                        f"declared {utils_sha256!r}, actual {actual!r} ({utils_file})"
                    )

        # SWE-bench: instances_sha256 + expected_item_count + uniqueness
        if bench_name == "swebench":
            errors.extend(_validate_swebench(repo, bench))

    return errors

def _resolve_repo_path(repo: Path, rel: str) -> Path | None:
    """Resolve *rel* against *repo* and return the absolute path, or None if
    the resolved path escapes the repo root.
    """
    repo = repo.resolve()
    candidate = (repo / rel).resolve()
    try:
        candidate.relative_to(repo)
        return candidate
    except ValueError:
        return None


def _validate_swebench(repo: Path, bench: dict) -> list[str]:
    """Validate the SWE-bench benchmark block."""
    errors: list[str] = []

    instance_set_file = bench.get("instance_set_file")
    instances_sha256 = bench.get("instances_sha256")
    expected_count = bench.get("expected_item_count")

    if not instance_set_file:
        errors.append("swebench: missing instance_set_file")
        return errors

    instances_path = _resolve_repo_path(repo, instance_set_file)
    if instances_path is None:
        errors.append(f"swebench: instance_set_file path escapes repo root: {instance_set_file!r}")
        return errors

    if not instances_path.exists():
        errors.append(f"swebench: instance_set_file not found: {instances_path}")
        return errors

    # Hash integrity
    actual_hash = sha256_file(instances_path)
    if instances_sha256 and actual_hash != instances_sha256:
        errors.append(
            f"swebench: instances_sha256 mismatch — "
            f"declared {instances_sha256!r}, actual {actual_hash!r}"
        )

    # Load and validate contents
    try:
        instances = json.loads(instances_path.read_text())
    except Exception as exc:
        errors.append(f"swebench: cannot parse instance set: {exc}")
        return errors

    if not isinstance(instances, list):
        errors.append("swebench: instance set must be a JSON array")
        return errors

    # Uniqueness
    if len(instances) != len(set(instances)):
        seen: set[str] = set()
        dups: set[str] = set()
        for iid in instances:
            if iid in seen:
                dups.add(iid)
            seen.add(iid)
        errors.append(f"swebench: duplicate instance IDs found: {sorted(dups)[:5]}")

    # Count vs expected_item_count
    if expected_count is not None and len(instances) != expected_count:
        errors.append(
            f"swebench: expected_item_count={expected_count} but "
            f"instance set contains {len(instances)} IDs"
        )

    # Non-empty
    if len(instances) == 0:
        errors.append("swebench: instance set is empty")

    # scaffold_file / scaffold_sha256
    errors.extend(_validate_file_hash_pair(repo, bench, "swebench", "scaffold_file", "scaffold_sha256"))

    return errors


def _validate_file_hash_pair(
    repo: Path,
    bench: dict,
    bench_name: str,
    file_key: str,
    hash_key: str,
) -> list[str]:
    """Validate a declared file/hash pair within a benchmark block.

    Returns [] if the pair is absent (not declared) or valid.
    Returns list[str] of errors if declared but invalid (path escape, missing,
    or hash mismatch).
    """
    errors: list[str] = []
    file_rel = bench.get(file_key)
    declared_hash = bench.get(hash_key)

    if not file_rel:
        return errors  # not declared; nothing to check

    resolved = _resolve_repo_path(repo, file_rel)
    if resolved is None:
        errors.append(
            f"{bench_name}: {file_key} path escapes repo root: {file_rel!r}"
        )
        return errors

    if not resolved.exists():
        errors.append(
            f"{bench_name}: {file_key} not found: {resolved}"
        )
        return errors

    if declared_hash:
        actual = sha256_file(resolved)
        if actual != declared_hash:
            errors.append(
                f"{bench_name}: {hash_key} mismatch — "
                f"declared {declared_hash!r}, actual {actual!r} ({file_rel})"
            )

    return errors


# ---------------------------------------------------------------------------
# Adapter validator
# ---------------------------------------------------------------------------

def validate_adapter(repo: Path, adapter_path: Path) -> list[str]:
    """Validate a serving adapter YAML at *adapter_path*.

    Checks performed:
    1. The adapter file resolves inside the repo (no path escape)
    2. JSON Schema validation against suite/schemas/adapter.schema.json
       (schema uses additionalProperties:false, so experiment-level keys
       such as generation_ceiling, sampling, task, instance_ids, etc. are
       automatically rejected)

    Returns [] on success, list[str] of error messages on failure.
    """
    repo = Path(repo).resolve()
    adapter_path = Path(adapter_path).resolve()
    errors: list[str] = []

    # -- 1. Path escape check --------------------------------------------------
    try:
        adapter_path.relative_to(repo)
    except ValueError:
        errors.append(
            f"adapter path escapes repo root — "
            f"adapter is outside the repository: {adapter_path} "
            f"(repo: {repo})"
        )
        return errors

    # -- 2. Load adapter -------------------------------------------------------
    try:
        adapter = load_yaml(adapter_path)
    except Exception as exc:
        return [f"Cannot load adapter YAML: {exc}"]

    # -- 3. Schema validation --------------------------------------------------
    schema_path = repo / "suite" / "schemas" / "adapter.schema.json"
    if not schema_path.exists():
        errors.append(f"Adapter schema not found: {schema_path}")
        return errors

    schema_errors = validate_json(adapter, schema_path)
    errors.extend(schema_errors)

    return errors
