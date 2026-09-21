"""
viz/contract.py — warpcore-v1 contract validation library.

Public API:
    load_yaml(path: Path) -> dict
    sha256_file(path: Path) -> str
    validate_json(instance: dict, schema_path: Path) -> list[str]
    validate_suite(repo: Path, suite_path: Path) -> list[str]
    validate_adapter(repo: Path, adapter_path: Path) -> list[str]
    validate_adapter_campaign_ready(adapter: dict, slug: str, suite: dict | None) -> list[str]
    validate_adapters_dir(repo: Path, adapters_dir: Path) -> list[str]

validate_json enforces JSON Schema format annotations via FormatChecker.
validate_suite and validate_adapter return [] on success, list[str] of
human-readable error messages on failure.

validate_adapter_campaign_ready checks whether a schema-valid adapter may
launch a canonical warpcore-v1 campaign.  It is separate from validate_adapter
because a noncanonical adapter is schema-valid (for historical documentation)
but may not drive a canonical run.

validate_adapters_dir validates every *.yaml file in a directory against the
adapter schema and also performs cross-adapter checks (duplicate slugs).

Context-length validation note:
    Campaign readiness requires measured tokenized prompt maxima from the pinned
    task/tokenizer path.  For each quality benchmark it proves
    prompt_tokens + generation_ceiling <= max_model_len.  Missing evidence blocks
    readiness rather than treating output-ceiling-only arithmetic as proof.

Exit codes (for CLI callers):
    0 — valid
    1 — diagnosed defect
    2 — unreadable / inconclusive input
"""
from __future__ import annotations

import hashlib
import json
import re
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

    # Type guard: every element must be a non-empty string.
    # This must run BEFORE set(instances) so unhashable objects (dicts, lists,
    # etc.) do not cause a TypeError — they are a diagnosed contract defect
    # (exit 1), not an unreadable input (exit 2).
    type_errors: list[str] = []
    empty_errors: list[str] = []
    for idx, iid in enumerate(instances):
        if not isinstance(iid, str):
            type_errors.append(
                f"swebench: instance ID at position {idx} is not a string "
                f"(got {type(iid).__name__!r}: {iid!r})"
            )
        elif iid == "":
            empty_errors.append(
                f"swebench: instance ID at position {idx} is an empty string"
            )
    errors.extend(type_errors)
    errors.extend(empty_errors)

    # Only run hash-based uniqueness check when all IDs are non-empty strings;
    # otherwise the set() call would crash on unhashable objects, and duplicate
    # detection on empty strings is already covered above.
    if not type_errors and not empty_errors:
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

    # Suite-owned launch qualification set. Delegated to the authoritative gate
    # module so the derivation rule has exactly one implementation; imported
    # lazily because that module imports this one.
    import sys as _sys
    _viz_dir = str(Path(__file__).resolve().parent)
    if _viz_dir not in _sys.path:
        _sys.path.insert(0, _viz_dir)
    from swebench_qualification import validate_suite_qualification_config

    errors.extend(validate_suite_qualification_config(repo, bench))

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


# ---------------------------------------------------------------------------
# Campaign-readiness validator
# ---------------------------------------------------------------------------

def validate_adapter_campaign_ready(
    adapter: dict,
    slug: str,
    suite: dict | None = None,
    prompt_token_maxima: dict[str, int] | None = None,
) -> list[str]:
    """Check whether a schema-valid adapter may launch a canonical warpcore-v1 campaign.

    This check is SEPARATE from schema validation.  A noncanonical adapter is
    schema-valid (it documents historical serving configuration) but must not
    drive a canonical campaign.

    Checks:
    1. campaign_status must be 'canonical'.  If 'noncanonical', return errors
       naming every unresolved field specified in noncanonical_reason.
    2. model.revision must be a 40-char lowercase hex string (not 'unresolved').
    3. serving.image must match the canonical repo@sha256:<64hex> pattern.
    4. Context feasibility: when a suite is supplied, measured tokenized prompt
       maxima must also be supplied for every benchmark with a generation ceiling,
       and prompt_tokens + generation_ceiling must fit max_model_len.

    Returns [] if the adapter is campaign-ready; list[str] of errors otherwise.
    """
    errors: list[str] = []
    status = adapter.get("campaign_status", "canonical")

    # -- 1. campaign_status gating --------------------------------------------
    if status == "noncanonical":
        reason = adapter.get("noncanonical_reason", "(no reason given)")
        errors.append(
            f"adapter for {slug!r} is noncanonical and may not launch a canonical campaign — "
            f"unresolved provenance: {reason}"
        )
        # Also enumerate which specific immutable fields are unresolved
        model_rev = (adapter.get("model") or {}).get("revision", "")
        serving = adapter.get("serving") or {}
        serving_image = serving.get("image", "")
        if model_rev == "unresolved":
            errors.append(
                f"model.revision is 'unresolved' for {slug!r}: "
                "the historical HuggingFace commit SHA was never recorded. "
                "Verify the exact revision used and resolve before enabling canonical campaigns."
            )
        if serving_image == "unresolved":
            errors.append(
                f"serving.image is 'unresolved' for {slug!r}: "
                "the historical container image digest was never recorded. "
                "Verify the exact image digest used and resolve before enabling canonical campaigns."
            )
        for field in ("max_model_len", "gpu_memory_utilization", "max_num_seqs"):
            if serving.get(field) == "unresolved":
                errors.append(
                    f"serving.{field} is 'unresolved' for {slug!r}: verify the effective "
                    "serving value before enabling canonical campaigns."
                )
        # Return early — further checks assume a canonical adapter
        return errors

    # -- 2. Immutable revision check ------------------------------------------
    _HEX40 = re.compile(r"^[0-9a-f]{40}$")
    model_rev = (adapter.get("model") or {}).get("revision", "")
    if not _HEX40.match(model_rev):
        errors.append(
            f"model.revision for {slug!r} is not a valid 40-char lowercase hex SHA: {model_rev!r}"
        )

    # -- 3. Immutable image digest check ---------------------------------------
    _DIGEST = re.compile(r"^(?:.+@)?sha256:[0-9a-f]{64}$")
    serving_image = (adapter.get("serving") or {}).get("image", "")
    if not _DIGEST.match(serving_image):
        errors.append(
            f"serving.image for {slug!r} does not carry an immutable digest "
            f"(expected repo@sha256:<64hex> or local sha256:<64hex>): {serving_image!r}"
        )

    # -- 4. Context feasibility -----------------------------------------------
    # The gate requires measured tokenized prompt maxima from the real pinned
    # task/tokenizer path. Without that evidence, readiness is inconclusive and
    # therefore blocks canonical launch.
    if suite is not None:
        if prompt_token_maxima is None:
            errors.append(
                f"tokenized prompt evidence is missing for {slug!r}: cannot prove "
                "prompt tokens plus suite output ceiling fit serving.max_model_len"
            )
            return errors

        max_model_len = (adapter.get("serving") or {}).get("max_model_len")
        if not isinstance(max_model_len, int):
            errors.append(
                f"serving.max_model_len for {slug!r} is unresolved; context feasibility cannot be proven"
            )
            return errors

        benchmarks = suite.get("benchmarks", {})
        for bench_name, bench in benchmarks.items():
            if not isinstance(bench, dict):
                continue
            ceiling = bench.get("generation_ceiling")
            if ceiling is None:
                continue
            prompt_tokens = prompt_token_maxima.get(bench_name)
            if not isinstance(prompt_tokens, int) or prompt_tokens < 0:
                errors.append(
                    f"tokenized prompt maximum is missing or invalid for {bench_name!r}"
                )
                continue
            required = prompt_tokens + ceiling
            if max_model_len < required:
                errors.append(
                    f"serving.max_model_len={max_model_len} for {slug!r} is less than "
                    f"the {bench_name!r} tokenized prompt plus output requirement={required} "
                    f"({prompt_tokens}+{ceiling})"
                )

    return errors


# ---------------------------------------------------------------------------
# Directory-level adapter validator
# ---------------------------------------------------------------------------

def validate_adapters_dir(
    repo: Path,
    adapters_dir: Path,
    suite: dict | None = None,
    prompt_token_maxima_by_slug: dict[str, dict[str, int]] | None = None,
) -> list[str]:
    """Validate every *.yaml file in *adapters_dir* as a serving adapter.

    Checks performed:
    1. Each file is schema-valid against suite/schemas/adapter.schema.json.
    2. Every canonical adapter passes campaign-readiness validation.
    3. No two adapters share the same model.slug (duplicate slug detection).

    Noncanonical adapters are retained as schema-valid drafts and are not treated
    as CI failures solely because their unresolved provenance blocks launch.
    """
    repo = Path(repo).resolve()
    adapters_dir = Path(adapters_dir).resolve()
    errors: list[str] = []

    if not adapters_dir.is_dir():
        return [f"Adapters directory not found: {adapters_dir}"]

    schema_path = repo / "suite" / "schemas" / "adapter.schema.json"
    if not schema_path.exists():
        errors.append(f"Adapter schema not found: {schema_path}")
        return errors

    # Collect all yaml files
    yaml_files = sorted(adapters_dir.glob("*.yaml"))

    slug_to_files: dict[str, list[str]] = {}  # slug -> [filename, ...]

    for adapter_path in yaml_files:
        filename = adapter_path.name

        # Load
        try:
            adapter = load_yaml(adapter_path)
        except Exception as exc:
            errors.append(f"{filename}: cannot load YAML: {exc}")
            continue

        # Schema validation
        schema_errors = validate_json(adapter, schema_path)
        for err in schema_errors:
            errors.append(f"{filename}: {err}")

        # Track slugs (even if schema-invalid, to surface duplicate errors)
        if isinstance(adapter, dict):
            slug = (adapter.get("model") or {}).get("slug")
            if slug:
                slug_to_files.setdefault(slug, []).append(filename)
                if not schema_errors and adapter.get("campaign_status") == "canonical":
                    maxima = (
                        (prompt_token_maxima_by_slug or {}).get(slug)
                        if prompt_token_maxima_by_slug is not None
                        else None
                    )
                    readiness_errors = validate_adapter_campaign_ready(
                        adapter,
                        slug,
                        suite=suite,
                        prompt_token_maxima=maxima,
                    )
                    for err in readiness_errors:
                        errors.append(f"{filename}: {err}")

    # Duplicate slug detection
    for slug, files in slug_to_files.items():
        if len(files) > 1:
            errors.append(
                f"Duplicate model slug {slug!r} found in multiple adapter files: "
                f"{sorted(files)} — each model slug must be unique across all adapters."
            )

    return errors
