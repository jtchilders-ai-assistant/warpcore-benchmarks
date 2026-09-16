"""
viz/create_campaign.py — Create a new warpcore-v1 benchmark campaign.

Public API:
    create_campaign(repo, suite_path, adapter_path, benchmark, run_id) -> Path
    validate_safe_path_component(component) -> None

Exceptions:
    OutputDirectoryCollisionError   — run directory already exists (without valid resume)
    NoncanonicalAdapterError        — adapter is not canonical
    UnsafePathComponentError        — path component contains unsafe characters
    SuiteValidationError            — suite YAML fails contract.validate_suite checks
    InventoryConflictError          — caller item_count conflicts with suite expected_item_count
    ResumeIdentityMismatchError     — resume=True but existing dir identity/inventory mismatch

create_campaign:
  1. Validates path components (run_id, model slug, benchmark) for safety.
  2. Validates the full suite via contract.validate_suite (fails on unknown benchmark,
     missing/escaped input, hash mismatch).
  3. Validates the adapter is canonical (campaign-ready).
  4. Resolves and records all suite input hashes (including suite YAML) and adapter
     hash before any command generation.
  5. Derives expected item inventory from suite expected_item_count; requires explicit
     positive item_count for benchmarks without it; rejects conflicting caller item_count.
  6. Validates manifest and status against their JSON schemas before any write.
  7. Creates the normalized run directory transactionally:
       - Stages in a sibling directory
       - Writes manifest.json and status.json into staging
       - Atomically renames staging -> final
       - Any exception cleans staging and leaves no final run dir
  8. With resume=True: reuses an existing dir only when manifest/status validate and
     identity+inventory exactly equal expected (suite/run/benchmark, adapter hash,
     suite input hashes, serving profile digest, model identity, expected item count).
     Otherwise fails closed. resume=False always raises on collision.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from typing import Optional

# Allow import from viz/ when this file is run/tested directly
_VIZ_DIR = pathlib.Path(__file__).parent
for _p in (str(_VIZ_DIR),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import campaign_state
from contract import (
    load_yaml,
    validate_adapter,
    validate_adapter_campaign_ready,
    validate_json,
    validate_suite,
)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class OutputDirectoryCollisionError(Exception):
    """Raised when the target run directory already exists and resume is not valid."""


class NoncanonicalAdapterError(Exception):
    """Raised when the adapter is not campaign-ready (noncanonical)."""


class UnsafePathComponentError(Exception):
    """Raised when a path component contains unsafe characters."""


class SuiteValidationError(Exception):
    """Raised when the suite YAML fails contract.validate_suite checks."""


class InventoryConflictError(Exception):
    """Raised when caller item_count conflicts with suite expected_item_count."""


class ResumeIdentityMismatchError(Exception):
    """Raised when resume=True but existing dir identity/inventory does not match."""


# ---------------------------------------------------------------------------
# Safe path component validation
# ---------------------------------------------------------------------------

# Allow alphanumerics, hyphens, underscores, dots, and colons.
# Explicitly disallow: slash, backslash, null byte, spaces, and traversal sequences.
_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]*$")


def validate_safe_path_component(component: str) -> None:
    """Validate that *component* is safe to use as a directory or file name.

    Rejects:
    - Empty strings
    - Strings containing path separators (/ or \\)
    - Strings containing null bytes
    - Strings containing spaces or control characters
    - Relative traversal sequences (..)
    - Strings not matching the safe pattern

    Raises UnsafePathComponentError on failure.
    """
    if not isinstance(component, str) or not component:
        raise UnsafePathComponentError(
            f"Path component must be a non-empty string; got: {component!r}"
        )
    if "\x00" in component:
        raise UnsafePathComponentError(
            f"Path component contains null byte: {component!r}"
        )
    if "/" in component or "\\" in component:
        raise UnsafePathComponentError(
            f"Path component must not contain path separators: {component!r}"
        )
    if " " in component:
        raise UnsafePathComponentError(
            f"Path component must not contain spaces: {component!r}"
        )
    if ".." in component.split("/"):
        raise UnsafePathComponentError(
            f"Path component must not contain traversal sequences: {component!r}"
        )
    if not _SAFE_COMPONENT_RE.match(component):
        raise UnsafePathComponentError(
            f"Path component contains unsafe characters: {component!r}. "
            "Allowed: alphanumerics, hyphens, underscores, dots, colons, at-signs."
        )


# ---------------------------------------------------------------------------
# Hash resolution helpers
# ---------------------------------------------------------------------------


def _sha256_hex(path: pathlib.Path) -> str:
    """Return the SHA-256 hex digest of *path*."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _collect_suite_input_hashes(
    repo: pathlib.Path,
    suite: dict,
    suite_path: pathlib.Path,
    benchmark: str,
) -> dict[str, str]:
    """Resolve and return SHA-256 hashes for suite inputs used by *benchmark*.

    Always includes the suite YAML itself.
    Keys are repo-relative paths under suite/ (e.g. 'suite/tasks/gsm8k_clean_v1.yaml').
    Values are 64-char lowercase hex SHA-256 digests.

    Only files declared in the benchmark block (task_file, utils_file, instance_set_file,
    scaffold_file) are included; empty or absent fields are skipped.
    """
    hashes: dict[str, str] = {}
    repo = repo.resolve()
    suite_path = suite_path.resolve()

    # Always include the suite YAML hash
    try:
        suite_rel = suite_path.relative_to(repo)
    except ValueError as exc:
        raise SuiteValidationError(
            f"Suite path {suite_path} is outside repository {repo}"
        ) from exc
    if not suite_rel.parts or suite_rel.parts[0] != "suite":
        raise SuiteValidationError(
            f"Suite path must be under {repo / 'suite'}; got {suite_path}"
        )
    hashes[str(suite_rel)] = _sha256_hex(suite_path)

    benchmarks = suite.get("benchmarks", {})
    bench = benchmarks.get(benchmark, {})
    if not isinstance(bench, dict):
        return hashes

    for field in ("task_file", "utils_file", "instance_set_file", "scaffold_file"):
        rel = bench.get(field)
        if not rel:
            continue
        abs_path = (repo / rel).resolve()
        # Safety: must resolve inside repo
        try:
            abs_path.relative_to(repo)
        except ValueError as exc:
            raise SuiteValidationError(
                f"Suite input {rel!r} resolves outside repository {repo}"
            ) from exc
        if not abs_path.is_file():
            raise SuiteValidationError(f"Suite input not found or not a file: {rel}")
        hashes[rel] = _sha256_hex(abs_path)

    return hashes


def _serving_profile_digest(adapter: dict, hardware_id: str = "dgx-spark-gb10") -> str:
    """Compute a deterministic SHA-256 over the full serving profile identity.

    Identity components: model ID, checkpoint revision, image (from adapter),
    effective engine args (sorted), environment, hardware_id, and all parser/
    tokenizer/moe/backend/resource controls.

    Returns 'sha256:<64-char-hex>'.
    """
    model = adapter.get("model", {})
    serving = adapter.get("serving", {})
    identity = {
        "model_id": model.get("id", ""),
        "revision": model.get("revision", ""),
        "image": serving.get("image", ""),
        "engine": serving.get("engine", ""),
        "engine_version": serving.get("engine_version", ""),
        "quantization": serving.get("quantization"),
        "reasoning_parser": serving.get("reasoning_parser"),
        "tool_call_parser": serving.get("tool_call_parser"),
        "tokenizer": serving.get("tokenizer"),
        "moe_backend": serving.get("moe_backend"),
        "max_model_len": serving.get("max_model_len"),
        "gpu_memory_utilization": serving.get("gpu_memory_utilization"),
        "max_num_seqs": serving.get("max_num_seqs"),
        "environment": serving.get("environment") or {},
        "hardware_id": hardware_id,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return f"sha256:{digest}"


def _build_effective_args(serving: dict) -> list[str]:
    """Build the effective_args list from adapter serving settings.

    Includes all non-None/non-empty serving fields that affect model behavior.
    """
    args: list[str] = []
    field_map = [
        ("quantization", "--quantization"),
        ("reasoning_parser", "--reasoning-parser"),
        ("tool_call_parser", "--tool-call-parser"),
        ("tokenizer", "--tokenizer"),
        ("moe_backend", "--moe-backend"),
        ("max_model_len", "--max-model-len"),
        ("gpu_memory_utilization", "--gpu-memory-utilization"),
        ("max_num_seqs", "--max-num-seqs"),
        ("engine", "--engine"),
        ("engine_version", "--engine-version"),
    ]
    for key, flag in field_map:
        val = serving.get(key)
        if val is not None:
            args.append(f"{flag}={val}")
    # Environment vars
    env = serving.get("environment") or {}
    for k, v in sorted(env.items()):
        args.append(f"--env={k}={v}")
    return args


# ---------------------------------------------------------------------------
# Resume validation helpers
# ---------------------------------------------------------------------------

def _load_existing_manifest(run_dir: pathlib.Path) -> dict:
    """Load and return manifest.json from an existing run directory.

    Raises FileNotFoundError if manifest.json does not exist.
    """
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json not found in existing run directory: {run_dir}. "
            "Cannot resume without a valid manifest."
        )
    return json.loads(manifest_path.read_text())


def _validate_resume_identity(
    existing_manifest: dict,
    expected_suite_id: str,
    expected_run_id: str,
    expected_benchmark: str,
    expected_adapter_hash: str,
    expected_suite_input_hashes: dict,
    expected_profile_digest: str,
    expected_model_slug: str,
    expected_model_id: str,
    expected_model_revision: str,
    expected_item_count: int,
) -> None:
    """Validate that an existing manifest's identity matches all expected values.

    Raises ResumeIdentityMismatchError if any field differs.
    """
    mismatches: list[str] = []

    def _check(field: str, actual, expected):
        if actual != expected:
            mismatches.append(f"{field}: existing={actual!r}, expected={expected!r}")

    _check("suite_id", existing_manifest.get("suite_id"), expected_suite_id)
    _check("run_id", existing_manifest.get("run_id"), expected_run_id)
    _check("benchmark", existing_manifest.get("benchmark"), expected_benchmark)
    _check("adapter_hash", existing_manifest.get("adapter_hash"), expected_adapter_hash)
    _check("serving_profile_digest", existing_manifest.get("serving_profile_digest"), expected_profile_digest)

    # Model identity
    m = existing_manifest.get("model", {})
    _check("model.slug", m.get("slug"), expected_model_slug)
    _check("model.id", m.get("id"), expected_model_id)
    _check("model.revision", m.get("revision"), expected_model_revision)

    # Suite input hashes are an exact identity map; extras are also drift.
    existing_hashes = existing_manifest.get("suite_input_hashes", {})
    _check("suite_input_hashes", existing_hashes, expected_suite_input_hashes)

    # Item inventory
    existing_count = (existing_manifest.get("item_inventory") or {}).get("expected")
    _check("item_inventory.expected", existing_count, expected_item_count)

    if mismatches:
        raise ResumeIdentityMismatchError(
            "resume=True but existing run directory identity does not match current call:\n"
            + "\n".join(f"  {m}" for m in mismatches)
        )


# ---------------------------------------------------------------------------
# Inventory resolution
# ---------------------------------------------------------------------------

def _resolve_item_count(
    suite: dict,
    benchmark: str,
    caller_item_count: Optional[int],
) -> int:
    """Derive the expected item count for *benchmark*.

    Rules:
    - If suite declares expected_item_count, use it.
    - If caller_item_count is provided and matches suite value, accepted.
    - If caller_item_count conflicts with suite value, raise InventoryConflictError.
    - If suite has no expected_item_count and caller_item_count is positive, use it.
    - If suite has no expected_item_count and no caller_item_count, raise ValueError.
    """
    bench = (suite.get("benchmarks") or {}).get(benchmark, {})
    suite_count = bench.get("expected_item_count") if isinstance(bench, dict) else None

    if suite_count is not None:
        # Suite declares a count
        if caller_item_count is not None and caller_item_count != suite_count:
            raise InventoryConflictError(
                f"caller item_count={caller_item_count} conflicts with suite "
                f"expected_item_count={suite_count} for benchmark {benchmark!r}. "
                "Pass item_count matching the suite value, or omit it."
            )
        return suite_count
    else:
        # No suite count — caller must supply a positive value
        if caller_item_count is None:
            raise ValueError(
                f"Benchmark {benchmark!r} has no expected_item_count in the suite. "
                "Provide an explicit positive item_count argument."
            )
        if not isinstance(caller_item_count, int) or caller_item_count <= 0:
            raise ValueError(
                f"item_count must be a positive integer; got: {caller_item_count!r}"
            )
        return caller_item_count


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def create_campaign(
    repo: pathlib.Path,
    suite_path: pathlib.Path,
    adapter_path: pathlib.Path,
    benchmark: str,
    run_id: str,
    hardware_id: str = "dgx-spark-gb10",
    item_count: Optional[int] = None,
    resume: bool = False,
    prompt_token_maxima: Optional[dict[str, int]] = None,
) -> pathlib.Path:
    """Create a new campaign run directory.

    Steps:
    1. Validate path components.
    2. Validate the full suite via contract.validate_suite.
    3. Validate the adapter is canonical (campaign-ready).
    4. Resolve all suite input hashes (including suite YAML) and adapter hash.
    5. Derive expected item inventory from suite expected_item_count.
    6. Build manifest and status; validate both against JSON schemas.
    7. Create the normalized run directory transactionally (staging + atomic rename).
    8. If resume=True and dir exists: validate identity/inventory before returning.

    Parameters
    ----------
    repo:
        Repository root path.
    suite_path:
        Absolute path to the suite YAML.
    adapter_path:
        Absolute path to the adapter YAML.
    benchmark:
        Benchmark name (e.g. 'gsm8k').
    run_id:
        Run identifier (e.g. 'run-2026-09-15T00-00-00').
    hardware_id:
        Hardware identity string for the serving profile digest.
    item_count:
        Expected item count. If suite declares expected_item_count, must match or be None.
        If suite lacks expected_item_count, must be a positive integer.
    resume:
        If True, reuse an existing run dir when identity+inventory match exactly.
        If False (default), raise OutputDirectoryCollisionError on existing dir.
    prompt_token_maxima:
        Measured tokenized prompt maxima keyed by benchmark. Required to prove
        prompt-plus-output context feasibility for canonical campaign creation.

    Returns
    -------
    pathlib.Path
        The created (or resumed) run directory.

    Raises
    ------
    OutputDirectoryCollisionError
        If the run directory already exists and resume=False.
    ResumeIdentityMismatchError
        If resume=True but existing dir identity/inventory doesn't match.
    NoncanonicalAdapterError
        If the adapter is not campaign-ready.
    UnsafePathComponentError
        If any path component is unsafe.
    SuiteValidationError
        If the suite fails contract.validate_suite checks.
    InventoryConflictError
        If caller item_count conflicts with suite expected_item_count.
    ValueError
        If suite lacks expected_item_count and no item_count provided.
    """
    repo = pathlib.Path(repo).resolve()
    suite_path = pathlib.Path(suite_path).resolve()
    adapter_path = pathlib.Path(adapter_path).resolve()

    # -- 1. Validate path components -----------------------------------------
    validate_safe_path_component(benchmark)
    validate_safe_path_component(run_id)

    # -- 2. Load and fully validate suite ------------------------------------
    suite = load_yaml(suite_path)

    # Run full suite validation via contract.validate_suite
    suite_errors = validate_suite(repo, suite_path)
    if suite_errors:
        raise SuiteValidationError(
            f"Suite validation failed for {suite_path}:\n"
            + "\n".join(suite_errors)
        )

    # Verify the requested benchmark exists in the suite
    benchmarks = suite.get("benchmarks", {})
    if benchmark not in benchmarks:
        raise SuiteValidationError(
            f"Benchmark {benchmark!r} not found in suite {suite_path}. "
            f"Available benchmarks: {sorted(benchmarks.keys())}"
        )

    # -- 3. Load and validate adapter ----------------------------------------
    adapter = load_yaml(adapter_path)

    # Validate adapter schema
    adapter_errors = validate_adapter(repo, adapter_path)
    if adapter_errors:
        raise NoncanonicalAdapterError(
            f"Adapter schema validation failed:\n" + "\n".join(adapter_errors)
        )

    # Validate campaign readiness (canonical)
    model_slug = (adapter.get("model") or {}).get("slug", "")
    validate_safe_path_component(model_slug)

    readiness_errors = validate_adapter_campaign_ready(
        adapter,
        model_slug,
        suite=suite,
        prompt_token_maxima=prompt_token_maxima,
    )
    if readiness_errors:
        raise NoncanonicalAdapterError(
            f"Adapter is not campaign-ready:\n" + "\n".join(readiness_errors)
        )

    # -- 4. Resolve hashes before building paths or commands -----------------
    adapter_hash = _sha256_hex(adapter_path)
    suite_input_hashes = _collect_suite_input_hashes(repo, suite, suite_path, benchmark)
    profile_digest = _serving_profile_digest(adapter, hardware_id=hardware_id)

    # -- 5. Resolve item inventory -------------------------------------------
    expected_item_count = _resolve_item_count(suite, benchmark, item_count)

    # -- 6. Build run directory path -----------------------------------------
    suite_id = suite.get("suite_id", "warpcore-v1")
    run_dir = repo / "results" / model_slug / "runs" / suite_id / benchmark / run_id

    # -- 7. Build manifest and status dicts ----------------------------------
    now_utc = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    model = adapter.get("model", {})
    serving = adapter.get("serving", {})

    manifest: dict = {
        "schema_version": 1,
        "suite_id": suite_id,
        "suite_schema_version": suite.get("suite_schema_version", 1),
        "run_id": run_id,
        "benchmark": benchmark,
        "adapter_hash": adapter_hash,
        "suite_input_hashes": suite_input_hashes,
        "serving_profile_digest": profile_digest,
        "model": {
            "slug": model.get("slug", ""),
            "id": model.get("id", ""),
            "revision": model.get("revision", ""),
        },
        "serving": {
            "image_digest": "unrecorded",  # filled below
            "engine": serving.get("engine", ""),
            "engine_version": serving.get("engine_version", ""),
            "effective_args": _build_effective_args(serving),
            "environment": serving.get("environment") or {},
            "hardware_id": hardware_id,
        },
        "item_inventory": {
            "expected": expected_item_count,
            "submitted": 0,
        },
        "timing": {
            "started_utc": now_utc,
            "completed_utc": None,
        },
        "artifact_inventory": {
            "samples_jsonl_gz": False,
            "per_item_csv": False,
            "run_log": False,
            "command_txt": False,
            "done_sentinel": False,
        },
    }

    # Extract image digest from canonical adapter image reference
    image_field = serving.get("image", "unrecorded")
    if isinstance(image_field, str) and "@sha256:" in image_field:
        digest_part = image_field.split("@", 1)[1]  # e.g. 'sha256:<hex>'
        manifest["serving"]["image_digest"] = digest_part
    else:
        manifest["serving"]["image_digest"] = "unrecorded"

    status: dict = {
        "schema_version": 1,
        "run_id": run_id,
        "suite_id": suite_id,
        "execution_state": "planned",
        "lifecycle": "current",
        "history": [
            {
                "state": "planned",
                "timestamp": now_utc,
                "note": "Campaign initialized by create_campaign",
            }
        ],
    }

    # -- 8. Validate manifest and status before any filesystem write ----------
    schema_dir = repo / "suite" / "schemas"
    manifest_schema = schema_dir / "manifest.schema.json"
    status_schema = schema_dir / "result-status.schema.json"

    manifest_errors = validate_json(manifest, manifest_schema)
    if manifest_errors:
        raise ValueError(
            f"Manifest schema validation failed before write:\n"
            + "\n".join(manifest_errors)
        )

    status_errors = validate_json(status, status_schema)
    if status_errors:
        raise ValueError(
            f"Status schema validation failed before write:\n"
            + "\n".join(status_errors)
        )

    # -- 9. Handle existing directory (collision / resume) -------------------
    if run_dir.exists():
        if not resume:
            raise OutputDirectoryCollisionError(
                f"Run directory already exists: {run_dir}. "
                "Refuse to overwrite. Use resume=True if identity and inventory match."
            )
        # resume=True: validate that existing dir matches current identity
        try:
            existing_manifest = _load_existing_manifest(run_dir)
            existing_status = json.loads((run_dir / "status.json").read_text())
        except (OSError, ValueError, TypeError) as exc:
            raise ResumeIdentityMismatchError(
                f"Cannot resume {run_dir}: manifest/status is missing or unreadable: {exc}"
            ) from exc
        existing_manifest_errors = validate_json(existing_manifest, manifest_schema)
        existing_status_errors = validate_json(existing_status, status_schema)
        if existing_manifest_errors or existing_status_errors:
            details = existing_manifest_errors + existing_status_errors
            raise ResumeIdentityMismatchError(
                "Cannot resume: existing manifest/status fails schema validation:\n"
                + "\n".join(details)
            )
        _validate_resume_identity(
            existing_manifest,
            expected_suite_id=suite_id,
            expected_run_id=run_id,
            expected_benchmark=benchmark,
            expected_adapter_hash=adapter_hash,
            expected_suite_input_hashes=suite_input_hashes,
            expected_profile_digest=profile_digest,
            expected_model_slug=model.get("slug", ""),
            expected_model_id=model.get("id", ""),
            expected_model_revision=model.get("revision", ""),
            expected_item_count=expected_item_count,
        )
        status_mismatches = []
        if existing_status.get("run_id") != run_id:
            status_mismatches.append("status.run_id")
        if existing_status.get("suite_id") != suite_id:
            status_mismatches.append("status.suite_id")
        if status_mismatches:
            raise ResumeIdentityMismatchError(
                "Cannot resume: status identity mismatch in " + ", ".join(status_mismatches)
            )
        # Identity matches — return existing dir
        return run_dir

    # -- 10. Transactional directory creation --------------------------------
    # Stage in a sibling directory, then atomically rename to final.
    staging_dir: Optional[pathlib.Path] = None
    try:
        # Create parent first so we can create a sibling staging dir
        run_dir.parent.mkdir(parents=True, exist_ok=True)

        # Create staging sibling
        staging_dir = pathlib.Path(
            tempfile.mkdtemp(
                dir=run_dir.parent,
                prefix=f".staging_{run_id}_",
            )
        )

        # Write manifest.json into staging
        manifest_path = staging_dir / "manifest.json"
        campaign_state.write_status(manifest_path, manifest, run_dir=staging_dir)

        # Write status.json into staging
        status_path = staging_dir / "status.json"
        campaign_state.write_status(status_path, status, run_dir=staging_dir)

        # Atomically rename staging -> final
        staging_dir.rename(run_dir)
        staging_dir = None  # Prevent cleanup: rename succeeded

    except BaseException:
        # Clean up staging dir on any failure
        if staging_dir is not None and staging_dir.exists():
            try:
                shutil.rmtree(staging_dir)
            except OSError:
                pass
        raise

    return run_dir
