"""Fail-closed artifact registry loader for warpcore-benchmarks.

Usage
-----
from viz_registry import lookup_path, load_registry

# Raises KeyError if the path is not registered:
entry = lookup_path("results/ornith-35b/raw/quality/gsm8k/results_2026-08-18T22-22-03.403814.json")
print(entry["status"])  # "historical"

# Full registry dict (cached):
registry = load_registry()

Lifecycle statuses
------------------
historical   — pre-contract evidence; artifact is real, provenance is incomplete
superseded   — replaced by a newer run; kept for auditability
diagnostic   — deliberate diagnostic-only run, not for publication
replay       — composite of an earlier run + targeted replay of failing items;
               must carry base_run and optionally superseded_by
invalid      — run aborted, corrupt, or infrastructurally blocked; not quotable
current      — v1-contract run; requires v1_validated=true in the registry entry

Fail-closed contract
--------------------
lookup_path() raises KeyError for any path not in the registry.
collect_matrix.py and audit_provenance.py must call lookup_path() so that a
newly-added result file cannot silently enter the published matrix without a
registry classification.  Unregistered == blocked.
"""
from __future__ import annotations

import json
import pathlib
from typing import Optional

REPO = pathlib.Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO / "results" / "registry.json"

# ---------------------------------------------------------------------------
# Internal cache
# ---------------------------------------------------------------------------

_REGISTRY: Optional[dict] = None
_PATH_INDEX: Optional[dict[str, dict]] = None

VALID_STATUSES = {"historical", "superseded", "diagnostic", "replay", "invalid", "current"}


def validate_registry(registry: dict, *, check_paths: bool = False) -> None:
    """Validate structure/lifecycle; optionally verify repository files exist."""
    if registry.get("registry_version") != 1 or not isinstance(registry.get("entries"), list):
        raise ValueError("registry_version must be 1 and entries must be a list")
    ids: set[str] = set()
    paths: set[str] = set()
    for entry in registry["entries"]:
        entry_id = entry.get("id")
        if not isinstance(entry_id, str) or not entry_id:
            raise ValueError("registry entry id must be a non-empty string")
        if entry_id in ids:
            raise ValueError(f"duplicate registry id: {entry_id}")
        ids.add(entry_id)
        status = entry.get("status")
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid registry status for {entry_id}: {status!r}")
        entry_paths = entry.get("paths")
        if not isinstance(entry_paths, list) or not entry_paths:
            raise ValueError(f"registry entry {entry_id} must have non-empty paths")
        for path in entry_paths:
            if not isinstance(path, str) or not path.startswith("results/"):
                raise ValueError(f"invalid registry path in {entry_id}: {path!r}")
            if path in paths:
                raise ValueError(f"duplicate registry path: {path}")
            if check_paths and not (REPO / path).is_file():
                raise ValueError(f"registered path does not exist: {path}")
            paths.add(path)
        if not entry.get("note"):
            raise ValueError(f"registry entry {entry_id} requires a classification note")
        if status == "current" and entry.get("v1_validated") is not True:
            raise ValueError(f"current entry {entry_id} requires v1_validated=true")
        if status == "replay":
            if len(entry_paths) < 2:
                raise ValueError(f"replay entry {entry_id} requires at least two component paths")
            if entry.get("base_run") not in entry_paths:
                raise ValueError(f"replay entry {entry_id} base_run must name a registered component")


def _build_index(registry: dict) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for entry in registry["entries"]:
        for path in entry.get("paths", []):
            index[path] = entry
    return index


def load_registry() -> dict:
    """Return the full registry dict (cached after first load)."""
    global _REGISTRY, _PATH_INDEX
    if _REGISTRY is None:
        if not REGISTRY_PATH.exists():
            raise FileNotFoundError(
                f"results/registry.json not found at {REGISTRY_PATH}. "
                "Create it as part of Task 8."
            )
        raw_registry = json.loads(REGISTRY_PATH.read_text())
        validate_registry(raw_registry, check_paths=True)
        _REGISTRY = raw_registry
        _PATH_INDEX = _build_index(raw_registry)
    return _REGISTRY


def _path_index() -> dict[str, dict]:
    load_registry()
    assert _PATH_INDEX is not None
    return _PATH_INDEX


def lookup_path(rel_path: str) -> dict:
    """Return the registry entry for *rel_path* (repo-relative).

    Raises KeyError if the path is not registered.  This is intentional:
    unregistered paths must not silently enter the published matrix.
    """
    idx = _path_index()
    if rel_path not in idx:
        raise KeyError(
            f"Path not in registry: {rel_path!r}. "
            "Add a classified entry to results/registry.json before using "
            "this artifact in collect_matrix or audit_provenance."
        )
    return idx[rel_path]


def status_for_path(rel_path: str) -> str:
    """Convenience wrapper — returns just the status string."""
    return lookup_path(rel_path)["status"]


def is_historical(rel_path: str) -> bool:
    """True if the artifact is pre-contract historical evidence."""
    return status_for_path(rel_path) in {"historical", "superseded", "diagnostic",
                                          "replay", "invalid"}


def is_current(rel_path: str) -> bool:
    """True only when the artifact is a validated v1-contract run."""
    entry = lookup_path(rel_path)
    return entry.get("status") == "current" and bool(entry.get("v1_validated"))


def reload() -> None:
    """Invalidate the cache (useful in tests after writing a tmp registry)."""
    global _REGISTRY, _PATH_INDEX
    _REGISTRY = None
    _PATH_INDEX = None
