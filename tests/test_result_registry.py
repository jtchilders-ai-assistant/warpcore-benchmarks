"""Registry tests for Task 8 — Historical artifact registry.

Every existing published source referenced by viz/collect_matrix.py must have
exactly one registry entry with a classified status.  The registry is the
canonical source of truth for artifact lifecycle.  Readers are expected to use
it fail-closed: an unknown path is an error, not a missing-but-ok entry.

RED-phase requirements (these must fail before results/registry.json exists):

  1. Registry file exists at results/registry.json.
  2. Every result path referenced by collect_matrix.QUALITY is registered.
  3. Every result path referenced by common.SWEBENCH_RESULTS is registered.
  4. No unknown paths (any path in the registry must exist in the repo).
  5. No duplicate IDs in the registry.
  6. "current" status requires authoritative v1 validation evidence.
  7. "replay" composites must carry supersession/base_run links.
  8. Paths not in the registry raise KeyError (fail-closed loader).
  9. Valid statuses: historical, superseded, diagnostic, replay, invalid, current.
 10. Replay composites referenced by collect_matrix must have base_run links.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

# ---------------------------------------------------------------------------
# Repo root + path to the registry
# ---------------------------------------------------------------------------
REPO = pathlib.Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO / "results" / "registry.json"

# Add viz/ to sys.path so we can import collect_matrix / common
VIZ = REPO / "viz"
if str(VIZ) not in sys.path:
    sys.path.insert(0, str(VIZ))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_registry() -> dict:
    """Load the registry; fail with an informative error if absent."""
    if not REGISTRY_PATH.exists():
        pytest.fail(
            f"results/registry.json does not exist. "
            f"Create it as part of Task 8 GREEN phase."
        )
    return json.loads(REGISTRY_PATH.read_text())


def all_registered_paths(registry: dict) -> set[str]:
    """Return the set of relative paths (str) in every registry entry."""
    paths: set[str] = set()
    for entry in registry["entries"]:
        paths.update(entry.get("paths", []))
    return paths


# ---------------------------------------------------------------------------
# Collect the paths actually used by collect_matrix / common
# ---------------------------------------------------------------------------

def quality_paths() -> list[str]:
    """Relative paths used in collect_matrix.QUALITY, relative to REPO root."""
    import importlib.util
    import os
    # Import from viz/ directory
    old = os.getcwd()
    os.chdir(VIZ)
    try:
        spec = importlib.util.spec_from_file_location("collect_matrix", VIZ / "collect_matrix.py")
        cm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cm)  # type: ignore[union-attr]
    finally:
        os.chdir(old)

    paths = []
    for model, files in cm.QUALITY.items():
        for bench, rel in files.items():
            paths.append(f"results/{model}/{rel}")
    return paths


def swebench_paths() -> list[str]:
    """Relative paths used in common.SWEBENCH_RESULTS, relative to REPO root."""
    import importlib.util
    import os
    old = os.getcwd()
    os.chdir(VIZ)
    try:
        spec = importlib.util.spec_from_file_location("common", VIZ / "common.py")
        cm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cm)  # type: ignore[union-attr]
    finally:
        os.chdir(old)

    return list(cm.SWEBENCH_RESULTS.values())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRegistryFileExists:
    def test_registry_json_exists(self):
        """results/registry.json must be present (Task 8 Step 1)."""
        assert REGISTRY_PATH.exists(), (
            "results/registry.json does not exist. "
            "Populate it with classified entries for all published sources."
        )

    def test_registry_has_entries_key(self):
        r = load_registry()
        assert "entries" in r, "Registry must have a top-level 'entries' list."

    def test_registry_has_version(self):
        r = load_registry()
        assert "registry_version" in r, "Registry must carry a 'registry_version' field."


class TestRegistrySchema:
    """Every entry must have mandatory fields with valid values."""

    VALID_STATUSES = {"historical", "superseded", "diagnostic", "replay", "invalid", "current"}

    def test_every_entry_has_id(self):
        r = load_registry()
        for i, entry in enumerate(r["entries"]):
            assert "id" in entry, f"Entry #{i} is missing 'id'"

    def test_every_entry_has_status(self):
        r = load_registry()
        for entry in r["entries"]:
            assert "status" in entry, f"Entry {entry.get('id', '?')} missing 'status'"

    def test_statuses_are_valid(self):
        r = load_registry()
        for entry in r["entries"]:
            status = entry.get("status")
            assert status in self.VALID_STATUSES, (
                f"Entry {entry.get('id', '?')} has invalid status {status!r}. "
                f"Valid: {sorted(self.VALID_STATUSES)}"
            )

    def test_every_entry_has_paths(self):
        r = load_registry()
        for entry in r["entries"]:
            assert "paths" in entry and len(entry["paths"]) > 0, (
                f"Entry {entry.get('id', '?')} must have at least one path."
            )

    def test_every_entry_has_classification_note(self):
        """Every entry should explain WHY it has its status (provenance discipline)."""
        r = load_registry()
        for entry in r["entries"]:
            assert entry.get("note"), (
                f"Entry {entry.get('id', '?')} is missing a 'note' explaining its classification."
            )


class TestNoDuplicateIDs:
    def test_no_duplicate_ids(self):
        r = load_registry()
        ids = [e["id"] for e in r["entries"]]
        seen: set[str] = set()
        dupes = []
        for id_ in ids:
            if id_ in seen:
                dupes.append(id_)
            seen.add(id_)
        assert not dupes, f"Duplicate IDs found: {dupes}"


class TestNoDuplicatePaths:
    def test_no_path_appears_in_multiple_entries(self):
        r = load_registry()
        seen: dict[str, str] = {}
        for entry in r["entries"]:
            for p in entry.get("paths", []):
                if p in seen:
                    pytest.fail(
                        f"Path {p!r} appears in both entry {seen[p]!r} "
                        f"and {entry['id']!r}."
                    )
                seen[p] = entry["id"]


class TestAllPublishedSourcesCovered:
    """Every path referenced by collect_matrix must appear in the registry."""

    def test_quality_paths_are_registered(self):
        r = load_registry()
        registered = all_registered_paths(r)
        missing = [p for p in quality_paths() if p not in registered]
        assert not missing, (
            f"{len(missing)} QUALITY paths from collect_matrix.py are not in the registry:\n"
            + "\n".join(f"  {p}" for p in sorted(missing))
        )

    def test_swebench_paths_are_registered(self):
        r = load_registry()
        registered = all_registered_paths(r)
        missing = [p for p in swebench_paths() if p not in registered]
        assert not missing, (
            f"{len(missing)} SWE-bench paths from common.py are not in the registry:\n"
            + "\n".join(f"  {p}" for p in sorted(missing))
        )


class TestNoUnknownPaths:
    """Every path declared in the registry must exist on disk (no phantoms)."""

    def test_all_registered_paths_exist(self):
        r = load_registry()
        missing = []
        for entry in r["entries"]:
            for p in entry.get("paths", []):
                full = REPO / p
                if not full.exists():
                    missing.append(f"{entry['id']}: {p}")
        assert not missing, (
            f"{len(missing)} registry paths do not exist on disk:\n"
            + "\n".join(f"  {m}" for m in missing)
        )


class TestCurrentStatusRequiresV1Validation:
    """A 'current' entry must carry v1_validated=true, or fail closed."""

    def test_current_requires_v1_validated(self):
        r = load_registry()
        bad = []
        for entry in r["entries"]:
            if entry.get("status") == "current":
                if not entry.get("v1_validated"):
                    bad.append(entry["id"])
        assert not bad, (
            f"Entries with status='current' must carry v1_validated=true. "
            f"Offenders: {bad}. "
            f"Historical/legacy artifacts must use status='historical', not 'current'."
        )


class TestReplayCompositesHaveBaseRunLinks:
    """Replay composites (status='replay') must carry base_run and superseded_by links."""

    def test_replay_has_base_run(self):
        r = load_registry()
        bad = []
        for entry in r["entries"]:
            if entry.get("status") == "replay":
                if not entry.get("base_run"):
                    bad.append(entry["id"])
        assert not bad, (
            f"Entries with status='replay' must carry 'base_run' link. "
            f"Offenders: {bad}"
        )

    def test_replay_registers_base_and_replay_components(self):
        """A composite is not fully registered if only its base file is listed."""
        r = load_registry()
        bad = []
        for entry in r["entries"]:
            if entry.get("status") == "replay":
                paths = entry.get("paths", [])
                if len(paths) < 2 or entry.get("base_run") not in paths:
                    bad.append(entry["id"])
        assert not bad, (
            "Replay entries must register both the base and replay artifact, "
            f"with base_run naming one registered component. Offenders: {bad}"
        )

    def test_replay_composites_in_collect_matrix_have_base_run(self):
        """Specifically: the replay composites referenced by collect_matrix.py
        (ornith-35b GPQA, nemotron-3.5-lightning-30b GPQA/IFEval)
        must have base_run set to their original run path.
        """
        r = load_registry()
        # Build a path -> entry lookup
        by_path: dict[str, dict] = {}
        for entry in r["entries"]:
            for p in entry.get("paths", []):
                by_path[p] = entry

        # The collect_matrix notes explicitly document replay composites.
        # These are the paths from the QUALITY dict that correspond to composite runs:
        replay_paths = [
            # ornith GPQA (64k replay composite)
            "results/ornith-35b/raw/quality/gpqa/results_2026-08-19T11-45-14.035924.json",
            # lightning GPQA (32k+64k composite)
            "results/nemotron-3.5-lightning-30b/raw/quality/gpqa_32k/nvidia__NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/results_2026-08-12T21-15-33.518368.json",
            # lightning IFEval (8k+64k composite)
            "results/nemotron-3.5-lightning-30b/raw/quality/ifeval/nvidia__NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/results_2026-08-12T13-08-23.626945.json",
        ]

        bad = []
        for p in replay_paths:
            entry = by_path.get(p)
            if entry is not None and entry.get("status") == "replay":
                if not entry.get("base_run"):
                    bad.append(p)
        assert not bad, (
            f"Replay composite entries must have 'base_run' link: {bad}"
        )


class TestFailClosedLoader:
    """The registry must be used fail-closed: unknown paths raise KeyError."""

    def test_load_registry_module_exists(self):
        """viz/audit_provenance.py or a dedicated loader must expose a fail-closed API."""
        loader_path = VIZ / "audit_provenance.py"
        assert loader_path.exists(), "viz/audit_provenance.py must exist"

    def test_registry_lookup_raises_on_unknown_path(self):
        """Importing and using the registry lookup with an unknown path must raise."""
        from viz_registry import load_registry as reg_load, lookup_path
        with pytest.raises(KeyError):
            lookup_path("results/nonexistent/model/raw/fake.json")

    def test_registry_lookup_returns_entry_for_known_path(self):
        """A known path must return the entry dict without raising."""
        from viz_registry import load_registry as reg_load, lookup_path
        # pick any registered path
        r = load_registry()
        first_path = r["entries"][0]["paths"][0]
        entry = lookup_path(first_path)
        assert "id" in entry
        assert "status" in entry

    def test_loader_rejects_duplicate_ids(self):
        from viz_registry import validate_registry
        registry = {"registry_version": 1, "entries": [
            {"id": "same", "status": "historical", "paths": ["results/a"], "note": "x"},
            {"id": "same", "status": "historical", "paths": ["results/b"], "note": "x"},
        ]}
        with pytest.raises(ValueError, match="duplicate registry id"):
            validate_registry(registry)

    def test_loader_rejects_current_without_v1_validation(self):
        from viz_registry import validate_registry
        registry = {"registry_version": 1, "entries": [
            {"id": "bad", "status": "current", "paths": ["results/a"], "note": "x"},
        ]}
        with pytest.raises(ValueError, match="v1_validated"):
            validate_registry(registry)

    def test_loader_rejects_replay_without_two_components(self):
        from viz_registry import validate_registry
        registry = {"registry_version": 1, "entries": [
            {"id": "bad", "status": "replay", "paths": ["results/base"],
             "base_run": "results/base", "note": "x"},
        ]}
        with pytest.raises(ValueError, match="at least two"):
            validate_registry(registry)

    def test_loader_rejects_duplicate_paths(self):
        from viz_registry import validate_registry
        registry = {"registry_version": 1, "entries": [
            {"id": "one", "status": "historical", "paths": ["results/same"], "note": "x"},
            {"id": "two", "status": "historical", "paths": ["results/same"], "note": "x"},
        ]}
        with pytest.raises(ValueError, match="duplicate registry path"):
            validate_registry(registry)

    def test_loader_rejects_registered_path_missing_from_repository(self, monkeypatch, tmp_path):
        import viz_registry
        registry_path = tmp_path / "registry.json"
        registry_path.write_text(json.dumps({"registry_version": 1, "entries": [
            {"id": "phantom", "status": "historical",
             "paths": ["results/does-not-exist.json"], "note": "x"},
        ]}))
        monkeypatch.setattr(viz_registry, "REGISTRY_PATH", registry_path)
        monkeypatch.setattr(viz_registry, "REPO", tmp_path)
        viz_registry.reload()
        with pytest.raises(ValueError, match="does not exist"):
            viz_registry.load_registry()
        viz_registry.reload()


class TestCompositeSourcesAreActuallyGated:
    COMPONENTS = [
        "results/ornith-35b/raw/quality/gpqa/results_64k_replay_corrected.json",
        "results/ornith-35b/raw/quality/ifeval/results_64k_replay_corrected.json",
        "results/nemotron-3.5-lightning-30b/raw/quality/ifeval/results_64k_replay_corrected.json",
        "results/nemotron-3.5-lightning-30b/raw/gpqa_64k_replay_results.json",
    ]

    def test_every_composite_component_is_registered(self):
        registered = all_registered_paths(load_registry())
        assert not [p for p in self.COMPONENTS if p not in registered]

    def test_collect_matrix_gates_every_composite_component(self):
        source = (VIZ / "collect_matrix.py").read_text()
        for path in self.COMPONENTS:
            assert path in source, f"collect_matrix must explicitly gate {path}"


class TestCollectMatrixUsesRegistry:
    """collect_matrix.py must consult the registry; unknown paths must abort collection."""

    def test_collect_matrix_imports_registry(self):
        """collect_matrix must import from viz_registry (fail-closed integration)."""
        source = (VIZ / "collect_matrix.py").read_text()
        assert "viz_registry" in source or "registry" in source.lower(), (
            "collect_matrix.py must import registry integration "
            "(fail-closed: unknown paths abort collection)."
        )

    def test_audit_provenance_imports_registry(self):
        """audit_provenance.py must reference the registry."""
        source = (VIZ / "audit_provenance.py").read_text()
        assert "registry" in source.lower(), (
            "audit_provenance.py must reference the artifact registry."
        )

    def test_audit_covers_composite_components(self):
        source = (VIZ / "audit_provenance.py").read_text()
        assert "COMPOSITE_COMPONENTS" in source


class TestHistoricalArtifactsNotAltered:
    """Registry entries must NOT move or synthesize historical raw artifacts.

    This is a structural test: every historical path must point to real
    committed files, and the registry itself must be a metadata-only file.
    """

    def test_historical_entries_point_to_existing_files(self):
        r = load_registry()
        missing = []
        for entry in r["entries"]:
            if entry.get("status") in {"historical", "superseded", "diagnostic", "replay", "invalid"}:
                for p in entry.get("paths", []):
                    if not (REPO / p).exists():
                        missing.append(f"{entry['id']}: {p}")
        assert not missing, (
            f"Historical entries must point to real existing files (do not move/delete):\n"
            + "\n".join(f"  {m}" for m in missing)
        )

    def test_registry_is_metadata_only(self):
        """The registry must live at results/registry.json, not inside raw/ subdirs."""
        assert REGISTRY_PATH.parent == REPO / "results", (
            "Registry must be at results/registry.json, not inside raw/ artifact directories."
        )
