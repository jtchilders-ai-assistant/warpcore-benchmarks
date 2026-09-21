"""viz/derive_diagnostic.py — deterministic diagnostic summary for a failed campaign.

A campaign that ends in execution state ``failed`` leaves evidence but no score.
That evidence still has to be auditable: how many items reached a terminal state,
which frozen instances were never observed at all, what the terminal statuses
were, and the fact that nothing here may ever be published.

This tool derives that summary *from the retained evidence only* and fails closed
on any count, ID, or hash inconsistency. It never invents a score, a grading
result, an exit status, or a completion timestamp. The creation-time
``manifest.json`` is read as raw evidence of what the runner wrote — including
counters the aborted run never got to update — and is never repaired.

Usage
-----
    derive_diagnostic.py --run-dir <run> --emit      # write diagnostic_summary.json
    derive_diagnostic.py --run-dir <run> --verify    # re-derive and compare (exit 1 on drift)

Public API
----------
derive(run_dir, repo=None) -> dict
verify(run_dir, repo=None) -> dict
write_summary(run_dir, summary) -> pathlib.Path
DiagnosticEvidenceError

Evidence read
-------------
manifest.json, status.json, command.txt, TERMINATION.json,
raw/preds.json, raw/exit_statuses_*.yaml, raw/run.log, raw/minisweagent.log,
raw/trajectories.tar.gz, plus the suite's frozen instance set.

Fail-closed gates
-----------------
G1  Exactly one ``raw/exit_statuses_*.yaml`` exists and parses.
G2  No instance ID appears twice across exit-status buckets.
G3  preds.json keys equal the exit-status inventory, and each entry's embedded
    ``instance_id`` equals its key.
G4  The trajectory archive holds exactly one ``<id>/<id>.traj.json`` per
    inventory ID, and each trajectory's terminal ``exit`` message agrees with
    the exit-status inventory.
G5  The non-empty ``model_patch`` set equals the ``Submitted`` bucket.
G6  Every terminal ID is a member of the suite's frozen instance set.
G7  manifest ``item_inventory.expected`` equals the frozen instance count, and
    ``instance_ids_hash`` equals the canonical digest of the sorted frozen set.
G8  manifest ``adapter_hash`` and every ``suite_input_hashes`` entry match the
    run-owned input snapshot (``suite_input_snapshots/<basename>``) when that
    snapshot exists, or the current canonical repository file otherwise.  The
    snapshot is the authoritative provenance of the bytes the run actually used;
    a canonical file that evolved after the run was committed does not falsify
    historical evidence that carries its own snapshot.
G9  status records ``execution_state='failed'`` with a nonpublishable lifecycle.
G10 No DONE sentinel, no grading artifact, no per-item score table — a score
    must not be derivable from this run.
G11 TERMINATION.json attests an operator-initiated stop for this run_id and
    claims neither a score nor a grading result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
import tarfile
from typing import Optional

try:
    import yaml as _yaml  # type: ignore[import]
except ImportError:  # pragma: no cover - PyYAML is a hard dependency of the repo
    _yaml = None

_VIZ_DIR = pathlib.Path(__file__).resolve().parent
_REPO_DEFAULT = _VIZ_DIR.parent

SUMMARY_NAME = "diagnostic_summary.json"
TERMINATION_NAME = "TERMINATION.json"
ARCHIVE_REL = "raw/trajectories.tar.gz"
#: Subdirectory within the run that holds exact snapshots of suite input files
#: as they were at the moment the run was launched.  When a snapshot exists for
#: a ``suite_input_hashes`` entry, G8 verifies the declared hash against the
#: snapshot rather than the current canonical repository file, preserving
#: provenance for historical evidence across canonical-file evolution.
SNAPSHOTS_DIR = "suite_input_snapshots"
SCHEMA_VERSION = 1

#: Lifecycles a failed campaign may legally carry (design §8, §9).
_NONPUBLISHABLE_LIFECYCLES = frozenset({"invalid", "diagnostic"})

#: Artifacts whose presence would mean a score could be derived from this run.
_SCORE_BEARING_ARTIFACTS = (
    "raw/grading_results.json",
    "raw/exit_statuses.json",
    "per_item.csv",
    "DONE",
)

#: The serving defect signature carried by the retained raw API responses.
_TOOL_CALL_PARSE_SIGNATURE = "Error parsing tool call arguments"

#: Filesystem droppings that are never evidence. Everything else found in the run
#: directory is digested, so an unexpected file fails verification rather than
#: being silently ignored.
_NOT_EVIDENCE = frozenset({".DS_Store", SUMMARY_NAME})


class DiagnosticEvidenceError(Exception):
    """Raised when retained evidence is inconsistent, incomplete, or tampered with."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _validate_safe_rel(rel: str, label: str, errors: list) -> bool:
    """Validate that *rel* is a lexically safe, relative path.

    A safe relative path must not be absolute and must contain no ``..``
    components that could escape the intended root.  Returns True when safe,
    appends to *errors* and returns False otherwise.

    This check is intentionally strict: any ``..`` in *any* component of the
    path is rejected, even if the OS-resolved result would remain within the
    root.  Lexical safety is required before any filesystem operation.
    """
    if pathlib.PurePosixPath(rel).is_absolute() or pathlib.PureWindowsPath(rel).is_absolute():
        errors.append(
            f"unsafe path in {label}: {rel!r} is absolute; "
            "suite_input_hashes keys must be repo-relative paths."
        )
        return False
    parts = pathlib.PurePosixPath(rel).parts
    if ".." in parts or "." in parts:
        errors.append(
            f"unsafe path traversal in {label}: {rel!r} contains '..' or '.' components; "
            "path traversal is not permitted."
        )
        return False
    return True


def _validate_contained(resolved: pathlib.Path, root: pathlib.Path, label: str, errors: list) -> bool:
    """Confirm that *resolved* is strictly contained within *root*.

    Returns True when contained, appends to *errors* and returns False otherwise.
    """
    try:
        resolved.relative_to(root)
        return True
    except ValueError:
        errors.append(
            f"path containment violation in {label}: resolved path {resolved} "
            f"is not contained within {root}."
        )
        return False


def _read_json(path: pathlib.Path, label: str, errors: list) -> Optional[dict]:
    if not path.is_file():
        errors.append(f"{label} not found at {path}.")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{label} could not be read as JSON: {exc}")
        return None


def _preview(ids) -> str:
    ordered = sorted(ids)
    head = ordered[:5]
    more = len(ordered) - len(head)
    return f"{head}{f' ... and {more} more' if more else ''}"


# ---------------------------------------------------------------------------
# Evidence readers
# ---------------------------------------------------------------------------

def _read_exit_statuses(raw_dir: pathlib.Path, errors: list) -> dict[str, str]:
    """Return {instance_id: exit_status} from the single exit-status YAML (G1, G2)."""
    if _yaml is None:
        errors.append("PyYAML is not installed; exit_statuses YAML cannot be parsed.")
        return {}

    candidates = sorted(raw_dir.glob("exit_statuses_*.yaml"))
    if len(candidates) != 1:
        errors.append(
            f"expected exactly one raw/exit_statuses_*.yaml under {raw_dir}, "
            f"found {len(candidates)}: {[p.name for p in candidates]}. "
            "The terminal-status inventory must be unambiguous."
        )
        return {}

    try:
        doc = _yaml.safe_load(candidates[0].read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - any parse failure is fail-closed
        errors.append(f"raw/{candidates[0].name} could not be parsed: {exc}")
        return {}

    buckets = (doc or {}).get("instances_by_exit_status")
    if not isinstance(buckets, dict) or not buckets:
        errors.append(
            f"raw/{candidates[0].name} must contain a non-empty "
            "'instances_by_exit_status' mapping."
        )
        return {}

    inventory: dict[str, str] = {}
    duplicates: list[str] = []
    for status, ids in sorted(buckets.items()):
        if not isinstance(ids, list) or not ids:
            errors.append(
                f"exit status bucket {status!r} must be a non-empty list of instance IDs."
            )
            continue
        for iid in ids:
            if not isinstance(iid, str) or not iid.strip():
                errors.append(f"exit status bucket {status!r} contains a non-string ID: {iid!r}")
                continue
            if iid in inventory:
                duplicates.append(iid)
            inventory[iid] = str(status)
    if duplicates:
        errors.append(
            f"duplicate instance ID(s) across exit status buckets: {_preview(set(duplicates))}. "
            "An instance has exactly one terminal status."
        )
    return inventory


def _read_preds(raw_dir: pathlib.Path, errors: list) -> dict[str, str]:
    """Return {instance_id: model_patch} from preds.json (G3)."""
    preds = _read_json(raw_dir / "preds.json", "raw/preds.json", errors)
    if preds is None:
        return {}
    if not isinstance(preds, dict):
        errors.append(
            f"raw/preds.json must be a JSON object keyed by instance_id; "
            f"got {type(preds).__name__}."
        )
        return {}

    patches: dict[str, str] = {}
    for iid, entry in sorted(preds.items()):
        if not isinstance(entry, dict):
            errors.append(f"raw/preds.json entry {iid!r} is not an object.")
            continue
        embedded = entry.get("instance_id")
        if embedded != iid:
            errors.append(
                f"raw/preds.json key {iid!r} disagrees with its embedded "
                f"instance_id {embedded!r}."
            )
        patches[iid] = entry.get("model_patch") or ""
    return patches


def _read_trajectory_archive(
    archive: pathlib.Path, errors: list
) -> tuple[dict[str, dict], int]:
    """Stream the archive once.

    Returns ({instance_id: {...per-trajectory facts...}}, total uncompressed bytes).
    """
    if not archive.is_file():
        errors.append(f"trajectory archive not found at {archive}.")
        return {}, 0

    facts: dict[str, dict] = {}
    total_bytes = 0
    try:
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                if member.isdir():
                    continue
                if not member.isfile():
                    errors.append(
                        f"trajectory archive member {member.name!r} is not a regular file."
                    )
                    continue
                iid, _, leaf = member.name.partition("/")
                if not iid or leaf != f"{iid}.traj.json":
                    errors.append(
                        f"trajectory archive member {member.name!r} does not follow "
                        "'<instance_id>/<instance_id>.traj.json'."
                    )
                    continue
                if iid in facts:
                    errors.append(f"duplicate trajectory for instance {iid!r} in the archive.")
                    continue
                handle = tar.extractfile(member)
                if handle is None:  # pragma: no cover - defensive
                    errors.append(f"trajectory archive member {member.name!r} is unreadable.")
                    continue
                blob = handle.read()
                total_bytes += len(blob)
                facts[iid] = _trajectory_facts(iid, blob, errors)
    except (OSError, tarfile.TarError) as exc:
        errors.append(f"trajectory archive could not be read: {exc}")
        return {}, 0

    return facts, total_bytes


def _trajectory_facts(iid: str, blob: bytes, errors: list) -> dict:
    """Extract the auditable facts from one trajectory document.

    A mini-swe-agent trajectory records its terminal status twice — in
    ``info.exit_status`` and in the terminal ``exit`` message's
    ``extra.exit_status``. Both are read and required to agree, so a single
    edited field cannot change the derived status. The ``exit`` message's
    ``content`` is NOT the status: for a ``Submitted`` run it holds the patch.
    """
    fact = {
        "sha256": _sha256(blob),
        "bytes": len(blob),
        "exit_status": None,
        "submission": None,
        "api_calls": 0,
        "retained_response_objects": 0,
        "tool_call_parse_errors": 0,
    }
    try:
        doc = json.loads(blob)
    except json.JSONDecodeError as exc:
        errors.append(f"trajectory for {iid!r} is not valid JSON: {exc}")
        return fact

    if not isinstance(doc, dict):
        errors.append(f"trajectory for {iid!r} must be a JSON object.")
        return fact
    if doc.get("instance_id") != iid:
        errors.append(
            f"trajectory for {iid!r} declares instance_id {doc.get('instance_id')!r}."
        )

    messages = doc.get("messages")
    if not isinstance(messages, list) or not messages:
        errors.append(f"trajectory for {iid!r} has no messages.")
        return fact

    info = doc.get("info") if isinstance(doc.get("info"), dict) else {}
    terminal = messages[-1] if isinstance(messages[-1], dict) else {}
    if terminal.get("role") != "exit":
        errors.append(
            f"trajectory for {iid!r} does not end in a terminal 'exit' message; "
            "its terminal exit status cannot be established from the evidence."
        )
        return fact

    terminal_extra = terminal.get("extra") if isinstance(terminal.get("extra"), dict) else {}
    info_status = info.get("exit_status")
    message_status = terminal_extra.get("exit_status")
    if not isinstance(info_status, str) or not info_status:
        errors.append(f"trajectory for {iid!r} records no info.exit_status.")
        return fact
    if info_status != message_status:
        errors.append(
            f"trajectory for {iid!r} records info.exit_status {info_status!r} but its "
            f"terminal message records exit status {message_status!r}."
        )
        return fact
    fact["exit_status"] = info_status

    info_submission = info.get("submission")
    if info_submission != terminal_extra.get("submission"):
        errors.append(
            f"trajectory for {iid!r} records a different submission in info than in "
            "its terminal message."
        )
    fact["submission"] = info_submission if isinstance(info_submission, str) else ""

    stats = info.get("model_stats")
    if isinstance(stats, dict) and isinstance(stats.get("api_calls"), int):
        fact["api_calls"] = stats["api_calls"]

    for message in messages:
        if not isinstance(message, dict):
            continue
        extra = message.get("extra")
        if isinstance(extra, dict) and isinstance(extra.get("response"), dict):
            fact["retained_response_objects"] += 1
        content = message.get("content")
        if isinstance(content, str):
            fact["tool_call_parse_errors"] += content.count(_TOOL_CALL_PARSE_SIGNATURE)
    return fact


def _load_frozen_ids(
    repo: pathlib.Path, benchmark: str, errors: list
) -> tuple[Optional[list[str]], Optional[str]]:
    """Load the suite's frozen instance set. Returns (sorted IDs, instance_set_file)."""
    if _yaml is None:
        errors.append("PyYAML is not installed; the suite frozen instance set cannot be read.")
        return None, None

    suite_path = repo / "suite" / "warpcore-v1.yaml"
    if not suite_path.is_file():
        errors.append(f"suite YAML not found at {suite_path}.")
        return None, None
    try:
        suite = _yaml.safe_load(suite_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        errors.append(f"suite YAML could not be parsed: {exc}")
        return None, None

    cfg = (suite.get("benchmarks") or {}).get(benchmark)
    if not isinstance(cfg, dict):
        errors.append(f"suite YAML declares no '{benchmark}' benchmark.")
        return None, None

    rel = cfg.get("instance_set_file")
    if not rel:
        errors.append(f"suite '{benchmark}' config declares no instance_set_file.")
        return None, None

    instances_path = repo / rel
    if not instances_path.is_file():
        errors.append(f"suite instance_set_file not found at {instances_path}.")
        return None, rel

    blob = instances_path.read_bytes()
    declared = cfg.get("instances_sha256")
    actual = _sha256(blob)
    if declared and declared != actual:
        errors.append(
            f"suite instances_sha256 hash mismatch for {rel}: "
            f"suite declares {declared[:16]}..., file is {actual[:16]}..."
        )
        return None, rel

    try:
        ids = json.loads(blob)
    except json.JSONDecodeError as exc:
        errors.append(f"suite instance_set_file {rel} is not valid JSON: {exc}")
        return None, rel
    if not isinstance(ids, list) or not all(isinstance(i, str) and i for i in ids):
        errors.append(f"suite instance_set_file {rel} must be a JSON array of instance IDs.")
        return None, rel
    if len(set(ids)) != len(ids):
        errors.append(f"suite instance_set_file {rel} contains duplicate instance IDs.")
        return None, rel

    declared_count = cfg.get("expected_item_count")
    if declared_count is not None and declared_count != len(ids):
        errors.append(
            f"suite expected_item_count={declared_count} does not match the "
            f"{len(ids)} IDs in {rel}."
        )
        return None, rel
    return sorted(ids), rel


def _read_termination(run_dir: pathlib.Path, run_id: str, errors: list) -> dict:
    """Read and gate the operator termination attestation (G11)."""
    path = run_dir / TERMINATION_NAME
    if not path.is_file():
        errors.append(
            f"{TERMINATION_NAME} not found at {path}. A failed campaign must record how it "
            "ended; an unattested stop cannot be distinguished from an unexplained crash."
        )
        return {}
    doc = _read_json(path, TERMINATION_NAME, errors)
    if not isinstance(doc, dict):
        return {}

    if doc.get("run_id") != run_id:
        errors.append(
            f"{TERMINATION_NAME} attests run_id {doc.get('run_id')!r} but the campaign "
            f"run_id is {run_id!r}."
        )
    mode = doc.get("mode")
    if mode != "operator_initiated":
        errors.append(
            f"{TERMINATION_NAME} declares termination mode {mode!r}; this tool only "
            "summarises an 'operator_initiated' stop. Any other mode must be evidenced "
            "and handled explicitly rather than assumed."
        )
    if doc.get("score_claimed") is not False:
        errors.append(
            f"{TERMINATION_NAME} must record score_claimed=false — a failed campaign "
            "yields no score."
        )
    if doc.get("grading_performed") is not False:
        errors.append(
            f"{TERMINATION_NAME} must record grading_performed=false — this campaign "
            "was never graded."
        )
    for field in ("attested_by", "attested_utc", "reason"):
        if not doc.get(field):
            errors.append(f"{TERMINATION_NAME} is missing required field {field!r}.")
    evidence = doc.get("corroborating_evidence")
    if not isinstance(evidence, list) or not evidence:
        errors.append(
            f"{TERMINATION_NAME} must list corroborating_evidence drawn from the "
            "retained artifacts."
        )
    return doc


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------

def derive(run_dir, repo=None) -> dict:
    """Derive the diagnostic summary for *run_dir*, or raise DiagnosticEvidenceError."""
    run_dir = pathlib.Path(run_dir).resolve()
    repo = pathlib.Path(repo).resolve() if repo is not None else _REPO_DEFAULT
    raw_dir = run_dir / "raw"
    errors: list[str] = []

    if not run_dir.is_dir():
        raise DiagnosticEvidenceError(f"run directory not found: {run_dir}")

    manifest = _read_json(run_dir / "manifest.json", "manifest.json", errors) or {}
    status = _read_json(run_dir / "status.json", "status.json", errors) or {}
    if errors:
        raise DiagnosticEvidenceError(_format(run_dir, errors))

    run_id = manifest.get("run_id", "")
    benchmark = manifest.get("benchmark", "")
    model_slug = (manifest.get("model") or {}).get("slug", "")
    item_inv = manifest.get("item_inventory") or {}

    # --- Identity agreement -------------------------------------------------
    if run_id != run_dir.name:
        errors.append(
            f"manifest run_id {run_id!r} does not match run directory name {run_dir.name!r}."
        )
    if status.get("run_id") != run_id:
        errors.append(
            f"status.json run_id {status.get('run_id')!r} does not match manifest "
            f"run_id {run_id!r}."
        )

    termination = _read_termination(run_dir, run_id, errors)

    # --- G9: the run must actually be a failed, nonpublishable campaign -----
    execution_state = status.get("execution_state")
    lifecycle = status.get("lifecycle")
    if execution_state != "failed":
        errors.append(
            f"status.json execution_state is {execution_state!r}; this tool summarises "
            "only campaigns whose execution_state is 'failed'."
        )
    history = status.get("history")
    if not isinstance(history, list) or not history or \
            history[-1].get("state") != execution_state:
        errors.append(
            "status.json history must be a non-empty list whose tail state matches "
            "execution_state; the failure history must not be rewritten."
        )
    if lifecycle not in _NONPUBLISHABLE_LIFECYCLES:
        errors.append(
            f"status.json lifecycle is {lifecycle!r}; a failed campaign must carry a "
            f"nonpublishable lifecycle ({sorted(_NONPUBLISHABLE_LIFECYCLES)})."
        )

    # --- G10: nothing here may yield a score --------------------------------
    score_bearing = [rel for rel in _SCORE_BEARING_ARTIFACTS if (run_dir / rel).exists()]
    if score_bearing:
        errors.append(
            f"score-bearing artifact(s) present in a failed campaign: {score_bearing}. "
            "A DONE sentinel, grading result, or per-item score table means the run is "
            "not the ungraded diagnostic this summary describes."
        )

    # --- Terminal inventory (G1–G5) -----------------------------------------
    exit_statuses = _read_exit_statuses(raw_dir, errors)
    patches = _read_preds(raw_dir, errors)
    pred_ids = set(patches)
    patched_ids = {iid for iid, patch in patches.items() if patch.strip()}
    traj_facts, traj_bytes = _read_trajectory_archive(run_dir / ARCHIVE_REL, errors)

    inventory_ids = set(exit_statuses)
    if pred_ids != inventory_ids:
        errors.append(
            f"raw/preds.json instance IDs do not match the exit status inventory: "
            f"only in preds {_preview(pred_ids - inventory_ids)}; "
            f"only in exit statuses {_preview(inventory_ids - pred_ids)}."
        )
    traj_ids = set(traj_facts)
    if traj_ids != inventory_ids:
        errors.append(
            f"trajectory archive does not match the exit status inventory: "
            f"only in trajectories {_preview(traj_ids - inventory_ids)}; "
            f"missing trajectories {_preview(inventory_ids - traj_ids)}."
        )
    for iid in sorted(traj_ids & inventory_ids):
        recorded = traj_facts[iid]["exit_status"]
        if recorded != exit_statuses[iid]:
            errors.append(
                f"trajectory for {iid!r} ends in exit status {recorded!r} but the "
                f"exit status inventory records {exit_statuses[iid]!r}."
            )
    for iid in sorted(traj_ids & pred_ids):
        submission = traj_facts[iid]["submission"]
        if submission is not None and submission != patches[iid]:
            errors.append(
                f"the model_patch recorded in preds.json for {iid!r} differs from the "
                "submission retained in its trajectory."
            )

    submitted_ids = {iid for iid, st in exit_statuses.items() if st == "Submitted"}
    if patched_ids != submitted_ids:
        errors.append(
            f"the non-empty model_patch set does not match the 'Submitted' exit status "
            f"bucket: patch without Submitted {_preview(patched_ids - submitted_ids)}; "
            f"Submitted without patch {_preview(submitted_ids - patched_ids)}."
        )

    # --- Frozen instance set (G6, G7) ---------------------------------------
    frozen, instance_set_file = _load_frozen_ids(repo, benchmark, errors)
    unobserved: list[str] = []
    if frozen is not None:
        frozen_set = set(frozen)
        foreign = inventory_ids - frozen_set
        if foreign:
            errors.append(
                f"{len(foreign)} terminal instance ID(s) are not in the suite frozen set: "
                f"{_preview(foreign)}."
            )
        unobserved = sorted(frozen_set - inventory_ids)

        declared_expected = item_inv.get("expected")
        if declared_expected != len(frozen):
            errors.append(
                f"manifest item_inventory.expected={declared_expected!r} does not equal "
                f"the suite frozen instance count {len(frozen)}."
            )
        declared_hash = item_inv.get("instance_ids_hash", "")
        canonical_hash = _sha256(json.dumps(frozen, sort_keys=True).encode())
        if declared_hash != canonical_hash:
            errors.append(
                f"manifest item_inventory.instance_ids_hash {declared_hash[:16]}... does "
                f"not equal the canonical digest of the sorted frozen set "
                f"{canonical_hash[:16]}..."
            )

    # --- G8: declared input hashes must match the run-owned snapshot or repo file ----
    suite_input_hashes = manifest.get("suite_input_hashes") or {}
    snapshots_dir = run_dir / SNAPSHOTS_DIR
    snapshot_mode = snapshots_dir.is_dir()

    for rel, declared in sorted(suite_input_hashes.items()):
        # Validate lexical safety before any filesystem operation.
        if not _validate_safe_rel(rel, "suite_input_hashes", errors):
            continue
        snapshot = snapshots_dir / rel
        resolved_snapshot = snapshot.resolve()
        resolved_snapshots_root = snapshots_dir.resolve()
        if snapshot_mode:
            # In snapshot mode ALL entries must have a snapshot; no silent fallback.
            if not snapshot.is_file():
                errors.append(
                    f"{SNAPSHOTS_DIR}/{rel} snapshot missing: once suite_input_snapshots/ "
                    "exists every suite_input_hashes entry must have a run-owned snapshot "
                    "so that historical provenance is complete."
                )
                continue
            # Verify resolved path is still contained within snapshots root.
            if not _validate_contained(resolved_snapshot, resolved_snapshots_root,
                                       f"suite_input_hashes key {rel!r}", errors):
                continue
            actual = _sha256(snapshot.read_bytes())
            if actual != declared:
                errors.append(
                    f"{SNAPSHOTS_DIR}/{rel} hash mismatch: "
                    f"manifest records {str(declared)[:16]}..., "
                    f"snapshot is {actual[:16]}..."
                )
        else:
            # No snapshot present — fall back to current canonical file.
            target = repo / rel
            if not target.is_file():
                errors.append(f"manifest suite_input_hashes names a missing file: {rel}.")
                continue
            actual = _sha256(target.read_bytes())
            if actual != declared:
                errors.append(
                    f"manifest suite_input_hashes hash mismatch for {rel}: manifest records "
                    f"{str(declared)[:16]}..., file is {actual[:16]}..."
                )

    # --- G8 adapter: adapter snapshot required when snapshot mode is active ----------
    adapter_snapshot_rel: Optional[str] = None
    declared_adapter = manifest.get("adapter_hash", "")
    # Always validate slug path safety before building any filesystem path.
    slug_safe = True
    if model_slug:
        slug_parts = pathlib.PurePosixPath(model_slug).parts
        if ".." in slug_parts or "." in slug_parts or pathlib.PurePosixPath(model_slug).is_absolute():
            errors.append(
                f"unsafe adapter path: model_slug {model_slug!r} contains '..'  or '.' "
                "components or is absolute; adapter snapshot path traversal is not permitted."
            )
            slug_safe = False
    if snapshot_mode and slug_safe:
        adapter_snap_rel = f"adapters/{model_slug}.yaml"
        adapter_snap = snapshots_dir / adapter_snap_rel
        resolved_adapter_snap = adapter_snap.resolve()
        resolved_snapshots_root = snapshots_dir.resolve()
        if not _validate_contained(resolved_adapter_snap, resolved_snapshots_root,
                                   f"adapter snapshot for slug {model_slug!r}", errors):
            slug_safe = False
        elif not adapter_snap.is_file():
            errors.append(
                f"{SNAPSHOTS_DIR}/adapters/{model_slug}.yaml adapter snapshot missing: "
                "once suite_input_snapshots/ exists the adapter snapshot must also be "
                "present to preserve full provenance for this historical run."
            )
        else:
            actual_adapter_snap = _sha256(adapter_snap.read_bytes())
            if actual_adapter_snap != declared_adapter:
                errors.append(
                    f"{SNAPSHOTS_DIR}/adapters/{model_slug}.yaml adapter snapshot hash "
                    f"mismatch: manifest adapter_hash records {str(declared_adapter)[:16]}..., "
                    f"snapshot is {actual_adapter_snap[:16]}..."
                )
            else:
                # Snapshot verified: no need to re-check the canonical adapter file.
                adapter_snapshot_rel = f"{SNAPSHOTS_DIR}/{adapter_snap_rel}"
    if slug_safe and not snapshot_mode:
        # Non-snapshot mode: verify against the current canonical adapter file.
        adapter_path = repo / "adapters" / f"{model_slug}.yaml"
        if not adapter_path.is_file():
            errors.append(f"adapter file not found at {adapter_path}; adapter_hash is unverifiable.")
        else:
            actual_adapter = _sha256(adapter_path.read_bytes())
            if actual_adapter != declared_adapter:
                errors.append(
                    f"manifest adapter_hash mismatch for adapters/{model_slug}.yaml: manifest "
                    f"records {str(declared_adapter)[:16]}..., file is {actual_adapter[:16]}..."
                )
    if errors:
        raise DiagnosticEvidenceError(_format(run_dir, errors))

    # --- Summary ------------------------------------------------------------
    status_counts: dict[str, int] = {}
    for st in exit_statuses.values():
        status_counts[st] = status_counts.get(st, 0) + 1

    effective_args = ((manifest.get("serving") or {}).get("effective_args")) or []
    artifacts = {
        rel: {"bytes": (run_dir / rel).stat().st_size,
              "sha256": _sha256((run_dir / rel).read_bytes())}
        for rel in sorted(
            str(p.relative_to(run_dir)) for p in run_dir.rglob("*")
            if p.is_file() and p.name not in _NOT_EVIDENCE
        )
    }

    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": "failed_campaign_diagnostic",
        "run_id": run_id,
        "model_slug": model_slug,
        "suite_id": manifest.get("suite_id", ""),
        "benchmark": benchmark,
        "execution_state": execution_state,
        "lifecycle": lifecycle,
        "publishable": False,
        "score": None,
        "nonpublishable_reasons": [
            f"execution_state is {execution_state!r}; the campaign never completed.",
            f"artifact lifecycle is {lifecycle!r}, which the v1 validator refuses to certify.",
            "no grading was performed, so no item has a verdict and no score exists.",
            f"only {len(inventory_ids)} of {len(frozen or [])} frozen instances reached a "
            "terminal state; the declared denominator was never attempted.",
            "the creation-time manifest is unreconciled evidence (submitted=0, all artifact "
            "flags false, completed_utc null) and must not be repaired into a result.",
        ],
        "grading": {
            "present": False,
            "graded_instances": 0,
            "grading_artifacts": [],
            "note": "No grading harness was run; resolved/unresolved verdicts do not exist.",
        },
        "termination": {
            "mode": termination.get("mode"),
            "attested_by": termination.get("attested_by"),
            "attested_utc": termination.get("attested_utc"),
            "reason": termination.get("reason"),
            "corroborating_evidence": list(termination.get("corroborating_evidence") or []),
        },
        "item_inventory": {
            "expected": len(frozen or []),
            "terminal_observed": len(inventory_ids),
            "unobserved": len(unobserved),
            "instance_set_file": instance_set_file,
            "instance_ids_hash": item_inv.get("instance_ids_hash", ""),
            "terminal_instance_ids": sorted(inventory_ids),
            "unobserved_instance_ids": unobserved,
        },
        "terminal_exit_statuses": dict(sorted(status_counts.items())),
        "submitted_nonempty_patches": len(patched_ids),
        "trajectories": {
            "archive": ARCHIVE_REL,
            "count": len(traj_facts),
            "member_template": "<instance_id>/<instance_id>.traj.json",
            "uncompressed_bytes": traj_bytes,
            "archive_bytes": (run_dir / ARCHIVE_REL).stat().st_size,
        },
        "trajectory_digests": {
            iid: traj_facts[iid]["sha256"] for iid in sorted(traj_facts)
        },
        "defect_evidence": {
            "signature": _TOOL_CALL_PARSE_SIGNATURE,
            "trajectories_with_signature": sum(
                1 for f in traj_facts.values() if f["tool_call_parse_errors"]
            ),
            "signature_occurrences": sum(
                f["tool_call_parse_errors"] for f in traj_facts.values()
            ),
            "retained_response_objects": sum(
                f["retained_response_objects"] for f in traj_facts.values()
            ),
            "api_calls": sum(f["api_calls"] for f in traj_facts.values()),
            "tool_call_parser": _arg_value(effective_args, "tool-call-parser"),
            "reasoning_parser": _arg_value(effective_args, "reasoning-parser"),
        },
        "adapter_hash": declared_adapter,
        "adapter_snapshot": adapter_snapshot_rel,
        "suite_input_hashes": dict(sorted(suite_input_hashes.items())),
        "artifacts": artifacts,
    }
    return summary


def _arg_value(effective_args: list, name: str) -> Optional[str]:
    """Return the value of ``--<name>=<value>`` from effective_args, else None."""
    pattern = re.compile(rf"^--{re.escape(name)}=(.+)$")
    for arg in effective_args:
        if isinstance(arg, str):
            match = pattern.match(arg)
            if match:
                return match.group(1)
    return None


def _format(run_dir: pathlib.Path, errors: list) -> str:
    joined = "\n".join(f"  - {e}" for e in errors)
    return (
        f"retained evidence for {run_dir} is inconsistent "
        f"({len(errors)} finding(s)):\n{joined}"
    )


# ---------------------------------------------------------------------------
# Emit / verify
# ---------------------------------------------------------------------------

def write_summary(run_dir, summary: dict) -> pathlib.Path:
    """Write *summary* to ``<run_dir>/diagnostic_summary.json``."""
    path = pathlib.Path(run_dir) / SUMMARY_NAME
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path


def verify(run_dir, repo=None) -> dict:
    """Re-derive the summary and compare it with the committed one (fail-closed)."""
    run_dir = pathlib.Path(run_dir).resolve()
    derived = derive(run_dir, repo=repo)
    path = run_dir / SUMMARY_NAME
    if not path.is_file():
        raise DiagnosticEvidenceError(
            f"{SUMMARY_NAME} not found at {path}. Emit it with --emit before verifying."
        )
    try:
        committed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticEvidenceError(f"{SUMMARY_NAME} could not be read as JSON: {exc}") from exc

    if committed != derived:
        raise DiagnosticEvidenceError(
            f"committed {SUMMARY_NAME} does not match the summary derived from the "
            f"retained evidence:\n{_diff(committed, derived)}"
        )
    return derived


def _diff(committed: dict, derived: dict) -> str:
    keys = sorted(set(committed) | set(derived))
    lines = []
    for key in keys:
        left, right = committed.get(key, "<absent>"), derived.get(key, "<absent>")
        if left != right:
            lines.append(f"  - {key}: committed={_short(left)} derived={_short(right)}")
    return "\n".join(lines) or "  - documents differ but no top-level key differs"


def _short(value) -> str:
    text = json.dumps(value, sort_keys=True, default=str)
    return text if len(text) <= 160 else text[:157] + "..."


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True, type=pathlib.Path,
                        help="campaign run directory to summarise")
    parser.add_argument("--repo", type=pathlib.Path, default=None,
                        help="repository root (default: the parent of viz/)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit", action="store_true", help="write diagnostic_summary.json")
    mode.add_argument("--verify", action="store_true",
                      help="re-derive and compare with the committed summary")
    args = parser.parse_args(argv)

    try:
        if args.emit:
            summary = derive(args.run_dir, repo=args.repo)
            path = write_summary(args.run_dir, summary)
            print(f"Wrote {path}")
        else:
            summary = verify(args.run_dir, repo=args.repo)
            print(f"OK: {args.run_dir}/{SUMMARY_NAME} matches the retained evidence.")
    except DiagnosticEvidenceError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    inv = summary["item_inventory"]
    print(
        f"  {summary['run_id']}: execution_state={summary['execution_state']} "
        f"lifecycle={summary['lifecycle']} publishable={summary['publishable']} "
        f"score={summary['score']}"
    )
    print(
        f"  {inv['terminal_observed']}/{inv['expected']} instances reached a terminal "
        f"state; {inv['unobserved']} never observed; "
        f"terminal statuses {summary['terminal_exit_statuses']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
