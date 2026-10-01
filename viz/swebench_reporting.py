#!/usr/bin/env python3
"""Shared SWE-bench reporting parser for warpcore-benchmarks.

Derives submitted/resolved/decomposition from committed prediction and grading
artifacts without hand-entering counts. This is the single authoritative source
for every model's selected SWE-bench report; all generated data and figures must
consume it.

Definitions (per the approved design spec):
  - submitted: count of instances with a nonempty model_patch in predictions
    (when predictions are retained); for legacy aggregate-only reports, derived
    from resolved + unresolved where predictions are absent.
  - resolved: len(resolved_ids) from official grading output
  - grading_error: len(error_ids) — nonempty patches that reached the grader
    but failed before a terminal verdict; counted as submitted but NOT resolved
  - empty_patch: len(empty_patch_ids) — agent never produced a diff
  - unresolved: len(unresolved_ids) — gradeable patch with test failure
  - incomplete: len(incomplete_ids)
  - provenance: "normalized-complete" when predictions file is present and
    IDs agree with grading; "legacy-aggregate" when only the aggregate JSON
    is available and predictions are absent.

Two public functions:
  load_swebench_report(repo, bundle) -> dict   — parse a single bundle
  selected_swebench_reports(repo) -> dict[str, dict]  — return the chosen
      run per model, keyed by results/<dir> model slug
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from viz_registry import lookup_path


# ---------------------------------------------------------------------------
# Source bundle registry
#
# Each entry names the official grading evidence and optional prediction
# evidence. Do NOT encode result counts here — they must be derived.
#
# Bundle keys:
#   grading     Path  — official grading_results.json (normalized) or the
#                        legacy swebench_verified_n100_results.json
#   predictions Path  — optional per-prediction preds.json
#   lifecycle   str   — "current" | "historical" | "diagnostic"
#   is_legacy   bool  — True for pre-contract aggregate-format reports
# ---------------------------------------------------------------------------

def _source_bundles(repo: Path) -> dict[str, dict[str, Any]]:
    """Return the selected source bundle for each model."""
    r = repo
    return {
        "ornith-35b": {
            "grading": r / "results/ornith-35b/raw/swebench/swebench_verified_n100_results.json",
            "predictions": r / "results/ornith-35b/raw/swebench/swebench_verified_n100_preds.json",
            "lifecycle": "historical",
            "is_legacy": True,
        },
        "ornith-1.5-35b-a3b": {
            # recovered and publication-validated warpcore-v2 run
            "grading": r / "results/ornith-1.5-35b-a3b/runs/warpcore-v2/swebench/ornith15-swebench-n100-20260929/raw/grading_results.json",
            "predictions": r / "results/ornith-1.5-35b-a3b/runs/warpcore-v2/swebench/ornith15-swebench-n100-20260929/raw/preds.json",
            "lifecycle": "current",
            "is_legacy": False,
        },
        "laguna-s-2.1-118b": {
            "grading": r / "results/laguna-s-2.1-118b/raw/swebench/swebench_verified_n100_results.json",
            "predictions": r / "results/laguna-s-2.1-118b/raw/swebench/swebench_verified_n100_preds.json",
            "lifecycle": "historical",
            "is_legacy": True,
        },
        "nemotron-3.5-lightning-30b": {
            "grading": r / "results/nemotron-3.5-lightning-30b/raw/swebench_verified_n100_results.json",
            "predictions": r / "results/nemotron-3.5-lightning-30b/raw/swebench_verified_n100_preds.json",
            "lifecycle": "historical",
            "is_legacy": True,
        },
        "qwen3.6-35b-a3b": {
            # validated warpcore-v1 run qwen36-swebench-n100-20260919
            "grading": r / "results/qwen3.6-35b-a3b/runs/warpcore-v1/swebench/qwen36-swebench-n100-20260919/raw/grading_results.json",
            "predictions": r / "results/qwen3.6-35b-a3b/runs/warpcore-v1/swebench/qwen36-swebench-n100-20260919/raw/preds.json",
            "lifecycle": "current",
            "is_legacy": False,
        },
        "qwen3.5-122b-a10b": {
            # noncanonical diagnostic trial
            "grading": r / "results/qwen3.5-122b-a10b/runs/warpcore-v1/swebench/qwen35-swebench-trial-n100-20260928/raw/grading_results.json",
            "predictions": r / "results/qwen3.5-122b-a10b/runs/warpcore-v1/swebench/qwen35-swebench-trial-n100-20260928/raw/preds.json",
            "lifecycle": "diagnostic",
            "is_legacy": False,
        },
    }


# ---------------------------------------------------------------------------
# Legacy-format grading report (schema_version=2, total_instances=500)
# ---------------------------------------------------------------------------

def _parse_legacy_report(data: dict) -> dict[str, list[str]]:
    """Parse the legacy aggregate SWE-bench report format.

    The legacy format uses *_instances integer counts and *_ids list fields.
    Crucially, the legacy format was run against the full 500-instance set
    with only a 100-instance slice selected; the `incomplete_ids` field
    contains the other 400 instances from the full 500 that were NOT in the
    n=100 sample.  We exclude those 400 and treat `incomplete_ids` as empty
    for the n=100 accounting (these are "not in sample", not "failed").
    """
    resolved_ids = list(data.get("resolved_ids", []))
    unresolved_ids = list(data.get("unresolved_ids", []))
    empty_patch_ids = list(data.get("empty_patch_ids", []))
    error_ids = list(data.get("error_ids", []))
    # Do NOT include legacy incomplete_ids — they are the 400 non-selected
    # instances from the full 500, not failures within our n=100 sample.
    return {
        "resolved_ids": resolved_ids,
        "unresolved_ids": unresolved_ids,
        "empty_patch_ids": empty_patch_ids,
        "error_ids": error_ids,
        "incomplete_ids": [],  # normalized away for legacy format
    }


def _parse_normalized_report(data: dict) -> dict[str, list[str]]:
    """Parse the normalized grading_results.json format."""
    return {
        "resolved_ids": list(data.get("resolved_ids", [])),
        "unresolved_ids": list(data.get("unresolved_ids", [])),
        "empty_patch_ids": list(data.get("empty_patch_ids", [])),
        "error_ids": list(data.get("error_ids", [])),
        "incomplete_ids": list(data.get("incomplete_ids", [])),
    }


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_no_duplicates(partitions: dict[str, list[str]]) -> None:
    """Raise ValueError if any ID appears in more than one partition."""
    seen: dict[str, str] = {}
    for name, ids in partitions.items():
        for iid in ids:
            if iid in seen:
                raise ValueError(
                    f"Duplicate/overlapping ID '{iid}' found in both "
                    f"'{seen[iid]}' and '{name}'"
                )
            seen[iid] = name


def _validate_total(partitions: dict[str, list[str]], expected: int = 100) -> None:
    """Raise ValueError if the total ID count does not equal expected."""
    total = sum(len(v) for v in partitions.values())
    if total != expected:
        raise ValueError(
            f"Partition total {total} != expected {expected}; "
            f"breakdown: {{{', '.join(f'{k}:{len(v)}' for k, v in partitions.items())}}}"
        )


def _submitted_from_predictions(preds: dict[str, dict]) -> int:
    """Count instances with a nonempty model_patch."""
    return sum(1 for p in preds.values() if p.get("model_patch", ""))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_swebench_report(repo: Path, bundle: dict[str, Any]) -> dict[str, Any]:
    """Parse a single source bundle into a validated decomposition dict.

    Parameters
    ----------
    repo : Path
        Root of the warpcore-benchmarks repository.
    bundle : dict
        Must contain:
          - "grading": Path to the grading JSON
          - "lifecycle": str ("current" | "historical" | "diagnostic")
        May contain:
          - "predictions": Path to preds.json
          - "is_legacy": bool  (if True, grading is the legacy aggregate format)

    Returns
    -------
    dict with keys:
        expected, submitted, resolved, unresolved, empty_patch,
        grading_error, incomplete, source_paths, provenance, lifecycle
    """
    grading_path: Path = Path(bundle["grading"])
    lifecycle: str = bundle.get("lifecycle", "historical")
    is_legacy: bool = bundle.get("is_legacy", False)
    preds_path: Path | None = (
        Path(bundle["predictions"]) if bundle.get("predictions") else None
    )

    if not grading_path.exists():
        raise FileNotFoundError(f"Grading file not found: {grading_path}")

    raw = json.loads(grading_path.read_text())

    # Detect format: legacy uses total_instances; normalized uses only *_ids
    if is_legacy or "total_instances" in raw:
        partitions = _parse_legacy_report(raw)
    else:
        partitions = _parse_normalized_report(raw)

    # Validate partitions
    _validate_no_duplicates(partitions)
    _validate_total(partitions, expected=100)

    resolved_ids = partitions["resolved_ids"]
    unresolved_ids = partitions["unresolved_ids"]
    empty_patch_ids = partitions["empty_patch_ids"]
    error_ids = partitions["error_ids"]
    incomplete_ids = partitions["incomplete_ids"]

    resolved = len(resolved_ids)
    unresolved = len(unresolved_ids)
    empty_patch = len(empty_patch_ids)
    grading_error = len(error_ids)
    incomplete = len(incomplete_ids)

    try:
        grading_rel = grading_path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        # External paths are permitted only for isolated parser fixtures. Selected
        # repository reports are constructed under repo and checked below.
        grading_rel = grading_path.resolve().as_posix()
    else:
        lookup_path(grading_rel)
    source_paths: list[str] = [grading_rel]
    provenance: str

    if preds_path is not None and preds_path.exists():
        preds = json.loads(preds_path.read_text())
        try:
            preds_rel = preds_path.resolve().relative_to(repo.resolve()).as_posix()
        except ValueError:
            preds_rel = preds_path.resolve().as_posix()
        else:
            lookup_path(preds_rel)
        if set(preds) != set().union(*map(set, partitions.values())):
            raise ValueError("Prediction IDs do not equal the official n=100 grading inventory")
        submitted_ids = {
            iid for iid, pred in preds.items() if pred.get("model_patch", "")
        }
        submitted = len(submitted_ids)
        source_paths.append(preds_rel)

        # The nonempty-patch count from predictions must be internally consistent:
        # At minimum, resolved + unresolved items must have nonempty patches
        # (those received grading verdicts). grading_error items also normally
        # have nonempty patches, but we don't require it here since some edge
        # cases (e.g. mutation tests) may alter them.
        expected_min_submitted = resolved + unresolved
        if submitted < expected_min_submitted:
            raise ValueError(
                f"Predictions nonempty count ({submitted}) < "
                f"resolved+unresolved ({expected_min_submitted}). "
                f"Predictions and grading artifacts disagree."
            )
        provenance = "normalized-complete"
    else:
        # Fall back: derive submitted from the aggregate partition
        # submitted = resolved + unresolved (those with verdicts)
        # grading_error items count as submitted too (nonempty patch implied)
        submitted = resolved + unresolved + grading_error
        submitted_ids = set(resolved_ids) | set(unresolved_ids) | set(error_ids)
        provenance = "legacy-aggregate"

    # Final sanity checks
    if not (resolved <= submitted <= 100):
        raise ValueError(
            f"Impossible: resolved({resolved}) <= submitted({submitted}) <= 100 violated"
        )

    return {
        "expected": 100,
        "submitted": submitted,
        "resolved": resolved,
        "unresolved": unresolved,
        "empty_patch": empty_patch,
        "grading_error": grading_error,
        "incomplete": incomplete,
        "source_paths": source_paths,
        "provenance": provenance,
        "lifecycle": lifecycle,
        "resolved_ids": resolved_ids,
        "unresolved_ids": unresolved_ids,
        "empty_patch_ids": empty_patch_ids,
        "error_ids": error_ids,
        "incomplete_ids": incomplete_ids,
        "submitted_ids": sorted(submitted_ids),
    }


def selected_swebench_reports(repo: Path) -> dict[str, dict[str, Any]]:
    """Return the selected source-derived SWE-bench report for each model.

    This is the single source of truth for which run is selected per model
    and what its source-derived counts are.  All generated data and figures
    must call this function rather than hand-entering counts.

    Returns
    -------
    dict mapping model slug -> validated decomposition dict (same shape as
    load_swebench_report).
    """
    bundles = _source_bundles(repo)
    return {
        model: load_swebench_report(repo, bundle)
        for model, bundle in bundles.items()
    }
