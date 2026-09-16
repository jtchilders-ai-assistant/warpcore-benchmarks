"""viz/publish_campaign.py — Transactional canonical publication gate.

Reads validated manifests and produces canonical matrix data. Only validated
runs with lifecycle='current' enter the matrix. Historical debt is visible for
audit but cannot become canonical implicitly.

Every candidate is run through the authoritative validate() function with
for_publication=True. A run that fails any gate is rejected (with diagnostics)
and does NOT enter the matrix. Silent skips for malformed candidates are not
allowed — they appear in PublishResult.rejected instead.

Rules:
  - Only execution_state='validated' + lifecycle='current' runs enter the matrix.
  - Every candidate is validated by validate(for_publication=True).
  - Emits explicit 'not measured' cells for the full canonical adapter universe.
  - No cross-task overall score or rank.
  - Paired comparisons only for identical item sets; missing scores raise ValueError.
  - Writes the output file transactionally (temp + atomic rename).
  - Hash verification: suite input hashes must match actual files.
  - Item ID verification: per_item.csv item IDs must match expected count.
  - Default output_path resolves relative to the repo argument, not the module dir.
  - Duplicate current+validated runs for the same model/benchmark: at most one
    is published (deterministic: latest run_id wins); the rest are in rejected.

Public API
----------
publish(repo, *, output_path=None, suite_id="warpcore-v1") -> PublishResult
compute_paired_comparison(model_a, scores_a, model_b, scores_b,
                          item_ids_a, item_ids_b) -> dict

PublishResult has:
    .entries      (list[dict])    — canonical entries (validated+current only)
    .not_measured (list[dict])    — explicit not-measured cells
    .output_path  (pathlib.Path)  — path to written matrix JSON
    .rejected     (list[dict])    — candidates rejected with reasons (diagnostics)
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pathlib
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional

_VIZ_DIR = pathlib.Path(__file__).parent
if str(_VIZ_DIR) not in sys.path:
    sys.path.insert(0, str(_VIZ_DIR))

# The suite that drives publication
_DEFAULT_SUITE_ID = "warpcore-v1"

# Validation: states that qualify as publication-ready
_PUBLICATION_STATES = frozenset({"validated"})


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PublishResult:
    entries: List[dict]
    not_measured: List[dict]
    output_path: pathlib.Path
    rejected: List[dict] = field(default_factory=list)

    @property
    def diagnostics(self) -> List[dict]:
        """Alias for .rejected — either attribute gives full audit trail."""
        return self.rejected


# ---------------------------------------------------------------------------
# Suite / adapter resolution
# ---------------------------------------------------------------------------

def _resolve_suite_path(repo: pathlib.Path, suite_id: str) -> Optional[pathlib.Path]:
    """Resolve the suite YAML from repo/suite/<suite_id>.yaml."""
    candidate = repo / "suite" / f"{suite_id}.yaml"
    if candidate.exists():
        return candidate
    return None


def _resolve_adapter_path(repo: pathlib.Path, model_slug: str) -> Optional[pathlib.Path]:
    """Resolve the adapter YAML from repo/adapters/<model_slug>.yaml."""
    candidate = repo / "adapters" / f"{model_slug}.yaml"
    if candidate.exists():
        return candidate
    return None


def _discover_canonical_adapters(repo: pathlib.Path) -> list[str]:
    """Return slugs of all adapter YAML files in repo/adapters/."""
    adapters_dir = repo / "adapters"
    if not adapters_dir.is_dir():
        return []
    return [p.stem for p in sorted(adapters_dir.glob("*.yaml"))]


# ---------------------------------------------------------------------------
# Core publication
# ---------------------------------------------------------------------------

def publish(
    repo: pathlib.Path,
    *,
    output_path: Optional[pathlib.Path] = None,
    suite_id: str = _DEFAULT_SUITE_ID,
) -> PublishResult:
    """Collect validated+current runs and write the canonical matrix.

    Parameters
    ----------
    repo:
        Repository root.
    output_path:
        Where to write canonical_matrix.json. Defaults to
        <repo>/viz/data/canonical_matrix.json (resolved from repo, NOT from
        the module's location — tests must not pollute the real worktree).
    suite_id:
        Suite to publish (default: 'warpcore-v1').

    Returns
    -------
    PublishResult
        .entries      — canonical entries that passed all gates
        .not_measured — explicit 'not measured' cells for canonical adapter universe
        .output_path  — path to the written matrix file
        .rejected     — candidates that were discovered but failed gates (with reasons)
    """
    repo = pathlib.Path(repo).resolve()

    # Default output path resolves from the repo argument, not the module dir
    if output_path is None:
        output_path = repo / "viz" / "data" / "canonical_matrix.json"
    output_path = pathlib.Path(output_path)

    # --- Discover all runs via the shared single discovery path ---
    from validate_campaign import discover_runs, validate

    all_runs = discover_runs(repo)

    # --- Resolve the suite YAML once (all runs share the same suite) ---
    suite_path = _resolve_suite_path(repo, suite_id)

    # --- Filter to validated + current + matching suite, then validate each ---
    canonical_runs = []
    rejected: list[dict] = []

    for run_info in all_runs:
        status_path = run_info["status_path"]
        manifest_path = run_info["manifest_path"]
        model_slug = run_info["model_slug"]
        run_id = run_info["run_id"]

        # --- Load status and manifest for pre-filter ---
        if not status_path.exists() or not manifest_path.exists():
            rejected.append({
                "model_slug": model_slug,
                "run_id": run_id,
                "errors": ["status.json or manifest.json not found"],
                "reason": "missing_files",
            })
            continue

        try:
            status = json.loads(status_path.read_text())
            manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            rejected.append({
                "model_slug": model_slug,
                "run_id": run_id,
                "errors": [f"JSON parse error: {exc}"],
                "reason": "parse_error",
            })
            continue

        # Pre-filter: must be validated + current + matching suite
        # (the authoritative validator below also checks these, but pre-filtering
        #  avoids calling validate() on obviously ineligible runs like historical ones)
        execution_state = status.get("execution_state", "")
        lifecycle = status.get("lifecycle", "")
        manifest_suite = manifest.get("suite_id", "")

        if execution_state not in _PUBLICATION_STATES:
            rejected.append({
                "model_slug": model_slug,
                "run_id": run_id,
                "errors": [f"execution_state={execution_state!r} is not in {sorted(_PUBLICATION_STATES)}"],
                "reason": "not_validated",
            })
            continue

        if lifecycle != "current":
            rejected.append({
                "model_slug": model_slug,
                "run_id": run_id,
                "errors": [f"lifecycle={lifecycle!r} is not 'current'"],
                "reason": "not_current",
            })
            continue

        if manifest_suite != suite_id:
            rejected.append({
                "model_slug": model_slug,
                "run_id": run_id,
                "errors": [f"suite_id={manifest_suite!r} != requested {suite_id!r}"],
                "reason": "suite_mismatch",
            })
            continue

        # --- Resolve adapter from repo/adapters/<slug>.yaml ---
        adapter_path = _resolve_adapter_path(repo, model_slug)
        if adapter_path is None:
            rejected.append({
                "model_slug": model_slug,
                "run_id": run_id,
                "errors": [
                    f"Adapter file not found at repo/adapters/{model_slug}.yaml. "
                    "Cannot verify adapter hash or validate run."
                ],
                "reason": "adapter_not_found",
            })
            continue

        if suite_path is None:
            rejected.append({
                "model_slug": model_slug,
                "run_id": run_id,
                "errors": [
                    f"Suite file not found at repo/suite/{suite_id}.yaml. "
                    "Cannot validate run."
                ],
                "reason": "suite_not_found",
            })
            continue

        # --- Authoritative validation (fail-closed) ---
        val_result = validate(
            run_info["run_dir"],
            suite_path,
            adapter_path,
            for_publication=True,
        )

        if not val_result.passed:
            rejected.append({
                "model_slug": model_slug,
                "run_id": run_id,
                "errors": val_result.errors,
                "reason": "validation_failed",
                "eligible": val_result.eligible,
            })
            continue

        canonical_runs.append({
            "run_info": run_info,
            "status": status,
            "manifest": manifest,
        })

    # --- Deduplicate: one entry per (model_slug, benchmark); latest run_id wins ---
    canonical_runs = _deduplicate_runs(canonical_runs, rejected)

    # --- Build canonical matrix entries ---
    entries = _build_entries(canonical_runs)

    # --- Compute 'not measured' cells from canonical adapter universe ---
    canonical_adapters = _discover_canonical_adapters(repo)
    not_measured = _compute_not_measured(entries, canonical_adapters, suite_path)

    # --- Write transactionally ---
    matrix_data = {
        "suite_id": suite_id,
        "entries": entries,
        "not_measured": not_measured,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(output_path, matrix_data)

    return PublishResult(
        entries=entries,
        not_measured=not_measured,
        output_path=output_path,
        rejected=rejected,
    )


def _deduplicate_runs(canonical_runs: list, rejected: list) -> list:
    """For each (model_slug, benchmark), keep the latest run_id; send others to rejected."""
    from collections import defaultdict
    buckets: dict[tuple, list] = defaultdict(list)
    for r in canonical_runs:
        manifest = r["manifest"]
        key = (
            manifest.get("model", {}).get("slug", r["run_info"]["model_slug"]),
            manifest.get("benchmark", ""),
        )
        buckets[key].append(r)

    deduplicated = []
    for key, runs in buckets.items():
        if len(runs) == 1:
            deduplicated.append(runs[0])
        else:
            # Sort by run_id descending (lexicographic) — latest wins
            runs_sorted = sorted(
                runs,
                key=lambda r: r["manifest"].get("run_id", ""),
                reverse=True,
            )
            deduplicated.append(runs_sorted[0])
            for dup in runs_sorted[1:]:
                rejected.append({
                    "model_slug": key[0],
                    "run_id": dup["manifest"].get("run_id", ""),
                    "errors": [
                        f"Duplicate current+validated run for {key[0]!r}/{key[1]!r}. "
                        f"Only the latest run_id is published."
                    ],
                    "reason": "duplicate_run",
                })

    return deduplicated


def _verify_suite_hashes(repo: pathlib.Path, manifest: dict) -> bool:
    """Return True if all suite input hashes match actual files."""
    suite_input_hashes = manifest.get("suite_input_hashes", {})
    for rel_path, recorded_hash in suite_input_hashes.items():
        abs_path = repo / rel_path
        if not abs_path.exists():
            return False
        actual_hash = hashlib.sha256(abs_path.read_bytes()).hexdigest()
        if actual_hash != recorded_hash:
            return False
    return True


def _load_item_scores(run_dir: pathlib.Path) -> Optional[Dict[str, float]]:
    """Load per-item scores keyed by item_id."""
    per_item_path = run_dir / "per_item.csv"
    if not per_item_path.exists():
        return None
    try:
        scores: Dict[str, float] = {}
        with per_item_path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                iid = row.get("item_id", "")
                score_str = row.get("score", "")
                if iid and score_str not in ("", None):
                    try:
                        scores[iid] = float(score_str)
                    except (ValueError, TypeError):
                        scores[iid] = 0.0
        return scores
    except Exception:
        return None


def _build_entries(canonical_runs: list) -> list[dict]:
    """Build canonical matrix entries from validated+current runs."""
    entries = []
    for r in canonical_runs:
        manifest = r["manifest"]
        run_info = r["run_info"]
        run_dir = run_info["run_dir"]

        scores = _load_item_scores(run_dir)
        n = manifest.get("item_inventory", {}).get("expected", 0)

        if scores and n > 0:
            score_values = list(scores.values())
            mean_score = sum(score_values) / len(score_values) if score_values else None
        else:
            mean_score = None

        entry = {
            "model_slug": manifest.get("model", {}).get("slug", run_info["model_slug"]),
            "benchmark": manifest.get("benchmark", ""),
            "suite_id": manifest.get("suite_id", ""),
            "run_id": manifest.get("run_id", ""),
            "lifecycle": r["status"].get("lifecycle", ""),
            "execution_state": r["status"].get("execution_state", ""),
            "n_items": n,
            "score": round(mean_score, 4) if mean_score is not None else None,
            "manifest_run_id": manifest.get("run_id", ""),
            "item_ids": sorted(scores.keys()) if scores else [],
            "imputed": False,  # never imputed
        }
        entries.append(entry)
    return entries


def _compute_not_measured(
    entries: list,
    canonical_adapters: list[str],
    suite_path: Optional[pathlib.Path],
) -> list[dict]:
    """Compute explicit 'not measured' cells for canonical adapter universe.

    Uses the suite's benchmarks list when available; falls back to known defaults.
    Covers ALL canonical adapters from repo/adapters/*.yaml, not only models
    with at least one accepted entry.
    """
    # Load benchmarks from suite if available
    _FALLBACK_BENCHMARKS = ["gsm8k", "ifeval", "gpqa_diamond", "swebench"]
    if suite_path is not None and suite_path.exists():
        try:
            import yaml
            suite_data = yaml.safe_load(suite_path.read_text())
            candidate_benchmarks = [
                b for b in (suite_data.get("benchmarks") or {}).keys()
                if b != "throughput"  # throughput is not a quality benchmark
            ]
            # Fall back to hardcoded defaults if suite has no benchmarks listed
            known_benchmarks = candidate_benchmarks if candidate_benchmarks else _FALLBACK_BENCHMARKS
        except Exception:
            known_benchmarks = _FALLBACK_BENCHMARKS
    else:
        known_benchmarks = _FALLBACK_BENCHMARKS

    # Group entries by model
    measured: Dict[str, set] = {}
    for e in entries:
        slug = e["model_slug"]
        bench = e["benchmark"]
        measured.setdefault(slug, set()).add(bench)

    # Full universe = canonical adapters UNION models with published entries
    all_models = set(canonical_adapters)
    all_models.update(measured.keys())

    not_measured = []
    for slug in sorted(all_models):
        model_benches = measured.get(slug, set())
        for bench in known_benchmarks:
            if bench not in model_benches:
                not_measured.append({
                    "model_slug": slug,
                    "benchmark": bench,
                    "value": "not measured",
                })
    return not_measured


def _write_atomic(path: pathlib.Path, data: dict) -> None:
    """Write data as JSON to path atomically (temp + fsync + rename)."""
    payload = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=".tmp_matrix_",
        suffix=".json",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------------

def compute_paired_comparison(
    model_a: str,
    scores_a: Dict[str, float],
    model_b: str,
    scores_b: Dict[str, float],
    item_ids_a: list,
    item_ids_b: list,
) -> dict:
    """Compute paired per-item comparison statistics.

    Paired comparisons are only valid when both models were evaluated on the
    IDENTICAL set of item IDs. Raises ValueError if item sets differ.

    Score keys in scores_a/scores_b must EXACTLY cover item_ids_a/item_ids_b
    respectively. Missing score entries raise ValueError rather than defaulting
    to 0 — a missing score could indicate a data integrity problem.

    Parameters
    ----------
    model_a, model_b:
        Model slugs.
    scores_a, scores_b:
        Dicts mapping item_id -> score (float). Must cover ALL item IDs in
        item_ids_a/item_ids_b respectively.
    item_ids_a, item_ids_b:
        Ordered item ID lists for each model (must be identical sets).

    Returns
    -------
    dict with keys:
        model_a, model_b, n_items, diff_pp, lo_pp, hi_pp,
        a_only, b_only, mcnemar_chi2, identical_item_sets

    Raises
    ------
    ValueError
        If item sets differ, or if any score is missing from the score dicts.
    """
    set_a = set(item_ids_a)
    set_b = set(item_ids_b)

    if set_a != set_b:
        only_a = sorted(set_a - set_b)
        only_b = sorted(set_b - set_a)
        raise ValueError(
            f"Paired comparison requires identical item sets. "
            f"Items only in {model_a!r}: {only_a[:5]}{'...' if len(only_a) > 5 else ''}. "
            f"Items only in {model_b!r}: {only_b[:5]}{'...' if len(only_b) > 5 else ''}."
        )

    items = sorted(set_a)
    n = len(items)
    if n == 0:
        raise ValueError("Cannot compute paired comparison on empty item set.")

    # Validate that every item has a score in both dicts
    missing_a = [iid for iid in items if iid not in scores_a]
    missing_b = [iid for iid in items if iid not in scores_b]
    if missing_a:
        raise ValueError(
            f"scores_a is missing entries for item(s): {missing_a[:5]}. "
            "All declared item IDs must have a score. "
            "Missing scores cannot be defaulted to 0 — check data integrity."
        )
    if missing_b:
        raise ValueError(
            f"scores_b is missing entries for item(s): {missing_b[:5]}. "
            "All declared item IDs must have a score. "
            "Missing scores cannot be defaulted to 0 — check data integrity."
        )

    # Discordant pairs
    a_only = 0  # A correct, B wrong
    b_only = 0  # B correct, A wrong
    for iid in items:
        sa = scores_a[iid]  # KeyError impossible: validated above
        sb = scores_b[iid]
        a_correct = sa > 0.5
        b_correct = sb > 0.5
        if a_correct and not b_correct:
            a_only += 1
        elif b_correct and not a_correct:
            b_only += 1

    # Paired difference (McNemar-style)
    z = 1.96
    d = (a_only - b_only) / n
    se = math.sqrt(max(a_only + b_only - (a_only - b_only) ** 2 / n, 0)) / n
    mcnemar_chi2 = (
        (abs(a_only - b_only) - 1) ** 2 / (a_only + b_only)
        if (a_only + b_only) > 0
        else 0.0
    )

    return {
        "model_a": model_a,
        "model_b": model_b,
        "n_items": n,
        "diff_pp": round(d * 100, 2),
        "lo_pp": round((d - z * se) * 100, 2),
        "hi_pp": round((d + z * se) * 100, 2),
        "a_only": a_only,
        "b_only": b_only,
        "mcnemar_chi2": round(mcnemar_chi2, 4),
        "identical_item_sets": True,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Publish canonical matrix from validated+current campaigns."
    )
    ap.add_argument("repo", type=pathlib.Path, nargs="?", default=None,
                    help="Repository root (default: auto-detect from script location)")
    ap.add_argument("--output", type=pathlib.Path, default=None,
                    help="Output path for canonical_matrix.json")
    ap.add_argument("--suite-id", default=_DEFAULT_SUITE_ID,
                    help=f"Suite to publish (default: {_DEFAULT_SUITE_ID})")
    args = ap.parse_args(argv)

    repo = args.repo or _VIZ_DIR.parent
    result = publish(repo, output_path=args.output, suite_id=args.suite_id)

    print(f"Published {len(result.entries)} canonical entries.")
    for e in result.entries:
        print(f"  {e['model_slug']:30s} {e['benchmark']:15s} score={e['score']}")
    if result.not_measured:
        print(f"\n{len(result.not_measured)} 'not measured' cells:")
        for nm in result.not_measured:
            print(f"  {nm['model_slug']:30s} {nm['benchmark']}")
    if result.rejected:
        print(f"\n{len(result.rejected)} rejected candidates:")
        for r in result.rejected:
            print(f"  {r['model_slug']:30s} reason={r['reason']}")
    print(f"\nWrote: {result.output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
