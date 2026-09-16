#!/usr/bin/env python3
"""viz/run_quality.py — Contract-aware quality runner for warpcore-v1 benchmarks.

Given suite, adapter, benchmark, endpoint, throughput, concurrency, and timeout,
builds a canonical lm-eval command from suite settings and executes it under the
required safety gates.

CONTRACT (from docs/superpowers/specs/2026-09-15-warpcore-v1-apples-to-apples-design.md):
- Generated command uses local-chat-completions (routes to /v1/chat/completions).
- Includes --apply_chat_template and tokenized_requests=False.
- Uses --include_path <task_dir> and --tasks <task_name_from_yaml>.
- Uses canonical task name from suite task_file YAML 'task:' field, not the file path.
- Includes --log_samples.
- Points output at normalized run directory's raw/ subdirectory.
- CLI does NOT accept overrides for task, ceiling, sampling, scoring, datasets, or instances.
- Lifecycle order: planned -> QualityPreflightGate -> preflight_passed -> running -> harness.
- Refuses long launch outside /usr/bin/screen except in --dry-run or explicit test mode.
- Uses dependency-injected process execution.
- Does not duplicate preflight_serving.py, check_output_budget.py, or timeout arithmetic.
- Writes exact argv to command.txt using shell-safe quoting.
- Captures harness stdout+stderr in run.log (run_dir/run.log).
- On success: verifies required raw files (aggregate result + nonempty samples) before writing DONE.
  DONE is written only after completed status is schema-validated and written successfully.
- On nonzero harness exit: does NOT write DONE; records exit_code in history entry;
  transitions to failed. Does NOT add top-level harness_exit_code (schema forbids it).
- State transitions are fatal: missing/invalid status.json or illegal transition aborts run.
  _read_status_strict validates against JSON schema, verifies run_id==run_dir.name,
  and verifies suite_id matches loaded suite.
- Noncanonical adapters are rejected at construction time (no allow_noncanonical_adapter bypass).
- Run directory must be exactly repo/results/<adapter-slug>/runs/<suite_id>/<bench>/<run_id>.
- Completion is atomic: schema-validate completed status, write status, then write DONE.
  If status write fails, return nonzero — never return success with running status.
- Evidence validator recursively finds artifacts under raw/ subdirectories.

USAGE
-----
    # Dry-run — inspect command without network/GPU work:
    python3 viz/run_quality.py \\
        --suite suite/warpcore-v1.yaml \\
        --adapter adapters/qwen3.6-35b-a3b.yaml \\
        --benchmark gsm8k \\
        --endpoint http://csi370295.alcf.anl.gov:8000/v1 \\
        --throughput 64 \\
        --concurrency 8 \\
        --timeout 14400 \\
        --run-id run-2026-09-15T12-00-00 \\
        --prompt-tokens gsm8k=500 \\
        --dry-run

    # Live run (must be inside /usr/bin/screen):
    screen -S quality-gsm8k
    python3 viz/run_quality.py \\
        --suite suite/warpcore-v1.yaml \\
        --adapter adapters/qwen3.6-35b-a3b.yaml \\
        --benchmark gsm8k \\
        --endpoint http://csi370295.alcf.anl.gov:8000/v1 \\
        --throughput 64 \\
        --concurrency 8 \\
        --timeout 14400 \\
        --run-id run-2026-09-15T12-00-00 \\
        --prompt-tokens gsm8k=500

EXIT CODES
----------
    0   Success: harness exited 0 and required raw files exist; DONE written.
    1   Preflight defect, harness failure, or lifecycle/evidence error; DONE not written.
    2   Preflight inconclusive or screen guard triggered; DONE not written.
    3   Command construction, validation, or containment error.
"""
from __future__ import annotations

import argparse
import csv
import gzip as _gzip_module
import json
import os
import pathlib
import re
import shlex
import subprocess
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

from quality_preflight import QualityPreflightGate, ExitCode  # noqa: E402
import campaign_state  # noqa: E402
from contract import (  # noqa: E402
    load_yaml,
    validate_json,
    validate_adapter_campaign_ready,
)

# ---------------------------------------------------------------------------
# Schemas dir — always the real repo schemas, regardless of run-dir layout repo
# ---------------------------------------------------------------------------

_SCHEMAS_DIR = _REPO_DIR / "suite" / "schemas"
_STATUS_SCHEMA = _SCHEMAS_DIR / "result-status.schema.json"
_ADAPTER_SCHEMA = _SCHEMAS_DIR / "adapter.schema.json"

# ---------------------------------------------------------------------------
# Forbidden CLI override flags
# ---------------------------------------------------------------------------

#: CLI flags that override suite-owned experiment variables.
#: The runner refuses any invocation that passes these.
FORBIDDEN_OVERRIDE_FLAGS: List[str] = [
    "--tasks",
    "--num_fewshot",
    "--gen_kwargs",
    "--limit",
    "--include_path",
    "--max_gen_toks",
    "--temperature",
    "--do_sample",
    "--fewshot_config",
    "--dataset_path",
    "--dataset_name",
    "--dataset_kwargs",
    "--use_cache",
    "--cache_requests",
]

# ---------------------------------------------------------------------------
# UTC timestamp helper
# ---------------------------------------------------------------------------


def _utcnow() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Screen guard
# ---------------------------------------------------------------------------


def _is_under_screen() -> bool:
    """Return True when the process is running inside a GNU screen session."""
    return bool(os.environ.get("STY", ""))


# ---------------------------------------------------------------------------
# Evidence verification (recursive)
# ---------------------------------------------------------------------------

_AGGREGATE_RESULT_RE = re.compile(r"^results_.*\.json$")
_SAMPLES_JSONL_RE = re.compile(r"^samples_.*\.jsonl$")
_SAMPLES_JSONL_GZ_RE = re.compile(r"^samples_.*\.jsonl\.gz$")


def _normalize_lmeval_samples(raw_dir: pathlib.Path) -> List[pathlib.Path]:
    """Atomically gzip lm-eval 0.4.12's plain sample JSONL outputs.

    The pinned harness writes ``samples_*.jsonl`` despite the campaign contract
    requiring retained ``samples_*.jsonl.gz`` evidence. Existing compressed files
    are retained. A basename present in both forms is rejected as ambiguous rather
    than silently selecting one copy.
    """
    raw_dir = pathlib.Path(raw_dir)
    plain_files = sorted(
        p for p in raw_dir.rglob("*")
        if p.is_file() and _SAMPLES_JSONL_RE.match(p.name)
    )
    compressed = {
        p.resolve(): p for p in raw_dir.rglob("*")
        if p.is_file() and _SAMPLES_JSONL_GZ_RE.match(p.name)
    }

    for source in plain_files:
        destination = source.with_name(source.name + ".gz")
        if destination.resolve() in compressed or destination.exists():
            raise RuntimeError(
                f"Ambiguous sample evidence: both {source} and {destination} exist"
            )
        temporary = destination.with_name(destination.name + ".tmp")
        try:
            with source.open("rb") as src, _gzip_module.open(temporary, "wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
            os.replace(temporary, destination)
            source.unlink()
            compressed[destination.resolve()] = destination
        except Exception:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise

    return sorted(compressed.values())


def _response_text(rec: dict) -> str:
    """Return the raw model generation for a lm-eval sample record."""
    resps = rec.get("resps") or []
    if resps and isinstance(resps[0], list) and resps[0]:
        return resps[0][0] or ""
    if resps and isinstance(resps[0], str):
        return resps[0]
    filt = rec.get("filtered_resps") or []
    if filt and isinstance(filt[0], str):
        return filt[0]
    return ""


def _score_of(rec: dict) -> Optional[float]:
    """Extract the canonical score from a lm-eval sample record."""
    for key in ("exact_match", "acc", "prompt_level_strict_acc", "acc_norm"):
        if key in rec:
            try:
                return float(rec[key])
            except (TypeError, ValueError):
                pass
    return None


def _load_sidecar_metadata(sidecar_path: pathlib.Path) -> Dict[str, dict]:
    """Load sidecar metadata into a fingerprint-indexed dict.

    Returns dict[fingerprint_str, record] for fast lookup.
    If sidecar_path does not exist or is empty, returns {}.
    """
    if not sidecar_path or not pathlib.Path(sidecar_path).exists():
        return {}
    meta_index: Dict[str, dict] = {}
    try:
        with open(sidecar_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    fp = rec.get("fingerprint")
                    if fp and fp not in meta_index:
                        meta_index[fp] = rec
    except Exception:
        pass  # caller has already reconciled; any failure here is secondary
    return meta_index


def _fingerprint_from_sample(sample: dict) -> Optional[str]:
    """Compute request fingerprint from an lm-eval sample record.

    lm-eval 0.4.12 stores messages at:
        sample["arguments"]["gen_args_0"]["arg_0"][0]  (a JSON-encoded messages list)
    """
    import hashlib as _hashlib
    try:
        arg0 = sample["arguments"]["gen_args_0"]["arg_0"]
        if isinstance(arg0, list):
            msgs_str = arg0[0]
        else:
            msgs_str = arg0
        messages = json.loads(msgs_str)
        canonical = json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return _hashlib.sha256(canonical).hexdigest()
    except (KeyError, TypeError, json.JSONDecodeError, IndexError):
        return None


def _derive_per_item_data(
    run_dir: pathlib.Path,
    sidecar_path: Optional[pathlib.Path] = None,
) -> Dict[str, dict]:
    """Derive per-item evidence from retained samples_*.jsonl.gz files.

    Reads all samples_*.jsonl.gz under run_dir/raw/ (recursively).
    Deduplicates by doc_id so multi-filter tasks (e.g. GPQA's answer-line +
    flexible-fallback) produce exactly ONE row per item.

    When sidecar_path is provided, enriches each item with:
      - finish_reason: from sidecar metadata (not always available in lm-eval JSONL)
      - disposition: from sidecar classify_response() classification:
          "scored" | "budget" | "parser" | "empty"

    Returns a dict keyed by str(doc_id):
        {
            "item_id":       str,   # str(doc_id)
            "score":         str,   # float formatted, or "" if None
            "response_chars": int,
            "empty_content": int,   # 1 if empty, 0 otherwise
            "disposition":   str,   # from metadata or "scored"/"empty_response"
            "finish_reason": str,   # from metadata, or "" if unavailable
        }

    Raises RuntimeError if no samples files are found or if all are corrupt/empty.
    Raises gzip.BadGzipFile / json.JSONDecodeError for individual corrupt files.
    """
    raw_dir = run_dir / "raw"
    all_gz = [
        f for f in raw_dir.rglob("*")
        if f.is_file() and _SAMPLES_JSONL_GZ_RE.match(f.name)
    ]
    if not all_gz:
        raise RuntimeError(
            f"No samples_*.jsonl.gz found under {raw_dir} — cannot derive per-item evidence."
        )

    # Load sidecar metadata for enrichment (best-effort; reconciliation already passed)
    meta_index = _load_sidecar_metadata(sidecar_path) if sidecar_path else {}

    # Accumulate: for each doc_id keep the "best" view across all filter rows.
    # "Best" = non-empty text preferred over empty; best non-None score kept.
    # Also track the fingerprint so we can look up sidecar metadata.
    items: Dict[str, dict] = {}

    for gz_path in sorted(all_gz):
        with _gzip_module.open(gz_path, "rt", encoding="utf-8") as fh:
            content = fh.read()
        lines = [l.strip() for l in content.splitlines() if l.strip()]
        if not lines:
            raise RuntimeError(
                f"samples file {gz_path.name} contains no lines — "
                "evidence is absent; refusing to proceed."
            )
        for line in lines:
            rec = json.loads(line)
            doc_id = rec.get("doc_id")
            key = str(doc_id)
            text = _response_text(rec)
            score = _score_of(rec)
            fp = _fingerprint_from_sample(rec)

            prev = items.get(key)
            if prev is None:
                items[key] = {
                    "item_id": key,
                    "score": score,
                    "_text": text,
                    "_fp": fp,
                }
            else:
                # Keep non-empty text if any filter produced output
                if not prev["_text"].strip() and text.strip():
                    prev["_text"] = text
                # Keep fingerprint if we didn't have one
                if prev["_fp"] is None and fp is not None:
                    prev["_fp"] = fp
                # Keep best score (prefer non-None, then highest)
                if prev["score"] is None and score is not None:
                    prev["score"] = score
                elif (score is not None and prev["score"] is not None
                      and score > prev["score"]):
                    prev["score"] = score

    if not items:
        raise RuntimeError(
            "No item records found in any samples file — cannot derive per-item evidence."
        )

    # Import classifier for sidecar-based disposition
    _classify = None
    if meta_index:
        try:
            if str(_VIZ_DIR) not in sys.path:
                sys.path.insert(0, str(_VIZ_DIR))
            from lmeval_sidecar.reconcile import classify_response as _cr
            _classify = _cr
        except ImportError:
            pass

    # Finalize: compute derived fields, enrich from sidecar where available
    result: Dict[str, dict] = {}
    for key, item in items.items():
        text = item["_text"]
        rc = len(text)
        empty = 1 if rc == 0 else 0

        # Look up sidecar metadata via fingerprint
        fp = item.get("_fp")
        meta_rec = meta_index.get(fp) if fp and meta_index else None

        if meta_rec and _classify is not None:
            disposition = _classify(meta_rec)
            finish_reasons = meta_rec.get("finish_reasons") or []
            finish_reason = finish_reasons[0] if finish_reasons else ""
            if finish_reason is None:
                finish_reason = ""
        else:
            disposition = "empty_response" if empty else "scored"
            finish_reason = ""  # not stored in lm-eval JSONL without sidecar

        score_val = item["score"]
        result[key] = {
            "item_id": key,
            "score": "" if score_val is None else str(score_val),
            "response_chars": rc,
            "empty_content": empty,
            "disposition": disposition,
            "finish_reason": str(finish_reason),
        }
    return result




def _write_per_item_csv(run_dir: pathlib.Path,
                        items: Dict[str, dict]) -> pathlib.Path:
    """Write run_dir/per_item.csv with the canonical schema expected by the validator.

    Columns: item_id, score, response_chars, empty_content, disposition, finish_reason
    Rows sorted by item_id (numeric sort when possible).
    Returns the path to the written file.
    """
    out_path = run_dir / "per_item.csv"
    fieldnames = [
        "item_id", "score", "response_chars", "empty_content",
        "disposition", "finish_reason",
    ]

    def sort_key(k: str) -> tuple:
        try:
            return (0, int(k))
        except ValueError:
            return (1, k)

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for key in sorted(items.keys(), key=sort_key):
            writer.writerow(items[key])
    return out_path


def _update_manifest_on_completion(
    run_dir: pathlib.Path,
    submitted_count: int,
    completed_utc: str,
) -> None:
    """Update manifest.json with completion metadata.

    Sets:
      - timing.completed_utc = completed_utc
      - item_inventory.submitted = submitted_count
      - artifact_inventory.samples_jsonl_gz = True (if samples exist)
      - artifact_inventory.per_item_csv = True (if per_item.csv exists)
      - artifact_inventory.run_log = True (if run.log exists)
      - artifact_inventory.command_txt = True (if command.txt exists)
      - artifact_inventory.done_sentinel = True (always — DONE is written next)

    Raises OSError if manifest cannot be read or written.
    Raises json.JSONDecodeError if manifest is corrupt.
    """
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Update timing
    timing = manifest.setdefault("timing", {})
    timing["completed_utc"] = completed_utc

    # Update submitted count
    inventory = manifest.setdefault("item_inventory", {})
    inventory["submitted"] = submitted_count

    # Update artifact_inventory
    art = manifest.setdefault("artifact_inventory", {})
    raw_dir = run_dir / "raw"
    # samples_jsonl_gz: any samples file under raw/
    art["samples_jsonl_gz"] = any(
        f.is_file() and _SAMPLES_JSONL_GZ_RE.match(f.name)
        for f in raw_dir.rglob("*")
        if raw_dir.exists()
    )
    # per_item_csv: present at run_dir/per_item.csv
    art["per_item_csv"] = (run_dir / "per_item.csv").exists()
    # run_log: present at run_dir/run.log
    art["run_log"] = (run_dir / "run.log").exists()
    # command_txt: present at run_dir/command.txt
    art["command_txt"] = (run_dir / "command.txt").exists()
    # done_sentinel: about to be written; mark True now (atomicity: we write
    # manifest before DONE so that manifest is never wrong)
    art["done_sentinel"] = True

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _verify_and_reconcile_sidecar(
    run_dir: pathlib.Path,
    sidecar_path: pathlib.Path,
) -> int:
    """Verify sidecar file exists and passes reconciliation. Returns 0 on pass, 1 on fail.

    Fail-closed: any issue (missing, corrupt, reconciliation failure) returns 1.
    """
    import sys

    if not sidecar_path.exists():
        print(
            f"[run-quality] FATAL: Sidecar metadata file not found at {sidecar_path}. "
            "The sidecar runner must produce this file before DONE can be written. "
            "DONE not written.",
            file=sys.stderr,
        )
        return 1

    # Find the samples gz under raw/
    raw_dir = run_dir / "raw"
    if not raw_dir.exists():
        print(
            f"[run-quality] FATAL: raw/ directory does not exist at {raw_dir}.",
            file=sys.stderr,
        )
        return 1

    samples_files = [
        f for f in raw_dir.rglob("*")
        if f.is_file() and _SAMPLES_JSONL_GZ_RE.match(f.name)
    ]
    if not samples_files:
        print(
            f"[run-quality] FATAL: No samples_*.jsonl.gz found under {raw_dir} "
            "for sidecar reconciliation.",
            file=sys.stderr,
        )
        return 1

    # Import reconciler (in the same viz/ package)
    try:
        # Add viz/ to path so lmeval_sidecar is importable
        if str(_VIZ_DIR) not in sys.path:
            sys.path.insert(0, str(_VIZ_DIR))
        from lmeval_sidecar.reconcile import reconcile_inventory
    except ImportError as exc:
        print(
            f"[run-quality] FATAL: Cannot import lmeval_sidecar.reconcile: {exc}. "
            "Install the sidecar package or check PYTHONPATH.",
            file=sys.stderr,
        )
        return 1

    # Reconcile once across the complete inventory. Per-file reconciliation would
    # falsely classify metadata belonging to sibling sample files as foreign.
    try:
        report = reconcile_inventory(sorted(samples_files), sidecar_path)
    except Exception as exc:
        print(
            f"[run-quality] FATAL: Whole-inventory reconciliation raised: {exc}. "
            "DONE not written.",
            file=sys.stderr,
        )
        return 1

    if report["exit_code"] != 0:
        print(
            "[run-quality] FATAL: Sidecar reconciliation FAILED. DONE not written.\n"
            + "\n".join(report["errors"]),
            file=sys.stderr,
        )
        return 1

    return 0



def _verify_required_evidence(run_dir: pathlib.Path) -> List[str]:
    """Return a list of error strings if required harness artifacts are missing.

    Required minimum (subset of full required_evidence from suite):
      - At least one aggregate result JSON (results_*.json) that is nonempty
      - At least one nonempty valid gzip samples file (samples_*.jsonl.gz)

    Searches recursively under raw/ to find artifacts in subdirectories
    (lm-eval 0.4.12 writes into named subdirs under the output path).

    Returns [] on success, list of errors on failure.
    """
    import gzip as _gzip

    raw_dir = run_dir / "raw"
    errors: List[str] = []

    if not raw_dir.exists():
        errors.append(f"raw/ directory does not exist at {raw_dir}")
        return errors

    # Recursive search for artifacts under raw/
    all_files = list(raw_dir.rglob("*"))
    if not any(f.is_file() for f in all_files):
        errors.append(f"raw/ directory is empty at {raw_dir} — no harness artifacts produced")
        return errors

    # Find aggregate result JSONs (nonempty)
    aggregate_results = [
        f for f in all_files
        if f.is_file() and _AGGREGATE_RESULT_RE.match(f.name)
    ]
    nonempty_results = [f for f in aggregate_results if f.stat().st_size > 0]
    if not nonempty_results:
        if aggregate_results:
            errors.append(
                f"All results_*.json files in {raw_dir} (searched recursively) are empty. "
                "lm-eval must produce a nonempty aggregate result JSON."
            )
        else:
            errors.append(
                f"No aggregate result file (results_*.json) found under {raw_dir} "
                "(searched recursively). lm-eval must produce at least one results JSON."
            )

    # Find samples JSONL.GZ files — must be valid nonempty gzip
    samples_gz = [
        f for f in all_files
        if f.is_file() and _SAMPLES_JSONL_GZ_RE.match(f.name)
    ]
    if not samples_gz:
        errors.append(
            f"No samples file (samples_*.jsonl.gz) found under {raw_dir} "
            "(searched recursively). lm-eval must produce compressed samples (--log_samples required)."
        )
    else:
        valid_nonempty = []
        for gz_path in samples_gz:
            try:
                with _gzip.open(gz_path, "rb") as fh:
                    content = fh.read(1)
                if len(content) > 0:
                    valid_nonempty.append(gz_path)
                else:
                    errors.append(
                        f"samples file {gz_path.name} is valid gzip but contains no data. "
                        "Evidence must be a nonempty archive."
                    )
            except Exception:
                errors.append(
                    f"samples file {gz_path.name} is corrupt or not valid gzip. "
                    "Evidence must be a valid nonempty gzip archive."
                )

    return errors


# ---------------------------------------------------------------------------
# Task name extraction
# ---------------------------------------------------------------------------


def _read_task_name_from_yaml(task_file_path: pathlib.Path) -> str:
    """Read the registered 'task:' name from a lm-eval task YAML file."""
    import yaml as _yaml

    class _PermissiveLoader(_yaml.SafeLoader):
        pass

    _PermissiveLoader.add_multi_constructor(
        "",
        lambda loader, tag_suffix, node: None,
    )

    raw = pathlib.Path(task_file_path).read_text(encoding="utf-8")
    data = _yaml.load(raw, Loader=_PermissiveLoader)  # noqa: S506
    if not isinstance(data, dict):
        raise ValueError(
            f"Task YAML {task_file_path} did not parse to a mapping; got {type(data).__name__}."
        )
    task_name = data.get("task", "") or ""
    if not task_name:
        raise ValueError(
            f"Task YAML {task_file_path} does not define a 'task:' field."
        )
    return str(task_name)


# ---------------------------------------------------------------------------
# Normalized run directory validation
# ---------------------------------------------------------------------------


def _validate_run_dir_identity(
    run_dir: pathlib.Path,
    repo: pathlib.Path,
    adapter_slug: str,
    suite_id: str,
    benchmark: str,
    run_id: str,
) -> List[str]:
    """Validate that run_dir exactly equals repo/results/<slug>/runs/<suite_id>/<bench>/<run_id>.

    Returns [] on success, list of error strings on failure.
    """
    expected = (
        repo / "results" / adapter_slug / "runs" / suite_id / benchmark / run_id
    ).resolve()
    actual = run_dir.resolve()
    if actual != expected:
        return [
            f"Run directory identity mismatch: "
            f"expected {expected}, got {actual}. "
            f"Run directories must match the exact normalized layout: "
            f"<repo>/results/<adapter-slug>/runs/<suite-id>/<benchmark>/<run-id>."
        ]
    return []


# ---------------------------------------------------------------------------
# QualityRunner
# ---------------------------------------------------------------------------


class QualityRunner:
    """Contract-aware quality runner.

    Parameters
    ----------
    suite_path : pathlib.Path
        Path to warpcore-v1.yaml.
    adapter_path : pathlib.Path
        Path to adapter YAML.
    benchmark : str
        Benchmark key (e.g. "gsm8k").
    endpoint : str
        OpenAI-compatible base URL ending in /v1.
    throughput : float
        Measured aggregate token throughput (tok/s).
    concurrency : int
        Number of parallel lm-eval workers.
    timeout : float
        lm-eval --timeout value in seconds.
    run_dir : pathlib.Path
        Normalized run directory.
    repo : pathlib.Path or None
        Repository root for run directory layout. Defaults to parent of suite file.
    prompt_token_maxima : dict or None
        Measured tokenized prompt maxima keyed by benchmark.
    dry_run : bool
        If True, build and print command without invoking preflight or harness.
    allow_no_screen : bool
        Bypass the screen guard (tests and special ops only).
    preflight_runner : callable or None
        Injected callable(model_id) -> int.
    harness_runner : callable or None
        Injected callable(cmd: list, **kw) -> int.
    """

    def __init__(
        self,
        suite_path: pathlib.Path,
        adapter_path: pathlib.Path,
        benchmark: str,
        endpoint: str,
        throughput: float,
        concurrency: int,
        timeout: float,
        run_dir: pathlib.Path,
        repo: Optional[pathlib.Path] = None,
        prompt_token_maxima: Optional[Dict[str, int]] = None,
        dry_run: bool = False,
        allow_no_screen: bool = False,
        preflight_runner: Optional[Callable] = None,
        harness_runner: Optional[Callable] = None,
    ) -> None:
        self.suite_path = pathlib.Path(suite_path).resolve()
        self.adapter_path = pathlib.Path(adapter_path).resolve()
        self.benchmark = benchmark
        self.endpoint = endpoint
        self.throughput = float(throughput)
        self.concurrency = int(concurrency)
        self.timeout = float(timeout)
        self.run_dir = pathlib.Path(run_dir).resolve()
        self.dry_run = dry_run
        self.allow_no_screen = allow_no_screen
        self.prompt_token_maxima = prompt_token_maxima

        # _repo: used ONLY for run directory layout (not for schema lookup)
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
        # Always uses _ADAPTER_SCHEMA from the real repo (module-level constant),
        # not from self._repo (which may be a temp dir in tests).
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

        # -- Campaign-readiness validation (noncanonical adapters blocked) --
        model_slug = (self._adapter.get("model") or {}).get("slug", "")
        readiness_errors = validate_adapter_campaign_ready(
            self._adapter,
            model_slug,
            suite=self._suite,
            prompt_token_maxima=self.prompt_token_maxima,
        )
        if readiness_errors:
            raise ValueError(
                f"Adapter {self.adapter_path} is not campaign-ready (noncanonical):\n"
                + "\n".join(readiness_errors)
            )

        # Resolve benchmark config from suite
        benchmarks = self._suite.get("benchmarks", {})
        if benchmark not in benchmarks:
            raise ValueError(
                f"Benchmark {benchmark!r} not found in suite {suite_path}. "
                f"Available: {sorted(benchmarks.keys())}"
            )
        self._bench_cfg: dict = benchmarks[benchmark]

        # Suite-level IDs
        self._suite_id: str = self._suite.get("suite_id", "warpcore-v1")
        self._run_id: str = self.run_dir.name
        self._model_slug: str = model_slug

        # Injected runners
        self._preflight_runner = preflight_runner
        self._harness_runner = harness_runner

    # ------------------------------------------------------------------
    # Command construction
    # ------------------------------------------------------------------

    def build_command(self) -> List[str]:
        """Build the canonical lm-eval argv list from suite + adapter settings.

        Uses the sidecar wrapper (lmeval_sidecar_runner.py) instead of
        `python -m lm_eval` so that response metadata is captured for every
        request. The sidecar path is placed under run_dir/raw/ and passed via
        environment (not argv) so no credentials appear in the command line.

        The sidecar runner asserts lm_eval==0.4.12 at startup and fails closed
        if the wrong version is installed.
        """
        bench = self._bench_cfg

        task_file = bench.get("task_file", "")
        if task_file:
            # Resolve from the actual task file location (always in real repo)
            canonical_task_path = (_REPO_DIR / task_file).resolve()
            task_dir = str(canonical_task_path.parent)
            task_name = _read_task_name_from_yaml(canonical_task_path)
        else:
            task_dir = ""
            task_name = bench.get("task_name", self.benchmark)

        generation_ceiling: int = bench.get("generation_ceiling", 8192)
        retry_policy = bench.get("retry_policy", {})
        max_retries: int = retry_policy.get("max_retries", 0)
        sampling = bench.get("sampling", {})
        temperature = sampling.get("temperature", 0)
        do_sample = sampling.get("do_sample", False)

        model_id = (self._adapter.get("model") or {}).get("id", "")

        gen_kwargs = (
            f"max_gen_toks={generation_ceiling},"
            f"temperature={temperature},"
            f"do_sample={str(do_sample).lower()}"
        )
        # Sidecar path: under run_dir/raw/, passed via model_args (not env)
        # so it is visible in command.txt for auditability; contains no credentials.
        sidecar_path = str(self.run_dir / "raw" / "response_metadata.jsonl")
        model_args = (
            f"base_url={self.endpoint},"
            f"model={model_id},"
            f"num_concurrent={self.concurrency},"
            f"max_retries={max_retries},"
            f"timeout={int(self.timeout)},"
            f"tokenized_requests=False,"
            f"sidecar_path={sidecar_path}"
        )

        output_path = str(self.run_dir / "raw")

        # Sidecar runner path (always in the same directory as run_quality.py)
        runner_path = str(_VIZ_DIR / "lmeval_sidecar_runner.py")

        cmd = [
            sys.executable, runner_path,
            "--model", "sidecar-chat-completions",
            "--model_args", model_args,
            "--apply_chat_template",
            "--tasks", task_name,
            "--gen_kwargs", gen_kwargs,
            "--output_path", output_path,
            "--log_samples",
            "--num_fewshot", "0",
        ]

        if task_dir:
            cmd.extend(["--include_path", task_dir])

        return cmd

    # ------------------------------------------------------------------
    # Core run logic
    # ------------------------------------------------------------------

    def run(self) -> int:
        """Execute the full quality run lifecycle.

        Lifecycle order (for live runs):
          1. Read and strict-validate status.json (schema + run_id + suite_id).
          2. Validate normalized run directory identity.
          3. Accept only 'planned' state.
          4. Build command, write command.txt.
          5. Run QualityPreflightGate (while state is 'planned').
          6. On preflight pass: transition planned -> preflight_passed.
          7. Transition preflight_passed -> running.
          8. Execute harness (stdout+stderr -> run.log).
          9. On success: verify evidence, schema-validate completed status, write status,
             then write DONE. If any step fails, return nonzero.
         10. On failure: transition to failed (best-effort), return nonzero.

        Returns an exit code (0 = success, nonzero = failure).
        """
        # --- Dry-run: validate/build/print only; persist nothing ---
        if self.dry_run:
            cmd = self.build_command()
            print("[dry-run] Command:")
            print(" ".join(shlex.quote(a) for a in cmd))
            return 0

        # --- Screen guard ---
        if not self.allow_no_screen and not _is_under_screen():
            print(
                "ERROR: Long quality runs must be launched inside /usr/bin/screen. "
                "Start a screen session first:\n  screen -S quality-run\n"
                "Or pass --allow-no-screen to bypass (tests/special ops only).",
                file=sys.stderr,
            )
            return 2

        # --- Read and strict-validate status.json ---
        try:
            status = self._read_status_strict()
        except Exception as exc:
            print(
                f"[run-quality] FATAL: Cannot read or validate status.json: {exc}",
                file=sys.stderr,
            )
            return 1

        # --- Validate normalized run directory identity ---
        identity_errors = _validate_run_dir_identity(
            run_dir=self.run_dir,
            repo=self._repo,
            adapter_slug=self._model_slug,
            suite_id=self._suite_id,
            benchmark=self.benchmark,
            run_id=self._run_id,
        )
        if identity_errors:
            for err in identity_errors:
                print(f"[run-quality] FATAL: {err}", file=sys.stderr)
            return 1

        # --- Accept only 'planned' state ---
        current_state = status.get("execution_state", "")
        if current_state != "planned":
            print(
                f"[run-quality] FATAL: Run directory is in state {current_state!r}; "
                "expected 'planned'.",
                file=sys.stderr,
            )
            return 1

        # --- Build command and write command.txt ---
        cmd = self.build_command()
        self._write_command_txt(cmd)

        # --- Run preflight gate while status is 'planned' ---
        model_id = (self._adapter.get("model") or {}).get("id", "")
        generation_ceiling: int = self._bench_cfg.get("generation_ceiling", 8192)

        preflight_rc = self._run_preflight(
            model_id=model_id,
            generation_ceiling=generation_ceiling,
        )
        if preflight_rc != 0:
            print(
                f"[run-quality] Preflight gate failed (exit {preflight_rc}).",
                file=sys.stderr,
            )
            return preflight_rc

        # --- Transition planned -> preflight_passed ---
        try:
            status = self._read_status_strict()
            new_status = campaign_state.apply_transition(
                status=status,
                new_state="preflight_passed",
                timestamp=_utcnow(),
            )
            self._write_status(new_status)
        except Exception as exc:
            print(
                f"[run-quality] FATAL: Lifecycle transition to 'preflight_passed' failed: {exc}",
                file=sys.stderr,
            )
            return 1

        # --- Transition preflight_passed -> running ---
        try:
            status = self._read_status_strict()
            new_status = campaign_state.apply_transition(
                status=status,
                new_state="running",
                timestamp=_utcnow(),
            )
            self._write_status(new_status)
        except Exception as exc:
            print(
                f"[run-quality] FATAL: Lifecycle transition to 'running' failed: {exc}",
                file=sys.stderr,
            )
            return 1

        # --- Execute harness (stdout+stderr captured to run.log) ---
        harness_rc = self._execute_harness(cmd)

        # --- Post-run: normalize exact lm-eval 0.4.12 output, then verify evidence ---
        if harness_rc == 0:
            try:
                _normalize_lmeval_samples(self.run_dir / "raw")
            except Exception as exc:
                print(
                    f"[run-quality] FATAL: Cannot normalize lm-eval sample evidence: {exc}. "
                    "DONE not written.",
                    file=sys.stderr,
                )
                if not self._transition_status_failed(
                    harness_rc=0, reason="sample_normalization_failed"
                ):
                    return 1
                return 1
            evidence_errors = _verify_required_evidence(self.run_dir)
            if not evidence_errors:
                # --- Sidecar reconciliation gate (fail closed) ---
                # The sidecar MUST exist and reconcile cleanly before DONE is written.
                sidecar_path = self.run_dir / "raw" / "response_metadata.jsonl"
                sidecar_rc = _verify_and_reconcile_sidecar(self.run_dir, sidecar_path)
                if sidecar_rc != 0:
                    if not self._transition_status_failed(
                        harness_rc=0, reason="sidecar_reconciliation_failed"
                    ):
                        return 1
                    return 1

                # --- Derive per-item evidence and write per_item.csv ---
                # Fail closed: if derivation fails, do NOT proceed to DONE.
                # Uses sidecar metadata for finish_reason and disposition.
                try:
                    per_item_data = _derive_per_item_data(self.run_dir,
                                                          sidecar_path=sidecar_path)
                except Exception as exc:
                    print(
                        f"[run-quality] FATAL: Cannot derive per-item evidence: {exc}. "
                        "DONE not written.",
                        file=sys.stderr,
                    )
                    if not self._transition_status_failed(
                        harness_rc=0, reason="per_item_derivation_failed"
                    ):
                        return 1
                    return 1

                try:
                    _write_per_item_csv(self.run_dir, per_item_data)
                except Exception as exc:
                    print(
                        f"[run-quality] FATAL: Cannot write per_item.csv: {exc}. "
                        "DONE not written.",
                        file=sys.stderr,
                    )
                    if not self._transition_status_failed(
                        harness_rc=0, reason="per_item_csv_write_failed"
                    ):
                        return 1
                    return 1

                # --- Completion atomicity: schema-validate completed status, write it ---
                rc = self._transition_status_completed_atomic()
                if rc != 0:
                    return rc

                # --- Update manifest durably before writing DONE ---
                # Fail closed: if manifest update fails, return nonzero and
                # do NOT write DONE (status is already completed, but DONE
                # is never written — caller knows to re-inspect).
                completed_utc = _utcnow()
                submitted_count = len(per_item_data)
                try:
                    _update_manifest_on_completion(
                        self.run_dir,
                        submitted_count=submitted_count,
                        completed_utc=completed_utc,
                    )
                except Exception as exc:
                    print(
                        f"[run-quality] FATAL: Cannot update manifest.json: {exc}. "
                        "DONE not written.",
                        file=sys.stderr,
                    )
                    return 1

                # --- Write DONE sentinel (last — manifest is already durable) ---
                try:
                    (self.run_dir / "DONE").write_text("completed\n")
                except OSError as exc:
                    print(
                        f"[run-quality] FATAL: Cannot write DONE sentinel: {exc}",
                        file=sys.stderr,
                    )
                    return 1
                return 0
            else:
                print(
                    "ERROR: Harness exited 0 but required evidence is missing:\n"
                    + "\n".join(f"  {e}" for e in evidence_errors)
                    + "\nDONE not written.",
                    file=sys.stderr,
                )
                if not self._transition_status_failed(harness_rc=0, reason="evidence_missing"):
                    return 1
                return 1
        else:
            if not self._transition_status_failed(harness_rc=harness_rc):
                return 1
            return harness_rc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_command_txt(self, cmd: List[str]) -> None:
        """Write shell-safe command to command.txt in the run directory."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        shell_line = " ".join(shlex.quote(a) for a in cmd)
        (self.run_dir / "command.txt").write_text(shell_line + "\n", encoding="utf-8")

    def _run_preflight(self, model_id: str, generation_ceiling: int) -> int:
        """Run the QualityPreflightGate; return its exit code."""
        if self._preflight_runner is not None:
            return int(self._preflight_runner(model_id))

        try:
            gate = QualityPreflightGate(
                endpoint=self.endpoint,
                model=model_id,
                max_gen_toks=generation_ceiling,
                aggregate_tok_s=self.throughput,
                concurrency=self.concurrency,
                client_timeout_s=self.timeout,
            )
            return int(gate.run())
        except Exception as exc:
            print(f"[run-quality] Preflight raised: {exc}", file=sys.stderr)
            return int(ExitCode.INCONCLUSIVE)

    def _execute_harness(self, cmd: List[str]) -> int:
        """Execute the lm-eval harness; return its exit code.

        Default: captures stdout+stderr to run.log in the run directory.
        Injected harness_runner receives (cmd, **kw) and must return an int exit code.
        """
        if self._harness_runner is not None:
            return int(self._harness_runner(cmd))

        # Default subprocess execution: redirect stdout+stderr to run.log
        run_log = self.run_dir / "run.log"
        with open(run_log, "w", encoding="utf-8") as log_fh:
            result = subprocess.run(cmd, stdout=log_fh, stderr=log_fh, check=False)
        return result.returncode

    def _read_status_strict(self) -> dict:
        """Read and return status.json; raise if missing, unreadable, or invalid.

        Validates:
          1. status.json exists and is valid JSON.
          2. The document passes the result-status.schema.json JSON schema.
          3. status.run_id == run_dir.name (last path component).
          4. status.suite_id == the loaded suite's suite_id.

        Raises FileNotFoundError if status.json does not exist.
        Raises ValueError if any validation fails.
        Never adds missing identity fields — fails closed.
        """
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

        # Schema validation (uses real repo schema always)
        if _STATUS_SCHEMA.exists():
            schema_errors = validate_json(status, _STATUS_SCHEMA)
            if schema_errors:
                raise ValueError(
                    f"status.json in {self.run_dir} fails schema validation:\n"
                    + "\n".join(schema_errors)
                )
        else:
            raise ValueError(
                f"result-status schema not found at {_STATUS_SCHEMA}"
            )

        # run_id must match run_dir.name — fail closed, never add missing identity
        expected_run_id = self.run_dir.name
        actual_run_id = status.get("run_id", "")
        if actual_run_id != expected_run_id:
            raise ValueError(
                f"status.json run_id={actual_run_id!r} does not match "
                f"run_dir.name={expected_run_id!r}."
            )

        # suite_id must match loaded suite
        expected_suite_id = self._suite_id
        actual_suite_id = status.get("suite_id", "")
        if actual_suite_id != expected_suite_id:
            raise ValueError(
                f"status.json suite_id={actual_suite_id!r} does not match "
                f"loaded suite suite_id={expected_suite_id!r}."
            )

        return status

    def _write_status(self, status: dict) -> None:
        campaign_state.write_status(
            dest=self.run_dir / "status.json",
            status=status,
            run_dir=self.run_dir,
        )

    def _build_completed_status(self, status: dict) -> dict:
        """Build and schema-validate a completed status dict.

        Raises ValueError if schema validation fails (atomicity guard).
        """
        new_status = campaign_state.apply_transition(
            status=status,
            new_state="completed",
            timestamp=_utcnow(),
        )

        # Schema-validate before writing (atomicity)
        if _STATUS_SCHEMA.exists():
            schema_errors = validate_json(new_status, _STATUS_SCHEMA)
            if schema_errors:
                raise ValueError(
                    f"Completed status failed schema validation:\n"
                    + "\n".join(schema_errors)
                )
        return new_status

    def _transition_status_completed_atomic(self) -> int:
        """Apply completed transition atomically: validate schema then write.

        Returns 0 on success, 1 on failure.
        Never returns 0 if the status write fails.
        """
        try:
            status = self._read_status_strict()
            completed_status = self._build_completed_status(status)
            self._write_status(completed_status)
            return 0
        except Exception as exc:
            print(
                f"[run-quality] FATAL: Completed status write failed: {exc}. "
                "DONE not written — returning nonzero.",
                file=sys.stderr,
            )
            return 1

    def _transition_status_failed(
        self, harness_rc: int, reason: Optional[str] = None
    ) -> bool:
        """Transition to failed; return whether the durable state write succeeded."""
        try:
            status = self._read_status_strict()
            ts = _utcnow()
            new_status = campaign_state.apply_transition(
                status=status,
                new_state="failed",
                timestamp=ts,
                lifecycle="invalid",
            )
            # Record the harness exit code in the last history entry.
            # Schema allows exit_code and note in history entries.
            # Does NOT add top-level harness_exit_code (schema forbids it).
            if new_status["history"]:
                last_entry = new_status["history"][-1]
                last_entry["exit_code"] = harness_rc
                if reason:
                    last_entry["note"] = reason
            self._write_status(new_status)
            return True
        except Exception as exc:
            print(
                f"[run-quality] FATAL: status transition to 'failed' failed: {exc}",
                file=sys.stderr,
            )
            return False


# ---------------------------------------------------------------------------
# CLI prompt-tokens parser
# ---------------------------------------------------------------------------


def _parse_prompt_tokens(value: str) -> Dict[str, int]:
    """Parse --prompt-tokens: 'gsm8k=500,ifeval=2000' -> {'gsm8k': 500, 'ifeval': 2000}."""
    result: Dict[str, int] = {}
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise argparse.ArgumentTypeError(
                f"Invalid --prompt-tokens format {part!r}. "
                "Expected: benchmark=tokens (e.g. gsm8k=500,ifeval=2000)"
            )
        bench, _, tok_str = part.partition("=")
        bench = bench.strip()
        tok_str = tok_str.strip()
        try:
            tokens = int(tok_str)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid token count {tok_str!r} for benchmark {bench!r}."
            )
        if tokens < 0:
            raise argparse.ArgumentTypeError(
                f"Token count must be non-negative for benchmark {bench!r}; got {tokens}."
            )
        result[bench] = tokens
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Contract-aware quality runner for warpcore-v1 benchmarks."
        )
    )

    ap.add_argument("--suite", required=True, type=pathlib.Path)
    ap.add_argument("--adapter", required=True, type=pathlib.Path)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--throughput", required=True, type=float)
    ap.add_argument("--concurrency", required=True, type=int)
    ap.add_argument("--timeout", required=True, type=float)
    ap.add_argument("--prompt-tokens", type=str, default=None,
                    help="Measured prompt maxima: 'bench=N,...' (e.g. gsm8k=500,ifeval=2000).")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--run-dir", type=pathlib.Path, default=None)
    ap.add_argument("--repo", type=pathlib.Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-no-screen", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="Resume an existing campaign run directory.")

    # Detect and reject forbidden override flags
    if argv is not None:
        args_to_check = argv
    else:
        args_to_check = sys.argv[1:]

    for flag in FORBIDDEN_OVERRIDE_FLAGS:
        if flag in args_to_check:
            ap.error(
                f"'{flag}' is a suite-owned experiment variable and cannot be overridden."
            )

    args = ap.parse_args(argv)

    # Parse prompt_token_maxima
    prompt_token_maxima: Optional[Dict[str, int]] = None
    if args.prompt_tokens is not None:
        try:
            prompt_token_maxima = _parse_prompt_tokens(args.prompt_tokens)
        except argparse.ArgumentTypeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 3

    # Resolve repository root
    suite_path = pathlib.Path(args.suite).resolve()
    repo = pathlib.Path(args.repo).resolve() if args.repo else suite_path.parent.parent

    # Resolve run directory. Live execution always passes through
    # create_campaign(..., resume=...), including an explicit --run-dir; this
    # prevents hand-built or stale manifests from bypassing exact identity
    # validation. Dry-run only derives/validates a path and performs no writes.
    if args.run_dir is not None and args.dry_run:
        run_dir = pathlib.Path(args.run_dir).resolve()
        try:
            run_dir.relative_to(repo)
        except ValueError:
            print(
                f"ERROR: --run-dir {run_dir} resolves outside repository {repo}.",
                file=sys.stderr,
            )
            return 3
    else:
        import yaml
        with open(suite_path) as fh:
            suite = yaml.safe_load(fh)
        with open(pathlib.Path(args.adapter).resolve()) as fh:
            adapter = yaml.safe_load(fh)

        suite_id = suite.get("suite_id", "warpcore-v1")
        model_slug = (adapter.get("model") or {}).get("slug", "unknown")
        if args.run_dir is not None:
            requested_run_dir = pathlib.Path(args.run_dir).resolve()
            try:
                requested_run_dir.relative_to(repo)
            except ValueError:
                print(
                    f"ERROR: --run-dir {requested_run_dir} resolves outside repository {repo}.",
                    file=sys.stderr,
                )
                return 3
            run_id = requested_run_dir.name
            expected = (
                repo / "results" / model_slug / "runs"
                / suite_id / args.benchmark / run_id
            ).resolve()
            if requested_run_dir != expected:
                print(
                    f"ERROR: --run-dir must equal normalized campaign path {expected}; "
                    f"got {requested_run_dir}.",
                    file=sys.stderr,
                )
                return 3
        else:
            run_id = args.run_id or datetime.now(tz=timezone.utc).strftime("run-%Y-%m-%dT%H-%M-%S")

        if args.dry_run:
            # Dry-run derives the path only. It must not create campaign state,
            # command files, or even parent directories.
            run_dir = (
                repo / "results" / model_slug / "runs"
                / suite_id / args.benchmark / run_id
            )
            try:
                run_dir.relative_to(repo)
            except ValueError:
                print(
                    f"ERROR: Derived run directory {run_dir} would be outside repository {repo}.",
                    file=sys.stderr,
                )
                return 3
        else:
            try:
                import create_campaign as cc_mod
                run_dir = cc_mod.create_campaign(
                    repo=repo,
                    suite_path=suite_path,
                    adapter_path=pathlib.Path(args.adapter).resolve(),
                    benchmark=args.benchmark,
                    run_id=run_id,
                    resume=(args.resume or args.run_dir is not None),
                    prompt_token_maxima=prompt_token_maxima,
                )
            except Exception as exc:
                print(f"ERROR: create_campaign failed: {exc}", file=sys.stderr)
                return 3

        # Containment check
        try:
            run_dir.relative_to(repo)
        except ValueError:
            print(
                f"ERROR: Run directory {run_dir} resolves outside repository {repo}.",
                file=sys.stderr,
            )
            return 3

    try:
        runner = QualityRunner(
            suite_path=args.suite,
            adapter_path=args.adapter,
            benchmark=args.benchmark,
            endpoint=args.endpoint,
            throughput=args.throughput,
            concurrency=args.concurrency,
            timeout=args.timeout,
            run_dir=run_dir,
            repo=repo,
            prompt_token_maxima=prompt_token_maxima,
            dry_run=args.dry_run,
            allow_no_screen=args.allow_no_screen,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
