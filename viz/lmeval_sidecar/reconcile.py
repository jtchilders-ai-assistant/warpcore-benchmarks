"""viz/lmeval_sidecar/reconcile.py
===================================
Production fail-closed reconciler for lm-eval 0.4.12 response metadata.

Verifies:
 1. Every unique sample request (unique fingerprint) has exactly one metadata record.
 2. No missing records (sample with no metadata).
 3. No foreign records (metadata with no matching sample).
 4. No ambiguous duplicates (same fingerprint but different response content).
 5. Multi-filter rows (N filter rows per request) handled correctly as N:1.
 6. Per-record field validation:
    - finish_reason: nonempty string
    - completion_tokens: nonneg non-bool int
    - content key present (nullable)
    - reasoning and reasoning_content keys present (nullable)

Returns a report dict:
  {
    "exit_code": 0 or 1,
    "errors": [str, ...],
    "matched": int,
    "total_samples": int,
    "total_meta": int,
    "unique_requests": int,
    "classifications": {"scored": N, "budget": N, "parser": N, "empty": N},
  }
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pathlib
from collections import Counter
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Fingerprint (must match capture.py exactly)
# ---------------------------------------------------------------------------

def _fingerprint(messages: object) -> str:
    canonical = json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _fingerprint_from_sample(sample: dict) -> Optional[str]:
    """Extract messages from an lm-eval sample and compute its fingerprint.

    lm-eval 0.4.12 stores the prompt as:
        sample["arguments"]["gen_args_0"]["arg_0"]

    Where arg_0 is a LIST whose first element is a JSON-encoded string
    of the messages list (list[dict]) that was sent to the model.
    """
    try:
        arg0 = sample["arguments"]["gen_args_0"]["arg_0"]
        # arg_0 is always a list; the serialized messages are in arg_0[0]
        if isinstance(arg0, list):
            msgs_str = arg0[0]
        else:
            msgs_str = arg0  # fallback: may be a plain string in some tasks
        messages = json.loads(msgs_str)
        return _fingerprint(messages)
    except (KeyError, TypeError, json.JSONDecodeError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Disposition / content classification
# ---------------------------------------------------------------------------

def classify_response(record: dict) -> str:
    """Classify a metadata record's response disposition.

    Returns one of:
      - "scored"  : nonempty content (model answered)
      - "budget"  : empty/null content + finish_reason == "length" (token budget)
      - "parser"  : empty/null content + stop + reasoning/reasoning_content nonempty
      - "empty"   : empty/null content + stop + no reasoning
    """
    contents = record.get("content", [])
    finish_reasons = record.get("finish_reasons", [])
    reasoning_contents = record.get("reasoning_content", [])
    reasonings = record.get("reasoning", [])

    content = contents[0] if contents else None
    finish_reason = finish_reasons[0] if finish_reasons else None
    rc = reasoning_contents[0] if reasoning_contents else None
    r = reasonings[0] if reasonings else None

    if content and str(content).strip():
        return "scored"

    # content is None or empty string
    if finish_reason == "length":
        return "budget"

    # stop or other finish reason with no content
    has_reasoning = (rc and str(rc).strip()) or (r and str(r).strip())
    if has_reasoning:
        return "parser"

    return "empty"


# ---------------------------------------------------------------------------
# Per-record field validation
# ---------------------------------------------------------------------------

def _validate_record(record: dict) -> List[str]:
    """Validate a single metadata record. Returns list of error strings."""
    errors = []

    # finish_reason nonempty
    finish_reasons = record.get("finish_reasons", [])
    if not finish_reasons:
        errors.append("finish_reasons is empty list")
    else:
        for i, fr in enumerate(finish_reasons):
            if not fr or not str(fr).strip():
                errors.append(f"finish_reasons[{i}] is empty/null")

    # usage and all three integer token counts are mandatory canonical evidence.
    usage = record.get("usage")
    if not isinstance(usage, dict):
        errors.append("usage is missing or is not an object")
    else:
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(field)
            if value is None:
                errors.append(f"usage.{field} is missing")
            elif isinstance(value, bool):
                errors.append(f"usage.{field} is bool ({value!r}), must be int")
            elif not isinstance(value, int):
                errors.append(
                    f"usage.{field} is {type(value).__name__} ({value!r}), must be int"
                )
            elif value < 0:
                errors.append(f"usage.{field} is negative ({value})")

    # content key must be present (nullable)
    if "content" not in record:
        errors.append("content key is missing (must be present, may be null)")

    # reasoning and reasoning_content keys must be present (nullable)
    if "reasoning" not in record:
        errors.append("reasoning key is missing (must be present, may be null)")
    if "reasoning_content" not in record:
        errors.append("reasoning_content key is missing (must be present, may be null)")

    return errors


# ---------------------------------------------------------------------------
# File readers
# ---------------------------------------------------------------------------

def _read_jsonl(path: pathlib.Path) -> List[dict]:
    """Read JSONL or JSONL.GZ file, return list of records."""
    opener = gzip.open if str(path).endswith(".gz") else open
    records = []
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Main reconciliation logic
# ---------------------------------------------------------------------------

def reconcile_inventory(
    sample_paths: List[pathlib.Path],
    metadata_path: pathlib.Path,
) -> dict:
    """Reconcile the complete sample-file inventory with one metadata sidecar.

    report["exit_code"] == 0 means reconciliation passed.
    report["exit_code"] == 1 means one or more failures detected.
    """
    sample_paths = [pathlib.Path(path) for path in sample_paths]
    metadata_path = pathlib.Path(metadata_path)

    errors: List[str] = []

    # --- Load samples ---
    samples = []
    for samples_path in sample_paths:
        try:
            samples.extend(_read_jsonl(samples_path))
        except Exception as exc:
            return {
                "exit_code": 1,
                "errors": [f"Cannot read samples file {samples_path}: {exc}"],
                "matched": 0,
                "total_samples": len(samples),
                "total_meta": 0,
                "unique_requests": 0,
                "classifications": {},
            }

    # --- Load metadata ---
    try:
        metadata = _read_jsonl(metadata_path)
    except Exception as exc:
        return {
            "exit_code": 1,
            "errors": [f"Cannot read metadata file {metadata_path}: {exc}"],
            "matched": 0,
            "total_samples": len(samples),
            "total_meta": 0,
            "unique_requests": 0,
            "classifications": {},
        }

    # --- Build metadata index: fingerprint → list[record] ---
    meta_index: Dict[str, List[dict]] = {}
    for rec in metadata:
        fp = rec.get("fingerprint")
        if fp not in meta_index:
            meta_index[fp] = []
        meta_index[fp].append(rec)

    # --- Per-record field validation (all records) ---
    for i, rec in enumerate(metadata):
        rec_errors = _validate_record(rec)
        for e in rec_errors:
            errors.append(f"  metadata[{i}] fp={str(rec.get('fingerprint', ''))[:12]}…: {e}")

    # --- Compute sample fingerprints (deduplicate by fingerprint+doc_id) ---
    # lm-eval emits N filter rows per request (same fingerprint, same doc_id).
    # We track unique (fingerprint, doc_id) pairs to understand multi-filter rows.
    # For reconciliation: a fingerprint is "needed" once per unique request (unique fp).
    # But if two *different* doc_ids share the same fingerprint (same prompt, different items),
    # that's ambiguous and we must detect it.

    # Map: fingerprint → set of doc_ids
    fp_to_docids: Dict[str, set] = {}
    sample_fps = []

    for s in samples:
        fp = _fingerprint_from_sample(s)
        sample_fps.append(fp)
        if fp is None:
            continue
        doc_id = s.get("doc_id")
        if fp not in fp_to_docids:
            fp_to_docids[fp] = set()
        fp_to_docids[fp].add(doc_id)

    # --- Detect ambiguous duplicates: same fingerprint, multiple doc_ids ---
    for fp, doc_ids in fp_to_docids.items():
        if len(doc_ids) > 1:
            errors.append(
                f"  AMBIGUOUS: fingerprint {fp[:12]}… maps to multiple doc_ids "
                f"{sorted(str(doc_id) for doc_id in doc_ids)}. A request fingerprint "
                "cannot establish one-to-one item provenance for repeated prompts."
            )

    # --- Unique fingerprints in samples ---
    unique_fp_set = {fp for fp in sample_fps if fp is not None}

    # --- Reconcile: every unique fingerprint must have exactly 1 metadata record ---
    classifications: Counter = Counter()
    matched = 0

    for fp in unique_fp_set:
        records_for_fp = meta_index.get(fp, [])
        if len(records_for_fp) == 0:
            errors.append(f"  MISSING: fingerprint {fp[:12]}… has no metadata record")
        elif len(records_for_fp) > 1:
            # Check if content differs (ambiguous) or all same (retries)
            contents = [json.dumps(r.get("content")) for r in records_for_fp]
            unique_contents = set(contents)
            if len(unique_contents) > 1:
                errors.append(
                    f"  DUPLICATE+AMBIGUOUS: fingerprint {fp[:12]}… has "
                    f"{len(records_for_fp)} records with different contents {unique_contents!r}"
                )
            else:
                errors.append(
                    f"  DUPLICATE: fingerprint {fp[:12]}… has {len(records_for_fp)} identical "
                    "metadata records (retry?). Expected exactly 1."
                )
        else:
            cls = classify_response(records_for_fp[0])
            classifications[cls] += 1
            matched += 1

    # --- Foreign records (metadata with no matching sample) ---
    foreign_fps = [fp for fp in meta_index if fp not in unique_fp_set]
    if foreign_fps:
        errors.append(
            f"  FOREIGN: {len(foreign_fps)} metadata record(s) have no matching sample fingerprint"
        )
        for fp in foreign_fps[:3]:
            errors.append(f"    foreign fp={fp[:12]}…")

    # --- Uncomputable fingerprints ---
    bad_samples = sum(1 for fp in sample_fps if fp is None)
    if bad_samples:
        errors.append(
            f"  {bad_samples} sample(s) have no computable fingerprint "
            "(missing arguments.gen_args_0.arg_0)"
        )

    exit_code = 1 if errors else 0

    return {
        "exit_code": exit_code,
        "errors": errors,
        "matched": matched,
        "total_samples": len(samples),
        "total_meta": len(metadata),
        "unique_requests": len(unique_fp_set),
        "classifications": dict(classifications),
    }


def reconcile(samples_path: pathlib.Path, metadata_path: pathlib.Path) -> dict:
    """Backward-compatible one-file reconciliation entry point."""
    return reconcile_inventory([samples_path], metadata_path)
