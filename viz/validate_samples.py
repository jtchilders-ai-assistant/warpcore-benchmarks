"""Detect silent scoring failures in lm-eval sample files.

ISSUES #15: with `--reasoning-parser`, vLLM can emit `content: null` and put the
answer in `message.reasoning`. lm-eval reads only `content`, finds an empty
string, and scores the item 0 -- no error, no warning, no retry. The run looks
clean and the published number is wrong. Measured empty rates on retained
samples: Nemotron-3-Super GPQA-D 28.3%, Ornith GPQA-D 21.2%, Laguna GSM8K 14.1%,
Lightning IFEval 8.7%.

This is invisible in results_*.json, which carries only aggregates. It is only
detectable per item, which is why samples must be retained (see --emit-per-item).

An empty response is NOT a wrong answer -- it is a MISSING measurement. Scoring
it 0 understates the model; excluding it overstates the model (on Laguna the
recovered items scored 90.3% vs 97.09% for served items, so dropped items are
harder -- served-only rates are an optimistic upper bound, not a fix).

  validate_samples.py                          # audit every committed sample file
  validate_samples.py --max-empty-rate 0.02    # CI gate (default 2%)
  validate_samples.py --emit-per-item          # write slim per_item.csv next to each
  validate_samples.py PATH [PATH ...]          # audit specific files

Exit 0 = every task under the threshold. Exit 1 = at least one over.
Exit 0 with a warning if no sample files exist at all -- absence of evidence is
reported loudly but does not fail a repo that has not yet backfilled.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

# lm-eval writes one line per (doc_id, filter). GPQA runs two filters
# (answer-line + flexible-fallback), so per-LINE rates double-count items.
# Empty-ness is a property of the generation, not the filter -> dedupe by doc_id.


def open_maybe_gz(path: pathlib.Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def response_text(rec: dict) -> str:
    """The raw model generation for this item, as lm-eval stored it."""
    resps = rec.get("resps") or []
    if resps and isinstance(resps[0], list) and resps[0]:
        return resps[0][0] or ""
    if resps and isinstance(resps[0], str):
        return resps[0]
    # Fall back to the post-filter view if the raw one is absent.
    filt = rec.get("filtered_resps") or []
    if filt and isinstance(filt[0], str):
        return filt[0]
    return ""


def score_of(rec: dict):
    for key in ("exact_match", "acc", "prompt_level_strict_acc", "acc_norm"):
        if key in rec:
            try:
                return float(rec[key])
            except (TypeError, ValueError):
                pass
    return None


def audit_per_item_csv(path: pathlib.Path) -> dict:
    """Audit a committed per_item.csv (the git-visible fallback).

    samples_*.jsonl is gitignored, so on a clean checkout the raw samples are
    usually ABSENT even though the defect they document is real. Reading the
    slim CSV keeps CI honest: without this, CI audits only the handful of
    committed sample files and reports a falsely clean bill of health.
    """
    items: dict = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                doc_id = int(row["doc_id"])
            except (KeyError, ValueError):
                continue
            score = row.get("score", "")
            items[doc_id] = {
                "text": "" if row.get("empty_content") == "1" else "x",
                "score": float(score) if score not in ("", None) else None,
            }
    return summarize(path, items)


def summarize(path: pathlib.Path, items: dict) -> dict:
    total = len(items)
    empty_ids = sorted(d for d, v in items.items() if not v["text"].strip())
    scored = [v["score"] for v in items.values() if v["score"] is not None]
    served = [v["score"] for d, v in items.items()
              if d not in set(empty_ids) and v["score"] is not None]

    return {
        "path": path,
        "total": total,
        "empty": len(empty_ids),
        "empty_rate": (len(empty_ids) / total) if total else 0.0,
        "empty_doc_ids": empty_ids,
        "published": (sum(scored) / len(scored)) if scored else None,
        "served_only": (sum(served) / len(served)) if served else None,
        "items": items,
    }


def audit_file(path: pathlib.Path) -> dict:
    """Per-item audit of a raw lm-eval sample file, deduped by doc_id."""
    if path.suffix == ".csv":
        return audit_per_item_csv(path)

    items: dict = {}
    for line in open_maybe_gz(path):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        doc_id = rec.get("doc_id")
        text = response_text(rec)
        prev = items.get(doc_id)
        # Keep the non-empty view if ANY filter saw output for this item.
        if prev is None or (not prev["text"].strip() and text.strip()):
            items[doc_id] = {"text": text, "score": score_of(rec)}
        elif prev.get("score") in (None, 0.0) and score_of(rec):
            prev["score"] = score_of(rec)

    return summarize(path, items)


def write_per_item(path: pathlib.Path, res: dict) -> pathlib.Path:
    """Slim, git-friendly audit record (~50KB vs a 5MB JSONL).

    samples_*.jsonl is gitignored, so for 5 of 7 models the empty-response
    defect is currently unauditable and their scores can never be verified.
    This keeps every audit-relevant field at a size nobody objects to.
    """
    out = path.parent / (path.name.split(".jsonl")[0] + ".per_item.csv")
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["doc_id", "empty_content", "score", "response_chars"])
        for doc_id in sorted(res["items"], key=lambda d: (d is None, d)):
            v = res["items"][doc_id]
            w.writerow([doc_id, int(not v["text"].strip()),
                        "" if v["score"] is None else v["score"], len(v["text"])])
    return out


def discover() -> list:
    """Every auditable task, preferring raw samples over the slim CSV.

    Both may exist locally; only the CSV survives a clean checkout, because
    .gitignore excludes samples_*.jsonl. Keyed by task so the same run is not
    counted twice when both are present.
    """
    by_task: dict = {}
    for p in sorted(set(RESULTS.rglob("samples_*.jsonl"))
                    | set(RESULTS.rglob("samples_*.jsonl.gz"))):
        by_task[(p.parent, p.name.split(".jsonl")[0])] = p
    for p in sorted(RESULTS.rglob("*.per_item.csv")):
        by_task.setdefault((p.parent, p.name[:-len(".per_item.csv")]), p)
    return [by_task[k] for k in sorted(by_task, key=lambda k: (str(k[0]), k[1]))]


def rel(path: pathlib.Path) -> str:
    """Display path, tolerant of args given as relative or outside results/."""
    p = path.resolve()
    for base in (RESULTS, REPO):
        try:
            return str(p.relative_to(base))
        except ValueError:
            continue
    return str(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", type=pathlib.Path,
                    help="sample files to audit (default: all under results/)")
    ap.add_argument("--max-empty-rate", type=float, default=0.02,
                    help="fail above this empty-response fraction (default 0.02)")
    ap.add_argument("--emit-per-item", action="store_true",
                    help="write a slim per_item.csv beside each sample file")
    ap.add_argument("--warn-only", action="store_true",
                    help="report but always exit 0")
    args = ap.parse_args(argv)

    if args.paths:
        files = sorted(args.paths)
    else:
        files = discover()

    if not files:
        print("WARNING: no samples_*.jsonl found under results/.")
        print("  samples_*.jsonl is gitignored, so silent-zero defects "
              "(ISSUES #15) cannot be audited for any model.")
        print("  Retain samples, or commit per_item.csv (--emit-per-item).")
        return 0

    print(f"{'empty':>12}  {'rate':>7}  {'published':>9}  {'served':>7}  task")
    print("-" * 96)

    failures = []
    for path in files:
        res = audit_file(path)
        rate = res["empty_rate"]
        flag = "FAIL" if rate > args.max_empty_rate else "ok"
        pub = f"{100*res['published']:.2f}%" if res["published"] is not None else "n/a"
        srv = f"{100*res['served_only']:.2f}%" if res["served_only"] is not None else "n/a"
        print(f"{res['empty']:>5}/{res['total']:<6} {100*rate:6.1f}%  "
              f"{pub:>9}  {srv:>7}  [{flag}] {rel(path)}")
        if args.emit_per_item:
            print(f"{'':>14}-> {rel(write_per_item(path, res))}")
        if rate > args.max_empty_rate:
            failures.append(res)

    if failures:
        print(f"\nFAIL: {len(failures)} task(s) exceed "
              f"{100*args.max_empty_rate:.0f}% empty responses.\n")
        for res in failures:
            print(f"  {rel(res['path'])}")
            print(f"    {res['empty']}/{res['total']} items returned NO content "
                  f"and were scored 0.")
            if res["published"] is not None and res["served_only"] is not None:
                print(f"    published {100*res['published']:.2f}%  ->  "
                      f"{100*res['served_only']:.2f}% on served items only "
                      f"(an UPPER BOUND, not a corrected score)")
            head = ", ".join(str(d) for d in res["empty_doc_ids"][:10])
            more = "" if len(res["empty_doc_ids"]) <= 10 else \
                   f" (+{len(res['empty_doc_ids']) - 10} more)"
            print(f"    empty doc_ids: {head}{more}")
        print("\nThis is ISSUES #15. Re-serve the empty items; do not publish "
              "either the raw score or the served-only rate as a capability number.")
        if args.warn_only:
            print("[warn-only] exiting 0 by request.")
            return 0
        return 1

    print(f"\nOK: all {len(files)} task(s) under "
          f"{100*args.max_empty_rate:.0f}% empty responses.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
