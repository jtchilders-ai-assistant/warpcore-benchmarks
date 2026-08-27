#!/usr/bin/env python3
"""Classify lm-eval generative outcomes: capability vs truncation artifact.

WHY THIS EXISTS
A GPQA/GSM8K accuracy figure is meaningless without knowing what fraction of
items never emitted a parseable answer. The Lightning card's budget curve
(16k -> 53.03%, 32k -> 66.16%, 64k -> 76.26%) shows truncation is worth ~23
points, so a run with a high no-answer rate is measuring the OUTPUT BUDGET,
not the model. This script makes that distinction mechanical instead of a
judgement call.

It reports three numbers that must always travel together:
  * headline accuracy          (correct / n)  -- what lm-eval prints
  * no-answer rate             (items with no anchored answer line)
  * accuracy among answered    (correct / answered) -- an UPPER BOUND only,
                               conditioned on a non-random (easier) subset

PITFALL THIS SCRIPT ENCODES
`filtered_resps` holds only the EXTRACTED letter (or "[invalid]"), never the
raw text. Diagnosing against it makes 100% of items look answer-less. Always
read `resps` for the generation and `filtered_resps` only for the verdict.

USAGE
    python3 viz/diagnose_truncation.py SAMPLES.jsonl[.gz] [--budget-tokens 32768]
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import defaultdict

ANCHORED = re.compile(r"[Tt]he answer is \(?([A-D])\)?")
CHARS_PER_TOKEN = 3.0  # conservative; only used to flag near-cap generations


def _open(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def _raw_text(d: dict) -> str:
    r = d.get("resps") or d.get("filtered_resps") or [""]
    v = r[0]
    while isinstance(v, list):
        v = v[0] if v else ""
    return v or ""


def _verdict(d: dict) -> str:
    f = d.get("filtered_resps")
    v = f[0] if isinstance(f, list) else f
    while isinstance(v, list):
        v = v[0] if v else ""
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("samples")
    ap.add_argument("--budget-tokens", type=int, default=32768)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    near_cap_chars = a.budget_tokens * CHARS_PER_TOKEN

    rows = [json.loads(l) for l in _open(a.samples) if l.strip()]
    if not rows:
        print("no samples", file=sys.stderr)
        return 2

    # lm-eval writes one row PER FILTER; split so n is the true item count.
    by_filter = defaultdict(list)
    for d in rows:
        by_filter[d.get("filter", "?")].append(d)

    report = {"samples_file": a.samples, "budget_tokens": a.budget_tokens,
              "filters": {}}

    for name, rs in sorted(by_filter.items()):
        n = len(rs)
        correct = sum(1 for d in rs if d.get("exact_match"))
        no_answer = near_cap = invalid = 0
        lens = []
        for d in rs:
            raw = _raw_text(d)
            lens.append(len(raw))
            if "[invalid]" in _verdict(d):
                invalid += 1
            if not ANCHORED.findall(raw):
                no_answer += 1
                if len(raw) > near_cap_chars:
                    near_cap += 1
        lens.sort()
        answered = n - no_answer
        f = {
            "n": n,
            "correct": correct,
            "headline_accuracy_pct": round(100.0 * correct / n, 2),
            "no_answer": no_answer,
            "no_answer_pct": round(100.0 * no_answer / n, 1),
            "no_answer_near_cap": near_cap,
            "invalid_verdicts": invalid,
            "answered": answered,
            "accuracy_among_answered_pct": (
                round(100.0 * correct / answered, 2) if answered else None),
            "gen_chars_p50": lens[n // 2],
            "gen_chars_p90": lens[int(n * 0.9)] if n > 1 else lens[0],
            "gen_chars_max": lens[-1],
        }
        report["filters"][name] = f

    worst = max(x["no_answer_pct"] for x in report["filters"].values())
    report["verdict"] = (
        "TRUNCATION-DOMINATED - headline score measures the output budget, not "
        "capability; re-run at a larger budget before publishing"
        if worst >= 20 else
        "CLEAN - no-answer rate is low; headline score is a capability measurement")

    if a.json:
        print(json.dumps(report, indent=2))
        return 0

    for name, f in report["filters"].items():
        print(f"filter={name}")
        print(f"  headline            {f['correct']}/{f['n']} = "
              f"{f['headline_accuracy_pct']}%")
        print(f"  no answer line      {f['no_answer']}/{f['n']} = "
              f"{f['no_answer_pct']}%  (near cap: {f['no_answer_near_cap']})")
        print(f"  among answered      {f['correct']}/{f['answered']} = "
              f"{f['accuracy_among_answered_pct']}%  <- UPPER BOUND, biased subset")
        print(f"  gen chars p50/p90/max  {f['gen_chars_p50']} / "
              f"{f['gen_chars_p90']} / {f['gen_chars_max']}")
    print()
    print("VERDICT:", report["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
