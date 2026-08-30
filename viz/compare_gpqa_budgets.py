#!/usr/bin/env python3
"""Compare the Laguna GPQA 32k and 64k runs from committed artifacts.

Reproduces every number published in the Laguna card's GPQA section and in
top-level README footnote 11, from the two committed samples files alone.

    python3 viz/compare_gpqa_budgets.py \
        results/laguna-s-2.1-118b/raw/quality/gpqa/samples_gpqa_diamond_cot_zeroshot_clean.jsonl.gz \
        results/laguna-s-2.1-118b/raw/quality/gpqa/samples_gpqa_diamond_cot_zeroshot_clean_64k.jsonl.gz

Exit codes: 0 = the two runs are statistically indistinguishable (the finding),
1 = they differ significantly, 2 = inputs unusable.

WHY THIS EXISTS
A doubled output budget was expected to lift the score (the Lightning
precedent: 16k 53.03% -> 64k 76.26%). It did not. This script is the
falsifiable check on that claim: it counts, per item, whether the generation
terminated on its own or ran into the cap, and pairs the two runs item-by-item
so the comparison is not confounded by which questions happened to be answered.

PITFALL (cost a wrong public claim once, 2026-08-28): do NOT estimate token
counts from character length. Measured chars/token on this model ranges
2.05-3.98 depending on content (math and code tokenize far denser than prose).
An earlier analysis assumed a single 5.25 ratio derived from one max-length
sample, concluded generations sat comfortably below the ceiling, and reported
"0 items pinned at the cap" when the true figure was 104/198. Character length
is a proxy for nothing here. Either tokenize, or -- as this script does by
default -- rely on the exact per-item counts in token_census_32k_vs_64k.json,
which were produced with the model's own tokenizer.
"""
import argparse
import gzip
import json
import math
import re
import sys

ANCHOR = re.compile(r"[Tt]he answer is")


def load(path):
    """doc_id -> {raw, filters}. Handles one row per (doc, filter)."""
    opener = gzip.open if path.endswith(".gz") else open
    by = {}
    with opener(path, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            d = r["doc_id"]
            e = by.setdefault(d, {"raw": None, "f": {}})
            resps = r.get("resps") or []
            flat = [x for sub in resps
                    for x in (sub if isinstance(sub, list) else [sub])]
            if flat and e["raw"] is None:
                e["raw"] = flat[0]
            e["f"][r.get("filter")] = {
                "em": r.get("exact_match"),
                "fr": (r.get("filtered_resps") or [None])[0],
            }
    return by


def correct(entry, filt="answer-line"):
    v = (entry["f"].get(filt) or {}).get("em")
    return 1 if (v is not None and float(v) > 0) else 0


def anchored(entry):
    return bool(entry["raw"]) and bool(ANCHOR.search(entry["raw"]))


def parsed(entry, filt="answer-line"):
    """lm-eval extracted a real letter rather than [invalid]."""
    fr = (entry["f"].get(filt) or {}).get("fr")
    return fr is not None and str(fr).strip().upper() in ("A", "B", "C", "D")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return ((ctr - hw) * 100, (ctr + hw) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples_a", help="32k samples (.jsonl or .jsonl.gz)")
    ap.add_argument("samples_b", help="64k samples (.jsonl or .jsonl.gz)")
    ap.add_argument("--label-a", default="32k")
    ap.add_argument("--label-b", default="64k")
    ap.add_argument("--census", default=None,
                    help="token_census_32k_vs_64k.json for exact at-cap counts")
    ap.add_argument("--json", action="store_true")
    a_args = ap.parse_args()

    try:
        A = load(a_args.samples_a)
        B = load(a_args.samples_b)
    except Exception as exc:                       # noqa: BLE001
        print(f"ERROR: could not read samples: {exc}", file=sys.stderr)
        return 2

    docs = sorted(set(A) & set(B))
    n = len(docs)
    if n == 0:
        print("ERROR: no shared doc_ids between the two runs", file=sys.stderr)
        return 2

    out = {"n": n, "runs": {}}
    for lbl, S in ((a_args.label_a, A), (a_args.label_b, B)):
        cor = sum(correct(S[d]) for d in docs)
        anc = sum(anchored(S[d]) for d in docs)
        par = sum(parsed(S[d]) for d in docs)
        lo, hi = wilson(cor, n)
        out["runs"][lbl] = {
            "correct": cor,
            "overall_pct": round(cor / n * 100, 2),
            "ci95": [round(lo, 1), round(hi, 1)],
            "raw_anchor": anc,
            "lm_eval_parsed": par,
            "interior_pct_vs_parsed": round(cor / par * 100, 2) if par else None,
            "interior_pct_vs_anchor": round(cor / anc * 100, 2) if anc else None,
        }

    both = sum(1 for d in docs if correct(A[d]) and correct(B[d]))
    only_a = sum(1 for d in docs if correct(A[d]) and not correct(B[d]))
    only_b = sum(1 for d in docs if correct(B[d]) and not correct(A[d]))
    diff = (only_b - only_a) / n * 100

    paired = {"both": both, f"only_{a_args.label_a}": only_a,
              f"only_{a_args.label_b}": only_b,
              "neither": n - both - only_a - only_b,
              "diff_pp": round(diff, 2)}
    significant = False
    if only_a + only_b:
        chi = (abs(only_a - only_b) - 1) ** 2 / (only_a + only_b)
        p = math.erfc(math.sqrt(chi / 2))
        se = math.sqrt(only_a + only_b) / n * 100
        paired.update({
            "mcnemar_chi2": round(chi, 3),
            "p_value": round(p, 4),
            "diff_ci95_pp": [round(diff - 1.96 * se, 1), round(diff + 1.96 * se, 1)],
        })
        significant = p <= 0.05
    out["paired"] = paired

    if a_args.census:
        try:
            with open(a_args.census) as fh:
                c = json.load(fh)
            c.pop("both_at_cap_docs", None)
            out["token_census"] = c
        except Exception as exc:                   # noqa: BLE001
            out["token_census_error"] = str(exc)

    if a_args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"items compared: {n}\n")
        for lbl, r in out["runs"].items():
            print(f"=== {lbl} ===")
            print(f"  overall            : {r['correct']}/{n} = "
                  f"{r['overall_pct']}%  95% CI {r['ci95']}")
            print(f"  raw answer anchor  : {r['raw_anchor']}")
            print(f"  lm-eval parsed     : {r['lm_eval_parsed']}"
                  f"  -> interior {r['interior_pct_vs_parsed']}%")
            print()
        print("=== paired (answer-line) ===")
        for k, v in paired.items():
            print(f"  {k:>22} = {v}")
        if "token_census" in out:
            print("\n=== token census (exact, model tokenizer) ===")
            for lbl in (a_args.label_a, a_args.label_b):
                c = out["token_census"].get(lbl)
                if c:
                    print(f"  {lbl}: at cap {c['at_cap']}/{c['n']} "
                          f"({c['at_cap_pct']}%), median {c['tok_p50']:,} tok")
            if "both_at_cap" in out["token_census"]:
                print(f"  at cap in BOTH runs: "
                      f"{out['token_census']['both_at_cap']}")
        print()
        print("VERDICT: " + (
            "runs DIFFER significantly (p<=0.05)" if significant else
            "runs are STATISTICALLY INDISTINGUISHABLE -- doubling the output "
            "budget did not change the score"))
    return 1 if significant else 0


if __name__ == "__main__":
    sys.exit(main())
