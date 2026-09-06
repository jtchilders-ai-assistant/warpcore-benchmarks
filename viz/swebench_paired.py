#!/usr/bin/env python3
"""Paired head-to-head SWE-bench comparisons (McNemar) from committed artifacts.

Why this exists
---------------
Every model in this repo ran the **same seed-42 n=100 shuffle**, so any two of
them are a *paired* design: the same 100 problems, two attempts each. Comparing
two paired proportions by looking at whether their confidence intervals overlap
is a well-known fallacy (Schenker & Gentleman 2001) -- it is both the wrong test
and needlessly conservative. The correct test is McNemar's, computed on the
discordant pairs only.

The repo previously published "Ornith and Laguna are effectively tied
(73.0% vs 73.3%)". That compared `73/100` against `55/75` -- two rates measured
on **different populations**. Laguna's 25 infra-excluded instances are not a
random subset: Ornith solved 13 of them. Dropping them from Laguna's denominator
but not from Ornith's flattered Laguna, and the "tie" was an artifact of that
mismatch. Paired on identical instances Ornith leads on BOTH framings:

    all 100 shared      ornith 73.0% vs laguna 55.0%   p = 0.0015  (significant)
    laguna-fair 75      ornith 80.0% vs laguna 73.3%   p = 0.25    (inconclusive)

Neither is a tie. "Not statistically distinguishable on the fair subset" is a
statement about insufficient evidence, not about equality.

Method
------
* Instance sets are intersected and asserted identical before comparing.
* b = A-only resolves, c = B-only resolves (the discordant pairs).
* Reported: uncorrected chi2, Yates-corrected chi2, and the **exact** binomial
  test. The exact test is the one to quote at these counts; the chi2 forms are
  emitted so a reader reproducing by hand can tell which convention a number
  came from. Naming the convention matters -- footnote 5's "chi2=1.2" is the
  Yates value and reads as an error against the uncorrected 1.69.
* A `subset` comparison re-runs the pairing on the instances left after one
  model's infrastructure exclusions, which is the only like-for-like way to ask
  "on the problems where its parser worked, how did it do?".

Writes viz/data/swebench_paired.json. Usage: python3 viz/swebench_paired.py
"""
from __future__ import annotations

import json
import math

import yaml

from common import DATA, REPO, SWEBENCH_RESULTS
from swebench_fair import EXIT_STATUSES, INFRA_STATUSES, load_statuses

# Pairs worth publishing: (a, b, optional model whose infra exclusions define
# a like-for-like subset). Kept explicit rather than all-pairs so the output
# stays readable and every entry is one the README actually cites.
PAIRS = [
    ("ornith-35b", "laguna-s-2.1-118b", "laguna-s-2.1-118b"),
    ("nemotron-3.5-lightning-30b", "qwen3.6-35b-a3b", "qwen3.6-35b-a3b"),
    ("ornith-35b", "nemotron-3.5-lightning-30b", None),
]


def _norm_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def mcnemar(b: int, c: int) -> dict:
    """Paired test on the discordant pairs. b/c = A-only / B-only successes."""
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "discordant": 0, "note": "no discordant pairs"}
    chi2 = (b - c) ** 2 / n
    chi2_yates = (abs(b - c) - 1) ** 2 / n if abs(b - c) >= 1 else 0.0
    # Exact two-sided binomial: P(X <= min(b,c)) under X ~ Bin(n, 0.5), doubled.
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2 ** n)
    return {
        "b": b,
        "c": c,
        "discordant": n,
        "chi2_uncorrected": round(chi2, 3),
        "p_uncorrected": round(math.erfc(math.sqrt(chi2 / 2)), 4),
        "chi2_yates": round(chi2_yates, 3),
        "p_yates": round(math.erfc(math.sqrt(chi2_yates / 2)), 4),
        "p_exact": round(min(1.0, 2 * tail), 4),
    }


def paired_diff_ci(b: int, c: int, n: int, z: float = 1.96) -> list:
    """95% CI for the paired difference in proportions, in percentage points.

    Wald interval on (b-c)/n. Endpoints are rounded to one decimal and NOT
    truncated toward zero -- reporting [-3, +17] for [-3.55, +17.55] understates
    the interval on both sides.
    """
    diff = (b - c) / n
    se = math.sqrt((b + c) / n ** 2)
    return [round(100 * (diff - z * se), 1), round(100 * (diff + z * se), 1)]


def infra_excluded(model: str) -> set:
    """Instances this model was never fairly given: no verdict AND infra cause.

    Mirrors swebench_fair.compute() exactly -- verdict comes from the results
    JSON, exit statuses only attribute a cause.
    """
    if model not in EXIT_STATUSES:
        return set()
    res = json.loads((REPO / SWEBENCH_RESULTS[model]).read_text())
    verdict = set(res["resolved_ids"]) | set(res["unresolved_ids"])
    submitted = set(res["submitted_ids"])
    status_of = load_statuses(EXIT_STATUSES[model])
    return {
        inst for inst, st in status_of.items()
        if inst not in verdict and inst in submitted and st in INFRA_STATUSES
    }


def compare(a: str, b: str, subset_from: str | None) -> dict:
    ra = json.loads((REPO / SWEBENCH_RESULTS[a]).read_text())
    rb = json.loads((REPO / SWEBENCH_RESULTS[b]).read_text())
    sa, sb = set(ra["submitted_ids"]), set(rb["submitted_ids"])
    shared = sa & sb
    if sa != sb:
        raise SystemExit(
            f"{a} and {b} did not run identical instance sets "
            f"({len(sa)} vs {len(sb)}, {len(shared)} shared) -- refusing to "
            "publish a paired comparison across different problems."
        )
    res_a, res_b = set(ra["resolved_ids"]), set(rb["resolved_ids"])

    def stats(pool: set) -> dict:
        n = len(pool)
        only_a = len([i for i in pool if i in res_a and i not in res_b])
        only_b = len([i for i in pool if i in res_b and i not in res_a])
        both = len([i for i in pool if i in res_a and i in res_b])
        out = {
            "n": n,
            "a_resolved": len([i for i in pool if i in res_a]),
            "b_resolved": len([i for i in pool if i in res_b]),
            "both": both,
            "neither": n - both - only_a - only_b,
            "mcnemar": mcnemar(only_a, only_b),
            "paired_diff_pp": round(100 * (only_a - only_b) / n, 1),
            "paired_diff_ci95_pp": paired_diff_ci(only_a, only_b, n),
        }
        out["a_pct"] = round(100 * out["a_resolved"] / n, 1)
        out["b_pct"] = round(100 * out["b_resolved"] / n, 1)
        return out

    entry = {"model_a": a, "model_b": b, "full": stats(shared)}
    if subset_from:
        dropped = infra_excluded(subset_from)
        pool = shared - dropped
        entry["subset"] = stats(pool)
        entry["subset"]["defined_by"] = subset_from
        entry["subset"]["dropped"] = len(dropped)
        # The headline honesty check: were the dropped instances actually hard?
        entry["subset"]["a_solved_among_dropped"] = len(
            [i for i in dropped if i in res_a]
        )
        entry["subset"]["b_solved_among_dropped"] = len(
            [i for i in dropped if i in res_b]
        )
    return entry


def main() -> None:
    out = {}
    for a, b, subset_from in PAIRS:
        out[f"{a}__vs__{b}"] = compare(a, b, subset_from)
    path = DATA / "swebench_paired.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    for key, e in out.items():
        f = e["full"]
        print(f"{key}")
        print(
            f"  full n={f['n']:3d}  {e['model_a']} {f['a_pct']}%  vs  "
            f"{e['model_b']} {f['b_pct']}%   "
            f"b={f['mcnemar']['b']} c={f['mcnemar']['c']} "
            f"exact p={f['mcnemar']['p_exact']}"
        )
        if "subset" in e:
            s = e["subset"]
            print(
                f"  fair n={s['n']:3d}  {e['model_a']} {s['a_pct']}%  vs  "
                f"{e['model_b']} {s['b_pct']}%   "
                f"b={s['mcnemar']['b']} c={s['mcnemar']['c']} "
                f"exact p={s['mcnemar']['p_exact']}   "
                f"({s['dropped']} dropped for {s['defined_by']} infra; "
                f"{e['model_a']} solved {s['a_solved_among_dropped']} of them)"
            )
    print(f"\nwrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
