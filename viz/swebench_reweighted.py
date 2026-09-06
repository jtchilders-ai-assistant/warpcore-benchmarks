#!/usr/bin/env python3
"""Repo-balanced SWE-bench reweighting, with bootstrap uncertainty.

Why this exists
---------------
The seed-42 n=100 shuffle is **django-heavy** (56/100). Models differ in how
much of their score is django, so the sample's composition is itself a
confounder: reweighting each repo equally asks "what if the benchmark were not
django-dominated?".

The README published the point estimates (Laguna 55->62, Ornith 73->66) as bare
integers. They are reproducible, but they rest on per-repo cells of n=4-5, where
a single instance moves a repo's rate by 20-25 pp. Publishing them without an
interval invites the reader to treat a coin-flip as a finding.

This script emits both the point estimate and a bootstrap CI so the uncertainty
travels with the number.

Method
------
* Repos with n >= MIN_N instances are kept (below that a "rate" is noise).
* Balanced estimate = unweighted mean of per-repo resolve rates.
* CI = percentile bootstrap, resampling instances **within each repo** (the
  design is stratified by repo, so the resample must be too), fixed seed for
  reproducibility.

Writes viz/data/swebench_reweighted.json. Usage: python3 viz/swebench_reweighted.py
"""
from __future__ import annotations

import json
import random
from collections import defaultdict

from common import DATA, REPO, SWEBENCH_RESULTS

MIN_N = 4
B = 10000
SEED = 42


def repo_of(instance_id: str) -> str:
    """`django__django-16938` -> `django`. Instance ids are `<org>__<repo>-<n>`."""
    return instance_id.split("__", 1)[0]


def per_repo(model: str) -> dict[str, list[int]]:
    d = json.loads((REPO / SWEBENCH_RESULTS[model]).read_text())
    resolved = set(d["resolved_ids"])
    buckets: dict[str, list[int]] = defaultdict(list)
    for inst in d["submitted_ids"]:
        buckets[repo_of(inst)].append(1 if inst in resolved else 0)
    return dict(buckets)


def balanced(buckets: dict[str, list[int]], repos: list[str]) -> float:
    rates = [sum(buckets[r]) / len(buckets[r]) for r in repos if buckets.get(r)]
    return 100 * sum(rates) / len(rates)


def bootstrap_ci(buckets: dict[str, list[int]], repos: list[str]) -> list:
    rng = random.Random(SEED)
    draws = []
    for _ in range(B):
        rates = []
        for r in repos:
            obs = buckets[r]
            n = len(obs)
            rates.append(sum(rng.choice(obs) for _ in range(n)) / n)
        draws.append(100 * sum(rates) / len(rates))
    draws.sort()
    return [round(draws[int(0.025 * B)], 1), round(draws[int(0.975 * B)], 1)]


def main() -> None:
    models = [m for m in SWEBENCH_RESULTS if m != "gpt-oss-120b"]
    buckets = {m: per_repo(m) for m in models}

    # Repos meeting MIN_N in every model, so the comparison is like-for-like.
    common = set.intersection(*(set(b) for b in buckets.values()))
    repos = sorted(r for r in common
                   if all(len(buckets[m][r]) >= MIN_N for m in models))

    out = {
        "method": (
            f"unweighted mean of per-repo resolve rates, repos with n>={MIN_N} "
            f"in every model; {B}-draw stratified percentile bootstrap, seed {SEED}"
        ),
        "repos": repos,
        "repo_sizes": {r: len(buckets[models[0]][r]) for r in repos},
        "models": {},
    }
    for m in models:
        d = json.loads((REPO / SWEBENCH_RESULTS[m]).read_text())
        nominal = 100 * len(d["resolved_ids"]) / len(d["submitted_ids"])
        out["models"][m] = {
            "nominal_pct": round(nominal, 1),
            "balanced_pct": round(balanced(buckets[m], repos), 1),
            "balanced_ci95_pp": bootstrap_ci(buckets[m], repos),
            "per_repo": {r: f"{sum(buckets[m][r])}/{len(buckets[m][r])}"
                         for r in repos},
        }

    path = DATA / "swebench_reweighted.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    print(f"repos kept (n>={MIN_N} in all models): {', '.join(repos)}")
    print(f"  sizes: {out['repo_sizes']}")
    print(f"\n  {'model':<30}{'nominal':>9}{'balanced':>10}   95% CI")
    for m, e in sorted(out["models"].items(),
                       key=lambda kv: -kv[1]["balanced_pct"]):
        lo, hi = e["balanced_ci95_pp"]
        print(f"  {m:<30}{e['nominal_pct']:>8.1f}%{e['balanced_pct']:>9.1f}%"
              f"   [{lo}, {hi}]  (width {round(hi - lo, 1)} pp)")
    print(f"\nwrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
