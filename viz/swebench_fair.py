#!/usr/bin/env python3
"""Derive infrastructure-fair SWE-bench denominators from committed artifacts.

Why this exists
---------------
The headline SWE-bench number is `resolved / 100`. That denominator silently
mixes two very different events:

  * the model tried and failed            -- a capability result
  * the model was never asked             -- an infrastructure result
                                             (docker pull timeout, 5xx,
                                              tool-call parser fault)

Charging the second kind to the model makes the comparison a measurement of
harness luck. The 2026-08-04 Qwen3.6 run lost 22 instances to a 120 s docker
pull timeout; Laguna lost 25 to a tool-call-parser defect (ISSUES #15). Neither
model was invoked on those instances.

Method (deliberately conservative)
----------------------------------
1. Ground truth for "did this instance get a test verdict?" is the results
   JSON -- resolved_ids | unresolved_ids. Exit statuses are only used to
   ATTRIBUTE a cause, never to decide the verdict, because exit-status files
   can cover a partial re-run segment rather than all 100 instances.
2. An instance is excluded only if it has NO verdict AND its exit status is in
   INFRA_STATUSES. Everything else -- step limits, context-window exhaustion,
   a wrong patch -- stays in the denominator as the model's own outcome.
3. A model with no exit-status artifact is reported with fair_n = 100 and
   `attributed: false`, never silently "adjusted".

Writes viz/data/swebench_fair.json.  Usage: python3 viz/swebench_fair.py
"""
from __future__ import annotations

import json

import yaml

from common import DATA, REPO, SWEBENCH_RESULTS

# Statuses where the harness/serving stack failed before the model could answer.
INFRA_STATUSES = {
    "TimeoutExpired",        # docker pull / test timeout, model never invoked
    "InternalServerError",   # vLLM 5xx (the GB10 long-context wedge)
    "RepeatedFormatError",   # tool-call parser defect, no tool call emitted
    "APIError",
    "ConnectionError",
}

# Model-side statuses, kept in the denominator. Listed explicitly so an
# unrecognised status is reported rather than quietly treated as model-side.
MODEL_STATUSES = {"LimitsExceeded", "ContextWindowExceededError", "Submitted"}

# Exit-status artifacts, relative to the repo root. A model absent here has no
# committed exit statuses -- see results/ornith-35b (a known provenance gap).
EXIT_STATUSES = {
    "qwen3.6-35b-a3b":
        "results/qwen3.6-35b-a3b/raw/swebench/exit_statuses_shuffle100.yaml",
    "laguna-s-2.1-118b":
        "results/laguna-s-2.1-118b/raw/swebench/exit_statuses_n100.yaml",
    "nemotron-3.5-lightning-30b":
        "results/nemotron-3.5-lightning-30b/raw/swebench/exit_statuses_final_segment_n28.yaml",
}

N_TOTAL = 100


def load_statuses(rel: str) -> dict[str, str]:
    """instance_id -> exit status, from the committed YAML."""
    doc = yaml.safe_load((REPO / rel).read_text())
    by_status = doc["instances_by_exit_status"]
    return {inst: status for status, insts in by_status.items() for inst in insts}


def compute() -> dict:
    out: dict[str, dict] = {}

    for model, rel in SWEBENCH_RESULTS.items():
        d = json.loads((REPO / rel).read_text())
        resolved = set(d["resolved_ids"])
        verdict = resolved | set(d["unresolved_ids"])
        no_verdict = N_TOTAL - len(verdict)

        entry: dict = dict(
            resolved=len(resolved),
            verdicts=len(verdict),
            no_verdict=no_verdict,
            nominal_pct=round(100 * len(resolved) / N_TOTAL, 1),
            src=rel,
        )

        rel_yaml = EXIT_STATUSES.get(model)
        if rel_yaml is None:
            # No artifact: report honestly, do not adjust.
            entry.update(attributed=False, infra_excluded=0, fair_n=N_TOTAL,
                         fair_pct=round(100 * len(resolved) / N_TOTAL, 1),
                         causes={}, note="no exit_statuses artifact committed")
        else:
            status_of = load_statuses(rel_yaml)
            submitted = set(d["submitted_ids"])
            causes: dict[str, int] = {}
            infra = 0
            foreign: list[str] = []
            for inst, status in status_of.items():
                if inst in verdict:
                    continue  # got a verdict; cause is moot
                if inst not in submitted:
                    # This instance was never part of the run. A stale file, a
                    # different segment, or a hand-edit can name instances from
                    # the wider 500-problem pool; counting one as "infra" would
                    # shrink the denominator for work never attempted here.
                    foreign.append(inst)
                    continue
                causes[status] = causes.get(status, 0) + 1
                if status in INFRA_STATUSES:
                    infra += 1
            unknown = sorted(set(causes) - INFRA_STATUSES - MODEL_STATUSES)

            # An exclusion must correspond to a real missing verdict. If this
            # trips, the exit-status file disagrees with the results JSON and
            # the results JSON wins -- refuse rather than publish.
            if infra > no_verdict:
                raise SystemExit(
                    f"{model}: {infra} infra exclusions but only {no_verdict} "
                    f"instances lack a verdict ({rel_yaml}). The exit-status "
                    "file disagrees with the results JSON; refusing to emit a "
                    "denominator."
                )

            fair_n = N_TOTAL - infra
            entry.update(attributed=True, infra_excluded=infra, fair_n=fair_n,
                         fair_pct=round(100 * len(resolved) / fair_n, 1),
                         causes=dict(sorted(causes.items())),
                         exit_statuses_src=rel_yaml)
            if unknown:
                entry["unrecognised_statuses"] = unknown
            if foreign:
                entry["ignored_not_in_run"] = len(foreign)
            # The exit-status file may cover only a re-run segment; say so.
            entry["statuses_cover"] = len(status_of)
            # Partial coverage is safe only if every no-verdict instance is
            # accounted for. Refuse to publish an attributed fair denominator
            # when even one submitted no-verdict instance has no status.
            uncovered = sorted(
                (submitted - verdict) - set(status_of)
            )
            if uncovered:
                raise SystemExit(
                    f"{model}: {len(uncovered)} submitted instance(s) have no "
                    f"verdict and no exit status in {rel_yaml}; refusing to "
                    "publish an attributed fair denominator."
                )

        out[model] = entry

    return out


def main() -> None:
    data = compute()
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "swebench_fair.json"
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    print(f"wrote {path.relative_to(REPO)}")

    hdr = f"  {'model':<28}{'nominal':>9}{'infra':>7}{'fair n':>8}{'fair %':>9}"
    print(hdr)
    for m, e in sorted(data.items(), key=lambda kv: -kv[1]["fair_pct"]):
        flag = "" if e["attributed"] else "  (unattributed)"
        print(f"  {m:<28}{e['nominal_pct']:>8.1f}%{e['infra_excluded']:>7}"
              f"{e['fair_n']:>8}{e['fair_pct']:>8.1f}%{flag}")


if __name__ == "__main__":
    main()
