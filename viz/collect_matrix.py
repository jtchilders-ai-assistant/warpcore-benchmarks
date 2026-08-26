#!/usr/bin/env python3
"""Collect the per-model benchmark matrix from committed artifacts.

Writes viz/data/bench_matrix.json. Values are cross-checked against the model
cards; every deviation from a raw harness number is annotated with a `note`
explaining why, so the figures can surface caveats rather than launder them.

Usage:  python3 viz/collect_matrix.py
"""
from __future__ import annotations

import json
import os
import re

from common import DATA, REPO, SWEBENCH_RESULTS

# Quality result files, relative to results/<model>/
QUALITY = {
    "gpt-oss-120b": dict(
        gsm8k="raw/gsm8k_results.json",
        ifeval="raw/ifeval_results.json",
        gpqa="raw/gpqa_diamond_results.json"),
    "nemotron-3-super-120b": dict(
        gsm8k="raw/gsm8k_results.json",
        ifeval="raw/ifeval_results.json",
        gpqa="raw/gpqa_diamond_results.json"),
    "qwen3.6-35b-a3b": dict(
        gsm8k="raw/gsm8k_results.json",
        ifeval="raw/ifeval_results.json",
        gpqa="raw/gpqa_nothink_results.json"),
    "nemotron-3.5-lightning-30b": dict(
        gsm8k="raw/quality/gsm8k_cot_zeroshot_clean/nvidia__NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/results_2026-08-12T11-08-12.708266.json",
        ifeval="raw/quality/ifeval/nvidia__NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/results_2026-08-12T13-08-23.626945.json",
        gpqa="raw/quality/gpqa_32k/nvidia__NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/results_2026-08-12T21-15-33.518368.json"),
    "ornith-35b": dict(
        gsm8k="raw/quality/gsm8k/results_2026-08-18T22-22-03.403814.json",
        ifeval="raw/quality/ifeval/results_2026-08-19T01-18-15.177081.json",
        gpqa="raw/quality/gpqa/results_2026-08-19T11-45-14.035924.json"),
    "laguna-s-2.1-118b": dict(
        gsm8k="raw/quality/gsm8k/results_2026-08-22T04-37-53.389704.json",
        ifeval="raw/quality/ifeval/results_2026-08-22T06-10-16.893966.json"),
}

SWEBENCH = SWEBENCH_RESULTS

METRIC_KEYS = {
    "gsm8k": ["exact_match,flexible-fallback", "exact_match,flexible-extract"],
    "ifeval": ["prompt_level_strict_acc,none"],
    "gpqa": ["exact_match,answer-line"],
}


def pick(results: dict, keys: list[str]):
    """Pull the first matching metric + its stderr out of an lm-eval results block."""
    for task in results.values():
        for k in keys:
            if k in task:
                stderr_key = (k.replace("exact_match,", "exact_match_stderr,")
                               .replace("_acc,", "_acc_stderr,"))
                return 100 * task[k], 100 * task.get(stderr_key, 0)
    return None, None


def collect() -> dict:
    out: dict[str, dict] = {}

    for model, files in QUALITY.items():
        out[model] = {}
        for bench, rel in files.items():
            path = REPO / "results" / model / rel
            if not path.exists():
                continue
            results = json.loads(path.read_text())["results"]
            value, stderr = pick(results, METRIC_KEYS[bench])
            if value is None or stderr is None:
                continue
            out[model][bench] = dict(value=round(value, 2),
                                     stderr=round(stderr, 2), src=rel)

    # ---- documented corrections and caveats (see the model cards / ISSUES.md)
    out["laguna-s-2.1-118b"]["gsm8k"]["value"] = 96.13
    out["laguna-s-2.1-118b"]["gsm8k"]["note"] = "CORRECTED re-serve (raw harness 83.40)"
    out["laguna-s-2.1-118b"]["ifeval"]["note"] = "floor: 5.4% empty"
    out["ornith-35b"]["gpqa"]["note"] = "UNDERESTIMATE: 21.2% empty (ISSUES #15)"
    out["ornith-35b"]["ifeval"]["note"] = "floor: 5.2% empty"
    out["qwen3.6-35b-a3b"]["gpqa"]["note"] = "non-thinking mode"

    # Lightning's headline GPQA is the 64k composite: the 32k run plus a 64k
    # replay of the 41 truncated items (151/198). The raw 32k file alone is 66.16.
    light = out["nemotron-3.5-lightning-30b"]["gpqa"]
    light["value"], light["stderr"] = 76.26, 3.03
    light["note"] = "64k budget composite (32k=66.16, 16k=53.03)"
    light["src"] += " + raw/gpqa_64k_replay_results.json"

    # ---- pi-30
    for model in sorted(os.listdir(REPO / "results")):
        for fname in ("SUMMARY.txt", "RESULTS.txt"):
            path = REPO / "results" / model / "raw" / "pi30" / fname
            if not path.exists():
                continue
            text = path.read_text(errors="ignore")
            scores = re.findall(r"SCORE:\s*(\d+)/30", text)
            if scores:
                passed = int(scores[-1])
            else:
                # Older RESULTS.txt files have no SCORE line; count distinct passes.
                passed = len({int(x) for x in re.findall(r"^P(\d+): PASS", text, re.M)})
            out.setdefault(model, {})["pi30"] = dict(
                value=round(100 * passed / 30, 2), passed=passed,
                src=str(path.relative_to(REPO)))
            break

    # ---- SWE-bench Verified
    #
    # `value` is the headline resolved count. Two extra fields matter for honest
    # reporting and are carried through to the figures:
    #   empty_patch  -- misses where the agent never submitted a patch at all.
    #                   These are NOT "wrong answer"; they are budget/format
    #                   ceilings, and they differ by 17x across these models.
    #   resolve_rate -- resolved / submitted. Separates "can it fix bugs?" from
    #                   "can it drive the harness to completion?".
    for model, rel in SWEBENCH.items():
        d = json.loads((REPO / rel).read_text())
        resolved = len(d["resolved_ids"])
        empty = len(d["empty_patch_ids"])
        submitted = resolved + len(d["unresolved_ids"])
        entry: dict = dict(value=float(resolved), empty_patch=empty,
                           submitted=submitted,
                           resolve_rate=round(100 * resolved / submitted, 1) if submitted else None,
                           src=rel)
        # exit statuses explain WHY a patch was empty; committed per PROVENANCE.md §2
        exits = REPO / rel.rsplit("/", 1)[0] / "exit_statuses_n100.yaml"
        if exits.exists():
            counts: dict[str, int] = {}
            cur = None
            for line in exits.read_text().splitlines():
                if line.startswith("    - "):
                    if cur:
                        counts[cur] = counts.get(cur, 0) + 1
                elif line.startswith("  ") and line.rstrip().endswith(":"):
                    cur = line.strip().rstrip(":")
            if counts:
                entry["exit_statuses"] = counts
        out[model]["swebench"] = entry

    out["laguna-s-2.1-118b"]["swebench"]["note"] = (
        "35 empty patches (25 RepeatedFormatError) -- floor, not ceiling")

    return out


def main() -> None:
    out = collect()
    DATA.mkdir(parents=True, exist_ok=True)
    dest = DATA / "bench_matrix.json"
    dest.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {dest.relative_to(REPO)}")
    for model in sorted(out):
        print(f"  {model:28s}",
              {b: v.get("value") for b, v in sorted(out[model].items())})


if __name__ == "__main__":
    main()
