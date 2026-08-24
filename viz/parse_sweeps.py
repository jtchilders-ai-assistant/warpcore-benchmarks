#!/usr/bin/env python3
"""Parse every vLLM throughput-sweep artifact into one tidy CSV.

Reads ONLY committed artifacts under results/*/raw/throughput_sweep/ and
handles the three formats currently in the repo:

  A) already-parsed throughput_sweep.csv            (gpt-oss-120b)
  B) '=== concurrency=N num_prompts=M ... ==='      (nemotron x2, ornith, qwen3.x)
  C) '========= CONCURRENCY N =========' blocks     (laguna)

Usage:  python3 viz/parse_sweeps.py [out.csv]
Default output: viz/data/throughput_all.csv
"""
from __future__ import annotations

import csv
import glob
import os
import re
import sys

from common import DATA, REPO

HDR_B = re.compile(r"^=== concurrency=(\d+)\s+num_prompts=(\d+)")
HDR_C = re.compile(r"^=+\s*CONCURRENCY\s+(\d+)\s*=+\s*$")

FIELDS = {
    "duration_s":     re.compile(r"^Benchmark duration \(s\):\s+([\d.]+)"),
    "out_tok_s":      re.compile(r"^Output token throughput \(tok/s\):\s+([\d.]+)"),
    "mean_ttft_ms":   re.compile(r"^Mean TTFT \(ms\):\s+([\d.]+)"),
    "median_ttft_ms": re.compile(r"^Median TTFT \(ms\):\s+([\d.]+)"),
    "p99_ttft_ms":    re.compile(r"^P99 TTFT \(ms\):\s+([\d.]+)"),
    "mean_tpot_ms":   re.compile(r"^Mean TPOT \(ms\):\s+([\d.]+)"),
    "completed":      re.compile(r"^Successful requests:\s+(\d+)"),
}

COLS = ["model", "concurrency", "num_prompts", "completed", "duration_s",
        "out_tok_s", "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
        "mean_tpot_ms", "source"]


def parse_log(path: str, model: str) -> list[dict]:
    rows, cur = [], None
    with open(path, errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            m = HDR_B.match(line) or HDR_C.match(line)
            if m:
                if cur:
                    rows.append(cur)
                g = m.groups()
                cur = {
                    "model": model,
                    "concurrency": int(g[0]),
                    "num_prompts": int(g[1]) if len(g) > 1 else None,
                    "source": os.path.relpath(path, REPO),
                }
                continue
            if cur is None:
                continue
            for key, rx in FIELDS.items():
                mm = rx.match(line)
                if mm:
                    cur[key] = float(mm.group(1))
    if cur:
        rows.append(cur)
    # Drop blocks with no throughput number: crashed or truncated segments.
    return [r for r in rows if r.get("out_tok_s")]


def parse_csv(path: str, model: str) -> list[dict]:
    rows = []
    with open(path) as fh:
        for d in csv.DictReader(fh):
            rows.append({
                "model": model,
                "concurrency": int(d["concurrency"]),
                "num_prompts": int(d["num_prompts"]),
                "completed": int(d["num_prompts"]),
                "duration_s": float(d["duration_s"]),
                "out_tok_s": float(d["out_tok_s"]),
                "mean_ttft_ms": float(d["mean_ttft_ms"]),
                "median_ttft_ms": float(d["median_ttft_ms"]),
                "p99_ttft_ms": float(d["p99_ttft_ms"]),
                "mean_tpot_ms": float(d["mean_tpot_ms"]),
                "source": os.path.relpath(path, REPO),
            })
    return rows


def collect() -> list[dict]:
    all_rows: list[dict] = []
    for mdir in sorted(glob.glob(str(REPO / "results" / "*"))):
        model = os.path.basename(mdir)
        tdir = os.path.join(mdir, "raw", "throughput_sweep")
        if not os.path.isdir(tdir):
            continue
        csvp = os.path.join(tdir, "throughput_sweep.csv")
        if os.path.exists(csvp):
            all_rows += parse_csv(csvp, model)
            continue
        for log in sorted(glob.glob(os.path.join(tdir, "sweep*.log"))):
            if "CONTAMINATED" in log:  # excluded by the Laguna card
                continue
            all_rows += parse_log(log, model)

    # De-duplicate on (model, concurrency); later files (sweep_hi) win.
    dedup = {(r["model"], r["concurrency"]): r for r in all_rows}
    return sorted(dedup.values(), key=lambda r: (r["model"], r["concurrency"]))


def main() -> None:
    rows = collect()
    out = sys.argv[1] if len(sys.argv) > 1 else str(DATA / "throughput_all.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    models = sorted({r["model"] for r in rows})
    print(f"wrote {out}: {len(rows)} rows, {len(models)} models")
    for m in models:
        mrows = [r for r in rows if r["model"] == m]
        cs = sorted(r["concurrency"] for r in mrows)
        peak = max((r["out_tok_s"], r["concurrency"]) for r in mrows)
        print(f"  {m:28s} c={cs}  peak {peak[0]:.1f} tok/s @c={peak[1]}")


if __name__ == "__main__":
    main()
