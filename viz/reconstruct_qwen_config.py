#!/usr/bin/env python3
"""Reconstruct the Qwen3.6-35B SWE-bench agent config from its own trajectories.

The config file used for that run was never committed. mini-swe-agent 2.4.6 embeds the
fully-resolved config inside every `*.traj.json` under `info.config`, so the effective
configuration is recoverable from the run's own output -- and is stronger evidence than a
config file, because it is what actually ran rather than what was intended.

Usage:  python3 reconstruct_qwen_config.py <traj_dir> <out.yaml>

Verifies the config is byte-identical across every trajectory before emitting it, and
refuses to write if any key varies (which would mean the run was not uniform).
"""
import json
import pathlib
import sys
from collections import Counter

import yaml


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    traj_dir, out_path = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    files = sorted(traj_dir.glob("*/*.traj.json"))
    if not files:
        print(f"no trajectories under {traj_dir}", file=sys.stderr)
        return 1

    # environment.image is legitimately per-instance; everything else must be uniform.
    seen: Counter = Counter()
    versions: Counter = Counter()
    for f in files:
        info = json.loads(f.read_text())["info"]
        versions[info.get("mini_version")] += 1
        cfg = json.loads(json.dumps(info["config"]))
        cfg["environment"].pop("image", None)
        seen[json.dumps(cfg, sort_keys=True)] += 1

    print(f"trajectories: {len(files)}")
    print(f"mini_version: {dict(versions)}")
    print(f"distinct configs (ignoring per-instance image): {len(seen)}")
    if len(seen) != 1:
        print("REFUSING TO WRITE: config varied across instances", file=sys.stderr)
        for i, (blob, n) in enumerate(seen.most_common(), 1):
            print(f"  variant {i}: {n} instances", file=sys.stderr)
        return 1

    cfg = json.loads(next(iter(seen)))
    header = (
        "# RECONSTRUCTED -- not the original file.\n"
        "#\n"
        "# The agent config for the Qwen3.6-35B-A3B-FP8 SWE-bench n=100 run was never committed.\n"
        "# This file was recovered from `info.config` embedded in the run's own trajectories\n"
        f"# ({len(files)} of them, all identical apart from the per-instance container image),\n"
        "# so it records the configuration that ACTUALLY executed.\n"
        "#\n"
        "# `environment.image` is omitted because it is per-instance.\n"
        "# Reproduce with: viz/reconstruct_qwen_config.py <traj_dir> <out.yaml>\n"
    )
    out_path.write_text(header + yaml.safe_dump(cfg, sort_keys=False, width=100))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
