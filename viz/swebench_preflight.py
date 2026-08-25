#!/usr/bin/env python3
"""Pre-flight check: verify every SWE-bench container image for a run is present locally.

WHY THIS EXISTS
---------------
The Qwen3.6-35B SWE-bench n=100 run (2026-08-04) lost 22 of 100 instances to
`subprocess.TimeoutExpired` raised by `docker run` hitting mini-swe-agent's
`pull_timeout` (default 120 s) against a cold image cache. Those instances never
started a container, so the model was never invoked -- yet they were scored as
zeros, indistinguishable from genuine failures in the final number.

Later runs (Ornith, Lightning) on the same instance set did not hit this, purely
because the earlier run had warmed the cache. That makes the cache an *implicit,
unprotected precondition*: a `docker system prune`, a Docker Desktop cleanup, or a
fresh machine silently restores the failure mode.

This script turns that implicit precondition into a verified one. Run it before any
SWE-bench generation run. It is a fast no-op when the cache is warm.

USAGE
-----
    # check only -- exits 1 if anything is missing
    swebench_preflight.py --instances <preds_or_results.json>

    # check, then pull whatever is missing (sequential, resumable)
    swebench_preflight.py --instances <file.json> --pull

    # emit the missing image refs and stop
    swebench_preflight.py --instances <file.json> --write-missing missing.txt

Instance IDs are read from a mini-swe-agent predictions file (`{instance_id: {...}}`)
or a swebench results file (`resolved_ids` + `unresolved_ids` + ...).

EXIT CODES
    0  every image present (safe to launch)
    1  images missing (do not launch, or re-run with --pull)
    2  usage / environment error
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ARCH = "x86_64"
REGISTRY = "docker.io/swebench"
# SWE-bench image naming replaces the "__" org/repo separator with "_1776_".
SEP = "_1776_"


def image_for(instance_id: str) -> str:
    return f"{REGISTRY}/sweb.eval.{ARCH}.{instance_id.replace('__', SEP)}:latest"


def load_instance_ids(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and any(
        k in data for k in ("resolved_ids", "unresolved_ids", "empty_patch_ids", "error_ids")
    ):
        ids: set[str] = set()
        for key in ("resolved_ids", "unresolved_ids", "empty_patch_ids", "error_ids"):
            ids |= set(data.get(key) or [])
        return sorted(ids)
    if isinstance(data, dict):
        return sorted(data)
    raise ValueError(f"cannot extract instance ids from {path}")


def local_image_tags() -> set[str]:
    """Every locally-present image tag, normalised with and without the docker.io/ prefix."""
    out = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    tags = set(out)
    tags |= {"docker.io/" + t for t in out}
    tags |= {t.removeprefix("docker.io/") for t in out}
    return tags


def docker_available() -> bool:
    try:
        subprocess.run(["docker", "info"], capture_output=True, check=True, timeout=60)
        return True
    except Exception:
        return False


def pull(image: str, timeout: int) -> tuple[bool, str]:
    try:
        r = subprocess.run(
            ["docker", "pull", "--quiet", image],
            capture_output=True, text=True, timeout=timeout,
        )
        return r.returncode == 0, (r.stderr or r.stdout).strip().splitlines()[-1:][0] if (r.stderr or r.stdout).strip() else ""
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", required=True, type=Path,
                    help="predictions or results JSON defining the instance set")
    ap.add_argument("--pull", action="store_true", help="pull any missing images")
    ap.add_argument("--pull-timeout", type=int, default=1800,
                    help="per-image pull timeout in seconds (default 1800; the "
                         "mini-swe-agent default of 120 is what caused the original failure)")
    ap.add_argument("--write-missing", type=Path, help="write missing image refs to this file")
    args = ap.parse_args()

    if not args.instances.exists():
        print(f"ERROR: no such file: {args.instances}", file=sys.stderr)
        return 2
    if not docker_available():
        print("ERROR: docker is not responding to `docker info`.", file=sys.stderr)
        return 2

    ids = load_instance_ids(args.instances)
    have = local_image_tags()
    missing = [i for i in ids if image_for(i) not in have]

    print(f"instance set      : {len(ids)} instances ({args.instances})")
    print(f"images cached     : {len(ids) - len(missing)}")
    print(f"images missing    : {len(missing)}")

    if args.write_missing:
        args.write_missing.write_text("".join(image_for(i) + "\n" for i in missing))
        print(f"wrote missing refs: {args.write_missing}")

    if not missing:
        print("\nPREFLIGHT PASS - every image is present. Safe to launch.")
        return 0

    if not args.pull:
        print("\nPREFLIGHT FAIL - missing images listed below.")
        for i in missing:
            print(f"  {i}")
        print("\nRe-run with --pull to fetch them, or launching now will score "
              "these instances as zeros without ever invoking the model.")
        return 1

    print(f"\npulling {len(missing)} image(s), timeout {args.pull_timeout}s each ...")
    failed: list[tuple[str, str]] = []
    for n, iid in enumerate(missing, 1):
        img = image_for(iid)
        t0 = time.time()
        ok, msg = pull(img, args.pull_timeout)
        dt = time.time() - t0
        status = "ok" if ok else "FAILED"
        print(f"  [{n}/{len(missing)}] {status:6s} {dt:6.1f}s  {iid}" + (f"  -- {msg}" if not ok else ""))
        if not ok:
            failed.append((iid, msg))

    if failed:
        print(f"\nPREFLIGHT FAIL - {len(failed)} image(s) could not be pulled:")
        for iid, msg in failed:
            print(f"  {iid}: {msg}")
        return 1

    print("\nPREFLIGHT PASS - all missing images pulled. Safe to launch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
