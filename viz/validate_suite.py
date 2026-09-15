#!/usr/bin/env python3
"""
viz/validate_suite.py — CLI for warpcore-v1 contract validation.

Usage:
    # Validate a suite YAML (discovers repo root from file location)
    python3 viz/validate_suite.py suite/warpcore-v1.yaml

    # Validate with explicit repo root
    python3 viz/validate_suite.py --repo /path/to/repo suite/warpcore-v1.yaml

    # Validate a serving adapter YAML (must be inside the repo)
    python3 viz/validate_suite.py --adapter adapters/my-model.yaml

    # Validate both suite and adapter
    python3 viz/validate_suite.py suite/warpcore-v1.yaml --adapter adapters/my-model.yaml

Exit codes:
    0  all inputs valid
    1  one or more diagnosed defects
    2  input unreadable or inconclusive (e.g. file not found, parse error)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Allow import from the viz/ directory when run as a script
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from contract import load_yaml, validate_suite, validate_adapter, validate_adapters_dir


def _find_repo_root(suite_path: Path) -> Path:
    """Walk up from *suite_path* looking for a directory that contains 'suite/'.

    Falls back to the grandparent of the suite YAML (the conventional location
    is repo/suite/warpcore-v1.yaml, so grandparent = repo).
    """
    candidate = suite_path.resolve().parent
    # Walk up until we find a directory that looks like the repo root
    for _ in range(8):
        if (candidate / "suite").is_dir() and (candidate / "viz").is_dir():
            return candidate
        if candidate.parent == candidate:
            break
        candidate = candidate.parent
    # Fallback: grandparent of suite file
    return suite_path.resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate warpcore-v1 suite and adapter files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "suite_path",
        nargs="?",
        metavar="SUITE_YAML",
        help="Path to the suite YAML file (e.g. suite/warpcore-v1.yaml). "
             "Required unless --adapter is provided without a suite.",
    )
    parser.add_argument(
        "--repo",
        metavar="REPO_ROOT",
        help="Repository root directory. Auto-detected from suite_path if omitted.",
    )
    parser.add_argument(
        "--adapter",
        metavar="ADAPTER_YAML",
        help="Path to a serving adapter YAML file to validate.",
    )
    args = parser.parse_args(argv)

    if args.suite_path is None and args.adapter is None:
        parser.error("Provide a suite YAML path and/or --adapter ADAPTER_YAML")

    all_errors: list[str] = []
    repo: Path | None = None

    # ---- Suite validation ---------------------------------------------------
    if args.suite_path is not None:
        suite_path = Path(args.suite_path)
        if not suite_path.exists():
            print(f"ERROR: suite file not found: {suite_path}", file=sys.stderr)
            return 2

        if args.repo:
            repo = Path(args.repo)
        else:
            repo = _find_repo_root(suite_path)

        try:
            errors = validate_suite(repo, suite_path)
        except (OSError, IOError, yaml.YAMLError) as exc:
            print(f"ERROR: cannot read/parse suite — {exc}", file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"ERROR: cannot validate suite — {exc}", file=sys.stderr)
            return 2

        if errors:
            for e in errors:
                print(f"SUITE ERROR: {e}")
            all_errors.extend(errors)

    # Validate all checked-in adapters whenever the canonical suite is checked.
    # Draft/noncanonical adapters are schema-valid, but malformed files, duplicate
    # slugs, and unsupported schema versions are contract defects.
    if args.suite_path is not None:
        if repo is None:
            print("ERROR: repo root could not be determined", file=sys.stderr)
            return 2
        suite_path = Path(args.suite_path)
        try:
            suite_document = load_yaml(suite_path)
            adapter_errors = validate_adapters_dir(
                repo,
                repo / "adapters",
                suite=suite_document,
            )
        except (OSError, IOError, yaml.YAMLError) as exc:
            print(f"ERROR: cannot read adapter inputs — {exc}", file=sys.stderr)
            return 2
        for error in adapter_errors:
            print(f"ADAPTER ERROR: {error}")
        all_errors.extend(adapter_errors)

    # ---- Adapter validation -------------------------------------------------
    if args.adapter is not None:
        adapter_path = Path(args.adapter)
        if not adapter_path.exists():
            print(f"ERROR: adapter file not found: {adapter_path}", file=sys.stderr)
            return 2

        # Determine repo root for adapter validation
        if args.repo:
            repo = Path(args.repo)
        elif args.suite_path:
            repo = _find_repo_root(Path(args.suite_path))
        else:
            # Auto-detect from adapter path
            repo = _find_repo_root(adapter_path)

        try:
            errors = validate_adapter(repo, adapter_path)
        except Exception as exc:
            print(f"ERROR: cannot validate adapter — {exc}", file=sys.stderr)
            return 2

        if errors:
            for e in errors:
                print(f"ADAPTER ERROR: {e}")
            all_errors.extend(errors)

    # ---- Summary -----------------------------------------------------------
    if all_errors:
        n = len(all_errors)
        print(f"\n{n} error{'s' if n > 1 else ''} found.")
        return 1

    if args.suite_path:
        print(f"OK: {args.suite_path} is valid.")
    if args.adapter:
        print(f"OK: {args.adapter} is valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
