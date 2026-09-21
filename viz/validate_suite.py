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

from contract import (
    load_yaml,
    validate_adapter,
    validate_adapter_campaign_ready,
    validate_adapters_dir,
    validate_suite,
)


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
    parser.add_argument(
        "--prompt-tokens",
        metavar="BENCH=N,...",
        help=(
            "Measured tokenized prompt maxima for campaign-readiness validation "
            "(for example gsm8k=256,ifeval=373,gpqa_diamond=2808,swebench_verified=0)."
        ),
    )
    args = parser.parse_args(argv)

    prompt_token_maxima: dict[str, int] | None = None
    if args.prompt_tokens is not None:
        prompt_token_maxima = {}
        try:
            for entry in args.prompt_tokens.split(","):
                name, value = entry.split("=", 1)
                name = name.strip()
                parsed = int(value)
                if not name or parsed < 0:
                    raise ValueError
                prompt_token_maxima[name] = parsed
        except (TypeError, ValueError):
            parser.error("--prompt-tokens must be BENCH=N[,BENCH=N...] with nonnegative integers")

    if args.suite_path is None and args.adapter is None:
        parser.error("Provide a suite YAML path and/or --adapter ADAPTER_YAML")

    all_errors: list[str] = []
    repo: Path | None = None
    suite_document: dict | None = None

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

        # The SWE-bench campaign circuit breaker is a suite-owned, versioned
        # control carried in its own file (suite/swebench/circuit_breaker_policy.yaml)
        # so that adding it did not rewrite the hash-pinned suite YAML. It is
        # validated here for the same reason every other suite input is: an
        # unverifiable abort control must not reach a live campaign.
        try:
            import swebench_circuit_breaker

            swebench_circuit_breaker.load_policy(repo)
        except Exception as exc:
            print(f"SUITE ERROR: {exc}")
            all_errors.append(str(exc))

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
                # Directory validation covers schemas and duplicate slugs. Campaign
                # readiness is adapter-specific because prompt maxima are measured
                # for the selected adapter's tokenizer and is checked below.
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
            if not errors and prompt_token_maxima is not None:
                adapter_document = load_yaml(adapter_path)
                slug = (adapter_document.get("model") or {}).get("slug", adapter_path.stem)
                errors.extend(
                    validate_adapter_campaign_ready(
                        adapter_document,
                        slug,
                        suite=suite_document if args.suite_path is not None else None,
                        prompt_token_maxima=prompt_token_maxima,
                    )
                )
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
