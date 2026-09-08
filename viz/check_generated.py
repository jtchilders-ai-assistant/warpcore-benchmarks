#!/usr/bin/env python3
"""Run a command and fail only if it changes generated artifacts."""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import subprocess
import sys
from collections.abc import Iterable


def digest(path: pathlib.Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(paths: Iterable[pathlib.Path]) -> dict[pathlib.Path, str | None]:
    return {path: digest(path) for path in paths}


def changed(
    before: dict[pathlib.Path, str | None],
    after: dict[pathlib.Path, str | None],
) -> list[pathlib.Path]:
    return [path for path in before if before[path] != after[path]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", action="append", required=True, type=pathlib.Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")

    before = snapshot(args.file)
    result = subprocess.run(command)
    if result.returncode:
        return result.returncode
    after = snapshot(args.file)
    stale = changed(before, after)
    if stale:
        print("STALE: generation changed these files:", file=sys.stderr)
        for path in stale:
            print(f"  {path}", file=sys.stderr)
        return 1
    print("OK: generated artifacts were already current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
