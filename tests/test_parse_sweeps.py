"""Regression tests for throughput artifact discovery."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "viz"))

import parse_sweeps  # noqa: E402


def test_collect_includes_extended_sweep_directories() -> None:
    rows = parse_sweeps.collect()
    by_key = {(row["model"], row["concurrency"]): row for row in rows}

    lightning = by_key[("nemotron-3.5-lightning-30b", 384)]
    assert lightning["out_tok_s"] == 926.18
    assert "throughput_sweep_extended" in lightning["source"]

    ornith = by_key[("ornith-35b", 384)]
    assert ornith["out_tok_s"] == 557.78
    assert "throughput_sweep_extended" in ornith["source"]
