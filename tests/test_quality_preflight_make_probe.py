"""Regression tests for Make's quality-preflight probe-budget forwarding."""
from __future__ import annotations

import subprocess
from pathlib import Path


def test_make_forwards_configured_serving_probe_budget() -> None:
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "make",
            "-n",
            "quality-preflight",
            "MODE=live",
            "ENDPOINT=http://example.invalid/v1",
            "MODEL=example/model",
            "MAX_TOKENS_PROBE=8192",
            "MAX_GEN_TOKS=65536",
            "AGGREGATE_TOK_S=100",
            "CONCURRENCY=4",
            "CLIENT_TIMEOUT=7200",
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--max-tokens-probe 8192" in result.stdout
