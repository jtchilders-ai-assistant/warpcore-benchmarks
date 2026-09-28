"""Regression tests for production runner serving-probe budget forwarding."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import sys
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "viz"))
import run_quality  # noqa: E402


def test_quality_runner_forwards_configured_probe_budget(tmp_path: Path) -> None:
    run_dir = REPO / "results" / "qwen3.6-35b-a3b" / "runs" / "warpcore-v1" / "gpqa_diamond" / "probe-budget-test"
    runner = run_quality.QualityRunner(
        suite_path=REPO / "suite" / "warpcore-v1.yaml",
        adapter_path=REPO / "adapters" / "qwen3.6-35b-a3b.yaml",
        benchmark="gpqa_diamond", endpoint="http://example.invalid/v1",
        throughput=64, concurrency=4, timeout=9000, run_dir=run_dir,
        prompt_token_maxima={"gsm8k": 256, "ifeval": 373, "gpqa_diamond": 2808},
        max_tokens_probe=8192, allow_no_screen=True,
    )
    with patch.object(run_quality, "QualityPreflightGate") as gate_cls:
        gate_cls.return_value.run.return_value = 0
        assert runner._run_preflight("example/model", 65536) == 0
    assert gate_cls.call_args.kwargs["max_tokens_probe"] == 8192


def test_cli_accepts_max_tokens_probe() -> None:
    rc = run_quality.main([
        "--suite", str(REPO / "suite" / "warpcore-v1.yaml"),
        "--adapter", str(REPO / "adapters" / "qwen3.6-35b-a3b.yaml"),
        "--benchmark", "gpqa_diamond", "--endpoint", "http://example.invalid/v1",
        "--throughput", "64", "--concurrency", "4", "--timeout", "9000",
        "--prompt-tokens", "gsm8k=256,ifeval=373,gpqa_diamond=2808",
        "--max-tokens-probe", "8192", "--run-id", "probe-budget-dry-run", "--dry-run",
    ])
    assert rc == 0
