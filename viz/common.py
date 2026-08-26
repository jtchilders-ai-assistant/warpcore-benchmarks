#!/usr/bin/env python3
"""Shared configuration for warpcore-benchmarks figures.

Everything here is deliberately deterministic: same inputs -> byte-identical
SVG output, so regenerated figures produce empty diffs when nothing changed.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------- paths
VIZ = Path(__file__).resolve().parent
REPO = VIZ.parent
DATA = VIZ / "data"
OUT = VIZ / "out"


def repo_path(*parts) -> Path:
    return REPO.joinpath(*parts)


# ---------------------------------------------------------------- identity
# Okabe-Ito colorblind-safe palette. Keys are results/<dir> names.
C = {
    "gpt-oss-120b": "#0072B2",
    "nemotron-3-super-120b": "#E69F00",
    "nemotron-3.5-lightning-30b": "#009E73",
    "ornith-35b": "#D55E00",
    "qwen3.5-122b-a10b": "#CC79A7",
    "qwen3.6-35b-a3b": "#56B4E9",
    "laguna-s-2.1-118b": "#7F7F7F",
}
SHORT = {
    "gpt-oss-120b": "gpt-oss-120b",
    "nemotron-3-super-120b": "Nemotron-3-Super-120B",
    "nemotron-3.5-lightning-30b": "Nemotron-3.5-Lightning-30B",
    "ornith-35b": "Ornith-1.0-35B",
    "qwen3.5-122b-a10b": "Qwen3.5-122B-int4",
    "qwen3.6-35b-a3b": "Qwen3.6-35B",
    "laguna-s-2.1-118b": "Laguna-S-2.1-118B",
}
TINY = {
    "ornith-35b": "Ornith",
    "nemotron-3.5-lightning-30b": "Lightning",
    "qwen3.6-35b-a3b": "Qwen3.6",
    "laguna-s-2.1-118b": "Laguna",
}

# The models with a SWE-bench Verified n=100 run on the IDENTICAL seed-42 set,
# best-first. Single source of truth -- collect_matrix and fig2/fig3 all import
# this, so a new run cannot be added to one and forgotten in the other.
SWEBENCH_RESULTS = {
    "ornith-35b":
        "results/ornith-35b/raw/swebench/swebench_verified_n100_results.json",
    "nemotron-3.5-lightning-30b":
        "results/nemotron-3.5-lightning-30b/raw/swebench_verified_n100_results.json",
    "laguna-s-2.1-118b":
        "results/laguna-s-2.1-118b/raw/swebench/swebench_verified_n100_results.json",
    "qwen3.6-35b-a3b":
        "results/qwen3.6-35b-a3b/raw/swebench/swebench_verified_shuffle100_report.json",
}
SWE_ORDER = ["ornith-35b", "laguna-s-2.1-118b",
             "nemotron-3.5-lightning-30b", "qwen3.6-35b-a3b"]

# Item counts per benchmark (harness configs / model cards).
N_ITEMS = {"gsm8k": 1319, "ifeval": 541, "gpqa": 198, "pi30": 30, "swebench": 100}

BENCHES = [
    ("gsm8k", "GSM8K"),
    ("ifeval", "IFEval"),
    ("gpqa", "GPQA-D"),
    ("pi30", "pi-30"),
    ("swebench", "SWE-bench"),
]

# ---------------------------------------------------------------- style
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    # deterministic SVG element ids -- without this every run rewrites the file
    "svg.hashsalt": "warpcore-benchmarks",
})


def save(fig, name: str) -> None:
    """Write PNG + SVG with no embedded timestamps."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", metadata={"Software": None})
    fig.savefig(OUT / f"{name}.svg", metadata={"Date": None})
    print(f"wrote {(OUT / name).relative_to(REPO)}.{{png,svg}}")


# ---------------------------------------------------------------- statistics
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a single proportion. Returns (lo, hi) in 0..1."""
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def paired_diff(a: set, b: set, n: int = 100, z: float = 1.96):
    """Paired difference of proportions on the SAME n items.

    Returns (diff_pp, lo_pp, hi_pp, a_only, b_only, mcnemar_chi2).
    Only the discordant pairs carry information, which is why this interval is
    much tighter than two independent Wilson intervals.
    """
    a_only, b_only = len(a - b), len(b - a)
    d = (a_only - b_only) / n
    se = math.sqrt(max(a_only + b_only - (a_only - b_only) ** 2 / n, 0)) / n
    chi = (abs(a_only - b_only) - 1) ** 2 / (a_only + b_only) if (a_only + b_only) else 0.0
    return d * 100, (d - z * se) * 100, (d + z * se) * 100, a_only, b_only, chi


def binomial_se(scores_pct, n_items: int) -> float:
    """Binomial SE (in pp) evaluated at the mean of the given scores."""
    p = (sum(scores_pct) / len(scores_pct)) / 100.0
    return 100.0 * math.sqrt(p * (1 - p) / n_items)
