#!/usr/bin/env python3
"""fig1 — the vLLM serving envelope on warpcore (GB10).

Left : latency-throughput Pareto frontier across all 7 models.
Right: the same data divided by concurrency -- what a single user feels.

Usage:  python3 viz/fig1_pareto.py
"""
from __future__ import annotations

import csv

import matplotlib
import matplotlib.pyplot as plt

from common import C, DATA, SHORT, save

SWEEP_CSV = DATA / "throughput_all.csv"


def load():
    by: dict[str, list[tuple[int, float, float, float]]] = {}
    with open(SWEEP_CSV) as fh:
        for r in csv.DictReader(fh):
            by.setdefault(r["model"], []).append((
                int(r["concurrency"]), float(r["out_tok_s"]),
                float(r["mean_tpot_ms"]), float(r["median_ttft_ms"])))
    for m in by:
        by[m].sort()
    return by


def main() -> None:
    by = load()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.4))

    for m, pts in sorted(by.items(), key=lambda kv: -max(p[1] for p in kv[1])):
        conc = [p[0] for p in pts]
        tok = [p[1] for p in pts]
        tpot = [p[2] for p in pts]
        col = C[m]
        ax.plot(tok, tpot, "-o", color=col, ms=3.6, lw=1.7, label=SHORT[m], zorder=3)
        ax.annotate(f"c={conc[0]}", (tok[0], tpot[0]), textcoords="offset points",
                    xytext=(-16, -3), fontsize=6.6, color=col)
        ax2.plot(conc, [t / c for t, c in zip(tok, conc)], "-o", color=col, ms=3.6, lw=1.7)

    ax.axhline(100, color="crimson", ls="--", lw=1.2, zorder=2)
    ax.text(0.985, 100, " TPOT = 100 ms  (interactive SLO)", color="crimson",
            fontsize=7.5, va="bottom", ha="right", transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("aggregate output throughput (tok/s)  \u2192  better")
    ax.set_ylabel("mean TPOT (ms)  \u2190  better")
    ax.set_title("Latency\u2013throughput Pareto frontier\n(512 in / 256 out, --ignore-eos)",
                 fontsize=10)
    ax.legend(fontsize=7.0, loc="lower right", framealpha=0.95, ncol=2,
              borderpad=0.5, labelspacing=0.35)
    ax.set_xticks([20, 50, 100, 200, 500, 1000])
    ax.set_yticks([20, 50, 100, 200, 500])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # The two structurally different ceilings: a config wall vs a real plateau.
    for m, note, dx in [
        ("nemotron-3.5-lightning-30b", "--max-num-seqs 128 wall\n(peak is a FLOOR)", (-8, 30)),
        ("laguna-s-2.1-118b", "genuine KV plateau\n(+2.9% c192\u2192256)", (-6, -52)),
    ]:
        if m not in by:
            continue
        last = max(by[m], key=lambda p: p[1])
        ax.annotate(note, (last[1], last[2]), textcoords="offset points", xytext=dx,
                    fontsize=6.8, color=C[m], ha="center", zorder=7,
                    bbox=dict(fc="white", ec="none", alpha=0.88, pad=1.4),
                    arrowprops=dict(arrowstyle="->", color=C[m], lw=0.9))

    ax2.axhline(10, color="grey", ls=":", lw=1)
    ax2.text(1.02, 10, "10 tok/s/stream\n(reading speed)", fontsize=6.8, color="grey",
             va="center", transform=ax2.get_yaxis_transform())
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("concurrency")
    ax2.set_ylabel("per-stream throughput (tok/s per request)")
    ax2.set_title("What one user actually feels\n(same data \u00f7 concurrency)", fontsize=10)
    ax2.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256])
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.set_yticks([1, 2, 5, 10, 20, 50])
    ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    fig.suptitle("Warpcore (DGX Spark / GB10) \u2014 vLLM serving envelope, 7 models",
                 fontsize=12, y=1.005)
    fig.text(0.5, -0.055,
             "source: results/*/raw/throughput_sweep/ \u00b7 Laguna sweep_CONTAMINATED.log excluded "
             "per card \u00b7 CAVEAT: num_prompts differs across sweeps (Laguna 3/concurrency, "
             "others 32), so curve SHAPE is comparable but point noise is not",
             ha="center", fontsize=6.8, color="#555")
    save(fig, "fig1_pareto")
    plt.close(fig)


if __name__ == "__main__":
    main()
