#!/usr/bin/env python3
"""fig3 — does the benchmark suite still tell these models apart?

Left  : discrimination table. Spread normalized by measurement noise, because a
        raw max-min range rewards small samples and ignores how many models were
        actually run. This is what identifies pi-30 as saturated.
Middle: rank heatmap -- every column gives a different ordering.
Right : SWE-bench reweighting sensitivity (the sample is 56% django).

Usage:  python3 viz/fig3_discrimination.py
"""
from __future__ import annotations

import json
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from common import (BENCHES, C, DATA, N_ITEMS, SHORT, SWE_ORDER, TINY,
                    binomial_se, save)
from fig2_swebench import load_swebench


# ------------------------------------------------------------------ panel 1
def discrimination_rows(BM: dict) -> list[dict]:
    """Pure function so the table can be checked without rendering anything."""
    rows = []
    for key, nice in BENCHES:
        vals = sorted((BM[m][key]["value"] for m in BM if key in BM[m]), reverse=True)
        if len(vals) < 2:
            continue
        se = binomial_se(vals, N_ITEMS[key])
        spread = max(vals) - min(vals)
        rows.append(dict(bench=nice, key=key, n_models=len(vals), n_items=N_ITEMS[key],
                         spread=spread, se=se, ratio=spread / se,
                         distinct=len({round(v, 4) for v in vals})))
    rows.sort(key=lambda r: r["ratio"], reverse=True)
    return rows


def draw_table(ax, BM: dict) -> list[dict]:
    rows = discrimination_rows(BM)
    ax.axis("off")

    headers = ["benchmark", "models", "spread", "\u00b1SE", "spread \u00f7 SE", "distinct\nscores"]
    colx = [0.005, 0.360, 0.485, 0.620, 0.790, 0.945]
    align = ["left", "center", "center", "center", "center", "center"]
    y0, dy = 0.90, 0.108

    for x, h, a in zip(colx, headers, align):
        ax.text(x, y0 + 0.055, h, fontsize=7.1, fontweight="bold", color="#222",
                ha=a, va="bottom", transform=ax.transAxes, linespacing=0.95)
    ax.plot([0, 1], [y0 + 0.035, y0 + 0.035], color="#222", lw=1.1,
            transform=ax.transAxes, clip_on=False)

    for i, r in enumerate(rows):
        y = y0 - i * dy
        # A benchmark is flagged dead if it cannot resolve models above noise,
        # or if it produces almost no distinct values at all.
        dead = r["distinct"] <= 2 or r["ratio"] < 2.0
        col = "#B00020" if dead else "#222"
        weight = "bold" if dead else "normal"
        if dead:
            ax.add_patch(mpatches.Rectangle(
                (-0.02, y - 0.048), 1.04, 0.096, transform=ax.transAxes,
                fc="#B00020", alpha=0.07, ec="none", zorder=0, clip_on=False))
        cells = [f"{r['bench']}  (n={r['n_items']})", f"{r['n_models']}",
                 f"{r['spread']:.1f} pp", f"{r['se']:.2f} pp",
                 f"{r['ratio']:.1f}\u00d7", f"{r['distinct']}"]
        for x, c, a in zip(colx, cells, align):
            ax.text(x, y, c, fontsize=8.0, color=col, fontweight=weight,
                    ha=a, va="center", transform=ax.transAxes)

    foot = y0 - len(rows) * dy
    ax.plot([0, 1], [foot + 0.5 * dy, foot + 0.5 * dy], color="#CCC", lw=0.8,
            transform=ax.transAxes, clip_on=False)
    ax.text(0, foot - 0.045,
            "pi-30: every model scores 29/30 or 30/30 \u2014 the entire 3.3 pp spread\n"
            "is ONE test case. It costs ~1h15m per model and no longer ranks them.",
            fontsize=7.5, color="#B00020", va="top", transform=ax.transAxes,
            linespacing=1.45)
    ax.text(0, foot - 0.175,
            "spread \u00f7 SE = (max\u2212min) \u00f7 binomial SE at the mean. Higher = better able to\n"
            "tell models apart. Model counts differ per benchmark, so compare with care.",
            fontsize=6.7, color="#666", va="top", transform=ax.transAxes,
            style="italic", linespacing=1.45)
    ax.set_title("pi-30 no longer discriminates:\n5 models, only 2 distinct scores",
                 fontsize=9.8)
    return rows


# ------------------------------------------------------------------ figure
def main() -> None:
    BM = json.loads((DATA / "bench_matrix.json").read_text())
    R, _EP, inst = load_swebench()
    repo_of = {i: i.split("__")[0] for i in inst}
    rc = Counter(repo_of.values())

    fig = plt.figure(figsize=(13.6, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.34, 1.10, 0.92], wspace=0.30)

    draw_table(fig.add_subplot(gs[0, 0]), BM)

    # ---- panel 2: rank heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    models = ([m for m in BM if "swebench" in BM[m]]
              + [m for m in BM if "swebench" not in BM[m]])
    grid = np.full((len(models), len(BENCHES)), np.nan)
    for j, (key, _) in enumerate(BENCHES):
        vals = sorted(((BM[m][key]["value"], m) for m in BM if key in BM[m]), reverse=True)
        for rank, (_, m) in enumerate(vals, 1):
            grid[models.index(m), j] = rank
    im = ax2.imshow(grid, cmap="RdYlGn_r", vmin=1, vmax=6, aspect="auto")
    for i in range(len(models)):
        for j in range(len(BENCHES)):
            if np.isnan(grid[i, j]):
                ax2.text(j, i, "\u2013", ha="center", va="center", fontsize=9, color="#999")
            else:
                v = int(grid[i, j])
                ax2.text(j, i, str(v), ha="center", va="center", fontsize=9.5,
                         fontweight="bold", color="white" if v >= 5 or v <= 1 else "#222")
    ax2.set_xticks(range(len(BENCHES)))
    ax2.set_xticklabels([n for _, n in BENCHES], fontsize=8.2)
    ax2.set_yticks(range(len(models)))
    ax2.set_yticklabels([SHORT[m] for m in models], fontsize=8)
    ax2.grid(False)
    for s in ax2.spines.values():
        s.set_visible(False)
    for j in (2, 4):  # GPQA and SWE-bench columns for Ornith
        ax2.add_patch(plt.Rectangle((j - 0.5, models.index("ornith-35b") - 0.5), 1, 1,
                                    fill=False, ec="black", lw=2.4, zorder=5))
    ax2.set_title("Rank (1 = best) per benchmark.\nOrnith: 4th on GPQA \u2192 1st on SWE-bench",
                  fontsize=9.8)
    cb = fig.colorbar(im, ax=ax2, fraction=0.035, pad=0.02, ticks=[1, 3, 6])
    cb.ax.set_yticklabels(["1st", "3rd", "6th"], fontsize=7.5)

    # ---- panel 3: reweighting sensitivity
    ax3 = fig.add_subplot(gs[0, 2])
    big = sorted(r for r in rc if rc[r] >= 4)
    obs, bal = [], []
    for m in SWE_ORDER:
        obs.append(len(R[m]))
        per = {r: sum(1 for i in inst if repo_of[i] == r and i in R[m]) / rc[r] for r in big}
        bal.append(100 * sum(per.values()) / len(per))
    x = np.arange(len(SWE_ORDER))
    w = 0.34
    ax3.bar(x - w / 2, obs, w, color=[C[m] for m in SWE_ORDER],
            label="observed (django-heavy)")
    ax3.bar(x + w / 2, bal, w, color=[C[m] for m in SWE_ORDER], alpha=0.45,
            hatch="///", edgecolor="white", label=f"repo-balanced ({len(big)} repos, n\u22654)")
    for i, (o, b) in enumerate(zip(obs, bal)):
        ax3.text(i - w / 2, o + 1.5, f"{o:.0f}", ha="center", fontsize=8.4, fontweight="bold")
        ax3.text(i + w / 2, b + 1.5, f"{b:.0f}", ha="center", fontsize=8.4, color="#444")
    ax3.set_xticks(x)
    ax3.set_xticklabels([TINY[m] for m in SWE_ORDER], fontsize=8.0)
    ax3.set_ylim(0, 88)
    ax3.set_ylabel("% resolved")
    # State the takeaway from the DATA, not a hardcoded sentence -- rank order and
    # the size of the leader's drop both change when a model is added.
    rank_obs = [m for _, m in sorted(zip(obs, SWE_ORDER), reverse=True)]
    rank_bal = [m for _, m in sorted(zip(bal, SWE_ORDER), reverse=True)]
    lead_i = SWE_ORDER.index(rank_obs[0])
    if rank_obs == rank_bal:
        head = "Ordering survives de-weighting django,"
    else:
        head = f"De-weighting django REORDERS the top: {TINY[rank_bal[0]]} leads,"
    ax3.set_title(f"{head}\nbut {TINY[rank_obs[0]]}'s lead shrinks "
                  f"{obs[lead_i]:.0f}\u2192{bal[lead_i]:.0f}",
                  fontsize=9.8)
    ax3.legend(fontsize=6.8, loc="upper right", framealpha=0.95)

    fig.suptitle("Does the benchmark suite still tell these models apart?",
                 fontsize=12.5, y=1.04)
    fig.text(0.5, -0.06,
             "source: results/*/raw/quality/*/results_*.json, raw/pi30/*.txt, "
             "raw/**/swebench*.json  \u00b7  Ornith GPQA is an underestimate (ISSUES #15)  \u00b7  "
             "Qwen3.6 GPQA = non-thinking mode  \u00b7  Lightning GPQA = 64k budget",
             ha="center", fontsize=6.8, color="#555")
    save(fig, "fig3_discrimination")
    plt.close(fig)


if __name__ == "__main__":
    main()
