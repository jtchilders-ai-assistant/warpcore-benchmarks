#!/usr/bin/env python3
"""fig2 — SWE-bench Verified, the same 100 instances for all three models.

Top   : per-instance solve matrix, sorted by difficulty.
Bottom: outcome decomposition (resolved / failed tests / never submitted) and
        paired differences with CIs.

Usage:  python3 viz/fig2_swebench.py
"""
from __future__ import annotations

import json
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from common import C, REPO, SHORT, SWE_ORDER, TINY, paired_diff, save

SWEBENCH = {
    "ornith-35b":
        "results/ornith-35b/raw/swebench/swebench_verified_n100_results.json",
    "nemotron-3.5-lightning-30b":
        "results/nemotron-3.5-lightning-30b/raw/swebench_verified_n100_results.json",
    "qwen3.6-35b-a3b":
        "results/qwen3.6-35b-a3b/raw/swebench/swebench_verified_shuffle100_report.json",
}


def load_swebench():
    data = {m: json.loads((REPO / p).read_text()) for m, p in SWEBENCH.items()}
    resolved = {m: set(d["resolved_ids"]) for m, d in data.items()}
    empty = {m: set(d["empty_patch_ids"]) for m, d in data.items()}
    submitted = {m: set(d["submitted_ids"]) for m, d in data.items()}
    sets = list(submitted.values())
    assert all(s == sets[0] for s in sets), \
        "instance sets differ across models -- paired statistics would be invalid"
    return resolved, empty, sorted(sets[0])


def main() -> None:
    R, EP, inst = load_swebench()
    repo_of = {i: i.split("__")[0] for i in inst}
    nsolve = {i: sum(i in R[m] for m in SWE_ORDER) for i in inst}
    inst_sorted = sorted(inst, key=lambda i: (-nsolve[i], repo_of[i], i))

    fig = plt.figure(figsize=(13.6, 7.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92], hspace=0.62, wspace=0.24)

    # ---------------------------------------------------- top: solve matrix
    axm = fig.add_subplot(gs[0, :])
    M = np.zeros((3, 100))
    for r, m in enumerate(SWE_ORDER):
        for c, i in enumerate(inst_sorted):
            M[r, c] = 2 if i in R[m] else (1 if i in EP[m] else 0)
    cmap = matplotlib.colors.ListedColormap(["#EDEDED", "#F4A582", "#2E7D32"])
    axm.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=2,
               interpolation="nearest", extent=(0, 100, 3, 0))
    for r in range(4):
        axm.axhline(r, color="white", lw=2)
    axm.set_yticks([0.5, 1.5, 2.5])
    axm.set_yticklabels([f"{SHORT[m]}\n{len(R[m])}/100 resolved \u00b7 {len(EP[m])} empty patch"
                         for m in SWE_ORDER], fontsize=7.8)
    axm.set_xticks([])
    axm.grid(False)

    bounds, prev = [], None
    for c, i in enumerate(inst_sorted):
        if nsolve[i] != prev:
            bounds.append(c)
            prev = nsolve[i]
    bounds.append(100)
    for b0, b1, k in zip(bounds[:-1], bounds[1:], sorted(set(nsolve.values()), reverse=True)):
        axm.axvline(b1, color="#444", lw=0.9, ls=":")
        axm.text((b0 + b1) / 2, -0.20, f"{k} of 3 solve\nn={b1 - b0}", ha="center",
                 fontsize=7.4, color="#333")
    axm.set_title("SWE-bench Verified \u2014 identical seed-42 n=100 set, sorted by difficulty",
                  fontsize=10.5, pad=26)
    axm.legend(handles=[Patch(fc="#2E7D32", label="resolved"),
                        Patch(fc="#F4A582", label="empty patch \u2014 agent never submitted"),
                        Patch(fc="#EDEDED", label="patch submitted but tests failed")],
               fontsize=7.2, ncol=3, loc="upper right",
               bbox_to_anchor=(1.0, 1.30), frameon=False)

    # ------------------------------------------- bottom-left: decomposition
    ax1 = fig.add_subplot(gs[1, 0])
    y = np.arange(3)
    res = [len(R[m]) for m in SWE_ORDER]
    ep = [len(EP[m]) for m in SWE_ORDER]
    fail = [100 - r - e for r, e in zip(res, ep)]
    ax1.barh(y, res, color="#2E7D32", label="resolved")
    ax1.barh(y, fail, left=res, color="#D9D9D9", label="patch failed tests")
    ax1.barh(y, ep, left=[r + f for r, f in zip(res, fail)], color="#F4A582",
             label="empty patch (never submitted)")
    for i, (r, f, e) in enumerate(zip(res, fail, ep)):
        ax1.text(r / 2, i, str(r), ha="center", va="center", fontsize=9,
                 color="white", fontweight="bold")
        if e >= 4:
            ax1.text(r + f + e / 2, i, str(e), ha="center", va="center",
                     fontsize=8.5, color="#7A2E00", fontweight="bold")
    ax1.set_yticks(y)
    ax1.set_yticklabels([TINY[m] for m in SWE_ORDER], fontsize=9)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(2.95, -0.55)
    ax1.set_xlabel("instances (of 100)")
    ax1.set_title("Qwen3.6's 44% is NOT all capability:\n"
                  "29 of its 56 misses never produced a patch", fontsize=9.8)
    ax1.annotate("agent-budget / harness ceiling,\nnot 'wrong answer'", xy=(85.5, 2.30),
                 xytext=(41, 2.74), fontsize=7.0, color="#7A2E00", ha="center",
                 annotation_clip=False,
                 arrowprops=dict(arrowstyle="->", color="#7A2E00", lw=0.9))

    # ------------------------------------------ bottom-right: paired diffs
    ax2 = fig.add_subplot(gs[1, 1])
    pairs = [("ornith-35b", "nemotron-3.5-lightning-30b"),
             ("ornith-35b", "qwen3.6-35b-a3b"),
             ("nemotron-3.5-lightning-30b", "qwen3.6-35b-a3b")]
    for i, (a, b) in enumerate(pairs):
        d, lo, hi, a_only, b_only, _chi = paired_diff(R[a], R[b])
        col = "#2E7D32" if lo > 0 else "#B00020"
        ax2.errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="o", color=col,
                     ms=8, capsize=5, lw=2)
        ax2.text(d, i + 0.30, f"{d:+.0f} pp  [{lo:+.0f}, {hi:+.0f}]",
                 ha="center", fontsize=7.8, color=col)
        ax2.text(-11.4, i, f"{TINY[a]} \u2212 {TINY[b]}\n({a_only} vs {b_only} discordant)",
                 ha="left", va="center", fontsize=7.6)
    ax2.axvline(0, color="black", lw=1.2)
    ax2.set_yticks([])
    ax2.set_xlim(-12, 46)
    ax2.set_ylim(2.75, -0.75)
    ax2.set_xlabel("paired difference in % resolved (95% CI, same 100 instances)")
    ax2.set_title("Ornith's lead is real. Lightning vs Qwen3.6 is a coin-flip:\n"
                  "its CI crosses zero", fontsize=9.8)
    ax2.legend(handles=[Patch(fc="#2E7D32", label="CI excludes 0 \u2014 real"),
                        Patch(fc="#B00020", label="CI crosses 0 \u2014 not distinguishable")],
               fontsize=6.9, loc="lower right", framealpha=0.95)

    fig.text(0.5, -0.005,
             "source: results/{ornith-35b,nemotron-3.5-lightning-30b,qwen3.6-35b-a3b}"
             "/raw/**/swebench*.json  \u00b7  paired (McNemar) CIs \u2014 every model saw the "
             "identical 100 instances",
             ha="center", fontsize=6.8, color="#555")
    save(fig, "fig2_swebench")
    plt.close(fig)


if __name__ == "__main__":
    main()
