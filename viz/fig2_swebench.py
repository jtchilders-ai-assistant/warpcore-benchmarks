#!/usr/bin/env python3
"""fig2 — SWE-bench Verified, the same 100 instances for all three models.

Top   : per-instance solve matrix, sorted by difficulty.
Bottom: outcome decomposition (resolved / failed tests / never submitted) and
        paired differences with CIs.

Usage:  python3 viz/fig2_swebench.py
"""
from __future__ import annotations

from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from common import C, REPO, SHORT, SWE_ORDER, TINY, paired_diff, save
from swebench_reporting import selected_swebench_reports


def load_swebench():
    reports = selected_swebench_reports(REPO)
    resolved = {m: set(reports[m]["resolved_ids"]) for m in SWE_ORDER}
    empty = {m: set(reports[m]["empty_patch_ids"]) for m in SWE_ORDER}
    partitions = [
        set(reports[m]["resolved_ids"])
        | set(reports[m]["unresolved_ids"])
        | set(reports[m]["empty_patch_ids"])
        | set(reports[m]["error_ids"])
        | set(reports[m]["incomplete_ids"])
        for m in SWE_ORDER
    ]
    if not partitions or any(ids != partitions[0] for ids in partitions):
        raise ValueError("instance sets differ across models -- paired statistics invalid")
    if len(partitions[0]) != 100:
        raise ValueError(f"expected exactly 100 SWE-bench instances, got {len(partitions[0])}")
    return resolved, empty, sorted(partitions[0])


def main() -> None:
    R, EP, inst = load_swebench()
    NM = len(SWE_ORDER)
    repo_of = {i: i.split("__")[0] for i in inst}
    nsolve = {i: sum(i in R[m] for m in SWE_ORDER) for i in inst}
    inst_sorted = sorted(inst, key=lambda i: (-nsolve[i], repo_of[i], i))

    fig = plt.figure(figsize=(13.6, 7.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92], hspace=0.62, wspace=0.24)

    # ---------------------------------------------------- top: solve matrix
    axm = fig.add_subplot(gs[0, :])
    M = np.zeros((NM, 100))
    for r, m in enumerate(SWE_ORDER):
        for c, i in enumerate(inst_sorted):
            M[r, c] = 2 if i in R[m] else (1 if i in EP[m] else 0)
    cmap = matplotlib.colors.ListedColormap(["#EDEDED", "#F4A582", "#2E7D32"])
    axm.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=2,
               interpolation="nearest", extent=(0, 100, NM, 0))
    for r in range(NM + 1):
        axm.axhline(r, color="white", lw=2)
    axm.set_yticks([r + 0.5 for r in range(NM)])
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
        # Bin labels go BELOW the matrix. Above the axes they crowd the title once
        # a 4th model row is added.
        axm.text((b0 + b1) / 2, NM + 0.30, f"{k} of {NM} solve\nn={b1 - b0}", ha="center",
                 va="top", fontsize=7.4, color="#333")
    axm.set_title("SWE-bench Verified \u2014 identical seed-42 n=100 set, sorted by difficulty",
                  fontsize=10.5, pad=10)
    # Legend below the bin labels, so title / labels / legend never collide.
    axm.legend(handles=[Patch(fc="#2E7D32", label="resolved"),
                        Patch(fc="#F4A582", label="empty patch \u2014 agent never submitted"),
                        Patch(fc="#EDEDED", label="patch submitted but tests failed")],
               fontsize=7.2, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, -0.30), frameon=False)

    # ------------------------------------------- bottom-left: decomposition
    #
    ax1 = fig.add_subplot(gs[1, 0])
    order = sorted(SWE_ORDER, key=lambda m: -len(R[m]))
    y = np.arange(NM)
    res = [len(R[m]) for m in order]
    ep = [len(EP[m]) for m in order]
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
    ax1.set_yticklabels([f"{TINY[m]}\n{len(R[m])}/100 resolved"
                         for m in order], fontsize=8)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(NM - 0.05, -0.55)
    ax1.set_xlabel("instances (of 100)")
    ax1.set_title("Full-denominator SWE-bench outcomes\n"
                  "resolved, unresolved, and empty patches",
                  fontsize=9.8)

    # ------------------------------------------ bottom-right: paired diffs
    #
    # Every pair on the identical instance set. Laguna-vs-Ornith is the pair that
    # decides whether Ornith's repo-best 73 is a real capability lead or partly a
    # harness-completion lead, so it goes first.
    ax2 = fig.add_subplot(gs[1, 1])
    pairs = [("ornith-35b", "laguna-s-2.1-118b"),
             ("laguna-s-2.1-118b", "nemotron-3.5-lightning-30b"),
             ("laguna-s-2.1-118b", "qwen3.6-35b-a3b"),
             ("ornith-35b", "nemotron-3.5-lightning-30b"),
             ("nemotron-3.5-lightning-30b", "qwen3.6-35b-a3b")]
    for i, (a, b) in enumerate(pairs):
        d, lo, hi, a_only, b_only, _chi = paired_diff(R[a], R[b])
        col = "#2E7D32" if lo > 0 else "#B00020"
        ax2.errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="o", color=col,
                     ms=7, capsize=4, lw=1.8)
        ax2.text(d, i + 0.34, f"{d:+.0f} pp  [{lo:+.0f}, {hi:+.0f}]",
                 ha="center", fontsize=7.2, color=col)
        ax2.text(-13.5, i, f"{TINY[a]} \u2212 {TINY[b]}\n({a_only} vs {b_only} discordant)",
                 ha="left", va="center", fontsize=7.0)
    ax2.axvline(0, color="black", lw=1.2)
    ax2.set_yticks([])
    ax2.set_xlim(-14, 46)
    ax2.set_ylim(len(pairs) - 0.25, -0.75)
    ax2.set_xlabel("paired difference in % resolved (95% CI, same 100 instances)")
    ax2.set_title("Paired resolved-rate differences\n"
                  "on the identical 100-instance set",
                  fontsize=9.8)
    ax2.legend(handles=[Patch(fc="#2E7D32", label="CI excludes 0 \u2014 real"),
                        Patch(fc="#B00020", label="CI crosses 0 \u2014 not distinguishable")],
               fontsize=6.9, loc="lower right", framealpha=0.95)

    fig.text(0.5, -0.005,
             "source: results/*/raw/**/swebench*.json  \u00b7  paired (McNemar) CIs \u2014 every "
             "model saw the identical seed-42 100 instances  \u00b7  empty patch = agent never "
             "emitted a diff (format/context/limit), not a wrong answer",
             ha="center", fontsize=6.8, color="#555")
    save(fig, "fig2_swebench")
    plt.close(fig)


if __name__ == "__main__":
    main()
