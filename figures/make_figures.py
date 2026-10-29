#!/usr/bin/env python3
"""Figures for "Polarity Subspaces in Dense Retrieval".

All values are taken from the experiment result files (seed means where
noted), including the final cross-encoder statistics of 28 October 2026.
Output: vector PDFs sized for a single-column Elsevier review layout.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np

# Validated categorical slots (light mode) + neutral greys; identity is also
# carried by marker shape and direct labels, so figures survive greyscale print.
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


def grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Figure 1: conceptual schematic
# ---------------------------------------------------------------------------
def fig1():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.9),
                               gridspec_kw={"width_ratios": [1, 1.25]})
    # (a) the mouse analogy
    a.set_xlim(0, 10); a.set_ylim(0, 10); a.axis("off")
    a.set_title("(a) Orientation analogy", fontsize=9, loc="left", color=INK)
    a.add_patch(Rectangle((0.6, 1.2), 3.2, 4.4, fill=False, ec=MUTED, lw=0.8,
                          ls=(0, (3, 2))))
    a.text(2.2, 0.6, "desk (hand frame)", ha="center", fontsize=7.5, color=INK2)
    a.add_patch(FancyArrowPatch((1.4, 2.4), (3.2, 4.6), arrowstyle="-|>",
                                mutation_scale=11, color=INK, lw=1.4))
    a.text(2.2, 5.9, "hand moves", ha="center", fontsize=7.5, color=INK)
    a.add_patch(Rectangle((5.6, 1.2), 3.8, 4.4, fill=False, ec=MUTED, lw=0.8))
    a.text(7.5, 0.6, "screen", ha="center", fontsize=7.5, color=INK2)
    a.add_patch(FancyArrowPatch((8.6, 4.6), (6.8, 2.4), arrowstyle="-|>",
                                mutation_scale=11, color=ORANGE, lw=1.4))
    a.text(7.5, 5.9, "cursor moves", ha="center", fontsize=7.5, color=INK)
    a.text(4.7, 3.4, "mouse\nrotated\n180°", ha="center", va="center",
           fontsize=7, color=INK2)
    a.text(0.3, 8.6, "Magnitude is measured correctly;\nthe frame of reference is wrong.",
           fontsize=8, color=INK, va="top")

    # (b) the low-rank model and the two readouts (Proposition 1)
    b.set_xlim(-0.4, 12.4); b.set_ylim(-2.2, 6.6)
    b.set_title("(b) Negation model and readouts", fontsize=9,
                loc="left", color=INK)
    for s_ in ("left", "bottom"):
        b.spines[s_].set_color(INK2)
    b.set_xticks([]); b.set_yticks([])
    b.set_xlabel("complement of span(U)", fontsize=8, color=INK2)
    b.set_ylabel("polarity subspace span(U)", fontsize=8, color=INK2)
    # two content items; each negated version = in-span transformation + cue offset d
    d = (1.1, 0.0)                      # component of d outside span(U)
    for k, (x, y) in enumerate([(1.0, 4.7), (4.6, 4.0)]):
        aff = (x, y); neg = (x + d[0], 1.5 - 0.3 * k)
        b.plot(*aff, "o", ms=7, color=BLUE, mec="white", mew=1.1, zorder=4)
        b.plot(*neg, "s", ms=7, color=ORANGE, mec="white", mew=1.1, zorder=4)
        b.annotate("", xy=neg, xytext=aff,
                   arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8,
                                   shrinkA=5, shrinkB=5))
        # polarity readout: drop the span(U) coordinate -> two distinct points
        for p_, mk, col in ((aff, "o", BLUE), (neg, "s", ORANGE)):
            b.plot([p_[0], p_[0]], [p_[1], 0.0], color=MUTED, lw=0.7, ls=(0, (2, 2)))
            b.plot(p_[0], 0.0, mk, ms=5.5, color=col, mec="white", mew=0.8,
                   zorder=5, clip_on=False)
        b.annotate("", xy=(neg[0] - 0.08, -0.45), xytext=(aff[0] + 0.08, -0.45),
                   arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=0.9),
                   annotation_clip=False)
        b.text(x + d[0] / 2, -0.85, "(I−UUᵀ)d", ha="center", va="top",
               fontsize=6.5, color=ORANGE)
        # augmented readout: pair coincides
        b.plot(x + d[0] / 2, -1.75, "D", ms=6, color=INK, mec="white", mew=0.9,
               zorder=5, clip_on=False)
    b.text(1.0 + 0.25, 5.25, "affirmative", fontsize=7, color=BLUE)
    b.text(2.35, 2.1, "negated", fontsize=7, color=ORANGE)
    b.text(6.2, 5.9, "b = a + d − c·UUᵀa", fontsize=7.5, color=INK, ha="center")
    b.text(12.3, 0.35, "readout V = U:\nsame offset for every pair",
           fontsize=6.8, ha="right", va="bottom", color=INK2)
    b.text(12.3, -1.75, "readout V = [U, d]:\npair coincides", fontsize=6.8,
           ha="right", va="center", color=INK2)
    fig.tight_layout(w_pad=1.5)
    fig.savefig("fig1_concept.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 (RQ1): NevIR accuracy, full vs polarity removed vs random removed
# ---------------------------------------------------------------------------
RQ1 = [  # label, full, polarity removed, random removed, mean difference removed (revision runs)
    ("E5, zero-shot", 0.187, 0.105, 0.186, 0.176),
    ("GTE, zero-shot", 0.145, 0.080, 0.144, 0.126),
    ("MPNet, zero-shot", 0.081, 0.062, 0.082, 0.073),
    ("Contriever, zero-shot", 0.097, 0.078, 0.096, 0.088),
    ("E5, fine-tuned (3 seeds)", 0.482, 0.249, 0.482, 0.473),
    ("MPNet, fine-tuned (2 seeds)", 0.510, 0.284, 0.510, 0.485),
    ("GTE, fine-tuned (2 seeds)", 0.332, 0.164, 0.331, 0.306),
]


def fig2():
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    ys = np.arange(len(RQ1))[::-1].astype(float)
    ys[:4] += 0.6                      # visual gap between zero-shot and fine-tuned
    for y, (lab, full, pol, rnd, md) in zip(ys, RQ1):
        ax.plot([pol, full], [y, y], color=GRID, lw=2.4, zorder=1, solid_capstyle="round")
        ax.plot(rnd, y, "s", ms=9, mfc="none", mec=MUTED, mew=1.1, zorder=2)
        ax.plot(md, y, "D", ms=5.5, mfc="none", mec=VIOLET, mew=1.1, zorder=2)
        ax.plot(full, y, "o", ms=7, color=BLUE, mec="white", mew=1.0, zorder=3)
        ax.plot(pol, y, "^", ms=7.5, color=ORANGE, mec="white", mew=1.0, zorder=3)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in RQ1])
    ax.axvline(0.25, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.text(0.254, ys[-1] - 0.75, "random ranking", fontsize=7, color=INK2)
    ax.set_xlim(0, 0.56); ax.set_xlabel("NevIR pairwise accuracy")
    ax.tick_params(axis="y", length=0)
    grid(ax, "x")
    h = [plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=7, label="full embedding"),
         plt.Line2D([], [], marker="^", ls="", color=ORANGE, ms=7.5,
                    label="polarity subspace removed"),
         plt.Line2D([], [], marker="s", ls="", mfc="none", mec=MUTED, ms=9,
                    label="random subspace removed"),
         plt.Line2D([], [], marker="D", ls="", mfc="none", mec=VIOLET, ms=5.5,
                    label="mean difference removed")]
    ax.set_ylim(ys[-1] - 0.95, ys[0] + 0.6)
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.42, 1.01), ncol=2,
              fontsize=7.5, handletextpad=0.3, columnspacing=1.2)
    fig.savefig("fig2_subspace_removal.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 (RQ2/RQ3): SciFact negated vs affirmative, three stages
# ---------------------------------------------------------------------------
RQ23 = [  # label, colour, marker, negated (ZS, trained, readout), affirmative (...); revision-run seed means
    ("E5", BLUE, "o", (0.722, 0.503, 0.686), (0.708, 0.607, 0.661)),
    ("MPNet", AQUA, "^", (0.692, 0.628, 0.685), (0.634, 0.615, 0.633)),
    ("GTE", VIOLET, "D", (0.743, 0.661, 0.673), (0.751, 0.695, 0.702)),
]


def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9), sharey=True)
    x = [0, 1, 2]
    stages = ["zero-shot", "fine-tuned", "fine-tuned,\npolarity readout"]
    for ax, idx, title in ((axes[0], 3, "(a) Negated claims (n = 35)"),
                           (axes[1], 4, "(b) Affirmative claims (n = 265)")):
        for row in RQ23:
            lab, col, mk = row[0], row[1], row[2]
            v = row[idx]
            ax.plot(x, v, color=col, lw=2, marker=mk, ms=6, mec="white", mew=0.9)
        ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=8)
        ax.set_xlim(-0.25, 2.25)
        ax.set_title(title, fontsize=9, loc="left")
        grid(ax, "y")
    axes[0].set_ylabel("SciFact nDCG@10")
    axes[0].set_ylim(0.45, 0.78)
    h = [plt.Line2D([], [], color=c, marker=m, lw=2, ms=6, mec="white", label=l)
         for l, c, m, _, _ in RQ23]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=3,
               fontsize=7.5, handlelength=2.2, columnspacing=1.4)
    fig.tight_layout(w_pad=1.2)
    fig.savefig("fig3_scifact_repair.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 (RQ4): ExcluIR pairwise RR per seed
# ---------------------------------------------------------------------------
RQ4 = [  # condition, per-seed values (E5, confirmatory three-seed run)
    ("Zero-shot", [0.6028]),
    ("Plain\nfine-tuning", [0.5947, 0.6075, 0.6364]),
    ("Random-subspace\ncontrol", [0.6095, 0.6118, 0.6118]),
    ("Anchored\nframe", [0.6225, 0.6231, 0.6231]),
]


def fig4():
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    for i, (lab, vals) in enumerate(RQ4):
        col = ORANGE if lab.startswith("Anchored") else BLUE
        jitter = np.linspace(-0.09, 0.09, len(vals)) if len(vals) > 1 else [0]
        ax.plot([i - 0.22, i + 0.22], [np.mean(vals)] * 2, color=INK, lw=1.2)
        ax.plot(i + np.array(jitter), vals, "o", ms=6.5, color=col, mec="white",
                mew=1.0, zorder=3)
    ax.set_xticks(range(len(RQ4))); ax.set_xticklabels([r[0] for r in RQ4], fontsize=8)
    ax.set_ylabel("ExcluIR pairwise RR")
    ax.set_ylim(0.585, 0.645)
    ax.tick_params(axis="x", length=0)
    grid(ax, "y")
    ax.text(3.0, 0.629, "0.623 in all seeds", ha="center", fontsize=7, color=INK2)
    ax.text(1.0, 0.641, "seed range 0.595–0.636", ha="center", fontsize=7, color=INK2)
    fig.savefig("fig4_exclusion_stability.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 (RQ2): negation and exclusion competence dissociate
# ---------------------------------------------------------------------------
ZS = {"E5": (0.187, 0.603), "GTE": (0.145, 0.629), "MPNet": (0.081, 0.713),
      "Contriever": (0.097, 0.617)}
FT = {"E5": (0.482, 0.606), "GTE": (0.332, 0.629), "MPNet": (0.510, 0.673)}  # revision runs


def fig5():
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for enc, (nx, ey) in ZS.items():
        ax.plot(nx, ey, "o", ms=7, mfc="white", mec=BLUE, mew=1.4, zorder=3)
        if enc in FT:
            fx, fy = FT[enc]
            ax.annotate("", xy=(fx, fy), xytext=(nx, ey),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9,
                                        shrinkA=5, shrinkB=5))
            ax.plot(fx, fy, "o", ms=7, color=BLUE, mec="white", mew=1.0, zorder=3)
    lab = {"E5": (0.180, 0.594, "right"), "GTE": (0.140, 0.634, "right"),
           "MPNet": (0.090, 0.716, "left"), "Contriever": (0.091, 0.611, "right")}
    for enc, (lx, ly, ha) in lab.items():
        ax.text(lx, ly, enc, fontsize=7.5, ha=ha, color=INK)
    ax.axhline(0.5, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.set_xlabel("NevIR pairwise accuracy (negation)")
    ax.set_ylabel("ExcluIR pairwise RR (exclusion)")
    ax.set_xlim(0.0, 0.58); ax.set_ylim(0.58, 0.73)
    grid(ax, "both")
    h = [plt.Line2D([], [], marker="o", ls="", mfc="white", mec=BLUE, mew=1.4, ms=7,
                    label="zero-shot"),
         plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=7,
                    label="fine-tuned on NevIR (plain)")]
    ax.legend(handles=h, loc="center right", bbox_to_anchor=(1.0, 0.42), fontsize=7.5)
    fig.savefig("fig5_dissociation.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6 (RQ2/RQ3): controlled-negation penalty per seed
# ---------------------------------------------------------------------------
CTRL = {  # zero-shot penalty, fine-tuned per seed, fine-tuned + polarity readout per seed
    "E5": (-0.016, [-0.176, -0.051, -0.118], [-0.023, -0.007, -0.017]),
    "MPNet": (-0.002, [-0.017, -0.041], [0.002, -0.005]),
    "GTE": (-0.017, [-0.016, -0.027], [0.000, -0.017]),
}


def fig6():
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    cols = (MUTED, ORANGE, BLUE); mks = ("o", "s", "^")
    labs = ("zero-shot", "fine-tuned", "fine-tuned, polarity readout")
    for gi, (enc, (z, ft, ro)) in enumerate(CTRL.items()):
        for si, vals in enumerate(([z], ft, ro)):
            x0 = gi * 3.6 + si
            jit = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else [0]
            ax.plot([x0 - 0.3, x0 + 0.3], [np.mean(vals)] * 2, color=INK, lw=1.1)
            ax.plot(x0 + np.array(jit), vals, mks[si], ms=6.5, color=cols[si],
                    mec="white", mew=0.9, zorder=3)
        ax.text(gi * 3.6 + 1, 0.022, enc, ha="center", fontsize=8.5, color=INK)
    ax.axhline(0, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.set_xticks([]); ax.set_xlim(-0.7, 2 * 3.6 + 2.7)
    ax.set_ylim(-0.195, 0.035)
    ax.set_ylabel("negation penalty\n(nDCG@10, negated − original)")
    grid(ax, "y")
    h = [plt.Line2D([], [], marker=m, ls="", color=c, ms=6.5, label=l)
         for m, c, l in zip(mks, cols, labs)]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              fontsize=7.5, handletextpad=0.3, columnspacing=1.2)
    fig.savefig("fig6_controlled_negation.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    print("figures written")
