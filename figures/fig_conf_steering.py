"""Confidence steering figure — dialing P(target) and entropy while holding the choice fixed.

Aggregates the 4 per-arm confidence-steering runs (re-aligned so index 0 = the target arm):
  A  P(target arm) along the confidence path: sharpens from ~0.4 → ~0.87 (manifold/linear/M_y).
  B  choice entropy along the path: drops toward certainty (vs the uniform ceiling log 4).
  C  identity preserved: the target stays the argmax throughout — confidence moves, choice doesn't.

    python -m figures.fig_conf_steering \
        --conf artifacts/manifolds/llama8b_fixed_conf_L19.pt \
        --steering artifacts/steering/llama8b_fixed_conf_steering_L19.pt \
        --out artifacts/figures/report_fig6.png
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

MAN, LIN, MY, POS = "#1f6feb", "#e8590c", "#343a40", "#2f9e44"
mpl.rcParams.update({
    "figure.facecolor": "white", "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "legend.fontsize": 8.5, "legend.frameon": False,
})


def entropy(P):
    return -(P * np.log(np.clip(P, 1e-12, 1))).sum(-1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conf", type=pathlib.Path, required=True)
    ap.add_argument("--steering", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    mf = torch.load(args.conf, weights_only=False)
    st = torch.load(args.steering, weights_only=False)
    res, K, na = st["results"], st["K"], st["n_arms"]
    frac = np.linspace(0, 1, K)

    # collect per-arm trajectories (mean over bases), aligned to the target arm
    pt = {"manifold": [], "linear_full": []}
    ent = {"manifold": [], "linear_full": []}
    my_pt, identity_ok = [], []
    for key, R in res.items():
        a = R["i"]                                       # target arm index
        dgrid = mf["pairs"][key]["dist_grid"].numpy()    # M_y [G, na]
        my_pt.append(dgrid[np.linspace(0, len(dgrid) - 1, K).round().astype(int), a])
        for mode in ("manifold", "linear_full"):
            d = R[f"dist_{mode}"].numpy().mean(0)         # [K, na+1]
            arms = d[:, :na] / d[:, :na].sum(1, keepdims=True)
            pt[mode].append(arms[:, a])
            ent[mode].append(entropy(arms))
            if mode == "manifold":
                identity_ok.append((arms.argmax(1) == a).mean())
    agg = lambda L: (np.mean(L, 0), np.std(L, 0) / np.sqrt(len(L)))

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))

    # A — P(target) sharpening
    m_m, m_s = agg(pt["manifold"]); l_m, l_s = agg(pt["linear_full"]); y_m, _ = agg(my_pt)
    ax[0].plot(np.linspace(0, 1, len(y_m)), y_m, color=MY, lw=2, alpha=0.55, label="$M_y$ (natural)")
    ax[0].plot(frac, m_m, color=MAN, lw=2.5, label="manifold")
    ax[0].fill_between(frac, m_m - m_s, m_m + m_s, color=MAN, alpha=0.15)
    ax[0].plot(frac, l_m, color=LIN, lw=2.5, ls="--", label="linear")
    ax[0].axhline(0.25, color="gray", ls=":", lw=1, label="chance (¼)")
    ax[0].set_xlabel("confidence path  (low gap → high gap)")
    ax[0].set_ylabel("P(target arm)")
    ax[0].set_title("Steering up confidence\nsharpens the choice")
    ax[0].set_ylim(0, 1); ax[0].legend(loc="center right")
    ax[0].text(-0.13, 1.06, "A", transform=ax[0].transAxes, fontsize=13, fontweight="bold")

    # B — entropy dropping
    em_m, em_s = agg(ent["manifold"]); el_m, el_s = agg(ent["linear_full"])
    ax[1].axhline(np.log(na), color="gray", ls=":", lw=1, label="uniform (log 4)")
    ax[1].plot(frac, em_m, color=MAN, lw=2.5, label="manifold")
    ax[1].fill_between(frac, em_m - em_s, em_m + em_s, color=MAN, alpha=0.15)
    ax[1].plot(frac, el_m, color=LIN, lw=2.5, ls="--", label="linear")
    ax[1].set_xlabel("confidence path  (low gap → high gap)")
    ax[1].set_ylabel("choice entropy (nats)")
    ax[1].set_title("…and lowers entropy\ntoward certainty")
    ax[1].legend(loc="upper right")
    ax[1].text(-0.13, 1.06, "B", transform=ax[1].transAxes, fontsize=13, fontweight="bold")

    # C — identity preserved: P(target) vs best competing arm (manifold)
    best_other = []
    for key, R in res.items():
        a = R["i"]; d = R["dist_manifold"].numpy().mean(0)
        arms = d[:, :na] / d[:, :na].sum(1, keepdims=True)
        oth = arms.copy(); oth[:, a] = -1
        best_other.append(oth.max(1))
    bo_m, _ = agg(best_other)
    ax[2].plot(frac, m_m, color=POS, lw=2.5, label="target arm")
    ax[2].plot(frac, bo_m, color="#adb5bd", lw=2.5, label="best competitor")
    ax[2].fill_between(frac, bo_m, m_m, color=POS, alpha=0.10)
    ax[2].set_xlabel("confidence path  (low gap → high gap)")
    ax[2].set_ylabel("P(arm)")
    ax[2].set_title(f"Identity preserved:\ntarget stays the choice ({100*np.mean(identity_ok):.0f}% of steps)")
    ax[2].set_ylim(0, 1); ax[2].legend(loc="center right")
    ax[2].text(-0.13, 1.06, "C", transform=ax[2].transAxes, fontsize=13, fontweight="bold")

    fig.suptitle("Confidence steering: dialing certainty without changing the choice",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"fig: P(target) {m_m[0]:.2f}→{m_m[-1]:.2f} (manifold); "
          f"entropy {em_m[0]:.2f}→{em_m[-1]:.2f}; identity kept {100*np.mean(identity_ok):.0f}%")


if __name__ == "__main__":
    main()
