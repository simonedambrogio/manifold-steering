"""Publication-quality figures for the value-steering report (one coherent style).

Generates artifacts/figures/report_fig{1..5}.png:
  1  Agent learns the bandit (performance above chance)
  2  A value latent is decodable (ridge R² by layer + isometry on the Q_max manifold)
  3  Causal steering: the channel works (interchange) and the lever is value (not recency)
  4  Concept geometry: the 4-arm simplex in behavior and activation space
  5  Manifold vs linear steering: trajectory + the two-metric verdict

    python -m figures.report_figures
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---- shared style ----
MAN, LIN, MY, POS = "#1f6feb", "#e8590c", "#343a40", "#2f9e44"
ARM = ["#1f6feb", "#e8590c", "#2f9e44", "#9c36b5"]
mpl.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelsize": 10, "legend.fontsize": 8.5, "legend.frameon": False,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
})
ART = pathlib.Path("artifacts")
FIG = ART / "figures"


def panel_tag(ax, s, dx=-0.12, dy=1.06):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")


def load(p):
    return torch.load(p, weights_only=False)


# ============================ Figure 1 — performance ============================
def fig1():
    logs = json.loads((ART / "datasets/llama8b_dataset.json").read_text())
    T = len(logs[0]["actions"])
    rel_ep, per_trial = [], np.zeros((len(logs), T))
    for e, l in enumerate(logs):
        pm = np.array(l["payoff_matrix"], float)          # [T, n_arms]
        a = np.array(l["actions"], int)
        actual = pm[np.arange(T), a]
        opt, rnd = pm.max(1), pm.mean(1)
        denom = np.clip(opt - rnd, 1e-6, None)
        per_trial[e] = (actual - rnd) / denom
        rel_ep.append((actual - rnd).sum() / (opt - rnd).sum())
    rel_ep = np.array(rel_ep)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].hist(rel_ep, bins=30, color=MAN, alpha=0.85, edgecolor="white")
    ax[0].axvline(0, color="gray", ls="--", lw=1.2, label="chance")
    ax[0].axvline(rel_ep.mean(), color=POS, lw=2, label=f"mean {rel_ep.mean():+.2f}")
    ax[0].axvline(1, color="k", ls=":", lw=1, label="optimal")
    ax[0].set_xlabel("relative reward  (0 = random, 1 = optimal)")
    ax[0].set_ylabel("episodes")
    ax[0].set_title("Llama plays the 4-armed bandit above chance")
    ax[0].legend(loc="upper left")
    panel_tag(ax[0], "A")

    m = per_trial.mean(0)
    k = 7
    sm = np.convolve(m, np.ones(k) / k, mode="valid")
    xs = np.arange(len(sm)) + k // 2
    ax[1].axhline(0, color="gray", ls="--", lw=1.2)
    ax[1].plot(xs, sm, color=MAN, lw=2)
    ax[1].fill_between(xs, 0, sm, color=MAN, alpha=0.12)
    ax[1].set_xlabel("trial within episode")
    ax[1].set_ylabel("relative reward (mean)")
    ax[1].set_title("Sustained above-chance value-tracking")
    panel_tag(ax[1], "B")
    fig.tight_layout()
    fig.savefig(FIG / "report_fig1.png", dpi=200, bbox_inches="tight")
    print(f"fig1: mean rel reward {rel_ep.mean():+.3f}, {100*(rel_ep>0).mean():.0f}% episodes >0")


# ============================ Figure 2 — the latent ============================
def fig2():
    sc = json.loads((ART / "locate/llama8b_bestrl_scalars.json").read_text())
    layers = [e["layer"] for e in sc["layers"]]
    qmax = [e["r2_test"]["Q_max"] for e in sc["layers"]]
    qch = [e["r2_test"]["Q_chosen"] for e in sc["layers"]]
    best = int(np.argmax(qmax))

    iso = load(ART / "locate/llama8b_isometry.pt")
    Dh_m = iso["D_h_manifold"].numpy(); Dh_l = iso["D_h_linear"].numpy()
    Dy = iso["D_y_hellinger"].numpy()
    n = Dh_m.shape[0]; iu = np.triu_indices(n, 1)
    rm, rl = iso["r_manifold"], iso["r_linear"]

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].axhline(0, color="gray", ls="--", lw=1)
    ax[0].plot(layers, qmax, "-o", color=MAN, ms=3, lw=1.6, label="$Q_{max}$")
    ax[0].plot(layers, qch, "-o", color="#adb5bd", ms=2.5, lw=1.2, label="$Q_{chosen}$")
    ax[0].plot(best, qmax[best], "*", color=POS, ms=16,
               label=f"L{best}: R²={qmax[best]:.2f}")
    ax[0].set_ylim(-0.15, 1.0)
    ax[0].set_xlabel("layer (residual stream)"); ax[0].set_ylabel("ridge probe  $R^2$ (test)")
    ax[0].set_title("A value latent ($Q_{max}$) is decodable")
    ax[0].legend(loc="lower center")
    panel_tag(ax[0], "A")

    def norm(D):
        x = D[iu]; return x / x.std()
    ax[1].scatter(norm(Dh_l), norm(Dy), s=6, color=LIN, alpha=0.25,
                  label=f"linear   r={rl:.2f}")
    ax[1].scatter(norm(Dh_m), norm(Dy), s=6, color=MAN, alpha=0.35,
                  label=f"manifold r={rm:.2f}")
    lim = [min(norm(Dh_m).min(), norm(Dy).min()), max(norm(Dh_m).max(), norm(Dy).max())]
    ax[1].plot(lim, lim, color="gray", ls=":", lw=1)
    ax[1].set_xlabel("activation-space distance (norm.)")
    ax[1].set_ylabel("behavior-space distance (norm.)")
    ax[1].set_title("Activation ↔ behavior isometry ($Q_{max}$)")
    ax[1].legend(loc="upper left")
    panel_tag(ax[1], "B")
    fig.tight_layout()
    fig.savefig(FIG / "report_fig2.png", dpi=200, bbox_inches="tight")
    print(f"fig2: Q_max peak L{best} R²={qmax[best]:.2f}; isometry man r={rm:.2f} lin r={rl:.2f}")


# ============================ Figure 3 — causal steering ============================
def fig3():
    ic = load(ART / "steering/llama8b_interchange.pt")
    tgt = ic["target_arm"].numpy(); n = len(tgt); na = 4
    base = ic["P_base"].numpy()[np.arange(n), tgt]
    src = ic["P_source_cached"].numpy()[np.arange(n), tgt]
    L = 19
    d = ic["by_layer"][f"L{L}"]
    def dP(key):
        arr = d[key].numpy()
        tp = arr[np.arange(n), tgt] - base
        oth = np.array([[arr[r, j] - ic["P_base"].numpy()[r, j]
                         for j in range(na) if j != tgt[r]] for r in range(n)])
        return tp.mean(), tp.std() / np.sqrt(n), oth.mean()

    conds = [("P_identity", "identity\n(control)"), ("P_full", "full\npatch"), ("P_subspace", "subspace\npatch")]
    vals = [dP(k) for k, _ in conds]

    vr = load(ART / "steering/llama8b_value_vs_recency.pt")
    tg = vr["target_arm"].numpy(); m = len(tg)
    Pb = vr["P_base"].numpy()[np.arange(m), tg]
    cells = ["HV_HR", "HV_LR", "LV_HR"]
    clabs = ["value\n+recency", "value\nno recency", "recency\nno value"]
    cdP, cse = [], []
    for c in cells:
        pres = vr["src_meta"][c]["present"].numpy().astype(bool)
        arr = vr["by_layer"]["L19"]["subspace"][c].numpy()[np.arange(m), tg]
        dd = (arr - Pb)[pres]
        cdP.append(dd.mean()); cse.append(dd.std() / np.sqrt(pres.sum()))

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    x = np.arange(3)
    bars = ax[0].bar(x, [v[0] for v in vals], yerr=[v[1] for v in vals],
                     color=[ "#adb5bd", MAN, POS], width=0.62, capsize=3)
    ax[0].plot(x, [v[2] for v in vals], "D", color=LIN, ms=7, label="ΔP(other arms)")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xticks(x); ax[0].set_xticklabels([c[1] for c in conds])
    ax[0].set_ylabel("ΔP(target arm)")
    ax[0].set_title(f"Interchange: an identity-carrying\npatch moves the choice (L{L})")
    ax[0].legend(loc="upper left")
    ax[0].text(0.42, 0.23, f"P(target):\n0.33 → {base.mean()+vals[2][0]:.2f}\n(ceiling {src.mean():.2f})",
               fontsize=8.5, ha="left", va="center", color=POS)
    panel_tag(ax[0], "A")

    xc = np.arange(3)
    cols = [POS, MAN, LIN]
    ax[1].bar(xc, cdP, yerr=cse, color=cols, width=0.62, capsize=3)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(xc); ax[1].set_xticklabels(clabs)
    ax[1].set_ylabel("ΔP(target arm)")
    ax[1].set_title("The lever is value, not recency")
    ax[1].text(0.97, 0.95, "regression (β, standardized):\n"
               r"$\beta_{value}=+0.08\ (t{=}{+}14)$" + "\n" + r"$\beta_{recency}=-0.07\ (t{=}{-}14)$",
               transform=ax[1].transAxes, ha="right", va="top", fontsize=8,
               bbox=dict(boxstyle="round", fc="#f1f3f5", ec="none"))
    panel_tag(ax[1], "B")
    fig.tight_layout()
    fig.savefig(FIG / "report_fig3.png", dpi=200, bbox_inches="tight")
    print(f"fig3: interchange subspace ΔP={vals[2][0]:+.2f}; HV-LR={cdP[1]:+.3f} LV-HR={cdP[2]:+.3f}")


# ============================ Figure 4 — concept geometry ============================
def fig4():
    sx = load(ART / "manifolds/llama8b_fixed_simplex_L19.pt")
    Y = sx["nodes_y"].numpy(); H = sx["nodes_h"].numpy()
    letters = sx["letters"]; na = sx["n_arms"]
    edges = sx["edge_specs"]
    T = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float) / np.sqrt(3)
    Y3 = Y @ T

    fig = plt.figure(figsize=(10, 4.4))
    axY = fig.add_subplot(1, 2, 1, projection="3d")
    for (i, j, s, e, pk) in edges:
        axY.plot(*Y3[s:e + 1].T, color="gray", lw=1.6, alpha=0.7)
    for X in range(na):
        axY.scatter(*Y3[X], color=ARM[X], s=170, marker="D", edgecolor="k", lw=0.5)
        axY.text(*Y3[X], f"  {letters[X]}", fontsize=13, fontweight="bold")
    axY.set_title("Behavior space: 4-arm simplex")
    for a in (axY.xaxis, axY.yaxis, axY.zaxis):
        a.set_ticklabels([])
    axY.grid(False)
    axY.text2D(0.0, 0.98, "A", transform=axY.transAxes, fontsize=13, fontweight="bold")

    # one curved M_h edge (C→D) + chord — use the smoothed manifold for a clean illustration
    mf = load(ART / "manifolds/llama8b_fixed_pairs_L19_s250.pt")
    pd = mf["pairs"]["1_2"]
    g = pd["manifold_grid"].numpy()[:, :3]
    rg = pd["r_grid"].numpy()
    arc = np.linalg.norm(np.diff(g, axis=0), axis=1).sum(); chord = np.linalg.norm(g[-1] - g[0])
    sub = np.linspace(0, len(g) - 1, 40).round().astype(int)
    axH = fig.add_subplot(1, 2, 2, projection="3d")
    axH.plot(*g.T, color="gray", lw=1, alpha=0.5)
    sc = axH.scatter(*g[sub].T, c=rg[sub], cmap="viridis", s=22)
    axH.plot(*np.stack([g[0], g[-1]]).T, color=LIN, ls="--", lw=2, label="linear chord")
    axH.scatter(*g[0], color=ARM[1], s=120, marker="D", edgecolor="k", lw=0.5)
    axH.scatter(*g[-1], color=ARM[2], s=120, marker="D", edgecolor="k", lw=0.5)
    axH.set_title("Activation space: $M_h$ for C→D\n(curved; chord cuts off-manifold)")
    for a in (axH.xaxis, axH.yaxis, axH.zaxis):
        a.set_ticklabels([])
    axH.grid(False); axH.legend(loc="upper left", fontsize=8)
    fig.colorbar(sc, ax=axH, shrink=0.5, pad=0.08, label="r = Q[D] − Q[C]")
    axH.text2D(0.0, 0.98, "B", transform=axH.transAxes, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "report_fig4.png", dpi=200, bbox_inches="tight")
    print("fig4: simplex + curved M_h (C→D)")


# ============================ Figure 5 — manifold vs linear ============================
def bc_nearest(p, grid):
    ov = (np.sqrt(p)[None] * np.sqrt(grid)).sum(1)
    return -np.log(np.clip(ov, 1e-12, 1.0)).min()


def fig5():
    mf = load(ART / "manifolds/llama8b_fixed_pairs_L19.pt")
    st = load(ART / "steering/llama8b_fixed_pair_steering_L19.pt")
    res, K, na = st["results"], st["K"], st["n_arms"]
    pk = "1_2"; R = res[pk]; i, j = R["i"], R["j"]; li, lj = R["letters"]
    rest = [a for a in range(na) if a not in (i, j)]
    dg = mf["pairs"][pk]["dist_grid"].numpy()
    frac = np.linspace(0, 1, K)

    def arms(mode):
        d = R[f"dist_{mode}"].numpy().mean(0)
        return d[:, :na] / d[:, :na].sum(1, keepdims=True)
    am, al = arms("manifold"), arms("linear_full")

    # aggregate both metrics over pairs
    ebc = {"manifold": [], "linear_full": []}
    trk = {"manifold": [], "linear_full": []}
    for k, Rp in res.items():
        dgp = mf["pairs"][k]["dist_grid"].numpy()
        tgt = dgp[np.linspace(0, len(dgp) - 1, K).round().astype(int)]
        for mode in ("manifold", "linear_full"):
            d = Rp[f"dist_{mode}"].numpy()
            ar = d[..., :na] / d[..., :na].sum(-1, keepdims=True)
            ebc[mode].append(np.mean([sum(bc_nearest(ar[b, w], dgp) for w in range(K))
                                      for b in range(ar.shape[0])]))
            trk[mode].append(((ar - tgt[None]) ** 2).mean())
    e_m, e_l = np.mean(ebc["manifold"]), np.mean(ebc["linear_full"])
    t_m, t_l = np.mean(trk["manifold"]) * 1e3, np.mean(trk["linear_full"]) * 1e3

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    ax[0].plot(np.linspace(0, 1, len(dg)), dg[:, j], color=MY, lw=2, alpha=0.55, label="$M_y$ (natural)")
    ax[0].plot(frac, am[:, j], color=MAN, lw=2.5, label="manifold")
    ax[0].plot(frac, al[:, j], color=LIN, lw=2.5, ls="--", label="linear")
    ax[0].set_xlabel(f"path fraction ({li} → {lj})")
    ax[0].set_ylabel(f"P(arm {lj})")
    ax[0].set_title(f"Steering {li}→{lj}: choice mass\nfollows the natural transition")
    ax[0].legend(loc="upper left"); ax[0].set_ylim(-0.02, 1)
    panel_tag(ax[0], "A")

    # normalize each metric to its own max so both fit one axis (lower = more natural)
    nm = [e_m / max(e_m, e_l), t_m / max(t_m, t_l)]
    nl = [e_l / max(e_m, e_l), t_l / max(t_m, t_l)]
    x = np.arange(2); w = 0.36
    ax[1].bar(x - w/2, nm, w, color=MAN, label="manifold")
    ax[1].bar(x + w/2, nl, w, color=LIN, label="linear")
    for xi, (vm, vl, rm, rl) in enumerate(zip(nm, nl, [e_m, t_m], [e_l, t_l])):
        ax[1].text(xi - w/2, vm + 0.02, f"{rm:.2g}", ha="center", fontsize=7.5, color=MAN)
        ax[1].text(xi + w/2, vl + 0.02, f"{rl:.2g}", ha="center", fontsize=7.5, color=LIN)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(["$E_{BC}$\n(nearest-pt, pacing-blind)",
                           "tracking MSE\n(order-aware)"])
    ax[1].set_ylabel("relative value  (lower = more natural)")
    ax[1].set_ylim(0, 1.2)
    ax[1].text(0, 1.10, "linear\nbetter (barely)", ha="center", fontsize=8, color=LIN)
    ax[1].text(1, 1.10, "manifold\nbetter", ha="center", fontsize=8, color=MAN)
    ax[1].set_title("Two metrics disagree by design")
    ax[1].legend(loc="lower left")
    panel_tag(ax[1], "B")
    fig.tight_layout()
    fig.savefig(FIG / "report_fig5.png", dpi=200, bbox_inches="tight")
    print(f"fig5: E_BC man {e_m:.2f} lin {e_l:.2f}; tracking MSE man {t_m:.2f} lin {t_l:.2f}")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fig1(); fig2(); fig3(); fig4(); fig5()
    print("\nall report figures saved.")
