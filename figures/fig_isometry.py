"""Isometry figure (paper Figs 2/3-style): distance-vs-distance scatter, manifold vs linear.

Two side-by-side panels — one dot per pair of bin centroids:

  Left  — D_h_manifold (geodesic on M_h) vs D_y_hellinger.  Pearson r in title.
  Right — D_h_linear  (straight-line in PCA) vs D_y_hellinger.  Pearson r in title.

The contrast `r_manifold ≫ r_linear` is the paper's headline finding: the *geometry*
of activations (not just their straight-line distance) is what aligns with behavior.

    python -m figures.fig_isometry \
        --isometry artifacts/locate/llama8b_isometry.pt \
        --out artifacts/figures/fig_isometry.png
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--isometry", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    d = torch.load(args.isometry, weights_only=False)
    Dh_m = d["D_h_manifold"].cpu().numpy()
    Dh_l = d["D_h_linear"].cpu().numpy()
    Dy = d["D_y_hellinger"].cpu().numpy()
    n_bins = int(d["n_bins"])
    bin_centers = d["bin_centers"].cpu().numpy()
    m_manifold = d.get("metrics_manifold")
    m_linear = d.get("metrics_linear")

    iu = np.triu_indices(n_bins, k=1)
    pair_mean_q = (bin_centers[iu[0]] + bin_centers[iu[1]]) / 2

    fig, (ax_m, ax_l) = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True)

    def _title(name: str, metrics: dict | None, fallback_r: float) -> str:
        if metrics is None:
            return f"{name}\nPearson r = {fallback_r:+.3f}"
        return (
            f"{name}\n"
            f"r = {metrics['pearson_r']:+.3f}   "
            f"R² = {metrics['r_squared']:.3f}   "
            f"nRMSE = {metrics['nrmse']:.3f}\n"
            f"Spearman ρ = {metrics['spearman_rho']:+.3f}   "
            f"ρ − r = {metrics['spearman_minus_pearson']:+.4f}"
        )

    for ax, x_data, xlabel, title in [
        (ax_m, Dh_m[iu],
         "On-manifold distance on M_h (Euclidean in 64-D PCA)",
         _title("Manifold vs behavior", m_manifold, float(d["r_manifold"]))),
        (ax_l, Dh_l[iu],
         "Linear distance in 64-D PCA (off-manifold baseline)",
         _title("Linear vs behavior",   m_linear,   float(d["r_linear"]))),
    ]:
        sc = ax.scatter(
            x_data, Dy[iu], c=pair_mean_q, cmap="viridis",
            s=18, alpha=0.75, edgecolor="black", linewidth=0.25,
        )
        # 1-D fit line through origin to guide the eye
        if len(x_data) > 1:
            slope = np.sum(x_data * Dy[iu]) / np.sum(x_data * x_data)
            xs = np.array([0, x_data.max()])
            ax.plot(xs, slope * xs, color="gray", ls="--", lw=0.7, alpha=0.7)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=10)
        fig.colorbar(sc, ax=ax, label="mean Q_max of pair")

    ax_m.set_ylabel("On-manifold distance on M_y (Hellinger)")

    fig.suptitle(
        f"Isometry test — {d['model_id']} → {d['value_model']} Q_max  "
        f"({n_bins} bin centroids, {d['n_pairs']} pairs)",
        fontsize=10,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"saved → {args.out}")
    if m_manifold and m_linear:
        print(
            f"  Δ(1-R²) = {m_linear['one_minus_r2'] - m_manifold['one_minus_r2']:+.4f}  "
            f"Δ(nRMSE) = {m_linear['nrmse'] - m_manifold['nrmse']:+.4f}  "
            f"Δ(ρ-r)  = {m_linear['spearman_minus_pearson'] - m_manifold['spearman_minus_pearson']:+.4f}"
        )


if __name__ == "__main__":
    main()
