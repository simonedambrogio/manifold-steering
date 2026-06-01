"""Isometry test: compare geodesic distances on M_h and M_y between bin-centroid pairs.

Three distance matrices over the shared bin centroids:
  - **D_h_manifold**  — cumulative Euclidean distance along the M_h spline in 64-D PCA space.
  - **D_h_linear**    — straight-line Euclidean distance between the same 64-D centroids
                        (off-manifold baseline — what linear steering "sees").
  - **D_y_hellinger** — cumulative Hellinger distance along the M_y sphere spline.

Then Pearson correlations:
  r(D_h_manifold, D_y_hellinger)  — the headline result (paper's were 0.89 to 0.99).
  r(D_h_linear,   D_y_hellinger)  — baseline; lower r demonstrates the manifold structure matters.

    python -m analyses.isometry \
        --manifold-h artifacts/manifolds/llama8b_layer19_Qmax.pt \
        --manifold-y artifacts/manifolds/llama8b_My_Qmax.pt \
        --out artifacts/locate/llama8b_isometry.pt
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr


def cumulative_arc_length(grid: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """`arc[i]` = cumulative Euclidean distance from grid[0] to grid[i], times `scale`."""
    diffs = np.diff(grid, axis=0)
    seg = np.linalg.norm(diffs, axis=1) * scale
    return np.concatenate([[0.0], np.cumsum(seg)])


def metric_suite(x: np.ndarray, y: np.ndarray) -> dict:
    """Pearson r + R² + nRMSE-of-residuals + Spearman ρ + curvature gap.

    nRMSE: RMSE of residuals from the OLS line, normalised by std(y). Measures the
           *typical deviation from a straight line*, scaled by data spread.
    Spearman − Pearson: positive when the relationship is monotone but non-linear
           (the "structured deviations" / curvature signature).
    """
    r, p = pearsonr(x, y)
    rho, _ = spearmanr(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    nrmse = float(np.sqrt(np.mean(resid ** 2)) / np.std(y))
    return {
        "pearson_r":   float(r),
        "pearson_p":   float(p),
        "r_squared":   float(r ** 2),
        "one_minus_r2": float(1.0 - r ** 2),
        "spearman_rho": float(rho),
        "spearman_minus_pearson": float(rho - r),
        "nrmse":       nrmse,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifold-h", type=pathlib.Path, required=True)
    parser.add_argument("--manifold-y", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    mh = torch.load(args.manifold_h, weights_only=False)
    my = torch.load(args.manifold_y, weights_only=False)

    # ---- arc lengths along each manifold's spline ----
    manifold_h_grid = mh["manifold_grid"].cpu().numpy()                           # [n_grid, 64]
    sphere_grid = my["sphere_grid"].cpu().numpy()                                  # [n_grid, n_arms]
    q_grid_h = mh["manifold_q_grid"].cpu().numpy()
    q_grid_y = my["manifold_q_grid"].cpu().numpy()
    assert np.allclose(q_grid_h, q_grid_y), "M_h and M_y must share the Q_max grid"
    q_grid = q_grid_h

    arc_h = cumulative_arc_length(manifold_h_grid, scale=1.0)                      # Euclidean in PCA space
    arc_y = cumulative_arc_length(sphere_grid, scale=1.0 / np.sqrt(2))             # Hellinger

    # ---- bin centers (assert they match between M_h and M_y) ----
    bc_h = mh["bin_centers"].cpu().numpy()
    bc_y = my["bin_centers"].cpu().numpy()
    assert np.allclose(bc_h, bc_y), "M_h and M_y bin centers should match"
    bin_centers = bc_h
    n_bins = len(bin_centers)

    grid_idx = np.array([int(np.argmin(np.abs(q_grid - bc))) for bc in bin_centers])

    # ---- pairwise distance matrices ----
    D_h_manifold = np.abs(arc_h[grid_idx][:, None] - arc_h[grid_idx][None, :])
    D_y_hellinger = np.abs(arc_y[grid_idx][:, None] - arc_y[grid_idx][None, :])

    # Linear distance in 64-D PCA space (off-manifold baseline)
    centroids_pca = mh["centroids_pca"].cpu().numpy()
    diff = centroids_pca[:, None, :] - centroids_pca[None, :, :]
    D_h_linear = np.linalg.norm(diff, axis=-1)

    # ---- metric suites on upper-triangular off-diagonal pairs ----
    iu = np.triu_indices(n_bins, k=1)
    m_manifold = metric_suite(D_h_manifold[iu], D_y_hellinger[iu])
    m_linear = metric_suite(D_h_linear[iu], D_y_hellinger[iu])

    print(f"Isometry test (n_bins = {n_bins}, n_pairs = {len(iu[0])}):\n")
    hdr = f"{'metric':<28} {'manifold':>12} {'linear':>12} {'manifold−linear':>18}"
    print(hdr); print("-" * len(hdr))
    for key, label in [
        ("pearson_r",               "Pearson r"),
        ("r_squared",               "R²"),
        ("one_minus_r2",            "1 − R²"),
        ("spearman_rho",            "Spearman ρ"),
        ("spearman_minus_pearson",  "Spearman ρ − Pearson r"),
        ("nrmse",                   "nRMSE of line residuals"),
    ]:
        mm, ll = m_manifold[key], m_linear[key]
        # for nrmse and 1-R² and spearman-pearson, "lower is better" for manifold;
        # for r, R², spearman, "higher is better" for manifold.
        diff = mm - ll
        arrow = "↓" if key in {"nrmse", "one_minus_r2"} and diff < 0 else (
                "↑" if key not in {"nrmse", "one_minus_r2", "spearman_minus_pearson"} and diff > 0 else "")
        print(f"{label:<28} {mm:>+12.4f} {ll:>+12.4f} {diff:>+18.4f} {arrow}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "D_h_manifold":  torch.tensor(D_h_manifold, dtype=torch.float32),
        "D_h_linear":    torch.tensor(D_h_linear, dtype=torch.float32),
        "D_y_hellinger": torch.tensor(D_y_hellinger, dtype=torch.float32),
        "bin_centers":   torch.tensor(bin_centers, dtype=torch.float32),
        "metrics_manifold": m_manifold,
        "metrics_linear":   m_linear,
        # legacy keys for compatibility with prior runs / figure
        "r_manifold":    m_manifold["pearson_r"],
        "r_linear":      m_linear["pearson_r"],
        "p_manifold":    m_manifold["pearson_p"],
        "p_linear":      m_linear["pearson_p"],
        "n_bins":        n_bins,
        "n_pairs":       int(len(iu[0])),
        "model_id":      mh["model_id"],
        "value_model":   mh["value_model"],
    }, args.out)
    print(f"\nsaved → {args.out}")


if __name__ == "__main__":
    main()
