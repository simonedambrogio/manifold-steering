"""Fit confidence manifolds — the value-magnitude axis, per target arm (fixed letters).

For each arm a, restrict to trials where a is best (argmax Q = a) and parameterize by the
confidence coordinate g = Q[a] − Q_2nd (a's advantage over the runner-up, which drives the
softmax P(a)). Low g = uncertain, high g = confident. Because we condition on a-best AND use
fixed letters, the centroid means "a is best by this much" — an identity-coherent, steerable
target, unlike the failed identity-blind scalar-Q_max centroid.

  M_h: PCA-64 centroids of a-best activations binned by g → spline.
  M_y: matching mean choice distributions → Hellinger spline (a curve toward the a-vertex).

Emits the SAME artifact schema as fit_pair_manifold (pairs[key] with manifold_grid, r_grid
[=g], centroids_pca, dist_grid, i, j, ...) so steer_pair.py / eval_pair_steering.py run on it
unchanged. For confidence we set i = j = a (so P(j) = P(a) is the tracked quantity and the
"off-axis" arms are the three non-a arms).

    python -m analyses.fit_conf_manifold \
        --activations artifacts/activations/llama8b_fixed_residual.pt \
        --q artifacts/value/bestrl_Q_fixed.pt \
        --dataset artifacts/datasets/llama8b_fixed.json \
        --layer 19 --n-bins 20 \
        --out artifacts/manifolds/llama8b_fixed_conf_L19.pt
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch

from analyses.fit_pair_manifold import _quantile_edges, _bin_centroids, _fit_splines
from analyses.fit_behavior_manifold import exp_map, log_map


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--activations", type=pathlib.Path, required=True)
    parser.add_argument("--q", type=pathlib.Path, required=True)
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--n-grid", type=int, default=200)
    parser.add_argument("--smoothing", type=float, default=None)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    act_data = torch.load(args.activations, weights_only=False)
    activations = act_data["activations"]
    ep_ids = act_data["episode_ids"]
    n_eps, T, n_layers, hidden = activations.shape

    q_data = torch.load(args.q, weights_only=False)
    Q = q_data["Q"][ep_ids].cpu().numpy()
    n_arms = Q.shape[2]

    logs = [json.loads(args.dataset.read_text())[e] for e in ep_ids]
    choice = np.array([l["choice_dists"] for l in logs], dtype=np.float32)[..., :n_arms]
    choice = choice / choice.sum(axis=-1, keepdims=True)
    letters = logs[0]["letters"]

    A = activations[:, 1:, args.layer, :].float().reshape(-1, hidden)
    A_mean = A.mean(dim=0)
    _, S, Vh = torch.linalg.svd(A - A_mean, full_matrices=False)
    pca_basis = Vh[: args.pca_dim]
    var_exp = (S ** 2) / (S ** 2).sum()
    A_pca = ((A - A_mean) @ pca_basis.T).cpu().numpy()
    print(f"layer {args.layer}: PCA-{args.pca_dim} cum var "
          f"{float(var_exp[:args.pca_dim].sum()):.3f}; letters={letters}")

    Qp = Q[:, 1:].reshape(-1, n_arms)
    Cp = choice[:, 1:].reshape(-1, n_arms)
    order = np.argsort(-Qp, axis=1)
    best = order[:, 0]
    sortedQ = -np.sort(-Qp, axis=1)
    gap = sortedQ[:, 0] - sortedQ[:, 1]                       # Q_max - Q_2nd

    pairs = {}
    for a in range(n_arms):
        sel = best == a
        n = int(sel.sum())
        if n < args.n_bins * 3:
            print(f"  arm {letters[a]}: only {n} best trials — skipped")
            continue
        g = gap[sel].astype(np.float64)
        A_arm, C_arm = A_pca[sel], Cp[sel]
        edges = _quantile_edges(g, args.n_bins)
        cen_pca, g_centers, counts = _bin_centroids(g, A_arm, edges)
        cen_dist, _, _ = _bin_centroids(g, C_arm, edges)
        n_used = len(g_centers)
        w = np.sqrt(counts.astype(np.float64))
        s = float(args.smoothing) if args.smoothing is not None else float(n_used)

        g_grid, manifold_grid = _fit_splines(g_centers, cen_pca, w, s, args.n_grid)

        sqrt_c = np.sqrt(cen_dist)
        b_star = sqrt_c.mean(axis=0); b_star = b_star / np.linalg.norm(b_star)
        tangents = log_map(b_star, sqrt_c)
        _, tangent_grid = _fit_splines(g_centers, tangents, w, s, args.n_grid)
        sphere_grid = exp_map(b_star, tangent_grid)
        dist_grid = sphere_grid ** 2
        dist_grid = dist_grid / dist_grid.sum(axis=1, keepdims=True)

        pairs[f"arm{a}"] = {
            "i": a, "j": a, "letters": [letters[a], letters[a]], "n": n,
            "centroids_pca": torch.tensor(cen_pca, dtype=torch.float32),
            "manifold_grid": torch.tensor(manifold_grid, dtype=torch.float32),
            "r_grid":        torch.tensor(g_grid, dtype=torch.float32),
            "r_centers":     torch.tensor(g_centers, dtype=torch.float32),
            "r_counts":      torch.tensor(counts, dtype=torch.long),
            "dist_centroids": torch.tensor(cen_dist, dtype=torch.float32),
            "dist_grid":     torch.tensor(dist_grid, dtype=torch.float32),
            "sphere_grid":   torch.tensor(sphere_grid, dtype=torch.float32),
            "base_point":    torch.tensor(b_star, dtype=torch.float32),
            "tangent_centroids": torch.tensor(tangents, dtype=torch.float32),
        }
        print(f"  arm {letters[a]}: n={n}, {n_used} bins; "
              f"P({letters[a]})→ {dist_grid[0, a]:.2f}@g_min → {dist_grid[-1, a]:.2f}@g_max")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "pairs":            pairs,
        "pca_basis":        pca_basis.cpu(),
        "pca_mean":         A_mean.cpu(),
        "pca_var_explained": var_exp[: args.pca_dim].cpu(),
        "layer":            args.layer,
        "letters":          letters,
        "n_arms":           n_arms,
        "n_bins":           args.n_bins,
        "n_grid":           args.n_grid,
        "coordinate":       "gap_Qmax_minus_Q2nd",
        "model_id":         act_data["model_id"],
        "value_model":      q_data["model_name"],
    }, args.out)
    print(f"\nsaved → {args.out}  ({len(pairs)} arms)")


if __name__ == "__main__":
    main()
