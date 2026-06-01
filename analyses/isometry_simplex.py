"""4-arm simplex isometry — the bandit analog of the paper's weekday isometry figure.

The bandit's concept geometry is not a cyclic loop (the 4 arms are symmetric); it is a
SIMPLEX: 4 "arm-X-is-best" vertices joined by 6 value-transition edges (the pair manifolds).
We test whether this simplex appears isometrically in activation space (M_h) and behavior
space (M_y), and whether the on-manifold (geodesic) structure matches behavior better than
the straight-line (linear) structure.

Construction (model-free; uses fitted pair manifolds + cached activations):
  vertices  arm-X-best centroid: mean over trials where X=argmax(Q) with top-tercile margin,
            in PCA-64 (activation) and on the simplex (behavior).
  edges     each pair's M_h grid (activation) and M_y dist_grid (behavior), subsampled.
  graph     nodes = 4 vertices + 6×M edge points; intra-edge links + edge-end→vertex links,
            weighted by PCA-64 Euclidean distance.
  D_h_manifold  graph-geodesic distance (along the simplex edges).
  D_h_linear    straight-line PCA-64 Euclidean distance (what linear steering sees).
  D_y           Hellinger distance between the nodes' behavior distributions.
Isometry = Pearson(D_h, D_y) for manifold vs linear. Classical MDS of each distance matrix
gives the structural-correspondence embeddings.

    python -m analyses.isometry_simplex \
        --pairs artifacts/manifolds/llama8b_fixed_pairs_L19.pt \
        --activations artifacts/activations/llama8b_fixed_residual.pt \
        --q artifacts/value/bestrl_Q_fixed.pt \
        --dataset artifacts/datasets/llama8b_fixed.json \
        --out artifacts/manifolds/llama8b_fixed_simplex_L19.pt
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch
from scipy.sparse.csgraph import shortest_path
from scipy.stats import pearsonr, spearmanr


def classical_mds(D, k=3):
    """Classical (Torgerson) MDS: distance matrix D [n,n] → coords [n,k]."""
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    vals, vecs = np.linalg.eigh(B)
    idx = np.argsort(vals)[::-1][:k]
    L = np.clip(vals[idx], 0, None)
    return vecs[:, idx] * np.sqrt(L)


def hellinger_matrix(P):
    """Pairwise Hellinger distance among rows of P [n, d] (distributions)."""
    S = np.sqrt(P)
    G = S @ S.T
    D2 = np.clip(2 - 2 * G, 0, None)
    return np.sqrt(D2) / np.sqrt(2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=pathlib.Path, required=True)
    ap.add_argument("--activations", type=pathlib.Path, required=True)
    ap.add_argument("--q", type=pathlib.Path, required=True)
    ap.add_argument("--dataset", type=pathlib.Path, required=True)
    ap.add_argument("--edge-points", type=int, default=12, help="samples per edge")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    mf = torch.load(args.pairs, weights_only=False)
    pca_basis = mf["pca_basis"].numpy().astype(np.float32)        # [pca_dim, 4096]
    pca_mean = mf["pca_mean"].numpy().astype(np.float32)
    L = mf["layer"]
    letters = mf["letters"]
    n_arms = mf["n_arms"]
    pairs = mf["pairs"]

    # ---- canonical vertices (arm-X-best) from raw data ----
    act = torch.load(args.activations, weights_only=False)
    A = act["activations"][:, 1:, L, :].float().reshape(-1, 4096).numpy()
    A_pca = ((A - pca_mean) @ pca_basis.T).astype(np.float32)      # [N, pca_dim]
    Q = torch.load(args.q, weights_only=False)["Q"][act["episode_ids"]].cpu().numpy()
    Qp = Q[:, 1:].reshape(-1, n_arms)
    logs = [json.loads(args.dataset.read_text())[e] for e in act["episode_ids"]]
    choice = np.array([l["choice_dists"] for l in logs], dtype=np.float32)[:, 1:, :n_arms]
    choice = (choice / choice.sum(-1, keepdims=True)).reshape(-1, n_arms)

    best = Qp.argmax(1)
    sortedQ = -np.sort(-Qp, axis=1)
    margin = sortedQ[:, 0] - sortedQ[:, 1]
    vert_h, vert_y = [], []
    for X in range(n_arms):
        m = best == X
        thr = np.quantile(margin[m], 2 / 3)
        sel = m & (margin >= thr)
        vert_h.append(A_pca[sel].mean(0))
        vert_y.append(choice[sel].mean(0))
        print(f"  vertex {letters[X]}: n={int(sel.sum())}, P({letters[X]})={vert_y[-1][X]:.2f}")
    vert_h = np.stack(vert_h)
    vert_y = np.stack(vert_y)

    # ---- assemble nodes: 4 vertices + 6 edges × M points ----
    nodes_h = [vert_h]
    nodes_y = [vert_y]
    node_arm = list(range(n_arms))                  # vertex coloring
    node_type = ["vertex"] * n_arms
    edge_specs = []                                  # (i, j, start_idx, end_idx) into node list
    M = args.edge_points
    for pk, pd in pairs.items():
        i, j = pd["i"], pd["j"]
        g = pd["manifold_grid"].numpy()
        dy = pd["dist_grid"].numpy()
        sub = np.linspace(0, len(g) - 1, M).round().astype(int)
        base = sum(len(x) for x in nodes_h)
        nodes_h.append(g[sub])
        nodes_y.append(dy[sub])
        node_arm += [-1] * M
        node_type += [pk] * M
        edge_specs.append((i, j, base, base + M - 1, pk))
    H = np.concatenate(nodes_h)                      # [n, pca_dim]
    Y = np.concatenate(nodes_y)                      # [n, n_arms]
    n = len(H)

    # ---- build the simplex graph (weighted by PCA-64 Euclidean) ----
    W = np.full((n, n), np.inf)
    np.fill_diagonal(W, 0)
    def link(a, b):
        d = np.linalg.norm(H[a] - H[b])
        W[a, b] = W[b, a] = d
    for (i, j, s, e, pk) in edge_specs:
        for p in range(s, e):                        # consecutive edge points
            link(p, p + 1)
        link(i, s)                                   # strong-i end → vertex i
        link(j, e)                                   # strong-j end → vertex j
    D_h_manifold = shortest_path(W, method="D", directed=False)
    D_h_linear = np.linalg.norm(H[:, None, :] - H[None, :, :], axis=2)
    D_y = hellinger_matrix(Y)

    # ---- isometry (upper triangle) ----
    iu = np.triu_indices(n, k=1)
    r_man, p_man = pearsonr(D_h_manifold[iu], D_y[iu])
    r_lin, p_lin = pearsonr(D_h_linear[iu], D_y[iu])
    rho_man = spearmanr(D_h_manifold[iu], D_y[iu]).correlation
    rho_lin = spearmanr(D_h_linear[iu], D_y[iu]).correlation
    print(f"\nIsometry (n={n} nodes, {len(iu[0])} pairs):")
    print(f"  manifold  Pearson r={r_man:.3f}  Spearman={rho_man:.3f}")
    print(f"  linear    Pearson r={r_lin:.3f}  Spearman={rho_lin:.3f}")

    mds = {
        "behavior":          classical_mds(D_y),
        "activation_manifold": classical_mds(D_h_manifold),
        "activation_linear": classical_mds(D_h_linear),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "nodes_h": torch.tensor(H), "nodes_y": torch.tensor(Y),
        "node_arm": node_arm, "node_type": node_type,
        "edge_specs": [(i, j, s, e, pk) for (i, j, s, e, pk) in edge_specs],
        "vert_h": torch.tensor(vert_h), "vert_y": torch.tensor(vert_y),
        "D_h_manifold": torch.tensor(D_h_manifold), "D_h_linear": torch.tensor(D_h_linear),
        "D_y": torch.tensor(D_y),
        "mds": {k: torch.tensor(v) for k, v in mds.items()},
        "r_manifold": float(r_man), "r_linear": float(r_lin),
        "rho_manifold": float(rho_man), "rho_linear": float(rho_lin),
        "layer": L, "letters": letters, "n_arms": n_arms, "edge_points": M,
        "model_id": act["model_id"],
    }, args.out)
    print(f"\nsaved → {args.out}")


if __name__ == "__main__":
    main()
