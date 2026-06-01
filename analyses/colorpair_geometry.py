"""Geometry of the learned color encoder phi: did a fair model fold the colors into a ring?

We never told the model colors are circular -- the input is one-hot, the comparison is a free MLP. So
the geometry of phi (the per-color embedding) is an honest read on what the reward structure forced. We
ask three things, numpy-only:

1. **Ring vs line** -- pairwise Euclidean distances between color embeddings vs circular and linear index
   distance. Decisive statistic: wraparound ratio R = d(color 0, color N-1) / mean d(adjacent colors).
   R ~ 1 for a closed ring (the seam colors are neighbours), R ~ N-1 for an open line.
2. **Decodability** -- leave-one-out circular ridge decode of the color angle from phi (mean cos error).
3. **Scaled isometry** -- finely-resampled ring arc-length (the manifold metric) vs true circular
   distance (r high) and chord length vs true circular distance (r lower): the paper's r=0.99/0.89 panel.
   Plus the effective dimensionality of phi (PCA variance in the top 2 components).

    .venv/bin/python -m analyses.colorpair_geometry --checkpoint artifacts/agent/colorpair_optB.pt
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch


def _pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def ring_resample(cents, n_out):
    """Resample the closed polyline through ordered points to n_out points by arc length."""
    closed = np.vstack([cents, cents[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, cum[-1], n_out, endpoint=False)
    return np.stack([np.interp(t, cum, closed[:, j]) for j in range(closed.shape[1])], axis=1)


def arc_dist_matrix(pts):
    """Geodesic distance along the closed ring through consecutively-ordered points."""
    n = len(pts)
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)]); total = cum[-1]
    d = np.abs(cum[:n, None] - cum[:n][None, :])
    return np.minimum(d, total - d)


def chord_dist_matrix(pts):
    return np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)


def loo_circular_decode(phi, theta, lam=1.0):
    """Leave-one-out circular ridge decode of angle theta from phi. Returns mean cos of angular error."""
    n = len(phi)
    Y = np.stack([np.cos(theta), np.sin(theta)], 1)
    cos_err = []
    for i in range(n):
        tr = np.arange(n) != i
        X = phi[tr]; mu = X.mean(0); sd = X.std(0) + 1e-6
        Xs = (X - mu) / sd
        W = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ (Y[tr] - Y[tr].mean(0)))
        yh = ((phi[i] - mu) / sd) @ W + Y[tr].mean(0)
        th = np.arctan2(yh[1], yh[0])
        cos_err.append(np.cos(theta[i] - th))
    return float(np.mean(cos_err))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=pathlib.Path, default=pathlib.Path("artifacts/agent/colorpair_optB.pt"))
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, weights_only=False)
    phi = ckpt["phi"].numpy().astype(np.float64)              # [N, k]
    n = ckpt["config"]["n_colors"]
    theta = 2 * np.pi * np.arange(n) / n

    # 1. ring vs line on the raw per-color embeddings
    D = chord_dist_matrix(phi)
    iu = np.triu_indices(n, 1)
    di, dj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    lin = np.abs(di - dj).astype(float)
    circ = np.minimum(lin, n - lin)
    adj = np.array([D[i, i + 1] for i in range(n - 1)])
    wrap = D[0, n - 1]
    print(f"N={n}, k={phi.shape[1]}")
    print(f"  corr(embed dist, circular)  = {_pearson(D[iu], circ[iu]):+.3f}")
    print(f"  corr(embed dist, linear)    = {_pearson(D[iu], lin[iu]):+.3f}")
    print(f"  wraparound ratio R          = {wrap / adj.mean():.3f}  (~1 ring, ~{n - 1} line)")

    # 2. decodability
    print(f"  LOO circular decode mean-cos = {loo_circular_decode(phi, theta):+.3f}  (1 = perfect)")

    # 3. scaled isometry + effective dim
    fine = ring_resample(phi, 120)
    fi = np.triu_indices(120, 1)
    arc = arc_dist_matrix(fine)[fi]; chord = chord_dist_matrix(fine)[fi]
    # true circular distance between the resampled ring positions (uniform in arc by construction)
    ang = 2 * np.pi * np.arange(120) / 120
    da = np.abs(ang[:, None] - ang[None, :]); true_circ = np.minimum(da, 2 * np.pi - da)[fi]
    r_man = _pearson(arc, true_circ); r_lin = _pearson(chord, true_circ)
    print(f"  scaled isometry r: manifold(arc)={r_man:.3f}  linear(chord)={r_lin:.3f}")

    mu = phi.mean(0); s = np.linalg.svd(phi - mu, compute_uv=False)
    var = s ** 2 / (s ** 2).sum()
    print(f"  PCA var top-2 = {var[:2].sum():.3f} ({var[0]:.3f}, {var[1]:.3f}); top-3 = {var[:3].sum():.3f}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"phi": ckpt["phi"], "n_colors": n,
                    "wraparound_ratio": float(wrap / adj.mean()),
                    "r_manifold": r_man, "r_linear": r_lin, "pca_var": torch.tensor(var)}, args.out)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
