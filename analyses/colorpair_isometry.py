"""Scaled isometry M_h <-> M_y for the color-pair task (the paper's panel (a), right two columns).

Both manifolds are sampled on the SAME ordered ring (resampled by arc length so vertex a in M_h matches
vertex a in M_y). We then measure, between every pair of vertices:
  D_Y[a,b]          = arc length along the BEHAVIOR ring M_y           (the reference)
  D_X_manifold[a,b] = arc length along the ACTIVATION ring M_h (geodesic)
  D_X_linear[a,b]   = straight chord between M_h[a], M_h[b]   (cuts the ring interior)
Pearson correlation over the strict upper triangle gives r_manifold and r_linear (the same D_Y, two
metrics on M_h). Scaling D_X to D_Y units is cosmetic (Pearson is scale-invariant). We also export the
geodesic and chord steering paths (A -> antipodal B) in M_h's 3D coords for the MDS path panels.

    .venv/bin/python -m analyses.colorpair_isometry --data artifacts/manifolds/colorpair_behavior.pt \
        --out artifacts/manifolds/colorpair_isometry.pt
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch


def ring_resample(cents, n_out):
    closed = np.vstack([cents, cents[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, cum[-1], n_out, endpoint=False)
    return np.stack([np.interp(t, cum, closed[:, j]) for j in range(closed.shape[1])], axis=1)


def polyline_resample(pts, n_out):
    """Resample an OPEN polyline (no wraparound) by arc length -- for the geodesic A->B path."""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, cum[-1], n_out)
    return np.stack([np.interp(t, cum, pts[:, j]) for j in range(pts.shape[1])], axis=1)


def arc_dist_matrix(pts):
    n = len(pts)
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)]); total = cum[-1]
    d = np.abs(cum[:n, None] - cum[:n][None, :])
    return np.minimum(d, total - d)


def chord_dist_matrix(pts):
    return np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)


def classical_mds(D, k=3):
    """Embed a distance matrix into k-D by classical (eigen) MDS."""
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(vals)[::-1][:k]
    return vecs[:, order] * np.sqrt(np.clip(vals[order], 0, None))


def pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=pathlib.Path, default=pathlib.Path("artifacts/manifolds/colorpair_behavior.pt"))
    ap.add_argument("--k-resample", type=int, default=80, help="ring vertices for the distance scatter")
    ap.add_argument("--n-path", type=int, default=60, help="points along each steering path")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("artifacts/manifolds/colorpair_isometry.pt"))
    args = ap.parse_args()

    d = torch.load(args.data, weights_only=False)
    Mh = d["pca3_Mh"].numpy()                 # activation ring (PCA-3D of phi), ordered by color index
    My = d["mds3"].numpy()                     # behavior ring (MDS-3D of D_beh), same order
    n = int(d["n_colors"])

    # --- isometry: same ordered vertices on both rings ---
    Mh_r = ring_resample(Mh, args.k_resample)
    My_r = ring_resample(My, args.k_resample)
    iu = np.triu_indices(args.k_resample, 1)
    D_Y = arc_dist_matrix(My_r)[iu]            # behavior arc length (reference)
    D_X_man = arc_dist_matrix(Mh_r)[iu]        # activation geodesic
    D_X_lin = chord_dist_matrix(Mh_r)[iu]      # activation chord
    # scale activation distances into behavior units (cosmetic; r unchanged)
    D_X_man_s = D_X_man * (D_Y.sum() / D_X_man.sum())
    D_X_lin_s = D_X_lin * (D_Y.sum() / D_X_lin.sum())
    r_man = pearson(D_X_man_s, D_Y); r_lin = pearson(D_X_lin_s, D_Y)
    print(f"N={n}: scaled isometry  manifold r={r_man:.3f}   linear(chord) r={r_lin:.3f}")

    # --- steering paths A -> antipodal B in M_h ambient (3D PCA) coords ---
    A, B = 0, n // 2
    arc_idx = list(range(A, B + 1))                                   # short arc 0..n/2
    man_path = polyline_resample(Mh[arc_idx], args.n_path)           # geodesic along the ring (open arc)
    ts = np.linspace(0, 1, args.n_path)
    lin_path = np.stack([(1 - t) * Mh[A] + t * Mh[B] for t in ts])   # straight chord

    # --- central-column MDS panels: TWO separate MDS embeddings (one per distance metric) ---
    # Each embeds the colors + the matching path jointly, so the path sits in the same MDS frame.
    closed = np.vstack([Mh, Mh[:1]])
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]); L = cum[-1]
    s_cent = cum[:n]                                                  # arc-length coordinate of each color
    # (1) MANIFOLD panel = MDS of the GEODESIC (arc-length) matrix. Geodesic ignores ambient bending,
    #     so equal arc spacing -> a clean uniform circle. The geodesic path lies on that circle.
    #     We embed in 2D (the circle lives in the top-2 components; the 3rd is a spurious harmonic of
    #     the non-Euclidean geodesic metric) and lay it flat on the floor (z = 0).
    def flat(coords2):
        return np.column_stack([coords2, np.zeros(len(coords2))])
    s_path = np.linspace(0, s_cent[B], args.n_path)                  # geodesic A->B arc coords
    s_all = np.concatenate([s_cent, s_path])
    ds = np.abs(s_all[:, None] - s_all[None, :]); Dg = np.minimum(ds, L - ds)
    G = flat(classical_mds(Dg, 2))
    man_ring, man_path_mds = G[:n], G[n:]
    # (2) LINEAR panel = MDS of the CHORD (ambient Euclidean) matrix. Recovers the real ambient shape
    #     (non-uniform spacing); the straight chord path cuts through the interior. Chord distances are
    #     Euclidean, so 2D MDS = the dominant plane of the ambient ring.
    pts = np.vstack([Mh, lin_path])
    Ln = flat(classical_mds(chord_dist_matrix(pts), 2))
    lin_ring, lin_path_mds = Ln[:n], Ln[n:]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "n_colors": n, "A": A, "B": B,
        "Mh": torch.tensor(Mh), "My": torch.tensor(My),
        "man_ring": torch.tensor(man_ring), "man_path_mds": torch.tensor(man_path_mds),
        "lin_ring": torch.tensor(lin_ring), "lin_path_mds": torch.tensor(lin_path_mds),
        "scatter": {"dx_man": torch.tensor(D_X_man_s), "dx_lin": torch.tensor(D_X_lin_s),
                    "dy": torch.tensor(D_Y)},
        "r_manifold": r_man, "r_linear": r_lin,
    }, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
