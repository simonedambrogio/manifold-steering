"""Behavior manifold M_y for the color-pair task, reconstructed from CHOICES ONLY (paper's M_h<->M_y).

The action is one bit (which pair is closer), so there is no ring to read off a single forward pass. We
recover the concept ring *as expressed in behavior* the way a psychophysicist would -- by using the
binary choice as a discrimination probe. For every color pair (i, j) we measure how often the model
judges it the *farther* pair when pitted against other pairs:

    s(i, j) = P_r[ model picks reference pair r as closer than (i, j) ]    (averaged over all pairs r)

s is a purely behavioral perceived-distance for the pair (i, j): it never inspects the model's internal
distance head g, only the model's choices. We assemble the symmetric color-by-color matrix D_beh[i,j] =
s(i,j) and run classical MDS. If the model *acts* on a faithful ring, D_beh is circulant and MDS returns
a closed ring = M_y. (Because the policy is deterministic, s is the rank-transform of the model's implied
metric, so this airtight-behavioral construction necessarily agrees with the g-based proxy -- by design.)

Saves D_beh, the MDS embedding (M_y), and the phi PCA embedding (M_h) for the figure + isometry steps.

    .venv/bin/python -m analyses.colorpair_behavior_manifold --checkpoint artifacts/agent/colorpair_optB.pt \
        --out artifacts/manifolds/colorpair_behavior.pt
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch

from colorpair.model import ColorPairNet


def classical_mds(D: np.ndarray, k: int = 3) -> np.ndarray:
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(vals)[::-1][:k]
    return vecs[:, order] * np.sqrt(np.clip(vals[order], 0, None))


def pca_3d(phi: np.ndarray) -> np.ndarray:
    mu = phi.mean(0)
    _, _, Vt = np.linalg.svd(phi - mu, full_matrices=False)
    return (phi - mu) @ Vt[:3].T


@torch.no_grad()
def behavioral_distance_matrix(model: ColorPairNet, n: int) -> np.ndarray:
    """D_beh[i,j] from choices only: how often pair (i,j) is judged the farther pair vs all other pairs."""
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    P = np.asarray(pairs)
    npair = len(P)
    ti = np.repeat(np.arange(npair), npair)
    ri = np.tile(np.arange(npair), npair)
    colors = np.column_stack([P[ti, 0], P[ti, 1], P[ri, 0], P[ri, 1]])     # [npair^2, 4]
    logit = model(torch.tensor(colors)).numpy()                            # >0 => target (pair 0) closer
    L = logit.reshape(npair, npair)
    farther = (L < 0).astype(float)                                        # target judged farther than ref
    np.fill_diagonal(farther, np.nan)                                      # skip r == target
    s = np.nanmean(farther, axis=1)                                        # [npair] behavioral farness
    D = np.zeros((n, n))
    for idx, (i, j) in enumerate(pairs):
        D[i, j] = D[j, i] = s[idx]
    return D


def ring_stats(coords2: np.ndarray, n: int) -> dict:
    """coords2 ordered by color index. Wraparound ratio + corr of MDS distance with circular distance."""
    Dc = np.linalg.norm(coords2[:, None, :] - coords2[None, :, :], axis=2)
    adj = np.array([Dc[i, i + 1] for i in range(n - 1)])
    wrap = Dc[0, n - 1]
    di, dj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    lin = np.abs(di - dj).astype(float)
    circ = np.minimum(lin, n - lin)
    iu = np.triu_indices(n, 1)
    return {"wraparound_ratio": float(wrap / adj.mean()),
            "corr_circular": float(np.corrcoef(Dc[iu], circ[iu])[0, 1])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=pathlib.Path, default=pathlib.Path("artifacts/agent/colorpair_optB.pt"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("artifacts/manifolds/colorpair_behavior.pt"))
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, weights_only=False)
    n = int(ckpt["config"]["n_colors"]); k = int(ckpt["config"]["k"])
    model = ColorPairNet(n, k=k); model.load_state_dict(ckpt["state_dict"]); model.eval()

    D_beh = behavioral_distance_matrix(model, n)
    mds3 = classical_mds(D_beh, 3)                                          # M_y (behavior ring)
    phi = ckpt["phi"].numpy().astype(np.float64)
    pca3 = pca_3d(phi)                                                      # M_h (representation ring)

    st = ring_stats(mds3[:, :2], n)
    print(f"N={n}: behavioral ring (M_y) from choices only")
    print(f"  wraparound ratio R = {st['wraparound_ratio']:.3f}  (~1 closed ring)")
    print(f"  corr(M_y dist, circular) = {st['corr_circular']:+.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"D_beh": torch.tensor(D_beh), "mds3": torch.tensor(mds3),
                "pca3_Mh": torch.tensor(pca3), "n_colors": n, "ring_stats": st}, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
