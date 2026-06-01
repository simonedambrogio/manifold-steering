"""Causal steering on the color ring: geodesic (manifold) vs linear (chord), read out from CHOICES only.

We steer one slot's color embedding phi(c) from A to its antipode B, along two paths:
  - manifold: the geodesic, resampled through the real color embeddings (stays ON the ring),
  - linear:   the straight chord phi(A)->phi(B) (cuts through the ring interior; midpoint = ring centre).
At each waypoint w we inject w as the slot's embedding and ask, using ONLY the model's binary choices,
"what color does the model behave as if it holds?":

  similarity(w, r) = P_{r'}[ model judges pair (w, r) closer than pair (w, r') ]      (win rate over r')

This is a purely behavioral profile over reference colors r (never touches the distance head g). From it:
  decoded angle = circular resultant of the (centred) profile;  coherence = its resultant length
A valid on-ring waypoint -> sharply peaked profile -> clean decoded color, coherence ~1. The chord's
midpoint sits at the ring centre, equidistant from every color -> flat profile -> coherence -> 0 (collapse).

    .venv/bin/python -m analyses.steer_colorpair --checkpoint artifacts/agent/colorpair_optB.pt \
        --out artifacts/steering/colorpair_steer.pt
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch

from colorpair.model import ColorPairNet


def geodesic_waypoints(phi, A, off, K):
    """K embeddings along the short ring arc A -> A+off, resampled by arc length (open polyline)."""
    n = len(phi)
    idx = [(A + i) % n for i in range(off + 1)]
    poly = phi[idx]
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, cum[-1], K)
    return np.stack([np.interp(t, cum, poly[:, j]) for j in range(poly.shape[1])], axis=1)


def linear_waypoints(phi, A, off, K):
    n = len(phi)
    a, b = phi[A], phi[(A + off) % n]
    return np.stack([(1 - t) * a + t * b for t in np.linspace(0, 1, K)], axis=0)


@torch.no_grad()
def behavioral_profile(model, phi_all, w, n):
    """similarity(w, r) over reference colors r, from choices only (win rate of (w,r) vs (w,r'))."""
    rr = torch.arange(n)
    R1 = rr.repeat_interleave(n); R2 = rr.repeat(n)            # all (r, r') ordered pairs
    emb = torch.empty(n * n, 4, model.k)
    emb[:, 0] = w; emb[:, 1] = phi_all[R1]; emb[:, 2] = w; emb[:, 3] = phi_all[R2]
    logit = model.logits_from_embeddings(emb).view(n, n)      # >0 => (w,r) closer than (w,r')
    win = (logit > 0).float()
    win.fill_diagonal_(np.nan)                                # skip r' == r
    return torch.nan_to_num(win, nan=0.0).sum(1).numpy() / (n - 1)   # mean win rate per r


def decode(sim, n):
    """Decoded angle + coherence from the centred circular resultant of the similarity profile."""
    theta = 2 * np.pi * np.arange(n) / n
    w = np.clip(sim - sim.min(), 0, None)
    tot = w.sum()
    if tot < 1e-9:
        return 0.0, 0.0
    R = (w * np.exp(1j * theta)).sum() / tot
    return float(np.angle(R)), float(np.abs(R))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=pathlib.Path, default=pathlib.Path("artifacts/agent/colorpair_optB.pt"))
    ap.add_argument("--k-way", type=int, default=13, help="waypoints A->B")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("artifacts/steering/colorpair_steer.pt"))
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, weights_only=False)
    n = int(ckpt["config"]["n_colors"]); k = int(ckpt["config"]["k"])
    model = ColorPairNet(n, k=k); model.load_state_dict(ckpt["state_dict"]); model.eval()
    with torch.no_grad():
        phi = model.embed_all().numpy().astype(np.float64)
    phi_t = torch.tensor(phi, dtype=torch.float32)
    off = n // 2; K = args.k_way

    # per (start A, mode, waypoint): decoded angle + coherence
    man_ang = np.zeros((n, K)); man_coh = np.zeros((n, K))
    lin_ang = np.zeros((n, K)); lin_coh = np.zeros((n, K))
    for A in range(n):
        for wp, ang, coh in [(geodesic_waypoints(phi, A, off, K), man_ang, man_coh),
                             (linear_waypoints(phi, A, off, K), lin_ang, lin_coh)]:
            for ki in range(K):
                sim = behavioral_profile(model, phi_t, torch.tensor(wp[ki], dtype=torch.float32), n)
                ang[A, ki], coh[A, ki] = decode(sim, n)

    theta = 2 * np.pi * np.arange(n) / n

    def aggregate(ang, coh):
        """A-relative behaviour circle: coherence-weighted mean of decoded direction over starts A.
        Its magnitude = how *consistently* across starts the steer lands at the intended progress
        (the real coherence signal): high on-manifold, ~0 where the off-manifold steer teleports
        to inconsistent colors."""
        rel = (ang - theta[:, None] + np.pi) % (2 * np.pi) - np.pi          # decoded angle minus start
        agg = (coh * np.exp(1j * rel)).mean(0)                              # average over starts A
        return np.stack([agg.real, agg.imag], 1)                            # circle [K,2]

    man_circ = aggregate(man_ang, man_coh); lin_circ = aggregate(lin_ang, lin_coh)
    norm = np.linalg.norm(man_circ, axis=1).mean()                          # manifold baseline -> 1
    man_circ /= norm; lin_circ /= norm
    man_coh_m = np.linalg.norm(man_circ, axis=1); lin_coh_m = np.linalg.norm(lin_circ, axis=1)
    print(f"N={n}, K={K}, antipodal off={off}  (coherence normalized: manifold baseline = 1)")
    print(f"  coherence midpoint: manifold {man_coh_m[K // 2]:.2f}  vs  linear {lin_coh_m[K // 2]:.2f}")
    print(f"  coherence min:      manifold {man_coh_m.min():.2f}  vs  linear {lin_coh_m.min():.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "n_colors": n, "K": K, "off": off,
        "man_ang0": man_ang[0], "man_coh0": man_coh[0],     # A=0 sweep (for the color strips)
        "lin_ang0": lin_ang[0], "lin_coh0": lin_coh[0],
        "man_circ": man_circ, "lin_circ": lin_circ,         # aggregated behaviour circle
        "man_coh": man_coh_m, "lin_coh": lin_coh_m,         # aggregated coherence vs progress
        "phi": torch.tensor(phi),
    }, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
