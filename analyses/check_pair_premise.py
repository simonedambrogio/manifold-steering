"""Phase-2 premise gate for V2 steering (model-free, fast).

(1) M_h curvature — the manifold-vs-linear contrast only exists if M_h is curved. Per pair,
    in the PCA-64 subspace where steering happens, report:
      arc/chord  = arc length of the M_h spline ÷ straight chord between its endpoints
      max_dev    = max orthogonal distance of the spline from that chord, ÷ chord length
    arc/chord ≈ 1 and max_dev ≈ 0 ⇒ flat ⇒ manifold ≈ linear (pivot, report honestly).
(2) Letter bias — fixed letters risk a per-letter preference confound. Report the baseline
    mean P(letter) over all trials and the first-trial (history-free) P(letter).

    python -m analyses.check_pair_premise \
        --pairs artifacts/manifolds/llama8b_fixed_pairs_L19.pt \
        --dataset artifacts/datasets/llama8b_fixed.json
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch


def curve_stats(grid):
    """grid [G, D] in PCA space → (arc/chord ratio, max orthogonal deviation / chord)."""
    seg = np.linalg.norm(np.diff(grid, axis=0), axis=1)
    arc = seg.sum()
    a, b = grid[0], grid[-1]
    chord_vec = b - a
    chord = np.linalg.norm(chord_vec) + 1e-12
    u = chord_vec / chord
    rel = grid - a
    proj = rel @ u
    perp = rel - proj[:, None] * u[None, :]
    max_dev = np.linalg.norm(perp, axis=1).max() / chord
    return arc / chord, max_dev


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=pathlib.Path, required=True)
    ap.add_argument("--dataset", type=pathlib.Path, required=True)
    args = ap.parse_args()

    mf = torch.load(args.pairs, weights_only=False)
    letters = mf["letters"]
    n_arms = mf["n_arms"]

    print("=== (1) M_h curvature (PCA-64, where steering happens) ===")
    print(f"{'pair':>8}  {'arc/chord':>10}  {'max_dev':>8}   verdict")
    ratios = []
    for pk, pd in mf["pairs"].items():
        ratio, dev = curve_stats(pd["manifold_grid"].numpy())
        ratios.append(ratio)
        verdict = "curved" if (ratio > 1.05 or dev > 0.05) else "~flat"
        ll = "-".join(pd["letters"])
        print(f"{pk:>8}  {ratio:>10.3f}  {dev:>8.3f}   {verdict}  ({ll})")
    print(f"mean arc/chord = {np.mean(ratios):.3f}  "
          f"→ {'CURVED, proceed' if np.mean(ratios) > 1.05 else 'FLAT, consider pivot'}")

    print("\n=== (2) Letter bias ===")
    logs = json.loads(args.dataset.read_text())
    cd = np.array([l["choice_dists"] for l in logs], dtype=np.float32)[..., :n_arms]
    cd = cd / cd.sum(axis=-1, keepdims=True)
    overall = cd.reshape(-1, n_arms).mean(axis=0)
    first = cd[:, 0, :].mean(axis=0)
    print(f"{'letter':>8}  {'overall P':>10}  {'first-trial P':>14}")
    for a in range(n_arms):
        print(f"{letters[a]:>8}  {overall[a]:>10.3f}  {first[a]:>14.3f}")
    spread = overall.max() - overall.min()
    print(f"overall spread = {spread:.3f}  "
          f"→ {'mild' if spread < 0.10 else 'NOTABLE — flag in interpretation'}")


if __name__ == "__main__":
    main()
