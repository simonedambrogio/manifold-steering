"""Locate figure: per-layer ridge probe R² for several scalar targets derived from Q.

Two panels (1 × 2):

  Left  — heatmap of test R² per (target, layer).
  Right — line plot, one line per target, of test R² across layers.

    python -m figures.fig_locate \
        --locate artifacts/locate/llama8b_bestrl_scalars.json \
        --out artifacts/figures/fig_locate.png
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--locate", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.locate.read_text())
    layers = data["layers"]
    target_names = data["target_names"]
    n_layers = len(layers)
    n_targets = len(target_names)

    r2_test = np.array(
        [[d["r2_test"][t] for t in target_names] for d in layers]
    )  # [n_layers, n_targets]

    # ---- figure ----
    fig, (ax_h, ax_l) = plt.subplots(
        1, 2, figsize=(12.0, 4.5), gridspec_kw={"width_ratios": [1.3, 1.0]}
    )

    # heatmap: targets (rows) × layers (cols); fixed [-1, 1] so the meaningful range
    # ([0, 1] for an actual fit) dominates the colours. Off-scale values saturate.
    im = ax_h.imshow(
        r2_test.T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-1.0, vmax=1.0,
    )
    ax_h.set_xlabel("Layer (0 = embedding output)")
    ax_h.set_yticks(range(n_targets))
    ax_h.set_yticklabels(target_names)
    ax_h.set_title("Test R²: linear probe activation → scalar target")
    plt.colorbar(im, ax=ax_h, label="R²  (clipped to [-1, 1])", fraction=0.04)

    # mark the per-target best layer
    for ti in range(n_targets):
        best_layer = int(np.argmax(r2_test[:, ti]))
        ax_h.scatter([best_layer], [ti], marker="o", facecolor="none",
                     edgecolor="black", s=80, lw=1.2)

    # line plot — clip y to the meaningful range; report any off-screen values in legend
    colors = plt.cm.tab10(np.arange(n_targets))
    for ti, name in enumerate(target_names):
        ax_l.plot(range(n_layers), r2_test[:, ti], color=colors[ti],
                  lw=1.8, marker="o", markersize=3, label=name)
        best_layer = int(np.argmax(r2_test[:, ti]))
        ax_l.scatter([best_layer], [r2_test[best_layer, ti]],
                     marker="o", facecolor="none", edgecolor=colors[ti], s=80, lw=1.4)

    ax_l.axhline(0.0, color="gray", lw=0.6, ls=":")
    ax_l.set_xlabel("Layer (0 = embedding output)")
    ax_l.set_ylabel("Test R² (clipped)")
    ax_l.set_title("Per-target R² across layers")
    ax_l.legend(loc="lower right", fontsize=8)
    ax_l.set_xlim(-0.5, n_layers - 0.5)
    ax_l.set_ylim(-1.0, 1.0)

    fig.suptitle(
        f"{data['model_id']}  →  {data['value_model']} Q-derived scalars  "
        f"(train {data['n_eps_train']}/val {data['n_eps_val']}/test {data['n_eps_test']} eps × "
        f"{data['n_predict_trials']} trials)",
        fontsize=10,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"saved → {args.out}")
    for name in target_names:
        scores = r2_test[:, target_names.index(name)]
        best_layer = int(np.argmax(scores))
        print(f"  {name:>22}: best layer = {best_layer:2d}  R² = {scores[best_layer]:+.3f}")


if __name__ == "__main__":
    main()
