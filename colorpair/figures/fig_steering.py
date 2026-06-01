"""Figure 4 (the headline): causal steering on the color ring — geodesic vs linear.

Four panels:
  (1) Activation ring (PCA-2D) with the geodesic (manifold) and chord (linear) steering paths -> the cause.
  (2) Behaviour circle: across-start aggregate of the decoded color (angle) weighted by consistency
      (radius). The manifold steer rides the rim A->B; the linear steer collapses to the centre at the fold.
  (3) Perceived-color strips (example: steering color 0 -> 12): each cell's hue = the color the model
      behaves as if it holds. Manifold = a smooth rainbow sweep; linear = teleports to wrong colors.
  (4) Across-start coherence vs progress: manifold flat ~1; linear collapses to 0 at the midpoint.

    PYTHONPATH=. .venv/bin/python colorpair/figures/fig_steering.py \
        --data artifacts/steering/colorpair_steer.pt --out colorpair/figures/colorpair_steering
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch
import matplotlib
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CYC = matplotlib.colormaps["hsv"]
MAN_C, LIN_C = "#2c5fa8", "#c0392b"


def pca2(phi):
    mu = phi.mean(0); _, _, Vt = np.linalg.svd(phi - mu, full_matrices=False)
    return (phi - mu) @ Vt[:2].T


def color_strip(angs, n):
    """RGB row [K,3] from decoded angles -> hue (full saturation)."""
    hue = (np.asarray(angs) % (2 * np.pi)) / (2 * np.pi)
    hsv = np.stack([hue, np.ones_like(hue), np.ones_like(hue)], 1)
    return (mcolors.hsv_to_rgb(hsv) * 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=pathlib.Path, default=pathlib.Path("artifacts/steering/colorpair_steer.pt"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("colorpair/figures/colorpair_steering"))
    args = ap.parse_args()

    d = torch.load(args.data, weights_only=False)
    n, K, off = int(d["n_colors"]), int(d["K"]), int(d["off"])
    phi = d["phi"].numpy(); ring = pca2(phi)
    man_circ, lin_circ = d["man_circ"], d["lin_circ"]
    man_coh, lin_coh = d["man_coh"], d["lin_coh"]
    prog = np.linspace(0, 1, K)

    fig = make_subplots(
        rows=2, cols=2, vertical_spacing=0.13, horizontal_spacing=0.13,
        subplot_titles=("Activation ring — steering paths", "Decoded recall (behaviour circle)",
                        "Perceived color along the steer (0 → 12)", "Across-start coherence"),
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]])

    # (1) activation ring + paths
    cols = [mcolors.to_hex(CYC(i / n)) for i in range(n)]
    closed = np.vstack([ring, ring[:1]])
    arc = list(range(0, off + 1))
    geo = ring[arc]                                              # geodesic hugs the ring
    chord = np.stack([(1 - t) * ring[0] + t * ring[off] for t in np.linspace(0, 1, 30)])
    fig.add_trace(go.Scatter(x=closed[:, 0], y=closed[:, 1], mode="lines",
                             line=dict(color="lightgray", width=1.5), showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=ring[:, 0], y=ring[:, 1], mode="markers",
                             marker=dict(size=11, color=cols, symbol="diamond",
                                         line=dict(width=0.5, color="black")), showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=geo[:, 0], y=geo[:, 1], mode="lines", line=dict(color=MAN_C, width=4),
                             name="manifold"), 1, 1)
    fig.add_trace(go.Scatter(x=chord[:, 0], y=chord[:, 1], mode="lines", line=dict(color=LIN_C, width=4),
                             name="linear"), 1, 1)

    # (2) behaviour circle
    th = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(go.Scatter(x=np.cos(th), y=np.sin(th), mode="lines",
                             line=dict(color="lightgray", width=1.5), showlegend=False), 1, 2)
    for circ, c in [(man_circ, MAN_C), (lin_circ, LIN_C)]:
        fig.add_trace(go.Scatter(x=circ[:, 0], y=circ[:, 1], mode="lines+markers",
                                 line=dict(color=c, width=2), marker=dict(size=5), showlegend=False), 1, 2)
    fig.add_trace(go.Scatter(x=[1, -1], y=[0, 0], mode="markers+text", text=["A", "B"],
                             textposition="top center", marker=dict(size=9, color="black"),
                             showlegend=False), 1, 2)

    # (3) perceived-color strips (example A=0)
    img = np.stack([color_strip(d["man_ang0"], n), color_strip(d["lin_ang0"], n)])  # [2,K,3]
    fig.add_trace(go.Image(z=img), 2, 1)

    # (4) coherence
    fig.add_trace(go.Scatter(x=prog, y=man_coh, mode="lines+markers", line=dict(color=MAN_C, width=3),
                             name="manifold", showlegend=False), 2, 2)
    fig.add_trace(go.Scatter(x=prog, y=lin_coh, mode="lines+markers", line=dict(color=LIN_C, width=3),
                             name="linear", showlegend=False), 2, 2)

    # axes styling
    fig.update_xaxes(visible=False, row=1, col=1); fig.update_yaxes(visible=False, scaleanchor="x", row=1, col=1)
    fig.update_xaxes(visible=False, row=1, col=2); fig.update_yaxes(visible=False, scaleanchor="x2", row=1, col=2)
    fig.update_xaxes(tickvals=[0, (K - 1) / 2, K - 1], ticktext=["A", "0.5", "B"], row=2, col=1)
    fig.update_yaxes(tickvals=[0, 1], ticktext=["manifold", "linear"], row=2, col=1, autorange="reversed")
    fig.update_xaxes(title_text="path progress  A → B", row=2, col=2)
    fig.update_yaxes(title_text="coherence", rangemode="tozero", row=2, col=2)
    fig.update_layout(width=1000, height=900, paper_bgcolor="white", plot_bgcolor="white",
                      margin=dict(l=40, r=20, t=60, b=40),
                      title=dict(text="Steering a remembered color: manifold vs linear", x=0.5, xanchor="center"),
                      legend=dict(x=0.0, y=0.62, font=dict(size=10)))
    for r, c in [(2, 2)]:
        fig.update_xaxes(showgrid=False, linecolor="black", row=r, col=c)
        fig.update_yaxes(showgrid=False, linecolor="black", row=r, col=c)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    html = args.out.with_suffix(".html"); fig.write_html(str(html),
        config={"responsive": False, "toImageButtonOptions": {"format": "png", "scale": 6}})
    print(f"saved -> {html}")
    try:
        fig.write_image(str(args.out.with_suffix(".pdf")), scale=3)
        print(f"saved -> {args.out.with_suffix('.pdf')}")
    except Exception as e:
        print(f"(static export skipped: {e})")


if __name__ == "__main__":
    main()
