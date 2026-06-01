"""Capstone steering figure: manifold vs linear induced choice trajectories on the M_y tetrahedron.

For one arm pair we take the choice trajectories induced by steering the activation along the
M_h manifold (even steps in r) vs along the straight linear chord (V2 artifact, mean over base
prompts), and overlay them on the behavior simplex M_y (barycentric tetrahedron). Waypoints are
colored by the steering fraction t so PACING is visible: manifold steps should spread evenly
along the natural M_y edge; linear steps should bunch at the ends and jump through the middle.

Quantified by matched-fraction tracking MSE (induced trajectory vs the natural M_y edge resampled
at the same fractions) — the order-aware metric on which the manifold wins. Off-axis = mean mass
on the two arms that are NOT i,j (does the i→j handoff stay on the edge?).

    .venv/bin/python -m figures.steer_capstone --pair 1_2 --outdir bandit/figures
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

ARM_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
TETRA = np.array([[0, 0, 0], [1, 0, 0],
                  [0.5, np.sqrt(3) / 2, 0], [0.5, np.sqrt(3) / 6, np.sqrt(6) / 3]])


def _bary(P):
    return np.asarray(P) @ TETRA


def _sqH(p, q):
    return 0.5 * ((np.sqrt(p) - np.sqrt(q)) ** 2).sum(-1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steering", type=pathlib.Path,
                   default=pathlib.Path("artifacts/steering/llama8b_fixed_pair_steering_L19.pt"))
    p.add_argument("--pairs", type=pathlib.Path,
                   default=pathlib.Path("artifacts/manifolds/llama8b_fixed_pairs_L19.pt"))
    p.add_argument("--q", type=pathlib.Path, default=pathlib.Path("artifacts/value/bestrl_Q_fixed.pt"))
    p.add_argument("--dataset", type=pathlib.Path, default=pathlib.Path("artifacts/datasets/llama8b_fixed.json"))
    p.add_argument("--pair", default="1_2", help="i_j arm indices (default C-D)")
    p.add_argument("--linear", default="linear_insub", choices=["linear_insub", "linear_full"])
    p.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("bandit/figures"))
    args = p.parse_args()

    st = torch.load(args.steering, weights_only=False)
    pf = torch.load(args.pairs, weights_only=False)
    letters = st["letters"]; n_arms = st["n_arms"]
    i, j = (int(x) for x in args.pair.split("_"))
    pr = st["results"][args.pair]; K = st["K"]
    # tag the output by anything after "L<layer>" in the steering filename (e.g. "_t95-105")
    lkey = f"L{st['layer']}"
    tag = args.steering.stem.split(lkey, 1)[1] if lkey in args.steering.stem else ""

    def arms4(D):
        A = D[:, :4]
        return A / A.sum(1, keepdims=True)
    man = arms4(pr["dist_manifold"].numpy().mean(0))           # [K,4] mean over bases
    lin = arms4(pr[f"dist_{args.linear}"].numpy().mean(0))     # [K,4]
    # off-axis leak = mean mass on the two arms that are NOT i,j (does the handoff stay on the edge?)
    off = [a for a in range(n_arms) if a not in (i, j)]
    other_man = float(man[:, off].sum(1).mean())
    other_lin = float(lin[:, off].sum(1).mean())

    # natural M_y edge for this pair, resampled at the K steering fractions
    g = pf["pairs"][args.pair]["dist_grid"].numpy()            # [n_grid,4]
    frac_idx = np.linspace(0, len(g) - 1, K).round().astype(int)
    myf = g[frac_idx]
    mse_man = _sqH(man, myf).mean() * 1e3
    mse_lin = _sqH(lin, myf).mean() * 1e3

    # tetrahedron context: vertices + all 6 M_y edges
    Q = torch.load(args.q, weights_only=False)["Q"].cpu().numpy()
    logs = json.loads(args.dataset.read_text())
    choice = np.array([l["choice_dists"] for l in logs])[..., :n_arms]
    choice = choice / choice.sum(-1, keepdims=True)
    Qp = Q[:, 1:].reshape(-1, n_arms); Cp = choice[:, 1:].reshape(-1, n_arms)
    order = np.argsort(-Qp, axis=1); argmax = Qp.argmax(1)
    vertices = np.stack([Cp[argmax == x].mean(0) for x in range(n_arms)])

    t = np.linspace(0, 1, K)
    panels = [("Manifold steering", man, mse_man, other_man),
              (f"Linear steering ({args.linear})", lin, mse_lin, other_lin)]
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}] * 2],
                        subplot_titles=[f"{nm} — tracking MSE={m:.2f}e-3, off-axis={o:.2f}"
                                        for nm, _, m, o in panels], horizontal_spacing=0.03)
    for col, (nm, traj, _, _) in enumerate(panels, start=1):
        # faint tetrahedron context (all 6 M_y edges + vertices)
        for a, b in itertools.combinations(range(n_arms), 2):
            pk = f"{a}_{b}"
            if pk not in pf["pairs"]:
                continue
            ge = _bary(pf["pairs"][pk]["dist_grid"].numpy())
            fig.add_trace(go.Scatter3d(x=ge[:, 0], y=ge[:, 1], z=ge[:, 2], mode="lines",
                line=dict(color="lightgray", width=2), showlegend=False, hoverinfo="skip"), 1, col)
        for x in range(n_arms):
            V = _bary(vertices[x][None])[0]
            fig.add_trace(go.Scatter3d(x=[V[0]], y=[V[1]], z=[V[2]], mode="markers+text",
                text=[letters[x]], textposition="top center", showlegend=False,
                marker=dict(size=6, color=ARM_COLORS[x], line=dict(color="black", width=1))), 1, col)
        # focal natural M_y edge (bold black)
        gf = _bary(g)
        fig.add_trace(go.Scatter3d(x=gf[:, 0], y=gf[:, 1], z=gf[:, 2], mode="lines",
            line=dict(color="black", width=4), name="M_y (natural)", showlegend=(col == 1),
            legendgroup="my"), 1, col)
        # induced trajectory: line + waypoint markers colored by t
        P = _bary(traj)
        fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="lines+markers",
            line=dict(color="gray", width=3), showlegend=False,
            marker=dict(size=5, color=t, colorscale="Viridis", showscale=(col == 2),
                        colorbar=dict(title="steering t", x=1.02)),
            customdata=t, hovertemplate="t=%{customdata:.2f}<extra></extra>"), 1, col)

    fig.update_layout(
        title=f"Capstone: steering {letters[i]}→{letters[j]} along M_h, induced choice on the M_y "
              f"tetrahedron (L{st['layer']}). Manifold tracks the natural edge in order; "
              f"linear lags. Tracking MSE {mse_man:.2f} vs {mse_lin:.2f} (×1e-3).",
        height=560, width=1250, margin=dict(l=0, r=0, t=80, b=0), legend=dict(x=0, y=0.5))
    for sc in ("scene", "scene2"):
        fig.update_layout(**{sc: dict(xaxis_title="", yaxis_title="", zaxis_title="",
                                      xaxis_showticklabels=False, yaxis_showticklabels=False,
                                      zaxis_showticklabels=False)})
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / f"steer_capstone_{letters[i]}{letters[j]}_L{st['layer']}{tag}.html"
    fig.write_html(out, include_plotlyjs="cdn")

    # --- natural readout: induced P(action) vs steering fraction t ---
    fig2 = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.05,
                         subplot_titles=(f"Manifold steering (tracking MSE {mse_man:.2f}e-3)",
                                         f"Linear steering (tracking MSE {mse_lin:.2f}e-3)"))
    for col, traj in enumerate([man, lin], start=1):
        for a in range(n_arms):
            wide = a in (i, j)
            fig2.add_trace(go.Scatter(x=t, y=traj[:, a], mode="lines",
                line=dict(color=ARM_COLORS[a], width=3.5 if wide else 1.2),
                name=f"P({letters[a]})", legendgroup=letters[a], showlegend=(col == 1)), 1, col)
        for a in (i, j):  # natural (unintervened) target for the two transitioning arms
            fig2.add_trace(go.Scatter(x=t, y=myf[:, a], mode="lines", opacity=0.55,
                line=dict(color=ARM_COLORS[a], width=2, dash="dot"),
                name=f"P({letters[a]}) natural", legendgroup=f"{letters[a]}nat",
                showlegend=(col == 1)), 1, col)
    fig2.update_xaxes(title_text="steering fraction t")
    fig2.update_yaxes(title_text="P(action)", row=1, col=1)
    fig2.update_layout(
        title=f"Induced P(action) steering {letters[i]}→{letters[j]} (L{st['layer']}): solid=induced, "
              f"dotted=natural M_y. Manifold tracks the natural crossover pace; linear lags then jumps.",
        height=440, width=1050, margin=dict(l=60, r=20, t=80, b=50))
    out2 = args.outdir / f"steer_paction_{letters[i]}{letters[j]}_L{st['layer']}{tag}.html"
    fig2.write_html(out2, include_plotlyjs="cdn")

    print(f"{letters[i]}→{letters[j]}: tracking MSE manifold {mse_man:.2f} vs linear {mse_lin:.2f} (×1e-3); "
          f"off-axis manifold {other_man:.3f} vs linear {other_lin:.3f}")
    print(f"saved → {out}\nsaved → {out2}")


if __name__ == "__main__":
    main()
