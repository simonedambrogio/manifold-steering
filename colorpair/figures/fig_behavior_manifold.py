"""Figure 2 (color-pair task): the behavior manifold M_y, reconstructed from CHOICES ONLY, via MDS.

The paper's structural-correspondence figure pairs an activation ring (M_h) with a behavior ring (M_y).
Here M_y is not a direct model output (the action is one bit) -- it is the concept ring as expressed in
behavior, recovered by `analyses.colorpair_behavior_manifold` from the model's binary choices and embedded
with classical MDS. A closed rainbow loop here means the model's *behavior* (not just its representation)
honours the ring, with the seam color sitting next to color 0.

Reuses the M_h figure's aesthetic (grey floor, cyclic diamonds, shadow). Run under the interpreter that
has plotly (+kaleido), from the repo root:

    .venv/bin/python colorpair/figures/fig_behavior_manifold.py \
        --data artifacts/manifolds/colorpair_behavior.pt --out colorpair/figures/colorpair_behavior_manifold
"""

from __future__ import annotations

import argparse
import pathlib

import torch

from colorpair.figures.fig_pca3d import build_figure


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=pathlib.Path, default=pathlib.Path("artifacts/manifolds/colorpair_behavior.pt"))
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("colorpair/figures/colorpair_behavior_manifold"))
    ap.add_argument("--title", type=str, default="Behavior manifold")
    args = ap.parse_args()

    d = torch.load(args.data, weights_only=False)
    coords = d["mds3"].numpy()                     # [N, 3] behavioral ring, ordered by color index
    n = int(d["n_colors"])
    st = d["ring_stats"]
    print(f"N={n}: M_y wraparound R={st['wraparound_ratio']:.3f}, corr_circular={st['corr_circular']:+.3f}")

    fig = build_figure(coords, n, args.title)   # angled "on the floor" view, matching the activation-manifold figure
    args.out.parent.mkdir(parents=True, exist_ok=True)
    html = args.out.with_suffix(".html"); fig.write_html(str(html),
        config={"responsive": False, "toImageButtonOptions": {"format": "png", "scale": 6}})
    print(f"saved -> {html}")
    try:
        pdf = args.out.with_suffix(".pdf"); fig.write_image(str(pdf), scale=3)
        print(f"saved -> {pdf}")
    except Exception as e:
        print(f"(static export skipped: {e})")


if __name__ == "__main__":
    main()
