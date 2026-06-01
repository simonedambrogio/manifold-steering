"""Train the color-pair comparison net and run the necessity (wrap held-out / seam) tests.

Three evaluations, all on pairs/trials the model was *not* trained on:
  - heldout non-seam pairs : compositional generalization to unseen pairs that do NOT need the wrap.
    High acc => the model learned a reusable distance metric, not a per-pair lookup.
  - seam trials            : the diagnostic -- a circularly-near seam pair vs a linearly-near pair,
    where circular and linear distance disagree. A folding (ring) code scores high; a line scores ~0.
  - k=1 control            : same training, encoder dim 1 (colors forced onto a line). It cannot fold,
    so it must fail the seam trials -- the necessity result.

Saves the Option-B checkpoint and its phi embeddings [N, k] for the geometry/steering analyses.

    .venv/bin/python -m colorpair.train --out artifacts/agent/colorpair_optB.pt
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch

from colorpair import env as E
from colorpair.model import ColorPairNet


def _acc(model, colors, labels, device) -> float:
    c = torch.tensor(colors, device=device)
    y = torch.tensor(labels, device=device)
    logit = model(c)
    return ((logit > 0).float() == y).float().mean().item()


def train_one(k: int, split: E.Split, iters: int, batch: int, lr: float,
              device: str, seed: int, log_every: int = 1000) -> tuple[ColorPairNet, dict]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1)
    n = split.n_colors
    model = ColorPairNet(n, k=k).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = torch.nn.BCEWithLogitsLoss()

    heldout_nonseam = [p for p in split.heldout_pairs if not E.is_seam_pair(*p, n)]
    heldout_seam = [p for p in split.heldout_pairs if E.is_seam_pair(*p, n)]
    seam_pool = heldout_seam or E.seam_pairs(n)        # test generalization to unseen seam pairs

    model.train()
    for it in range(iters):
        colors, labels = E.make_trials(rng, batch, n, split.train_pairs)
        c = torch.tensor(colors, device=device)
        y = torch.tensor(labels, device=device)
        opt.zero_grad()
        loss = bce(model(c), y)
        loss.backward()
        opt.step()
        if log_every and (it + 1) % log_every == 0:
            model.eval()
            with torch.no_grad():
                tr_c, tr_y = E.make_trials(rng, 4000, n, split.train_pairs)
                sc, sl, _ = E.seam_trials(rng, 4000, n, seam_pool=seam_pool)
                msg = (f"  k={k} it {it + 1:5d}  loss {loss.item():.4f}  "
                       f"train-pair acc {_acc(model, tr_c, tr_y, device):.3f}  "
                       f"seam acc {_acc(model, sc, sl, device):.3f}")
            print(msg, flush=True)
            model.train()

    model.eval()
    with torch.no_grad():
        rng_e = np.random.default_rng(12345)
        tr_c, tr_y = E.make_trials(rng_e, 8000, n, split.train_pairs)
        report = {"train_pair_acc": _acc(model, tr_c, tr_y, device)}
        if heldout_nonseam:
            ho_c, ho_y = E.make_trials(rng_e, 8000, n, heldout_nonseam)
            report["heldout_nonseam_acc"] = _acc(model, ho_c, ho_y, device)
        sc, sl, _ = E.seam_trials(rng_e, 8000, n, seam_pool=seam_pool)
        report["seam_acc"] = _acc(model, sc, sl, device)
    return model, report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-colors", type=int, default=E.N_COLORS)
    ap.add_argument("--k", type=int, default=8, help="encoder dim for Option B")
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--heldout-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--no-control", action="store_true", help="skip the k=1 line control")
    ap.add_argument("--holdout-all-seam", action="store_true",
                    help="control regime: hold out every seam pair (wrap never demanded -> model stays linear)")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("artifacts/agent/colorpair_optB.pt"))
    args = ap.parse_args()

    split = E.make_split(args.n_colors, args.heldout_frac, args.seed, args.holdout_all_seam)
    train_seam = sum(E.is_seam_pair(*p, args.n_colors) for p in split.train_pairs)
    regime = "HOLD OUT ALL SEAM (control)" if args.holdout_all_seam else "seam pairs in training"
    print(f"N={args.n_colors} [{regime}]: train pairs {len(split.train_pairs)} ({train_seam} seam), "
          f"heldout {len(split.heldout_pairs)}\n")

    print(f"=== Option B (k={args.k}, learned comparison) ===")
    model, rep = train_one(args.k, split, args.iters, args.batch, args.lr, args.device, args.seed)
    print("  final:", {kk: round(vv, 3) for kk, vv in rep.items()})

    ctrl_rep = None
    if not args.no_control:
        print("\n=== Control: k=1 (colors forced onto a line; cannot fold) ===")
        _, ctrl_rep = train_one(1, split, args.iters, args.batch, args.lr, args.device, args.seed)
        print("  final:", {kk: round(vv, 3) for kk, vv in ctrl_rep.items()})

    print("\n--- necessity summary ---")
    print(f"  Option B (k={args.k}) seam acc: {rep['seam_acc']:.3f}")
    if ctrl_rep is not None:
        print(f"  control  (k=1) seam acc: {ctrl_rep['seam_acc']:.3f}  "
              f"(low = a line cannot represent the ring -> folding is necessary)")

    with torch.no_grad():
        emb = model.embed_all(args.device).cpu().numpy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": {"n_colors": args.n_colors, "k": args.k},
        "split": {"train_pairs": split.train_pairs, "heldout_pairs": split.heldout_pairs},
        "phi": torch.tensor(emb),                       # [N, k] color embeddings
        "report": rep, "control_report": ctrl_rep,
    }, args.out)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
