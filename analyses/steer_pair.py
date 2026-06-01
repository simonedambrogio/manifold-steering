"""V2 between-arms steering — manifold vs linear, c_i → c_j (fixed letters).

For each arm pair, steer the decision-token residual along the M_h spline (manifold) and
along the straight chord between endpoint centroids (linear), injecting K waypoints into a
fixed set of base prompts and reading the induced choice distribution. The headline test:
does manifold steering move probability mass i → (uncertain) → j *smoothly and naturally*
(hugging M_y), while linear steering teleports / leaks mass off the i↔j axis?

Modes:
  manifold      M_h spline path; replace PCA-64 projection, keep base off-subspace.
  linear_insub  straight chord in PCA-64; same in-subspace swap (isolates PATH SHAPE only).
  linear_full   straight chord; replace the ENTIRE residual (paper-exact linear baseline).

Naturalness = cumulative Bhattacharyya distance from each waypoint's induced distribution
to the nearest point on M_y (the paper's E_BC energy; lower = more natural).

Output (`artifacts/steering/<name>.pt`): per pair → per mode dist_per_base [n_bases, K, 5]
(4 arms + other), E_BC [n_bases], manifold/chord paths, r-along-path; plus base list and a
paired manifold-vs-linear summary.

    python -m analyses.steer_pair \
        --pairs artifacts/manifolds/llama8b_fixed_pairs_L19.pt \
        --activations artifacts/activations/llama8b_fixed_residual.pt \
        --dataset artifacts/datasets/llama8b_fixed.json \
        --n-waypoints 25 --n-bases 16 \
        --out artifacts/steering/llama8b_fixed_pair_steering.pt
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch

from llm_data.llm import letter_token_ids, load_model
from llm_data.prompt import LabelMap, build_chat_prompt, choice_distribution
from analyses.interchange import _install_steer_hook

MODES = ("manifold", "linear_insub", "linear_full")


def bc_to_my(p_arms, dist_grid):
    """Min Bhattacharyya distance from a 4-arm dist to points on M_y (dist_grid [G, 4])."""
    overlap = (np.sqrt(p_arms)[None, :] * np.sqrt(dist_grid)).sum(axis=1)
    return float(-np.log(np.clip(overlap, 1e-12, 1.0)).min())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pairs", type=pathlib.Path, required=True)
    parser.add_argument("--activations", type=pathlib.Path, required=True)
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--n-waypoints", type=int, default=25)
    parser.add_argument("--n-bases", type=int, default=16)
    parser.add_argument("--base-t-lo", type=int, default=30)
    parser.add_argument("--base-t-hi", type=int, default=120)
    parser.add_argument("--modes", nargs="+", default=list(MODES))
    parser.add_argument("--chunk", type=int, default=10,
                        help="waypoints per batched forward (memory knob)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    mf = torch.load(args.pairs, weights_only=False)
    pair_data = mf["pairs"]
    pca_basis = mf["pca_basis"].numpy().astype(np.float32)
    pca_mean = mf["pca_mean"].numpy().astype(np.float32)
    L = int(mf["layer"])
    hook_layer = L - 1
    letters = mf["letters"]
    n_arms = int(mf["n_arms"])
    K = args.n_waypoints

    act_data = torch.load(args.activations, weights_only=False)
    activations = act_data["activations"]
    ep_ids = [int(e) for e in act_data["episode_ids"]]
    logs = json.loads(args.dataset.read_text())

    # ---- fixed base prompts (shared across pairs) ----
    rng = np.random.default_rng(args.seed)
    base_eps = rng.choice(ep_ids, size=args.n_bases, replace=False)
    base_ts = rng.integers(args.base_t_lo, args.base_t_hi, size=args.n_bases)
    lm = LabelMap(letters=letters)

    model, tok = load_model(args.model)
    lab_ids = letter_token_ids(tok, letters)
    print(f"loaded {args.model} on {model.device}; layer {L}, letters {letters}", flush=True)

    bases = []
    for ep, t in zip(base_eps, base_ts):
        local = ep_ids.index(int(ep))
        history = list(zip(logs[int(ep)]["actions"][:int(t)], logs[int(ep)]["rewards"][:int(t)]))
        enc = tok(build_chat_prompt(tok, lm, history), return_tensors="pt").to(model.device)
        dp = enc["input_ids"].shape[1] - 1
        base_act = activations[local, int(t), L, :].float().numpy()
        base_off = base_act - (pca_basis.T @ (pca_basis @ (base_act - pca_mean)) + pca_mean)
        bases.append({"enc": enc, "dp": dp, "off": base_off, "ep": int(ep), "t": int(t)})

    results = {}
    pair_keys = list(pair_data.keys())
    n_fwd = len(pair_keys) * len(bases) * len(args.modes) * int(np.ceil(K / args.chunk))
    print(f"\n{len(pair_keys)} pairs × {len(bases)} bases × {len(args.modes)} modes, "
          f"K={K} batched in chunks of {args.chunk} → {n_fwd} batched forwards", flush=True)
    t0, n_done = time.time(), 0

    for pk in pair_keys:
        pd = pair_data[pk]
        i, j = pd["i"], pd["j"]
        manifold_grid = pd["manifold_grid"].numpy()                 # [G, pca_dim]
        r_grid = pd["r_grid"].numpy()
        cen = pd["centroids_pca"].numpy()                           # [n_used, pca_dim]
        dist_grid = pd["dist_grid"].numpy()                         # [G, n_arms] (M_y)

        # waypoints: manifold = sample spline grid at K even r; linear = chord of endpoint centroids
        r_way = np.linspace(r_grid[0], r_grid[-1], K)
        man_path = np.stack([manifold_grid[np.argmin(np.abs(r_grid - r))] for r in r_way])
        lin_path = np.stack([(1 - a) * cen[0] + a * cen[-1] for a in np.linspace(0, 1, K)])
        # reconstruct all K waypoints to 4096-D once: [K, pca_dim] @ [pca_dim, 4096]
        man_4096 = (man_path @ pca_basis + pca_mean).astype(np.float32)        # [K, 4096]
        lin_4096 = (lin_path @ pca_basis + pca_mean).astype(np.float32)

        dist_per = {m: np.zeros((len(bases), K, n_arms + 1)) for m in args.modes}
        for bi, b in enumerate(bases):
            ids, am, dp = b["enc"]["input_ids"], b["enc"]["attention_mask"], b["dp"]
            for m in args.modes:
                base_path = man_4096 if m == "manifold" else lin_4096
                steered_all = base_path + (b["off"][None, :] if m != "linear_full" else 0.0)
                # batch the K waypoints (same prompt, per-row injected vector) in chunks
                for s in range(0, K, args.chunk):
                    c = min(args.chunk, K - s)
                    st = torch.from_numpy(steered_all[s:s + c]).to(model.device)   # [c, 4096]
                    enc_b = {"input_ids": ids.repeat(c, 1), "attention_mask": am.repeat(c, 1)}
                    h = _install_steer_hook(model, hook_layer, dp, st)
                    try:
                        with torch.no_grad():
                            logits = model(**enc_b).logits[:, dp, :].float().cpu().numpy()  # [c, vocab]
                    finally:
                        h.remove()
                    for r in range(c):
                        dist_per[m][bi, s + r] = choice_distribution(logits[r], lab_ids)
                    n_done += 1
            if (bi + 1) % 4 == 0:
                el = time.time() - t0
                print(f"  {pk}: base {bi+1}/{len(bases)}  {el/60:.1f}min  "
                      f"eta {el/n_done*(n_fwd-n_done)/60:.1f}min", flush=True)

        # naturalness E_BC per base per mode (4-arm renormalized vs M_y)
        e_bc = {}
        for m in args.modes:
            d = dist_per[m]
            arms = d[..., :n_arms] / d[..., :n_arms].sum(axis=-1, keepdims=True)
            e_bc[m] = np.array([sum(bc_to_my(arms[bi, k], dist_grid) for k in range(K))
                                for bi in range(len(bases))])

        results[pk] = {
            "i": i, "j": j, "letters": pd["letters"],
            "r_way": torch.tensor(r_way, dtype=torch.float32),
            "manifold_path_pca": torch.tensor(man_path, dtype=torch.float32),
            "linear_path_pca": torch.tensor(lin_path, dtype=torch.float32),
            **{f"dist_{m}": torch.tensor(dist_per[m], dtype=torch.float32) for m in args.modes},
            **{f"ebc_{m}": torch.tensor(e_bc[m], dtype=torch.float32) for m in args.modes},
        }
        msg = "  ".join(f"{m}={e_bc[m].mean():.2f}" for m in args.modes)
        print(f"  {pk} [{pd['letters'][0]}→{pd['letters'][1]}] E_BC: {msg}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "results": results, "modes": args.modes, "layer": L, "letters": letters,
        "n_arms": n_arms, "K": K,
        "bases": [{"ep": b["ep"], "t": b["t"]} for b in bases],
        "model_id": act_data["model_id"], "pairs_artifact": str(args.pairs),
    }, args.out)
    print(f"\nsaved → {args.out}")

    # ---- paired manifold-vs-linear summary over (pair, base) ----
    if "manifold" in args.modes:
        for lin in [m for m in args.modes if m.startswith("linear")]:
            man = np.concatenate([results[pk]["ebc_manifold"].numpy() for pk in pair_keys])
            lim = np.concatenate([results[pk][f"ebc_{lin}"].numpy() for pk in pair_keys])
            d = lim - man
            t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
            print(f"\nE_BC  manifold={man.mean():.2f}  {lin}={lim.mean():.2f}  "
                  f"ratio={lim.mean()/man.mean():.2f}×  paired t={t:+.1f} (n={len(d)})")


if __name__ == "__main__":
    main()
