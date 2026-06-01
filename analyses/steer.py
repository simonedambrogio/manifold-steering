"""Step 7 — manifold vs linear steering of Llama's layer-19 residual.

For each of K waypoints along the M_h spline (manifold) and along the straight line in
64-D PCA space between two endpoint centroids (linear), patch the residual stream output
of model.model.layers[18] at the decision-token position with the steered activation, then
read the resulting choice distribution. Average across base prompts after sorting each
prompt's distribution by its own Q ranking at that trial.

  Manifold steering: keep base's *off-subspace* residual; replace the top-64 PCA
                     components with the waypoint. Surgical.
  Linear steering:   replace the entire residual with the PCA-reconstructed linear-interp
                     waypoint. No off-subspace from base.

Output (`artifacts/steering/<name>.pt`):
  waypoint_q_values         [K]            — Q_max values along the path
  manifold_path_pca         [K, 64]        — waypoints in PCA space
  linear_path_pca           [K, 64]        — waypoints (linear interp in PCA space)
  manifold_dist_per_base    [P, K, n_arms] — choice dist per (base, waypoint), Q-ranked
  linear_dist_per_base      [P, K, n_arms]
  manifold_dist_avg         [K, n_arms]    — mean across base prompts
  linear_dist_avg           [K, n_arms]
  base_prompts              list[(ep, t)]
  K, layer, q_low, q_high
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


def _install_steer_hook(model, layer0_idx: int, position: int, steered_4096: torch.Tensor):
    """Forward hook on `model.model.layers[layer0_idx]` that replaces the residual at
    `position` (token index, last token) with `steered_4096`. Returns the handle."""
    target = model.model.layers[layer0_idx]

    def hook(_module, _input, output):
        if isinstance(output, tuple):
            hs = output[0].clone()
            hs[:, position, :] = steered_4096.to(hs.dtype).to(hs.device)
            return (hs,) + output[1:]
        hs = output.clone()
        hs[:, position, :] = steered_4096.to(hs.dtype).to(hs.device)
        return hs

    return target.register_forward_hook(hook)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifold-h", type=pathlib.Path, required=True)
    parser.add_argument("--activations", type=pathlib.Path, required=True,
                        help="for the base-prompt activations (the off-subspace residual)")
    parser.add_argument("--q", type=pathlib.Path, required=True)
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--n-waypoints", type=int, default=20)
    parser.add_argument("--low-bin", type=int, default=2,
                        help="index into bin_centers; endpoint for low Q_max")
    parser.add_argument("--high-bin", type=int, default=27,
                        help="index into bin_centers; endpoint for high Q_max")
    parser.add_argument("--base-eps", type=int, nargs="+",
                        default=list(range(80, 90)),
                        help="episode indices to use as base prompts")
    parser.add_argument("--base-trial", type=int, default=75,
                        help="trial index within each base episode")
    args = parser.parse_args()

    # ---- load manifold + base data ----
    mh = torch.load(args.manifold_h, weights_only=False)
    centroids_pca = mh["centroids_pca"].numpy()           # [n_bins, 64]
    manifold_grid = mh["manifold_grid"].numpy()            # [n_grid, 64]
    manifold_q_grid = mh["manifold_q_grid"].numpy()
    pca_basis = mh["pca_basis"].numpy()                    # [64, 4096]
    pca_mean = mh["pca_mean"].numpy()                      # [4096]
    bin_centers = mh["bin_centers"].numpy()
    activation_layer_idx = int(mh["layer"])                # in hidden_states indexing
    hook_layer_idx = activation_layer_idx - 1              # 0-indexed transformer layer

    act_data = torch.load(args.activations, weights_only=False)
    activations = act_data["activations"]                   # [n_eps, T, n_layers, 4096] fp16
    ep_ids = act_data["episode_ids"]

    q_data = torch.load(args.q, weights_only=False)
    Q_all = q_data["Q"]                                     # [n_total_eps, T, n_arms]

    logs = json.loads(args.dataset.read_text())
    T = len(logs[0]["actions"])

    # ---- endpoints + waypoints ----
    q_low = float(bin_centers[args.low_bin])
    q_high = float(bin_centers[args.high_bin])
    print(f"steering endpoints: Q_max from {q_low:+.1f} (bin {args.low_bin}) "
          f"to {q_high:+.1f} (bin {args.high_bin})")
    print(f"  patching at hidden_states[{activation_layer_idx}] "
          f"(= output of model.model.layers[{hook_layer_idx}])")

    K = args.n_waypoints
    q_waypoints = np.linspace(q_low, q_high, K)

    # Manifold path: nearest grid point to each q_waypoint
    manifold_path_pca = np.stack([
        manifold_grid[np.argmin(np.abs(manifold_q_grid - q))] for q in q_waypoints
    ])                                                       # [K, 64]
    # Linear path: linear interp in PCA space between endpoint *centroids*
    linear_path_pca = np.stack([
        (1 - t) * centroids_pca[args.low_bin] + t * centroids_pca[args.high_bin]
        for t in np.linspace(0, 1, K)
    ])                                                       # [K, 64]

    # ---- model ----
    model, tok = load_model(args.model)
    print(f"loaded {args.model} on {model.device}", flush=True)
    n_arms = 4

    # ---- per-base-prompt loop ----
    bases = []                                               # list of (ep_global_idx, t)
    base_offsubspaces = []                                   # list[ndarray 4096]
    base_q_rank_idx = []                                     # list[ndarray (4,)]
    base_label_token_ids = []
    base_prompts = []
    base_decision_pos = []                                   # last-token index in tokenization
    base_enc = []

    for ep in args.base_eps:
        log = logs[ep]
        lm = LabelMap(letters=log["letters"])
        history = list(zip(log["actions"][: args.base_trial], log["rewards"][: args.base_trial]))
        prompt = build_chat_prompt(tok, lm, history)
        enc = tok(prompt, return_tensors="pt").to(model.device)
        dp = enc["input_ids"].shape[1] - 1

        # Base activation at our layer + the off-subspace residual
        # Find the position of `ep` in ep_ids
        try:
            local_idx = ep_ids.index(ep) if isinstance(ep_ids, list) else int(np.where(np.asarray(ep_ids) == ep)[0][0])
        except (ValueError, IndexError):
            raise SystemExit(f"episode {ep} is not in activations (ep_ids={ep_ids[:5]}...)")
        base_act = activations[local_idx, args.base_trial, activation_layer_idx, :].float().numpy()
        base_pca_proj = pca_basis @ (base_act - pca_mean)                  # [64]
        base_pca_reconstructed = pca_basis.T @ base_pca_proj + pca_mean    # [4096]
        base_offsubspace = base_act - base_pca_reconstructed               # [4096]

        # Q ranking at this trial (descending → best first)
        Q_t = Q_all[ep, args.base_trial, :].cpu().numpy()
        rank_indices = np.argsort(-Q_t)                                    # [n_arms]

        bases.append((int(ep), int(args.base_trial)))
        base_offsubspaces.append(base_offsubspace)
        base_q_rank_idx.append(rank_indices)
        base_label_token_ids.append(letter_token_ids(tok, lm.letters))
        base_prompts.append(prompt)
        base_decision_pos.append(dp)
        base_enc.append(enc)

    P = len(bases)
    print(f"\n{P} base prompts × {K} waypoints × 2 strategies = {P*K*2} forward passes")

    manifold_dist = np.zeros((P, K, n_arms))
    linear_dist = np.zeros((P, K, n_arms))

    t0 = time.time()
    n_done = 0
    n_total = P * K * 2
    for p in range(P):
        enc = base_enc[p]
        dp = base_decision_pos[p]
        lab_ids = base_label_token_ids[p]
        rank_idx = base_q_rank_idx[p]
        off = base_offsubspaces[p]

        for k in range(K):
            for mode, dist_array, path_pca in [
                ("manifold", manifold_dist, manifold_path_pca),
                ("linear",   linear_dist,   linear_path_pca),
            ]:
                wp_pca = path_pca[k]
                wp_4096 = pca_basis.T @ wp_pca + pca_mean
                if mode == "manifold":
                    steered_np = wp_4096 + off
                else:
                    steered_np = wp_4096
                steered_t = torch.from_numpy(steered_np.astype(np.float32)).to(model.device)

                handle = _install_steer_hook(model, hook_layer_idx, dp, steered_t)
                try:
                    with torch.no_grad():
                        out = model(**enc)
                    logits = out.logits[0, dp, :].float().cpu().numpy()
                finally:
                    handle.remove()

                dist_full = choice_distribution(logits, lab_ids)            # [n_arms + 1]
                arm_dist = dist_full[:n_arms]
                arm_dist = arm_dist / arm_dist.sum()                        # normalize over arms
                # Sort by Q ranking (rank 0 = best Q arm)
                dist_array[p, k] = arm_dist[rank_idx]

                n_done += 1
                if n_done % 20 == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / n_done * (n_total - n_done)
                    print(f"  [{n_done}/{n_total}] elapsed {elapsed/60:.1f}min, "
                          f"eta {eta/60:.1f}min", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "waypoint_q_values":      torch.tensor(q_waypoints, dtype=torch.float32),
        "manifold_path_pca":      torch.tensor(manifold_path_pca, dtype=torch.float32),
        "linear_path_pca":        torch.tensor(linear_path_pca, dtype=torch.float32),
        "manifold_dist_per_base": torch.tensor(manifold_dist, dtype=torch.float32),
        "linear_dist_per_base":   torch.tensor(linear_dist, dtype=torch.float32),
        "manifold_dist_avg":      torch.tensor(manifold_dist.mean(axis=0), dtype=torch.float32),
        "linear_dist_avg":        torch.tensor(linear_dist.mean(axis=0), dtype=torch.float32),
        "base_prompts":           bases,
        "K":                      K,
        "layer":                  activation_layer_idx,
        "hook_layer":             hook_layer_idx,
        "q_low":                  q_low,
        "q_high":                 q_high,
        "model_id":               args.model,
        "value_model":            q_data["model_name"],
        "low_bin":                args.low_bin,
        "high_bin":               args.high_bin,
    }, args.out)
    print(f"\nsaved → {args.out}")
    print(f"  manifold P(rank 0) at low/high endpoint: "
          f"{manifold_dist.mean(0)[0, 0]:.3f} → {manifold_dist.mean(0)[-1, 0]:.3f}")
    print(f"  linear   P(rank 0) at low/high endpoint: "
          f"{linear_dist.mean(0)[0, 0]:.3f} → {linear_dist.mean(0)[-1, 0]:.3f}")


if __name__ == "__main__":
    main()
