"""Within-episode interchange test — does an identity-carrying patch move the choice?

Diagnostic for the steering failure. Bin-centroid steering (`steer.py`) injects a patch
that has been *averaged over which arm is best* (letters are randomized per episode, and
each Q_max bin pools trials with different best arms), so the injected vector cannot tell
the model which of the 4 arms to favor. This script tests the opposite: a patch that DOES
carry arm identity — the real residual of a trial where arm `a` is clearly best — injected
into a base trial *from the same episode* (hence the same letter map) where `a` is a loser.

  source trial t_src:  argmax_a Q[e,t] == a, with a large Q_max - Q_2nd margin
  base trial   t_base: same episode, `a` is low-rank (a different arm is best)

If injecting the source residual at the decision token raises P(a) in the base — and does
so *specifically* for arm `a`, not all arms — then the steering channel works when identity
is preserved, and the centroid approach failed precisely because it averaged identity away.

This is the paper's `locate`/interchange primitive (causalab `layer_scan.py`), adapted to
our bandit. Two intervention modes mirror `steer.py`:
  full      replace the entire decision-token residual with the source activation
            (pure interchange / paper's linear style)
  subspace  replace only the PCA-64 projection, preserve the base off-subspace residual
            (paper's manifold style)
Controls: `identity` (inject the base's own residual → expect ΔP≈0) and `baseline` (no hook).

CONFOUND (stated, not hidden): source and base differ in history, so the patch carries
recency/perseveration alongside value. A positive, arm-specific ΔP(a) shows the channel
carries arm-favoring information; it does NOT by itself isolate value from recency. The
recency-decoupling control is deferred to a round-2 run.

Output (`artifacts/steering/<name>.pt`):
  target_arm        [n]               — arm a each triple steers toward
  triple_meta       list[dict]        — ep, a, t_src, t_base, src_margin, base_rank_a, ...
  P_base            [n, n_arms]        — unhooked base distribution over arms (renormalized)
  P_source_cached   [n, n_arms]        — source trial's natural dist (from collection)
  by_layer[f"L{L}"] = {
      P_identity    [n, n_arms]
      P_full        [n, n_arms]
      P_subspace    [n, n_arms]
  }
  layers, model_id, value_model, args
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
    `position` (token index) with `steered_4096`. Mirrors `steer.py`."""
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


def select_triples(Q, eps, triples_per_arm, min_margin, base_min_rank, rng):
    """Pick (episode, target arm, source trial, base trial) triples from Q trajectories.

    Source = a trial where arm `a` is clearly best (large Q_max - Q_2nd margin).
    Base   = same-episode trial where `a` is rank >= base_min_rank (a loser).
    Returns a list of dicts. `triples_per_arm` controls how many bases per (ep, arm)."""
    triples = []
    n_arms = Q.shape[2]
    for e in eps:
        Qe = Q[e].cpu().numpy()                                   # [T, n_arms]
        T = Qe.shape[0]
        order = np.argsort(-Qe, axis=1)                           # order[t, 0] = best arm
        best = order[:, 0]
        sortedQ = -np.sort(-Qe, axis=1)                           # descending per trial
        margin = sortedQ[:, 0] - sortedQ[:, 1]                    # Q_max - Q_2nd
        rank = np.empty((T, n_arms), dtype=int)
        for t in range(T):
            rank[t, order[t]] = np.arange(n_arms)                 # rank[t, arm] (0 = best)
        gap_a = sortedQ[:, 0][:, None] - Qe                       # Q_max - Q[arm], per trial

        valid_t = np.arange(T) >= 1                               # skip trial 0 (init value)
        for a in range(n_arms):
            src_mask = (best == a) & valid_t
            if not src_mask.any():
                continue
            src_idx = np.where(src_mask)[0]
            t_src = int(src_idx[np.argmax(margin[src_idx])])      # cleanest separation
            if margin[t_src] < min_margin:
                continue

            base_mask = (rank[:, a] >= base_min_rank) & valid_t
            base_mask[t_src] = False
            if not base_mask.any():
                continue
            base_idx = np.where(base_mask)[0]
            # bases where `a` is most clearly NOT best (largest Q_max - Q[a])
            ranked = base_idx[np.argsort(-gap_a[base_idx, a])]
            for t_base in ranked[:triples_per_arm]:
                t_base = int(t_base)
                triples.append({
                    "ep": int(e),
                    "arm": int(a),
                    "t_src": t_src,
                    "t_base": t_base,
                    "src_margin": float(margin[t_src]),
                    "base_rank_a": int(rank[t_base, a]),
                    "base_gap_a": float(gap_a[t_base, a]),
                    "base_best_arm": int(best[t_base]),
                })
    return triples


def _arm_dist(logits, lab_ids, n_arms):
    """Decision-token logits → distribution over the n_arms (renormalized, drop 'other')."""
    dist_full = choice_distribution(logits, lab_ids)
    arm = dist_full[:n_arms]
    return arm / arm.sum()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifolds", type=pathlib.Path, nargs="+", required=True,
                        help="one M_h .pt per layer; supplies pca_basis/pca_mean and layer")
    parser.add_argument("--activations", type=pathlib.Path, required=True)
    parser.add_argument("--q", type=pathlib.Path, required=True)
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--n-eps", type=int, default=40,
                        help="number of cached episodes to draw triples from")
    parser.add_argument("--triples-per-arm", type=int, default=1)
    parser.add_argument("--src-margin-quantile", type=float, default=0.5,
                        help="source margin must exceed this global-quantile of Q_max - Q_2nd")
    parser.add_argument("--base-min-rank", type=int, default=2,
                        help="target arm must be at least this rank in the base (0=best)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # ---- PCA per layer ----
    layer_pca = {}                                               # L -> (basis [64,4096], mean [4096])
    for mpath in args.manifolds:
        mh = torch.load(mpath, weights_only=False)
        L = int(mh["layer"])
        layer_pca[L] = (mh["pca_basis"].numpy().astype(np.float32),
                        mh["pca_mean"].numpy().astype(np.float32))
    layers = sorted(layer_pca)
    print(f"layers under test (hidden_states index): {layers}")

    # ---- data ----
    act_data = torch.load(args.activations, weights_only=False)
    activations = act_data["activations"]                        # [n_eps, T, n_layers, 4096] fp16
    ep_ids = act_data["episode_ids"]
    ep_pos = {int(e): i for i, e in enumerate(ep_ids)}

    q_data = torch.load(args.q, weights_only=False)
    Q_all = q_data["Q"]                                          # [n_total_eps, T, n_arms]
    n_arms = int(q_data["n_arms"])

    logs = json.loads(args.dataset.read_text())

    # ---- triple selection ----
    rng = np.random.default_rng(args.seed)
    cached_eps = [int(e) for e in ep_ids][: args.n_eps]
    all_margins = (-np.sort(-Q_all[cached_eps].cpu().numpy(), axis=2))
    all_margins = (all_margins[:, 1:, 0] - all_margins[:, 1:, 1]).ravel()
    min_margin = float(np.quantile(all_margins, args.src_margin_quantile))
    print(f"source margin threshold (q={args.src_margin_quantile}): {min_margin:.2f}")

    triples = select_triples(Q_all, cached_eps, args.triples_per_arm,
                             min_margin, args.base_min_rank, rng)
    n = len(triples)
    print(f"selected {n} triples across {len(cached_eps)} episodes")
    if n == 0:
        raise SystemExit("no triples selected — loosen --src-margin-quantile / --base-min-rank")

    # ---- model ----
    model, tok = load_model(args.model)
    print(f"loaded {args.model} on {model.device}", flush=True)

    target_arm = np.array([t["arm"] for t in triples])
    P_base = np.zeros((n, n_arms))
    P_source_cached = np.zeros((n, n_arms))
    by_layer = {L: {k: np.zeros((n, n_arms)) for k in ("P_identity", "P_full", "P_subspace")}
                for L in layers}

    n_fwd_per = 1 + len(layers) * 3
    n_total = n * n_fwd_per
    print(f"\n{n} triples × {n_fwd_per} forward passes = {n_total} total", flush=True)

    t0 = time.time()
    n_done = 0
    for i, tr in enumerate(triples):
        e, a, t_src, t_base = tr["ep"], tr["arm"], tr["t_src"], tr["t_base"]
        local = ep_pos[e]
        log = logs[e]
        lm = LabelMap(letters=log["letters"])
        lab_ids = letter_token_ids(tok, lm.letters)

        # source ceiling from collection-time distribution (context only)
        src_cd = np.asarray(log["choice_dists"][t_src][:n_arms], dtype=np.float64)
        P_source_cached[i] = src_cd / src_cd.sum()

        # base prompt
        history = list(zip(log["actions"][:t_base], log["rewards"][:t_base]))
        enc = tok(build_chat_prompt(tok, lm, history), return_tensors="pt").to(model.device)
        dp = enc["input_ids"].shape[1] - 1

        # baseline (no hook)
        with torch.no_grad():
            logits = model(**enc).logits[0, dp, :].float().cpu().numpy()
        P_base[i] = _arm_dist(logits, lab_ids, n_arms)
        n_done += 1

        for L in layers:
            basis, mean = layer_pca[L]
            hook_layer = L - 1
            src_act = activations[local, t_src, L, :].float().numpy()
            base_act = activations[local, t_base, L, :].float().numpy()

            src_insub = basis.T @ (basis @ (src_act - mean)) + mean
            base_insub = basis.T @ (basis @ (base_act - mean)) + mean
            base_off = base_act - base_insub

            patches = {
                "P_identity": base_act,                          # expect ≈ baseline
                "P_full":     src_act,                           # full interchange
                "P_subspace": src_insub + base_off,              # subspace swap, keep base off
            }
            for key, patch in patches.items():
                steered = torch.from_numpy(patch.astype(np.float32)).to(model.device)
                handle = _install_steer_hook(model, hook_layer, dp, steered)
                try:
                    with torch.no_grad():
                        logits = model(**enc).logits[0, dp, :].float().cpu().numpy()
                finally:
                    handle.remove()
                by_layer[L][key][i] = _arm_dist(logits, lab_ids, n_arms)
                n_done += 1

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / n_done * (n_total - n_done)
            print(f"  [{i+1}/{n}] elapsed {elapsed/60:.1f}min, eta {eta/60:.1f}min", flush=True)

    # ---- save ----
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "target_arm":      torch.tensor(target_arm),
        "triple_meta":     triples,
        "P_base":          torch.tensor(P_base, dtype=torch.float32),
        "P_source_cached": torch.tensor(P_source_cached, dtype=torch.float32),
        "by_layer":        {f"L{L}": {k: torch.tensor(v, dtype=torch.float32)
                                      for k, v in d.items()}
                            for L, d in by_layer.items()},
        "layers":          layers,
        "model_id":        args.model,
        "value_model":     q_data["model_name"],
        "args":            vars(args) | {"manifolds": [str(p) for p in args.manifolds],
                                         "activations": str(args.activations),
                                         "q": str(args.q), "dataset": str(args.dataset),
                                         "out": str(args.out)},
    }
    torch.save(out, args.out)
    print(f"\nsaved → {args.out}")

    # ---- summary ----
    tgt = target_arm
    base_tgt = P_base[np.arange(n), tgt]
    src_tgt = P_source_cached[np.arange(n), tgt]
    print(f"\nbaseline P(target arm) = {base_tgt.mean():.3f} ± {base_tgt.std()/np.sqrt(n):.3f}")
    print(f"source   P(target arm) = {src_tgt.mean():.3f} (cached ceiling)\n")
    for L in layers:
        d = by_layer[L]
        for key in ("P_identity", "P_full", "P_subspace"):
            arr = d[key]
            tgt_p = arr[np.arange(n), tgt]
            dP_tgt = tgt_p - base_tgt
            # specificity: mean ΔP over the non-target arms
            others = np.array([[arr[r, j] - P_base[r, j]
                                for j in range(n_arms) if j != tgt[r]] for r in range(n)])
            succ = float((dP_tgt > 0).mean())
            print(f"L{L} {key:11s}  ΔP(target)={dP_tgt.mean():+.3f} ± "
                  f"{dP_tgt.std()/np.sqrt(n):.3f}  |  ΔP(others)={others.mean():+.3f}  |  "
                  f"P(target)→{tgt_p.mean():.3f}  |  success={succ:.0%}")


if __name__ == "__main__":
    main()
