"""Collect Llama's residual-stream activations at the decision token, every layer.

For each of the first `--n-eps` episodes in the dataset, replay the saved
action+reward sequence and at each trial extract the residual stream at the decision
token at every layer. Saves a `[n_eps, T, n_layers + 1, hidden_dim]` fp16 tensor —
the input to Step 4 (locate) probing.

Uses the same KV-cache trick as the rollout (LCP, crop, append divergent tail), so each
trial only forwards ~10–30 new tokens. Saves incrementally every `--save-every` episodes
so a long run is crash-safe.

    python -m analyses.collect_activations \
        --dataset artifacts/datasets/llama8b_dataset.json \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --n-eps 100 \
        --out artifacts/activations/llama8b_residual.pt
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import torch

from llm_data.llm import CachedEpisode, load_model
from llm_data.prompt import LabelMap


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--n-eps", type=int, default=100)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--save-every", type=int, default=25,
                        help="checkpoint frequency (episodes)")
    args = parser.parse_args()

    logs = json.loads(args.dataset.read_text())
    selected = list(range(min(args.n_eps, len(logs))))
    T = len(logs[0]["actions"])

    model, tok = load_model(args.model)
    print(f"loaded {args.model} on {model.device}", flush=True)

    hidden_dim = model.config.hidden_size
    n_layers = model.config.num_hidden_layers + 1  # +1 for embedding output
    n_eps = len(selected)
    size_gb = n_eps * T * n_layers * hidden_dim * 2 / 1e9
    print(
        f"hidden_dim={hidden_dim}, n_layers (incl. embedding)={n_layers}; "
        f"output: [{n_eps}, {T}, {n_layers}, {hidden_dim}] fp16 = {size_gb:.1f} GB",
        flush=True,
    )

    activations = torch.empty(n_eps, T, n_layers, hidden_dim, dtype=torch.float16)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for i, ep_idx in enumerate(selected):
        log = logs[ep_idx]
        lm = LabelMap(letters=log["letters"])
        ep = CachedEpisode(model, tok, lm)

        for t in range(T):
            h = ep.decision_token_hidden_states()                  # [n_layers, hidden_dim]
            activations[i, t] = h.to(torch.float16).cpu()
            ep.append(log["actions"][t], log["rewards"][t])

        if (i + 1) % args.save_every == 0 or i == n_eps - 1:
            torch.save({
                "activations":    activations[: i + 1].clone(),
                "episode_ids":    selected[: i + 1],
                "model_id":       args.model,
                "n_layers":       n_layers,
                "hidden_dim":     hidden_dim,
                "n_trials":       T,
            }, args.out)
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_eps - i - 1)
            print(
                f"[{i + 1}/{n_eps}] saved; elapsed {elapsed / 60:.1f} min, "
                f"eta {eta / 60:.1f} min",
                flush=True,
            )

    print(f"done → {args.out}")


if __name__ == "__main__":
    main()
