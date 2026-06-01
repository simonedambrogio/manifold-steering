"""Throwaway sanity check for the steering forward hook.

Four interventions on the SAME base prompt at layer 19's decision token:
  1. BASELINE  — no hook installed.
  2. IDENTITY  — patch with the captured base activation. Should ≈ baseline (verifies
                 the hook *passes through* without corrupting).
  3. ZERO      — patch with a zero vector. Should dramatically change the distribution
                 (verifies the hook *actually fires* and reaches downstream layers).
  4. RANDOM    — patch with a large Gaussian. Same expectation as ZERO.

If IDENTITY matches BASELINE and ZERO/RANDOM differ wildly, the hook is correct and the
weak steering effect we saw is a real property of layer-19 intervention, not a bug.
"""

from __future__ import annotations
import json, pathlib
import numpy as np
import torch

from llm_data.llm import load_model, letter_token_ids
from llm_data.prompt import LabelMap, build_chat_prompt, choice_distribution

EP_IDX = 80      # global episode index
LOCAL_IDX = 0    # corresponding index in activations.pt (we collected eps 0..99 sequentially)
T = 75
HOOK_LAYER = 18  # patches output of model.model.layers[18] = hidden_states[19]
DECISION_POS = -1

model, tok = load_model("meta-llama/Llama-3.1-8B-Instruct")

logs = json.loads(pathlib.Path("artifacts/datasets/llama8b_dataset.json").read_text())
log = logs[EP_IDX]
lm = LabelMap(letters=log["letters"])
history = list(zip(log["actions"][:T], log["rewards"][:T]))
prompt = build_chat_prompt(tok, lm, history)
enc = tok(prompt, return_tensors="pt").to(model.device)
dp = enc["input_ids"].shape[1] - 1
lab_ids = letter_token_ids(tok, lm.letters)

act_data = torch.load("artifacts/activations/llama8b_residual.pt", weights_only=False)
base_act_np = act_data["activations"][LOCAL_IDX, T, 19, :].float().numpy()
base_act = torch.from_numpy(base_act_np.astype(np.float32)).to(model.device)
print(f"base activation L19: norm={float(np.linalg.norm(base_act_np)):.2f}, "
      f"std={float(base_act_np.std()):.4f}\n")


def install_hook(steered: torch.Tensor):
    target = model.model.layers[HOOK_LAYER]
    def hook(_m, _i, output):
        if isinstance(output, tuple):
            hs = output[0].clone()
            hs[:, dp, :] = steered.to(hs.dtype).to(hs.device)
            return (hs,) + output[1:]
        hs = output.clone()
        hs[:, dp, :] = steered.to(hs.dtype).to(hs.device)
        return hs
    return target.register_forward_hook(hook)


def run_get_dist():
    with torch.no_grad():
        out = model(**enc)
    logits = out.logits[0, dp, :].float().cpu().numpy()
    return choice_distribution(logits, lab_ids)


def run(name: str, patched: torch.Tensor | None):
    if patched is None:
        d = run_get_dist()
    else:
        h = install_hook(patched)
        try:
            d = run_get_dist()
        finally:
            h.remove()
    print(f"{name:<32} choice dist (4 arms + other) = {np.round(d, 4)}")
    return d


d_base = run("1. BASELINE (no hook)", None)
d_id   = run("2. IDENTITY (patch base act)", base_act)
d_zero = run("3. ZERO patch", torch.zeros_like(base_act))
torch.manual_seed(0)
rnd = torch.randn_like(base_act) * (base_act.std() * 5)
d_rnd = run("4. RANDOM patch (5× std)", rnd)

print()
print(f"L1 dist (IDENTITY vs BASELINE): {float(np.abs(d_id - d_base).sum()):.5f}  "
      f"(expect ~0)")
print(f"L1 dist (ZERO vs BASELINE):     {float(np.abs(d_zero - d_base).sum()):.5f}  "
      f"(expect large)")
print(f"L1 dist (RANDOM vs BASELINE):   {float(np.abs(d_rnd - d_base).sum()):.5f}  "
      f"(expect large)")
