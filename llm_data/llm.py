"""Model loading and choice readout for the LLM bandit (PyTorch; MPS/CUDA/CPU).

Deliberately separate from causalab's intervention pipeline: this is the lightweight
interface the closed-loop rollout (and the learning-gate baseline) use to (a) load an
instruct model and (b) read the next-token choice distribution after the `You press <<`
prefill. The manifold/steering analyses later use causalab's `LMPipeline` instead.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from llm_data.prompt import ELICIT_STUB, LabelMap, build_chat_prompt, choice_distribution


def resolve_device(device: str | None = None) -> str:
    """Pick a device: explicit value, else cuda -> mps -> cpu."""
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_id: str, dtype: str = "float16", device: str | None = None):
    """Load an instruct model + tokenizer onto the resolved device, in eval mode."""
    device = resolve_device(device)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = (
        AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype))
        .to(device)
        .eval()
    )
    return model, tok


def letter_token_ids(tokenizer, letters: list[str]) -> list[int]:
    """Token id emitted for each label letter immediately after `You press <<`.

    Raises if a letter is not a clean single token in this tokenizer (so we never silently
    mis-read the choice distribution).
    """
    base = tokenizer(ELICIT_STUB, add_special_tokens=False)["input_ids"]
    ids: list[int] = []
    for letter in letters:
        enc = tokenizer(ELICIT_STUB + letter, add_special_tokens=False)["input_ids"]
        extra = enc[len(base):] if enc[: len(base)] == base else None
        if not extra or len(extra) != 1:
            raise ValueError(
                f"label {letter!r} is not a single token after '<<' for this tokenizer"
            )
        ids.append(extra[0])
    return ids


@torch.no_grad()
def read_choice_distribution(model, tokenizer, prompt: str, label_ids: list[int]) -> np.ndarray:
    """Forward pass; return the (n_arms + 1,) choice distribution at the final token."""
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    logits = model(**enc).logits[0, -1].float().cpu().numpy()
    return choice_distribution(logits, label_ids)


def _longest_common_prefix(a: list[int], b) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == int(b[i]):
        i += 1
    return i


class CachedEpisode:
    """Incremental KV-cached prompt for one episode (chat-format instruct models).

    Each trial we tokenize the full prompt (cheap, on CPU) but only *forward* the tokens
    that differ from what's cached: keep the longest common prefix of the cache and the new
    token sequence, crop the cache to it, and run the model only over the divergent tail
    (the new history line + the fixed question / `You press <<` suffix). This feeds the
    model the *exact* same tokens as the non-cached reader — so results match it to fp
    precision — while each trial processes only ~tens of tokens, not the whole prompt.
    """

    def __init__(self, model, tokenizer, label_map: LabelMap):
        self.model = model
        self.tok = tokenizer
        self.lm = label_map
        self.label_ids = letter_token_ids(tokenizer, label_map.letters)
        self.history: list[tuple[int, int]] = []
        self.cache = DynamicCache()
        self.cached_ids: list[int] = []

    @torch.no_grad()
    def choice_distribution(self) -> np.ndarray:
        """Read the choice distribution at the current history."""
        prompt = build_chat_prompt(self.tok, self.lm, self.history)
        full = self.tok(prompt, return_tensors="pt").input_ids[0]
        k = _longest_common_prefix(self.cached_ids, full)
        if len(self.cached_ids) > k:
            self.cache.crop(k)
        new_ids = full[k:].unsqueeze(0).to(self.model.device)
        out = self.model(new_ids, past_key_values=self.cache, use_cache=True)
        self.cache = out.past_key_values
        self.cached_ids = full.tolist()
        logits = out.logits[0, -1].float().cpu().numpy()
        return choice_distribution(logits, self.label_ids)

    def append(self, arm: int, reward: int) -> None:
        """Record a chosen arm + observed reward (consumed at the next read)."""
        self.history.append((int(arm), int(reward)))

    @torch.no_grad()
    def decision_token_hidden_states(self) -> torch.Tensor:
        """Residual-stream activations at the decision token, at every layer.

        Returns shape `[n_layers + 1, hidden_dim]` (index 0 = embedding output;
        index `i ≥ 1` = output of transformer layer i). Same incremental-cache
        contract as `choice_distribution`: only the divergent tail is forwarded.
        """
        prompt = build_chat_prompt(self.tok, self.lm, self.history)
        full = self.tok(prompt, return_tensors="pt").input_ids[0]
        k = _longest_common_prefix(self.cached_ids, full)
        if len(self.cached_ids) > k:
            self.cache.crop(k)
        new_ids = full[k:].unsqueeze(0).to(self.model.device)
        out = self.model(
            new_ids,
            past_key_values=self.cache,
            use_cache=True,
            output_hidden_states=True,
        )
        self.cache = out.past_key_values
        self.cached_ids = full.tolist()
        # hidden_states: tuple of (n_layers + 1) tensors each [1, seq, hidden_dim].
        # Take the last suffix token (= decision token) at every layer.
        return torch.stack([h[0, -1] for h in out.hidden_states], dim=0)


@torch.no_grad()
def read_choice_distributions_batched(
    model, tokenizer, prompts: list[str], label_ids_list: list[list[int]]
) -> np.ndarray:
    """Batched choice readout: one forward pass over left-padded prompts.

    Left padding keeps every sequence's final (real) token at position -1, so logits[:, -1]
    is the decision token for all rows regardless of length. Returns (B, n_arms + 1).
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    logits = model(**enc).logits[:, -1, :].float().cpu().numpy()
    return np.stack(
        [choice_distribution(logits[i], lids) for i, lids in enumerate(label_ids_list)]
    )
