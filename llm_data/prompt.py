"""Prompt construction and choice readout for the LLM 4-armed bandit.

Design (see dev/context/value-steering/step1.md, "Prompt design — constraints from prior art"):

- **Raw per-trial history**, never summarized — value integration must happen inside the
  network so there is an internal latent to probe (Krishnamurthy et al. 2024 found
  summaries aid exploration, but that defeats our purpose).
- **Direct choice, no chain-of-thought** — keeps the value computation localized at one
  decision-token activation.
- **`<< >>` delimiter** (Centaur / Psych-101): trials read `You press <<X>> and get R
  points.`; the next choice is elicited by ending the prompt at `You press <<` and reading
  the next-token distribution over the arm letters (+ "other").
- **Randomized, non-adjacent letter labels, remapped per episode** — avoids ordinal,
  position, and token-identity biases. Letters are drawn from a consonant pool (no `A`, no
  obvious sequence).

The choice-distribution reader is backend-agnostic: it takes a `next_token_logits` tensor,
so it works with a raw HF model, causalab's `LMPipeline`, or a stub in tests.

Run as a script for the model-free self-test:

    python prompt.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Consonant pool: reliably single-token across BPE vocabularies, no `A` (avoids a
# "first/default" prior), no natural ordinal run.
LABEL_POOL: tuple[str, ...] = tuple("BCDFGHJKLMNPQRSTVWXZ")

INSTRUCTION_TEMPLATE: str = (
    "You repeatedly choose among {n} slot machines labelled {labels}. "
    "Each pays a varying number of points; the payouts drift slowly over time, "
    "so the best machine changes. Try to win as many points as possible.\n\n"
)

TRIAL_TEMPLATE: str = "You press <<{letter}>> and get {reward} points.\n"
ELICIT_STUB: str = "You press <<"


@dataclass
class LabelMap:
    """Per-episode assignment of arm index -> single-letter label, and its inverse."""

    letters: list[str]  # letters[arm] = label shown for that arm

    @property
    def n_arms(self) -> int:
        return len(self.letters)

    @property
    def to_arm(self) -> dict[str, int]:
        return {letter: arm for arm, letter in enumerate(self.letters)}

    def arm_of(self, letter: str) -> int | None:
        return self.to_arm.get(letter)


def sample_label_map(rng: np.random.Generator, n_arms: int = 4) -> LabelMap:
    """Draw a fresh random arm->letter assignment for one episode."""
    letters = list(rng.choice(np.array(LABEL_POOL), size=n_arms, replace=False))
    return LabelMap(letters=[str(x) for x in letters])


def build_prompt(
    labels: LabelMap,
    history: list[tuple[int, int]],
    elicit: bool = True,
    instruction: str | None = None,
) -> str:
    """Render the full bandit prompt.

    Args:
        labels: the episode's arm->letter map.
        history: list of (arm_index, reward) for past trials, in order.
        elicit: if True, append the `You press <<` stub so the next token is the choice.
        instruction: override the default preamble (e.g. to match NHB wording exactly).
    """
    preamble = instruction if instruction is not None else INSTRUCTION_TEMPLATE.format(
        n=labels.n_arms, labels=", ".join(labels.letters)
    )
    lines = [
        TRIAL_TEMPLATE.format(letter=labels.letters[arm], reward=int(reward))
        for arm, reward in history
    ]
    text = preamble + "".join(lines)
    if elicit:
        text += ELICIT_STUB
    return text


# Chat-format templates for instruct models (verified necessary — see step1.md).
SYSTEM_TEMPLATE: str = (
    "You are playing a game with {n} slot machines labelled {labels}. "
    "Each pays a varying number of points; the payouts drift slowly over time, "
    "so the best machine can change. Try to win as many points as possible. "
    "Respond with only the machine you press, written as <<X>>."
)
USER_SUFFIX: str = "\nWhich machine do you press next?"


def build_chat_prompt(
    tokenizer,
    labels: LabelMap,
    history: list[tuple[int, int]],
    system: str | None = None,
    elicit: bool = True,
) -> str:
    """Render the prompt through an instruct model's chat template.

    Instruct models must be addressed via their chat template: feeding a raw completion
    prompt leaves them off-distribution and yields near-uniform, value-insensitive choices
    (verified on Qwen2.5-7B — see step1.md). The history goes in the user turn; the
    assistant turn is prefilled with `You press <<` so the next token is the choice,
    preserving the `<<X>>` readout and a clean decision-token position.
    """
    sys_msg = system if system is not None else SYSTEM_TEMPLATE.format(
        n=labels.n_arms, labels=", ".join(labels.letters)
    )
    user_msg = (
        "".join(
            TRIAL_TEMPLATE.format(letter=labels.letters[arm], reward=int(reward))
            for arm, reward in history
        )
        + USER_SUFFIX
    )
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if elicit:
        text += ELICIT_STUB
    return text


def parse_choice(generated: str, labels: LabelMap) -> int | None:
    """Map a model continuation after the `You press <<` stub to an arm index.

    Used for closed-loop sampling; returns None if no valid label is found.
    """
    for ch in generated:
        if ch in labels.to_arm:
            return labels.to_arm[ch]
    return None


def choice_distribution(
    next_token_logits: np.ndarray,
    label_token_ids: list[int],
) -> np.ndarray:
    """Convert next-token logits into a distribution over arms + an "other" bin.

    Args:
        next_token_logits: (vocab,) logits for the token following `You press <<`.
        label_token_ids: token id emitted for each arm's letter (index = arm).

    Returns:
        (n_arms + 1,) distribution; last entry is residual "other" mass.
    """
    logits = np.asarray(next_token_logits, dtype=np.float64)
    logits = logits - logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    arm_probs = probs[np.asarray(label_token_ids)]
    other = max(0.0, 1.0 - arm_probs.sum())
    return np.concatenate([arm_probs, [other]])


# --- model-free self-test ---------------------------------------------------


def _selftest() -> None:
    rng = np.random.default_rng(0)

    # label map: distinct letters from the pool, deterministic under seed
    lm = sample_label_map(rng, n_arms=4)
    assert lm.n_arms == 4
    assert len(set(lm.letters)) == 4, "labels must be distinct"
    assert all(c in LABEL_POOL for c in lm.letters)
    assert "A" not in lm.letters
    print("label map:", lm.letters, "->", lm.to_arm)

    # prompt rendering
    history = [(0, 58), (0, 61), (2, 22), (1, 47)]
    prompt = build_prompt(lm, history)
    print("\n--- example prompt ---\n" + prompt + "  <<read distribution here>>")
    assert prompt.endswith(ELICIT_STUB)
    assert prompt.count("You press <<") == len(history) + 1  # history + elicitation
    for arm, reward in history:
        assert f"<<{lm.letters[arm]}>> and get {reward} points." in prompt

    # round-trip parse of a simulated continuation
    for arm in range(4):
        cont = f"{lm.letters[arm]}>> and get 31 points."
        assert parse_choice(cont, lm) == arm
    assert parse_choice(">> nonsense", lm) is None

    # choice_distribution: arms get their logit mass, rest -> other, sums to 1
    vocab = 50
    label_ids = [3, 9, 17, 42]
    logits = np.zeros(vocab)
    logits[label_ids] = [2.0, 1.0, 0.0, -1.0]
    dist = choice_distribution(logits, label_ids)
    assert dist.shape == (5,)
    assert abs(dist.sum() - 1.0) < 1e-9
    assert np.argmax(dist[:4]) == 0  # highest-logit arm wins
    print("\nchoice distribution (4 arms + other):", np.round(dist, 3))

    print("\nself-test OK")


if __name__ == "__main__":
    _selftest()
