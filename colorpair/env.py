"""The color-pair comparison task: choose the pair of colors closer on the (hidden) color ring.

A trial shows four colors forming two pairs, (c0,c1) and (c2,c3). The agent must pick the pair whose
two colors are closer in **circular** distance on a ring of N colors:

    d_circ(i, j) = min(|i - j|, N - |i - j|)

The ring is *latent*: it is never given to the agent. The input is a **one-hot** color code, under which
all colors are equidistant (any two distinct one-hots are sqrt(2) apart) -- it asserts no order, no
adjacency, no ring. The ring lives *entirely in the reward labels* (which pair is closer). The agent can
only recover it from the task. This is the fairest possible input: a scalar index would impose a line (and
bias against the wraparound); a cos/sin code would hand over the ring outright.

The decisive structure is the **seam**: a pair like (N-1, 1) is circularly *near* (distance 2) but
linearly *far* (|N-1 - 1| = N-2). A model that only learns linear distance gets seam pairs wrong. To pass
them it must fold the colors into a ring -- the folding geometry the manifold-steering test needs.

Two regimes (cf. compositional-generalization design in `dev/context/abcd-loop/status.md`):
  - `make_trials(..., allowed_pairs=...)`: random trials drawn only from a permitted set of color pairs.
    We hold out a fraction of pairs (and all seam pairs) from training, so the model cannot memorize a
    lookup of the N*(N-1)/2 distances; generalizing to held-out pairs *requires* the ring metric.
  - `seam_trials(...)`: held-out diagnostic trials that pit a circularly-near seam pair against a
    linearly-near non-seam pair, where circular and linear distance disagree on the answer.

Model-free; run as a script for the self-test:

    python -m colorpair.env
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_COLORS: int = 24  # ring size; even, so an antipodal color exists (offset N/2)


def circular_distance(i, j, n: int = N_COLORS):
    """Shorter arc length between colors i, j on the ring (array-aware)."""
    d = np.abs(np.asarray(i) - np.asarray(j))
    return np.minimum(d, n - d)


def linear_distance(i, j):
    """Naive non-circular distance |i - j| -- what a line (non-folding) code computes."""
    return np.abs(np.asarray(i) - np.asarray(j))


def all_pairs(n: int = N_COLORS) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def is_seam_pair(i: int, j: int, n: int = N_COLORS) -> bool:
    """True if the short arc between i and j wraps through the seam (|i - j| > N/2).

    These are exactly the pairs on which circular and linear distance disagree (circularly near but
    linearly far), so they are the pairs a non-folding linear code must get wrong.
    """
    return abs(i - j) > n / 2


def seam_pairs(n: int = N_COLORS) -> list[tuple[int, int]]:
    return [(i, j) for (i, j) in all_pairs(n) if is_seam_pair(i, j, n)]


@dataclass
class Split:
    """A train/held-out partition of color pairs (the compositional-generalization split)."""

    n_colors: int
    train_pairs: list[tuple[int, int]]
    heldout_pairs: list[tuple[int, int]]   # pairs never shown in training (incl. all seam pairs)


def make_split(n: int = N_COLORS, heldout_frac: float = 0.15, seed: int = 0,
               holdout_all_seam: bool = False) -> Split:
    """Train/held-out partition of color pairs. Two regimes:

    Default (`holdout_all_seam=False`) -- hold out a random `heldout_frac` of **all** pairs (seam and
    non-seam alike). Seam pairs are then *present in training*, so the wraparound is part of the task:
    a non-folding code cannot fit it, and the model must build the ring. Withholding whole pairs still
    blocks a memorized distance lookup, so `phi` must carry a reusable ring metric (not a table).

    Contrast (`holdout_all_seam=True`) -- hold out *every* seam pair (plus a random `heldout_frac` of
    the non-seam pairs). Now training never demands the wrap; the model extrapolates linearly and fails
    the seam. This is the control showing the ring is built *only when the task requires it*.
    """
    rng = np.random.default_rng(seed)
    if holdout_all_seam:
        seam = set(seam_pairs(n))
        non_seam = [p for p in all_pairs(n) if p not in seam]
        rng.shuffle(non_seam)
        k = int(round(heldout_frac * len(non_seam)))
        heldout = list(seam) + non_seam[:k]
        train = non_seam[k:]
    else:
        pairs = all_pairs(n)
        rng.shuffle(pairs)
        k = int(round(heldout_frac * len(pairs)))
        heldout = pairs[:k]
        train = pairs[k:]
    return Split(n_colors=n, train_pairs=train, heldout_pairs=heldout)


def _sample_trials(rng, n_trials: int, pool: list[tuple[int, int]], n: int):
    """Draw `n_trials` from a pool of pairs: two distinct pairs per trial, no distance ties.

    Returns colors [T, 4] = (c0, c1, c2, c3) and labels [T] where label = 1.0 iff pair 0 (the first
    pair) is the closer pair. Which pair is presented first is randomized (no positional bias).
    """
    pool_arr = np.asarray(pool)
    colors = np.empty((n_trials, 4), dtype=np.int64)
    labels = np.empty((n_trials,), dtype=np.float32)
    t = 0
    while t < n_trials:
        a, b = pool_arr[rng.integers(0, len(pool_arr), size=2)]
        d0, d1 = circular_distance(*a, n), circular_distance(*b, n)
        if d0 == d1:
            continue
        if rng.random() < 0.5:                       # randomize presentation order
            a, b, d0, d1 = b, a, d1, d0
        # randomize within-pair color order too
        p0 = a[::-1] if rng.random() < 0.5 else a
        p1 = b[::-1] if rng.random() < 0.5 else b
        colors[t] = [p0[0], p0[1], p1[0], p1[1]]
        labels[t] = 1.0 if d0 < d1 else 0.0
        t += 1
    return colors, labels


def make_trials(rng, n_trials: int, n: int = N_COLORS,
                allowed_pairs: list[tuple[int, int]] | None = None):
    """Random comparison trials. `allowed_pairs=None` uses every pair; pass a split's train_pairs to
    restrict training to the non-held-out pairs."""
    pool = allowed_pairs if allowed_pairs is not None else all_pairs(n)
    return _sample_trials(rng, n_trials, pool, n)


def seam_trials(rng, n_trials: int, n: int = N_COLORS,
                seam_pool: list[tuple[int, int]] | None = None,
                nonseam_pool: list[tuple[int, int]] | None = None):
    """Diagnostic trials: a **seam** pair (pair 0) vs a non-seam pair (pair 1), kept only where
    circular and linear distance give *opposite* answers. A linear (non-folding) code scores 0 here
    by construction; a ring code scores high. Pass `seam_pool=<held-out seam pairs>` to test
    generalization of the ring metric to seam pairs the model never trained on.
    """
    seam = seam_pool if seam_pool is not None else seam_pairs(n)
    non_seam = nonseam_pool if nonseam_pool is not None else [
        p for p in all_pairs(n) if not is_seam_pair(*p, n)]
    seam_arr, non_arr = np.asarray(seam), np.asarray(non_seam)
    colors = np.empty((n_trials, 4), dtype=np.int64)
    labels = np.empty((n_trials,), dtype=np.float32)
    lin_label = np.empty((n_trials,), dtype=np.float32)
    t = 0
    while t < n_trials:
        a = seam_arr[rng.integers(0, len(seam_arr))]          # circularly near, linearly far
        b = non_arr[rng.integers(0, len(non_arr))]            # non-seam
        dc0, dc1 = circular_distance(*a, n), circular_distance(*b, n)
        dl0, dl1 = linear_distance(*a), linear_distance(*b)
        if dc0 == dc1 or dl0 == dl1:
            continue
        circ_ans = 1.0 if dc0 < dc1 else 0.0
        lin_ans = 1.0 if dl0 < dl1 else 0.0
        if circ_ans == lin_ans:                               # keep only discriminating trials
            continue
        colors[t] = [a[0], a[1], b[0], b[1]]                  # pair 0 = seam pair
        labels[t] = circ_ans
        lin_label[t] = lin_ans
        t += 1
    return colors, labels, lin_label


def _self_test() -> None:
    n = N_COLORS
    rng = np.random.default_rng(0)
    # circular distance basics
    assert circular_distance(0, 1, n) == 1
    assert circular_distance(n - 1, 0, n) == 1               # the wraparound is adjacent
    assert circular_distance(0, n // 2, n) == n // 2         # antipodal is the max
    assert is_seam_pair(n - 1, 1, n) and circular_distance(n - 1, 1, n) == 2

    split = make_split(n, heldout_frac=0.15, seed=0)                # default: seam pairs in training
    assert set(split.train_pairs).isdisjoint(split.heldout_pairs)
    train_seam = [p for p in split.train_pairs if is_seam_pair(*p, n)]
    assert len(train_seam) > 0                                      # the wrap is part of the task
    total = len(all_pairs(n))
    print(f"N={n}: {total} pairs  ->  train {len(split.train_pairs)} ({len(train_seam)} seam)  "
          f"heldout {len(split.heldout_pairs)}")

    ctrl = make_split(n, heldout_frac=0.15, seed=0, holdout_all_seam=True)
    assert set(seam_pairs(n)).issubset(set(ctrl.heldout_pairs))     # control: all seam held out

    colors, labels = make_trials(rng, 5000, n, split.train_pairs)
    # label sanity: pair-0 closer exactly when label==1
    d0 = circular_distance(colors[:, 0], colors[:, 1], n)
    d1 = circular_distance(colors[:, 2], colors[:, 3], n)
    assert np.all((d0 < d1) == (labels == 1.0))
    assert abs(labels.mean() - 0.5) < 0.03                   # no positional bias
    print(f"train trials: {len(labels)}, frac pair-0-closer = {labels.mean():.3f}")

    sc, sl, sll = seam_trials(rng, 3000, n)
    # by construction the linear answer is the opposite of the circular answer on every seam trial
    assert np.all(sl != sll)
    # a linear-distance classifier scores 0 on these; circular (the labels) scores 1
    print(f"seam trials: {len(sl)}, linear-classifier acc on them = {(sll == sl).mean():.3f} (=0 by design)")
    print("self-test passed.")


if __name__ == "__main__":
    _self_test()
