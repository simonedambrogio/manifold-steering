"""Reward-schedule generator for the 4-armed restless bandit (Daw et al. 2006 / NHB 2026).

Each arm's latent mean follows a mean-reverting Gaussian random walk; the observed
reward is a noisy, integer-rounded readout clipped to [1, 100]. Parameters follow
Eckstein et al. (2026, Nat. Hum. Behav.) Methods, p980 (= the canonical Daw 2006 process):

    mu[1, i] ~ Uniform(1, 100)                         (independent per arm)
    mu[t, i] ~ Normal(lam * mu[t-1, i] + (1-lam)*50, sigma_d)   (reverts toward 50)
    r[t, i]  ~ Normal(mu[t, i], sigma_o)  -> round, clip to [1, 100]

with lam = 0.9836, sigma_d = 2.8, sigma_o = 4. A fresh schedule (seed) is drawn per
episode. The latent mean is left unclipped per the equation as written; mean-reversion
keeps it bounded in practice (stationary std ~= sigma_d / sqrt(1 - lam**2) ~= 15.5).

Run as a script to regenerate a few schedules and sanity-check them against NHB Fig 1e:

    python reward_schedule.py --n-schedules 3 --out schedules.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --- NHB / Daw-2006 reward-schedule parameters (Methods, p980) --------------
N_ARMS: int = 4
N_TRIALS: int = 150
LAMBDA: float = 0.9836  # centrality / decay toward the revert target
SIGMA_D: float = 2.8  # diffusion (drift) noise std on the latent mean
SIGMA_O: float = 4.0  # observation noise std on the realized reward
REVERT_TARGET: float = 50.0  # latent means revert toward this value
REWARD_MIN: int = 1
REWARD_MAX: int = 100


@dataclass
class BanditSchedule:
    """A single procedurally generated reward schedule.

    Rewards are pre-sampled for *every* arm on every trial (the full counterfactual
    payoff matrix); a rollout reveals only the chosen arm's entry, `rewards[t, a]`.
    """

    means: np.ndarray  # (n_trials, n_arms) latent reward means mu[t, i]
    rewards: np.ndarray  # (n_trials, n_arms) integer payoff matrix in [1, 100]
    seed: int
    params: dict = field(default_factory=dict)

    @property
    def n_trials(self) -> int:
        return self.means.shape[0]

    @property
    def n_arms(self) -> int:
        return self.means.shape[1]

    @property
    def optimal_arm(self) -> np.ndarray:
        """(n_trials,) index of the highest-paying arm each trial (by realized reward)."""
        return self.rewards.argmax(axis=1)

    def reward(self, trial: int, arm: int) -> int:
        """Reward delivered for choosing `arm` on `trial`."""
        return int(self.rewards[trial, arm])


def generate_schedule(
    seed: int,
    n_trials: int = N_TRIALS,
    n_arms: int = N_ARMS,
    lam: float = LAMBDA,
    sigma_d: float = SIGMA_D,
    sigma_o: float = SIGMA_O,
    revert_target: float = REVERT_TARGET,
    reward_min: int = REWARD_MIN,
    reward_max: int = REWARD_MAX,
) -> BanditSchedule:
    """Generate one restless-bandit reward schedule (Daw 2006 / NHB process)."""
    rng = np.random.default_rng(seed)

    means = np.empty((n_trials, n_arms), dtype=float)
    means[0] = rng.uniform(reward_min, reward_max, size=n_arms)
    for t in range(1, n_trials):
        drift_mean = lam * means[t - 1] + (1.0 - lam) * revert_target
        means[t] = rng.normal(drift_mean, sigma_d)

    rewards_cont = rng.normal(means, sigma_o)
    rewards = np.clip(np.round(rewards_cont), reward_min, reward_max).astype(int)

    return BanditSchedule(
        means=means,
        rewards=rewards,
        seed=seed,
        params={
            "n_trials": n_trials,
            "n_arms": n_arms,
            "lam": lam,
            "sigma_d": sigma_d,
            "sigma_o": sigma_o,
            "revert_target": revert_target,
            "reward_min": reward_min,
            "reward_max": reward_max,
        },
    )


def relative_reward(actions: np.ndarray, payoff_matrix: np.ndarray) -> np.ndarray:
    """Per-trial relative reward (NHB Fig 1f learning metric).

    `r_rel = (r_t - mean(p_t)) / (max(p_t) - mean(p_t))`, where `p_t` is the row of
    rewards available on trial `t`. 0 = random selection, 1 = always best (impossible
    to sustain), > 0 = systematically prefers above-average arms. Trials where all arms
    pay equally (zero denominator) contribute 0.

    Args:
        actions: (n_trials,) chosen arm index per trial.
        payoff_matrix: (n_trials, n_arms) realized rewards for every arm.
    """
    actions = np.asarray(actions)
    trials = np.arange(len(actions))
    obtained = payoff_matrix[trials, actions].astype(float)
    p_max = payoff_matrix.max(axis=1).astype(float)
    p_mean = payoff_matrix.mean(axis=1)
    denom = p_max - p_mean
    return np.where(denom > 0, (obtained - p_mean) / denom, 0.0)


# --- self-test / sanity check ----------------------------------------------


def _summarize(schedule: BanditSchedule) -> str:
    m, r = schedule.means, schedule.rewards
    return (
        f"seed={schedule.seed}: "
        f"mu in [{m.min():.1f}, {m.max():.1f}] (mean {m.mean():.1f}); "
        f"reward in [{r.min()}, {r.max()}]; "
        f"reward bounds ok: {r.min() >= REWARD_MIN and r.max() <= REWARD_MAX}"
    )


def _check_metric(seed: int = 0) -> str:
    """Random policy should score ~0; greedy-on-truth should score ~1."""
    sched = generate_schedule(seed)
    rng = np.random.default_rng(seed)
    random_actions = rng.integers(0, sched.n_arms, size=sched.n_trials)
    best_actions = sched.optimal_arm
    r_rand = relative_reward(random_actions, sched.rewards).mean()
    r_best = relative_reward(best_actions, sched.rewards).mean()
    return f"relative reward: random={r_rand:+.3f} (~0 expected), best={r_best:+.3f} (=1 expected)"


def _plot(schedules: list[BanditSchedule], out: Path) -> None:
    import matplotlib.pyplot as plt

    n = len(schedules)
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.4 * n), squeeze=False, sharex=True)
    for ax, sched in zip(axes[:, 0], schedules):
        for i in range(sched.n_arms):
            ax.plot(sched.means[:, i], lw=1.6, label=f"arm {i}")
            ax.scatter(
                np.arange(sched.n_trials), sched.rewards[:, i], s=6, alpha=0.25
            )
        ax.axhline(REVERT_TARGET, color="k", ls=":", lw=0.8)
        ax.set_ylim(0, 100)
        ax.set_ylabel("points")
        ax.set_title(f"schedule seed={sched.seed}", fontsize=9, loc="left")
    axes[-1, 0].set_xlabel("trial")
    axes[0, 0].legend(loc="upper right", ncol=4, fontsize=8)
    fig.suptitle("Restless 4-armed bandit reward schedules (cf. NHB Fig 1e)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"saved plot -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanity-check the restless-bandit reward-schedule generator.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n-schedules", type=int, default=3)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--seed", type=int, default=0, help="base seed; schedule i uses seed+i")
    parser.add_argument("--out", type=Path, default=None, help="path to save a plot (PNG)")
    args = parser.parse_args()

    schedules = [
        generate_schedule(args.seed + i, n_trials=args.n_trials)
        for i in range(args.n_schedules)
    ]
    for sched in schedules:
        print(_summarize(sched))
    print(_check_metric(args.seed))

    if args.out is not None:
        _plot(schedules, args.out)


if __name__ == "__main__":
    main()
