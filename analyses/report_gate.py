"""Viability gate for the value-report experiment.

Premise of the whole report-experiment arc (see dev/context/value-steering/report_experiment.md):
the model must produce a *graded* value report that tracks the latent value. This script
tests that premise cheaply, reusing the existing fixed-letter histories and BestRL Q — no
new rollouts.

For a sample of (episode, trial t) contexts we re-prompt the model with a value-report
question for *each* of the 4 arms, given the history of trials 0..t-1, and parse the
reported number. We then ask two questions, matching the two regimes that matter:

  (1) Best arm (argmax Q):   does the report track Q_max across weak -> dominant?
  (2) Non-best arms:         is the report graded with Q[a], or flat / floored?
                             (rank 1 = runner-up / near-crossover; rank 3 = clear loser.)

A point-estimate (greedy-decoded number) suffices for the gate; the full reported-value
*distribution* (the behavior manifold M_y^report) comes later only if this passes.

Verdict heuristic (printed): the gate PASSES if reports parse reliably, discriminate best
from worst arms, and correlate with Q for the best arm. The key headline signal is whether
non-best (especially rank-1 / rising) arms are reported gradedly.

    .venv/bin/python -m analyses.report_gate \
        --dataset artifacts/datasets/llama8b_fixed.json \
        --qfile artifacts/value/bestrl_Q_fixed.pt \
        --n-contexts 60 --out artifacts/value/report_gate.pt
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import time

import numpy as np
import torch

from llm_data.llm import load_model
from llm_data.prompt import LabelMap, TRIAL_TEMPLATE

REPORT_SYSTEM: str = (
    "You are playing a game with {n} slot machines labelled {labels}. "
    "Each pays a varying number of points; the payouts drift slowly over time, "
    "so the best machine can change. Based on the outcomes you have seen, "
    "estimate how many points a machine will pay next."
)
REPORT_USER_SUFFIX: str = (
    "\nBased on what you have seen so far, how many points do you expect "
    "machine {letter} to pay if you press it next? Answer with a single number."
)
REPORT_PREFILL: str = "I expect machine {letter} to pay about "


def build_report_prompt(
    tokenizer, labels: LabelMap, history: list[tuple[int, int]], letter: str
) -> str:
    """Chat-format prompt eliciting the expected value of `letter`, prefilled for a number."""
    sys_msg = REPORT_SYSTEM.format(n=labels.n_arms, labels=", ".join(labels.letters))
    user_msg = (
        "".join(
            TRIAL_TEMPLATE.format(letter=labels.letters[arm], reward=int(reward))
            for arm, reward in history
        )
        + REPORT_USER_SUFFIX.format(letter=letter)
    )
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return text + REPORT_PREFILL.format(letter=letter)


@torch.no_grad()
def report_value(model, tokenizer, prompt: str, max_new_tokens: int = 4) -> float:
    """Greedy-decode a few tokens after the prefill and parse the first integer; nan if none."""
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **enc, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new = tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
    m = re.search(r"\d+", new)
    return float(m.group()) if m else float("nan")


def _last_reward_for_arm(history: list[tuple[int, int]], arm: int) -> float:
    for a, r in reversed(history):
        if a == arm:
            return float(r)
    return float("nan")


def _corr(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Pearson r over rows where both are finite; returns (r, n)."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return float("nan"), int(m.sum())
    return float(np.corrcoef(x[m], y[m])[0, 1]), int(m.sum())


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--dataset", type=pathlib.Path,
                   default=pathlib.Path("artifacts/datasets/llama8b_fixed.json"))
    p.add_argument("--qfile", type=pathlib.Path,
                   default=pathlib.Path("artifacts/value/bestrl_Q_fixed.pt"))
    p.add_argument("--n-contexts", type=int, default=60,
                   help="(episode, trial) contexts; each queries all 4 arms")
    p.add_argument("--t-min", type=int, default=15)
    p.add_argument("--t-max", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=pathlib.Path,
                   default=pathlib.Path("artifacts/value/report_gate.pt"))
    args = p.parse_args()

    ds = json.loads(args.dataset.read_text())
    qd = torch.load(args.qfile, map_location="cpu")
    Q = qd["Q"]  # [n_ep, T, n_arms]
    n_ep, T, n_arms = Q.shape
    rng = np.random.default_rng(args.seed)

    # Sample (episode, trial) contexts, then query every arm in each.
    eps = rng.integers(0, min(n_ep, len(ds)), size=args.n_contexts)
    ts = rng.integers(args.t_min, min(args.t_max, T), size=args.n_contexts)

    model, tok = load_model(args.model)
    print(f"loaded {args.model} on {model.device}")

    records: list[dict] = []
    t0 = time.time()
    for i, (e, t) in enumerate(zip(eps.tolist(), ts.tolist())):
        epi = ds[e]
        labels = LabelMap(letters=list(epi["letters"]))
        history = list(zip(epi["actions"][:t], epi["rewards"][:t]))
        q = Q[e, t].numpy()                       # value going into trial t, per arm
        ranks = (-q).argsort().argsort()          # 0 = best (highest Q)
        payoff = epi["payoff_matrix"][t]          # true per-arm payoff at trial t
        for arm in range(n_arms):
            letter = labels.letters[arm]
            prompt = build_report_prompt(tok, labels, history, letter)
            v = report_value(model, tok, prompt)
            records.append(dict(
                episode=int(e), trial=int(t), arm=int(arm), letter=letter,
                v_report=v, q=float(q[arm]), q_rank=int(ranks[arm]),
                is_best=bool(ranks[arm] == 0), true_payoff=float(payoff[arm]),
                last_reward=_last_reward_for_arm(history, arm),
            ))
        if (i + 1) % 10 == 0:
            el = (time.time() - t0) / 60
            print(f"[{i+1}/{args.n_contexts}] elapsed {el:.1f} min, "
                  f"eta {el/(i+1)*(args.n_contexts-i-1):.1f} min")

    # ---- analysis -----------------------------------------------------------
    vr = np.array([r["v_report"] for r in records])
    q = np.array([r["q"] for r in records])
    rank = np.array([r["q_rank"] for r in records])
    payoff = np.array([r["true_payoff"] for r in records])
    last = np.array([r["last_reward"] for r in records])

    n = len(records)
    parsed = np.isfinite(vr)
    best = rank == 0
    nonbest = rank != 0

    def line(name, mask):
        sub = vr[mask & parsed]
        return (f"{name:<16} n={mask.sum():3d}  parsed={ (mask & parsed).sum():3d}  "
                f"mean={np.nanmean(vr[mask]):6.1f}  std={np.nanstd(vr[mask]):5.1f}")

    print("\n=== REPORT-GATE RESULTS ===")
    print(f"parse rate: {parsed.mean():.2f}  ({parsed.sum()}/{n})")
    print("\nReported value by Q-rank (0=best .. 3=worst):")
    for rk in range(n_arms):
        print("  " + line(f"rank {rk}", rank == rk))

    print("\nCorrelations (Pearson r):")
    for nm, x in [("Q", q), ("true_payoff", payoff), ("last_reward", last)]:
        r_all, n_all = _corr(vr, x)
        r_b, n_b = _corr(vr[best], x[best])
        r_nb, n_nb = _corr(vr[nonbest], x[nonbest])
        print(f"  v_report vs {nm:<12} all r={r_all:+.2f} (n={n_all})   "
              f"best r={r_b:+.2f} (n={n_b})   non-best r={r_nb:+.2f} (n={n_nb})")

    # discrimination: best vs worst reported value
    mb = np.nanmean(vr[best]); mw = np.nanmean(vr[rank == n_arms - 1])
    print(f"\nbest-vs-worst reported mean: {mb:.1f} vs {mw:.1f}  (gap {mb-mw:+.1f})")

    r_qb, _ = _corr(vr[best], q[best])
    r_qnb, _ = _corr(vr[nonbest], q[nonbest])
    verdict = (
        parsed.mean() > 0.8 and (mb - mw) > 5 and np.isfinite(r_qb) and r_qb > 0.3
    )
    print(f"\nVERDICT: {'PASS' if verdict else 'CHECK'} "
          f"(parse>{0.8}, best-vs-worst gap>5, best-arm r(Q)>0.3)")
    print(f"  headline signal — non-best graded? r(v_report,Q | non-best) = {r_qnb:+.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"records": records, "model": args.model, "args": vars(args)}, args.out)
    summary = {
        "n": n, "parse_rate": float(parsed.mean()),
        "best_vs_worst_gap": float(mb - mw),
        "r_q_best": float(r_qb), "r_q_nonbest": float(r_qnb),
        "verdict_pass": bool(verdict),
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved -> {args.out}  (+ .json summary)")


if __name__ == "__main__":
    main()
