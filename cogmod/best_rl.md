# BestRL

A 6-parameter Q-learner with three cognitive extensions: **forgetting toward a learned
baseline**, **persistent perseveration**, and **transient perseveration**. Ported from
Eckstein et al. (2026)'s `CogMod` (`hybrid-rnn/code/cogmod.py`); `BestRL` = `CogMod` with
all flags on.

Implementation: [`cogmod/best_rl.py`](best_rl.py); ABC: [`cogmod/base.py`](base.py).

---

## State

At the start of trial $t$ the model carries

$$
\text{state}_t \;=\; \big(\mathbf{Q}_t,\; \mathbf{a}_{t-1}\big)
$$

- $\mathbf{Q}_t \in \mathbb{R}^{n_{\text{arms}}}$ — per-arm value estimates,
- $\mathbf{a}_{t-1} \in \{0,1\}^{n_{\text{arms}}}$ — one-hot of the previous trial's action
  (zeros at $t=0$).

---

## Per-trial dynamics

### 1.  Choose — `policy(state) → P(a_t)`

Add the **transient perseveration** bonus to the just-chosen arm, then softmax:

$$
\tilde{\mathbf{Q}}_t \;=\; \mathbf{Q}_t + \kappa_t\,\mathbf{a}_{t-1},
\qquad
P(a_t = i) \;=\; \frac{\exp\!\big(\beta\,\tilde{Q}_{t,i}\big)}{\sum_j \exp\!\big(\beta\,\tilde{Q}_{t,j}\big)}.
$$

$\kappa_t$ is **not stored** — it only biases the very next choice.

### 2.  Observe — sample $a_t \sim P(\cdot)$, environment returns reward $r_t$.

### 3.  Update — `update(state, a_t, r_t) → state_{t+1}`

Three sub-steps applied in order:

**(a) Delta rule on the chosen arm only.** Reward prediction error $\delta_t = r_t - Q_t^{(a_t)}$; update is masked to the action that was actually taken:

$$
\mathbf{Q}'_t \;=\; \mathbf{Q}_t + \alpha\,\delta_t\,\mathbf{a}_t.
$$

**(b) Forget toward init** (applied to *all* arms — chosen *and* unchosen):

$$
\mathbf{Q}''_t \;=\; (1-f)\,\mathbf{Q}'_t + f\,Q_{\text{init}}.
$$

**(c) Persistent perseveration on the chosen arm:**

$$
\mathbf{Q}_{t+1} \;=\; \mathbf{Q}''_t + \kappa_p\,\mathbf{a}_t.
$$

New state: $\big(\mathbf{Q}_{t+1},\;\mathbf{a}_t\big)$.

Note $\kappa_p$ goes **into stored value**, so repeated choices build up a stickiness
bonus that itself **decays** via the forgetting step on the next trial.

---

## Parameters (6)

| Symbol | Range | What it controls | Reparameterization |
|---|---|---|---|
| $\alpha$ | $(0,1)$ | **Learning rate.** How fast Q tracks new rewards. High $\alpha$ → recency; low → uses long history. | $\alpha=\sigma(\text{raw})$ |
| $\beta$ | $(0,\infty)$ | **Inverse temperature.** How sharply the model picks the higher-value arm. High $\beta$ → near-greedy; low → near-uniform. | $\beta=\mathrm{softplus}(\text{raw})$ |
| $f$ | $(0,1)$ | **Forgetting.** Per-step decay of all Q toward $Q_{\text{init}}$. High $f$ → short memory; low → values persist. | $f=\sigma(\text{raw})$ |
| $\kappa_p$ | $(-1,1)$ | **Persistent perseveration.** Stickiness baked into stored value. $+$ = repeat-bias; $-$ = switch-bias. Decays via forgetting. | $\kappa_p=\tanh(\text{raw})$ |
| $\kappa_t$ | $(-1,1)$ | **Transient perseveration.** One-trial repeat bias on the very next choice. Not stored. | $\kappa_t=\tanh(\text{raw})$ |
| $Q_{\text{init}}$ | $\mathbb{R}$ | **Initial value.** Starting Q for every arm and the asymptote of forgetting. | unconstrained scalar |

Bounded params are fit on their unconstrained "raw" representation, so L-BFGS sees a
smooth unconstrained landscape (matches `hybrid-rnn`'s convention).

---

## Initialization (trial $t = 0$)

- $\mathbf{Q}_0 = (Q_{\text{init}},\,\dots,\,Q_{\text{init}})$ — every arm starts at $Q_{\text{init}}$.
- $\mathbf{a}_{-1} = \mathbf{0}$ — no previous action, so $\kappa_t$ contributes nothing.
- $P(a_0)$ is therefore uniform (softmax of a constant vector). No information has arrived yet.

---

## Information flow (one trial, visual)

```
   state in:  (Q_t,  a_{t-1})
                  │
                  ▼
       ┌────────────────────────────────────────┐
       │  policy:                               │
       │    Q̃ = Q_t + κ_t · a_{t-1}             │   ← transient persev
       │    P = softmax(β · Q̃)                  │
       └────────────────────────────────────────┘
                  │
                  ▼   sample
              a_t ~ P
                  │
                  ▼   observe
              r_t  (environment)
                  │
                  ▼
       ┌────────────────────────────────────────┐
       │  update:                               │
       │    δ_t = r_t − Q_t[a_t]                │   ← RPE
       │    Q'_t  = Q_t + α · δ_t · a_t         │   ← delta rule (chosen only)
       │    Q''_t = (1 − f) · Q'_t  + f · Q_init │   ← forget (all arms)
       │    Q_{t+1} = Q''_t + κ_p · a_t          │   ← persistent persev
       └────────────────────────────────────────┘
                  │
                  ▼
   state out:  (Q_{t+1},  a_t)
```

---

## What's recovered (Llama-3.1-8B, Step 3 fit)

| $\alpha$ | $\beta$ | $f$ | $\kappa_p$ | $\kappa_t$ | $Q_{\text{init}}$ |
|---:|---:|---:|---:|---:|---:|
| 0.16 | 0.016 | 0.04 | **1.0** (boundary) | **1.0** (boundary) | −222 |

Both perseveration scalars are pegged at the $\tanh$ boundary and $\beta$ is very small —
Llama's choices on this task are **dominated by perseveration**, with value-driven choice
contributing only modestly on top. See [`../dev/context/value-steering/status.md`](../dev/context/value-steering/status.md)
for the full interpretation.

---

## Relation to `SimpleRL`

`SimpleRL` is `BestRL` with $f=\kappa_p=\kappa_t=0$ and $Q_{\text{init}}$ fixed at the
reward-range midpoint (50). That is, **drop steps (b) and (c) of the update, and drop the
transient term in step (1)**. Two free parameters remain: $\alpha$ and $\beta$.
