# Value Steering — Project Summary

A standalone overview: the goal, the logic of each analysis, the results, the conclusions
they license, and what remains open.

---

## 1. Goal

Test whether **manifold steering** (Goodfire 2026) extends from a model's *output concept*
(their setup — weekdays, months, letters: the concept *is* the answer token) to a **latent
intermediate variable**: the subjective **value** a model assigns while deciding.

Concretely, Llama-3.1-8B-Instruct plays a 4-armed restless bandit. We fit a cognitive model
(BestRL) to its choices to recover the per-arm value **Q** it appears to track. We then ask:

- **(a) Geometry.** Does Llama's residual stream encode a value-like latent whose structure
  matches its behavior?
- **(b) Control.** Can we *move its choices* by intervening on that representation — and does
  *respecting the representation's geometry* (manifold vs linear steering) matter for doing
  so coherently?

This targets the paper's own #1 stated open problem: steering intermediate *belief* variables
rather than outputs. The aim is to **actually move the model**, not to produce a null.

---

## 2. Logic of the analysis

The investigation is a chain, each step motivated by the previous result.

1. **Substrate.** Before steering, confirm value is *there*: the model learns the task, a
   cognitive model recovers a value latent Q, and Q is decodable from activations.
2. **First steering attempt** along a Q_max manifold (bin centroids). It failed — so the
   logic turned diagnostic.
3. **Diagnosis.** Identify *why* it failed by isolating the candidate causes (layer, subspace,
   hook, target construction) one at a time.
4. **Channel test (interchange).** Once the diagnosis pointed at *target construction*,
   test the channel directly with an identity-carrying patch.
5. **Lever test (value vs recency).** A working channel could be transmitting value *or*
   recency; a factorial design dissociates them.
6. **Corrected manifold experiment (V2).** With the channel and the lever established, rebuild
   the *manifold-vs-linear* comparison on the correct concept ("which arm is best" × value),
   with the geometry checked *before* steering (curvature gate) so the comparison is meaningful.

The guiding discipline throughout: separate **observation** from **inference**, and test the
**premise** of each comparison before running it.

---

## 3. Results

### 3.1 Substrate exists
- Llama learns the bandit: mean relative reward **+0.37** over chance (0 = random, 1 = optimal).
- **BestRL** is the best-fitting cognitive model (BIC-best; recovers per-trial Q). Strong
  perseveration, modest but real value term (β=0.016).
- **Q_max is decodable** from the residual stream: ridge **R² ≈ 0.84–0.91**, peaking at layer 19;
  best-arm-letter accuracy 95% vs 80% for pure recency.

### 3.2 Naïve steering failed — and the cause is target construction
Bin-centroid Q_max steering gave **flat behavior + level collapse**. Root cause: with letters
randomized per episode, each Q_max bin averaged over trials where *different arms were best*, so
the steering vector **marginalized away "which arm is best"** — the one thing the choice circuit
needs. Not a layer/subspace/hook problem.

### 3.3 The steering channel works (interchange test)
Injecting a real trial's activation where arm *a* is clearly best, into a same-episode base
where *a* is a loser, moved:

| | baseline | after patch | source ceiling |
|---|---|---|---|
| P(target arm) | 0.325 | **0.702** | 0.731 |

≈93% of the natural ceiling recovered, **arm-specifically** (the other 3 arms each lost ~0.13),
94% directional success. This *also cleared every prior suspect*: identity patch = exact no-op
(hook valid), subspace ≈ full (signal lives in PCA-64), **L19 ≈ L28** (layer was fine).

### 3.4 The lever is value, not recency (decoupling test)
A 2×2 factorial dissociating value from recency (which BestRL bundles):

| source | isolates | ΔP(target) |
|---|---|---|
| high value, **not** recent | **value w/o recency** | **+0.031** (~3σ) |
| recent, **low** value | recency w/o value | −0.001 (≈0) |

Regression (n=420): **β_value = +0.077 (t=+14.3)**, β_recency = −0.074 (t=−13.7). A valuable
arm steers even when not recently pressed; a recently-pressed-but-low-value arm does not.
**Steering Llama's choice is a *value* operation.**

### 3.5 Corrected manifold experiment (V2, fixed letters B,C,D,F)
Concept = "which arm is best" (4-way) × relative value **r = Q[j] − Q[i]** (intrinsic coord).

- **Curvature gate PASSED**: M_h strongly curved (L19 mean arc/chord 1.95; L28 2.14) → the
  manifold-vs-linear comparison is meaningful.
- **Structural correspondence (qualitative)**: the 4-arm concept **simplex** appears in both
  behavior space (clean tetrahedron) and activation space.
- **Isometry does NOT favor manifold**: within-edge Pearson **r ≈ 0.975 ≈ 0.975** (manifold ≈
  linear). Reason: the value edges are curved but **non-folding** (open monotone arcs), so
  distance cannot separate the two. The paper's isometry win is specific to **folding** (cyclic)
  geometry. *Honest finding, not a bug.*
- **Steering (L19 & L28) — depends on the metric, and the right metric favors manifold:**

  | metric | L19 man | L19 lin | L28 man | L28 lin |
  |--------|--------:|--------:|--------:|--------:|
  | E_BC (nearest-pt to M_y; pacing-blind) | 9.77 | **9.18** | 10.51 | **9.90** |
  | **full MSE ×1e3 (matched-fraction tracking)** | **3.63** | 5.69 | **1.06** | 1.23 |
  | off-axis leakage | 0.153 | 0.142 | 0.102 | 0.098 |

  **Manifold steering reproduces the natural transition better** — trajectory tracking, full
  MSE 3.6 vs 5.7 at L19, manifold lower in all 6 pairs, paired t=−4.8, p<1e-4 (directional but
  n.s. at L28, where both track ~3× better and converge). The **nearest-point E_BC marginally
  favors linear** (t≈+8) but is *pacing-blind*: it asks "is the output on M_y?" not "does it
  traverse M_y in order?". Linear does NOT teleport off the manifold here (non-folding geometry)
  — it traverses the manifold at the wrong *pace*, which only the tracking metric catches.
- **Smoothing the manifold (s=250) is behaviorally inert**: it removes the visible spline
  wobble (sampling noise, ~3-5% of manifold extent) but moves every steering metric by <1%.
  The wobble — and indeed most of M_h's fine geometry — lives in behaviorally inert PCA
  directions. *Current data suffices; no larger collection needed for the steering result.*

---

## 4. Conclusions

1. **We can causally steer Llama's choices via its value representation.** This is the project's
   core payoff, established independently of the manifold question (§3.3–3.4): a value-carrying
   intervention moves the choice predictably (0.33 → 0.70), specifically, and through the *value*
   pathway (not recency).

2. **Manifold steering has a real but DIFFERENT advantage here than in the paper** (§3.5).
   It reproduces the natural transition with better *trajectory fidelity* (matched-fraction
   MSE, significant at L19) — because interpolating in the intrinsic value coordinate gives
   the right *pacing*. It does NOT win on the paper's *nearest-point* naturalness, because
   that metric is pacing-blind and our manifold doesn't fold. (An earlier headline that
   "manifold doesn't beat linear" was an artifact of using only the nearest-point metric.)

3. **Why the divergence from the paper, precisely.** Linear steering is pathological when
   off-manifold regions are *behaviorally* unnatural — which happens when the concept
   manifold **folds** (cyclic: a straight chord cuts across the loop into wrong-answer
   territory) and shows up as large nearest-point distance. A value continuum does NOT fold,
   so linear stays near the curve-as-a-set (E_BC ≈ tied) but traverses it at the wrong pace
   (tracking MSE worse). The failure mode shifts from "teleport off-manifold" to "wrong-pace
   traversal." Most of M_h's fine geometry (wobble, curvature-vs-chord) is behaviorally inert
   here — smoothing it changes the picture but not the behavior.

4. **Net scientific contribution.** (i) A positive, confound-controlled demonstration that an
   LLM's *latent value* variable is causally steerable — the paper's open problem. (ii) A
   refined account of *when and how* manifold steering helps: for a non-folding latent, the
   benefit is in trajectory *pacing/fidelity*, not in avoiding off-manifold teleportation; and
   the right way to measure it is order-aware (tracking), not pacing-blind (nearest-point).
   Neither is a dressed-up null.

---

## 5. Status / open

The experiment is complete; conclusions are stable. Done: L19 + L28 steering, both
naturalness metrics, and the s=250 smoothing test (all settle the points in §3.5–§4).

**Optional polish (not required):**
- Bias-balanced D→F showcase + a tight 2-panel "activation-paths-differ / behavior-identical"
  figure to make the behaviorally-inert point crisp.
- L28 steering figure (only L19 rendered).

**Caveat (flagged, not a confound):** fixed letters induce a first-listed-token bias
(first-trial P(B)=0.55); the within-pair manifold-vs-linear comparison is unaffected.

**Deferred strengtheners (future work):**
- **Weekdays positive control** in our pipeline — would confirm our machinery *can* reproduce a
  manifold advantage when the concept genuinely folds, making the bandit scope result airtight.
- **Synthetic-history value/recency decoupling** — construct prompts that set value and recency
  independently by design (vs the natural-trial dissociation used in §3.4).
- **V3 factored 2-D control** — steer "which arm" and "confidence" as independent coordinates.
