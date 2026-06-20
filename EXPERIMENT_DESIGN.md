# Experiment Design Plan
## Strengthening the Empirical Evaluation of the LLM Reward-Coaching Study

**Target venue:** IEEE Access · **Paper:** *When Do LLM Coaches Help? …* · **Status:** plan for experiments the authors must run on their own SUMO + NumPy-RL + Ollama codebase. No results below are fabricated; this document specifies *what to run* and *how to analyse it*.

---

## 0. Why these experiments (the decision they serve)

The current manuscript honestly reports that (i) adaptive RL clearly beats the fixed-time plan, but (ii) the LLM/RAG coaching conditions differ from plain RL by only ~1 %, the best overall cell uses **no** coaching, and at 5 seeds nothing is statistically established. A reviewer's three lethal questions follow directly:

1. **"Your baseline is a strawman."** → Add strong adaptive baselines (max-pressure, actuated).
2. **"Where is the evidence the LLM helps?"** → Add control coaches (random, rule-based) and a grid-search ceiling, then test significance.
3. **"Is 1 % real or noise?"** → Proper statistical protocol with enough seeds.

These experiments are designed so that *whatever the outcome*, the paper has a defensible, publishable story (see the decision tree in §8).

---

## 1. Experiment summary (priority-ordered)

| # | Experiment | Purpose | Reviewer impact | Effort |
|---|------------|---------|-----------------|--------|
| E1 | Max-pressure baseline | Replace strawman baseline | **Critical** | Low–Med |
| E2 | Actuated (gap-out) baseline | Standard adaptive reference | High | Low (SUMO built-in) |
| E3 | Random-in-safety-layer coach | Isolate LLM vs safety layer | **Critical** | Low (drop-in) |
| E4 | Rule-based coach | Isolate LLM vs simple heuristic | **Critical** | Low |
| E5 | Grid-search optimal weights (ceiling) | Explain the (likely) null result | High | Med (compute) |
| E6 | Increase seeds to ≥15–20 + significance tests | Make any claim valid | **Critical** | Low (compute) |
| E7 | Ablations: k, η, δ, temperature | Justify design choices | Med | Med |
| E8 | Cost/overhead measurement | Quantify the "cost" side | Med | Low |
| E9 | Webster re-timed fixed plan (optional) | Stronger static baseline | Low–Med | Low |

> **Minimum viable revision** = E1 + E3 + E4 + E6 (and ideally E5). These four answer the three lethal questions. Everything else strengthens but is secondary.

---

## 2. Part A — Strong baselines (raise the bar above fixed-time)

### E1. Max-pressure control *(the key adaptive baseline)*

Max-pressure is the standard, theoretically grounded, training-free adaptive controller; beating a static plan is unconvincing, but beating max-pressure is meaningful.

**Pressure definition.** For each candidate phase \(p \in P\) serving a set of movements \(M(p)\), each movement \(m\) going from incoming lane \(\ell^{\text{in}}_m\) to outgoing lane \(\ell^{\text{out}}_m\):

\[
\text{Pressure}(p) \;=\; \sum_{m \in M(p)} s_m \big( x(\ell^{\text{in}}_m) - x(\ell^{\text{out}}_m) \big)
\]

where \(x(\cdot)\) is the queue length (use `getLastStepHaltingNumber`) and \(s_m\) the saturation rate (set \(s_m=1\) if unknown). For an **isolated** intersection where downstream lanes drain freely, \(x(\ell^{\text{out}}_m)\approx 0\), so pressure reduces to the total queue served by the phase.

**Policy (pseudocode).**
```
G_min = 10 s         # minimum green
Y     = 4 s          # yellow (match Table: 4 s clearance)
dt    = 5 s          # decision interval
loop every simulation step:
    if time_in_current_green >= G_min and (step % dt == 0):
        p_star = argmax_p Pressure(p)            # over all phases
        if p_star != current_green_phase:
            switch: current_green -> Y (yellow) -> p_star (green)
            reset time_in_current_green
```
**TraCI notes.** Read per-lane queues with `traci.lane.getLastStepHaltingNumber(laneID)`; map lanes→movements→phases from the SUMO `.net.xml` connections (you already built the high-fidelity connection model). Drive phases with `traci.trafficlight.setPhase` or `setRedYellowGreenState`. Use the *same* yellow durations as the fixed plan for fairness.

### E2. Actuated control (gap-out / max-out)
SUMO has built-in actuated logic — cheapest strong baseline. In the `.add.xml` set `<tlLogic ... type="actuated">` with `minDur`, `maxDur`, and detector `gap`. Suggested: `minDur=10`, `maxDur=45`, `gap=2.0 s`. No code; just a config + run.

### E9. (Optional) Webster re-timed fixed plan
Compute a Webster-optimal cycle/splits **for the observed demand** (the 6 410-vehicle profile) and run it as a second static baseline. Shows the learned controller beats not just *a* fixed plan but a *well-tuned* one. Formula: \(C_0 = \frac{1.5L+5}{1-Y}\), green splits ∝ critical flow ratios.

---

## 3. Part B — Control coaches: does the LLM add anything? *(the crux)*

All three conditions below **share the exact safety layer** (clip to ±δ, project to simplex, EMA with η). The *only* thing that changes is the source of the raw weight proposal. This is what lets you attribute (or not) the effect to the language model.

### E3. Random-in-safety-layer coach *(decisive control)*
```
def random_coach(current_weights, delta_cap=0.10):
    noise = np.random.uniform(-delta_cap, +delta_cap, size=3)
    return current_weights + noise          # safety layer then clips/projects/EMAs
```
If **LLM ≈ Random** (no significant difference), the LLM's "reasoning" contributes nothing beyond stochastic perturbation inside a stabilising safety layer — a critical thing to know and disclose.

### E4. Rule-based coach
```
def rule_coach(summary, current_weights, step=0.05):
    # summary.dW, dQ, dCO2 = change vs previous episode (positive = worsened)
    worsened = {'alpha': summary.dW, 'beta': summary.dQ, 'gamma': summary.dCO2}
    target = max(worsened, key=worsened.get)   # objective that regressed most
    w = current_weights.copy(); w[target] += step
    return w                                   # safety layer then clips/projects/EMAs
```
A 4-line deterministic heuristic. If **LLM ≈ Rule**, an 8B model is unjustified versus trivial logic.

### E5. Grid-search optimal fixed weights *(performance ceiling + explanation)*
```
for algo A in {DQN, DDPG, PPO, A2C}:
    best = None
    for (a,b,g) in simplex_grid(step=0.1):     # ~66 weight triples summing to 1
        score = mean over 3 seeds of final_waiting(A, fixed_weights=(a,b,g))
        if score < best.score: best = (a,b,g,score)
    report best                                 # ceiling any reward-weight tuner could reach
```
**Why this is powerful:** if the default `(0.6, 0.3, 0.1)` (PURE) is already within the CI of the grid optimum, then **there is nothing for any coach to find**, and your null result is *explained*, not just observed. This single experiment can turn a weak negative into a strong, principled one.

---

## 4. Part C — Ablations on the proposed method (E7)

Run on the 1–2 algorithms where coaching is most active (e.g., A2C and PPO) to limit cost:

| Knob | Values to sweep | Question |
|------|-----------------|----------|
| Retrieval depth `k` | 0, 1, 3, 5, 10 | Does episodic memory help monotonically, or saturate/hurt? (`k=0` = memory-less) |
| EMA coefficient `η` | 0.5, 0.8, 1.0 | Is the *smoothing* doing the work? (`η=1.0` = ignore LLM entirely) |
| Safety cap `δ` | 0.05, 0.10, 0.20 | Sensitivity to update aggressiveness |
| LLM temperature | 0.0, 0.3, 0.7 | Determinism vs exploration of the coach |
| LLM size (if feasible) | 8B vs larger | Does a stronger model change the per-algorithm verdict? |

The `η=1.0` and `k=0` cells are especially diagnostic: they bracket "no LLM influence" and "no memory."

---

## 5. Part D — Statistical protocol (make any claim valid) (E6)

1. **Seeds.** Raise from 5 to **≥15 (target 20–30)**. Justify with a power calculation: with a paired test, the seeds needed to detect a difference \(\Delta\) at power \(1-\beta\) is roughly \(n \approx \dfrac{(z_{1-\alpha/2}+z_{1-\beta})^2\,\sigma_d^2}{\Delta^2}\), where \(\sigma_d\) is the SD of the *paired differences* (estimate from a small pilot). For ~1 % effects this typically lands well above 5.
2. **Per-cell reporting.** Mean ± SD **and** 95 % CI (bootstrap or t) for *every* metric in *every* cell.
3. **Tests.** Per algorithm, for each pre-registered contrast (PURE vs LLM; LLM vs RAG; best-coach vs best-baseline), use the **Wilcoxon signed-rank test** (paired on seed; robust for small, possibly non-normal samples). Report the median paired difference with bootstrap CI.
4. **Effect size.** Report **Cliff's δ** (or matched-pairs rank-biserial) alongside every p-value — distinguishes a real-but-tiny effect from a negligible one.
5. **Multiple comparisons.** Apply **Holm–Bonferroni** within each contrast family (4 algorithms).
6. **Pre-registration.** Fix the primary hypothesis and the exact comparison list *before* running, to avoid post-hoc cherry-picking. State the significance level (α = 0.05).
7. **Language rule (already in the paper).** Call a difference an "improvement" only if it is consistent in direction **and** significant after correction; otherwise report "not statistically distinguishable."

**Stats script skeleton (Python / SciPy):**
```python
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

# results[algo][cond] -> 1D array of per-seed final avg-waiting (len = n_seeds)
def paired_compare(a, b, n_boot=10000):
    a, b = np.asarray(a), np.asarray(b)
    stat, p = wilcoxon(a, b)                 # paired, two-sided
    d = a - b
    boots = [np.median(np.random.choice(d, len(d), replace=True)) for _ in range(n_boot)]
    ci = np.percentile(boots, [2.5, 97.5])
    cliffs = np.mean(np.sign(a[:,None] - b[None,:]))   # Cliff's delta
    return dict(p=p, median_diff=np.median(d), ci=tuple(ci), cliffs_delta=cliffs)

# collect p-values across the family, then correct:
pvals = [...]
reject, p_adj, _, _ = multipletests(pvals, alpha=0.05, method="holm")
```

---

## 6. Part E — Cost / overhead measurement (E8)

You claim practicality ("municipal deployment"); quantify the cost side.

- **Wall-clock split per episode:** time the RL update vs the LLM call separately (`time.perf_counter()` around the Ollama request).
- **Per training run:** total time and number of LLM calls (25 per run for LLM/RAG; 0 for PURE).
- **LLM latency:** mean and p95 of the generate call; note model build/quantization (e.g., `llama3.1:8b-instruct-q4_K_M`) for reproducibility.
- **Energy (if available):** GPU via `nvidia-smi --query-gpu=power.draw` integrated over time (or `pynvml`); CPU via RAPL/`powercap`.
- **Report** a small table: overhead of LLM/RAG vs PURE, so the cost–benefit trade-off is explicit.

---

## 7. Part F — Full experimental matrix & compute budget

**Non-learning baselines** (run once each, but average over the same seed set for the stochastic demand): Fixed-time, Webster (opt.), Actuated, **Max-pressure**.

**Learning conditions, per RL algorithm** {DQN, DDPG, PPO, A2C}: PURE, **+Random-coach**, **+Rule-coach**, +LLM (`k=0`), +LLM+RAG (`k=3`) → 5 conditions.

**Ceiling:** Grid-search optimal fixed weights (per algorithm).

Run-count estimate (training runs), with `S` seeds:
```
non-learning:        4 * S
learning:            4 algos * 5 conditions * S          = 20S
grid-search ceiling: 4 algos * 66 triples * 3 seeds      = 792   (one-off)
ablations (E7):      ~ (5+3+3+3) k/η/δ/temp * 2 algos * S
```
With `S = 20`: ≈ `4*20 + 20*20 + 792 + (14*2*20)` ≈ **1 832 runs** (+ ablations). Each run is short (25 episodes, 2 sim-hours); the LLM-in-loop runs dominate wall-clock. Budget the LLM conditions first and parallelise seeds.

---

## 8. Part G — Reporting artifacts to produce

1. **Table — full per-metric results:** waiting time, queue, speed, CO₂, each as mean ± SD (95 % CI), for *all* conditions × 4 algorithms. (Replaces the current single-metric Table.)
2. **Table — significance & effect size:** per contrast, median diff, CI, Wilcoxon p (Holm-adjusted), Cliff's δ.
3. **Table — baselines:** fixed-time / Webster / actuated / max-pressure vs best learned controller.
4. **Table — cost/overhead.**
5. **Figure — learning curves with CI bands** (you have this; add CIs).
6. **Figure — weight trajectories** \((\alpha,\beta,\gamma)\) over episodes per coach (LLM vs random vs rule) — shows *what* each coach does.
7. **Table/inset — grid-search ceiling** vs PURE per algorithm.

---

## 9. Part H — Decision tree: how outcomes reshape the paper

- **Outcome A — LLM (or RAG) significantly beats PURE *and* Random *and* Rule on ≥1 algorithm (post-correction).**
  → Genuine positive contribution. Promote it; the title can move back toward the method; report precisely where and why it helps, with the cost table justifying the overhead.

- **Outcome B — LLM ≈ Random ≈ Rule, and all ≈ PURE; grid-search shows `(0.6,0.3,0.1)` already near-optimal.**
  → The LLM adds nothing beyond its safety layer *because the weight problem is already saturated*. This is a **strong, publishable cautionary result**: the contribution is the auditable framework + a rigorous "when LLM reward-coaching does **not** help, and a principled explanation." Keep the current honest framing; add the grid-search evidence as the clincher.

- **Outcome C — coaching significantly *worse* than PURE.**
  → Report as a cautionary finding; emphasise that the safety layer bounds the harm, but recommend against coaching in this regime.

- **Cross-cutting — learned controllers do *not* beat max-pressure.**
  → Important on its own. Reframe the contribution around the real-intersection benchmark + the analysis, not state-of-the-art performance, and discuss why (short training horizon, compact NumPy agents, single intersection).

In **every** branch the paper has a clear, honest claim — which is exactly what survives review.

---

## 10. Part I — Implementation checklist

- [ ] **E1** Implement max-pressure controller (pressure from TraCI queues; min-green + yellow respected).
- [ ] **E2** Add actuated `tlLogic` config and run.
- [ ] **E3** Add `random_coach` (drop-in for the LLM call; same safety layer).
- [ ] **E4** Add `rule_coach` (drop-in; same safety layer).
- [ ] **E5** Grid-search over the weight simplex (per algorithm, 3 seeds).
- [ ] **E6** Re-run all conditions with **≥15–20 seeds**, fixed and logged.
- [ ] **E6** Stats script: Wilcoxon + bootstrap CI + Cliff's δ + Holm.
- [ ] **E7** Sweep `k`, `η`, `δ`, temperature on A2C/PPO.
- [ ] **E8** Instrument wall-clock / latency / energy; build cost table.
- [ ] Regenerate the 7 artifacts in §8 and fold into the paper (replacing the two `\todo{}` placeholders).
- [ ] Pin and report the exact Ollama model build for reproducibility.

---

## 11. Part J — Suggested order (sprints)

1. **Sprint 1 (answers the lethal questions):** E1 max-pressure, E3 random coach, E4 rule coach, E6 seeds+stats. → You can now make *valid* statements and have a real baseline.
2. **Sprint 2 (explains the result):** E5 grid-search ceiling, E2 actuated. → Turns a null into a principled finding.
3. **Sprint 3 (depth & cost):** E7 ablations, E8 cost/overhead, E9 Webster, weight-trajectory figure.

After Sprint 1 the paper is already defensible; after Sprint 2 it is genuinely solid for IEEE Access.

---

*Prepared as a planning document. All numbers to be produced by the authors from real runs; do not report any value here as a result.*
