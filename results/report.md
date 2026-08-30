# Continual Skill Learning via Skill Cloning: Minimal Prototype — Results

*Implements Phases 1–4 of [issue #23](https://github.com/kobros-tech/6.86x/issues/23). The authoritative current experiment is the fixed-target prerequisite matrix in PR #4; the earlier source→target curriculum is retained below only as historical context.*

> **Historical-controller note (PR #1/#2):** The original source→target run used a compatibility controller that has since been fixed. Independent review found that `τ_solve=0.90` alone could let the controller call a merely-related-but-unsolved task "reuse" (zero training). PR #2 fixed this by requiring an independently measured solve check. The old powers result (clone-and-adapt held-out MSE ≈ 3.71) therefore **must not be treated as a current result**. Against the fixed controller, the original curriculum's powers clone-and-adapt held-out MSE was **0.206 ± 0.076**, versus **0.227** for scratch. The historical narrative that cloning was fast but generalized worse was an artifact of the false-reuse bug and is preserved only in the historical section below.

## 1. What was built

A dependency-free (numpy only) implementation of a controlled continual skill-acquisition mechanism:

| Component | File | What it does |
| --- | --- | --- |
| Skill representation | `skill.py` | `TinyMLP` (2→32→1, tanh hidden layer) trained by full-batch Adam; `clone()` deep-copies `W_j → W_new^(0)` per Section 5 |
| Task family | `tasks.py` | Small-integer arithmetic regression tasks used by the experiments |
| Compatibility score `P(T \| s_i)` | `compatibility.py` | `exp(-MSE_i(T)/60)` evaluated on a 64-example probe batch, with no gradient step |
| Decision rule | `compatibility.py` | reuse / clone / scratch against thresholds `τ_solve=0.90`, `τ_clone=0.15` |
| Three training strategies | `strategies.py` | **shared** (Baseline A: one network, no protection), **scratch** (Baseline B: independent network per task), **clone-and-adapt** (proposed mechanism) |
| Experiment drivers | `experiment.py`, `experiments/relatedness_pairs.py` | Historical source→target comparison and current fixed-target prerequisite matrix |
| Statistics | `analysis.py` | Paired comparisons matched by seed |

## 2. Design choices

- **Compatibility score $P(T\mid s_i)$:** exponentially squashed MSE of the *frozen* skill on a probe batch from the new task. It is cheap (no training) and asks whether the existing skill already solves the target.
- **Thresholds:** `τ_solve = 0.90` and `τ_clone = 0.15`, chosen once rather than tuned separately for each target.
- **Stopping rule:** target training stops when the skill reaches 85% accuracy (±0.5 tolerance) on its training batch, capped at 1500 epochs. In the fixed-target experiment, reported mean acquisition steps include **all seeds**; an unsuccessful run contributes the full **1500-epoch budget**. Therefore these means are budgeted acquisition cost, not mean convergence time among successful runs.
- **Task representation:** raw `(a,b)` pairs, with integer inputs scaled by `/10` for the network; targets remain unscaled.

## 3. Current authoritative experiment: fixed-target prerequisite matrix

PR #4 changes the central question from ranking isolated source→target pairs to testing the same target under increasing amounts of previously acquired knowledge. Each condition uses 15 matched seeds and allows the controller to choose reuse, clone-and-adapt, or scratch.

For non-empty histories, each requested prerequisite must be successfully acquired before the history is considered available to the target controller. Failed prerequisite acquisitions are recorded in the raw results, but unsuccessful skills are not exposed to the controller as acquired skills.

### 3.1 Results

| Target | Prior history | Success | Mean budgeted acquisition steps | Held-out MSE |
| --- | --- | ---: | ---: | ---: |
| Subtraction | none | 15/15 | 33.5 | 0.153 |
| Subtraction | addition | 15/15 | 62.3 | 0.141 |
| Subtraction | addition + multiplication | 15/15 | 73.2 | 0.144 |
| Division | none | 15/15 | 515.2 | 0.244 |
| Division | addition | 15/15 | 616.2 | 0.228 |
| Division | addition + multiplication | 15/15 | 617.5 | 0.231 |
| Squares | none | 3/15 | 1486.7 | 0.207 |
| Squares | addition | 3/15 | 1490.0 | 0.208 |
| Squares | addition + multiplication | 2/15 | 1499.2 | 0.206 |
| Powers | none | 15/15 | 471.6 | 0.249 |
| Powers | addition | 15/15 | 355.2 | 0.193 |
| Powers | addition + multiplication | 15/15 | 237.3 | 0.169 |

### 3.2 Interpretation

The results do **not** support a universal rule that more prior skills always improve acquisition. Under this protocol, prior knowledge was associated with slower budgeted acquisition for subtraction and division, had little effect on the already difficult squares target, and substantially reduced acquisition cost for powers. The powers result is especially clear: mean budgeted acquisition steps decreased from **471.6** with no prior history to **355.2** after addition and **237.3** after addition + multiplication, while held-out MSE also decreased from **0.249** to **0.193** and **0.169**, respectively.

These findings support a narrower claim: **the usefulness of prior knowledge is target- and history-dependent.** They do not establish that addition or multiplication are formal mathematical prerequisites, nor do they imply that a particular skill ordering is universally optimal.

### 3.3 Squares: interpret the budget carefully

Squares is near the training-budget limit in all three histories. Only 3/15 runs succeeded with no prior history, 3/15 after addition, and 2/15 after addition + multiplication. The mean values near 1500 should therefore **not** be described as convergence times. They are means of the budgeted acquisition steps with unsuccessful runs assigned 1500. The held-out MSE is similar across conditions, so the current experiment does not show a reliable acquisition benefit from the tested prior histories for squares.

## 4. Compatibility score notation and meaning

The controller's compatibility quantity is

\[
P(T\mid s_i) = \exp\left(-\frac{\operatorname{MSE}(s_i,T)}{60}\right).
\]

It is a frozen-probe heuristic: it measures how well skill `s_i` already predicts examples from target task `T`. A high value can justify reuse when the independent solve gate also passes, while an intermediate value can make cloning eligible. The score should not be interpreted as a probability calibrated in the statistical sense, nor as a guarantee that cloning will accelerate learning.

## 5. Historical source→target experiment (superseded controller)

The following observations are retained for reproducibility and provenance, but **are not authoritative current results**.

The original curriculum compared addition, subtraction, multiplication, and powers as isolated source→target events. In that pre-fix run, powers was incorrectly allowed to trigger reuse on 9 of 15 seeds because the controller treated a merely related frozen skill as solved. This produced the old held-out MSE ≈ 3.71 result. PR #2 corrected the reuse gate using independent target-solve accuracy. The corrected rerun gave powers clone-and-adapt held-out MSE **0.206 ± 0.076**, versus scratch **0.227**, so the old claim that cloning itself caused worse generalization is **superseded**.

The historical experiment did contain useful evidence that transfer can be heterogeneous, but its pre-fix powers result must not be mixed with the current fixed-target matrix.

## 6. Honest limitations

- The fixed-target experiment uses one arithmetic task family, one architecture size, one probe-batch design, and fixed controller thresholds.
- Fifteen matched seeds provide a controlled comparison but do not establish universal behavior.
- The stopping rule is based on training accuracy and can make early stopping an imperfect proxy for final solution quality; the budgeted-step metric also assigns 1500 steps to unsuccessful runs.
- Squares has very low success under the current 1500-epoch budget, so its acquisition-cost comparisons are weak.
- The prerequisite matrix tests usefulness of previously learned representations; it does **not** prove formal prerequisite relationships.
- No composition of multiple skills into a new computation was tested.
- The current arithmetic domain does not establish robustness to a broader or signed input domain; that is a natural follow-up experiment.

## 7. Next step

The clean follow-up is to extend the same fixed-target protocol to a broader signed domain containing negative values, while keeping the architecture, controller, seeds, stopping rule, and analysis fixed. The signed-domain experiment should be treated as a separate robustness/domain-sensitivity study so that it does not silently change the interpretation of the current results.
