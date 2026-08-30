# Continual Skill Learning via Skill Cloning: Minimal Prototype — Results

*Implements Phases 1–4 of [issue #23](https://github.com/kobros-tech/6.86x/issues/23). The authoritative current experiment is the fixed-target prerequisite matrix in PR #4; the earlier source→target curriculum is retained below only as historical context.*

> **Historical-controller note (PR #1/#2):** The original source→target run used a compatibility controller that has since been fixed. Independent review found that `τ_solve=0.90` alone could let the controller call a merely-related-but-unsolved task "reuse" (zero training). PR #2 fixed this by requiring an independently measured solve check. The old powers result (clone-and-adapt held-out MSE ≈ 3.71) therefore **must not be treated as a current result**. Against the fixed controller, the original curriculum's powers clone-and-adapt held-out MSE was **0.206 ± 0.076**, versus **0.227** for scratch. The historical narrative that cloning was fast but generalized worse was an artifact of the false-reuse bug and is preserved only in the historical section below.

> **Update (post PR #1/#2, current as of PR #3):** §3.1 and §3.3 below describe the *original* run, made against a compatibility controller that has since been fixed. Independent review found that `τ_solve=0.90` alone was letting the controller call a merely-related-but-unsolved task "reuse" (zero training) — for the powers task specifically, this fired on 9 of 15 seeds, using the raw multiplication network on powers with no adaptation at all (real held-out MSE ≈ 6, far from solved). PR #2 fixed this by gating `reuse` on an independently measured accuracy check, not just the frozen-probe score. **Re-running the original curriculum against the current, fixed controller changes the headline powers number:** clone-and-adapt's final held-out MSE on powers is now **0.206 ± 0.076** (was 3.71 ± 3.10), which is now *on par with* independent scratch (0.227), not five times worse. The §3.3 "mixed result" — fast convergence but worse generalization — **no longer reproduces** against the current codebase; it was an artifact of the false-reuse bug, not a real property of cloning. The rest of this document is left as the original write-up (a snapshot of what the original, now-fixed controller produced) for the historical record; treat §3.3's narrative as superseded by this note, and see PR #2's description / `results/compatibility_calibration_summary.csv` for the fix details. §3.1's retention conclusion (shared network forgets catastrophically, clone-and-adapt does not) is unaffected and still holds.

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
| Division | addition + multiplication | 14/15 | 621.8 | 0.231 |
| Squares | none | 3/15 | 1486.7 | 0.207 |
| Squares | addition | 3/15 | 1490.0 | 0.208 |
| Squares | addition + multiplication | 2/15 | 1499.2 | 0.206 |
| Powers | none | 15/15 | 471.6 | 0.249 |
| Powers | addition | 15/15 | 355.2 | 0.193 |
| Powers | addition + multiplication | 15/15 | 237.3 | 0.169 |

**Source-of-truth correction:** The latest successful workflow run (`run #199`, commit `1e58f44`) reports **621.785714** mean budgeted acquisition steps for division with addition + multiplication, from **14 valid target attempts after one prerequisite failure**. The manuscript-facing value is therefore rounded to **621.8**, not 617.5. The 14/15 success/history-validity distinction is important: one prerequisite failed, so that seed never attempted the target.

### 3.2 Interpretation

The results do **not** support a universal rule that more prior skills always improve acquisition. Under this protocol, prior knowledge was associated with slower budgeted acquisition for subtraction and division, had little effect on the already difficult squares target, and substantially reduced acquisition cost for powers. The powers result is especially clear: mean budgeted acquisition steps decreased from **471.6** with no prior history to **355.2** after addition and **237.3** after addition + multiplication, while held-out MSE also decreased from **0.249** to **0.193** and **0.169**, respectively.

These findings support a narrower claim: **the usefulness of prior knowledge is target- and history-dependent.** They do not establish that addition or multiplication are formal mathematical prerequisites, nor do they imply that a particular skill ordering is universally optimal.

### 3.3 Squares: interpret the budget carefully

Squares is near the training-budget limit in all three histories. Only 3/15 runs succeeded with no prior history, 3/15 after addition, and 2/15 after addition + multiplication. The mean values near 1500 should therefore **not** be described as convergence times. They are means of the budgeted acquisition steps with unsuccessful runs assigned 1500. The held-out MSE is similar across conditions, so the current experiment does not show a reliable acquisition benefit from the tested prior histories for squares.

## 4. Compatibility score notation and meaning

The controller's compatibility quantity is

$$
P(T\mid s_i) = \exp\left(-\frac{\mathrm{MSE}(s_i,T)}{60}\right).
$$

It is a frozen-probe heuristic: it measures how well skill `s_i` already predicts examples from target task `T`. A high value can justify reuse when the independent solve gate also passes, while an intermediate value can make cloning eligible. The score should not be interpreted as a probability calibrated in the statistical sense, nor as a guarantee that cloning will accelerate learning.

## 5. Historical source→target experiment (superseded controller)

The following observations are retained for reproducibility and provenance, but **are not authoritative current results**.

The original curriculum compared addition, subtraction, multiplication, and powers as isolated source→target events. In that pre-fix run, powers was incorrectly allowed to trigger reuse on 9 of 15 seeds because the controller treated a merely related frozen skill as solved. This produced the old held-out MSE ≈ 3.71 result. PR #2 corrected the reuse gate using independent target-solve accuracy. The corrected rerun gave powers clone-and-adapt held-out MSE **0.206 ± 0.076**, versus scratch **0.227**, so the old claim that cloning itself caused worse generalization is **superseded**.

The historical experiment did contain useful evidence that transfer can be heterogeneous, but its pre-fix powers result must not be mixed with the current fixed-target matrix.

## 6. Signed-domain follow-up

The latest workflow also completed the signed-domain transfer-robustness experiment. The paired comparison is valid only for seeds with successful source acquisition in both domains. The resulting speedups are:

| Pair | Valid matched seeds | Non-negative speedup | Signed speedup | Paired difference (non-negative − signed) | Paired p-value | Direction reversed? |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| multiplication → powers | 5/15 | 2.217 ± 0.530 | 0.703 ± 0.141 | 1.514 ± 0.451 | 0.00168 | Yes |
| multiplication → squares | 5/15 | 1.266 ± 0.178 | 1.031 ± 0.069 | 0.235 ± 0.220 | 0.0757 | No; erodes toward null |
| addition → subtraction | 15/15 | 0.408 ± 0.100 | 1.001 ± 0.224 | −0.594 ± 0.243 | 1.88×10⁻⁷ | Yes; negative transfer neutralizes |
| addition → multiplication | 15/15 | 1.145 ± 0.138 | 1.176 ± 0.128 | −0.031 ± 0.218 | 0.591 | No |

The multiplication → powers effect reverses direction and remains statistically different under the paired test. The multiplication → squares comparison is **not** conventionally statistically significant at p = 0.0757; it is more appropriate to describe the effect as eroding toward the null. The addition → multiplication null control shows no statistically detectable domain difference.

The signed-domain history experiment also shows substantial attrition for some prerequisite histories. Squares has 0% target-acquisition success in all signed histories, while powers remains reliably acquired when its prerequisite history is valid but requires roughly 573–576 budgeted steps rather than the much lower 237–472 steps observed in the corresponding non-negative histories. These results support domain sensitivity, not a claim that negative values alone are the causal explanation.

## 7. Retention/isolation mechanism check

The latest workflow re-ran the skill-isolation invariant check over 4 representative three-skill sequences, with 15 matched seeds per sequence. Across all reported checks, mean pre/post accuracy change was exactly **0.0000**, maximum absolute change was **0.0**, all deltas were exactly zero, and the retention pass rate under a five-percentage-point tolerance was **100%**.

This is an implementation/mechanism check: later acquisition uses an independent parameter copy, so the stored parent is not modified. It should **not** be described as an empirical demonstration of immunity to catastrophic forgetting. A meaningful forgetting comparison requires an interference-prone shared-parameter baseline.

## 8. Honest limitations

- The fixed-target experiment uses one arithmetic task family, one architecture size, one probe-batch design, and fixed controller thresholds.
- Fifteen matched seeds provide a controlled comparison but do not establish universal behavior.
- The stopping rule is based on training accuracy and can make early stopping an imperfect proxy for final solution quality; the budgeted-step metric also assigns 1500 steps to unsuccessful runs.
- Squares has very low success under the current 1500-epoch budget, so its acquisition-cost comparisons are weak.
- The prerequisite matrix tests usefulness of previously learned representations; it does **not** prove formal prerequisite relationships.
- No composition of multiple skills into a new computation was tested.
- The signed-domain analysis changes the operand distribution and therefore establishes domain sensitivity under the tested manipulation, not robustness to arbitrary distribution shifts.
- The retention result is an isolation invariant rather than a comparison against an interference-prone learning system.

## 9. Authoritative quantitative evidence

The latest successful workflow is the source of truth for the current branch's numerical results. In particular, the fixed-target division/addition+multiplication condition is **621.785714 → 621.8**, and the signed-domain paired comparisons use the valid matched-seed counts shown above. Historical pre-fix powers results remain excluded from current claims.

The workflow completed all experiment stages and the full regression suite: **29 tests ran and all passed**. The generated results artifact was uploaded successfully.
