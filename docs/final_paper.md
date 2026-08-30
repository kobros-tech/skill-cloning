# Continual Skill Acquisition via Reuse, Clone-and-Adapt, and Scratch Learning

## Abstract

This study evaluates a small reproducible framework for continual skill acquisition in which an incoming task can reuse an already-solved skill, clone a related skill and adapt it, or learn from scratch. The experiments use small arithmetic regression tasks and matched deterministic seeds to separate acquisition reliability, transfer efficiency, domain sensitivity, and the preservation of previously acquired skills.

The central result is that prior knowledge is not uniformly beneficial: transfer depends on the source-target relationship and on the input distribution. The authoritative fixed-target experiment shows heterogeneous transfer: powers benefit from additional prior history, division exhibits negative transfer, and squares remains difficult. A signed-domain follow-up further shows distribution sensitivity. On the valid matched seeds, multiplication → powers changes from 2.217× to 0.703× and reverses direction (`p=0.00168`); multiplication → squares changes from 1.265× to 1.031× without a conventionally significant domain difference (`p=0.0757`); addition → subtraction changes from 0.408× to 1.001× (`p≈1.88×10⁻⁷`); and the addition → multiplication null control changes from 1.145× to 1.176× with no statistically detectable domain difference (`p=0.591`). The powers and squares signed-domain comparisons have only 5/15 valid matched seeds because signed-domain multiplication prerequisite acquisition fails in 10 seeds.

A separate skill-isolation check confirms that stored parent parameters are unchanged during later acquisitions. This is an implementation invariant of the current architecture, not a statistical demonstration that catastrophic forgetting is absent.

## 1. Introduction

A continual-learning system should acquire new capabilities without unnecessarily overwriting earlier ones. One strategy is to store learned capabilities independently and choose among reuse, clone-and-adapt, and scratch learning for each new target.

This work studies that mechanism as a controlled research prototype. The goal is not to claim a universal theory of prerequisites or a universal advantage for cloning. Instead, the experiments ask when previous skills help, when they hurt, whether transfer depends on the task distribution, and whether independent skill storage preserves earlier skills.

The research questions are:

1. Can a new target skill be acquired reliably?
2. Can an already-solved target be reused without additional training?
3. When reuse is insufficient, does cloning provide a useful initialization compared with scratch learning?
4. Does independent skill storage prevent later acquisition from modifying earlier stored skills?
5. Does transfer remain stable when the operand distribution is expanded to include signed values?

## 2. Experimental system

The prototype uses a two-input, one-output neural network with a 32-unit hidden layer. The task family contains addition, subtraction, multiplication, powers, squares, and division. The controller has three routes:

* **Reuse:** keep an existing skill unchanged when compatibility and an independent target-solve check both pass.
* **Clone + adapt:** deep-copy a source skill and train the copy on the target.
* **Scratch:** train a fresh network.

The corrected reuse gate prevents a merely related but unsolved skill from being treated as a zero-training solution.

### 2.1 Mathematical formulation

Let target task be \(T\) and an acquired skill be \(s_i\) with parameters \(\theta_i\). For a target probe set \(D_T^{\mathrm{probe}}=\{(x_j,y_j)\}_{j=1}^{m}\), frozen compatibility is:

$$
\mathrm{MSE}(T,s_i)=\frac{1}{m}\sum_{j=1}^{m}
\left(f(x_j;\theta_i)-y_j\right)^2
$$

$$
P(T\mid s_i)=\exp\left(-\frac{\mathrm{MSE}(T,s_i)}{60}\right).
$$

The score is computed without modifying the source network. Let \(A(T,s_i)\) be an independent target-solve accuracy. With \(\tau_{\mathrm{solve}}=0.90\), \(\tau_{\mathrm{clone}}=0.15\), and solve accuracy threshold 0.85:

$$
\mathrm{action}(T,s_i)=
\begin{cases}
\mathrm{reuse} & P(T\mid s_i)\geq\tau_{\mathrm{solve}}\text{ and }A(T,s_i)\geq0.85,\\
\mathrm{clone} & P(T\mid s_i)\geq\tau_{\mathrm{clone}}\text{ and reuse is not selected},\\
\mathrm{scratch} & \text{otherwise.}
\end{cases}
$$

For cloning, \(\theta_T^{(0)}=\theta_i\); for scratch, the target starts from an independent initialization. Target training minimizes mean squared error.

### 2.2 Acquisition metric

For a valid matched seed, the budget-capped training-cost ratio is:

$$
S^{(r)}=\frac{E_{\mathrm{scratch}}^{(r)}}{E_{\mathrm{clone}}^{(r)}}.
$$

A value above 1 favors cloning. Training is capped at 1500 epochs. If an attempted target does not reach the acquisition criterion, its recorded epoch count is the full 1500-epoch budget. Consequently, this ratio is a **budget-capped training-cost comparison**, not a convergence-time comparison restricted to successful runs.

Acquisition reliability is reported separately as the fraction reaching the declared criterion within the budget.

### 2.3 Skill-isolation invariant

For an acquired skill, let \(\theta_i^{\mathrm{pre}}\) and \(\theta_i^{\mathrm{post}}\) denote its stored parameters before and after later acquisition. The intended invariant is:

$$
\theta_i^{\mathrm{post}}=\theta_i^{\mathrm{pre}}.
$$

The retention check evaluates the same frozen network on the same deterministic retention set before and after later acquisition. Therefore zero accuracy change is expected by construction. This verifies the implementation invariant; it does not constitute an empirical interference experiment.

## 3. Research design

### 3.1 Fixed-target prerequisite matrix

The authoritative current experiment holds the target fixed while varying prior history across no prior skill, addition only, and addition + multiplication. The targets are subtraction, division, squares, and powers. Each nominal condition has 15 seeds.

A requested prerequisite must actually reach the acquisition criterion before it is exposed to the controller. If a prerequisite fails, the history is marked invalid, the failed skill is unavailable, and the target is not attempted for that seed.

### 3.2 Signed-domain follow-up

The signed-domain experiment compares the original non-negative configuration with an explicitly configured signed configuration. Task ranges are task-specific: addition, subtraction, multiplication, and squares use operands from `[-9,9]`; powers use a signed base range with a non-negative exponent; division uses signed numerators and nonzero signed divisors. Inputs remain scaled by 10 and the same architecture, optimizer, training budget, controller, and seed protocol are used.

The comparison therefore tests domain/distribution sensitivity rather than isolating a causal effect of negative numbers alone.

## 4. Results

### 4.1 Historical relatedness results

The earlier source-target analysis remains useful as historical context: multiplication → powers was approximately 2.31×, multiplication → squares approximately 1.26×, addition → subtraction approximately 0.33×, and addition → multiplication approximately 1.12×. These are historical relatedness results and are not mixed with the authoritative fixed-target matrix.

### 4.2 Authoritative fixed-target results

| Target | No prior skill | Addition only | Addition + Multiplication |
|---|---:|---:|---:|
| Subtraction | 33.5 | 62.3 | 73.2 |
| Division | 515.2 | 616.2 | **621.8** |
| Squares | 20.0% success | 20.0% success | 13.3% success |
| Powers | 471.6 | 355.2 | 237.3 |

For subtraction, division, and powers, these are mean budgeted target-adaptation steps among seeds whose prerequisite history was valid. A target failure after a valid history contributes the full 1500-epoch budget. Invalid prerequisite histories are excluded because the target was never attempted. Squares is reported as success rate because most runs do not reach the criterion within the budget.

The results show heterogeneous transfer. Additional history substantially helps powers, while division becomes slower and squares remains difficult. These results support target- and history-dependent transfer, not a formal mathematical prerequisite hierarchy.

### 4.3 Skill-isolation check

The CI retention run reports zero pre/post accuracy change for all recorded checks and a 100% pass rate under the five-percentage-point sanity tolerance. Because the stored parent network and evaluation set are unchanged, this outcome is expected by construction. The legitimate conclusion is that the implementation's isolation guarantee holds; it is not evidence that a system with interference-capable parameters would resist catastrophic forgetting.

A genuine empirical forgetting experiment would require an at-risk baseline, such as a shared-parameter model in which later learning can modify parameters supporting earlier skills.

### 4.4 Signed-domain transfer results

The current CI artifact gives the following paired domain comparisons. Pairing is performed only for seeds that have a valid source acquisition in both domains.

| Pair | Valid matched seeds | Non-negative | Signed | Difference (non-negative − signed) | Paired p |
|---|---:|---:|---:|---:|---:|
| multiplication → powers | 5/15 | 2.217 ± 0.530 | 0.703 ± 0.141 | 1.514 ± 0.451 | 0.00168 |
| multiplication → squares | 5/15 | 1.265 ± 0.178 | 1.031 ± 0.069 | 0.235 ± 0.220 | 0.0757 |
| addition → subtraction | 15/15 | 0.408 ± 0.100 | 1.001 ± 0.224 | −0.594 ± 0.243 | 1.88×10⁻⁷ |
| addition → multiplication (null control) | 15/15 | 1.145 ± 0.138 | 1.176 ± 0.128 | −0.031 ± 0.218 | 0.591 |

The first two comparisons have only 5/15 valid matched seeds because signed-domain multiplication acquisition fails in 10 seeds. Those failures are explicitly reported and are not converted into artificial speedup observations.

**Multiplication → powers reverses direction.** The paired mean changes from 2.217× to 0.703× and the domain comparison is statistically different (`p=0.00168`). Sign-specific diagnostics show substantially higher error on negative-base odd-exponent cases. This is consistent with a harder sub-problem under the signed distribution, but does not prove the mechanism causally.

**Multiplication → squares moves toward no effect.** The paired mean changes from 1.265× to 1.031×. The paired comparison is not conventionally statistically significant (`p=0.0757`), so the appropriate conclusion is erosion toward no effect rather than a demonstrated significant domain difference.

**Addition → subtraction neutralizes.** Negative transfer in the non-negative domain (0.408×) becomes approximately neutral in the signed domain (1.001×), with all 15 seeds valid.

**Addition → multiplication remains a null control.** The ratio changes from 1.145× to 1.176×, with no statistically detectable domain difference (`p=0.591`). This does not prove equivalence; it means the tested paired comparison did not detect a statistically significant domain effect.

### 4.5 Signed-domain prerequisite reliability

The signed fixed-target experiment also reveals prerequisite attrition. Examples include 7/15 invalid histories for subtraction after addition + multiplication, 9/15 invalid histories for division after addition + multiplication, 3/15 invalid histories for squares after addition + multiplication, and 5/15 invalid histories for powers after addition + multiplication. These invalid histories are fail-closed: the failed prerequisite is not exposed to the controller and the target is not attempted.

Signed-domain squares has 0/15 target successes for all three tested histories. This establishes failure within the declared 1500-epoch budget for this protocol; it does not establish impossibility of learning signed squares in general.

### 4.6 Compatibility sensitivity

The controller's input is itself distribution-sensitive. For division with an addition parent, the frozen compatibility score is approximately 0.28–0.31 in the non-negative domain and approximately 0.03–0.06 in the signed domain. The non-negative values exceed the clone threshold while the signed values are below it, so controller behavior shifts toward scratch under the signed configuration.

## 5. Statistical interpretation

Paired comparisons use the matched seed as the unit of analysis. For signed-domain comparisons, only seeds valid in both domains are included. This is essential because prerequisite acquisition failure changes the estimand: a failed source does not provide a valid clone-versus-scratch transfer comparison.

The signed-domain paired t-tests therefore describe changes in the budget-capped training-cost ratio among valid matched source acquisitions. They do not include the 10 multiplication-prerequisite failures in the powers and squares paired tests.

The retention check is intentionally not analyzed with confidence intervals or effect sizes. Its zero delta is a direct consequence of evaluating an unchanged stored network on an unchanged evaluation set.

## 6. Limitations

1. The task family is small and arithmetic; results may not generalize to broader domains.
2. The model is small, and outcomes may depend on architecture and optimization.
3. Fifteen seeds provide controlled matched comparisons but do not establish universal behavior.
4. Controller thresholds and compatibility probes are protocol-dependent.
5. The skill-isolation result is an implementation invariant, not an interference-capable retention experiment.
6. The fixed-target experiment tests usefulness of prior representations, not formal prerequisite relationships.
7. The signed-domain manipulation changes the input distribution in addition to introducing negative values, so conclusions concern domain/distribution sensitivity rather than negative numbers alone.
8. The powers and squares signed-domain pair tests use only 5/15 valid matched seeds, limiting precision and power.
9. The budget-capped speedup ratio incorporates 1500-epoch caps for unsuccessful target attempts; it should not be described as convergence speed among successful runs.
10. Acquisition failure within the budget is not evidence that the task is impossible to learn.

## 7. Reproducibility

The repository contains experiment drivers, regression tests, workflow configuration, raw CSV outputs, statistical summaries, and plots. CI executes the baseline experiment suite, stopping-rule confound analysis, squares relatedness analysis, compatibility calibration, fixed-target history analysis, skill-isolation invariant check, signed-domain experiment, and regression tests, then uploads the generated results artifact.

The signed-domain experiment records per-seed pair validity, source-acquisition failures, domain-specific outcomes, paired statistics, history validity, and sign-specific diagnostics. The publication tables report the effective matched sample size for the paired domain tests.

The non-negative configuration remains the default. Signed-domain behavior is an explicit experimental configuration and does not silently alter the baseline task distribution.

## 8. Conclusion

The experiments support a conservative conclusion: continual skill acquisition benefits from retaining multiple acquisition routes rather than assuming that previous knowledge is always useful. Prior knowledge can accelerate some targets, slow others, and fail to help difficult targets. The usefulness of a cloned representation is therefore source-, target-, and distribution-dependent.

The signed-domain follow-up strengthens this conclusion. Multiplication → powers reverses from positive to negative transfer, multiplication → squares moves toward no effect without a statistically significant paired domain difference, and addition → subtraction's negative transfer neutralizes. The addition → multiplication null control shows no statistically detectable domain difference. These findings are conditional on the tested arithmetic task family and domain expansion.

The independent skill-storage architecture also provides a clear implementation-level isolation guarantee: later acquisition does not modify stored parent parameters. This is a useful architectural property, but it should not be confused with an empirical demonstration of resistance to catastrophic forgetting under interference-capable learning.

Future work should therefore combine the present transfer and domain-sensitivity protocol with an explicit at-risk shared-parameter baseline, broader task families, and larger models.

## Reproducibility checklist

* [x] Reuse / clone / scratch routes implemented.
* [x] Corrected reuse gate requires independent solved-target evidence.
* [x] Fail-closed prerequisite semantics implemented.
* [x] Matched seeds used for paired comparisons.
* [x] Signed-domain configuration is explicit and tested.
* [x] Signed-domain source-acquisition attrition is reported.
* [x] Signed-domain paired statistics are reported with effective sample sizes.
* [x] Skill-isolation invariant is tested on stable skill-specific evaluation sets.
* [x] Retention result is described as an implementation invariant, not a statistical forgetting result.
* [x] CI runs the full experiment suite and regression tests.
* [x] Generated artifacts are uploaded by CI.
* [ ] At-risk empirical retention baseline remains future work.
* [ ] Broader task families and larger models remain future work.
