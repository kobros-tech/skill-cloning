# Continual Skill Acquisition via Reuse, Clone-and-Adapt, and Scratch Learning

## Abstract

Continual learning requires deciding not only how to learn a new skill, but also whether previously acquired skills should be reused, adapted, or ignored. We study this problem with a controlled three-route framework that chooses among **reuse**, **clone-and-adapt**, and **scratch** learning. Using small arithmetic regression tasks, matched deterministic seeds, and separated training, compatibility-probe, solve/calibration, and held-out evaluation data, we isolate acquisition reliability, transfer efficiency, parent identity, and distribution sensitivity.

The results show that prior knowledge is not uniformly beneficial. In the authoritative fixed-target analysis, additional prior history improves acquisition for some targets, while division exhibits negative transfer and squares remains difficult. A matched three-arm parent-control experiment provides stronger evidence that parent identity matters beyond generic pretraining. For multiplication, the relevant clone is 117.2 steps faster than scratch and 298.3 steps faster than an unrelated parent; the unrelated parent is 181.1 steps slower than scratch. For powers, the corresponding differences are 260.5, 125.9, and 134.7 steps in favor of the first-named arm. All six parent-control paired tests remain significant after Holm correction across the six hypotheses. A signed-domain follow-up further shows that transfer is distribution-sensitive: multiplication→powers reverses direction from 2.217× to 0.703× on the valid matched seeds, while addition→subtraction changes from 0.408× to approximately 1.001×. The multiplication→multiplication null control changes only from 1.145× to 1.176× (p=0.591).

The retention experiment establishes a narrower architectural invariant: previously acquired skills remain unchanged while later skills are learned in isolated parameter copies. We therefore interpret the study as a controlled empirical analysis of skill transfer under the tested acquisition protocol, rather than as evidence of general immunity to catastrophic forgetting or broad generalization to larger architectures and domains.

## 1. Introduction

Continual learning is not only a question of whether a model can learn a new task; it is also a question of how previously acquired capabilities should be used when a new task arrives. If all tasks continually update one shared parameter set, learning a new task can interfere with capabilities acquired earlier. An alternative is to treat acquired capabilities as persistent skills and to make an explicit acquisition decision: reuse a skill when it already solves the target, clone a useful parent and adapt the copy when prior knowledge may help, or fall back to scratch learning when prior knowledge is unsuitable.

The key question studied here is narrower and more testable than whether skill reuse is universally beneficial: **when does previously acquired knowledge help with a new skill, when does it hurt, and does the identity of the reused parent matter?** A continual learner that always reuses prior knowledge can suffer negative transfer, while a learner that always starts from scratch discards potentially useful information. A useful acquisition mechanism should preserve all three options and make their consequences measurable.

We study this question in a deliberately small and controlled experimental setting. The task family consists of arithmetic regression problems with a common input/output structure, and the model is a two-input, one-output neural network with a 32-unit hidden layer. This simplicity is intentional: it allows the source-target relationship, acquisition route, training budget, seed, and data distribution to be controlled independently. The results should therefore be read as a controlled study of transfer dynamics, not as evidence that the same behavior must hold across arbitrary architectures or application domains.

The framework evaluates three acquisition routes. **Reuse** keeps an existing skill unchanged when independent evidence indicates that it already solves the target. **Clone-and-adapt** copies a selected parent and trains the copy on the new task, leaving the stored parent intact. **Scratch** starts from a fresh initialization. This design makes it possible to ask not merely whether pretraining helps, but whether a particular previously acquired skill provides a better initialization than another one.

The experiments are organized around five questions:

1. **Acquisition reliability:** Can a new target skill be acquired within the declared budget, and how does reliability vary across targets and prior histories?
2. **Reuse:** Can an already-solved target be recognized and reused without additional optimization?
3. **Clone versus scratch:** When reuse is insufficient, does cloning reduce the training cost of acquiring a new skill compared with a fresh initialization?
4. **Parent identity:** Does the benefit of cloning depend on which previously acquired skill supplies the initialization, beyond the generic effect of pretraining?
5. **Domain sensitivity:** Does observed transfer remain stable when the operand distribution is expanded from non-negative to signed values?

These questions are answered with matched deterministic seeds, explicit separation of training, compatibility, calibration, and held-out evaluation data, a three-arm parent-control experiment, and an explicit signed-domain follow-up. The evidence is intentionally interpreted as conditional on the tested protocol.

### Contributions

This work makes four contributions:

1. **A controlled three-route skill-acquisition framework.** We define and evaluate a reproducible protocol with three explicit alternatives—reuse, clone-and-adapt, and scratch—together with separate compatibility and independent solve checks.

2. **Evidence for heterogeneous transfer and parent-specific effects.** Matched experiments show that prior skills can produce positive, negative, or limited transfer depending on the target. The three-arm parent-control experiment holds the target, seed, data, architecture, optimizer, and budget fixed while varying parent identity, providing evidence that the identity of the transferred skill matters under the tested protocol.

3. **Evidence that transfer is sensitive to the task distribution.** The signed-domain follow-up changes the operand distribution from non-negative to signed values. The resulting changes include both substantial shifts in transfer magnitude and a reversal of transfer direction.

4. **A reproducible methodology for controlled transfer studies.** The study combines matched deterministic seeds, separated training/probe/calibration/evaluation data, explicit budget-capped acquisition metrics, prerequisite-validity accounting, and multiplicity-corrected paired inference. Retention is deliberately reported as an architectural isolation invariant rather than evidence of general immunity to catastrophic forgetting.

Together, these contributions establish a bounded empirical claim: **under the tested controlled protocol, prior skills can be useful, harmful, or largely neutral, and their effect depends on the target, parent identity, and data distribution.**

## 2. Experimental System

The prototype uses a small two-input, one-output neural network with a 32-unit hidden layer. The task family consists of addition, subtraction, multiplication, powers, squares, and division. The same architecture, optimizer, learning procedure, convergence criterion, and training budget are used across the controlled comparisons; the signed-domain experiment changes only the explicitly documented input distribution.

The controller exposes three acquisition routes:

- **Reuse:** retain an existing skill unchanged when both the compatibility gate and an independent solved-target accuracy check are satisfied.
- **Clone + adapt:** deep-copy an existing skill and train the copy on the target task.
- **Scratch:** create a fresh network and train it on the target task.

The corrected reuse gate is important. A high frozen compatibility score alone is not sufficient evidence that the source skill already solves the target. The independent solve-accuracy check prevents a merely related skill from being incorrectly treated as a zero-training solution.

### 2.1 Mathematical Formulation

Let the incoming target task be $T$ and let the repository contain previously acquired skills $s_i$, each represented by a parameter vector $\theta_i$. A skill maps an input $x$ to an output through $f(x;\theta_i)$.

For a target compatibility-probe set $D_T^{\mathrm{probe}}=\{(x_j,y_j)\}_{j=1}^{m}$, the frozen compatibility score is computed from mean squared error:

$$
\mathrm{MSE}(T,s_i)=\frac{1}{m}\sum_{j=1}^{m}
\left(f(x_j;\theta_i)-y_j\right)^2.
$$

The controller transforms this error into a compatibility score:

$$
P(T\mid s_i)=\exp\left(-\frac{\mathrm{MSE}(T,s_i)}{60}\right).
$$

The compatibility score is computed without updating $\theta_i$. It therefore measures how well the frozen parent already matches the target; it is not itself a measure of post-adaptation transfer benefit.

The controller then evaluates the same frozen parent on a separate **solve/calibration batch** $D_T^{\mathrm{solve}}=\{(x_j,y_j)\}_{j=1}^{64}$, sampled independently from the compatibility probe. The target-solve accuracy is:

$$
A(T,s_i)=\frac{1}{64}\sum_{j=1}^{64}
\mathbf{1}\!\left[
\left|f(x_j;\theta_i)-y_j\right|\leq0.5
\right].
$$

The calibration batch is separate from training data and from the compatibility-probe batch. The held-out test set is never consulted for the reuse decision. The implementation uses the independent solve-probe seed offset $20{,}000$ and the same task-sampling procedure as the experiments.

With compatibility thresholds $\tau_{\mathrm{solve}}=0.90$ and $\tau_{\mathrm{clone}}=0.15$, the controller selects:

$$
\mathrm{action}(T,s_i)=
\begin{cases}
\mathrm{reuse} & \text{if } P(T\mid s_i)\geq\tau_{\mathrm{solve}} \text{ and } A(T,s_i)\geq0.85,\\
\mathrm{clone} & \text{if } P(T\mid s_i)\geq\tau_{\mathrm{clone}} \text{ and the reuse condition is not satisfied},\\
\mathrm{scratch} & \text{otherwise.}
\end{cases}
$$

When cloning from parent $s_i$, the target is initialized by:

$$
\theta_T^{(0)}=\theta_i.
$$

Scratch learning uses an independent initialization. The adapted parameters minimize target mean squared error:

$$
\theta_T^*=\arg\min_{\theta}\;\frac{1}{n}\sum_{j=1}^{n}
\left(f(x_j;\theta)-y_j\right)^2.
$$

These equations define the mechanism evaluated in the experiments. They do not imply that a larger compatibility score must produce a larger transfer benefit.

### 2.2 Acquisition Metrics

For matched seed $r$, let $E_{\mathrm{scratch}}^{(r)}$ and $E_{\mathrm{clone}}^{(r)}$ denote the epochs required to reach the predeclared acquisition criterion. The per-seed training-cost ratio is:

$$
S^{(r)}=\frac{E_{\mathrm{scratch}}^{(r)}}{E_{\mathrm{clone}}^{(r)}}.
$$

Thus $S>1$ favors cloning, whereas $S<1$ indicates that cloning is slower than scratch. Training is capped at 1500 epochs. If an attempted target does not reach the criterion, its recorded epoch count is the full 1500-epoch budget. The ratio is therefore a **budget-capped training-cost comparison**, not a convergence-time estimate restricted to successful runs.

Acquisition reliability is reported separately. Let $I_r=1$ when a target reaches the declared criterion within the allowed budget and $I_r=0$ otherwise. The empirical success rate is:

$$
R=\frac{1}{N}\sum_{r=1}^{N}I_r.
$$

This distinction is important for difficult targets such as squares, where a method can have low success despite informative successful-run speed comparisons.

### 2.3 Parent-Control Design

To distinguish useful parent identity from generic pretraining, a three-arm matched control compares a relevant clone, an unrelated previously acquired clone, and scratch initialization. For each target comparison, all three arms use the same target, seed, training data, architecture, optimizer, and budget. Only skills acquired before the target are eligible as parents.

The relevant parent is the highest-compatibility previously acquired skill under the controller's ranking. The unrelated parent is a different previously acquired skill. Both cloned arms are deep copies of their respective stored networks and are trained on the same target data as the scratch arm.

The inferential unit is the matched seed within target. Descriptive arm means are secondary; paired differences are the primary inferential summaries. The six parent-control hypotheses are corrected jointly with Holm's step-down procedure.

### 2.4 Retention and Isolation Invariant

For an acquired skill $s_i$, let $\theta_i^{\mathrm{pre}}$ and $\theta_i^{\mathrm{post}}$ denote its stored parameters immediately before and after a later skill is acquired. The intended isolated-skill invariant is:

$$
\theta_i^{\mathrm{post}}=\theta_i^{\mathrm{pre}},
$$

or equivalently $\Delta\theta_i=0$.

The retention experiment evaluates the same stored skill on the same stable retention set before and after later acquisition. Its diagnostic accuracy change is:

$$
\Delta A_i=A_{i,\mathrm{post}}-A_{i,\mathrm{pre}}.
$$

The predeclared practical diagnostic criterion is $\Delta A_i\geq-0.05$.

Because the stored parent is not optimized after acquisition and the evaluation set is unchanged, zero change is expected under the mechanism. Retention is therefore interpreted as an implementation/invariance diagnostic, not as a statistical estimate of resistance to interference.

### Data Separation and Seeds

The experiment separates target-training data, compatibility-probe data, solve/calibration data, and held-out evaluation data. Held-out evaluation data are not used to choose thresholds, budgets, stopping rules, or model-selection decisions.

The main relatedness, parent-control, and retention checks use 15 deterministic seeds per condition. The signed-domain follow-up also uses 15 nominal seeds per condition. Pairing by seed permits within-seed comparisons wherever the same source acquisition is valid in both conditions.

## 3. Research Design

### 3.1 Historical Relatedness Analysis and Authoritative Fixed-Target Matrix

The initial source-target experiments examined multiplication→squares, multiplication→powers, addition→subtraction, and addition→multiplication. These results are retained as a **historical relatedness analysis** and are useful for showing that transfer can be heterogeneous.

The **expanded fixed-target prerequisite matrix is the authoritative current quantitative analysis**. It holds the target fixed and varies the prior-skill history across no prior skill, addition only, and addition + multiplication. The matrix includes subtraction, division, squares, and powers. This design distinguishes the effect of relevant prior knowledge from the mere presence of additional training history.

A curriculum order is not treated as proof that one task is a mathematical prerequisite for another. The experiment only tests whether a previously learned representation is useful for acquiring the target under the stated protocol. A requested prerequisite must actually reach the acquisition criterion before it is exposed to the controller. If a prerequisite fails, the history is marked invalid, the failed skill is unavailable, and the target is not attempted for that seed.

### 3.2 Parent-Identity Control

The parent-control experiment is a matched three-arm test of the claim that relevant cloning provides more than generic pretraining. The relevant and unrelated parents are both drawn from the set of skills already acquired before the target; no future task or target-trained representation is exposed to parent selection. For each seed, the relevant, unrelated, and scratch arms are trained independently on the same target data.

The experiment reports paired differences for relevant minus scratch, relevant minus unrelated, and unrelated minus scratch. Six paired hypotheses are tested across multiplication and powers. Holm's step-down correction is applied across these six paired $t$-test $p$-values.

This control is stronger than a clone-versus-scratch comparison alone. If all pretrained networks helped equally, relevant and unrelated cloning would be expected to perform similarly. A difference between the two cloned parents therefore provides evidence that parent identity matters under the tested protocol, without by itself establishing a causal representation-level mechanism.

### 3.3 Retention and Catastrophic Forgetting

The retention code performs sequential acquisition and re-evaluates every previously acquired skill after each later acquisition on a stable, skill-specific evaluation set. This verifies the intended isolation invariant: a stored parent skill is not modified when a later skill is trained as an independent copy.

For each retention check the implementation records pre/post accuracy, accuracy change, retention ratio, and whether the change remains within the predeclared five-percentage-point practical tolerance. These measurements are useful diagnostics of the invariant, but they are not a conventional test of catastrophic forgetting because the isolated-skill architecture does not expose the stored parent to subsequent optimization.

A genuine empirical test of resistance to interference would require an at-risk comparison arm in which later learning can modify previously learned parameters, such as a shared-network baseline. That comparison is outside the scope of this study.

### 3.4 Signed-Domain Follow-Up

The signed-domain experiment compares the original non-negative configuration with an explicitly configured signed configuration. Task ranges are task-specific and are defined by the experiment configuration. The purpose is to test whether observed transfer relationships remain stable under a concrete change in operand distribution, not to isolate negative values as the sole causal variable.

## 4. Results

### 4.1 Authoritative Fixed-Target Acquisition Results

The fixed-target prerequisite matrix is the primary quantitative result. The target is held fixed while prior-skill history varies across no prior skill, addition only, and addition + multiplication.

| Target | No prior skill | Addition only | Addition + multiplication |
|---|---:|---:|---:|
| Subtraction | 33.5 epochs | 62.3 epochs | 73.2 epochs |
| Division | 515.2 epochs | 616.2 epochs | 617.5 epochs |
| Squares | 20.0% success | 20.0% success | 13.3% success |
| Powers | 471.6 epochs | 355.2 epochs | 237.3 epochs |

For subtraction, division, and powers, values are acquisition epochs to the declared criterion, with an unsuccessful target after a valid prerequisite history contributing the full 1500-epoch budget. Invalid prerequisite histories are excluded because the target was never attempted. Squares is reported as success rate because many runs do not reach the acquisition criterion within the allowed budget.

The matrix shows heterogeneous transfer rather than a universal benefit from additional prior knowledge. Powers improves substantially as prior history expands, whereas division becomes slower and squares remains difficult. These results are evidence about transfer under the tested protocol, not proof of formal mathematical prerequisite relationships.

### 4.2 Parent-Identity Control Results

The three-arm parent-control experiment provides the strongest current evidence that the benefit of cloning is not explained solely by generic pretraining. All six paired $t$-test comparisons remain below 0.05 after Holm correction.

| Target | Comparison | Mean paired difference (steps) | Raw paired $p$ | Holm-adjusted $p$ |
|---|---|---:|---:|---:|
| Multiplication | relevant − scratch | −117.2 | 0.012672 | 0.012672 |
| Multiplication | relevant − unrelated | −298.3 | 5.91×10^-6 | 1.77×10^-5 |
| Multiplication | unrelated − scratch | +181.1 | 9.18×10^-4 | 1.84×10^-3 |
| Powers | relevant − scratch | −260.5 | 6.01×10^-11 | 3.61×10^-10 |
| Powers | relevant − unrelated | −125.9 | 1.01×10^-6 | 4.03×10^-6 |
| Powers | unrelated − scratch | −134.7 | 2.40×10^-8 | 1.20×10^-7 |

Negative differences favor the first-named arm because the outcome is convergence steps. For multiplication, relevant cloning is faster than both scratch and unrelated cloning, while unrelated cloning is slower than scratch. For powers, both pretrained arms outperform scratch, but the relevant parent is still significantly faster than the unrelated parent.

The pattern supports a narrower claim than “cloning helps”: **parent identity matters under the tested protocol**. The multiplication result shows that an unrelated parent can be worse than scratch, whereas the powers result shows that unrelated pretraining can itself help. Thus the relevant-parent advantage cannot be reduced to a simple pretrained-versus-untrained distinction. These results do not imply that every relevant parent will outperform every unrelated parent on arbitrary tasks.

### 4.3 Historical Relatedness Results

The earlier relatedness-pair experiment produced heterogeneous transfer: multiplication→powers showed a mean paired speedup of approximately 2.31×, multiplication→squares approximately 1.26×, addition→subtraction approximately 0.33×, and addition→multiplication approximately 1.12×.

These values are retained as historical relatedness results. They are not substituted for the authoritative fixed-target matrix or the parent-control analysis. Their purpose is to show why source-target relatedness must be evaluated empirically rather than assumed from task labels or intuitive prerequisite order.

### 4.4 Signed-Domain Transfer Results

The signed-domain follow-up compares the original non-negative configuration with an explicitly configured signed configuration. The primary outcome is the matched-seed training-cost ratio $S$.

| Pair | Valid matched seeds | Non-negative | Signed | Difference (non-negative − signed) | Paired $p$ |
|---|---:|---:|---:|---:|---:|
| multiplication→powers | 5/15 | 2.217±0.530 | 0.703±0.141 | 1.514±0.451 | 0.00168 |
| multiplication→squares | 5/15 | 1.265±0.178 | 1.031±0.069 | 0.235±0.220 | 0.0757 |
| addition→subtraction | 15/15 | 0.408±0.100 | 1.001±0.224 | −0.594±0.243 | ≈1.88×10^-7 |
| addition→multiplication (null control) | 15/15 | 1.145±0.138 | 1.176±0.128 | −0.031±0.218 | 0.591 |

The first two comparisons have only 5/15 valid matched seeds because signed-domain multiplication prerequisite acquisition fails in 10 seeds. Those failures are explicitly reported and are not converted into artificial speedup observations.

Multiplication→powers reverses direction: the paired mean changes from 2.217× to 0.703×, with a statistically detectable domain difference ($p=0.00168$). Multiplication→squares moves toward no effect, from 1.265× to 1.031×; the paired comparison is not conventionally significant ($p=0.0757$), so this should not be described as proof of equivalence. Addition→subtraction changes from negative transfer (0.408×) to approximately neutral transfer (1.001×), with a strong paired difference. The addition→multiplication null control changes only from 1.145× to 1.176× and shows no statistically detectable domain difference ($p=0.591$).

These results support domain sensitivity under the tested configuration, but the signed-domain comparison is not a clean causal test of negative numbers alone because the input distribution changes more broadly.

### 4.5 Signed-Domain Prerequisite Reliability

The signed-domain experiment also reveals prerequisite attrition. Invalid histories are fail-closed: the failed prerequisite is not exposed to the controller and the target is not attempted.

Signed-domain squares has 0/15 target successes for all three tested histories. This establishes failure within the declared 1500-epoch budget for this protocol; it does not establish impossibility of learning signed squares in general.

### 4.6 Compatibility Sensitivity

The controller itself is distribution-sensitive. For division with an addition parent, the frozen compatibility score is approximately 0.28–0.31 in the non-negative domain and approximately 0.03–0.06 in the signed domain. The non-negative values exceed the clone threshold while the signed values fall below it, so controller behavior shifts toward scratch under the signed configuration.

This observation is consistent with the broader domain-sensitivity results: the decision mechanism is based on the frozen behavior of a stored parent on the current target distribution, so changing that distribution can change both measured compatibility and the selected acquisition route.

### 4.7 Retention Mechanism Check

The retention run reports zero change in the repeated pre/post checks and a 100% pass rate under the five-percentage-point practical tolerance. These values are consistent with the implementation invariant that previously acquired skills are stored independently and are not modified while a new skill is adapted.

Because the same unchanged skill is evaluated on the same skill-specific retention set before and after later acquisitions, zero change is an expected consequence of the mechanism. We therefore interpret the result as an implementation-level verification of parameter isolation, not as evidence that the architecture is immune to catastrophic forgetting under interference.

## 5. Statistical Analysis

The inferential unit for matched strategy comparisons is the seed. For parent-control comparisons, the relevant, unrelated, and scratch arms are evaluated on the same target and matched seed, and paired differences are tested within target.

The six parent-control hypotheses form one multiplicity family. Holm's step-down procedure is applied to the six raw paired $t$-test $p$-values. The adjusted values remain below 0.05 for all six comparisons. This correction reduces the risk that the parent-identity conclusion depends on an uncorrected collection of six tests.

For signed-domain comparisons, only seeds valid in both domains are included. The paired powers and squares comparisons therefore use 5/15 valid matched seeds, not 15/15. The 10 signed multiplication-prerequisite failures are reported separately and are not converted into missing-at-random speed observations.

A non-significant $p$-value is not interpreted as evidence of exact equality or equivalence. This is particularly important for multiplication→squares and the addition→multiplication null control. Conversely, statistically significant paired differences are interpreted as evidence of a difference under the tested protocol, not as proof of a particular representation-level causal mechanism.

Retention is treated differently. Since the stored parent is not optimized after acquisition, zero change is expected under the isolation architecture. Statistical inference on the retention delta would therefore not answer the stronger scientific question of whether a system resists interference when interference is possible.

## 6. Discussion

The combined findings support a bounded view of skill transfer. Prior knowledge is neither uniformly beneficial nor uniformly harmful. Its effect depends on the relationship between the source and target, the identity of the parent selected for cloning, and the input distribution on which the skills are evaluated.

The parent-control experiment is especially informative because it separates generic pretraining from source-specific transfer. An unrelated parent can be beneficial for one target and harmful for another, while a relevant parent can provide an additional advantage. The evidence therefore supports the statement that **parent identity matters under the tested protocol**, rather than the broader statement that any pretrained parent is beneficial.

The fixed-target matrix complements this result. Powers benefits from additional prior history, while division exhibits negative transfer and squares remains difficult. These heterogeneous outcomes argue against treating curriculum order as a universal prerequisite graph. A prior skill can improve acquisition under one target while adding little or imposing a cost under another.

The signed-domain follow-up strengthens the conditional interpretation. Several transfer ratios change substantially when the operand distribution changes, including a reversal for multiplication→powers. Addition→subtraction changes from negative transfer to approximately neutral transfer, while the addition→multiplication null control shows no statistically detectable domain difference. The results therefore suggest that transfer relationships are properties of the source, target, and distribution together, rather than immutable properties of task names.

The retention result has a deliberately narrower interpretation. Because skills are stored independently and later training operates on copies, preservation of the original parameters is an architectural consequence of isolation. It is useful as an implementation invariant, but it does not replace a shared-parameter continual-learning baseline for studying catastrophic forgetting under interference.

## 7. Limitations and Threats to Validity

**Task-family scope.** The experiments use a small arithmetic task family. Arithmetic provides strong control over source-target relationships but cannot establish that the same transfer dynamics occur in language, vision, control, or other domains.

**Model scale.** The model is small, with a 32-unit hidden layer. Results may depend on architecture, parameter count, optimizer, and training dynamics.

**Finite seeds.** Fifteen seeds provide useful matched comparisons but do not establish universal population-level behavior. The signed-domain powers and squares comparisons are more limited because only 5/15 seeds are valid in both domains.

**Controller dependence.** The behavior depends on the compatibility probe, thresholds, and independent solve gate. The study evaluates this specified decision mechanism rather than claiming that the thresholds are optimal or universally appropriate.

**Budget dependence.** Acquisition cost is capped at 1500 epochs. A capped ratio can therefore reflect budgeted failure as well as successful convergence speed and should not be interpreted as an unconstrained optimization-time estimate.

**Parent-control scope.** The relevant/unrelated/scratch comparison covers two targets in one small arithmetic family. It supports parent-identity dependence under the tested protocol, not a universal ranking rule for source skills.

**Signed-domain confounding.** The signed-domain manipulation changes the input distribution in addition to introducing negative values. The conclusions therefore concern domain/distribution sensitivity rather than negative numbers alone.

**Retention interpretation.** Perfect retention is expected from isolated storage. The study does not include an at-risk shared-network baseline in which later training can modify parameters supporting earlier skills. Consequently, it does not establish general resistance to catastrophic forgetting.

**Prerequisite interpretation.** Earlier acquisition in a curriculum is evidence about transfer under the stated protocol, not proof of a formal mathematical prerequisite relationship.

**Failure interpretation.** Failure within the declared budget does not establish that a task is impossible to learn. It establishes only that the acquisition criterion was not reached under the tested configuration and budget.

## 8. Reproducibility

The repository contains the experiment drivers, regression tests, workflow configuration, raw result artifacts, statistical summaries, and documentation needed to audit the headline results. CI executes the relevant experiments and uploads generated artifacts.

The parent-control experiment records per-seed arm outcomes and paired statistics for relevant, unrelated, and scratch arms. The final statistical analysis applies Holm correction across the six parent-control paired tests.

The signed-domain experiment records per-seed pair validity, source-acquisition failures, domain-specific outcomes, paired statistics, history validity, and sign-specific diagnostics. The manuscript reports the effective matched sample size for the paired domain tests.

The non-negative configuration remains the default. Signed-domain behavior is an explicit experimental configuration and does not silently alter the baseline task distribution.

## 9. Conclusion

This study evaluates a controlled continual skill-acquisition mechanism in which an incoming target can be handled by reuse, clone-and-adapt, or scratch learning. The experiments do not support the claim that cloning always helps, nor do they establish general immunity to catastrophic forgetting.

The authoritative fixed-target analysis instead shows heterogeneous transfer: powers benefits from additional prior history, division exhibits negative transfer, and squares remains difficult. The matched parent-control experiment provides stronger evidence that the source identity matters: relevant cloning is faster than scratch and unrelated cloning for both multiplication and powers after Holm correction, while unrelated pretraining can either hurt or help depending on the target. Thus the benefit is not explained solely by generic pretraining; **parent identity matters under the tested protocol**.

The signed-domain follow-up further shows that transfer can change with the input distribution. Multiplication→powers reverses direction, multiplication→squares moves toward no effect without a conventionally significant paired domain difference, and addition→subtraction changes from negative to approximately neutral transfer. The addition→multiplication null control shows no statistically detectable domain difference under the tested manipulation.

Finally, the isolated-skill mechanism preserves stored parent parameters during later acquisition as an architectural invariant. This is a property of independent storage and copy-based adaptation, not a demonstration that an interference-capable continual learner would resist catastrophic forgetting.

The resulting contribution is therefore a controlled empirical foundation for studying **when and why previously acquired skills help or hurt the acquisition of new skills**. Future work should extend the parent-control design to broader task families and add explicit at-risk shared-parameter baselines so that transfer and forgetting can be studied under the same experimental framework.
