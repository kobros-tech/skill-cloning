# Continual Skill Acquisition via Reuse, Clone-and-Adapt, and Scratch Learning

## Abstract

Continual acquisition systems must learn new skills without unnecessarily discarding useful prior knowledge. This study evaluates a small, reproducible skill-acquisition framework in which an incoming task is handled by one of three routes: reuse an existing skill when it already solves the target, clone a related skill and adapt it when reuse is insufficient, or learn from a fresh initialization when prior knowledge is not useful. The experiments use small arithmetic regression tasks and matched deterministic seeds to separate acquisition reliability, acquisition efficiency, domain sensitivity, parent identity, and retention.

The central result is that prior knowledge is not uniformly beneficial: the observed transfer depends on the source-target relationship and on how the controller evaluates that relationship. The expanded fixed-target prerequisite design provides the current authoritative quantitative evidence: powers benefit from additional prior history, whereas division exhibits negative transfer and squares remains difficult. A three-arm matched parent-control experiment further shows that relevant parent identity matters beyond generic pretraining under the tested protocol. For multiplication, relevant cloning is faster than scratch (paired difference $-117.2$ steps, Holm-adjusted $p=0.012672$) and faster than an unrelated parent ($-298.3$ steps, adjusted $p=1.77\times10^{-5}$); for powers, the corresponding differences are $-260.5$ steps (adjusted $p=3.61\times10^{-10}$) and $-125.9$ steps (adjusted $p=4.03\times10^{-6}$). An unrelated parent is slower than scratch for multiplication ($+181.1$ steps, adjusted $p=0.00184$), but faster than scratch for powers ($-134.7$ steps, adjusted $p=1.20\times10^{-7}$), showing that generic pretraining can help while parent identity still matters. A signed-domain follow-up further shows distribution sensitivity. On the valid matched seeds, multiplication $\rightarrow$ powers changes from $2.217\times$ to $0.703\times$ and reverses direction ($p=0.00168$); multiplication $\rightarrow$ squares changes from $1.265\times$ to $1.031\times$ without a conventionally significant domain difference ($p=0.0757$); addition $\rightarrow$ subtraction changes from $0.408\times$ to $1.001\times$ ($p\approx1.88\times10^{-7}$); and the addition $\rightarrow$ multiplication null control changes from $1.145\times$ to $1.176\times$ with no statistically detectable domain difference ($p=0.591$). The powers and squares signed-domain comparisons have only 5/15 valid matched seeds because signed-domain multiplication prerequisite acquisition fails in 10 seeds.

The retention checks verify a narrower architectural property: under the tested isolated-skill mechanism, previously acquired skill parameters remain unchanged while later skills are acquired. This is an implementation/invariance check rather than evidence that catastrophic forgetting is impossible in continual-learning systems generally.

## 1. Introduction

A continual-learning system should be able to acquire a new capability while preserving capabilities that it has already learned. A simple shared-network strategy can update parameters for every new task, making the system vulnerable to interference. An alternative is to treat each learned capability as an independently stored skill and decide, for each incoming task, whether an existing skill should be reused, cloned and adapted, or whether learning should begin from scratch.

This work studies that mechanism as a controlled research prototype. The goal is not to claim a universal theory of prerequisites or a universal advantage for cloning. Instead, the experiments ask when previously acquired skills help, when they do not, whether parent identity contributes beyond generic pretraining, whether transfer depends on the task distribution, and whether the proposed isolated-skill mechanism preserves earlier skills during subsequent acquisition.

The research is organized around five questions:

1. Can a new target skill be acquired reliably?
2. Can an already-solved target be reused without additional training?
3. When reuse is not possible, does cloning provide a useful initialization compared with scratch learning?
4. Does the benefit of cloning depend on which previously acquired parent is selected?
5. Does transfer remain stable when the operand distribution is expanded to include signed values?

## 2. Experimental system

The prototype uses a small two-input, one-output neural network with a 32-unit hidden layer. The task family consists of addition, subtraction, multiplication, powers, squares, and division. All conditions use the same architecture, optimizer, learning rate, training procedure, convergence criterion, and fixed training budget unless explicitly stated otherwise for the signed-domain configuration.

The controller exposes three acquisition routes:

- **Reuse:** retain an existing skill unchanged when both the compatibility gate and an independent solved-target accuracy check are satisfied.
- **Clone + adapt:** deep-copy an existing skill and train the copy on the target task.
- **Scratch:** create a fresh network and train it on the target task.

The corrected reuse gate is important. A high frozen compatibility score alone is not sufficient evidence that the source skill already solves the target. The independent solve-accuracy check prevents a merely related skill from being incorrectly treated as a zero-training solution.

### 2.1 Mathematical formulation

Let the incoming target task be $T$ and let the repository contain previously acquired skills $s_i$, each represented by a parameter vector $\theta_i$. A skill maps an input $x$ to an output through $f(x;\theta_i)$.

For a target compatibility-probe set $D_T^{\mathrm{probe}}=\{(x_j,y_j)\}_{j=1}^{m}$, the frozen compatibility score used by the controller is the exponentially transformed mean squared error:

$$
\mathrm{MSE}(T,s_i)=\frac{1}{m}\sum_{j=1}^{m}
\left(f(x_j;\theta_i)-y_j\right)^2
$$

$$
P(T\mid s_i)=\exp\left(-\frac{\mathrm{MSE}(T,s_i)}{60}\right).
$$

The compatibility score is computed without updating $\theta_i$. It therefore measures how well the frozen parent already matches the target, rather than how useful its internal representation will necessarily be after adaptation.

The controller then evaluates the same frozen parent on a separate **solve/calibration batch** $D_T^{\mathrm{solve}}=\{(x_j,y_j)\}_{j=1}^{64}$, sampled independently from the compatibility probe. The target-solve accuracy is defined as the fraction of calibration examples whose absolute prediction error is at most the experiment's accuracy tolerance $\epsilon=0.5$:

$$
A(T,s_i)=\frac{1}{64}\sum_{j=1}^{64}
\mathbf{1}\!\left[
\left|f(x_j;\theta_i)-y_j\right|\leq0.5
\right].
$$

Thus $A(T,s_i)$ is an independently measured estimate of whether the frozen parent actually solves the target, rather than a relabeling derived from the compatibility score. The solve/calibration batch is separate from both the training data and the compatibility-probe batch, and the held-out test set is never consulted by the reuse decision. In the implementation, the calibration batch uses the independent solve-probe seed offset $20{,}000$ and the same task sampling procedure as the experiments.

With thresholds $\tau_{\mathrm{solve}}=0.90$ and $\tau_{\mathrm{clone}}=0.15$, the controller selects an action according to:

$$
\mathrm{action}(T,s_i)=
\begin{cases}
\mathrm{reuse} & \text{if } P(T\mid s_i)\geq\tau_{\mathrm{solve}} \text{ and } A(T,s_i)\geq0.85,\\
\mathrm{clone} & \text{if } P(T\mid s_i)\geq\tau_{\mathrm{clone}} \text{ and the reuse condition is not satisfied},\\
\mathrm{scratch} & \text{otherwise.}
\end{cases}
$$

When clone-and-adapt is selected from parent $s_i$, the target model is initialized from the parent's parameters:

$$
\theta_T^{(0)}=\theta_i.
$$

For scratch learning, $\theta_T^{(0)}$ is independently initialized. The adapted parameters are then obtained by minimizing the target training loss, represented here by mean squared error:

$$
\theta_T^*=\arg\min_{\theta}\;\frac{1}{n}\sum_{j=1}^{n}
\left(f(x_j;\theta)-y_j\right)^2.
$$

These equations define the mechanism evaluated in the experiments; they do not assume that a larger compatibility score must imply a larger transfer benefit.

### 2.2 Acquisition metrics

For a matched seed $r$, let $E_{\mathrm{scratch}}^{(r)}$ and $E_{\mathrm{clone}}^{(r)}$ denote the numbers of epochs required to reach the predeclared training criterion. The per-seed training-cost ratio is:

$$
S^{(r)}=\frac{E_{\mathrm{scratch}}^{(r)}}{E_{\mathrm{clone}}^{(r)}}.
$$

Thus $S>1$ favors clone-and-adapt, while $S<1$ indicates that cloning is slower than scratch. Training is capped at 1500 epochs. If an attempted target does not reach the acquisition criterion, its recorded epoch count is the full 1500-epoch budget. Consequently, this ratio is a **budget-capped training-cost comparison**, not a convergence-time comparison restricted to successful runs.

Acquisition reliability is kept separate from efficiency. Let $I_r=1$ when the target reaches the declared acquisition criterion within the allowed budget and $I_r=0$ otherwise. The empirical success rate is:

$$
R=\frac{1}{N}\sum_{r=1}^{N} I_r.
$$

This separation is important for difficult targets such as squares, where a method may have a low success rate even if successful runs can be compared for convergence speed.

### 2.3 Parent-control design

To distinguish useful parent identity from generic pretraining, a three-arm matched control compares a relevant clone, an unrelated previously acquired clone, and scratch initialization. For each target comparison, all three arms use the same target, seed, training data, architecture, optimizer, and budget. Only skills acquired before the target are eligible as parents.

The relevant parent is the highest-compatibility previously acquired skill under the controller's ranking. The unrelated parent is a different previously acquired skill. Both cloned arms are deep copies of their respective stored networks and are trained on the same target data as the scratch arm. The comparison therefore tests whether parent identity contributes to acquisition efficiency beyond the generic effect of starting from a pretrained network.

The inferential unit is the matched seed within target. Descriptive arm means are retained for context, but paired differences are the primary inferential summaries.

### 2.4 Retention and isolation invariant

For an acquired skill $s_i$, let $\theta_i^{\mathrm{pre}}$ and $\theta_i^{\mathrm{post}}$ denote its stored parameters immediately before and after a later skill is acquired. The intended isolated-skill invariant is:

$$
\theta_i^{\mathrm{post}}=\theta_i^{\mathrm{pre}}.
$$

Equivalently,

$$
\Delta\theta_i=\theta_i^{\mathrm{post}}-\theta_i^{\mathrm{pre}}=0.
$$

The retention experiment evaluates the same stored skill on the same stable retention set before and after later acquisition. Its diagnostic accuracy change is:

$$
\Delta A_i=A_{i,\mathrm{post}}-A_{i,\mathrm{pre}}.
$$

The predeclared practical diagnostic criterion is:

$$
\Delta A_i\geq-0.05.
$$

Because the stored parent is not optimized after acquisition and the evaluation set is unchanged, $\Delta A_i=0$ is expected under the mechanism. Therefore this quantity is reported as an implementation/invariance diagnostic, not as a statistical estimate of resistance to interference.

### Data separation

The experiment separates target-training data, compatibility-probe data, solve/calibration data, and held-out evaluation data. Held-out evaluation data are not used to choose thresholds, budgets, stopping rules, or model-selection decisions.

### Seeds

Experiments use matched deterministic seeds. The main relatedness, parent-control, and retention checks use 15 seeds per condition; the signed-domain follow-up also uses 15 nominal seeds per condition. Pairing by seed allows within-seed comparisons between acquisition strategies where applicable.

## 3. Research design

### 3.1 Historical relatedness analysis and authoritative fixed-target matrix

The initial source-target experiments examined multiplication $\rightarrow$ squares, multiplication $\rightarrow$ powers, addition $\rightarrow$ subtraction, and addition $\rightarrow$ multiplication. These results are retained as a **historical relatedness analysis** and are useful for showing that transfer can be heterogeneous.

The **expanded fixed-target prerequisite matrix is the authoritative current quantitative analysis**. It holds the target fixed and varies the prior-skill history across no prior skill, addition only, and addition + multiplication. The matrix includes subtraction, division, squares, and powers. This design distinguishes the effect of relevant prior knowledge from the mere presence of additional training history.

The interpretation is deliberately conservative. A curriculum order is not treated as proof that one task is a mathematical prerequisite for another. The experiment only tests whether a previously learned representation is useful for acquiring the target under the stated protocol.

A requested prerequisite must actually reach the acquisition criterion before it is exposed to the controller. If a prerequisite fails, the history is marked invalid, the failed skill is unavailable, and the target is not attempted for that seed.

### 3.2 Parent-identity control

The parent-control experiment is designed as a matched three-arm test of the claim that relevant cloning provides more than generic pretraining. The relevant and unrelated parent are both drawn from the set of skills already acquired before the target; no future task or target-trained representation is exposed to parent selection. For each seed, the relevant, unrelated, and scratch arms are trained independently on the same target data.

The experiment reports paired differences for relevant minus scratch, relevant minus unrelated, and unrelated minus scratch. Six paired hypotheses are tested across multiplication and powers. Holm's step-down correction is applied across these six paired $t$-test $p$-values. The manuscript reports the adjusted values alongside the raw paired results and does not treat arm means as independent observations.

This control is intentionally stronger than a comparison of clone versus scratch alone: if all pretrained networks helped equally, relevant and unrelated cloning would be expected to perform similarly. A difference between the two cloned parents therefore provides evidence that parent identity matters under the tested protocol.

### 3.3 Retention and catastrophic forgetting

The retention code performs sequential acquisition and re-evaluates every previously acquired skill after each later acquisition on a stable, skill-specific evaluation set. This verifies the intended isolation invariant: a stored parent skill is not modified when a later skill is trained as an independent copy.

For each retention check the implementation records pre/post accuracy, accuracy change, retention ratio, and whether the change remains within a predeclared five-percentage-point practical tolerance. These measurements are useful diagnostics of the invariant, but they should not be interpreted as a conventional statistical test of forgetting because the isolated-skill architecture does not expose the stored parent to subsequent optimization. In particular, repeated evaluation of an unchanged network on a stable evaluation set is expected to produce the same result.

A genuine empirical test of resistance to interference would require an at-risk comparison arm in which later learning can modify previously learned parameters, such as a shared-network baseline. That comparison is outside the scope of this PR and is not claimed here.

### 3.4 Signed-domain follow-up

The signed-domain experiment compares the original non-negative configuration with an explicitly configured signed configuration. Task ranges are task-specific: addition, subtraction, multiplication, and squares use operands from `[-9,9]`; powers use a signed base range with a non-negative exponent; division uses signed numerators and nonzero signed divisors. Inputs remain scaled by 10 and the same architecture, optimizer, training budget, controller, and seed protocol are used.

The comparison therefore tests domain/distribution sensitivity rather than isolating a causal effect of negative numbers alone.

## 4. Results

### 4.1 Historical relatedness-pair results

The earlier relatedness-pair experiment produced heterogeneous transfer. Multiplication $\rightarrow$ powers showed a mean paired speedup of approximately $2.31\times$, multiplication $\rightarrow$ squares approximately $1.26\times$, addition $\rightarrow$ subtraction approximately $0.33\times$, and addition $\rightarrow$ multiplication approximately $1.12\times$.

These observations are explicitly **historical relatedness results**. They are not mixed with the newer fixed-target matrix and should not be interpreted as the current controller's final quantitative summary. They also do not support a simple monotonic rule in which a larger frozen compatibility score always predicts a larger transfer benefit.

### 4.2 Authoritative fixed-target prerequisite matrix

The current fixed-target experiment provides the main quantitative evidence for how additional prior history affects acquisition. The authoritative values are:

| Target | No prior skill | Addition only | Addition + Multiplication |
|---|---:|---:|---:|
| Subtraction | 33.5 epochs | 62.3 epochs | 73.2 epochs |
| Division | 515.2 epochs | 616.2 epochs | **617.5 epochs** |
| Squares | 20.0% success | 20.0% success | 13.3% success |
| Powers | 471.6 epochs | 355.2 epochs | 237.3 epochs |

For subtraction, division, and powers, the values are acquisition epochs to the declared criterion, with an unsuccessful target after a valid prerequisite history contributing the full 1500-epoch budget. Invalid prerequisite histories are excluded because the target was never attempted. Squares is reported as success rate because many runs do not reach the acquisition criterion within the allowed budget.

The matrix shows heterogeneous transfer rather than a universal benefit from additional prior knowledge. Powers improves substantially as prior history expands, while division becomes slower and squares remains difficult. These results are evidence about transfer under the tested protocol, not proof of formal mathematical prerequisite relationships.

### 4.3 Parent-identity control results

The three-arm parent-control experiment provides the strongest current evidence that the effect of cloning is not explained solely by generic pretraining. All six paired $t$-test comparisons remain below 0.05 after Holm correction.

| Target | Comparison | Mean paired difference | Raw paired $p$ | Holm-adjusted $p$ |
|---|---|---:|---:|---:|
| multiplication | relevant − scratch | $-117.2$ steps | 0.012672 | 0.012672 |
| multiplication | relevant − unrelated | $-298.3$ steps | $5.91\times10^{-6}$ | $1.77\times10^{-5}$ |
| multiplication | unrelated − scratch | $+181.1$ steps | $9.18\times10^{-4}$ | $1.84\times10^{-3}$ |
| powers | relevant − scratch | $-260.5$ steps | $6.01\times10^{-11}$ | $3.61\times10^{-10}$ |
| powers | relevant − unrelated | $-125.9$ steps | $1.01\times10^{-6}$ | $4.03\times10^{-6}$ |
| powers | unrelated − scratch | $-134.7$ steps | $2.40\times10^{-8}$ | $1.20\times10^{-7}$ |

Negative differences favor the first-named arm because the outcome is convergence steps. For multiplication, relevant cloning is faster than both scratch and unrelated cloning, while unrelated cloning is slower than scratch. For powers, both pretrained arms outperform scratch, but the relevant parent is still significantly faster than the unrelated parent.

The pattern supports a narrower claim than "cloning helps": **parent identity matters under the tested protocol**. The powers result is especially informative because unrelated pretraining itself helps, showing that the relevant-parent advantage cannot be reduced to a simple pretrained-versus-untrained distinction. Conversely, the multiplication result shows that an unrelated parent can be actively worse than scratch in the same protocol. These findings do not imply that every relevant parent will outperform every unrelated parent on arbitrary tasks.

### 4.4 Retention mechanism check

The retention run reports zero change in the repeated pre/post checks and a 100% pass rate under the five-percentage-point practical tolerance. These values are consistent with the implementation invariant that previously acquired skills are stored independently and are not modified while a new skill is adapted.

Because the same unchanged skill is evaluated on the same skill-specific retention set before and after later acquisitions, the resulting zero change is **not an independent empirical estimate of protection against catastrophic forgetting**. It is a verification that the isolation mechanism and evaluation protocol behave as intended. We therefore do not report bootstrap confidence intervals or effect sizes for the retention delta as evidence of an interference effect.

### 4.5 Acquisition efficiency and reliability

Acquisition speed is treated as a secondary outcome. A small speedup is not automatically practically important, and reliability is evaluated separately from efficiency. A strong acquisition result is one in which prior knowledge increases fixed-budget success or substantially reduces training cost without sacrificing final held-out performance.

The fixed-target matrix is the primary current evidence for history-dependent acquisition behavior. The parent-control experiment adds evidence that the identity of the cloned parent matters beyond generic pretraining. The earlier relatedness-pair speedups remain useful contextual evidence, but they should not be interpreted as replacements for the expanded fixed-target or parent-control analyses.

### 4.6 Signed-domain transfer results

The current CI artifact gives the following paired domain comparisons. Pairing is performed only for seeds that have a valid source acquisition in both domains.

| Pair | Valid matched seeds | Non-negative | Signed | Difference (non-negative $-$ signed) | Paired $p$ |
|---|---:|---:|---:|---:|---:|
| multiplication $\rightarrow$ powers | 5/15 | $2.217\pm0.530$ | $0.703\pm0.141$ | $1.514\pm0.451$ | 0.00168 |
| multiplication $\rightarrow$ squares | 5/15 | $1.265\pm0.178$ | $1.031\pm0.069$ | $0.235\pm0.220$ | 0.0757 |
| addition $\rightarrow$ subtraction | 15/15 | $0.408\pm0.100$ | $1.001\pm0.224$ | $-0.594\pm0.243$ | $1.88\times10^{-7}$ |
| addition $\rightarrow$ multiplication (null control) | 15/15 | $1.145\pm0.138$ | $1.176\pm0.128$ | $-0.031\pm0.218$ | 0.591 |

The first two comparisons have only 5/15 valid matched seeds because signed-domain multiplication acquisition fails in 10 seeds. Those failures are explicitly reported and are not converted into artificial speedup observations.

**Multiplication $\rightarrow$ powers reverses direction.** The paired mean changes from $2.217\times$ to $0.703\times$ and the domain comparison is statistically different ($p=0.00168$). Sign-specific diagnostics show substantially higher error on negative-base odd-exponent cases. This is consistent with a harder sub-problem under the signed distribution, but does not prove the mechanism causally.

**Multiplication $\rightarrow$ squares moves toward no effect.** The paired mean changes from $1.265\times$ to $1.031\times$. The paired comparison is not conventionally statistically significant ($p=0.0757$), so the appropriate conclusion is erosion toward no effect rather than a demonstrated significant domain difference.

**Addition $\rightarrow$ subtraction neutralizes.** Negative transfer in the non-negative domain ($0.408\times$) becomes approximately neutral in the signed domain ($1.001\times$), with all 15 seeds valid.

**Addition $\rightarrow$ multiplication remains a null control.** The ratio changes from $1.145\times$ to $1.176\times$, with no statistically detectable domain difference ($p=0.591$). This does not prove equivalence; it means the tested paired comparison did not detect a statistically significant domain effect.

### 4.7 Signed-domain prerequisite reliability

The signed fixed-target experiment also reveals prerequisite attrition. Invalid histories are fail-closed: the failed prerequisite is not exposed to the controller and the target is not attempted.

Signed-domain squares has 0/15 target successes for all three tested histories. This establishes failure within the declared 1500-epoch budget for this protocol; it does not establish impossibility of learning signed squares in general.

### 4.8 Compatibility sensitivity

The controller's input is itself distribution-sensitive. For division with an addition parent, the frozen compatibility score is approximately 0.28--0.31 in the non-negative domain and approximately 0.03--0.06 in the signed domain. The non-negative values exceed the clone threshold while the signed values are below it, so controller behavior shifts toward scratch under the signed configuration.

## 5. Statistical analysis

For paired strategy comparisons, the unit of analysis is the matched seed. Reported summaries include the mean paired difference, variability, interval estimates, effect sizes, and paired significance tests where appropriate for the acquisition comparisons.

The three-arm parent-control analysis uses the same matched seed within each target for the relevant-clone, unrelated-clone, and scratch arms. The six paired $t$-test hypotheses across multiplication and powers form one multiplicity family. Holm's step-down procedure is applied to the six raw paired $p$-values. The resulting adjusted values are $0.012672$, $1.77\times10^{-5}$, $0.00184$, $3.61\times10^{-10}$, $4.03\times10^{-6}$, and $1.20\times10^{-7}$ in the order reported in Section 4.3. All six remain below 0.05. This correction limits the risk that the parent-control conclusion depends on an uncorrected collection of six tests.

For signed-domain comparisons, only seeds valid in both domains are included. The paired $t$-tests therefore describe changes in the budget-capped training-cost ratio among valid matched source acquisitions. They do not include the 10 multiplication-prerequisite failures in the powers and squares paired tests.

The retention check is intentionally treated differently. Its primary diagnostic quantity is:

$$
\Delta A_i=A_{i,\mathrm{post}}-A_{i,\mathrm{pre}}.
$$

The practical diagnostic rule used by the experiment is:

$$
\Delta A_i\geq-0.05.
$$

This five-percentage-point value is a declared practical tolerance. It is not a statistical equivalence margin and is not used to claim that the architecture has been shown equivalent in performance before and after acquisition.

Because the isolated parent network is not modified between the two evaluations, a zero retention delta is an expected consequence of the mechanism. Statistical inference on this delta would therefore not answer the stronger scientific question of whether a system resists interference when interference is possible.

## 6. Discussion

The combined findings support a simple design principle: a continual skill-acquisition system should not assume that every previous skill is useful, but it should preserve the option to exploit previous skills when they are useful.

The three-route controller is therefore important. Reuse is appropriate when an existing skill genuinely solves the target. Clone-and-adapt provides a way to exploit a useful initialization without modifying the parent. Scratch remains necessary because prior knowledge can be irrelevant or negatively transferable.

The new parent-control evidence sharpens this conclusion. Relevant cloning is faster than scratch for both multiplication and powers, but the comparison against an unrelated previously acquired parent shows that the effect is not explained solely by generic pretraining. For multiplication, the unrelated parent is actually slower than scratch, whereas for powers the unrelated parent still helps. Thus the tested results support the narrower claim that **parent identity matters under the tested protocol**, rather than the broader claim that any pretrained parent is beneficial.

The authoritative fixed-target results complement this control: Powers benefits from additional prior history, while Division exhibits negative transfer and Squares remains difficult. This heterogeneity is scientifically important because it prevents the paper from reducing the conclusion to "cloning always helps."

The signed-domain follow-up strengthens the same conclusion from a distribution-sensitivity perspective. The multiplication-to-powers comparison reverses from positive to negative transfer on the valid matched seeds, multiplication-to-squares moves toward no effect, and addition-to-subtraction neutralizes its earlier negative transfer. The addition-to-multiplication null control shows no statistically detectable domain difference. These observations concern the tested domain expansion and do not isolate negative values as the sole causal factor.

The retention checks provide implementation-level evidence that the independent-skill storage mechanism preserves stored parent parameters during later acquisition. They should not be confused with a comparative forgetting experiment. Demonstrating robustness to interference would require a condition in which later learning can actually alter parameters supporting earlier skills.

At the same time, the heterogeneous transfer results show that the harder scientific question is not simply whether cloning works. It is **why some skills benefit from prior knowledge while others do not**. The parent-control result makes source identity part of that question: useful transfer is not determined only by whether a network has been pretrained, but also by which previously acquired skill supplies the initialization. That question can form a follow-up study based on the present experimental framework.

## 7. Limitations and threats to validity

1. **Small task family.** The experiments use a small set of arithmetic functions and therefore cannot establish behavior across broad classes of machine-learning tasks.
2. **Small model.** Results may depend on the architecture, parameter count, optimizer, and training dynamics.
3. **Finite seeds.** Fifteen seeds provide useful matched comparisons but do not establish universal population-level behavior.
4. **Controller dependence.** The results depend on the compatibility probe, thresholds, and solved-target gate.
5. **Parent-control scope.** The relevant/unrelated/scratch control covers two targets and one small arithmetic family. It supports parent-identity dependence under the tested protocol, not a universal ranking rule for parents.
6. **Multiplicity and finite hypotheses.** Holm correction addresses the six parent-control tests in this analysis family, but the finite number of targets and hypotheses still limits generalization.
7. **Retention protocol.** The retention checks verify the tested isolation mechanism; they do not test an at-risk interference condition in which parent parameters can be overwritten.
8. **No universal prerequisite claim.** Earlier acquisition in a curriculum is evidence about transfer under that protocol, not proof of a formal prerequisite relationship.
9. **Potential task-family confounds.** Arithmetic tasks share representations and input structure, so transfer behavior may differ substantially in other domains.
10. **Signed-domain confounding.** The signed-domain manipulation changes the input distribution in addition to introducing negative values, so the conclusions concern domain/distribution sensitivity rather than negative numbers alone.
11. **Small signed-domain matched samples.** The powers and squares signed-domain pair tests use only 5/15 valid matched seeds, limiting precision and power.
12. **Budget-capped speedup.** The speedup ratio incorporates the 1500-epoch cap for unsuccessful target attempts and should not be described as convergence speed among successful runs.
13. **Acquisition failure interpretation.** Failure within the declared budget is not evidence that the task is impossible to learn.
14. **Practical tolerance.** The five-percentage-point retention tolerance is a declared diagnostic criterion, not a statistical equivalence margin derived from an external validation study.

## 8. Reproducibility

The repository contains the experiment drivers, regression tests, workflow configuration, raw CSV outputs, statistical summaries, and plots. CI reruns the experiments and uploads the generated artifacts. The retention experiment additionally records the exact seed, target sequence, acquisition strategy, source skill, compatibility diagnostics, pre/post accuracy, retention change, and retention decision for each measured run.

The parent-control experiment records per-seed arm outcomes and paired statistics for relevant, unrelated, and scratch arms. The final statistical summary applies Holm correction across the six parent-control paired tests.

The signed-domain experiment records per-seed pair validity, source-acquisition failures, domain-specific outcomes, paired statistics, history validity, and sign-specific diagnostics. The publication tables report the effective matched sample size for the paired domain tests.

The non-negative configuration remains the default. Signed-domain behavior is an explicit experimental configuration and does not silently alter the baseline task distribution.

## 9. Conclusion

This prototype demonstrates a controlled approach to continual skill acquisition in which the system can choose among reuse, clone-and-adapt, and scratch learning while preserving previously acquired skills through independent storage.

The most defensible conclusion is not that cloning always improves learning, nor that catastrophic forgetting has been eliminated. Instead, the authoritative fixed-target experiments show that prior history can have positive or negative effects depending on the target: powers benefit from additional prior skills, while division exhibits negative transfer and squares remains difficult. The matched parent-control experiment adds an important qualification: relevant cloning outperforms scratch and unrelated cloning for both multiplication and powers after Holm correction, while unrelated pretraining itself can either hurt or help depending on the target. Thus the benefit is not explained solely by generic pretraining; **parent identity matters under the tested protocol**. The signed-domain follow-up further shows that transfer can change when the operand distribution is expanded: multiplication-to-powers reverses direction, multiplication-to-squares erodes toward no effect without a conventionally significant paired domain difference, and addition-to-subtraction neutralizes its earlier negative transfer. The addition-to-multiplication null control shows no statistically detectable domain difference under the tested manipulation.

The isolated-skill mechanism preserves previously stored skills during later acquisition as an implementation invariant under the tested conditions. This is an architectural property rather than a demonstration that a system with interference-capable parameters would resist catastrophic forgetting.

The work therefore establishes a useful experimental foundation for a broader research program: first characterize reliable skill acquisition and the conditions under which transfer helps or hurts, then test interference resistance using explicit at-risk baselines and broader task families.

## 10. Reproducibility checklist

- [x] Reuse / clone / scratch routes implemented.
- [x] Corrected reuse gate requires independent solved-target evidence.
- [x] Genuine scratch fallback retained.
- [x] Fail-closed prerequisite semantics implemented.
- [x] Matched seeds used for paired comparisons.
- [x] Three-arm relevant/unrelated/scratch parent-control implemented.
- [x] Parent-control paired statistics reported with Holm correction.
- [x] Held-out evaluation separated from training and controller data.
- [x] Retention evaluated on stable skill-specific evaluation sets.
- [x] Retention tolerance declared before interpretation.
- [x] Per-seed retention data and summary outputs generated.
- [x] Signed-domain configuration is explicit and tested.
- [x] Signed-domain source-acquisition attrition is reported.
- [x] Signed-domain paired statistics are reported with effective sample sizes.
- [x] CI executes the full experiment suite and regression tests.
- [x] Generated artifacts are uploaded by CI.
- [x] Retention claims explicitly separated from statistical evidence of interference resistance.
- [ ] An at-risk shared-network retention comparison is outside the scope of this PR and should be added only as a separate, explicitly controlled experiment.
