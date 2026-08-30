# Continual Skill Acquisition via Reuse, Clone-and-Adapt, and Scratch Learning

## Abstract

Continual acquisition systems must learn new skills without unnecessarily discarding useful prior knowledge. This study evaluates a small, reproducible skill-acquisition framework in which an incoming task is handled by one of three routes: reuse an existing skill when it already solves the target, clone a related skill and adapt it when reuse is insufficient, or learn from a fresh initialization when prior knowledge is not useful. The experiments use small arithmetic regression tasks and matched random seeds to separate acquisition reliability, acquisition efficiency, transfer, and the preservation of previously acquired skills.

The central result is that prior knowledge is not uniformly beneficial: observed transfer depends on the source-target relationship and on how the controller evaluates that relationship. Earlier experiments showed both positive and negative transfer, motivating a fixed-target prerequisite design rather than a simple ranking of source-target pairs. A signed-domain follow-up further shows that transfer behavior can be distribution-sensitive: expanding the operand domain from non-negative to signed integers reversed the direction of the strongest transfer result (multiplication → powers, 2.22× → 0.70× on the valid matched seeds) and moved another toward no effect (multiplication → squares, 1.27× → 1.03×). The addition → subtraction effect also moved from negative transfer toward neutral, while no statistically detectable domain difference was observed for the addition → multiplication null control. The signed-domain powers and squares comparisons have only 5/15 valid matched seeds because signed-domain multiplication prerequisite acquisition failed in 10 seeds; this attrition is reported explicitly rather than hidden. A separate skill-isolation check confirmed that previously acquired skills' stored parameters are not modified by later acquisitions. This is an implementation invariant of the current architecture, not a statistical finding about resistance to catastrophic forgetting.

## 1. Introduction

A continual-learning system should be able to acquire a new capability while preserving capabilities that it has already learned. A simple shared-network strategy can update parameters for every new task, making the system vulnerable to interference. An alternative is to treat each learned capability as an independently stored skill and decide, for each incoming task, whether an existing skill should be reused, cloned and adapted, or whether learning should begin from scratch.

This work studies that mechanism as a controlled research prototype. The goal is not to claim a universal theory of prerequisites or a universal advantage for cloning. Instead, the experiments ask when previously acquired skills help, when they do not, whether transfer depends on the task distribution, and whether the isolated-skill mechanism preserves earlier skills during subsequent acquisition.

The research is organized around four questions:

1. Can a new target skill be acquired reliably?
2. Can an already-solved target be reused without additional training?
3. When reuse is not possible, does cloning provide a useful initialization compared with scratch learning?
4. Does the isolated-skill mechanism preserve previously acquired skills during later acquisition?

## 2. Experimental system

The prototype uses a small two-input, one-output neural network with a 32-unit hidden layer. The task family consists of addition, subtraction, multiplication, powers, and squares. All conditions use the same architecture, optimizer, learning rate, training procedure, convergence criterion, and fixed training budget.

The controller exposes three acquisition routes:

* **Reuse:** retain an existing skill unchanged when both the compatibility gate and an independent solved-target accuracy check are satisfied.
* **Clone + adapt:** deep-copy an existing skill and train the copy on the target task.
* **Scratch:** create a fresh network and train it on the target task.

The corrected reuse gate is important. A high frozen compatibility score alone is not sufficient evidence that the source skill already solves the target. The independent solve-accuracy check prevents a merely related skill from being incorrectly treated as a zero-training solution.

### 2.1 Mathematical formulation

Let the incoming target task be \(T\) and let the repository contain previously acquired skills \(s_i\), each represented by a parameter vector \(\theta_i\). A skill maps an input \(x\) to an output through \(f(x;\theta_i)\).

For a target probe set \(D_T^{\mathrm{probe}}=\{(x_j,y_j)\}_{j=1}^{m}\), the frozen compatibility score used by the controller is the exponentially transformed mean squared error:

$$
\mathrm{MSE}(T,s_i)=\frac{1}{m}\sum_{j=1}^{m}
\left(f(x_j;\theta_i)-y_j\right)^2
$$

$$
P(T\mid s_i)=
\exp\left(-\frac{\mathrm{MSE}(T,s_i)}{60}\right).
$$

The compatibility score is computed without updating \(\theta_i\). It therefore measures how well the frozen parent already matches the target, rather than how useful its internal representation will necessarily be after adaptation.

Let \(A(T,s_i)\) denote the independent target-solve accuracy used by the corrected reuse gate. With thresholds \(\tau_{\mathrm{solve}}=0.90\) and \(\tau_{\mathrm{clone}}=0.15\), the controller selects an action according to:

$$
\mathrm{action}(T,s_i)=
\begin{cases}
\mathrm{reuse} &
\text{if } P(T\mid s_i)\geq\tau_{\mathrm{solve}}
\text{ and } A(T,s_i)\geq 0.85,\\
\mathrm{clone} &
\text{if } P(T\mid s_i)\geq\tau_{\mathrm{clone}}
\text{ and the reuse condition is not satisfied},\\
\mathrm{scratch} &
\text{otherwise.}
\end{cases}
$$

When clone-and-adapt is selected from parent \(s_i\), the target model is initialized from the parent's parameters:

$$
\theta_T^{(0)}=\theta_i.
$$

For scratch learning, \(\theta_T^{(0)}\) is independently initialized. The adapted parameters are then obtained by minimizing the target training loss, represented here by mean squared error:

$$
\theta_T^*=
\arg\min_{\theta}\;
\frac{1}{n}\sum_{j=1}^{n}
\left(f(x_j;\theta)-y_j\right)^2.
$$

These equations define the mechanism evaluated in the experiments; they do not assume that a larger compatibility score must imply a larger transfer benefit.

### 2.2 Acquisition metrics

For a matched seed \(r\), let \(E_{\mathrm{scratch}}^{(r)}\) and \(E_{\mathrm{clone}}^{(r)}\) denote the numbers of epochs required to reach the predeclared training criterion. The per-seed convergence speedup is:

$$
S^{(r)}=
\frac{E_{\mathrm{scratch}}^{(r)}}
{E_{\mathrm{clone}}^{(r)}}.
$$

Thus \(S>1\) favors clone-and-adapt, while \(S<1\) indicates that cloning is slower than scratch. The reported paired comparisons treat the matched seed as the unit of analysis.

Acquisition reliability is kept separate from efficiency. Let \(I_r=1\) when the target reaches the declared acquisition criterion within the allowed budget and \(I_r=0\) otherwise. The empirical success rate is:

$$
R=\frac{1}{N}\sum_{r=1}^{N} I_r.
$$

This separation is important for difficult targets such as squares, where a method may have a low success rate even if successful runs can be compared for convergence speed.

### 2.3 Retention and isolation invariant

For an acquired skill \(s_i\), let \(\theta_i^{\mathrm{pre}}\) and \(\theta_i^{\mathrm{post}}\) denote its stored parameters immediately before and after a later skill is acquired. The intended isolated-skill invariant is:

$$
\theta_i^{\mathrm{post}}=\theta_i^{\mathrm{pre}}.
$$

or equivalently,

$$
\Delta\theta_i=
\theta_i^{\mathrm{post}}-\theta_i^{\mathrm{pre}}=0.
$$

The retention check evaluates the same stored skill on the same stable retention set before and after later acquisition. Its diagnostic accuracy change is:

$$
\Delta A_i=
A_{i,\mathrm{post}}-A_{i,\mathrm{pre}}.
$$

The predeclared practical diagnostic criterion is:

$$
\Delta A_i\geq-0.05.
$$

Because the stored parent is not optimized after acquisition and the evaluation set is unchanged, \(\Delta A_i=0\) is expected under the mechanism. Therefore this quantity is reported as an implementation/invariance diagnostic, not as a statistical estimate of resistance to interference.

### 2.4 Data separation

The experiment separates target-training data, compatibility-probe data, solve/accuracy data, and held-out evaluation data. Held-out evaluation data are not used to choose thresholds, budgets, stopping rules, or model-selection decisions.

### 2.5 Seeds

Experiments use matched deterministic seeds. The main relatedness, signed-domain, and retention experiments use 15 seeds per condition. Pairing by seed allows within-seed comparisons between acquisition strategies where applicable.

## 3. Research design

### 3.1 Historical relatedness analysis and fixed-target prerequisite matrix

The initial source-target experiments examined multiplication → squares, multiplication → powers, addition → subtraction, and addition → multiplication. These results are retained as a **historical relatedness analysis** and are useful for showing that transfer can be heterogeneous.

The **expanded fixed-target prerequisite matrix is the authoritative current quantitative analysis**. It holds the target fixed and varies the prior-skill history across no prior skill, addition only, and addition + multiplication. The matrix includes subtraction, division, squares, and powers. This design distinguishes the effect of relevant prior knowledge from the mere presence of additional training history.

The interpretation is deliberately conservative. A curriculum order is not treated as proof that one task is a mathematical prerequisite for another. The experiment only tests whether a previously learned representation is useful for acquiring the target under the stated protocol.

### 3.2 Skill-isolation invariant check

A separate check evaluates sequential acquisition. After each new skill is acquired, every previously acquired skill is evaluated again on its own stable, skill-specific evaluation set.

Unlike the transfer experiments, this check does not provide statistical evidence about resistance to interference. In the current architecture, a `Skill`'s network is never modified once training on it stops. There is no code path through which a later acquisition's gradient updates can reach an already-stored skill.

Because the pre- and post-acquisition evaluations use the identical frozen network on the identical deterministic evaluation batch, the recorded change is mathematically expected to be exactly zero. Consequently, a confidence interval, standard deviation, or effect size over this quantity would not represent meaningful sampling uncertainty.

What the check legitimately establishes is narrower: the isolation guarantee holds in the implementation, with no observed code path allowing later acquisition to modify a supposedly frozen skill. This is a useful regression property, but it is not evidence that catastrophic forgetting is absent in continual-learning systems generally.

A genuine empirical forgetting experiment would require an at-risk comparison arm in which later learning can modify parameters supporting earlier skills, such as a shared-network baseline.

## 4. Results

### 4.1 Historical relatedness-pair results

The earlier relatedness-pair experiment produced heterogeneous transfer. Multiplication → powers showed a mean paired speedup of approximately 2.31×, multiplication → squares approximately 1.26×, addition → subtraction approximately 0.33×, and addition → multiplication approximately 1.12×.

These observations are explicitly **historical relatedness results**. They are not mixed with the newer fixed-target matrix and should not be interpreted as the current controller's final quantitative summary.

They also do not support a simple monotonic rule in which a larger frozen compatibility score always predicts a larger transfer benefit. A frozen output-compatibility measure and usefulness as a parameter initialization are not necessarily the same property.

### 4.2 Authoritative fixed-target prerequisite matrix

The current fixed-target experiment provides the main quantitative evidence for how additional prior history affects acquisition.

| Target      | No prior skill | Addition only | Addition + Multiplication |
| ----------- | -------------: | ------------: | ------------------------: |
| Subtraction |    33.5 epochs |   62.3 epochs |               73.2 epochs |
| Division    |   515.2 epochs |  616.2 epochs |              617.5 epochs |
| Squares     |  20.0% success | 20.0% success |             13.3% success |
| Powers      |   471.6 epochs |  355.2 epochs |              237.3 epochs |

For subtraction, division, and powers, the values are acquisition epochs to the declared criterion. Squares is reported as success rate because many runs do not reach the acquisition criterion within the allowed budget.

The matrix shows heterogeneous transfer rather than a universal benefit from additional prior knowledge. Powers improves substantially as prior history expands, while division becomes slower and squares remains difficult. These results support the conclusion that transfer depends on the source-target relationship and that additional prior skills can introduce negative transfer.

These results are evidence about transfer under the tested protocol, not proof of formal mathematical prerequisite relationships.

### 4.3 Skill-isolation invariant check

The retention run reports zero change in the repeated pre/post checks and a 100% pass rate under the five-percentage-point practical tolerance. These values are consistent with the implementation invariant that previously acquired skills are stored independently and are not modified while a new skill is adapted.

Because the same unchanged skill is evaluated on the same skill-specific retention set before and after later acquisitions, the resulting zero change is **not an independent empirical estimate of protection against catastrophic forgetting**. It is a verification that the isolation mechanism and evaluation protocol behave as intended.

The correct interpretation is therefore narrow: **the implementation's skill-isolation guarantee holds**. This is a useful regression property because it rules out a class of implementation bugs in which later training accidentally modifies a supposedly frozen skill.

### 4.4 Domain-sensitivity analysis: signed-domain follow-up

**Research question:** does the transfer behavior observed in Section 4.1 hold when the arithmetic operand domain is expanded from non-negative integers to include negative values, or is it sensitive to the specific input distribution used by the original experiments?

This follow-up reruns the four relatedness pairs from Section 4.1 under matched seeds in two domains: the original non-negative domain (`{0,...,9}` for most tasks) and a signed domain (`{-9,...,9}`). The powers exponent remains non-negative to avoid confounding operand sign with fractional targets, and division's divisor is nonzero by construction in both domains.

Only the operand domain changes. Architecture, optimizer, learning rate, stopping criterion, training budget, seed protocol, compatibility calculation, and data-role separation are otherwise held fixed.

The experiment's current non-negative-domain rerun gives mean speedups of 2.217×, 1.265×, 0.408×, and 1.145× for multiplication → powers, multiplication → squares, addition → subtraction, and addition → multiplication, respectively. These values are the current artifact's statistics rather than the older rounded values used in the earlier draft.

**Result: transfer is not uniformly robust to this domain expansion.**

| Pair | Valid matched seeds | Non-negative speedup | Signed speedup | Paired difference (non-negative − signed) | Paired t-test p | Direction reversed? |
| ---------------------------------------- | --------------: | -------------------: | -------------: | ----------------------------------------: | --------------: | ---------------------------------- |
| multiplication → powers | 5/15 | 2.217 ± 0.530 | 0.703 ± 0.141 | 1.514 ± 0.451 | 0.00168 | **Yes** |
| multiplication → squares | 5/15 | 1.265 ± 0.178 | 1.031 ± 0.069 | 0.235 ± 0.220 | 0.0757 | No; erodes toward null |
| addition → subtraction | 15/15 | 0.408 ± 0.100 | 1.001 ± 0.224 | −0.594 ± 0.243 | 1.88×10⁻⁷ | Yes; negative transfer neutralizes |
| addition → multiplication (null control) | 15/15 | 1.145 ± 0.138 | 1.176 ± 0.128 | −0.031 ± 0.218 | 0.591 | No |

The powers and squares comparisons have only **5/15 valid matched seeds**. In the signed domain, the multiplication prerequisite failed acquisition in 10 of 15 seeds for each pair, so only five seeds had a valid source skill and a corresponding clone/scratch comparison. Those failures are not silently converted into speedup values and are not treated as successful matched observations. The reduced paired sample size is therefore part of the result and must accompany the corresponding p-values.

**Multiplication → powers reverses direction.** A positive transfer effect in the non-negative domain (2.217×) becomes negative transfer in the signed domain (0.703×) on the five valid matched seeds. The paired comparison remains statistically different (`p=0.00168`). A sign-specific diagnostic breakdown of the trained clone's held-out error provides a plausible, but not proven, mechanism: negative-base/odd-exponent inputs, which do not occur in the non-negative domain, have substantially higher error than positive-base inputs. The signed domain therefore introduces a harder sub-problem that the multiplication-derived initialization appears poorly suited to under this protocol.

**Multiplication → squares erodes toward no effect.** The observed speedup changes from 1.265× to 1.031× on the five valid matched seeds. The current paired test is **not conventionally statistically significant** (`p=0.0757`). The appropriate conclusion is therefore that the observed positive transfer erodes toward no effect; the experiment does not establish a statistically significant domain difference for this pair. The sign-specific diagnostic shows essentially symmetric error for positive- and negative-base inputs (0.486 versus 0.489 mean absolute error), which is consistent with the sign-invariance of squares, \((-a)^2=a^2\).

**Addition → subtraction's negative transfer neutralizes.** The negative transfer observed in the non-negative domain (0.408×) becomes approximately neutral in the signed domain (1.001×), with all 15 seeds valid. The sign breakdown shows that same-sign subtraction pairs are learned more accurately than mixed-sign pairs, consistent with same-sign subtraction sharing some structure with the addition parent while mixed-sign subtraction introduces a different input relationship.

**Addition → multiplication remains a null control.** The speedup changes from 1.145× to 1.176×, with no statistically detectable difference between domains (`p=0.591`). No statistically detectable domain difference was observed for this negative-control pair, while the other three pairs change qualitatively. This supports the narrower interpretation that the observed changes are not simply a generic effect of the domain manipulation on every pair.

### Acquisition reliability under the signed domain

The fixed-target prerequisite-history matrix was also evaluated under both domains. Squares' success rate in the non-negative domain is 20.0%, 20.0%, and 13.3% across the three prior histories, and it falls to exactly 0.0% in the corresponding signed-domain histories.

Subtraction and powers have 100% target-acquisition success in all valid signed-domain histories. Division also has 100% target success in every history that reaches the target attempt. Some signed-domain histories fail earlier because a requested prerequisite is not acquired; those histories are marked invalid and the failed prerequisite is never exposed to the controller.

This is a reliability failure in addition to the transfer-efficiency changes. It should not be interpreted as evidence that signed inputs universally make squares difficult; it is evidence that this particular system and training protocol failed to acquire the signed-domain square task within the declared budget.

### Compatibility and controller behavior

The compatibility score itself is domain-sensitive, which has a direct behavioral consequence for the controller.

For division with an addition parent, compatibility falls from approximately 0.28–0.31 in the non-negative domain to approximately 0.03–0.06 in the signed domain. The former is above the clone threshold \(\tau_{\mathrm{clone}}=0.15\), whereas the latter is below it. Consequently, the controller switches from cloning toward scratch in 14–15 of 15 signed-domain seeds.

This is consistent with the intended controller behavior: it does not blindly preserve a source-target decision when the evidence used by the controller changes under the new input distribution.

### Interpretation

The signed-domain follow-up establishes that transfer **can be domain-sensitive** for this system under the tested expansion. It does not establish that transfer is always domain-sensitive, nor that negative numbers alone are responsible for the observed changes.

The manipulation expands `{0,...,9}`-scale ranges to `{-9,...,9}`-scale ranges and therefore changes the input distribution in more than one way. The result should consequently be interpreted as evidence of **distribution sensitivity**, rather than as a distribution-independent causal effect of negative numbers.

The null-control result strengthens this interpretation: no statistically detectable domain difference was observed for addition → multiplication while the other three pairs change qualitatively. Nevertheless, the conclusions remain conditional on this task family, architecture, controller, and specific domain expansion.

### 4.5 Acquisition efficiency and reliability

Acquisition speed is treated as a secondary outcome. A small speedup is not automatically practically important, and reliability is evaluated separately from efficiency.

A strong acquisition result is one in which prior knowledge increases fixed-budget success or substantially reduces training cost without sacrificing final held-out performance.

The fixed-target matrix is the primary current evidence for history-dependent acquisition behavior. The historical relatedness-pair speedups remain useful contextual evidence, while the signed-domain follow-up demonstrates that those transfer effects can change under a distribution shift.

## 5. Statistical analysis

For paired strategy comparisons, the unit of analysis is the matched seed. Reported summaries include the mean paired difference, variability, interval estimates, effect sizes, and paired significance tests where appropriate for acquisition comparisons.

For the signed-domain comparisons, each seed is paired across the non-negative and signed conditions **only when both conditions yield a valid source acquisition and therefore a valid clone/scratch pair**. The paired test therefore evaluates within-seed changes without treating prerequisite failures as artificial speedup observations. The effective paired sample size is reported explicitly for each pair.

The retention check is intentionally treated differently. Its primary diagnostic quantity is:

$$
\Delta A_i=
A_{i,\mathrm{post}}-A_{i,\mathrm{pre}},
$$

where accuracy is measured on the same retention set for the same skill and seed. The practical diagnostic rule is:

$$
\Delta A_i\geq-0.05.
$$

This five-percentage-point value is a declared practical tolerance. It is not a statistical equivalence margin derived from an external validation study.

Because the isolated parent network is not modified between the two evaluations, a zero retention delta is an expected consequence of the mechanism. Statistical inference on this delta therefore does not answer the stronger scientific question of whether a system resists interference when interference is possible. A genuine empirical forgetting study would require an at-risk baseline in which later learning can alter parameters supporting earlier skills.

Success rate is reported separately from convergence speed. If all conditions reach the fixed budget successfully, training-cost and convergence distributions become the more informative secondary outcomes.

## 6. Discussion

The combined findings support a simple design principle: a continual skill-acquisition system should not assume that every previous skill is useful, but it should preserve the option to exploit previous skills when they are useful.

The three-route controller is therefore important. Reuse is appropriate when an existing skill genuinely solves the target. Clone-and-adapt provides a way to exploit a useful initialization without modifying the parent. Scratch remains necessary because prior knowledge can be irrelevant or negatively transferable.

The authoritative fixed-target results strengthen this interpretation: powers benefit from additional prior history, while division exhibits negative transfer and squares remains difficult. These heterogeneous outcomes prevent the conclusion from being reduced to "cloning always helps."

The signed-domain follow-up adds an important qualification. Transfer benefits are not fixed properties of source-target task labels alone. Under the tested distribution expansion, multiplication → powers reverses direction (2.217× to 0.703× on the valid matched seeds), multiplication → squares approaches no effect (1.265× to 1.031×, with `p=0.0757`), and addition → subtraction's negative transfer disappears (0.408× to 1.001×), while no statistically detectable domain difference was observed for the null control (1.145× to 1.176×, `p=0.591`). The powers and squares paired comparisons use only 5 of 15 seeds because signed-domain multiplication prerequisite acquisition failed for 10 seeds; this limits the strength of those domain-comparison conclusions and is itself a reproducibility-relevant result.

This suggests that the usefulness of a cloned representation depends jointly on the source-target relationship and the distribution on which that relationship is evaluated.

The skill-isolation invariant check supports the architectural design rationale: independently storing skills and avoiding updates to stored parent parameters provides a direct mechanism for accumulating capabilities without overwriting earlier skills. This is a design-level guarantee confirmed by the implementation check, not an empirical demonstration that catastrophic forgetting is absent from continual-learning systems generally.

At the same time, the heterogeneous transfer results show that the harder scientific question is not simply whether cloning works. It is **why some skills benefit from prior knowledge while others do not**, and why the same source-target relationship can behave differently under a changed input distribution. That question provides a natural direction for future work.

## 7. Limitations and threats to validity

1. **Small task family.** The experiments use a small set of arithmetic functions and therefore cannot establish behavior across broad classes of machine-learning tasks.

2. **Small model.** Results may depend on the architecture, parameter count, optimizer, and training dynamics.

3. **Finite seeds.** Fifteen seeds provide useful matched comparisons but do not establish universal population-level behavior.

4. **Controller dependence.** The results depend on the compatibility probe, thresholds, and solved-target gate.

5. **The skill-isolation check is an implementation invariant, not an empirical retention result.** Under the tested architecture, a stored skill's parameters cannot change once acquired. The zero-delta outcome therefore confirms the isolation implementation rather than providing statistical evidence about resistance to interference.

6. **No universal prerequisite claim.** Earlier acquisition in a curriculum is evidence about transfer under that protocol, not proof of a formal prerequisite relationship.

7. **Potential task-family confounds.** Arithmetic tasks share representations and input structure, so transfer behavior may differ substantially in other domains.

8. **Practical tolerance.** The five-percentage-point tolerance in the isolation check is a sanity bound, not a statistical equivalence margin. A violation would indicate an implementation problem rather than establish a population-level effect.

9. **Specific signed-domain expansion.** The signed-domain follow-up tests one particular expansion from `{0,...,9}`-scale ranges to `{-9,...,9}`-scale ranges on the tested arithmetic tasks. It demonstrates that transfer can be domain-sensitive for this system, not that negative numbers universally cause transfer changes or that the result generalizes to arbitrary distribution shifts.

10. **Distribution shift versus sign effect.** Expanding the domain changes the input distribution in ways beyond simply introducing negative values. The experiment therefore supports a claim about domain/distribution sensitivity rather than an isolated causal claim about the presence of negative numbers.

11. **Matched-pair attrition.** The signed-domain multiplication→powers and multiplication→squares comparisons have only 5/15 valid matched seeds because signed-domain multiplication prerequisite acquisition fails in 10 seeds. The resulting p-values describe the five valid matched pairs, not all 15 nominal seeds. This reduces statistical power and makes the signed-domain result less precise for those pairs.

12. **Acquisition reliability.** The fixed training budget creates a distinction between failure to acquire a skill within the budget and impossibility of acquiring that skill. The reported success rates should therefore be interpreted relative to the declared budget and stopping criterion.

## 8. Reproducibility

The repository contains the experiment drivers, regression tests, workflow configuration, raw CSV outputs, statistical summaries, and plots. CI reruns the experiments and uploads generated artifacts.

The skill-isolation check records the seed, target sequence, acquisition strategy, source skill, compatibility diagnostics, pre/post accuracy, recorded accuracy change, and pass/fail decision.

The signed-domain follow-up records both domain conditions under matched seeds and includes per-seed pair results, summary statistics, sign-specific diagnostic breakdowns, and generated plots. The publication tables report the effective valid matched-seed count alongside the paired statistics.

The non-negative domain remains the default task configuration. The signed domain is an explicit experimental configuration and does not silently alter the baseline task distribution.

## 9. Conclusion

This prototype demonstrates a controlled approach to continual skill acquisition in which the system can choose among reuse, clone-and-adapt, and scratch learning while preserving previously acquired skills through independent storage.

The most defensible conclusion is not that cloning always improves learning. Instead, the experiments show that prior knowledge can have positive, negligible, or negative effects depending on the target and source relationship, and that these transfer effects can themselves be sensitive to the input distribution. The signed-domain follow-up demonstrates this sensitivity directly: multiplication → powers reverses from positive to negative transfer, multiplication → squares moves toward no effect without a conventionally significant paired domain difference, and addition → subtraction's negative transfer neutralizes, while no statistically detectable domain difference was observed for the null control. The strongest signed-domain pair comparisons are limited by prerequisite-acquisition attrition, with only 5/15 valid matched seeds for multiplication→powers and multiplication→squares.

The isolated-skill mechanism provides a separate implementation-level guarantee: previously stored skills are not modified during later acquisition under the tested architecture. This is a useful architectural property, but it should not be confused with an empirical demonstration of resistance to catastrophic forgetting.

The work therefore establishes an experimental foundation for a broader research program: characterize reliable skill acquisition, understand when and why transfer helps or hurts, study how transfer depends on task distributions, and eventually evaluate retention using explicit at-risk interference baselines and broader task families.

## 10. Reproducibility checklist

* [x] Reuse / clone / scratch routes implemented.
* [x] Corrected reuse gate requires independent solved-target evidence.
* [x] Genuine scratch fallback retained.
* [x] Matched seeds used for paired comparisons.
* [x] Held-out evaluation separated from training and controller data.
* [x] Skill-isolation invariant evaluated on stable skill-specific evaluation sets.
* [x] Isolation-check tolerance declared before interpretation as a sanity bound.
* [x] Per-seed isolation-check data and summary outputs generated.
* [x] CI executes the skill-isolation check and regression tests.
* [x] Isolation-check results are described as an implementation invariant rather than a statistical retention experiment.
* [x] Historical relatedness-pair results retained separately from the authoritative fixed-target matrix.
* [x] Fixed-target prerequisite-history analysis included as the current quantitative analysis.
* [x] Signed-domain follow-up added as an explicit experimental configuration.
* [x] Non-negative baseline preserved as the default domain.
* [x] Signed-domain comparisons use matched seeds.
* [x] Signed-domain analysis holds architecture, optimizer, training budget, controller, thresholds, and data-role separation fixed.
* [x] Signed-domain results include paired statistical comparisons.
* [x] Signed-domain publication tables report valid matched-seed counts and prerequisite-acquisition attrition.
* [x] Signed-domain conclusions are explicitly limited to the tested domain/distribution expansion.
* [ ] A genuine at-risk empirical retention experiment with a shared-parameter interference baseline remains future work.
* [ ] Broader task families and larger models remain future validation targets.
