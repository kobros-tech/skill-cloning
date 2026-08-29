# Continual Skill Acquisition via Reuse, Clone-and-Adapt, and Scratch Learning

## Abstract

Continual acquisition systems must learn new skills without unnecessarily discarding useful prior knowledge. This study evaluates a small, reproducible skill-acquisition framework in which an incoming task is handled by one of three routes: reuse an existing skill when it already solves the target, clone a related skill and adapt it when reuse is insufficient, or learn from a fresh initialization when prior knowledge is not useful. The experiments use small arithmetic regression tasks and matched random seeds to separate acquisition reliability, acquisition efficiency, and retention.

The central result is that prior knowledge is not uniformly beneficial: the observed transfer depends on the source-target relationship and on how the controller evaluates that relationship. Earlier experiments showed both substantial positive transfer and negative transfer, motivating a fixed-target prerequisite design rather than a simple ranking of source-target pairs. The retention checks verify a narrower architectural property: under the tested isolated-skill mechanism, previously acquired skill parameters remain unchanged while later skills are acquired. This is an implementation/invariance check rather than evidence that catastrophic forgetting is impossible in continual-learning systems generally.

## 1. Introduction

A continual-learning system should be able to acquire a new capability while preserving capabilities that it has already learned. A simple shared-network strategy can update parameters for every new task, making the system vulnerable to interference. An alternative is to treat each learned capability as an independently stored skill and decide, for each incoming task, whether an existing skill should be reused, cloned and adapted, or whether learning should begin from scratch.

This work studies that mechanism as a controlled research prototype. The goal is not to claim a universal theory of prerequisites or a universal advantage for cloning. Instead, the experiments ask when previously acquired skills help, when they do not, and whether the proposed isolated-skill mechanism preserves earlier skills during subsequent acquisition.

The research is organized around four questions:

1. Can a new target skill be acquired reliably?
2. Can an already-solved target be reused without additional training?
3. When reuse is not possible, does cloning provide a useful initialization compared with scratch learning?
4. Does the isolated-skill mechanism preserve previously acquired skills during later acquisition?

## 2. Experimental system

The prototype uses a small two-input, one-output neural network with a 32-unit hidden layer. The task family consists of addition, subtraction, multiplication, powers, and squares. All conditions use the same architecture, optimizer, learning rate, training procedure, convergence criterion, and fixed training budget.

The controller exposes three acquisition routes:

- **Reuse:** retain an existing skill unchanged when both the compatibility gate and an independent solved-target accuracy check are satisfied.
- **Clone + adapt:** deep-copy an existing skill and train the copy on the target task.
- **Scratch:** create a fresh network and train it on the target task.

The corrected reuse gate is important. A high frozen compatibility score alone is not sufficient evidence that the source skill already solves the target. The independent solve-accuracy check prevents a merely related skill from being incorrectly treated as a zero-training solution.

### 2.1 Mathematical formulation

Let the incoming target task be $T$ and let the repository contain previously acquired skills $s_i$, each represented by a parameter vector $\theta_i$. A skill maps an input $x$ to an output through $f(x;\theta_i)$.

For a target probe set $D_T^{\mathrm{probe}}=\{(x_j,y_j)\}_{j=1}^{m}$, the frozen compatibility score used by the controller is the exponentially transformed mean squared error:

$$
\mathrm{MSE}(T,s_i)=\frac{1}{m}\sum_{j=1}^{m}\left(f(x_j;\theta_i)-y_j\right)^2
$$

$$
P(T\mid s_i)=\exp\left(-\frac{\mathrm{MSE}(T,s_i)}{60}\right).
$$

The compatibility score is computed without updating $\theta_i$. It therefore measures how well the frozen parent already matches the target, rather than how useful its internal representation will necessarily be after adaptation.

Let $A(T,s_i)$ denote the independent target-solve accuracy used by the corrected reuse gate. With thresholds $\tau_{\mathrm{solve}}=0.90$ and $\tau_{\mathrm{clone}}=0.15$, the controller selects an action according to:

$$
\mathrm{action}(T,s_i)=
\begin{cases}
\mathrm{reuse} & \text{if } P(T\mid s_i)\geq\tau_{\mathrm{solve}} \text{ and } A(T,s_i)\geq 0.85,\\
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
\theta_T^*=\arg\min_{\theta}\;\frac{1}{n}\sum_{j=1}^{n}\left(f(x_j;\theta)-y_j\right)^2.
$$

These equations define the mechanism evaluated in the experiments; they do not assume that a larger compatibility score must imply a larger transfer benefit.

### 2.2 Acquisition metrics

For a matched seed $r$, let $E_{\mathrm{scratch}}^{(r)}$ and $E_{\mathrm{clone}}^{(r)}$ denote the numbers of epochs required to reach the predeclared training criterion. The per-seed convergence speedup is:

$$
S^{(r)}=\frac{E_{\mathrm{scratch}}^{(r)}}{E_{\mathrm{clone}}^{(r)}}.
$$

Thus $S>1$ favors clone-and-adapt, while $S<1$ indicates that cloning is slower than scratch. The reported paired comparisons treat the matched seed as the unit of analysis.

Acquisition reliability is kept separate from efficiency. Let $I_r=1$ when the target reaches the declared acquisition criterion within the allowed budget and $I_r=0$ otherwise. The empirical success rate is:

$$
R=\frac{1}{N}\sum_{r=1}^{N} I_r.
$$

This separation is important for difficult targets such as squares, where a method may have a low success rate even if successful runs can be compared for convergence speed.

### 2.3 Retention and isolation invariant

For an acquired skill $s_i$, let $\theta_i^{\mathrm{pre}}$ and $\theta_i^{\mathrm{post}}$ denote its stored parameters immediately before and after a later skill is acquired. The intended isolated-skill invariant is:

$$
\theta_i^{\mathrm{post}}=\theta_i^{\mathrm{pre}},
$$

or equivalently,

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

The experiment separates target-training data, compatibility-probe data, solve/accuracy data, and held-out evaluation data. Held-out evaluation data are not used to choose thresholds, budgets, stopping rules, or model-selection decisions.

### Seeds

Experiments use matched deterministic seeds. The main relatedness and retention checks use 15 seeds per condition. Pairing by seed allows within-seed comparisons between acquisition strategies where applicable.

## 3. Research design

### 3.1 Relatedness and prior-history experiments

The initial source-target experiments examined multiplication → squares, multiplication → powers, addition → subtraction, and addition → multiplication. These results motivated a stronger fixed-target design: hold the target fixed and vary the prior-skill history.

The prerequisite matrix includes each of subtraction, division, squares, and powers under three histories: no prior skill, addition only, and addition + multiplication. This design distinguishes the effect of relevant prior knowledge from the mere presence of additional training history.

The interpretation is deliberately conservative. A curriculum order is not treated as proof that one task is a mathematical prerequisite for another. The experiment only tests whether a previously learned representation is useful for acquiring the target.

### 3.2 Retention and catastrophic forgetting

The retention code performs sequential acquisition and re-evaluates every previously acquired skill after each later acquisition on a stable, skill-specific evaluation set. This verifies the intended isolation invariant: a stored parent skill is not modified when a later skill is trained as an independent copy.

For each retention check the implementation records pre/post accuracy, accuracy change, retention ratio, and whether the change remains within a predeclared five-percentage-point practical tolerance. These measurements are useful diagnostics of the invariant, but they should not be interpreted as a conventional statistical test of forgetting because the isolated-skill architecture does not expose the stored parent to subsequent optimization. In particular, repeated evaluation of an unchanged network on a stable evaluation set is expected to produce the same result.

A genuine empirical test of resistance to interference would require an at-risk comparison arm in which later learning can modify previously learned parameters, such as the shared-network baseline. That comparison is outside the scope of this final documentation PR and is not claimed here.

## 4. Results

### 4.1 Transfer is relationship-dependent

The earlier relatedness-pair experiment produced heterogeneous transfer. Multiplication → powers showed a mean paired speedup of approximately 2.31×, while multiplication → squares showed approximately 1.26×. Addition → subtraction instead produced approximately 0.33×, meaning clone-and-adapt was slower than scratch. Addition → multiplication was approximately 1.12×.

These observations do not support a simple monotonic rule in which a larger frozen compatibility score always predicts a larger transfer benefit. In particular, a frozen output-compatibility measure and usefulness as a parameter initialization are not necessarily the same property.

### 4.2 Retention mechanism check

The retention run reports zero change in the repeated pre/post checks and a 100% pass rate under the five-percentage-point practical tolerance. These values are consistent with the implementation invariant that previously acquired skills are stored independently and are not modified while a new skill is adapted.

Because the same unchanged skill is evaluated on the same skill-specific retention set before and after later acquisitions, the resulting zero change is **not an independent empirical estimate of protection against catastrophic forgetting**. It is a verification that the isolation mechanism and evaluation protocol behave as intended. We therefore do not report bootstrap confidence intervals or effect sizes for the retention delta as evidence of a population-level effect.

### 4.3 Acquisition efficiency and reliability

Acquisition speed is treated as a secondary outcome. A small speedup is not automatically practically important, and reliability is evaluated separately from efficiency. A strong acquisition result is one in which prior knowledge increases fixed-budget success or substantially reduces training cost without sacrificing final held-out performance.

The earlier experiments illustrate why these outcomes must remain separate: clone initialization can reach an early training threshold much faster while the resulting model quality depends on the stopping and evaluation protocol. The fixed-budget follow-up therefore provides an important control against interpreting early threshold crossing as a universal quality improvement.

## 5. Statistical analysis

For paired strategy comparisons, the unit of analysis is the matched seed. Reported summaries include the mean paired difference, variability, interval estimates, effect sizes, and paired significance tests where appropriate for the acquisition comparisons.

The retention check is intentionally treated differently. Its primary diagnostic quantity is $\Delta_i=A_{i,\mathrm{post}}-A_{i,\mathrm{pre}}$, where accuracy is measured on the same held-out retention set for the same skill and seed. The practical diagnostic rule used by the experiment is $\Delta_i\geq-0.05$.

This five-percentage-point value is a declared practical tolerance. It is not a statistical equivalence margin and is not used to claim that the architecture has been shown equivalent in performance before and after acquisition.

Because the isolated parent network is not modified between the two evaluations, a zero retention delta is an expected consequence of the mechanism. Statistical inference on this delta would therefore not answer the stronger scientific question of whether a system resists interference when interference is possible.

## 6. Discussion

The combined findings support a simple design principle: a continual skill-acquisition system should not assume that every previous skill is useful, but it should preserve the option to exploit previous skills when they are useful.

The three-route controller is therefore important. Reuse is appropriate when an existing skill genuinely solves the target. Clone-and-adapt provides a way to exploit a useful initialization without modifying the parent. Scratch remains necessary because prior knowledge can be irrelevant or negatively transferable.

The retention checks provide implementation-level evidence that the independent-skill storage mechanism preserves stored parent parameters during later acquisition. They should not be confused with a comparative forgetting experiment. Demonstrating robustness to interference would require a condition in which later learning can actually alter parameters supporting earlier skills.

At the same time, the heterogeneous transfer results show that the harder scientific question is not simply whether cloning works. It is **why some skills benefit from prior knowledge while others do not**. That question is intentionally separated from the present paper's main acquisition-and-retention objective and can form a follow-up study based on the present experimental framework.

## 7. Limitations and threats to validity

1. **Small task family.** The experiments use a small set of arithmetic functions and therefore cannot establish behavior across broad classes of machine-learning tasks.
2. **Small model.** Results may depend on the architecture, parameter count, optimizer, and training dynamics.
3. **Finite seeds.** Fifteen seeds provide useful matched comparisons but do not establish universal population-level behavior.
4. **Controller dependence.** The results depend on the compatibility probe, thresholds, and solved-target gate.
5. **Retention protocol.** The retention checks verify the tested isolation mechanism; they do not test an at-risk interference condition in which parent parameters can be overwritten.
6. **No universal prerequisite claim.** Earlier acquisition in a curriculum is evidence about transfer under that protocol, not proof of a formal prerequisite relationship.
7. **Potential task-family confounds.** Arithmetic tasks share representations and input structure, so transfer behavior may differ substantially in other domains.
8. **Practical tolerance.** The five-percentage-point retention tolerance is a declared diagnostic criterion, not a statistical equivalence margin derived from an external validation study.

## 8. Reproducibility

The repository contains the experiment drivers, regression tests, workflow configuration, raw CSV outputs, statistical summaries, and plots. CI reruns the experiments and uploads the generated artifacts. The retention experiment additionally records the exact seed, target sequence, acquisition strategy, source skill, compatibility diagnostics, pre/post accuracy, retention change, and retention decision for each measured run.

## 9. Conclusion

This prototype demonstrates a controlled approach to continual skill acquisition in which the system can choose among reuse, clone-and-adapt, and scratch learning while preserving previously acquired skills through independent storage.

The most defensible conclusion is not that cloning always improves learning, nor that catastrophic forgetting has been eliminated. Instead, the experiments show that prior knowledge can have positive, negligible, or negative effects depending on the target and source relationship, while the isolated-skill mechanism preserves previously stored skills during later acquisition as an implementation invariant under the tested conditions.

The work therefore establishes a useful experimental foundation for a broader research program: first characterize reliable skill acquisition and the conditions under which transfer helps or hurts, then test interference resistance using explicit at-risk baselines and broader task families.

## 10. Reproducibility checklist

- [x] Reuse / clone / scratch routes implemented.
- [x] Corrected reuse gate requires independent solved-target evidence.
- [x] Genuine scratch fallback retained.
- [x] Matched seeds used for paired comparisons.
- [x] Held-out evaluation separated from training and controller data.
- [x] Retention evaluated on stable skill-specific evaluation sets.
- [x] Retention tolerance declared before interpretation.
- [x] Per-seed retention data and summary outputs generated.
- [x] CI executes the retention experiment and regression tests.
- [x] Retention claims explicitly separated from statistical evidence of interference resistance.
- [ ] An at-risk shared-network retention comparison is outside the scope of this PR and should be added only as a separate, explicitly controlled experiment.
- [ ] Final prerequisite-matrix expansion and its final statistical tables should be included only after that experiment is the authoritative merged result.
