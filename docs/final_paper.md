# Continual Skill Acquisition via Reuse, Clone-and-Adapt, and Scratch Learning

## Abstract

Continual acquisition systems must learn new skills without unnecessarily discarding useful prior knowledge. This study evaluates a small, reproducible skill-acquisition framework in which an incoming task is handled by one of three routes: reuse an existing skill when it already solves the target, clone a related skill and adapt it when reuse is insufficient, or learn from a fresh initialization when prior knowledge is not useful. The experiments use small arithmetic regression tasks and matched random seeds to separate acquisition reliability, acquisition efficiency, and retention.

The central result is that prior knowledge is not uniformly beneficial: the observed transfer depends on the source-target relationship and on how the controller evaluates that relationship. Earlier experiments showed both substantial positive transfer and negative transfer, motivating a fixed-target prerequisite design rather than a simple ranking of source-target pairs. The retention experiment then evaluated whether previously acquired skills remained usable after subsequent acquisitions. Under the tested isolated-skill architecture, no measurable degradation was observed in the retention checks. This supports the narrower claim that the proposed isolation mechanism can acquire additional skills without overwriting the parameters of previously stored skills; it does not establish that catastrophic forgetting is impossible in continual-learning systems generally.

## 1. Introduction

A continual-learning system should be able to acquire a new capability while preserving capabilities that it has already learned. A simple shared-network strategy can update parameters for every new task, making the system vulnerable to interference. An alternative is to treat each learned capability as an independently stored skill and decide, for each incoming task, whether an existing skill should be reused, cloned and adapted, or whether learning should begin from scratch.

This work studies that mechanism as a controlled research prototype. The goal is not to claim a universal theory of prerequisites or a universal advantage for cloning. Instead, the experiments ask when previously acquired skills help, when they do not, and whether adding a new skill damages previously acquired ones.

The research is organized around four questions:

1. Can a new target skill be acquired reliably?
2. Can an already-solved target be reused without additional training?
3. When reuse is not possible, does cloning provide a useful initialization compared with scratch learning?
4. Does acquiring a new skill cause measurable degradation of earlier skills?

## 2. Experimental system

The prototype uses a small two-input, one-output neural network with a 32-unit hidden layer. The task family consists of addition, subtraction, multiplication, powers, and squares. All conditions use the same architecture, optimizer, learning rate, training procedure, convergence criterion, and fixed training budget.

The controller exposes three acquisition routes:

- **Reuse:** retain an existing skill unchanged when both the compatibility gate and an independent solved-target accuracy check are satisfied.
- **Clone + adapt:** deep-copy an existing skill and train the copy on the target task.
- **Scratch:** create a fresh network and train it on the target task.

The corrected reuse gate is important. A high frozen compatibility score alone is not sufficient evidence that the source skill already solves the target. The independent solve-accuracy check prevents a merely related skill from being incorrectly treated as a zero-training solution.

### Data separation

The experiment separates target-training data, compatibility-probe data, solve/accuracy data, and held-out evaluation data. Held-out evaluation data are not used to choose thresholds, budgets, stopping rules, or model-selection decisions.

### Seeds

Experiments use matched deterministic seeds. The main relatedness and retention experiments use 15 seeds per condition. Pairing by seed allows within-seed comparisons between acquisition strategies and retention measurements.

## 3. Research design

### 3.1 Relatedness and prior-history experiments

The initial source-target experiments examined multiplication → squares, multiplication → powers, addition → subtraction, and addition → multiplication. These results motivated a stronger fixed-target design: hold the target fixed and vary the prior-skill history.

The planned prerequisite matrix includes each of subtraction, division, squares, and powers under three histories: no prior skill, addition only, and addition + multiplication. This design distinguishes the effect of relevant prior knowledge from the mere presence of additional training history.

The interpretation is deliberately conservative. A curriculum order is not treated as proof that one task is a mathematical prerequisite for another. The experiment only tests whether a previously learned representation is useful for acquiring the target.

### 3.2 Retention and catastrophic forgetting

The retention experiment evaluates sequential acquisition. After each new skill is acquired, every previously acquired skill is evaluated again on its own stable, skill-specific evaluation set. For each old skill we record:

- pre-acquisition accuracy;
- post-acquisition accuracy;
- absolute retention change;
- retention ratio;
- whether the change stays within the predeclared five-percentage-point practical tolerance.

The same evaluation set is used for the paired pre/post comparison of a given skill, preventing changes in the evaluation sample from being mistaken for learning or forgetting.

The experiment does not assume that forgetting is absent. Negative retention is recorded as a valid outcome. The result is therefore an empirical measurement rather than a construction of a desired answer.

## 4. Results

### 4.1 Transfer is relationship-dependent

The earlier relatedness-pair experiment produced heterogeneous transfer. Multiplication → powers showed a mean paired speedup of approximately 2.31×, while multiplication → squares showed approximately 1.26×. Addition → subtraction instead produced approximately 0.33×, meaning clone-and-adapt was slower than scratch. Addition → multiplication was approximately 1.12×.

These observations do not support a simple monotonic rule in which a larger frozen compatibility score always predicts a larger transfer benefit. In particular, a frozen output-compatibility measure and usefulness as a parameter initialization are not necessarily the same property.

### 4.2 Retention

The retention experiment evaluated four representative three-skill sequences across 15 seeds. The generated retention summary reported zero mean accuracy change for the measured retention checks, with a 100% retention-pass rate under the predeclared five-percentage-point tolerance. The corresponding bootstrap intervals were degenerate at zero in the reported checks.

The correct interpretation is narrow: **no measurable catastrophic forgetting was observed under the tested isolated-skill mechanism and evaluation protocol.** The result is consistent with the implementation design in which a previously acquired skill is stored independently and is not modified when a new skill is adapted.

This should not be phrased as proof that catastrophic forgetting is impossible. It demonstrates that the tested architecture can add new skills while preserving the measured performance of previously stored skills.

### 4.3 Acquisition efficiency and reliability

Acquisition speed is treated as a secondary outcome. A small speedup is not automatically practically important, and reliability is evaluated separately from efficiency. A strong acquisition result is one in which prior knowledge increases fixed-budget success or substantially reduces training cost without sacrificing final held-out performance.

The earlier experiments illustrate why these outcomes must remain separate: clone initialization can reach an early training threshold much faster while the resulting model quality depends on the stopping and evaluation protocol. The fixed-budget follow-up therefore provides an important control against interpreting early threshold crossing as a universal quality improvement.

## 5. Statistical analysis

For paired strategy comparisons, the unit of analysis is the matched seed. Reported summaries should include the mean paired difference, standard deviation, confidence or bootstrap interval, effect size, and an appropriate paired significance test where sample size and distributional assumptions permit.

For retention, the primary quantity is:

\[
\Delta_i = A_{i,\mathrm{post}} - A_{i,\mathrm{pre}},
\]

where accuracy is measured on the same held-out retention set for the same skill and seed. The practical retention rule used by the experiment is:

\[
\Delta_i \ge -0.05.
\]

This five-percentage-point value is a practical tolerance, not a claim of statistical equivalence. Statistical intervals and effect sizes should accompany it.

Success rate is reported separately from convergence speed. If all conditions reach the fixed budget successfully, training cost and convergence distributions become the more informative secondary outcomes.

## 6. Discussion

The combined findings support a simple design principle: a continual skill-acquisition system should not assume that every previous skill is useful, but it should preserve the option to exploit previous skills when they are useful.

The three-route controller is therefore important. Reuse is appropriate when an existing skill genuinely solves the target. Clone-and-adapt provides a way to exploit a useful initialization without modifying the parent. Scratch remains necessary because prior knowledge can be irrelevant or negatively transferable.

The retention result strengthens the continual-learning interpretation of the mechanism. Learning a new skill does not require sacrificing the old one when skills are isolated and stored independently. This is particularly important because the purpose of skill acquisition is not merely to produce one successful new model; it is to accumulate capabilities over time.

At the same time, the heterogeneous transfer results show that the harder scientific question is not simply whether cloning works. It is **why some skills benefit from prior knowledge while others do not**. That question is intentionally separated from the present paper's main acquisition-and-retention objective and can form a follow-up study based on the present experimental framework.

## 7. Limitations and threats to validity

1. **Small task family.** The experiments use a small set of arithmetic functions and therefore cannot establish behavior across broad classes of machine-learning tasks.
2. **Small model.** Results may depend on the architecture, parameter count, optimizer, and training dynamics.
3. **Finite seeds.** Fifteen seeds provide useful matched comparisons but do not establish universal population-level behavior.
4. **Controller dependence.** The results depend on the compatibility probe, thresholds, and solved-target gate.
5. **Retention protocol.** The absence of measured forgetting is conditional on the independent-skill storage mechanism and the tested sequence lengths.
6. **No universal prerequisite claim.** Earlier acquisition in a curriculum is evidence about transfer under that protocol, not proof of a formal prerequisite relationship.
7. **Potential task-family confounds.** Arithmetic tasks share representations and input structure, so transfer behavior may differ substantially in other domains.
8. **Practical tolerance.** The five-percentage-point retention tolerance is a declared practical criterion, not a statistical equivalence margin derived from an external validation study.

## 8. Reproducibility

The repository contains the experiment drivers, regression tests, workflow configuration, raw CSV outputs, statistical summaries, and plots. CI reruns the experiments and uploads the generated artifacts. The retention experiment additionally records the exact seed, target sequence, acquisition strategy, source skill, compatibility diagnostics, pre/post accuracy, retention change, and retention decision for each measured run.

## 9. Conclusion

This prototype demonstrates a controlled approach to continual skill acquisition in which the system can choose among reuse, clone-and-adapt, and scratch learning while preserving previously acquired skills through independent storage.

The most defensible conclusion is not that cloning always improves learning. Instead, the experiments show that prior knowledge can have positive, negligible, or negative effects depending on the target and source relationship, while the isolated-skill mechanism can add new capabilities without measurable degradation of previously stored skills under the tested conditions.

The work therefore establishes a useful experimental foundation for a broader research program: first characterize reliable skill acquisition and retention, then investigate the factors that determine why one skill transfers well and another does not.

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
- [ ] Final prerequisite-matrix expansion and its final statistical tables should be included only after that experiment is the authoritative merged result.
