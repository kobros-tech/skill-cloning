# Continual Skill Acquisition via Reuse, Clone-and-Adapt, and Scratch Learning

## Abstract

Continual learning requires deciding not only how to learn a new skill, but also whether previously acquired skills should be reused, adapted, or ignored. We study this problem with a controlled three-route framework that chooses among reuse, clone-and-adapt, and scratch learning. Using small arithmetic regression tasks, matched deterministic seeds, and separated training, probe, calibration, and evaluation data, we isolate acquisition reliability, transfer efficiency, parent identity, and distribution sensitivity.

Our results show that prior knowledge is not uniformly beneficial. Transfer depends on the source-target relationship, the identity of the transferred parent, and the task distribution. In the fixed-target prerequisite analysis, additional prior history improves acquisition for some targets, while division exhibits negative transfer and squares remains difficult. A matched three-arm parent control further shows that parent identity matters beyond generic pretraining: relevant parents can accelerate acquisition relative to both scratch and unrelated parents, while an unrelated parent can either hurt or help depending on the target. A signed-domain follow-up demonstrates that these transfer effects can change substantially under distribution shift, including a reversal in transfer direction for multiplication → powers. Together, these results show that useful prior knowledge is conditional rather than universally beneficial.

The retention experiment establishes a narrower architectural invariant: previously acquired skills remain unchanged while later skills are learned in isolated parameter copies. We therefore interpret the study as a controlled empirical analysis of skill transfer under the proposed acquisition protocol, rather than as evidence of general immunity to catastrophic forgetting or broad generalization to larger architectures and domains.

## 1. Introduction

Continual learning is not only a question of whether a model can learn a new task; it is also a question of how previously acquired capabilities should be used when a new task arrives. If all tasks continually update one shared parameter set, learning a new task can interfere with capabilities acquired earlier. An alternative is to treat acquired capabilities as persistent skills and to make an explicit acquisition decision: reuse a skill when it already solves the target, clone a useful parent and adapt the copy when prior knowledge may help, or fall back to scratch learning when prior knowledge is unsuitable.

The key question studied here is therefore narrower and more testable than whether skill reuse is universally beneficial: **when does previously acquired knowledge help with a new skill, when does it hurt, and does the identity of the reused parent matter?** This question matters because a continual learner that always reuses prior knowledge can suffer negative transfer, while a learner that always starts from scratch discards potentially useful information. A useful acquisition mechanism should preserve all three options and make their consequences measurable.

We study this question in a deliberately small and controlled experimental setting. The task family consists of arithmetic regression problems with a common input/output structure, and the model is a two-input, one-output neural network with a 32-unit hidden layer. This simplicity is intentional: it allows the source-target relationship, acquisition route, training budget, seed, and data distribution to be controlled independently, making it possible to distinguish several effects that would be confounded in a large benchmark. The results should therefore be read as a controlled study of transfer dynamics, not as evidence that the same behavior must hold across arbitrary architectures or application domains.

The framework evaluates three acquisition routes. **Reuse** keeps an existing skill unchanged when independent evidence indicates that it already solves the target. **Clone-and-adapt** copies a selected parent and trains the copy on the new task, leaving the stored parent intact. **Scratch** starts from a fresh initialization. This design makes it possible to ask not merely whether pretraining helps, but whether a particular previously acquired skill provides a better initialization than another one.

The experiments are organized around five questions:

1. **Acquisition reliability:** Can a new target skill be acquired within the declared budget, and how does reliability vary across targets and prior histories?
2. **Reuse:** Can an already-solved target be recognized and reused without additional optimization?
3. **Clone versus scratch:** When reuse is insufficient, does cloning reduce the training cost of acquiring a new skill compared with a fresh initialization?
4. **Parent identity:** Does the benefit of cloning depend on which previously acquired skill supplies the initialization, beyond the generic effect of pretraining?
5. **Domain sensitivity:** Does observed transfer remain stable when the operand distribution is expanded from non-negative to signed values?

These questions are answered with matched deterministic seeds, explicit separation of training, compatibility, calibration, and held-out evaluation data, a three-arm parent-control experiment, and an explicit signed-domain follow-up. The resulting evidence is intentionally interpreted as conditional on the tested protocol. The main conclusion is not that cloning always helps, nor that prior skills constitute universal prerequisites. Rather, the experiments show that transfer is heterogeneous, that parent identity can matter beyond generic pretraining, and that the direction and magnitude of transfer can change with the task distribution.

### Contributions

This work makes four contributions:

1. **A controlled three-route skill-acquisition framework.** We define and evaluate a reproducible acquisition protocol with three explicit alternatives—reuse, clone-and-adapt, and scratch—together with separate compatibility and independent solve checks. This separates the decision to reuse an existing capability from the decision to use that capability as an initialization for further learning.

2. **Evidence for heterogeneous transfer and parent-specific effects.** Matched experiments show that prior skills can produce positive, negative, or limited transfer depending on the target. The three-arm parent-control experiment goes beyond a generic pretrained-versus-scratch comparison by holding the target, seed, data, architecture, optimizer, and budget fixed while varying only the parent identity; the resulting differences provide evidence that the identity of the transferred skill matters under the tested protocol.

3. **Evidence that transfer is sensitive to the task distribution.** The signed-domain follow-up tests whether source-target transfer relationships remain stable when operand distributions change from non-negative to signed values. The observed changes include both substantial shifts in magnitude and a reversal of transfer direction, showing that transfer is not a fixed property of task labels alone.

4. **A reproducible methodology for controlled transfer studies.** The study combines matched deterministic seeds, separated training/probe/calibration/evaluation data, explicit budget-capped acquisition metrics, prerequisite-validity accounting, and multiplicity-corrected paired inference. Retention is deliberately reported as an architectural isolation invariant rather than as evidence of general immunity to catastrophic forgetting.

Together, these contributions establish a bounded empirical claim: under the tested controlled protocol, prior skills can be useful, harmful, or largely neutral, and their effect depends on the target, parent identity, and data distribution. The work does not claim that these transfer patterns necessarily generalize to larger architectures or unrelated domains; instead, it provides a controlled setting in which the relevant effects can be isolated and measured.

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

Thus $A(T,s_i)$ is an independently measured estimate of whether the frozen parent actually solves the target, rather than a relabeling derived from the compatibility score. The solve/calibration batch is separate from both the training data and the compatibility-probe batch, and the held-out test set is never consulted for the reuse decision. In the implementation, the calibration batch uses the independent solve-probe seed offset $20{,}000$ and the same task sampling procedure as the experiments.

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

For each retention check the implementation records pre/post accuracy, accuracy change, retention ratio, and whether the change remains within a predeclared five-percentage-point practical tolerance. These measurements are useful diagnostics of the invariant, but they should not be interpreted as a conventional statistical test of forgetting because the isolated-skill architecture does not expose the stored parent to subsequent optimization.

In particular, repeated evaluation of an unchanged network on a stable evaluation set is expected to produce the same result.

A genuine empirical test of resistance to interference would require an at-risk comparison arm in which later learning can modify previously learned parameters, such as a shared-network baseline. That comparison is outside the scope of this PR and is not claimed here.

### 3.4 Signed-domain follow-up

The signed-domain experiment compares the original non-negative configuration with an explicitly configured signed configuration. Task ranges are task-specific: addition, subtraction, multiplication, and squares use the signed sampling ranges documented in the experiment configuration. The purpose is not to establish a universal domain-shift benchmark, but to test whether the observed transfer relationships are stable under a concrete change in operand distribution.

## 4. Results

### 4.1 Fixed-target acquisition

The **fixed-target prerequisite matrix is the primary quantitative result**. Holding the target fixed while varying prior-skill history shows that the effect of additional prior knowledge is target-dependent rather than monotonic. Powers benefit from additional prior history under the authoritative fixed-target design, while division exhibits negative transfer and squares remains difficult. These results support heterogeneous transfer: accumulating more skills does not guarantee faster or more reliable acquisition.

Acquisition reliability and acquisition efficiency should be distinguished. The matrix first determines whether a target can be acquired within the declared budget; only then are matched training-cost comparisons interpreted. This distinction is especially important for difficult targets such as squares, where failures cannot be treated as ordinary successful convergence observations.

### 4.2 Parent identity

The matched three-arm parent-control experiment provides the strongest direct test of whether transfer depends on **which** prior skill is used, rather than merely on whether the model was pretrained. For multiplication, the relevant clone reaches the criterion 117.2 steps earlier than scratch on average and 298.3 steps earlier than the unrelated parent; the unrelated parent is itself 181.1 steps slower than scratch. For powers, the relevant clone is 260.5 steps faster than scratch and 125.9 steps faster than the unrelated parent, while the unrelated parent is 134.7 steps faster than scratch. Holm-adjusted paired tests support all six reported contrasts under the declared correction procedure.

The pattern is important because it is not consistent with a simple rule that "pretraining helps." The unrelated parent hurts multiplication but helps powers, while the relevant parent outperforms both baselines for both targets. Under this protocol, the source skill therefore contains information about transfer that is not captured by the generic fact that it was pretrained.

### 4.3 Domain sensitivity

The signed-domain follow-up tests whether these transfer relationships remain stable when the input distribution changes. On the valid matched seeds, multiplication $\rightarrow$ powers changes from $2.217\times$ to $0.703\times$ and reverses direction ($p=0.00168$). Multiplication $\rightarrow$ squares changes from $1.265\times$ to $1.031\times$ without a conventionally significant domain difference ($p=0.0757$). Addition $\rightarrow$ subtraction changes from $0.408\times$ to $1.001\times$ ($p\approx1.88\times10^{-7}$), while the addition $\rightarrow$ multiplication null control changes from $1.145\times$ to $1.176\times$ with no statistically detectable domain difference ($p=0.591$).

The powers and squares comparisons have only 5/15 valid matched seeds because signed-domain multiplication prerequisite acquisition fails in 10 seeds. These results are therefore interpreted as limited domain-sensitivity evidence rather than as high-powered general conclusions.

### 4.4 Retention

Under the isolated-skill architecture, previously acquired parameters remain unchanged during later acquisition, and the stable retention evaluations remain unchanged within the implementation tolerance. This result verifies the intended storage/isolation invariant. It does not establish that a shared-network continual learner would avoid catastrophic forgetting.

## 5. Discussion

The results support a bounded view of skill transfer. Prior knowledge is neither uniformly beneficial nor uniformly harmful. Its effect depends on the relationship between the source and target, the identity of the parent selected for cloning, and the input distribution on which the skills are evaluated.

The parent-control experiment is especially informative because it separates two effects that are often conflated: generic pretraining and task-specific transfer. An unrelated parent can be beneficial for one target and harmful for another, while a relevant parent can provide an additional advantage. This suggests that the utility of a stored skill is not determined solely by its training history; the relationship between the learned parameters and the incoming task matters.

The fixed-target prerequisite matrix also argues against interpreting curriculum order as a universal prerequisite graph. A prior skill can improve acquisition under one target while adding little or even imposing a cost under another. The appropriate conclusion is therefore conditional: prior histories alter acquisition dynamics under the tested protocol, but they do not define universal prerequisites.

The signed-domain follow-up strengthens this conditional interpretation. Several transfer ratios change substantially when the operand distribution changes, including a reversal for multiplication $\rightarrow$ powers. This indicates that transfer relationships are sensitive to the data distribution and should not be treated as invariant properties of task names alone.

The retention result has a deliberately narrower interpretation. Because skills are stored independently and later training operates on copies, preservation of the original parameters is an architectural consequence of isolation. It is useful as an implementation invariant, but it does not replace a shared-parameter continual-learning baseline for studying catastrophic forgetting under interference.

## 6. Limitations

The most important limitation is external validity. The experiments use a small arithmetic task family and a 32-unit network. This design provides unusually strong control over source-target relationships, but it does not establish that the same transfer dynamics occur in larger networks, perceptual tasks, language tasks, or other continual-learning settings.

The signed-domain follow-up has an additional limitation: 10 of 15 nominal seeds fail to acquire the signed-domain multiplication prerequisite, leaving only 5 valid matched seeds for the multiplication $\rightarrow$ powers and multiplication $\rightarrow$ squares comparisons. These results are consequently low-powered and are presented as domain-sensitivity evidence rather than definitive estimates of generalization.

The isolated-skill architecture also limits what can be concluded about catastrophic forgetting. Since later training does not modify stored parents, perfect retention is expected by construction. A meaningful comparison to conventional interference-based continual learning would require an at-risk shared-parameter baseline, which is not part of this study.

Finally, the controller's compatibility and solve thresholds are fixed by the experimental protocol. The study evaluates the behavior of this specified decision mechanism rather than claiming that these thresholds are optimal or universally appropriate.

## 7. Reproducibility and statistical protocol

All primary comparisons use deterministic matched seeds and preserve the same target data across the relevant acquisition arms. Training, compatibility probes, independent solve/calibration batches, and held-out evaluation sets are separated. The parent-control analysis uses paired within-seed comparisons and applies Holm's step-down correction across the six planned paired hypotheses. Failed prerequisite acquisitions are explicitly recorded rather than silently treated as successful observations.

The repository contains the experiment configurations, scripts, workflows, result artifacts, and documentation needed to reproduce the reported analyses.