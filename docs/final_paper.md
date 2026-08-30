# Continual Skill Acquisition via Reuse, Clone-and-Adapt, and Scratch Learning

## Abstract

Continual acquisition systems must learn new skills without unnecessarily discarding useful prior knowledge. This study evaluates a small, reproducible skill-acquisition framework in which an incoming task is handled by one of three routes: reuse an existing skill when it already solves the target, clone a related skill and adapt it when reuse is insufficient, or learn from a fresh initialization when prior knowledge is not useful. The experiments use small arithmetic regression tasks and matched random seeds to separate acquisition reliability, acquisition efficiency, and retention.

The central result is that prior knowledge is not uniformly beneficial: the observed transfer depends on the source-target relationship and on how the controller evaluates that relationship. Earlier experiments showed both substantial positive transfer and negative transfer, motivating a fixed-target prerequisite design rather than a simple ranking of source-target pairs. A signed-domain follow-up further shows this transfer behavior is itself distribution-sensitive: expanding the operand domain from non-negative to signed integers reversed the direction of the strongest transfer result (multiplication → powers, 2.17× → 0.72×) and eroded another toward no effect (multiplication → squares, 1.23× → 1.01×), while a null-control pair stayed stable, indicating the reversal is not simply an artifact of the manipulation itself. A separate check confirmed that the implementation's skill-isolation guarantee holds: a previously acquired skill's stored parameters are never modified by later acquisitions, so its accuracy on a fixed evaluation set is unchanged before and after. This is an implementation invariant of the current architecture, not a statistical finding about resistance to catastrophic forgetting -- there is no mechanism in this experiment through which a frozen skill's parameters could change, so the check could not have come out any other way. A genuine empirical test of forgetting would require a baseline in which interference is actually possible (e.g. a shared-parameter architecture); this paper reports the isolation-guarantee check on its own terms rather than as evidence against catastrophic forgetting in general.

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

### 3.2 Skill-isolation invariant check (not a statistical retention experiment)

A separate check evaluates sequential acquisition. After each new skill is acquired, every previously acquired skill is evaluated again on its own stable, skill-specific evaluation set. For each old skill we record pre-acquisition accuracy, post-acquisition accuracy, the absolute change between them, and whether the change stays within a predeclared five-percentage-point practical tolerance.

Unlike the transfer experiments in Section 3.1, this check does not have statistical uncertainty to report. In the current architecture, a `Skill`'s network is never modified once training on it stops -- there is no code path in this implementation through which a later acquisition's gradient updates could reach an already-stored skill. Because the pre- and post-acquisition evaluations use the identical frozen network on the identical deterministic evaluation batch (same seed, same data), the recorded change is mathematically guaranteed to be exactly zero; it could not have come out any other way given the code as written. Reporting a confidence interval, standard deviation, or effect size over that quantity would describe sampling uncertainty that does not exist, so this paper does not compute or report those statistics for this check.

What the check legitimately establishes is narrower and still useful: that the isolation guarantee actually holds in the implementation, i.e. there is no bug that lets a later acquisition leak into a supposedly frozen skill. That is a real regression check on the code, not an empirical finding about continual learning in general. A genuine empirical test of forgetting would require an architecture in which interference is actually possible -- for example, re-running the original shared-parameter baseline (Section 4.1's precursor, described in the historical results) across this same sequential-acquisition protocol, so that a nonzero result is a possible outcome the check could detect. That comparison is left for future work rather than attempted here, to keep this PR's scope to the reframing itself.

## 4. Results

### 4.1 Transfer is relationship-dependent

The earlier relatedness-pair experiment produced heterogeneous transfer. Multiplication → powers showed a mean paired speedup of approximately 2.31×, while multiplication → squares showed approximately 1.26×. Addition → subtraction instead produced approximately 0.33×, meaning clone-and-adapt was slower than scratch. Addition → multiplication was approximately 1.12×.

These observations do not support a simple monotonic rule in which a larger frozen compatibility score always predicts a larger transfer benefit. In particular, a frozen output-compatibility measure and usefulness as a parameter initialization are not necessarily the same property.

### 4.4 Domain-sensitivity analysis (signed-domain follow-up)

**Research question:** does the transfer behavior in Section 4.1 hold up when the arithmetic operand domain is expanded from non-negative integers to include negative values, or is it sensitive to the specific input distribution the original experiments used?

This follow-up reruns the four pairs from Section 4.1 under matched seeds in two domains -- the original non-negative domain (`{0,...,9}` for most tasks) and a signed domain (`{-9,...,9}`, with the powers exponent deliberately kept non-negative to avoid confounding operand sign with a change to fractional targets, and division's divisor drawn to be nonzero in both domains). Only the operand domain changes; architecture, optimizer, learning rate, stopping criterion, training budget, seed protocol, and compatibility calculation are identical to Section 4.1's setup. This experiment's own re-run of the non-negative domain (2.17×, 1.23×, 0.41×, 1.15× for the four pairs respectively) replicates Section 4.1's numbers (2.31×, 1.26×, 0.33×, 1.12×) closely under an independently constructed seed scheme -- not identically, since the two experiments draw data differently, but in the same direction and comparable magnitude for every pair, which is itself a useful cross-check.

**Result: transfer is not uniformly robust to this domain expansion.**

| Pair | Non-negative speedup | Signed speedup | Paired t-test p | Direction reversed? |
|---|---|---|---|---|
| multiplication → powers | 2.17× | 0.72× | 5.6×10⁻⁸ | **Yes** |
| multiplication → squares | 1.23× | 1.01× | 7.0×10⁻⁵ | No (erodes toward null) |
| addition → subtraction | 0.41× | 1.00× | 1.9×10⁻⁷ | Yes (negative transfer neutralizes) |
| addition → multiplication (null control) | 1.15× | 1.18× | 0.59 (n.s.) | No |

Three qualitatively different outcomes appear across four pairs, all from the same protocol:

- **multiplication → powers reverses direction**: a substantial positive transfer (2.17×) becomes a substantial negative one (0.72×), the "Case C" outcome. A sign-specific diagnostic breakdown of the trained clone's held-out error offers a plausible (not proven) mechanism: negative-base/odd-exponent inputs -- a case that does not exist at all in the non-negative domain, since the base was never negative there -- have by far the highest error (mean absolute error 0.89, 67% within-tolerance accuracy) of any quadrant, compared to 0.23 / 93% for positive-base inputs. The signed domain introduces a genuinely harder sub-problem that the multiplication-derived initialization is not well suited to, which is consistent with (though does not prove) the observed reversal.
- **multiplication → squares erodes toward no effect** ("Case B"): the already-modest 1.23× shrinks to 1.01×, statistically indistinguishable from no speedup. The sign breakdown shows essentially symmetric error for positive- and negative-base inputs (0.486 vs 0.489 mean absolute error), consistent with squares' sign-invariance (`(-a)² = a²`) -- the erosion here looks more like the frozen compatibility score itself collapsing (0.0006 → ~4×10⁻¹⁴, since a signed multiplication network's output on squares' now-broader input region is an even worse frozen match) than a base-sign-specific difficulty effect.
- **addition → subtraction's negative transfer neutralizes**: 0.41× (significantly worse than scratch) becomes 1.00× (no significant difference). The sign breakdown shows same-sign subtraction pairs, e.g. (+,+) and (-,-), are learned much more accurately (95-97% within-tolerance accuracy) than mixed-sign pairs (71%), consistent with same-sign subtraction behaving more like addition (which is what the parent skill actually learned) than mixed-sign subtraction does.
- **addition → multiplication (the null control) stays stable**: 1.15× and 1.18×, not a significant difference (p=0.59). A null control that stays null under a manipulation that changes other pairs substantially is itself evidence the other three effects are real rather than an artifact of, say, reduced training-set effective size in the signed domain.

**Acquisition reliability is also domain-sensitive, independent of speedup.** The fixed-target prerequisite-history matrix (Section 3.1's design) run under both domains shows squares' acquisition success rate -- already low in the non-negative domain (13-20% within the shared training budget, across prior-skill histories) -- drops to exactly 0% in the signed domain, for every history condition. This is a reliability collapse on top of an already-unreliable case, not merely a change in the size of an existing effect.

**The compatibility score itself is domain-sensitive**, which has a direct behavioral consequence for the controller: division's compatibility score against an addition parent drops from 0.28-0.31 in the non-negative domain (comfortably above `τ_clone=0.15`, so the controller reliably chose `clone`) to 0.03-0.06 in the signed domain (below `τ_clone`, so the controller switched to `scratch` in 14-15 of 15 seeds). The controller is not making a fixed decision about a fixed task relationship -- its own inputs shift with the domain.

**Interpretation, held to the same standard as the rest of this paper:** the transfer effect was *not* robust to this tested domain expansion for two of the four pairs (multiplication→powers reversed, multiplication→squares eroded to null), *was* robust for the null control, and addition→subtraction's negative-transfer finding did not survive the expansion. This supports treating the Section 4.1 transfer results as **distribution-sensitive rather than fixed properties of the source-target task relationship** -- consistent with Section 4.1's own observation that a frozen compatibility score is an incomplete predictor of transfer benefit; here we see the score, and not only the benefit it's meant to predict, move with the domain. This conclusion is conditional on the tested non-negative/signed expansion specifically (`{0,...,9}` vs. `{-9,...,9}`-scale domains on these six arithmetic tasks); it should not be read as a claim about negative numbers in general or about domain robustness for architectures or task families not tested here.

### 4.2 Skill-isolation invariant check

The check evaluated four representative three-skill sequences across 15 seeds. Every recorded accuracy change was exactly 0.0, and every check passed the predeclared five-percentage-point tolerance. This is the expected result given the architecture, not a discovery: as described in Section 3.2, a frozen skill's parameters cannot change under this implementation, so a nonzero result was never a possible outcome for this particular check to find.

The correct interpretation is narrow: **the implementation's skill-isolation guarantee holds -- previously acquired skills' stored parameters are provably unaffected by later acquisitions in this codebase.** This is a real and useful regression property (it rules out a class of bugs where a later training step accidentally touches a supposedly frozen skill), but it should not be read as empirical evidence that catastrophic forgetting is absent or difficult to produce in continual-learning systems generally -- this check's design has no mechanism through which forgetting could have appeared even if the underlying question were false for some other architecture.

A genuine empirical retention result would require a comparison arm where interference is actually possible (Section 3.2); that experiment is not part of this check and is noted as future work.

### 4.3 Acquisition efficiency and reliability

Acquisition speed is treated as a secondary outcome. A small speedup is not automatically practically important, and reliability is evaluated separately from efficiency. A strong acquisition result is one in which prior knowledge increases fixed-budget success or substantially reduces training cost without sacrificing final held-out performance.

The earlier experiments illustrate why these outcomes must remain separate: clone initialization can reach an early training threshold much faster while the resulting model quality depends on the stopping and evaluation protocol. The fixed-budget follow-up therefore provides an important control against interpreting early threshold crossing as a universal quality improvement.

## 5. Statistical analysis

For paired strategy comparisons, the unit of analysis is the matched seed. Reported summaries should include the mean paired difference, standard deviation, confidence or bootstrap interval, effect size, and an appropriate paired significance test where sample size and distributional assumptions permit.

For the skill-isolation invariant check, the recorded quantity is the same form:

\[
\Delta_i = A_{i,\mathrm{post}} - A_{i,\mathrm{pre}},
\]

where accuracy is measured on the same evaluation set for the same skill and seed, and the practical pass rule is \(\Delta_i \ge -0.05\). Unlike the paired strategy comparisons above, \(\Delta_i\) has zero variance across seeds under the current architecture (Section 3.2), so no confidence interval or effect size is reported for it -- those tools describe uncertainty in a sampling process, and there is none here to describe. The five-percentage-point tolerance is retained as a sanity bound: a violation would indicate an implementation bug, not a statistically surprising result.

Success rate is reported separately from convergence speed. If all conditions reach the fixed budget successfully, training cost and convergence distributions become the more informative secondary outcomes.

## 6. Discussion

The combined findings support a simple design principle: a continual skill-acquisition system should not assume that every previous skill is useful, but it should preserve the option to exploit previous skills when they are useful.

The three-route controller is therefore important. Reuse is appropriate when an existing skill genuinely solves the target. Clone-and-adapt provides a way to exploit a useful initialization without modifying the parent. Scratch remains necessary because prior knowledge can be irrelevant or negatively transferable.

The skill-isolation invariant check supports the architectural design rationale: storing skills independently and never modifying a stored skill's parameters is, by construction, a way to accumulate capabilities without overwriting earlier ones. This is a property of the design that the check confirms holds in the implementation, not a discovered empirical resistance to forgetting -- the distinction matters because the purpose of skill acquisition is to accumulate capabilities over time, and a design-level guarantee is a reasonable way to pursue that, but it is a different kind of claim than an empirical result obtained from an architecture where forgetting was actually possible.

At the same time, the heterogeneous transfer results show that the harder scientific question is not simply whether cloning works. It is **why some skills benefit from prior knowledge while others do not**. That question is intentionally separated from the present paper's main acquisition-and-retention objective and can form a follow-up study based on the present experimental framework.

## 7. Limitations and threats to validity

1. **Small task family.** The experiments use a small set of arithmetic functions and therefore cannot establish behavior across broad classes of machine-learning tasks.
2. **Small model.** Results may depend on the architecture, parameter count, optimizer, and training dynamics.
3. **Finite seeds.** Fifteen seeds provide useful matched comparisons but do not establish universal population-level behavior.
4. **Controller dependence.** The results depend on the compatibility probe, thresholds, and solved-target gate.
5. **The skill-isolation check is an implementation invariant, not an empirical retention result.** Under the tested architecture, a stored skill's parameters cannot change once acquired, so the check's zero-delta outcome was the only possible result given the code as written -- it demonstrates the isolation guarantee holds (a real, useful regression property) but says nothing about whether forgetting would occur in an architecture where interference is actually possible. No confidence interval or effect size is reported for this check because the quantity it measures has no sampling variance to describe.
6. **No universal prerequisite claim.** Earlier acquisition in a curriculum is evidence about transfer under that protocol, not proof of a formal prerequisite relationship.
7. **Potential task-family confounds.** Arithmetic tasks share representations and input structure, so transfer behavior may differ substantially in other domains.
8. **Practical tolerance.** The five-percentage-point tolerance in the isolation check is a sanity bound retained from the original design, not a statistical equivalence margin -- a violation would indicate a bug, not a surprising result.
9. **The signed-domain follow-up (Section 4.4) tested one specific expansion.** `{0,...,9}`-scale ranges to `{-9,...,9}`-scale ranges, on the same six arithmetic tasks, same architecture, same seeds. It demonstrates that transfer *can* be domain-sensitive for this system, not that it always is, nor how it would generalize to other kinds of distribution shift (different scale, different sparsity, non-arithmetic tasks, larger models). Expanding a domain also changes more than "whether negative values are present" -- it changes the input distribution as a whole (Section 11 of the underlying plan raises this explicitly); this experiment tests domain/distribution sensitivity, not an abstract, distribution-independent property of negative numbers.

## 8. Reproducibility

The repository contains the experiment drivers, regression tests, workflow configuration, raw CSV outputs, statistical summaries, and plots. CI reruns the experiments and uploads the generated artifacts. The skill-isolation invariant check additionally records the exact seed, target sequence, acquisition strategy, source skill, compatibility diagnostics, pre/post accuracy, the recorded accuracy change, and the pass/fail decision for each measured run.

## 9. Conclusion

This prototype demonstrates a controlled approach to continual skill acquisition in which the system can choose among reuse, clone-and-adapt, and scratch learning while preserving previously acquired skills through independent storage.

The most defensible conclusion is not that cloning always improves learning. Instead, the experiments show that prior knowledge can have positive, negligible, or negative effects depending on the target and source relationship, that this transfer behavior is itself sensitive to the input domain rather than a fixed property of a source-target pair (Section 4.4), while the isolated-skill mechanism's design guarantees -- confirmed to hold in the implementation by the skill-isolation check -- add new capabilities without modifying previously stored skills' parameters under the tested conditions.

The work therefore establishes a useful experimental foundation for a broader research program: first characterize reliable skill acquisition (and confirm the isolation guarantee the architecture depends on), then investigate the factors that determine why one skill transfers well and another does not and how sensitive that determination is to the task distribution, and eventually test retention in a setting where interference is actually possible.

## 10. Reproducibility checklist

- [x] Reuse / clone / scratch routes implemented.
- [x] Corrected reuse gate requires independent solved-target evidence.
- [x] Genuine scratch fallback retained.
- [x] Matched seeds used for paired comparisons.
- [x] Held-out evaluation separated from training and controller data.
- [x] Skill-isolation invariant evaluated on stable skill-specific evaluation sets.
- [x] Isolation-check tolerance declared before interpretation; documented as a sanity bound, not a statistical margin.
- [x] Per-seed isolation-check data and summary outputs generated.
- [x] CI executes the skill-isolation check and regression tests.
- [x] Isolation check's summary and paper text describe it as an implementation invariant, not a statistical retention experiment (no bootstrap CI or effect size reported for a zero-variance quantity).
- [ ] Final prerequisite-matrix expansion and its final statistical tables should be included only after that experiment is the authoritative merged result.
- [ ] A genuine empirical retention test (interference-risking baseline extended to this experiment's task sequences) is noted as future work, not yet implemented.
- [x] Signed-domain follow-up: non-negative domain preserved as the default and verified byte-identical to pre-follow-up results; signed domain added as an explicit, separate configuration (not a silent range change).
- [x] Signed-domain comparisons use matched seeds, identical architecture/optimizer/budget/controller/thresholds/data-role separation to the non-negative baseline -- only the operand domain differs.
- [x] Signed-domain results reported with paired statistics (mean, std, paired t-test) and explicitly flagged as conditional on the tested domain expansion, not a general claim about negative numbers.
