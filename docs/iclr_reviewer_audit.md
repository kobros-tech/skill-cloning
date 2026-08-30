# ICLR Reviewer Audit

Use this as a hostile-but-fair reviewer checklist before submission. A checked item should be supported by code, data, or a precise statement in the manuscript.

## 1. Contribution

- [ ] Can the contribution be stated in 2–3 sentences without describing the repository implementation first?
- [ ] Is the novelty relative to continual learning, transfer learning, modular learning, and parameter isolation explicit?
- [ ] Does the paper avoid presenting a controller heuristic as a general theory of prerequisites?

## 2. Experimental validity

- [ ] Are target, training, probe, solve/accuracy, and held-out data separated?
- [ ] Are all random seeds deterministic and paired where comparisons require pairing?
- [ ] Are unsuccessful runs and failed prerequisites reported rather than silently removed?
- [ ] Are stopping rules fixed independently of observed results?
- [ ] Is the 1500-epoch cap clearly described wherever capped cost is analyzed?
- [ ] Are reliability and efficiency reported as distinct outcomes?
- [ ] Is final held-out quality reported alongside acquisition cost?

## 3. Transfer claims

- [ ] Is scratch the baseline for every claimed acquisition benefit?
- [ ] Where possible, is an unrelated-parent clone control included?
- [ ] Is the distinction between frozen compatibility and actual transfer benefit explicit?
- [ ] Are historical pairwise results separated from the authoritative fixed-target matrix?
- [ ] Does the paper avoid implying that statistical significance proves a causal mechanism?

## 4. Domain-sensitivity claims

- [ ] Are domain configurations explicitly defined?
- [ ] Are matched seeds used for paired comparisons?
- [ ] Is matched-seed attrition reported?
- [ ] Are the 5/15 comparisons treated as lower-confidence evidence rather than pooled as if n=15?
- [ ] Is the null control included and interpreted as a null control rather than proof of equivalence?

## 5. Forgetting / retention

- [ ] Is the isolated-skill invariant mathematically defined?
- [ ] Is the current zero-change retention result described as an architectural/invariance check?
- [ ] Does the paper avoid claiming that this experiment demonstrates general immunity to catastrophic forgetting?
- [ ] If an empirical forgetting claim is desired, is there a shared-parameter comparison arm?

## 6. Statistics

- [ ] Is the statistical test specified for every headline comparison?
- [ ] Are paired tests used when the same seeds are compared?
- [ ] Are effect sizes or interpretable differences reported, not only p-values?
- [ ] Are confidence intervals reported when they materially clarify uncertainty?
- [ ] Are small effective sample sizes made prominent?
- [ ] Are multiple-comparison issues addressed or the scope of inference limited?

## 7. Reproducibility

- [ ] Does a clean checkout reproduce the reported analysis?
- [ ] Does CI execute the relevant experiments and tests?
- [ ] Is the exact code/configuration commit for the paper recorded?
- [ ] Are generated artifacts sufficient to audit headline numbers?
- [ ] Are dependencies pinned or otherwise reproducible enough for the claimed results?

## 8. Scope and limitations

The final paper should explicitly acknowledge:

- the small arithmetic task family;
- the small neural architecture;
- the finite seed count;
- the training-budget dependence of acquisition cost;
- the controller thresholds and their calibration;
- the signed-domain matched-seed attrition;
- the absence of a shared-network forgetting baseline unless one is added;
- the fact that transfer observations do not establish universal prerequisite structure.

## Reviewer red flags to eliminate

1. **"The authors cherry-picked successful runs."**
   - Counter with explicit failed-prerequisite accounting and matched-seed reporting.

2. **"Compatibility score is being treated as transferability."**
   - Counter by distinguishing frozen compatibility from post-adaptation benefit and, where possible, adding parent controls.

3. **"Zero forgetting is trivial because parameters are isolated."**
   - Agree. Label it an invariance check and do not overclaim.

4. **"The domain result has too few matched seeds."**
   - Report 5/15 prominently, treat it as exploratory/conditional evidence, and strengthen the experiment if compute permits.

5. **"The paper is only a toy arithmetic demonstration."**
   - Explain that arithmetic is deliberately used as a controlled environment for isolating transfer variables, while explicitly limiting generalization claims.
