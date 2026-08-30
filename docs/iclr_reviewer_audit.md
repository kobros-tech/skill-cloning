# ICLR Reviewer Audit

Use this as a hostile-but-fair reviewer checklist before submission. A checked item should be supported by code, data, or a precise statement in the manuscript.

## 1. Contribution

- [ ] Can the contribution be stated in 2–3 sentences without describing the repository implementation first?
- [ ] Is the novelty relative to continual learning, transfer learning, modular learning, and parameter isolation explicit?
- [x] Does the paper avoid presenting a controller heuristic as a general theory of prerequisites?

## 2. Experimental validity

- [x] Are target, training, probe, solve/accuracy, and held-out data separated?
- [x] Are all random seeds deterministic and paired where comparisons require pairing?
- [x] Are unsuccessful runs and failed prerequisites reported rather than silently removed?
- [x] Are stopping rules fixed independently of observed results?
- [x] Is the 1500-epoch cap clearly described wherever capped cost is analyzed?
- [x] Are reliability and efficiency reported as distinct outcomes?
- [x] Is final held-out quality reported alongside acquisition cost?

## 3. Transfer claims

- [x] Is scratch the baseline for every claimed acquisition benefit?
- [x] Is an unrelated-parent clone control included?
- [x] Is the distinction between frozen compatibility and actual transfer benefit explicit?
- [x] Are historical pairwise results separated from the authoritative fixed-target matrix?
- [x] Does the paper avoid implying that statistical significance proves a causal mechanism?
- [x] Does the parent-control implementation use only previously acquired skills?
- [x] Does the parent-control report matched-seed pairwise statistics rather than only arm means?

## 4. Domain-sensitivity claims

- [x] Are domain configurations explicitly defined?
- [x] Are matched seeds used for paired comparisons?
- [x] Is matched-seed attrition reported?
- [x] Are the 5/15 comparisons treated as lower-confidence evidence rather than pooled as if n=15?
- [x] Is the null control included and interpreted as a null control rather than proof of equivalence?

## 5. Forgetting / retention

- [x] Is the isolated-skill invariant mathematically defined?
- [x] Is the current zero-change retention result described as an architectural/invariance check?
- [x] Does the paper avoid claiming that this experiment demonstrates general immunity to catastrophic forgetting?
- [ ] If an empirical forgetting claim is desired, is there a shared-parameter comparison arm?

## 6. Statistics

- [x] Is the statistical test specified for every current headline comparison?
- [x] Are paired tests used when the same seeds are compared?
- [x] Are interpretable paired differences reported, not only p-values?
- [ ] Are confidence intervals reported when they materially clarify uncertainty?
- [x] Are small effective sample sizes made prominent?
- [ ] Is the final manuscript's multiple-comparison policy explicitly stated?

## 7. Reproducibility

- [x] Does a clean checkout reproduce the relevant experiment drivers?
- [x] Does CI execute the relevant experiments and tests?
- [ ] Is the exact final paper commit recorded after manuscript freeze?
- [x] Are generated artifacts sufficient to audit headline numbers?
- [ ] Are dependencies pinned or otherwise reproducible enough for the final submission?

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
   - Counter by distinguishing frozen compatibility from post-adaptation benefit and by using the relevant/unrelated/scratch parent control.

3. **"Zero forgetting is trivial because parameters are isolated."**
   - Agree. Label it an invariance check and do not overclaim.

4. **"The domain result has too few matched seeds."**
   - Report 5/15 prominently, treat it as exploratory/conditional evidence, and strengthen the experiment if compute permits.

5. **"The paper is only a toy arithmetic demonstration."**
   - Explain that arithmetic is deliberately used as a controlled environment for isolating transfer variables, while explicitly limiting generalization claims.

6. **"Relevant cloning may only look better because any pretrained network helps."**
   - Address with the three-arm matched control and seed-level paired statistics: relevant clone vs scratch, relevant clone vs unrelated clone, and unrelated clone vs scratch.
