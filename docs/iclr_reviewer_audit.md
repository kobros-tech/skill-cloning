# ICLR Reviewer Audit

Use this as a hostile-but-fair reviewer checklist before submission. A checked item should be supported by code, data, or a precise statement in the manuscript.

## 1. Contribution

- [x] Contribution can be summarized independently of repository implementation.
- [x] The paper distinguishes the proposed skill-reuse/clone-and-adapt mechanism from generic transfer learning and parameter isolation.
- [x] The paper avoids presenting a controller heuristic as a general theory of prerequisites.

## 2. Experimental validity

- [x] Target, training, probe, solve/accuracy, and held-out data are separated.
- [x] Random seeds are deterministic and paired where comparisons require pairing.
- [x] Unsuccessful runs and failed prerequisites are reported rather than silently removed.
- [x] Stopping rules are fixed independently of observed results.
- [x] The 1500-epoch cap is clearly described wherever capped cost is analyzed.
- [x] Reliability and efficiency are reported as distinct outcomes.
- [x] Final held-out quality is reported alongside acquisition cost.

## 3. Transfer claims

- [x] Scratch is the baseline for claimed acquisition benefit.
- [x] An unrelated-parent clone control is included.
- [x] Frozen compatibility is distinguished from actual post-adaptation transfer benefit.
- [x] Historical pairwise results are separated from the authoritative fixed-target matrix.
- [x] The paper avoids implying that statistical significance proves a causal mechanism.
- [x] The parent-control implementation uses only previously acquired skills.
- [x] The parent-control reports matched-seed pairwise statistics rather than only arm means.
- [x] Parent-control p-values have a stated Holm multiple-comparison policy.

## 4. Domain-sensitivity claims

- [x] Domain configurations are explicitly defined.
- [x] Matched seeds are used for paired comparisons.
- [x] Matched-seed attrition is reported.
- [x] The 5/15 comparisons are treated as lower-confidence evidence rather than pooled as if n=15.
- [x] The null control is included and interpreted as a null control rather than proof of equivalence.

## 5. Forgetting / retention

- [x] The isolated-skill invariant is mathematically defined.
- [x] The zero-change retention result is described as an architectural/invariance check.
- [x] The paper avoids claiming general immunity to catastrophic forgetting.
- [ ] An empirical shared-parameter forgetting baseline is not currently included; any final forgetting claim must remain limited to the isolation invariant.

## 6. Statistics

- [x] The statistical test is specified for current headline comparisons.
- [x] Paired tests are used when the same seeds are compared.
- [x] Interpretable paired differences are reported, not only p-values.
- [x] The parent-control family has a Holm correction policy.
- [x] Small effective sample sizes are made prominent.
- [x] The final paper should distinguish exploratory signed-domain analyses from the 15-seed parent-control analysis.

## 7. Reproducibility

- [x] A clean checkout reproduces the relevant experiment drivers.
- [x] CI executes the relevant experiments and tests.
- [x] Generated artifacts are sufficient to audit headline numbers.
- [ ] The exact final paper commit must be recorded after manuscript freeze.
- [ ] Dependency pinning should be reviewed before final archival submission.

## 8. Scope and limitations

The final paper should explicitly acknowledge:

- the small arithmetic task family;
- the small neural architecture;
- the finite seed count;
- the training-budget dependence of acquisition cost;
- the controller thresholds and their calibration;
- the signed-domain matched-seed attrition;
- the absence of a shared-network forgetting baseline;
- the fact that transfer observations do not establish universal prerequisite structure.

## Reviewer red flags to eliminate

1. **"The authors cherry-picked successful runs."**
   - Counter with explicit failed-prerequisite accounting and matched-seed reporting.

2. **"Compatibility score is being treated as transferability."**
   - Counter by distinguishing frozen compatibility from post-adaptation benefit and by using the relevant/unrelated/scratch parent control.

3. **"Zero forgetting is trivial because parameters are isolated."**
   - Agree. Label it an invariance check and do not overclaim.

4. **"The domain result has too few matched seeds."**
   - Report 5/15 prominently and treat it as exploratory/conditional evidence.

5. **"The paper is only a toy arithmetic demonstration."**
   - Explain that arithmetic is deliberately used as a controlled environment for isolating transfer variables, while explicitly limiting generalization claims.

6. **"Relevant cloning may only look better because any pretrained network helps."**
   - Address with the three-arm matched control and seed-level paired statistics. After Holm correction, all six paired t-test comparisons remain below 0.05 in this run.
