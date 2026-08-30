# ICLR 2027 Submission Plan

This document tracks the publication-focused work for the `submission/iclr-2027` branch. It is intentionally separate from `main`: the main branch remains the stable research implementation, while this branch is used to harden the scientific presentation, reproducibility, and submission materials.

## Target contribution

The paper should make one central claim:

> The benefit of transferring a previously learned skill is conditional on the source-target relationship and input distribution; cloning is therefore a mechanism for testing and exploiting transfer, not a universally beneficial learning strategy.

The paper should avoid claiming a universal prerequisite hierarchy, universal superiority of cloning, or general resistance to catastrophic forgetting.

## Work packages

### P0 — Scientific integrity (must pass)

- [x] Keep the corrected reuse gate: compatibility alone does not establish that the source already solves the target.
- [x] Treat the fixed-target prerequisite matrix as the authoritative current quantitative analysis.
- [x] Report invalid prerequisite histories explicitly rather than silently dropping them.
- [x] Report matched-seed attrition in the signed-domain study.
- [x] Keep retention framed as an isolation/invariance check rather than a general anti-forgetting claim.
- [ ] Audit every number in the manuscript against the generated CSV/CI artifact.
- [ ] Ensure historical results are visibly separated from authoritative results.

### P1 — Experimental strengthening

Priority additions, in order:

1. Add a matched irrelevant-parent control where feasible: scratch vs clone from a relevant parent vs clone from an unrelated parent.
2. Report success rate separately from capped training cost.
3. Where feasible, report conditional convergence cost among successful runs in addition to the budget-capped metric.
4. Report held-out target performance alongside acquisition cost so speed does not substitute for quality.
5. If compute permits, increase the number of seeds for the headline comparisons, especially signed-domain comparisons with only 5/15 valid matched seeds.
6. Add a genuine shared-parameter interference baseline if we want to make an empirical catastrophic-forgetting comparison. Otherwise keep the current retention claim explicitly architectural.

### P2 — Scientific positioning

- [ ] Add a proper Related Work section covering continual/lifelong learning, transfer learning, modular/parameter-isolated learning, and adaptation from prior skills.
- [ ] Explain precisely what is new about the controlled skill-repository formulation and its experimental protocol.
- [ ] Distinguish the controller's compatibility score from a causal measure of transferability.
- [ ] State why the arithmetic environment is useful as a controlled testbed and what remains unknown outside it.

### P3 — ICLR manuscript

- [ ] Convert the current research report into a concise ICLR-style manuscript.
- [ ] Keep the main text within the 9-page limit.
- [ ] Move detailed tables, additional diagnostics, and implementation details into supplementary material where appropriate.
- [ ] Prepare anonymized author/title information for double-blind review.
- [ ] Add an explicit limitations section.
- [ ] Add an AI-use statement as required by the venue.
- [ ] Verify references, equations, captions, and statistical notation.

### P4 — Reproducibility package

- [x] Keep a single end-to-end `run_all.py` entry point.
- [x] Keep automated experiments and regression tests in CI.
- [x] Upload generated CI results as artifacts.
- [ ] Add a concise reproducibility guide with expected runtime and hardware assumptions.
- [ ] Record the exact commit/configuration corresponding to the paper's final numbers.
- [ ] Add a final results manifest containing experiment name, seed count, configuration, and output files.

## Submission freeze criteria

The branch should not be considered submission-ready until all of the following are true:

- every headline number is traceable to a generated result;
- no superseded result is presented as authoritative;
- all exclusions and failed prerequisite runs are reported;
- the main claims are supported by the strongest available controls;
- the paper clearly separates mechanism verification from empirical transfer/forgetting evidence;
- the manuscript fits the venue format and is anonymized;
- a clean checkout can reproduce the reported analyses.
