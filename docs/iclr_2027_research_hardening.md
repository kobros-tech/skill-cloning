# ICLR 2027 Research Hardening

This document records the concrete changes being made on `submission/iclr-2027` before manuscript generation.

## Scientific priorities

1. **Baseline separation.** Distinguish the benefit of a pretrained initialization from the benefit of a task-relevant parent by comparing scratch, relevant-parent clone, and unrelated-parent clone where the existing framework permits a clean matched comparison.
2. **Numerical audit.** Every headline number in the manuscript must trace to an experiment artifact or a deterministic calculation from one. Historical results and authoritative fixed-target results must remain clearly separated.
3. **Contribution framing.** The paper should make a narrow, testable claim: transfer is heterogeneous and depends on source-target relationship and input distribution; cloning is a mechanism for testing that hypothesis, not a universal solution.
4. **Related work.** Position the work against continual learning, transfer learning, modular/parameter-isolated networks, progressive/lateral transfer, and related skill-reuse approaches.
5. **Statistical reporting.** Report matched sample sizes, paired tests, effect sizes where appropriate, and failure/attrition counts. Do not treat invalid prerequisite runs as successful observations or silently discard them.
6. **Forgetting baseline.** Retain the isolated-skill invariant as an implementation check. Where computationally feasible, add a shared-parameter interference comparison so that forgetting is evaluated empirically rather than inferred from parameter isolation.

## Rules for this branch

- Do not invent or manually tune experimental results to improve the narrative.
- New results must come from executable experiments and be recorded in reproducible artifacts.
- Negative transfer and null results remain first-class findings.
- Exploratory low-power comparisons (for example 5/15 matched signed-domain seeds) must remain explicitly labeled as such.
- The Word/PDF manuscript is deferred until the scientific content is frozen.

## Current strongest evidence

The fixed-target history matrix is the authoritative evidence for history-dependent transfer. The signed-domain study is a robustness/domain-sensitivity follow-up. The retention study is an architectural isolation check unless a genuine shared-parameter comparison is added.

## Reviewer questions this branch must answer

- Is the contribution distinct from ordinary transfer learning or continual learning with parameter isolation?
- Does cloning help because the parent is relevant, or merely because it is pretrained?
- Are conclusions robust to random seed and target difficulty?
- Are domain effects separable from source-acquisition failures?
- Are stopping-budget choices driving the apparent efficiency effects?
- Does the method preserve prior skills because of a demonstrated advantage, or simply because their parameters are never updated?
- Can an independent researcher reproduce the headline tables from the repository?
