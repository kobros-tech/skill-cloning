# Publication Tables

These tables are intended as the compact statistical presentation for the final paper. Values are copied from the reproducible experiment outputs already present in the repository; no new experiment is implied by this document.

## Table 1 — Relatedness-pair acquisition

| Source → target | Mean frozen compatibility | Mean paired speedup (scratch / clone) | Paired p-value | Interpretation |
|---|---:|---:|---:|---|
| Multiplication → Squares | 0.001 | 1.26× | 3.9×10⁻⁹ | Positive transfer despite low frozen-output score |
| Multiplication → Powers | 0.915 | 2.31× | 6.0×10⁻¹¹ | Strong positive transfer |
| Addition → Subtraction | 0.156 | 0.33× | 2.3×10⁻¹² | Negative transfer |
| Addition → Multiplication | 0.005 | 1.12× | 0.013 | Small positive control effect |

The table demonstrates why compatibility score should not be interpreted as a complete predictor of transfer usefulness.

## Table 2 — Historical retention comparison

The original shared-network experiment showed large degradation of old-task performance after later tasks were trained. The isolated clone-and-adapt mechanism kept prior skills stable. The original report gives the following paired MSE comparisons:

| Task | Shared-network mean MSE | Proposed mean MSE | Paired t-test p | Wilcoxon p |
|---|---:|---:|---:|---:|
| Addition | 1171.7 | 0.15 | 2.7×10⁻¹² | 0.00006 |
| Subtraction | 1963.2 | 0.14 | 1.1×10⁻¹³ | 0.00006 |
| Multiplication | 446.4 | 0.38 | 8.7×10⁻¹⁰ | 0.00006 |
| Powers | 0.29 | 3.71 | 7.4×10⁻⁴ | 0.026 |

**Important:** the powers row belongs to the historical pre-fix analysis documented in `results/report.md`. It must not be presented as the current controller's final result without also presenting the corrected rerun. The report explicitly states that the false-reuse bug changed the powers result.

## Table 3 — Skill-isolation invariant check (not a statistical retention experiment)

| Quantity | Definition | Reported result |
|---|---|---|
| Seeds per sequence | Matched deterministic seeds | 15 |
| Sequences | Representative 3-skill arithmetic sequences | 4 |
| Evaluation set | Stable, skill-specific held-out set | 300 examples / check |
| Practical tolerance (sanity bound) | Maximum allowed absolute accuracy loss before a check is flagged as a bug | 5 percentage points |
| Max absolute accuracy change observed | max(\|post accuracy − pre accuracy\|) across all checks | 0.0 (exact, every check) |
| All deltas exactly zero | Whether every check landed at exactly 0.0 | True |
| Checks passing the sanity bound | Fraction with delta ≥ −0.05 | 100% |

No bootstrap interval or effect size is reported here. In the current architecture, a stored skill's parameters are never modified once acquired, and each check re-evaluates the same frozen network on the same deterministic batch before and after — so a delta of exactly 0.0 is a mathematical guarantee of the code as written, not a sampled outcome. Computing a confidence interval or effect size over a quantity with no variance would misrepresent this as statistical evidence; it isn't. This table verifies an **implementation invariant** (the isolation guarantee holds, with no bug leaking a later gradient update into a frozen skill) — it is a regression check, not an empirical measurement of resistance to catastrophic forgetting. A genuine empirical retention result would require a comparison arm in which interference is actually possible (e.g. extending Table 2's shared-network baseline to this experiment's sequences), which is not part of this table and is left as future work.

## Table 3.5 — Signed-domain robustness (non-negative vs. signed domain, matched seeds)

**Speedup by pair** (scratch epochs / clone epochs; forced-clone methodology, same as the pairwise experiments this extends):

| Pair | Non-negative speedup | Signed speedup | Paired diff (nonneg − signed) | Paired t-test p | Direction reversed? |
|---|---|---|---|---|---|
| multiplication → powers | 2.173 ± 0.530 | 0.719 ± 0.177 | 1.455 ± 0.541 | 5.6×10⁻⁸ | **Yes** |
| multiplication → squares | 1.225 ± 0.178 | 1.010 ± 0.040 | 0.215 ± 0.150 | 7.0×10⁻⁵ | No (erodes to null) |
| addition → subtraction | 0.408 ± 0.100 | 1.001 ± 0.224 | −0.594 ± 0.243 | 1.9×10⁻⁷ | Yes (negative transfer neutralizes) |
| addition → multiplication (null control) | 1.145 ± 0.138 | 1.176 ± 0.128 | −0.031 ± 0.218 | 0.59 (n.s.) | No |

**Acquisition success rate** (fixed-target prerequisite-history matrix, fixed training budget, 15 seeds/condition): squares' success rate drops from 13.3-20.0% (non-negative, already low across all three prior-skill histories) to exactly 0.0% (signed, all three histories). Subtraction, division, and powers remain at 100% success in both domains.

**Compatibility-score domain sensitivity**: division's frozen compatibility score against an addition parent drops from 0.28-0.31 (non-negative; above `τ_clone=0.15`, controller chooses `clone` 15/15 seeds) to 0.03-0.06 (signed; below `τ_clone`, controller chooses `scratch` 14-15/15 seeds) — the controller's own decisions change with the domain, not only the transfer outcome those decisions were meant to predict.

**Sign-specific diagnostic breakdown** (signed domain, trained clone network, held-out accuracy at ±0.5 tolerance):

| Pair | Quadrant | Mean abs. error | Accuracy |
|---|---|---|---|
| multiplication → powers | positive base | 0.230 | 93.2% |
| multiplication → powers | negative base, even exponent | 0.410 | 75.4% |
| multiplication → powers | negative base, odd exponent | 0.890 | 67.0% |
| multiplication → squares | positive base | 0.486 | 67.4% |
| multiplication → squares | negative base | 0.489 | 69.7% |
| addition → subtraction | same-sign, (+,+) | 0.220 | 94.9% |
| addition → subtraction | same-sign, (-,-) | 0.206 | 96.7% |
| addition → subtraction | mixed-sign, (+,-) | 0.455 | 70.6% |
| addition → subtraction | mixed-sign, (-,+) | 0.458 | 71.0% |

These breakdowns are diagnostic, not causal proof: they show the domain effects are *plausibly explained* by the added sign structure (negative-base/odd-exponent cases, which don't exist in the non-negative domain, are the hardest quadrant for multiplication→powers; mixed-sign inputs are harder than same-sign inputs for addition→subtraction; multiplication→squares shows no sign asymmetry, consistent with squares being sign-invariant) rather than establishing the mechanism definitively.

Full per-seed data: `results/signed_domain_pairs.csv`, `results/signed_domain_history.csv`, `results/signed_domain_sign_breakdown.csv`.

## Table 4 — Main claim boundaries

| Supported by the experiments | Not established by the experiments |
|---|---|
| Skills can be stored independently, and the implementation's isolation guarantee (frozen skills are never modified by later acquisitions) is confirmed to hold in code. | Universal absence of catastrophic forgetting -- no interference-risking baseline was tested in this check. |
| Reuse, clone-and-adapt, and scratch are all useful acquisition routes. | Universal superiority of cloning. |
| Transfer can be positive or negative depending on the source-target pair. | A universal prerequisite hierarchy. |
| Frozen compatibility score alone does not fully predict transfer benefit. | Generalization to large models or arbitrary domains. |
| Matched-seed evaluation can separate acquisition reliability and efficiency. | An empirical (as opposed to invariant-by-construction) test of retention under interference. |
| Transfer behavior (direction and magnitude) can be sensitive to the operand domain/distribution, for at least 2 of 4 tested pairs. | That negative numbers specifically, independent of distribution shift, cause the change -- the non-negative → signed expansion changes the input distribution as a whole, not only "whether negative values are present." |
