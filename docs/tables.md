# Publication Tables

These tables are intended as the compact statistical presentation for the final paper. Values are copied from the reproducible experiment outputs already present in the repository; no new experiment is implied by this document.

## Table 1 — Relatedness-pair acquisition

| Source $\rightarrow$ target | Mean frozen compatibility | Mean paired speedup (scratch / clone) | Paired $p$-value | Interpretation |
|---|---:|---:|---:|---|
| Multiplication $\rightarrow$ Squares | 0.001 | $1.26\times$ | $3.9\times10^{-9}$ | Positive transfer despite low frozen-output score |
| Multiplication $\rightarrow$ Powers | 0.915 | $2.31\times$ | $6.0\times10^{-11}$ | Strong positive transfer |
| Addition $\rightarrow$ Subtraction | 0.156 | $0.33\times$ | $2.3\times10^{-12}$ | Negative transfer |
| Addition $\rightarrow$ Multiplication | 0.005 | $1.12\times$ | 0.013 | Small positive control effect |

**Historical-result note:** Table 1 summarizes the earlier source$\rightarrow$target analysis. The pre-fix Powers generalization result associated with the false-reuse controller is superseded and is not represented by this table's speedup statistic.

The table demonstrates why compatibility score should not be interpreted as a complete predictor of transfer usefulness.

## Table 2 — Fixed-target prerequisite matrix

The expanded experiment holds the target fixed while varying the prior-skill history. This separates the effect of relevant prior knowledge from the mere presence of additional training history.

| Target | No prior skill | Addition only | Addition + Multiplication |
|---|---:|---:|---:|
| Subtraction | 33.5 budgeted target-adaptation steps | 62.3 budgeted target-adaptation steps | 73.2 budgeted target-adaptation steps |
| Division | 515.2 budgeted target-adaptation steps | 616.2 budgeted target-adaptation steps | **621.8 budgeted target-adaptation steps** |
| Squares | 20.0% success | 20.0% success | 13.3% success |
| Powers | 471.6 budgeted target-adaptation steps | 355.2 budgeted target-adaptation steps | 237.3 budgeted target-adaptation steps |

For subtraction, division, and powers, the values are **mean budgeted target-adaptation steps among the 15 nominal seeds whose prerequisite history was valid**. If the target itself is attempted but does not reach the acquisition criterion, its `adaptation_steps` value is the full 1500-epoch budget and is included in the mean. A seed whose prerequisite history is invalid is excluded from the target metric because the target was never attempted. These values are therefore budgeted acquisition cost, not mean convergence time among successful target acquisitions.

If a requested prerequisite fails, that history is marked invalid and the failed prerequisite is never exposed to the controller as an acquired skill; target metrics are not computed for that seed. Squares is reported as success rate because only a small minority of runs reach the acquisition criterion within the allowed budget.

The results show heterogeneous transfer: additional prior skills substantially help powers, while division exhibits negative transfer and squares remains difficult. These results are evidence about transfer under the tested protocol, not proof of formal mathematical prerequisite relationships.

## Table 3 — Retention mechanism check

| Quantity | Experimental definition | Reported result |
|---|---|---|
| Seeds per sequence | Matched deterministic seeds | 15 |
| Sequences | Representative 3-skill arithmetic sequences | 4 |
| Evaluation set | Stable, skill-specific held-out set | 300 examples / check |
| Practical retention tolerance | Maximum diagnostic accuracy loss | 5 percentage points |
| Repeated-check mean delta | Post accuracy $-$ pre accuracy | 0.0000 in reported checks |
| Maximum absolute accuracy change | $\max(|\text{post accuracy}-\text{pre accuracy}|)$ across all checks | 0.0 |
| All deltas exactly zero | Whether every check landed at exactly 0.0 | True |
| Retention pass rate | Fraction with delta $\geq-0.05$ | 100% |

**Interpretation:** these zero-change checks are consistent with the isolated-skill invariant: the stored parent network is not modified during later skill acquisition, and the same network is evaluated on the same retention set. They are therefore an **implementation/mechanism check, not a statistical demonstration that the system is robust to catastrophic forgetting**.

Bootstrap confidence intervals and effect sizes for this quantity are intentionally not presented as evidence of an interference effect. A genuine empirical retention experiment would require a comparison arm in which later learning can actually modify parameters supporting earlier skills, such as a shared-parameter baseline.

## Table 3.5 — Signed-domain robustness

The signed-domain follow-up compares the original non-negative operand domain with an expanded signed domain using matched seeds and the same acquisition protocol.

### Speedup by pair

Speedup is defined as scratch epochs divided by clone epochs. Domain comparisons are paired by seed **only among seeds for which both domain conditions produced a valid source acquisition and therefore a valid clone/scratch pair**. The number of valid matched seeds is reported explicitly because source-acquisition failures reduce the paired sample size. Runs that fail to reach the target criterion contribute the declared 1500-epoch cap to the epoch ratio; therefore these ratios are budget-capped training-cost comparisons rather than convergence-time comparisons restricted to successful runs. The mean $\pm$ standard deviation values are descriptive summaries of per-seed speedup ratios; the paired $t$-test is performed on those per-seed ratios, not on the displayed summary values.

| Pair | Valid matched seeds | Non-negative speedup | Signed speedup | Paired difference (non-negative $-$ signed) | Paired $t$-test $p$ | Direction reversed? |
|---|---:|---:|---:|---:|---:|---|
| multiplication $\rightarrow$ powers | 5/15 | $2.217\pm0.530$ | $0.703\pm0.141$ | $1.514\pm0.451$ | 0.00168 | **Yes** |
| multiplication $\rightarrow$ squares | 5/15 | $1.265\pm0.178$ | $1.031\pm0.069$ | $0.235\pm0.220$ | 0.0757 | No — erodes toward null |
| addition $\rightarrow$ subtraction | 15/15 | $0.408\pm0.100$ | $1.001\pm0.224$ | $-0.594\pm0.243$ | $1.88\times10^{-7}$ | Yes — negative transfer neutralizes |
| addition $\rightarrow$ multiplication (null control) | 15/15 | $1.145\pm0.138$ | $1.176\pm0.128$ | $-0.031\pm0.218$ | 0.591 | No |

The powers and squares comparisons have only **5/15 valid matched seeds** because the signed-domain multiplication prerequisite failed acquisition in 10 seeds. The 10 failed seeds are not silently treated as zero or full-budget source-acquisition speedups and are excluded from the paired speedup test. This attrition is itself reported as part of the signed-domain result.

The multiplication $\rightarrow$ powers comparison remains statistically different under the paired test ($p=0.00168$) and reverses direction. For multiplication $\rightarrow$ squares, the current paired comparison is **not conventionally statistically significant** ($p=0.0757$); the appropriate conclusion is that the observed positive transfer erodes toward no effect, not that a statistically significant domain difference has been established.

### Acquisition success rate

For the fixed-target prerequisite-history matrix, squares' success rate is 20.0%, 20.0%, and 13.3% across the three non-negative prior histories, versus exactly 0.0% in the corresponding signed-domain histories.

Subtraction, division, and powers remain at 100% target-acquisition success in the histories that are valid for the target attempt. Some signed-domain histories have prerequisite-acquisition failures and are therefore marked invalid rather than treated as target failures.

### Compatibility-score domain sensitivity

Division's frozen compatibility score against an addition parent changes substantially with the domain:

* **Non-negative domain:** approximately 0.28--0.31, above $\tau_{\mathrm{clone}}=0.15$; the controller chooses clone in 15/15 seeds.
* **Signed domain:** approximately 0.03--0.06, below $\tau_{\mathrm{clone}}$; the controller switches toward scratch in 14--15/15 seeds.

Thus, the controller's own decision inputs are domain-sensitive, not only the resulting transfer outcome.

### Sign-specific diagnostic breakdown

The following diagnostics are from the signed-domain trained clone network, evaluated on held-out examples at $\pm0.5$ tolerance.

| Pair | Quadrant | Mean absolute error | Accuracy |
|---|---|---:|---:|
| multiplication $\rightarrow$ powers | positive base | 0.230 | 93.2% |
| multiplication $\rightarrow$ powers | negative base, even exponent | 0.410 | 75.4% |
| multiplication $\rightarrow$ powers | negative base, odd exponent | 0.890 | 67.0% |
| multiplication $\rightarrow$ squares | positive base | 0.486 | 67.4% |
| multiplication $\rightarrow$ squares | negative base | 0.489 | 69.7% |
| addition $\rightarrow$ subtraction | same-sign, $(+,+)$ | 0.220 | 94.9% |
| addition $\rightarrow$ subtraction | same-sign, $(-,-)$ | 0.206 | 96.7% |
| addition $\rightarrow$ subtraction | mixed-sign, $(+,-)$ | 0.455 | 70.6% |
| addition $\rightarrow$ subtraction | mixed-sign, $(-,+)$ | 0.458 | 71.0% |

These breakdowns are **diagnostic, not causal proof**. They show patterns consistent with the observed domain effects rather than establishing the mechanism definitively.

Full per-seed data are available in:

* `results/signed_domain_pairs.csv`
* `results/signed_domain_history.csv`
* `results/signed_domain_sign_breakdown.csv`

## Table 4 — Main claim boundaries

| Supported by the experiments | Not established by the experiments |
|---|---|
| Skills can be stored independently, and the implementation's isolation guarantee is confirmed to hold in code. | Universal absence of catastrophic forgetting — no interference-risking baseline was tested in the isolation check. |
| Reuse, clone-and-adapt, and scratch are all useful acquisition routes. | Universal superiority of cloning. |
| Transfer can be positive or negative depending on the source-target pair. | A universal prerequisite hierarchy. |
| Frozen compatibility score alone does not fully predict transfer benefit. | Generalization to large models or arbitrary domains. |
| Matched-seed evaluation can separate acquisition reliability, efficiency, and mechanism-level retention checks. | An empirical retention result under conditions where interference is possible. |
| Transfer behavior can be sensitive to the operand domain/distribution for the tested task pairs. | That negative numbers specifically, independent of distribution shift, cause the observed changes. |
| No statistically detectable domain difference was observed for the signed-domain null control under the tested manipulation. | Robustness to unrelated types of distribution shift or broader task families. |

## Historical-result note

The repository also contains an earlier shared-network comparison in `results/report.md`. That analysis predates the corrected reuse controller and includes historical results that are not part of the current final-controller evidence.

Those historical results are intentionally not reproduced as a publication table here, to avoid mixing pre-fix and current results.

The current paper should use the **corrected experiments and the fixed-target prerequisite matrix as the authoritative quantitative evidence**, with the signed-domain analysis presented separately as a robustness/domain-sensitivity follow-up.
