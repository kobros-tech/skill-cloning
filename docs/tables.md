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

## Table 3 — Current retention experiment

| Quantity | Experimental definition | Reported result |
|---|---|---|
| Seeds per sequence | Matched deterministic seeds | 15 |
| Sequences | Representative 3-skill arithmetic sequences | 4 |
| Evaluation set | Stable, skill-specific held-out set | 300 examples / check |
| Practical retention tolerance | Maximum allowed absolute accuracy loss | 5 percentage points |
| Mean retention delta | Post accuracy − pre accuracy | 0.0000 in reported checks |
| Retention pass rate | Fraction with delta ≥ −0.05 | 100% in reported checks |
| Bootstrap interval | 95% bootstrap interval for mean delta | [0, 0] in reported checks |

The current retention result should be described as **no measurable forgetting under the tested isolated-skill mechanism**, not as proof that catastrophic forgetting cannot occur in general.

## Table 4 — Main claim boundaries

| Supported by the experiments | Not established by the experiments |
|---|---|
| Skills can be stored independently and retained while new skills are acquired. | Universal absence of catastrophic forgetting. |
| Reuse, clone-and-adapt, and scratch are all useful acquisition routes. | Universal superiority of cloning. |
| Transfer can be positive or negative depending on the source-target pair. | A universal prerequisite hierarchy. |
| Frozen compatibility score alone does not fully predict transfer benefit. | Generalization to large models or arbitrary domains. |
| Matched-seed evaluation can separate acquisition reliability, efficiency, and retention. | Causal explanations for why every task transfers differently. |
