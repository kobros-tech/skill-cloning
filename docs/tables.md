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

## Table 2 — Fixed-target prerequisite matrix

The expanded experiment holds the target fixed while varying the prior-skill history. This separates the effect of relevant prior knowledge from the mere presence of additional training history.

| Target | No prior skill | Addition only | Addition + Multiplication |
|---|---:|---:|---:|
| Subtraction | 33.5 epochs | 62.3 epochs | 73.2 epochs |
| Division | 515.2 epochs | 616.2 epochs | 617.5 epochs |
| Squares | 20.0% success | 20.0% success | 13.3% success |
| Powers | 471.6 epochs | 355.2 epochs | 237.3 epochs |

For subtraction, division, and powers, the values are acquisition epochs to the declared criterion. Squares is reported as success rate because many runs do not reach the acquisition criterion within the allowed budget. The results show heterogeneous transfer: additional prior skills substantially help powers, while division exhibits negative transfer and squares remains difficult.

These results are evidence about transfer under the tested protocol, not proof of formal mathematical prerequisite relationships.

## Table 3 — Retention mechanism check

| Quantity | Experimental definition | Reported result |
|---|---|---|
| Seeds per sequence | Matched deterministic seeds | 15 |
| Sequences | Representative 3-skill arithmetic sequences | 4 |
| Evaluation set | Stable, skill-specific held-out set | 300 examples / check |
| Practical retention tolerance | Maximum diagnostic accuracy loss | 5 percentage points |
| Repeated-check mean delta | Post accuracy − pre accuracy | 0.0000 in reported checks |
| Retention pass rate | Fraction with delta ≥ −0.05 | 100% in reported checks |

**Interpretation:** these zero-change checks are consistent with the isolated-skill invariant: the stored parent network is not modified during later skill acquisition, and the same network is evaluated on the same retention set. They are therefore an implementation/mechanism check, not a statistical demonstration that the system is robust to catastrophic forgetting. Bootstrap confidence intervals and effect sizes for this quantity are intentionally not presented as evidence of an interference effect.

## Table 4 — Main claim boundaries

| Supported by the experiments | Not established by the experiments |
|---|---|
| Skills can be stored independently and remain unchanged while new skills are acquired. | Universal absence of catastrophic forgetting. |
| Reuse, clone-and-adapt, and scratch are all useful acquisition routes. | Universal superiority of cloning. |
| Transfer can be positive or negative depending on the source-target pair. | A universal prerequisite hierarchy. |
| Frozen compatibility score alone does not fully predict transfer benefit. | Generalization to large models or arbitrary domains. |
| Matched-seed evaluation can separate acquisition reliability, efficiency, and mechanism-level retention checks. | Causal explanations for why every task transfers differently. |
| The isolated-skill mechanism prevents later training from overwriting a stored parent skill in the tested protocol. | Robustness to interference when later training is allowed to modify shared parameters. |

## Historical-result note

The repository also contains an earlier shared-network comparison in `results/report.md`. That analysis predates the corrected reuse controller and includes a historical Powers comparison that is not part of the current final controller results. It is intentionally not reproduced as a publication table here, to avoid mixing pre-fix and current results. The current paper should use the corrected experiments and the fixed-target matrix above as the authoritative quantitative evidence.
