# ICLR Claim–Evidence Matrix

This ledger is a guardrail against accidental overclaiming. Every headline claim should map to a reproducible experiment and a concrete limitation.

| Claim | Evidence | Current strength | Limitation / required action |
|---|---|---|---|
| Prior knowledge can help target acquisition | Fixed-target prerequisite matrix; powers improve from 471.6 to 237.3 epochs as history expands | Strong under the tested protocol | Arithmetic task family and controller-specific |
| Prior knowledge can hurt target acquisition | Fixed-target matrix; division worsens from 515.2 to 617.5 budgeted epochs | Strong under the tested protocol | Do not generalize to arbitrary tasks |
| Prior knowledge need not help | Fixed-target squares success remains low and decreases with expanded history | Moderate | Low success rate limits inference |
| Relevant parent choice matters beyond generic pretraining | Three-arm matched control with relevant clone, unrelated clone, and scratch; 15 matched seeds per target; paired t-tests and Wilcoxon tests | Strong under the tested protocol | Small arithmetic benchmark; report multiplicity policy and effect sizes carefully |
| Relevant cloning beats scratch for multiplication | Relevant minus scratch = -117.2 steps, paired t p=0.0127; Holm-adjusted p=0.0127 | Moderate | One target and one architecture; effect is seed-matched, not universal |
| Relevant cloning beats unrelated cloning for multiplication | Relevant minus unrelated = -298.3 steps, paired t p=5.91e-6; Holm-adjusted p=1.77e-5 | Strong under the tested protocol | One target family; use as evidence that parent identity matters |
| Unrelated cloning is worse than scratch for multiplication | Unrelated minus scratch = +181.1 steps, paired t p=9.18e-4; Holm-adjusted p=0.00184 | Strong under the tested protocol | Does not imply all unrelated parents are harmful |
| Relevant cloning beats scratch for powers | Relevant minus scratch = -260.5 steps, paired t p=6.01e-11; Holm-adjusted p=3.61e-10 | Strong under the tested protocol | One target family and finite seeds |
| Relevant cloning beats unrelated cloning for powers | Relevant minus unrelated = -125.9 steps, paired t p=1.01e-6; Holm-adjusted p=4.03e-6 | Strong under the tested protocol | One target family and finite seeds |
| Unrelated cloning beats scratch for powers | Unrelated minus scratch = -134.7 steps, paired t p=2.40e-8; Holm-adjusted p=1.20e-7 | Strong under the tested protocol | Shows that generic pretraining can help here; relevant parent still performs better |
| Transfer is domain-sensitive | Signed-domain paired comparisons | Moderate | Two key comparisons have only 5/15 valid matched seeds |
| Multiplication→powers changes direction across domains | 2.217× → 0.703×, p=0.00168 | Moderate/strong statistical signal | Effective n=5; signed source acquisition attrition |
| Multiplication→squares shows no conventional domain difference | 1.265× → 1.031×, p=0.0757 | Weak/null evidence | n=5; absence of significance is not proof of equivalence |
| Addition→subtraction changes across domains | 0.408× → 1.001×, p≈1.88e−7 | Strong under protocol | Domain comparison remains task-family specific |
| Addition→multiplication is a useful null control | 1.145× → 1.176×, p=0.591 | Moderate | A non-significant result does not establish exact equality |
| Previously stored skills are unchanged by later independent acquisition | Parameter-isolation invariant and retention test | Strong as an implementation property | Not evidence of general catastrophic-forgetting resistance |
| Frozen compatibility is not sufficient to establish reuse | Corrected controller uses compatibility plus independent solve accuracy | Strong mechanistic design choice | Threshold calibration remains a limitation |
| There is no universal prerequisite hierarchy | Heterogeneous target/history outcomes | Moderate | This is a negative/generalization claim within the tested family, not a theorem |

## Statistical reporting policy

The three-arm parent-control experiment uses the same seed, target data, architecture, optimizer, and budget for each arm. The inferential unit is the **matched seed within target**, not the collection of arm means.

The experiment writes:

- `results/unrelated_parent_control.csv` — per-seed arm-level outcomes;
- `results/unrelated_parent_control_stats.csv` — paired differences and paired tests.

For the six parent-control hypotheses, Holm's step-down correction is applied to the paired t-test p-values. The resulting adjusted p-values are:

| Target | Comparison | Raw paired t p | Holm-adjusted p |
|---|---|---:|---:|
| multiplication | relevant − scratch | 0.012672 | 0.012672 |
| multiplication | relevant − unrelated | 5.91×10^-6 | 1.77×10^-5 |
| multiplication | unrelated − scratch | 9.18×10^-4 | 1.84×10^-3 |
| powers | relevant − scratch | 6.01×10^-11 | 3.61×10^-10 |
| powers | relevant − unrelated | 1.01×10^-6 | 4.03×10^-6 |
| powers | unrelated − scratch | 2.40×10^-8 | 1.20×10^-7 |

The six adjusted tests remain below 0.05. This does not remove the finite-task/finite-seed limitations, but it prevents the parent-control conclusions from depending on uncorrected multiple testing.

The signed-domain comparisons remain a separate family of analyses and are reported with their actual matched sample sizes. In particular, the multiplication→powers and multiplication→squares comparisons use only 5/15 valid matched seeds.

## Rules for the final manuscript

- Use **"under the tested protocol"** when conclusions depend on the experimental environment.
- Use **"consistent with"** when the experiment cannot establish the proposed mechanism causally.
- Do not call the retention experiment evidence of general catastrophic-forgetting prevention unless a shared-parameter interference baseline is added.
- Do not call the signed-domain 5/15 comparisons definitive.
- Do not convert a non-significant p-value into a claim of equivalence.
- Do not silently replace the authoritative fixed-target results with historical relatedness-pair results.
- Do not use arm means as substitutes for matched-seed inference.
- Do not claim universal superiority of relevant cloning; the evidence is limited to the tested targets, architecture, and protocol.
