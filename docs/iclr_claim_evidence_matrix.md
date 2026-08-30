# ICLR Claim–Evidence Matrix

This ledger is a guardrail against accidental overclaiming. Before submission, every headline claim in the manuscript should map to a reproducible experiment and a concrete limitation.

| Claim | Evidence | Current strength | Limitation / required action |
|---|---|---|---|
| Prior knowledge can help target acquisition | Fixed-target prerequisite matrix; powers improve from 471.6 to 237.3 epochs as history expands | Strong under the tested protocol | Arithmetic task family and controller-specific |
| Prior knowledge can hurt target acquisition | Fixed-target matrix; division worsens from 515.2 to 621.8 epochs | Strong under the tested protocol | Do not generalize to arbitrary tasks |
| Prior knowledge need not help | Fixed-target squares success remains low and decreases with expanded history | Moderate | Low success rate limits inference |
| Transfer is domain-sensitive | Signed-domain paired comparisons | Moderate | Two key comparisons have only 5/15 valid matched seeds |
| Multiplication→powers changes direction across domains | 2.217× → 0.703×, p=0.00168 | Moderate/strong statistical signal | Effective n=5; signed source acquisition attrition |
| Multiplication→squares shows no conventional domain difference | 1.265× → 1.031×, p=0.0757 | Weak/null evidence | n=5; absence of significance is not proof of equivalence |
| Addition→subtraction changes across domains | 0.408× → 1.001×, p≈1.88e−7 | Strong under protocol | Domain comparison remains task-family specific |
| Addition→multiplication is a useful null control | 1.145× → 1.176×, p=0.591 | Moderate | A non-significant result does not establish exact equality |
| Previously stored skills are unchanged by later independent acquisition | Parameter-isolation invariant and retention test | Strong as an implementation property | Not evidence of general catastrophic-forgetting resistance |
| Frozen compatibility is not sufficient to establish reuse | Corrected controller uses compatibility plus independent solve accuracy | Strong mechanistic design choice | Need broader validation of controller thresholds |
| There is no universal prerequisite hierarchy | Heterogeneous target/history outcomes | Moderate | This is a negative/generalization claim within the tested family, not a theorem |

## Rules for the final manuscript

- Use **"under the tested protocol"** when conclusions depend on the experimental environment.
- Use **"consistent with"** when the experiment cannot establish the proposed mechanism causally.
- Do not call the retention experiment evidence of general catastrophic-forgetting prevention unless a shared-parameter interference baseline is added.
- Do not call the signed-domain 5/15 comparisons definitive.
- Do not convert a non-significant p-value into a claim of equivalence.
- Do not silently replace the authoritative fixed-target results with the historical relatedness-pair results.
