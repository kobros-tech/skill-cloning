# Publication Tables — Deprecated

> **Do not use this file as the numerical source for the manuscript.**
>
> The previous version mixed historical/pre-fix source→target results with current workflow results. That made it possible for manuscript tables to contain numbers that could not be reconstructed from the latest workflow CSVs.

Use the authoritative, workflow-traceable publication tables instead:

- [`docs/publication_results.md`](publication_results.md)
- [`docs/publication_results_manifest.csv`](publication_results_manifest.csv)

The authoritative tables use only current-final-controller CSV outputs under `results/`, or statistics explicitly computed from those CSVs. Historical results remain in the repository for provenance but are not publication-authoritative.

## Important distinction

`results/relatedness_pairs.csv` and `results/relatedness_pairs_summary.csv` are the current **fixed-target prerequisite/history** experiment. They are not the historical source→target speedup experiment that was previously shown as Table 1 here.

For signed-domain paired speedups, use `results/signed_domain_pairs_comparison.csv`. Its per-seed paired analysis differs from the independent domain-level aggregation in `results/signed_domain_pairs_summary.csv`; the manuscript must not mix those statistics.
