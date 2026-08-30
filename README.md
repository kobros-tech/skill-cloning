# Continual Skill Learning via Skill Cloning — Minimal Prototype

Implementation of a controlled prototype for continual skill acquisition. For an incoming task, the system can reuse an existing solved skill, clone a useful skill and adapt it, or learn from scratch. The experiments focus on acquisition reliability and transfer efficiency; a separate check confirms the implementation's skill-isolation guarantee holds (see the note below -- this is a code invariant, not a statistical retention experiment).

The current research plan is tracked in [Issue #3](https://github.com/kobros-tech/skill-cloning/issues/3). The final paper draft is in [`docs/final_paper.md`](docs/final_paper.md), with compact publication tables in [`docs/tables.md`](docs/tables.md).

## Main findings

- Prior knowledge can produce positive, negligible, or negative transfer depending on the source-target pair; cloning is not universally beneficial.
- Earlier relatedness-pair experiments show approximately 2.31× speedup for multiplication → powers, 1.26× for multiplication → squares, and negative transfer (0.33×) for addition → subtraction.
- **A signed-domain follow-up shows this transfer behavior is itself distribution-sensitive, not a fixed property of a source-target pair.** Expanding the operand domain from non-negative (`{0,...,9}`) to signed (`{-9,...,9}`) integers, under matched seeds and an otherwise identical protocol, **reversed** the multiplication → powers result (2.17× → 0.72×, p=5.6×10⁻⁸) and eroded multiplication → squares toward no effect (1.23× → 1.01×, p=7.0×10⁻⁵), while a null-control pair (addition → multiplication) stayed stable (p=0.59) -- ruling out "the manipulation itself always changes results" as an explanation. Squares' already-low acquisition success rate also collapsed from 13-20% to exactly 0% in the signed domain. See `docs/final_paper.md` Section 4.4 and `docs/tables.md` Table 3.5 for the full analysis, including a sign-specific diagnostic breakdown.
- A corrected reuse gate requires both compatibility evidence and independent target-solve accuracy, preventing a merely related but unsolved skill from being treated as a zero-training solution.
- A skill-isolation invariant check confirms the implementation never modifies a previously acquired skill's stored parameters: every recorded accuracy change is exactly 0.0, for every seed and sequence. This is a **code-correctness/regression property confirmed to hold**, not a statistical retention experiment -- see the note below.

## Final paper

The final-paper PR is documentation and analysis only. It does not introduce a new learning algorithm. It organizes the existing evidence into:

- abstract and research questions;
- experimental setup and data separation;
- acquisition and retention results;
- statistical interpretation;
- limitations and threats to validity;
- reproducibility checklist;
- publication-ready tables.

See:

- [`docs/final_paper.md`](docs/final_paper.md)
- [`docs/tables.md`](docs/tables.md)
- [`results/report.md`](results/report.md)

## Continuous integration

Every push and pull request re-runs the experiment suite and regression tests through `.github/workflows/run-experiment.yml`. The workflow prints the report and statistics tables to the GitHub Actions Summary and uploads generated CSV/plot artifacts.

## Skill-isolation invariant check (not a statistical retention experiment)

`experiments/retention.py` runs sequential skill acquisition and re-evaluates every previously acquired skill after each later acquisition, on a stable, deterministic evaluation set per skill.

**Read this before citing this experiment's numbers:** in the current architecture, a `Skill`'s stored network is never modified once training on it stops, and each check re-evaluates the identical frozen network on the identical evaluation batch before and after. That means the recorded accuracy change is exactly 0.0 by mathematical construction, not by empirical result -- there is no code path here through which it could come out otherwise. A bootstrap confidence interval or effect size over that quantity would misrepresent zero sampling variance as a statistical finding, so this file does not compute either. What it does verify, legitimately, is an **implementation invariant**: that no bug lets a later acquisition's gradient updates leak into a supposedly frozen skill. That's a real and useful regression check on the code, not evidence that catastrophic forgetting is absent in continual-learning systems generally. A genuine empirical retention result would need a baseline where interference is actually possible (e.g. extending `results/report.md`'s original shared-network comparison to this experiment's task sequences) -- noted as future work, not implemented here.

Outputs:

- `results/retention.csv` — per-seed diagnostics.
- `results/retention_summary.csv` — aggregate summary, including `max_absolute_retention_delta` and `all_deltas_exactly_zero` (both of which should be `0.0`/`True` for every row -- a violation would indicate a real bug).
- `results/plot_retention.png` — pre/post accuracy per check (points overlap exactly, by construction).

The check uses 15 seeds per representative sequence and a five-percentage-point tolerance retained as a sanity bound, not a statistical margin.

Run it with:

```bash
python experiments/retention.py
python -m unittest discover -s tests -v
```

## Signed-domain transfer-robustness experiment

`experiments/signed_domain_transfer.py` reruns the four load-bearing relatedness pairs and the fixed-target prerequisite-history matrix under two domains -- the existing non-negative domain (`{0,...,9}`-scale ranges, `tasks.py`'s default, unchanged and byte-identical to every prior result) and a new signed domain (`{-9,...,9}`-scale ranges; powers' exponent stays non-negative to avoid confounding sign with fractional targets; division's divisor is nonzero by construction in both domains). Every comparison is paired by seed. Only the operand domain changes -- architecture, optimizer, learning rate, stopping criterion, training budget, seed protocol, and the compatibility/controller logic are identical to the non-negative baseline.

**Headline result: transfer is not uniformly robust to this domain expansion.** multiplication → powers reverses direction (2.17× → 0.72×); multiplication → squares erodes toward no effect (1.23× → 1.01×); addition → subtraction's negative transfer neutralizes (0.41× → 1.00×); the null control (addition → multiplication) stays stable (1.15× → 1.18×, not significant), which is itself evidence the other three changes are real rather than an artifact of the domain change alone. Squares' acquisition success rate collapses from an already-low 13-20% to exactly 0%. See `docs/final_paper.md` Section 4.4 for the full analysis, including sign-specific diagnostics that offer a plausible (not proven) mechanism for each effect.

This finding should be read narrowly: it establishes that transfer *can* be domain-sensitive for this system on this specific expansion, not that it always is, nor that "negative numbers" are the cause independent of the accompanying distribution shift (expanding `{0,...,9}` to `{-9,...,9}` changes more than sign -- see the limitations note in `docs/final_paper.md`).

Outputs:

- `results/signed_domain_pairs.csv`, `results/signed_domain_pairs_summary.csv`, `results/signed_domain_pairs_comparison.csv` — per-seed and summary results for the four relatedness pairs, both domains.
- `results/signed_domain_history.csv`, `results/signed_domain_history_summary.csv` — the prerequisite-history matrix, both domains.
- `results/signed_domain_sign_breakdown.csv` — sign-specific diagnostic breakdown for multiplication→powers, multiplication→squares, and addition→subtraction.
- `results/plot_signed_domain_speedup.png`, `results/plot_signed_domain_success_rate.png`, `results/plot_signed_domain_compatibility_vs_speedup.png`, `results/plot_signed_domain_convergence.png`.

Run it with:

```bash
python experiments/signed_domain_transfer.py
python -m unittest discover -s tests -v
```

## Reproducibility and scope

The project remains intentionally small and dependency-light. Results are conditional on the tested task family, architecture, optimizer, controller, thresholds, seed count, and sequence length. They should be interpreted as evidence for the proposed mechanism in this controlled setting rather than as a universal claim about continual learning.
