# Continual Skill Learning via Skill Cloning — Minimal Prototype

Implementation of a controlled prototype for continual skill acquisition. For an incoming task, the system can reuse an existing solved skill, clone a useful skill and adapt it, or learn from scratch. The experiments focus on acquisition reliability, transfer efficiency, and retention of previously acquired skills.

The current research plan is tracked in [Issue #3](https://github.com/kobros-tech/skill-cloning/issues/3). The final paper draft is in [`docs/final_paper.md`](docs/final_paper.md), with compact publication tables in [`docs/tables.md`](docs/tables.md).

## Main findings

- Prior knowledge can produce positive, negligible, or negative transfer depending on the source-target pair; cloning is not universally beneficial.
- Earlier relatedness-pair experiments show approximately 2.31× speedup for multiplication → powers, 1.26× for multiplication → squares, and negative transfer (0.33×) for addition → subtraction.
- The expanded fixed-target matrix confirms heterogeneous transfer: powers benefit from additional prior history (471.6 → 355.2 → 237.3 epochs), while division shows negative transfer (515.2 → 616.2 → 617.5 epochs) and squares remains difficult (20.0% → 20.0% → 13.3% success).
- A corrected reuse gate requires both compatibility evidence and independent target-solve accuracy, preventing a merely related but unsolved skill from being treated as a zero-training solution.
- The retention checks re-evaluate previously acquired skills after later acquisitions on stable skill-specific evaluation sets.
- The reported zero-change retention checks are consistent with the isolated-skill invariant: stored parent skills are not modified during later skill acquisition.
- These retention checks are deliberately treated as an implementation/mechanism verification, not as statistical evidence that the system is robust to catastrophic forgetting. A genuine interference experiment would require an at-risk comparison arm in which later learning can modify previously learned parameters.

## Final paper

The final-paper PR organizes the evidence into:

- abstract and research questions;
- experimental setup and data separation;
- formal mathematical formulation;
- acquisition and transfer results;
- fixed-target prerequisite-history analysis;
- mechanism-level retention verification;
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

## Retention / mechanism check

`experiments/retention.py` runs sequential skill acquisition and re-evaluates every previously acquired skill after each later acquisition. Each skill is evaluated on a stable, independent retention set. The resulting pre/post checks verify the intended isolation behavior: the stored parent skill is not modified when a later skill is trained as an independent copy.

Outputs:

- `results/retention.csv` — per-seed retention and acquisition diagnostics.
- `results/retention_summary.csv` — aggregate diagnostic summaries.
- `results/plot_retention.png` — retention-change figure.

The experiment uses 15 seeds per representative sequence and a predeclared five-percentage-point practical diagnostic tolerance. Because the parent network and evaluation set remain unchanged between the paired checks, a zero change is expected under the isolated-skill design and should not be interpreted as a population-level estimate of resistance to interference.

Run it with:

```bash
python experiments/retention.py
python -m unittest discover -s tests -v
```

## Reproducibility and scope

The project remains intentionally small and dependency-light. Results are conditional on the tested task family, architecture, optimizer, controller, thresholds, seed count, and sequence length. They should be interpreted as evidence for the proposed mechanism in this controlled setting rather than as a universal claim about continual learning.
