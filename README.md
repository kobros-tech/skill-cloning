# Continual Skill Learning via Skill Cloning — Minimal Prototype

Implementation of a controlled prototype for continual skill acquisition. For an incoming task, the system can reuse an existing solved skill, clone a useful skill and adapt it, or learn from scratch. The experiments focus on acquisition reliability, transfer efficiency, and retention of previously acquired skills.

The current research plan is tracked in [Issue #3](https://github.com/kobros-tech/skill-cloning/issues/3). The final paper draft is in [`docs/final_paper.md`](docs/final_paper.md), with compact publication tables in [`docs/tables.md`](docs/tables.md).

## Main findings

- Prior knowledge can produce positive, negligible, or negative transfer depending on the source-target pair; cloning is not universally beneficial.
- Earlier relatedness-pair experiments show approximately 2.31× speedup for multiplication → powers, 1.26× for multiplication → squares, and negative transfer (0.33×) for addition → subtraction.
- A corrected reuse gate requires both compatibility evidence and independent target-solve accuracy, preventing a merely related but unsolved skill from being treated as a zero-training solution.
- The retention experiment evaluates previously acquired skills after later acquisitions on stable skill-specific evaluation sets.
- Under the tested isolated-skill mechanism, the reported retention checks show 0.0000 mean accuracy change and 100% retention-pass rate at the predeclared five-percentage-point practical tolerance.
- The retention result is deliberately stated narrowly: it shows no measurable forgetting under the tested mechanism and protocol, not universal absence of catastrophic forgetting.

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

## Retention / catastrophic-forgetting experiment

`experiments/retention.py` runs sequential skill acquisition and re-evaluates every previously acquired skill after each later acquisition. Each skill is evaluated on a stable, independent retention set, making pre/post comparisons paired.

Outputs:

- `results/retention.csv` — per-seed retention and acquisition diagnostics.
- `results/retention_summary.csv` — aggregate retention statistics.
- `results/plot_retention.png` — retention-change figure.

The experiment uses 15 seeds per representative sequence and a predeclared five-percentage-point practical retention tolerance. Retention is evaluated separately from acquisition efficiency so that preserving old skills is not inferred from training speed.

Run it with:

```bash
python experiments/retention.py
python -m unittest discover -s tests -v
```

## Reproducibility and scope

The project remains intentionally small and dependency-light. Results are conditional on the tested task family, architecture, optimizer, controller, thresholds, seed count, and sequence length. They should be interpreted as evidence for the proposed mechanism in this controlled setting rather than as a universal claim about continual learning.
