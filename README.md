# Continual Skill Learning via Skill Cloning — Minimal Prototype

Implementation of the design proposed in [kobros-tech/6.86x#23](https://github.com/kobros-tech/6.86x/issues/23):
protect learned "skills" (small parametric units) from catastrophic forgetting by
reusing, cloning, or freshly initializing a skill depending on a compatibility
score against the incoming task, rather than always updating one shared set of
parameters.

This repo covers Phases 1–4 of the issue's implementation plan (mathematical
specification → minimal implementation → baselines → evaluation), run on the
Section 8 curriculum: **addition → subtraction → multiplication → powers**.

See **[results/report.md](results/report.md)** for the full write-up: methodology,
design choices made where the issue left things open, plots, and statistics.

## Continuous integration

Every push and pull request re-runs the full experiment from scratch
(`.github/workflows/run-experiment.yml`) and prints the report and every
statistics table directly onto the workflow run's **Summary** page, so results
are visible in GitHub without downloading anything. Plots and CSVs are also
uploaded as a downloadable `experiment-results` artifact on each run.

## Quick summary of findings

- Catastrophic forgetting is real and large in the shared-network baseline
  (MSE on old tasks grows 3–4 orders of magnitude); clone-and-adapt eliminates it
  (p < 10⁻⁹ on 3 of 4 tasks, paired by seed).
- Clone-and-adapt's convergence speedup is relatedness-dependent: 5.1× faster when
  cloning from a closely related skill (powers ← multiplication), but *slower*
  than random init when the parent is only weakly related (subtraction ← addition).
- A genuine limitation surfaced: on powers, the cloned skill converged faster but
  generalized *worse* than a from-scratch skill — likely a confound from the
  "stop at 85% training accuracy" rule, not from cloning itself. Flagged as the
  next thing to fix.

## Repo structure

```
.
├── .github/workflows/run-experiment.yml   # CI: reruns everything, prints results to the run summary
├── run_all.py          # reproduces everything end to end
├── requirements.txt
├── src/
│   ├── skill.py         # TinyMLP skill representation + clone() (Section 5)
│   ├── tasks.py          # addition/subtraction/multiplication/powers task generators
│   ├── compatibility.py  # P(T|s_i) definition + reuse/clone/scratch decision rule
│   ├── strategies.py     # 3 training strategies: shared, independent scratch, clone-and-adapt
│   ├── experiment.py     # runs all strategies across seeds (Phase 4 driver)
│   ├── analysis.py       # paired statistical tests (Phase 4 evaluation)
│   ├── make_plots.py     # generates results/plot_*.png
│   └── print_summary.py  # formats report + stats tables as markdown for the CI job summary
└── results/
    ├── report.md          # full write-up
    ├── plot_*.png          # figures
    ├── final.csv, convergence.csv, params.csv   # raw per-seed results
    └── t_*.csv, *_summary.csv                    # statistical test + summary tables
```

## Reproducing

```bash
pip install -r requirements.txt
python run_all.py
```

This regenerates every CSV, statistics table, and plot in `results/` (~15
random seeds; runs in well under a minute on CPU, no GPU/deep-learning
framework required — the whole thing is numpy).

## Status

Research hypothesis / design proposal — exploratory, not a claim of novelty
(see the issue for the full framing). This prototype is meant to test the
mechanism, not to be a finished system.
