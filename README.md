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
- Clone-and-adapt's convergence speedup is relatedness-dependent in the original
  experiment: 5.1× faster when cloning from a closely related skill (powers ←
  multiplication), but slower than random init when the parent is only weakly
  related (subtraction ← addition).
- The fixed-budget follow-up shows that the original powers generalization gap
  largely disappears at equal training budgets, so the original stopping-rule
  comparison was confounded.
- Independent review also identified a more fundamental controller issue:
  `τ_solve=0.90` can classify a merely related but unsolved task as `reuse`.
  PR #2 therefore audits/calibrates this decision rule before broader task-pair
  experiments are added.

## Stopping-rule confound follow-up

The original Phase 4 convergence metric stops each run when training accuracy
reaches 85%. Because a clone may reach that threshold earlier than a scratch
model, comparing held-out MSE at the stopping point can mix two effects:
initialization quality and training duration.

`experiments/stopping_rule_confound.py` provides a controlled follow-up for the
powers ← multiplication experiment. It records the original 85%-accuracy stopping
outcome, then re-runs clone and scratch from the same initial states at identical,
predeclared training budgets. Validation and test MSE are measured at every
budget without using the test set to choose a budget.

Outputs:

- `results/stopping_rule_confound.csv` — per-seed, per-budget raw results.
- `results/stopping_rule_confound_summary.csv` — mean/std/min/max summaries.
- `results/stopping_rule_confound_stopping_epochs.csv` — original stopping-rule
  epochs for clone vs scratch.

Run it with:

```bash
python experiments/stopping_rule_confound.py
```

This follow-up is deliberately diagnostic: it separates training-budget effects
from initialization effects.

## Squares relatedness follow-up

`experiments/squares_relatedness.py` tests cloning from **multiplication → squares**
against a scratch initialization. The squares task maps `(a, b)` to `a²` while
retaining the same two-input model interface.

The experiment measures convergence speed to the existing 85%-training-accuracy
threshold and does not use a test set to select a stopping budget. Across the
original 15 seeds it provides an additional observation consistent with transfer
from a structurally related parent.

Outputs:

- `results/squares_relatedness.csv` — per-seed convergence results.
- `results/squares_relatedness_summary.csv` — aggregate convergence and paired
  scratch/clone speedup statistics.

Run it with:

```bash
python experiments/squares_relatedness.py
```

## Compatibility decision-rule calibration

`experiments/compatibility_calibration.py` audits the original compatibility
controller before more task pairs are added. The original `τ_solve` rule used a
high frozen-skill compatibility score as sufficient evidence for zero-training
`reuse`. This is not necessarily valid: a related task can have a high score
without already being solved.

The calibration experiment trains skills on the existing source tasks, evaluates
every source → target pair, and records:

- frozen compatibility score on the compatibility probe;
- accuracy and MSE on an independent calibration batch;
- whether the target is actually solved under the predeclared 85%-accuracy rule;
- whether the original `τ_solve` would have selected `reuse`;
- false-reuse decisions;
- a calibration-only recommended score threshold.

No held-out test data are used to select the threshold. The controller itself is
also hardened so `reuse` requires both the compatibility threshold and the
independent solved-task accuracy criterion.

Outputs:

- `results/compatibility_calibration.csv` — per-seed, per-source/target diagnostics.
- `results/compatibility_calibration_summary.csv` — original vs calibration-only
  threshold and false-reuse counts.

Run it with:

```bash
python experiments/compatibility_calibration.py
```

Regression tests cover the key failure mode: a high compatibility score with
insufficient target accuracy must not trigger zero-training reuse.

## Repo structure

```
.
├── .github/workflows/run-experiment.yml   # CI: reruns everything, prints results to the run summary
├── run_all.py          # reproduces everything end to end
├── requirements.txt
├── src/
│   ├── skill.py         # TinyMLP skill representation + clone() (Section 5)
│   ├── tasks.py          # addition/subtraction/multiplication/powers/squares task generators
│   ├── compatibility.py  # P(T|s_i) definition + reuse/clone/scratch decision rule
│   ├── strategies.py     # 3 training strategies: shared, independent scratch, clone-and-adapt
│   ├── experiment.py     # runs all three strategies across seeds (Phase 4 driver)
│   ├── analysis.py       # paired statistical tests (Phase 4 evaluation)
│   ├── make_plots.py     # generates results/plot_*.png
│   └── print_summary.py  # formats report + stats tables as markdown for the CI job summary
├── experiments/
│   ├── stopping_rule_confound.py      # fixed-budget confound follow-up
│   ├── squares_relatedness.py         # multiplication → squares relatedness follow-up
│   └── compatibility_calibration.py   # compatibility/reuse decision calibration
├── tests/
│   └── test_compatibility.py          # regression tests for false reuse
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
python experiments/stopping_rule_confound.py
python experiments/squares_relatedness.py
python experiments/compatibility_calibration.py
python -m unittest discover -s tests -v
```

These commands regenerate the experiment CSVs, statistics, and calibration
diagnostics. The project remains dependency-light: numpy/pandas on CPU, no GPU
or deep-learning framework required.

## Status

Research hypothesis / design proposal — exploratory, not a claim of novelty
(see the issue for the full framing). This prototype is meant to test the
mechanism, not to be a finished system.
