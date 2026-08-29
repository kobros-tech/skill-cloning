"""
run_all.py — reproduces the full experiment end to end.

Usage:
    pip install -r requirements.txt
    python run_all.py

Writes all CSVs, statistics tables, and plots into results/.
"""
import os
import sys
import pickle

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)
os.chdir(RESULTS)

import experiment as ex
import analysis as an

print("Running experiment (3 strategies x N seeds)...")
out = ex.run_all()

out["final"].to_csv("final.csv", index=False)
out["convergence"].to_csv("convergence.csv", index=False)
out["params"].to_csv("params.csv", index=False)
with open("logs_illustrative.pkl", "wb") as f:
    pickle.dump(out["logs_illustrative"], f)
with open("decisions_illustrative.pkl", "wb") as f:
    pickle.dump(out["decisions_illustrative"], f)
print("Saved final.csv, convergence.csv, params.csv")

print("Computing statistics...")
tables = an.build_report_tables(out["final"], out["convergence"], out["params"])
tables["forgetting_shared_vs_proposed"].to_csv("t_forgetting.csv", index=False)
tables["convergence_scratch_vs_proposed"].to_csv("t_convergence.csv", index=False)
tables["retention_proposed_vs_scratch"].to_csv("t_retention.csv", index=False)
tables["final_mse_summary"].to_csv("final_mse_summary.csv", index=False)
tables["final_acc_summary"].to_csv("final_acc_summary.csv", index=False)
tables["convergence_summary"].to_csv("convergence_summary.csv", index=False)
tables["params_summary"].to_csv("params_summary.csv", index=False)
print("Saved t_forgetting.csv, t_convergence.csv, t_retention.csv, and summary tables")

print("Generating plots...")
import make_plots  # noqa: executes on import, writes plot_*.png into results/
print("Done. See results/report.md for the write-up.")
