"""
print_summary.py — formats results/ into GitHub-flavored markdown and prints
to stdout. Intended usage in CI:

    python src/print_summary.py >> "$GITHUB_STEP_SUMMARY"

which makes the report and statistics tables visible on the workflow summary.
"""
import os
import re
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+\.png)\)")


def render_report():
    path = os.path.join(RESULTS, "report.md")
    with open(path) as f:
        text = f.read()

    def swap(m):
        alt, fname = m.group(1), m.group(2)
        return f"*(figure: **{alt}** — see `{fname}` in the workflow artifact `experiment-results`)*"

    return IMG_RE.sub(swap, text)


def render_table(csv_name: str, title: str):
    path = os.path.join(RESULTS, csv_name)
    if not os.path.exists(path):
        return ""
    df = pd.read_csv(path)
    for col in df.select_dtypes(include="float").columns:
        df[col] = df[col].round(4)
    return f"### {title}\n\n{df.to_markdown(index=False)}\n"


def main():
    parts = []
    parts.append("# Continual Skill Learning — Experiment Results\n")
    parts.append("_Run generated automatically by CI._\n")
    parts.append(render_report())

    parts.append("\n---\n\n## Raw statistics tables\n")
    parts.append(render_table("t_forgetting.csv", "Forgetting: shared vs. clone-and-adapt (paired by seed)"))
    parts.append(render_table("t_convergence.csv", "Convergence speed: scratch vs. clone-and-adapt (paired by seed)"))
    parts.append(render_table("t_retention.csv", "Retention: clone-and-adapt vs. independent scratch (paired by seed)"))
    parts.append(render_table("final_mse_summary.csv", "Final MSE summary"))
    parts.append(render_table("final_acc_summary.csv", "Final accuracy summary"))
    parts.append(render_table("convergence_summary.csv", "Convergence steps summary"))
    parts.append(render_table("params_summary.csv", "Parameter growth summary"))
    parts.append(render_table(
        "compatibility_calibration_summary.csv",
        "Compatibility decision-rule calibration (tau_solve audit)",
    ))
    parts.append(render_table(
        "relatedness_pairs_summary.csv",
        "Prerequisite acquisition: fixed target, varying prior-skill history",
    ))

    print("\n".join(p for p in parts if p))


if __name__ == "__main__":
    main()
