import pandas as pd

# adjust path to  run_summary.csv
df = pd.read_csv("results/run_summary.csv")

print("Effective sample size fraction by dataset and shift intensity")
print("(closer to 1.0 = healthy; below ~0.3 = degenerate/untrustworthy)\n")

# mean ess_fraction across seeds, per dataset per lambda
summary = (
    df.groupby(["dataset", "lambda"])["ess_fraction"].agg(["mean", "min"]).reset_index()
)
for ds in summary["dataset"].unique():
    print(f"--- {ds} ---")
    sub = summary[summary.dataset == ds]
    for _, row in sub.iterrows():
        flag = "OK" if row["mean"] >= 0.3 else "LOW — degenerate"
        print(
            f"  lambda={row['lambda']}: ESS fraction mean={row['mean']:.3f}  min={row['min']:.3f}  {flag}"
        )
    print()
