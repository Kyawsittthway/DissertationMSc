import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# adjust path to  actual results CSV
df = pd.read_csv("results/comparisons_long.csv")

# CHECK 1: per-lambda confidence intervals on Cliff's delta (across seeds)
print("=" * 70)
print("CHECK 1 — per-lambda 95% CI on Cliff's delta (does it exclude zero?)")
print("=" * 70)
for ds in ["telco", "adult"]:
    for metric in ["spearman", "kendall", "jaccard"]:
        print(f"\n--- {ds} / {metric} ---")
        sub = df[(df.dataset == ds) & (df.metric == metric)]
        for lam in sorted(sub["lambda"].unique()):
            d = sub[sub["lambda"] == lam]["cliffs_delta"].values
            mean = d.mean()
            se = d.std(ddof=1) / np.sqrt(len(d))
            lo, hi = stats.t.interval(0.95, len(d) - 1, loc=mean, scale=se)
            flag = "excludes 0" if (hi < 0 or lo > 0) else "straddles 0"
            print(
                f"  lambda={lam}: delta={mean:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  {flag}"
            )

# CHECK 2: monotonicity (Spearman corr of lambda vs mean delta) ----
print("\n" + "=" * 70)
print("CHECK 2 — monotonicity: Spearman rho of lambda vs mean delta")
print("=" * 70)
for ds in ["telco", "adult"]:
    for metric in ["spearman", "kendall", "jaccard"]:
        sub = df[(df.dataset == ds) & (df.metric == metric)]
        lams = sorted(sub["lambda"].unique())
        means = [sub[sub["lambda"] == l]["cliffs_delta"].mean() for l in lams]
        rho, p = stats.spearmanr(lams, means)
        print(f"  {ds:6s} {metric:9s}: rho={rho:+.3f}, p={p:.4f}")

# CHECK 3: dataset x shift interaction
print("\n" + "=" * 70)
print("CHECK 3 — dataset x lambda interaction (spearman metric)")
print("=" * 70)
d2 = df[df.metric == "spearman"].copy()
d2["lam"] = d2["lambda"]
model = smf.ols("cliffs_delta ~ C(dataset) * lam", data=d2).fit()
for name, pval in model.pvalues.items():
    if ":" in name:
        print(f"  interaction term {name}: p = {pval:.2e}")
