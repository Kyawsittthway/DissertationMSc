"""
Validates the hand-rolled split conformal prediction against MAPIE,
on the real Telco and Adult datasets. Prints numbers for the appendix slide.
"""

import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from mapie.classification import SplitConformalClassifier

from data_loading import load_telco, load_adult
from experiment import split_three_way
from conformal import run_split_conformal

# ---- adjust these paths to real CSV locations ----
TELCO_CSV = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
ADULT_CSV = "data/adult.csv"  # or None to fetch via OpenML


def check(name, X, y, alpha=0.1, seed=0):
    Xtr, ytr, Xca, yca, Xte, yte = split_three_way(X, y, seed=seed)
    m = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    m.fit(Xtr, ytr)

    #  hand-rolled CP
    ours = run_split_conformal(m, Xca.to_numpy(), yca, Xte.to_numpy(), alpha=alpha)
    ours_cov = ours.sets[np.arange(len(yte)), yte].mean()

    # MAPIE 1.x split conformal, LAC score (= 1 - p(true), matches )
    mapie = SplitConformalClassifier(
        estimator=m,
        confidence_level=1 - alpha,
        conformity_score="lac",
        prefit=True,
    )
    mapie.conformalize(Xca.to_numpy(), yca)
    _, y_ps = mapie.predict_set(Xte.to_numpy())
    mapie_sets = y_ps[:, :, 0]
    mapie_sizes = mapie_sets.sum(axis=1)
    mapie_cov = mapie_sets[np.arange(len(yte)), yte].mean()

    agree = (ours.sets == mapie_sets).all(axis=1).mean()

    print(f"\n=== {name} (alpha={alpha}) ===")
    print(
        f"  ours  avg set size : {ours.set_sizes.mean():.4f}   coverage : {ours_cov:.4f}"
    )
    print(
        f"  mapie avg set size : {mapie_sizes.mean():.4f}   coverage : {mapie_cov:.4f}"
    )
    print(f"  per-instance sets identical : {agree*100:.1f}%")


# Telco
if os.path.exists(TELCO_CSV):
    ds = load_telco(TELCO_CSV)
    check("Telco", ds.X, ds.y)
else:
    print(f"Telco CSV not found at {TELCO_CSV}")

# Adult
ds = load_adult(ADULT_CSV if os.path.exists(ADULT_CSV) else None)
check("Adult", ds.X, ds.y)
