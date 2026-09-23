import sys


from data_loading import load_bank
from experiment import split_three_way
from sklearn.ensemble import RandomForestClassifier
from conformal import run_split_conformal
import numpy as np

ds = load_bank("data/bank-full.csv")  #  local copy
X_train, y_train, X_cal, y_cal, X_test, y_test = split_three_way(ds.X, ds.y, seed=0)

model = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]
print(f"Fraction with prob in [0.3, 0.7]: {((probs > 0.3) & (probs < 0.7)).mean():.4f}")

for test_alpha in [0.1, 0.05, 0.02, 0.01]:
    cp = run_split_conformal(
        model, X_cal.to_numpy(), y_cal, X_test.to_numpy(), alpha=test_alpha
    )
    n_uncertain = (cp.set_sizes == 2).sum()
    print(
        f"alpha={test_alpha}: uncertain={n_uncertain} ({100*n_uncertain/len(cp.set_sizes):.2f}%)"
    )
