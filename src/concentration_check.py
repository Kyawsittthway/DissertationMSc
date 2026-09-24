import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from data_loading import load_telco, load_adult
from experiment import split_three_way
from shap_stability import make_tree_explainer, shap_values_positive_class

for name, ds in [
    ("Telco", load_telco("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")),
    ("Adult", load_adult(None)),
]:
    Xtr, ytr, Xca, yca, Xte, yte = split_three_way(ds.X, ds.y, seed=0)
    m = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1).fit(
        Xtr, ytr
    )
    ex = make_tree_explainer(m, Xtr.sample(200, random_state=0))
    sv = shap_values_positive_class(ex, Xte.iloc[:400])
    imp = pd.Series(np.abs(sv).mean(0), index=Xte.columns).sort_values(ascending=False)
    share = imp / imp.sum()
    print(
        name,
        "top:",
        share.index[0],
        round(share.iloc[0], 3),
        "| shift feature:",
        ds.shift_feature,
        round(share[ds.shift_feature], 3),
        "rank",
        list(share.index).index(ds.shift_feature) + 1,
    )
