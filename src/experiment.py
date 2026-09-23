"""

Pipeline for a single (dataset, lambda, seed):
  1. Split into train / calibration / test (stratified).
  2. Train Random Forest on train.
  3. Calibrate split CP on calibration; build prediction sets on the ORIGINAL
     test set -> per-instance set sizes (the uncertainty signal).
  4. Build a fixed SHAP TreeExplainer (training-data background).
  5. Compute SHAP on the original test set -> pre-shift attributions.
  6. Resample the test set under covariate shift (intensity lambda).
     We map the shifted resample back to UNIQUE original test instances and
     compute SHAP on the shifted feature rows. Because we tilt the marginal of
     one feature but keep the model and background fixed, attribution changes
     reflect the input shift.
  7. For each ORIGINAL test instance that appears in the shifted sample, pair its
     pre-shift and post-shift attributions and compute rank-stability metrics.
  8. Compare stability between small-set (confident) and large-set (uncertain)
     instances with Mann-Whitney U + Cliff's delta.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from analysis import summarise_comparisons
from conformal import (
    average_set_size,
    empirical_coverage,
    run_split_conformal,
)
from data_loading import Dataset
from shap_stability import (
    make_tree_explainer,
    per_instance_stability,
    shap_values_positive_class,
)
from shift import resample_under_shift, verify_shift


@dataclass
class RunOutput:
    dataset: str
    lam: float
    seed: int
    coverage_original: float
    avg_set_size_original: float
    ess_fraction: float
    shift_mean_before: float
    shift_mean_after: float
    shift_sd_units: float
    comparisons: pd.DataFrame  # tidy table from summarise_comparisons
    per_instance: pd.DataFrame = field(default=None, repr=False)


def split_three_way(
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int,
    train_frac: float = 0.5,
    cal_frac: float = 0.25,
):
    """Stratified train / calibration / test split.

    Defaults: 50% train, 25% calibration, 25% test. The calibration set is
    required by split conformal prediction and must be disjoint from training.
    """
    # First carve off train.
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, train_size=train_frac, stratify=y, random_state=seed
    )
    # Of the remainder, split into calibration and test.
    rel_cal = cal_frac / (1.0 - train_frac)
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_rest, y_rest, train_size=rel_cal, stratify=y_rest, random_state=seed
    )
    return (
        X_train.reset_index(drop=True),
        y_train,
        X_cal.reset_index(drop=True),
        y_cal,
        X_test.reset_index(drop=True),
        y_test,
    )


def run_once(
    ds: Dataset,
    lam: float,
    seed: int,
    k: int = 5,
    n_estimators: int = 300,
    alpha: float = 0.1,
    keep_per_instance: bool = False,
) -> RunOutput:
    """Execute the full pipeline once for given intensity and seed."""
    rng = np.random.default_rng(seed)

    # 1. Split
    X_train, y_train, X_cal, y_cal, X_test, y_test = split_three_way(
        ds.X, ds.y, seed=seed
    )

    # 2. Train Random Forest
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # 3. Conformal prediction on the ORIGINAL test set
    cp = run_split_conformal(
        model, X_cal.to_numpy(), y_cal, X_test.to_numpy(), alpha=alpha
    )
    cov = empirical_coverage(cp, y_test)
    avg_size = average_set_size(cp)

    # 4. Fixed-background SHAP explainer
    # Use a sample of training data as background for speed and stability.
    bg_n = min(200, len(X_train))
    background = X_train.sample(bg_n, random_state=seed)
    explainer = make_tree_explainer(model, background=background)

    # 5. Pre-shift SHAP on original test set
    pre_shap = shap_values_positive_class(explainer, X_test)

    # 6. Build a covariate-shifted background distribution.
    # METHODOLOGY (shifted background):
    #   The trained model is held FIXED throughout. Covariate shift is modelled
    #   as a change in the operating distribution against which explanations are
    #   computed. We resample the test set under exponential tilting to obtain a
    #   shifted population, and use a sample of THAT population as the SHAP
    #   background. The same test instances are then explained twice:
    #     pre-shift  -> against the ORIGINAL training-data background
    #     post-shift -> against the SHIFTED background
    #   Any change in a feature's attribution reflects the sensitivity of the
    #   explanation to the distributional context, with the model itself fixed.
    #   This isolates the shift's effect on explanations and keeps the CP set
    #   sizes (computed on the fixed model) valid. Mechanism grounded in the
    #   documented SHAP background-distribution sensitivity (Yuan et al. 2022).
    shift_res = resample_under_shift(X_test, ds.shift_feature, lam=lam, rng=rng)
    shift_diag = verify_shift(X_test, ds.shift_feature, shift_res.indices)

    shifted_pop = X_test.iloc[shift_res.indices]
    shifted_bg_n = min(bg_n, len(shifted_pop))
    shifted_background = shifted_pop.sample(shifted_bg_n, random_state=seed)
    explainer_shifted = make_tree_explainer(model, background=shifted_background)

    # 7. Explain the SAME test instances against the shifted background.
    # We evaluate on the full original test set so every instance has a paired
    # pre/post attribution and a CP set size.
    post_shap = shap_values_positive_class(explainer_shifted, X_test)

    pre_shap_aligned = pre_shap
    post_shap_aligned = post_shap
    set_sizes_aligned = cp.set_sizes

    # 8. Per-instance stability + group comparison
    stability = per_instance_stability(pre_shap_aligned, post_shap_aligned, k=k)
    comparisons = summarise_comparisons(stability, set_sizes_aligned)

    per_inst = None
    if keep_per_instance:
        per_inst = stability.copy()
        per_inst["set_size"] = set_sizes_aligned

    return RunOutput(
        dataset=ds.name,
        lam=lam,
        seed=seed,
        coverage_original=cov,
        avg_set_size_original=avg_size,
        ess_fraction=shift_res.ess_fraction,
        shift_mean_before=shift_diag["mean_before"],
        shift_mean_after=shift_diag["mean_after"],
        shift_sd_units=shift_diag["shift_in_sd_units"],
        comparisons=comparisons,
        per_instance=per_inst,
    )
