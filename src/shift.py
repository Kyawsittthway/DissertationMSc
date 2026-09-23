"""
Simulated covariate shift via importance-weighted resampling.

Goal: change the marginal distribution P(X) of ONE continuous feature while
leaving the conditional P(Y | X) untouched. That is the definition of covariate
shift (Tibshirani et al. 2019), and reweighted resampling achieves it cleanly:
we never alter any label or any feature-to-target relationship; we only change
how frequently each existing test instance appears.

Method: exponential tilting on the chosen shift feature.
    weight_i  =  exp(lambda * z_i)
where z_i is the standardised value of the shift feature for instance i, and
lambda controls intensity:
    lambda = 0   -> no shift (uniform weights)
    lambda > 0   -> oversample HIGH values of the feature
    lambda < 0   -> oversample LOW values of the feature
We then resample the test set WITH REPLACEMENT using probabilities proportional
to the weights.

Standardising the feature first makes a given lambda comparable across datasets
(tenure in months vs age in years would otherwise need very different lambdas).

We also report the effective sample size (ESS), so we can flag intensities where
resampling has collapsed onto too few unique instances to be trustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ShiftResult:
    """Indices and diagnostics for one shifted resample of a test set.

    Attributes
    ----------
    indices : np.ndarray
        Row positions (into the original test set) of the resampled instances.
    lam : float
        The lambda (intensity) used.
    effective_sample_size : float
        Kish's ESS of the sampling weights. Low ESS = resample dominated by few
        instances = unreliable; flag if below ~0.3 * n.
    ess_fraction : float
        ESS divided by n, for an at-a-glance validity check.
    """

    indices: np.ndarray
    lam: float
    effective_sample_size: float
    ess_fraction: float


def _standardise(values: np.ndarray) -> np.ndarray:
    mu = values.mean()
    sd = values.std()
    if sd == 0:
        return np.zeros_like(values, dtype=float)
    return (values - mu) / sd


def compute_shift_weights(shift_values: np.ndarray, lam: float) -> np.ndarray:
    """Exponential-tilting weights on a (raw) shift-feature vector.

    The feature is standardised internally so lambda is dataset-comparable.
    Weights are normalised to sum to 1 (i.e. a sampling distribution).
    """
    z = _standardise(np.asarray(shift_values, dtype=float))
    # Subtract max before exp for numerical stability (softmax trick).
    raw = np.exp(lam * z - np.max(lam * z))
    weights = raw / raw.sum()
    return weights


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish's effective sample size: (sum w)^2 / sum(w^2).

    For normalised weights summing to 1 this is 1 / sum(w^2).
    """
    return float(1.0 / np.sum(weights**2))


def resample_under_shift(
    X_test: pd.DataFrame,
    shift_feature: str,
    lam: float,
    rng: np.random.Generator,
    size: int | None = None,
) -> ShiftResult:
    """Draw a covariate-shifted resample of the test set.

    Parameters
    ----------
    X_test : the original test feature frame.
    shift_feature : column name to tilt.
    lam : intensity (0 = no shift).
    rng : a numpy Generator for reproducibility.
    size : resample size; defaults to len(X_test).

    Returns a ShiftResult with the chosen row indices and ESS diagnostics.
    """
    n = len(X_test)
    if size is None:
        size = n
    weights = compute_shift_weights(X_test[shift_feature].to_numpy(), lam)
    ess = effective_sample_size(weights)
    indices = rng.choice(n, size=size, replace=True, p=weights)
    return ShiftResult(
        indices=indices,
        lam=lam,
        effective_sample_size=ess,
        ess_fraction=ess / n,
    )


def verify_shift(X_test: pd.DataFrame, shift_feature: str, indices: np.ndarray) -> dict:
    """Sanity check that the resample actually moved the feature distribution.

    Returns the mean of the shift feature before and after, plus the shift in
    standard-deviation units. Use this in the methodology chapter to evidence
    that a given lambda produced a meaningful, controlled shift.
    """
    original = X_test[shift_feature].to_numpy()
    shifted = X_test[shift_feature].to_numpy()[indices]
    sd = original.std() if original.std() != 0 else 1.0
    return {
        "feature": shift_feature,
        "mean_before": float(original.mean()),
        "mean_after": float(shifted.mean()),
        "shift_in_sd_units": float((shifted.mean() - original.mean()) / sd),
    }
