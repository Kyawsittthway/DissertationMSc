"""
Split (inductive) conformal prediction for binary classification.

Why hand-rolled rather than a library?
  Split CP is short and transparent, and the whole dissertation hinges on
  understanding exactly what a 'prediction set' is and where 'set size' comes
  from. Implementing it directly means every line is defensible in the viva.
  The implementation is cross-checked against MAPIE (see validate_mapie.py), which produces identical prediction sets on both datasets.A methodology rigour point.

The method (Vovk et al.; see Angelopoulos & Bates 2023 for a gentle treatment):

  1. Split labelled data into a training set and a calibration set.
  2. Train the model on the training set.
  3. On the calibration set, compute a 'nonconformity score' for each point.
     For classification we use   s_i = 1 - p_model(true label | x_i),
     i.e. one minus the predicted probability of the *correct* class. A high
     score means the model was 'surprised' by the true label.
  4. Find the threshold q_hat = the ceil((n+1)(1-alpha))/n empirical quantile
     of the calibration scores. This finite-sample correction is what gives the
     coverage guarantee.
  5. For a test point, include label k in its prediction set iff
        1 - p_model(k | x_test)  <=  q_hat
     equivalently  p_model(k | x_test) >= 1 - q_hat.

The prediction set can be {0}, {1}, or {0,1}. Its SIZE is the per-instance
uncertainty signal that this dissertation studies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConformalResult:
    """Holds the per-instance conformal output for a test set.

    Attributes
    ----------
    sets : np.ndarray of shape (n_test, n_classes), dtype bool
        sets[i, k] is True iff label k is in the prediction set for instance i.
    set_sizes : np.ndarray of shape (n_test,), dtype int
        Number of labels in each prediction set (1 or 2 for binary).
    q_hat : float
        The calibrated nonconformity threshold.
    alpha : float
        The miscoverage level used (target coverage = 1 - alpha).
    """

    sets: np.ndarray
    set_sizes: np.ndarray
    q_hat: float
    alpha: float


def _nonconformity_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Score = 1 - predicted probability of the true label.

    Parameters
    ----------
    probs : (n, n_classes) predicted class probabilities.
    labels : (n,) true labels as integer class indices.
    """
    n = probs.shape[0]
    true_class_prob = probs[np.arange(n), labels]
    return 1.0 - true_class_prob


def calibrate_threshold(
    cal_probs: np.ndarray, cal_labels: np.ndarray, alpha: float = 0.1
) -> float:
    """Compute the conformal threshold q_hat from calibration data.

    Uses the finite-sample-corrected quantile:
        level = ceil((n + 1) * (1 - alpha)) / n
    This correction (rather than the plain (1-alpha) quantile) is what makes the
    marginal coverage guarantee hold in finite samples.
    """
    scores = _nonconformity_scores(cal_probs, cal_labels)
    n = len(scores)
    # Rank of the quantile we need, with the +1 finite-sample correction.
    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    # np.quantile with 'higher' matches the conformal convention of taking the
    # ceil-ranked score. Clip level to [0,1] for tiny calibration sets.
    level = min(max(level, 0.0), 1.0)
    q_hat = np.quantile(scores, level, method="higher")
    return float(q_hat)


def build_prediction_sets(test_probs: np.ndarray, q_hat: float) -> ConformalResult:
    """Build prediction sets for test instances given a calibrated threshold.

    Label k is included iff  1 - p(k | x)  <= q_hat.
    """
    # scores[i, k] = 1 - p(k | x_i)
    scores = 1.0 - test_probs
    sets = scores <= q_hat  # boolean membership matrix
    set_sizes = sets.sum(axis=1).astype(int)
    # Guard: a degenerate threshold could in principle yield empty sets.
    # Conformal sets should never be empty; if they are, include the argmax.
    empty = set_sizes == 0
    if empty.any():
        best = test_probs[empty].argmax(axis=1)
        sets[np.where(empty)[0], best] = True
        set_sizes = sets.sum(axis=1).astype(int)
    return ConformalResult(sets=sets, set_sizes=set_sizes, q_hat=q_hat, alpha=None)


def run_split_conformal(
    model,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    alpha: float = 0.1,
) -> ConformalResult:
    """Full split-CP pass: calibrate on (X_cal, y_cal), predict sets on X_test.

    `model` must already be trained and expose predict_proba.
    """
    cal_probs = model.predict_proba(X_cal)
    test_probs = model.predict_proba(X_test)
    q_hat = calibrate_threshold(cal_probs, y_cal, alpha=alpha)
    result = build_prediction_sets(test_probs, q_hat)
    result.alpha = alpha
    return result


def empirical_coverage(result: ConformalResult, y_test: np.ndarray) -> float:
    """Fraction of test instances whose true label is in their prediction set.

    Should be >= 1 - alpha if the exchangeability assumption holds. Under
    covariate shift it may drop below the target; reporting that drop is part of
    the analysis.
    """
    n = len(y_test)
    in_set = result.sets[np.arange(n), y_test]
    return float(in_set.mean())


def average_set_size(result: ConformalResult) -> float:
    """Mean prediction-set size across the test set (efficiency metric)."""
    return float(result.set_sizes.mean())
