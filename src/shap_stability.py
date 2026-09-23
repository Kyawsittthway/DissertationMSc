"""
SHAP feature attributions and per-instance rank-stability metrics.

For each instance we extract its top-k features (by absolute SHAP value) and
then, comparing the pre-shift and post-shift attributions for the SAME instance,
measure how much that top-k ranking moved. Three complementary metrics:

  * Spearman's rho  - overall monotonic agreement between the two rankings.
  * Kendall's tau   - pairwise ordering agreement (robust, interpretable).
  * Jaccard         - overlap of the top-k SETS, ignoring order.

Why rank-based and not magnitude-based? Because users act on the ORDER of
features in an explanation, not the raw SHAP value (Goldwasser & Hooker 2024),
and magnitude changes are partly an artefact of the background distribution
(Yuan et al. 2022). Restricting to top-k focuses on the part of the ranking that
is both meaningful and (per Yuan's U-shape) most stable.

IMPORTANT methodological note on the background distribution:
  SHAP values depend on the background (reference) distribution used by the
  explainer. In this study the shift is applied to that background, not to the
  instances being explained: the same test instances are explained twice, once
  against a training-data background (pre-shift) and once against a background
  drawn from the covariate-shifted population (post-shift). The model and the
  instances are held fixed, so any change in attribution reflects the
  sensitivity of the explanation to the shifted reference distribution alone.
  This isolates the effect of shift on explanations and keeps the conformal
  set sizes (computed on the unshifted data) valid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from scipy.stats import kendalltau, spearmanr


def make_tree_explainer(model, background: pd.DataFrame | None = None):
    """Create a SHAP TreeExplainer for a fitted tree model.

    For Random Forests, TreeExplainer is exact and fast. The background is
    supplied by the caller: the pre-shift explanation uses a training-data
    background and the post-shift explanation a shifted background, so that
    the effect of the shift enters through the reference distribution.
    """
    if background is not None:
        return shap.TreeExplainer(
            model, data=background, feature_perturbation="interventional"
        )
    return shap.TreeExplainer(model)


def shap_values_positive_class(explainer, X: pd.DataFrame) -> np.ndarray:
    """Return a (n_instances, n_features) SHAP matrix for the positive class.

    Different SHAP/sklearn versions return slightly different shapes for binary
    classifiers; this normalises them to the positive-class contributions.
    """
    raw = explainer.shap_values(X, check_additivity=False)

    # Newer SHAP returns an ndarray; older returns a list per class.
    if isinstance(raw, list):
        # Binary -> list of length 2; take the positive class.
        arr = np.asarray(raw[1])
    else:
        arr = np.asarray(raw)
        # Could be (n, features, classes) or (n, features).
        if arr.ndim == 3:
            arr = arr[:, :, 1]  # positive class slice
    return arr


def top_k_indices(shap_row: np.ndarray, k: int) -> np.ndarray:
    """Indices of the top-k features by ABSOLUTE SHAP value, most important first."""
    order = np.argsort(-np.abs(shap_row))  # descending by |shap|
    return order[:k]


def spearman_topk(pre_row: np.ndarray, post_row: np.ndarray, k: int) -> float:
    """Spearman rho between pre/post rankings, restricted to the union of top-k.

    We rank the features that appear in EITHER top-k list (by |SHAP|) and
    correlate their pre vs post ranks. Features absent from a top-k still get a
    rank from the full ordering, so the correlation is well-defined.
    Returns NaN if degenerate (e.g. constant), which the caller filters out.
    """
    pre_top = top_k_indices(pre_row, k)
    post_top = top_k_indices(post_row, k)
    union = np.union1d(pre_top, post_top)

    # Full-ranking position (0 = most important) for each feature, pre and post.
    pre_full = np.argsort(np.argsort(-np.abs(pre_row)))
    post_full = np.argsort(np.argsort(-np.abs(post_row)))

    pre_ranks = pre_full[union]
    post_ranks = post_full[union]
    if len(union) < 2:
        return np.nan
    rho, _ = spearmanr(pre_ranks, post_ranks)
    return float(rho)


def kendall_topk(pre_row: np.ndarray, post_row: np.ndarray, k: int) -> float:
    """Kendall's tau between pre/post rankings over the union of top-k features."""
    pre_top = top_k_indices(pre_row, k)
    post_top = top_k_indices(post_row, k)
    union = np.union1d(pre_top, post_top)

    pre_full = np.argsort(np.argsort(-np.abs(pre_row)))
    post_full = np.argsort(np.argsort(-np.abs(post_row)))

    pre_ranks = pre_full[union]
    post_ranks = post_full[union]
    if len(union) < 2:
        return np.nan
    tau, _ = kendalltau(pre_ranks, post_ranks)
    return float(tau)


def jaccard_topk(pre_row: np.ndarray, post_row: np.ndarray, k: int) -> float:
    """Jaccard similarity of the two top-k SETS (membership, ignores order)."""
    pre_top = set(top_k_indices(pre_row, k).tolist())
    post_top = set(top_k_indices(post_row, k).tolist())
    inter = len(pre_top & post_top)
    union = len(pre_top | post_top)
    return float(inter / union) if union > 0 else np.nan


def per_instance_stability(
    pre_shap: np.ndarray, post_shap: np.ndarray, k: int = 5
) -> pd.DataFrame:
    """Compute all three stability metrics for every instance.

    Parameters
    ----------
    pre_shap, post_shap : (n_instances, n_features) SHAP matrices for the SAME
        instances, before and after shift.
    k : top-k cutoff.

    Returns a DataFrame with columns spearman, kendall, jaccard, one row per
    instance.
    """
    n = pre_shap.shape[0]
    rows = []
    for i in range(n):
        pre_row = pre_shap[i]
        post_row = post_shap[i]
        rows.append(
            {
                "spearman": spearman_topk(pre_row, post_row, k),
                "kendall": kendall_topk(pre_row, post_row, k),
                "jaccard": jaccard_topk(pre_row, post_row, k),
            }
        )
    return pd.DataFrame(rows)
