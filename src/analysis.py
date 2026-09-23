"""
The headline statistical analysis.

Research question, operationalised:
  Do instances with LARGE conformal prediction sets (uncertain) exhibit lower
  SHAP rank stability than instances with SMALL sets (confident)?
[Former research question]

We compare the two groups with the Mann-Whitney U test - a non-parametric test
appropriate for our data because rank-correlation scores are bounded in [-1, 1]
and heavily skewed toward 1, so the normality assumption of a t-test fails.

We always report an effect size alongside the p-value, because with thousands of
instances even a trivial difference can be 'significant'. Cliff's delta is the
natural non-parametric effect size: it is the probability that a randomly chosen
confident-group score exceeds a randomly chosen uncertain-group score, minus the
reverse. It ranges from -1 to 1; conventional thresholds are
|d| < 0.147 negligible, < 0.33 small, < 0.474 medium, else large.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


@dataclass
class ComparisonResult:
    metric: str
    n_confident: int
    n_uncertain: int
    median_confident: float
    median_uncertain: float
    u_statistic: float
    p_value: float
    cliffs_delta: float
    effect_label: str


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta effect size between groups a and b.

    delta = P(a > b) - P(a < b), computed via the Mann-Whitney U statistic to
    avoid an O(n*m) double loop:
        delta = 2U / (n*m) - 1
    where U is the U statistic for group a.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return np.nan
    u, _ = mannwhitneyu(a, b, alternative="two-sided")
    return float(2.0 * u / (n * m) - 1.0)


def _effect_label(d: float) -> str:
    ad = abs(d)
    if np.isnan(d):
        return "undefined"
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def compare_groups(
    stability: pd.DataFrame,
    set_sizes: np.ndarray,
    metric: str = "spearman",
) -> ComparisonResult:
    """Compare a stability metric between small-set and large-set instances.

    Parameters
    ----------
    stability : DataFrame with columns spearman/kendall/jaccard, one row/instance.
    set_sizes : per-instance conformal set sizes (1 = confident, 2 = uncertain).
    metric : which stability column to test.

    Confident group = set size 1; uncertain group = set size >= 2.
    Hypothesis: confident scores are HIGHER (one-sided), but we report the
    two-sided p-value to be conservative.
    """
    values = stability[metric].to_numpy()
    confident_mask = (set_sizes == 1) & ~np.isnan(values)
    uncertain_mask = (set_sizes >= 2) & ~np.isnan(values)

    confident = values[confident_mask]
    uncertain = values[uncertain_mask]

    if len(confident) == 0 or len(uncertain) == 0:
        return ComparisonResult(
            metric=metric,
            n_confident=len(confident),
            n_uncertain=len(uncertain),
            median_confident=float("nan"),
            median_uncertain=float("nan"),
            u_statistic=float("nan"),
            p_value=float("nan"),
            cliffs_delta=float("nan"),
            effect_label="undefined (a group was empty)",
        )

    u, p = mannwhitneyu(confident, uncertain, alternative="two-sided")
    delta = cliffs_delta(confident, uncertain)

    return ComparisonResult(
        metric=metric,
        n_confident=len(confident),
        n_uncertain=len(uncertain),
        median_confident=float(np.median(confident)),
        median_uncertain=float(np.median(uncertain)),
        u_statistic=float(u),
        p_value=float(p),
        cliffs_delta=delta,
        effect_label=_effect_label(delta),
    )


def summarise_comparisons(
    stability: pd.DataFrame, set_sizes: np.ndarray
) -> pd.DataFrame:
    """Run compare_groups for all three metrics and return a tidy table."""
    rows = []
    for metric in ("spearman", "kendall", "jaccard"):
        r = compare_groups(stability, set_sizes, metric)
        rows.append(
            {
                "metric": r.metric,
                "n_confident": r.n_confident,
                "n_uncertain": r.n_uncertain,
                "median_confident": r.median_confident,
                "median_uncertain": r.median_uncertain,
                "U": r.u_statistic,
                "p_value": r.p_value,
                "cliffs_delta": r.cliffs_delta,
                "effect": r.effect_label,
            }
        )
    return pd.DataFrame(rows)
