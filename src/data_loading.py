"""
Dataset-agnostic loading for the dissertation experiments.

Every loader returns a `Dataset` object with a standard interface so the rest of
the pipeline never needs to know which dataset it is working with. To add a new
dataset, write one new loader function that returns a `Dataset`.

The key field for this dissertation is `shift_feature`: the continuous feature
whose distribution we will perturb to simulate covariate shift. For Telco this is
`tenure`; for Adult it is `age`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    """A standard container so the pipeline is dataset-agnostic.

    Attributes
    ----------
    name : str
        Human-readable dataset name, used in output filenames and figures.
    X : pd.DataFrame
        Feature matrix (already numeric-encoded, ready for the model).
    y : np.ndarray
        Binary target, encoded as 0 / 1.
    shift_feature : str
        Name of the continuous feature we perturb to induce covariate shift.
        Must be a column in X.
    feature_names : list[str]
        Column names of X, in order. Convenience for SHAP ranking.
    positive_label : str
        Original name of the positive class (for reporting only).
    """

    name: str
    X: pd.DataFrame
    y: np.ndarray
    shift_feature: str
    feature_names: list[str] = field(default_factory=list)
    positive_label: str = "1"

    def __post_init__(self) -> None:
        if not self.feature_names:
            self.feature_names = list(self.X.columns)
        if self.shift_feature not in self.X.columns:
            raise ValueError(
                f"shift_feature '{self.shift_feature}' not found in X columns."
            )
        # Basic sanity: y must be binary 0/1
        unique = set(np.unique(self.y).tolist())
        if not unique.issubset({0, 1}):
            raise ValueError(f"y must be binary 0/1, got values {unique}")


# Telco Customer Churn loader
def load_telco(csv_path: str | Path) -> Dataset:
    """Load IBM Telco Customer Churn from the standard Kaggle/IBM CSV.

    Downloaded from:
      https://www.kaggle.com/datasets/blastchar/telco-customer-churn
    The file is usually named
      'WA_Fn-UseC_-Telco-Customer-Churn.csv'.

    Preprocessing performed here:
      * Drop the customerID column (identifier, not predictive).
      * Coerce TotalCharges to numeric (it has blank strings for new customers).
      * Drop the small number of rows with missing TotalCharges (~11 rows).
      * One-hot encode all categorical columns.
      * Encode the Churn target as 1 = 'Yes', 0 = 'No'.
    The continuous shift feature is `tenure`.
    """
    df = pd.read_csv(csv_path)

    # Drop identifier
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # TotalCharges arrives as object because new customers have blank strings.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)

    # Target
    y = (df["Churn"].str.strip().str.lower() == "yes").astype(int).to_numpy()
    df = df.drop(columns=["Churn"])

    # SeniorCitizen is already 0/1 numeric; everything else object -> one-hot.
    categorical = df.select_dtypes(include=["object"]).columns.tolist()
    X = pd.get_dummies(df, columns=categorical, drop_first=True)

    # Ensure all columns numeric and float (SHAP / sklearn happiest with float)
    X = X.astype(float)

    return Dataset(
        name="telco",
        X=X,
        y=y,
        shift_feature="tenure",
        positive_label="Churn=Yes",
    )


# Adult Census Income loader
def load_adult(csv_path: str | Path | None = None) -> Dataset:
    """Load UCI Adult Census Income.

    Two ways to use this:
      1. Pass a CSV path downloaded from UCI / Kaggle.
      2. Pass None to fetch via scikit-learn's OpenML interface (needs internet).

    Downloaded from:
      https://archive.ics.uci.edu/dataset/2/adult

    Preprocessing performed here:
      * Obtain the dataset via OpenML (combined train+test, 48,842 rows).
      * Drop the fnlwgt survey sampling weight (not predictive).
      * One-hot encode all categorical columns. NOTE: on the OpenML path,
        missing entries (in workclass / occupation / native-country) are
        retained and encoded as an explicit "missing" category rather than
        dropped; this is the behaviour used for the reported results.
      * Encode the income target as 1 = '>50K', 0 = '<=50K'.
    The continuous shift feature is `age`.
    """
    if csv_path is None:
        from sklearn.datasets import fetch_openml

        bunch = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
        df = bunch.frame.copy()
        # In OpenML v2, target column is named 'class'
        target_col = "class"
    else:
        # UCI raw CSV has no header; define column names per the data dictionary.
        col_names = [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country",
            "income",
        ]
        df = pd.read_csv(csv_path, header=None, names=col_names, skipinitialspace=True)
        target_col = "income"

    # Strip whitespace on any string columns
    for c in df.select_dtypes(include=["object", "category"]).columns:
        df[c] = df[c].astype(str).str.strip()

    # '?' marks missing in Adult; treat as NaN and drop
    df = df.replace("?", np.nan).dropna().reset_index(drop=True)

    # Target: '>50K' (sometimes '>50K.') is positive
    raw_target = df[target_col].astype(str).str.replace(".", "", regex=False)
    y = (raw_target.str.strip() == ">50K").astype(int).to_numpy()
    df = df.drop(columns=[target_col])

    # fnlwgt is a survey sampling weight, not predictive of income - drop it.
    if "fnlwgt" in df.columns:
        df = df.drop(columns=["fnlwgt"])

    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    X = pd.get_dummies(df, columns=categorical, drop_first=True)
    X = X.astype(float)

    return Dataset(
        name="adult",
        X=X,
        y=y,
        shift_feature="age",
        positive_label="income>50K",
    )


# Synthetic fallback - mimics the real schemas so the pipeline can be tested
# without internet or the real CSVs present.
def load_synthetic(which: str = "telco", n: int = 5000, seed: int = 0) -> Dataset:
    """Generate synthetic data that mimics Telco or Adult structure.

    This exists ONLY so the pipeline can be smoke-tested end-to-end without the
    real data. Real experiments will use load_telco / load_adult. The synthetic
    generator deliberately builds in a relationship between the shift feature and
    the target so that shifting it actually moves predictions and SHAP values.
    """
    rng = np.random.default_rng(seed)

    if which == "telco":
        shift_feat = "tenure"
        tenure = rng.integers(0, 73, size=n).astype(float)
        monthly = rng.normal(65, 30, size=n).clip(18, 120)
        total = tenure * monthly + rng.normal(0, 50, size=n)
        contract = rng.integers(0, 3, size=n).astype(float)  # 0,1,2
        internet = rng.integers(0, 3, size=n).astype(float)
        payment = rng.integers(0, 4, size=n).astype(float)
        senior = rng.integers(0, 2, size=n).astype(float)
        # Churn driven mostly by low tenure + high monthly + short contract
        logit = (
            -0.06 * tenure
            + 0.015 * monthly
            - 0.5 * contract
            + 0.2 * internet
            + 0.1 * payment
            + 0.3 * senior
            + 1.0
        )
        X = pd.DataFrame(
            {
                "tenure": tenure,
                "monthly_charges": monthly,
                "total_charges": total,
                "contract_type": contract,
                "internet_service": internet,
                "payment_method": payment,
                "senior_citizen": senior,
            }
        )
    elif which == "adult":
        shift_feat = "age"
        age = rng.integers(17, 90, size=n).astype(float)
        hours = rng.normal(40, 12, size=n).clip(1, 99)
        edu_num = rng.integers(1, 17, size=n).astype(float)
        cap_gain = (rng.random(size=n) < 0.1) * rng.exponential(5000, size=n)
        sex = rng.integers(0, 2, size=n).astype(float)
        workclass = rng.integers(0, 6, size=n).astype(float)
        # Income driven by age + hours + education
        logit = (
            0.04 * age
            + 0.03 * hours
            + 0.18 * edu_num
            + 0.0002 * cap_gain
            + 0.3 * sex
            - 6.0
        )
        X = pd.DataFrame(
            {
                "age": age,
                "hours_per_week": hours,
                "education_num": edu_num,
                "capital_gain": cap_gain,
                "sex": sex,
                "workclass": workclass,
            }
        )
    else:
        raise ValueError("which must be 'telco' or 'adult'")

    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(size=n) < prob).astype(int)

    return Dataset(
        name=f"synthetic_{which}",
        X=X.astype(float),
        y=y,
        shift_feature=shift_feat,
        positive_label="synthetic_positive",
    )
