# Trustworthy ML Under Distribution Shift — Experiment Pipeline

Conformal Prediction set size as a per-instance proxy for SHAP explanation
stability under covariate shift.

**Finding:** the coupling between conformal set size and SHAP explanation
stability is real and reproducible, but *conditional in sign* — under shift it
inverts on the Telco dataset (confident instances become less stable) but does
not invert on Adult. Conformal set size is therefore a conditional, not
universal, signal of explanation reliability.

## What this does

For each test instance it pairs two signals:
- **Conformal prediction set size** — the per-instance uncertainty signal
  (size 1 = confident, size 2 = uncertain) from split CP.
- **SHAP rank stability** — how much the instance's top-k SHAP ranking moves
  when the explanation's background distribution is shifted.

It then tests whether uncertain instances (large CP sets) have systematically
less stable explanations than confident ones (small CP sets), using
Mann-Whitney U with Cliff's delta as the effect size, across a range of shift
intensities (a dose-response curve) and multiple random seeds.

## Project structure

```
dissertation/
├── run_experiments.py        # top-level orchestration; run this
├── src/
│   ├── data_loading.py       # dataset-agnostic loaders (Telco, Adult, synthetic)
│   ├── conformal.py          # hand-rolled split conformal prediction
│   ├── shift.py              # covariate shift via importance-weighted resampling
│   ├── shap_stability.py     # SHAP + Spearman/Kendall/Jaccard rank stability
│   ├── analysis.py           # Mann-Whitney U + Cliff's delta group comparison
│   ├── experiment.py         # one full pipeline run (orchestrates the above)
│   └── validate_mapie.py     # cross-check of the hand-rolled CP against MAPIE
├── results/                  # CSV outputs land here
└── figures/                  # figures land here
```

## Methodology choices and justification

- **Random Forest** base model — a standard choice for tabular classification,
  and required here because SHAP's TreeExplainer computes *exact* Shapley values
  for tree ensembles, so measured instability reflects genuine change in the
  explanation rather than approximation noise.
- **Hand-rolled split CP** — transparent; the whole project hinges on
  understanding set size, and the implementation is cross-checked against
  MAPIE (see `validate_mapie.py`), producing identical prediction sets on both
  datasets.
- **alpha = 0.1** — 90% target coverage; a field convention that yields a
  meaningful spread of both confident (size-1) and uncertain (size-2) sets.
- **Covariate shift via exponential tilting** on one continuous feature
  (`tenure` for Telco, `age` for Adult), standardised so lambda is comparable
  across datasets. Effective sample size (ESS) is reported so degenerate
  high-intensity resamples can be flagged.
- **Shifted-background SHAP** — the model, its predictions, and the conformal
  sets are held fixed on the unshifted data; the same instances are explained
  against the original vs the shifted background distribution. This isolates the
  effect of distributional drift on explanations and keeps CP set sizes valid.
  Mechanism motivated by the SHAP background-distribution sensitivity result
  (Yuan et al. 2022).
- **Rank-based stability metrics** (Spearman, Kendall, Jaccard) on **top-k = 5** —
  rank matters more than magnitude (Goldwasser & Hooker 2024); top-k focuses on
  the stable, meaningful part of the ranking (Yuan's U-shape).

## Usage

Smoke test (no internet or CSVs needed — uses synthetic data):

```bash
python run_experiments.py --synthetic --synthetic-n 1500 --seeds 2 --quick
```

Reproduce the reported results (Adult fetched via OpenML, as used in the
dissertation):

```bash
python run_experiments.py \
    --telco data/WA_Fn-UseC_-Telco-Customer-Churn.csv \
    --adult "" \
    --seeds 20
```

Alternatively, supply a local Adult CSV instead of fetching from OpenML:

```bash
python run_experiments.py \
    --telco data/WA_Fn-UseC_-Telco-Customer-Churn.csv \
    --adult data/adult.csv \
    --seeds 20
```

## Data sources

- Telco Customer Churn:
  https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Adult Census Income:
  https://archive.ics.uci.edu/dataset/2/adult

## Outputs

- `results/comparisons_long.csv` — one row per (dataset, lambda, seed, metric)
  with U statistic, p-value, Cliff's delta, group medians and sizes.
- `results/run_summary.csv` — one row per (dataset, lambda, seed) with coverage,
  average set size, ESS fraction, and shift diagnostics.

The dose-response curve and the headline comparison are both reconstructable
from `comparisons_long.csv`.

## Environment

Python 3.10. Install dependencies with:

```bash
pip install -r requirements.txt
```

## Licence

Released for academic reference.