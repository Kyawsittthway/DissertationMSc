"""
Top-level orchestration for the dissertation experiments.

Runs the full grid:
    datasets x shift intensities (lambda) x random seeds
and writes tidy CSV results to results/. Designed to run unattended.

Usage
-----
    # Smoke test on synthetic data (no internet / CSVs needed):
    python run_experiments.py --synthetic --seeds 3 --quick

    # Real run once the CSV files are downloaded:
    python run_experiments.py \
        --telco path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv \
        --adult path/to/adult.csv \
        --seeds 20

Outputs
-------
    results/comparisons_long.csv  - one row per (dataset, lambda, seed, metric)
    results/run_summary.csv       - one row per (dataset, lambda, seed)

The dose-response curve (stability gap vs lambda) and the headline group
comparison are both reconstructable from comparisons_long.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make src importable whether run from repo root or src/.
SRC = Path(__file__).resolve().parent / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loading import load_adult, load_synthetic, load_telco  # noqa: E402
from experiment import run_once  # noqa: E402

DEFAULT_LAMBDAS = [0.0, 0.5, 1.0, 1.5, 2.0]
QUICK_LAMBDAS = [0.0, 1.0, 2.0]


def gather_datasets(args):
    datasets = []
    if args.synthetic:
        datasets.append(load_synthetic("telco", n=args.synthetic_n, seed=0))
        datasets.append(load_synthetic("adult", n=args.synthetic_n, seed=1))
        return datasets
    if args.telco:
        datasets.append(load_telco(args.telco))
    if args.adult is not None:
        # Empty string => fetch via OpenML
        datasets.append(load_adult(args.adult if args.adult else None))
    if not datasets:
        raise SystemExit(
            "No datasets specified. Use --synthetic, or --telco PATH and/or "
            "--adult PATH (or --adult '' to fetch Adult via OpenML)."
        )
    return datasets


def main():
    ap = argparse.ArgumentParser(description="Run dissertation CP+SHAP experiments")
    ap.add_argument("--telco", type=str, default=None, help="Path to Telco CSV")
    ap.add_argument(
        "--adult",
        type=str,
        default=None,
        help="Path to Adult CSV, or '' to fetch via OpenML",
    )
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data mimicking both schemas",
    )
    ap.add_argument("--synthetic-n", type=int, default=5000)
    ap.add_argument("--seeds", type=int, default=10, help="Number of seeds")
    ap.add_argument("--k", type=int, default=5, help="top-k for SHAP rankings")
    ap.add_argument("--alpha", type=float, default=0.1, help="CP miscoverage")
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Fewer lambdas and trees for a fast smoke test",
    )
    ap.add_argument("--outdir", type=str, default="results")
    args = ap.parse_args()

    lambdas = QUICK_LAMBDAS if args.quick else DEFAULT_LAMBDAS
    n_estimators = 80 if args.quick else args.n_estimators
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    datasets = gather_datasets(args)

    long_rows = []
    summary_rows = []

    for ds in datasets:
        print(
            f"\n=== Dataset: {ds.name} "
            f"(n={len(ds.X)}, features={ds.X.shape[1]}, "
            f"shift_feature={ds.shift_feature}, "
            f"churn_rate={ds.y.mean():.3f}) ==="
        )
        for lam in lambdas:
            for seed in range(args.seeds):
                out = run_once(
                    ds,
                    lam=lam,
                    seed=seed,
                    k=args.k,
                    n_estimators=n_estimators,
                    alpha=args.alpha,
                )
                summary_rows.append(
                    {
                        "dataset": out.dataset,
                        "lambda": out.lam,
                        "seed": out.seed,
                        "coverage_original": out.coverage_original,
                        "avg_set_size_original": out.avg_set_size_original,
                        "ess_fraction": out.ess_fraction,
                        "shift_mean_before": out.shift_mean_before,
                        "shift_mean_after": out.shift_mean_after,
                        "shift_sd_units": out.shift_sd_units,
                    }
                )
                c = out.comparisons.copy()
                c.insert(0, "seed", out.seed)
                c.insert(0, "lambda", out.lam)
                c.insert(0, "dataset", out.dataset)
                long_rows.append(c)
            print(f"  lambda={lam:>4}: done {args.seeds} seeds")

    comparisons_long = pd.concat(long_rows, ignore_index=True)
    run_summary = pd.DataFrame(summary_rows)

    comparisons_long.to_csv(outdir / "comparisons_long.csv", index=False)
    run_summary.to_csv(outdir / "run_summary.csv", index=False)

    print(f"\nWrote {outdir/'comparisons_long.csv'} " f"({len(comparisons_long)} rows)")
    print(f"Wrote {outdir/'run_summary.csv'} ({len(run_summary)} rows)")

    # Quick headline readout: mean Cliff's delta by dataset x lambda (spearman).
    sp = comparisons_long[comparisons_long["metric"] == "spearman"]
    pivot = sp.pivot_table(
        index=["dataset", "lambda"], values="cliffs_delta", aggfunc="mean"
    )
    print("\nMean Cliff's delta (spearman) by dataset x lambda:")
    print(pivot.round(3).to_string())


if __name__ == "__main__":
    main()
