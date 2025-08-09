import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from src.reporting.centers import unscale_centers, feature_contributions

def main():
    """
    CLI to inspect cluster centers and feature contributions.

    Usage
    -----
    python scripts/run_center_explorer.py --centers_csv reports/kmeans_centers.csv \
      --features_csv data/features.parquet --feature_cols "ret_1d,vol_21d,carry,curve_slope" \
      --means_stds_csv reports/feature_stats.csv --top 10 --out_prefix reports/centers

    centers_csv schema: rows=regimes, columns=features in z-space (or original if no stats provided).
    means_stds_csv schema: columns=[feature, mean, std].
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--centers_csv", required=True, help="CSV of centers (KxF) in z-space by default")
    parser.add_argument("--feature_cols", required=True, help="Comma-separated feature columns in order matching centers")
    parser.add_argument("--means_stds_csv", default="", help="Optional CSV with columns: feature, mean, std")
    parser.add_argument("--top", type=int, default=10, help="Top features per regime")
    parser.add_argument("--out_prefix", required=True, help="Output prefix")
    args = parser.parse_args()

    centers = pd.read_csv(args.centers_csv)
    feature_names = [c.strip() for c in args.feature_cols.split(",")]
    centers = centers[feature_names]
    Cz = centers.values.astype(float)

    mu = None
    sd = None
    if args.means_stds_csv:
        stats = pd.read_csv(args.means_stds_csv).set_index("feature").loc[feature_names]
        mu = stats["mean"].values.astype(float)
        sd = stats["std"].values.astype(float)

    C_orig = unscale_centers(Cz, mu, sd)
    pd.DataFrame(C_orig, columns=feature_names).to_csv(f"{args.out_prefix}_centers_original_scale.csv", index=False)

    contrib = feature_contributions(Cz, feature_names=feature_names, top=args.top, signed=True)
    contrib.to_csv(f"{args.out_prefix}_feature_contributions.csv", index=False)

if __name__ == "__main__":
    main()
