# scripts/validate_features.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import PROC_DIR
from src.data.utils import panel_quality_report


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for the features validation script.
    """
    p = argparse.ArgumentParser(description="Validate the features DataFrame and print summary statistics.")
    p.add_argument(
        "--features",
        type=Path,
        default=PROC_DIR / "features_custom.parquet",
        help="Path to the features parquet file.",
    )
    return p.parse_args()


def main() -> None:
    """
    Load the features file, print summary, first/last valid dates, and NaN report.
    """
    args = parse_args()
    df = pd.read_parquet(args.features)

    print("=== Features summary ===")
    print(f"path: {args.features}")
    print(f"shape: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"date range: {df.index.min()} -> {df.index.max()}")
    print(f"columns: {list(df.columns)}\n")

    print("=== First/Last valid per column ===")
    first = df.apply(lambda s: s.first_valid_index())
    last = df.apply(lambda s: s.last_valid_index())
    print(pd.DataFrame({"first_valid": first, "last_valid": last}))
    print()

    print("=== Quality report ===")
    print(panel_quality_report(df).sort_values("pct_na", ascending=False))


if __name__ == "__main__":
    main()
