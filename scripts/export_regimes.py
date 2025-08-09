# scripts/export_regimes.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import PANEL_PARQUET, KMEANS_LABELS_PARQUET, PROC_DIR
from src.viz.plots import load_labeled_features, load_price_series


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for exporting the final regimes dataset.
    """
    p = argparse.ArgumentParser(description="Export date, price, regime_id, and regime_name to a CSV file.")
    p.add_argument(
        "--labels",
        type=Path,
        default=str(KMEANS_LABELS_PARQUET),
        help="Path to labeled features parquet.",
    )
    p.add_argument(
        "--panel",
        type=Path,
        default=str(PANEL_PARQUET),
        help="Path to processed market panel parquet.",
    )
    p.add_argument(
        "--price-col",
        type=str,
        default="SPX",
        help="Price column to include in the export.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=str(PROC_DIR / "regimes_dataset.csv"),
        help="Path to save the output CSV.",
    )
    return p.parse_args()


def main() -> None:
    """
    Load labeled features and the price series, join them, and save to CSV.
    """
    args = parse_args()

    labeled = load_labeled_features(args.labels)
    price = load_price_series(args.panel, price_col=args.price_col)

    df = pd.DataFrame({"price": price}).join(labeled[["regime", "regime_name"]], how="inner")
    df = df.reset_index().rename(columns={"index": "date"})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"Exported regimes dataset to: {args.out}")


if __name__ == "__main__":
    main()
