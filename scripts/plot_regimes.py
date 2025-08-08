# scripts/plot_regimes.py
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import PROC_DIR
from src.viz.plots import load_labeled_features, load_price_series, plot_price_with_regimes


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for plotting a price series segmented by detected regimes.
    """
    p = argparse.ArgumentParser(description="Plot price with K-Means regime segmentation.")
    p.add_argument(
        "--labels",
        type=Path,
        default=PROC_DIR / "kmeans_labels.parquet",
        help="Path to labeled features parquet.",
    )
    p.add_argument(
        "--panel",
        type=Path,
        default=PROC_DIR / "market_panel.parquet",
        help="Path to processed market panel parquet.",
    )
    p.add_argument(
        "--price-col",
        type=str,
        default="SPX",
        help="Price column name to plot from the panel.",
    )
    p.add_argument(
        "--title",
        type=str,
        default="SPX with Market Regimes",
        help="Plot title.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=PROC_DIR / "regimes_plot.png",
        help="Path to save the output PNG. If omitted, the plot is shown.",
    )
    return p.parse_args()


def main() -> None:
    """
    Load labeled features and a price series, then render and save the regime plot.
    """
    args = parse_args()
    labeled = load_labeled_features(args.labels)
    price = load_price_series(args.panel, price_col=args.price_col)
    out = plot_price_with_regimes(
        labeled=labeled,
        price=price,
        label_col="regime",
        name_col="regime_name",
        title=args.title,
        out_path=args.out,
    )
    print(f"Saved regime plot to: {out}" if out else "Displayed regime plot.")


if __name__ == "__main__":
    main()
