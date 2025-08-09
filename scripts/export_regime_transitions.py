# scripts/export_regime_transitions.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import PANEL_PARQUET, KMEANS_LABELS_PARQUET, PROC_DIR
from src.viz.plots import load_labeled_features, load_price_series


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for exporting regime transition summary.
    """
    p = argparse.ArgumentParser(description="Export regime transitions with start/end dates, duration, prices, and returns.")
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
        help="Price column to compute start/end prices and returns.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=str(PROC_DIR / "regime_transitions.csv"),
        help="Path to save the transitions CSV.",
    )
    return p.parse_args()


def _extract_runs(df: pd.DataFrame, label_col: str = "regime", name_col: Optional[str] = "regime_name") -> pd.DataFrame:
    """
    Extract contiguous runs of the same regime from a labeled DataFrame.
    """
    df = df.copy()
    df["__regime"] = df[label_col].astype(int)
    df["__block"] = (df["__regime"].ne(df["__regime"].shift())).cumsum()

    grouped = df.groupby("__block")
    start = grouped.apply(lambda g: g.index.min())
    end = grouped.apply(lambda g: g.index.max())
    regime = grouped["__regime"].first()

    out = pd.DataFrame({"start_date": start, "end_date": end, "regime": regime})
    if name_col and name_col in df.columns:
        name = grouped[name_col].first()
        out["regime_name"] = name
    return out.reset_index(drop=True)


def _attach_prices(transitions: pd.DataFrame, price: pd.Series) -> pd.DataFrame:
    """
    Attach start/end prices and compute returns for each regime run.
    """
    p = price.sort_index()
    starts = p.reindex(transitions["start_date"]).to_numpy()
    ends = p.reindex(transitions["end_date"]).to_numpy()
    returns = np.where((starts > 0) & (ends > 0), ends / starts - 1.0, np.nan)

    out = transitions.copy()
    out["start_price"] = starts
    out["end_price"] = ends
    out["ret"] = returns
    out["duration_days"] = (out["end_date"] - out["start_date"]).dt.days + 1
    return out


def main() -> None:
    """
    Load labeled features and price series, compute regime transitions, and export to CSV.
    """
    args = parse_args()

    labeled = load_labeled_features(args.labels)
    if "regime" not in labeled.columns:
        raise ValueError("Missing 'regime' column in labeled features.")
    price = load_price_series(args.panel, price_col=args.price_col)

    joined = labeled[["regime"] + (["regime_name"] if "regime_name" in labeled.columns else [])].join(
        price.to_frame(name="price"), how="inner"
    )

    transitions = _extract_runs(joined, label_col="regime", name_col="regime_name" if "regime_name" in joined.columns else None)
    transitions = _attach_prices(transitions, price)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    transitions.to_csv(args.out, index=False)
    print(f"Exported regime transitions to: {args.out}")


if __name__ == "__main__":
    main()
