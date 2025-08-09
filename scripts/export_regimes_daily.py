from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main() -> None:
    """
    CLI entry point for exporting the full daily regimes dataset.

    The script joins the selected price series from the market panel with regime labels
    and exports a tidy daily table: date, price, regime, regime_name.

    Defaults are loaded from `src.config`. By default, both CSV and Parquet are written.

    Example
    -------
    python scripts/export_regimes_daily.py
    python scripts/export_regimes_daily.py --price-col SPX --csv-out data/processed/regimes_daily.csv
    """
    from src.config import PANEL_PARQUET, KMEANS_LABELS_PARQUET, PROC_DIR

    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=str, default=str(PANEL_PARQUET))
    ap.add_argument("--labels", type=str, default=str(KMEANS_LABELS_PARQUET))
    ap.add_argument("--price-col", type=str, default="SPX")
    ap.add_argument("--csv-out", type=str, default=str(PROC_DIR / "regimes_daily.csv"))
    ap.add_argument("--parquet-out", type=str, default=str(PROC_DIR / "regimes_daily.parquet"))
    args = ap.parse_args()

    panel = pd.read_parquet(Path(args.panel))
    labels = pd.read_parquet(Path(args.labels))

    cols = ["regime"] + (["regime_name"] if "regime_name" in labels.columns else [])
    df = panel[[args.price_col]].rename(columns={args.price_col: "price"}).join(labels[cols], how="inner").dropna()

    df.index.name = "date"
    Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv_out)
    df.to_parquet(args.parquet_out)

    print(f"daily dataset (csv)     -> {args.csv_out}")
    print(f"daily dataset (parquet) -> {args.parquet_out}")

if __name__ == "__main__":
    """
    Module execution guard.
    """
    main()
