# scripts/validate_panel.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import PROC_DIR
from src.data.utils import panel_quality_report


def _first_last_valid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute first and last valid timestamps for each column.
    """
    first = df.apply(lambda s: s.first_valid_index())
    last = df.apply(lambda s: s.last_valid_index())
    out = pd.DataFrame({"first_valid": first, "last_valid": last})
    return out


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for panel validation.
    """
    p = argparse.ArgumentParser(description="Validate processed market panel and print a quality report.")
    p.add_argument(
        "--panel",
        type=Path,
        default=PROC_DIR / "market_panel.parquet",
        help="Path to processed panel file (parquet).",
    )
    p.add_argument(
        "--save-report",
        type=Path,
        default=None,
        help="Optional path to save the quality report as CSV.",
    )
    return p.parse_args()


def main() -> None:
    """
    Load the processed panel, print summary stats and quality report, optionally save to CSV.
    """
    args = parse_args()
    df = pd.read_parquet(args.panel)

    print("=== Panel summary ===")
    print(f"path: {args.panel}")
    print(f"shape: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"date range: {df.index.min()} -> {df.index.max()}")
    print(f"columns: {list(df.columns)}")
    print()

    print("=== First/Last valid per column ===")
    fl = _first_last_valid(df)
    print(fl)
    print()

    print("=== Quality report ===")
    qr = panel_quality_report(df)
    print(qr.sort_values("pct_na", ascending=False))

    if args.save_report is not None:
        qr.join(fl).to_csv(args.save_report, index=True)
        print()
        print(f"Saved report to: {args.save_report}")


if __name__ == "__main__":
    main()
