import argparse
import pandas as pd
from pathlib import Path

from src.reporting.composites import regime_path_composites

def main():
    """
    CLI to compute normalized price path composites around regime starts.

    Usage
    -----
    python scripts/run_composites.py --price_csv data/px.csv --labels_csv reports/labels.csv \
      --regime 0 --lookback 5 --lookahead 20 --mode rebased --out_csv reports/composite_0.csv

    price_csv must have columns: date, price
    labels_csv must have columns: date, regime
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--price_csv", required=True, help="CSV with columns date, price")
    parser.add_argument("--labels_csv", required=True, help="CSV with columns date, regime")
    parser.add_argument("--regime", type=int, required=True, help="Regime id to anchor on")
    parser.add_argument("--lookback", type=int, default=5, help="Bars before event")
    parser.add_argument("--lookahead", type=int, default=20, help="Bars after event")
    parser.add_argument("--mode", default="rebased", choices=["rebased", "logret"], help="Normalization mode")
    parser.add_argument("--min_run", type=int, default=3, help="Minimum run length to count event")
    parser.add_argument("--out_csv", required=True, help="Output CSV path")
    args = parser.parse_args()

    px = pd.read_csv(args.price_csv, parse_dates=["date"]).set_index("date")["price"]
    lab = pd.read_csv(args.labels_csv, parse_dates=["date"]).set_index("date")["regime"]

    comp = regime_path_composites(
        price=px,
        labels=lab,
        regime_id=args.regime,
        lookback=args.lookback,
        lookahead=args.lookahead,
        mode=args.mode,
        min_run=args.min_run,
    )
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    comp.to_csv(args.out_csv, index=True)

if __name__ == "__main__":
    main()
