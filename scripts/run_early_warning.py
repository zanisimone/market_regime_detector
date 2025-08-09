import argparse
import pandas as pd
from pathlib import Path

from src.alerts.early_warning import early_warning_signal

def main():
    """
    CLI entry-point to compute early-warning heuristics on Risk-Off probability.

    Usage
    -----
    python scripts/run_early_warning.py --prob_csv data/risk_off_prob.csv --out_csv reports/early_warning.csv --prob_col prob
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--prob_csv", required=True, help="CSV with columns: date, prob (or specified via --prob_col).")
    parser.add_argument("--prob_col", default="prob", help="Column containing Risk-Off probability or indicator.")
    parser.add_argument("--ma_window", type=int, default=5, help="Short-term MA window.")
    parser.add_argument("--z_window", type=int, default=63, help="Baseline window for z-score.")
    parser.add_argument("--prob_threshold", type=float, default=0.60, help="MA probability trigger.")
    parser.add_argument("--z_threshold", type=float, default=1.0, help="Z-score trigger.")
    parser.add_argument("--slope_threshold", type=float, default=0.02, help="Slope trigger on MA.")
    parser.add_argument("--cool_off_days", type=int, default=5, help="Minimum distance between warnings.")
    parser.add_argument("--out_csv", required=True, help="Output CSV path.")
    args = parser.parse_args()

    df = pd.read_csv(args.prob_csv, parse_dates=["date"]).set_index("date")
    series = df[args.prob_col].astype(float)
    res = early_warning_signal(
        series,
        ma_window=args.ma_window,
        z_window=args.z_window,
        prob_threshold=args.prob_threshold,
        z_threshold=args.z_threshold,
        slope_threshold=args.slope_threshold,
        cool_off_days=args.cool_off_days,
    )
    out = res.reset_index().rename(columns={"index": "date"})
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()
