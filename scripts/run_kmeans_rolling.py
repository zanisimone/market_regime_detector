from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main() -> None:
    """
    CLI entry point for KMeans rolling backtest with reporting.

    This script:
    1) Iteratively fits KMeans on rolling windows.
    2) Predicts out-of-sample labels for each iteration.
    3) Saves the merged rolling labels and schedule.
    4) Generates a regime performance report for the labeled period.

    Example
    -------
    python scripts/run_kmeans_rolling.py \
        --start 2010-01-01 --end 2024-12-31 \
        --lookback-days 504 --oos-days 21 --step-days 21 \
        --k 4 --price-col SPX
    """
    from src.config import FEATURES_PARQUET, PROC_DIR, PANEL_PARQUET
    from src.models.kmeans_rolling import run_kmeans_rolling
    from src.eval.report import regime_report

    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, default=str(FEATURES_PARQUET))
    ap.add_argument("--panel", type=str, default=str(PANEL_PARQUET))
    ap.add_argument("--out-dir", type=str, default=str(PROC_DIR / "kmeans_rolling"))
    ap.add_argument("--start", type=str, required=True)
    ap.add_argument("--end", type=str, required=True)
    ap.add_argument("--lookback-days", type=int, default=504)
    ap.add_argument("--oos-days", type=int, default=21)
    ap.add_argument("--step-days", type=int, default=21)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--n-init", type=int, default=20)
    ap.add_argument("--price-col", type=str, default="SPX")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    labels_path, schedule_path = run_kmeans_rolling(
        features_path=Path(args.features),
        out_dir=out_dir,
        start=args.start,
        end=args.end,
        lookback_days=args.lookback_days,
        oos_days=args.oos_days,
        step_days=args.step_days,
        k=args.k,
        random_state=args.random_state,
        n_init=args.n_init,
    )

    panel = pd.read_parquet(Path(args.panel))
    labels = pd.read_parquet(labels_path)
    rpt = regime_report(panel, labels, price_col=args.price_col)

    rpt.per_regime.to_csv(out_dir / "rolling_per_regime.csv", index=False)
    rpt.overall.to_csv(out_dir / "rolling_overall.csv", index=False)
    rpt.segments.to_csv(out_dir / "rolling_segments.csv", index=False)

    print(f"labels   -> {labels_path}")
    print(f"schedule -> {schedule_path}")

if __name__ == "__main__":
    """
    Module execution guard.
    """
    main()
