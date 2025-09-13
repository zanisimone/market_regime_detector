from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main() -> None:
    """
    CLI entry point for HMM rolling backtest with reporting.
    """
    from src.config import FEATURES_PARQUET, PROC_DIR, PANEL_PARQUET, REPORTS_DIR
    from src.models.hmm_rolling import run_hmm_rolling
    from src.eval.report import regime_report

    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, default=str(FEATURES_PARQUET))
    ap.add_argument("--panel", type=str, default=str(PANEL_PARQUET))
    ap.add_argument("--out-dir", type=str, default=str(PROC_DIR / "hmm_rolling"))
    ap.add_argument("--reports-dir", type=str, default=str(REPORTS_DIR / "hmm_rolling"))
    ap.add_argument("--start", type=str, required=True)
    ap.add_argument("--end", type=str, required=True)
    ap.add_argument("--lookback-days", type=int, default=504)
    ap.add_argument("--oos-days", type=int, default=21)
    ap.add_argument("--step-days", type=int, default=21)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--n-iter", type=int, default=200)
    ap.add_argument("--covariance-type", type=str, default="full", choices=["full", "diag", "spherical", "tied"])
    ap.add_argument("--price-col", type=str, default="SPX")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    reports_dir = Path(args.reports_dir)
    
    # Create reports directory
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Create files directly in reports directory (no data/processed)
    labels_path, schedule_path = run_hmm_rolling(
        features_path=Path(args.features),
        out_dir=reports_dir,  # Create directly in reports_dir
        start=args.start,
        end=args.end,
        lookback_days=args.lookback_days,
        oos_days=args.oos_days,
        step_days=args.step_days,
        k=args.k,
        random_state=args.random_state,
        n_iter=args.n_iter,
        covariance_type=args.covariance_type,
    )

    panel = pd.read_parquet(Path(args.panel))
    labels = pd.read_parquet(labels_path)
    rpt = regime_report(panel, labels, price_col=args.price_col)

    # Save reports in reports directory
    rpt.per_regime.to_csv(reports_dir / "rolling_per_regime.csv", index=False)
    rpt.overall.to_csv(reports_dir / "rolling_overall.csv", index=False)
    rpt.segments.to_csv(reports_dir / "rolling_segments.csv", index=False)

    print(f"labels   -> {labels_path}")
    print(f"schedule -> {schedule_path}")
    print(f"reports  -> {reports_dir}")

    print("Note: HMM requires `pip install hmmlearn`.")

if __name__ == "__main__":
    main()
