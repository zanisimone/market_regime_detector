from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main() -> None:
    """
    CLI entry point for generating regime performance reports.

    This script loads a market panel and labeled regimes, computes:
    - per_regime.csv: daily statistics by regime
    - segments.csv: contiguous regime segments with returns and duration
    - overall.csv: overall daily statistics

    Default paths are taken from `src.config`.

    Example
    -------
    python scripts/report_regime_stats.py
    python scripts/report_regime_stats.py \
        --price-col SPX \
        --out-prefix data/processed/my_report
    """
    from src.config import PANEL_PARQUET, KMEANS_LABELS_PARQUET, PROC_DIR
    from src.eval.report import regime_report

    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=str, default=str(PANEL_PARQUET))
    ap.add_argument("--labels", type=str, default=str(KMEANS_LABELS_PARQUET))
    ap.add_argument("--price-col", type=str, default="SPX")
    ap.add_argument("--out-prefix", type=str, default=str(PROC_DIR / "regime_report"))
    args = ap.parse_args()

    panel = pd.read_parquet(Path(args.panel))
    labeled = pd.read_parquet(Path(args.labels))

    rpt = regime_report(panel, labeled, price_col=args.price_col)

    per_regime_path = Path(f"{args.out_prefix}_per_regime.csv")
    segments_path = Path(f"{args.out_prefix}_segments.csv")
    overall_path = Path(f"{args.out_prefix}_overall.csv")

    rpt.per_regime.to_csv(per_regime_path, index=False)
    rpt.segments.to_csv(segments_path, index=False)
    rpt.overall.to_csv(overall_path, index=False)

    print(f"per_regime -> {per_regime_path}")
    print(f"segments   -> {segments_path}")
    print(f"overall    -> {overall_path}")

if __name__ == "__main__":
    """
    Module execution guard.
    """
    main()
