from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main() -> None:
    """
    CLI entry point for Gaussian Mixture train/test split with reporting.
    """
    from src.config import FEATURES_PARQUET, PROC_DIR, PANEL_PARQUET, REPORTS_DIR
    from src.models.gmm_split import run_gmm_split
    from src.eval.report import regime_report

    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, default=str(FEATURES_PARQUET))
    ap.add_argument("--panel", type=str, default=str(PANEL_PARQUET))
    ap.add_argument("--out-dir", type=str, default=str(PROC_DIR / "gmm_split"))
    ap.add_argument("--reports-dir", type=str, default=str(REPORTS_DIR / "gmm_split"))
    ap.add_argument("--train-start", type=str, required=True)
    ap.add_argument("--train-end", type=str, required=True)
    ap.add_argument("--test-start", type=str, required=True)
    ap.add_argument("--test-end", type=str, required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--n-init", type=int, default=5)
    ap.add_argument("--covariance-type", type=str, default="full", choices=["full", "tied", "diag", "spherical"])
    ap.add_argument("--price-col", type=str, default="SPX")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    reports_dir = Path(args.reports_dir)
    
    # Create reports directory
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    train_labels_path, test_labels_path, model_path, centers_path = run_gmm_split(
        features_path=Path(args.features),
        out_dir=out_dir,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        k=args.k,
        random_state=args.random_state,
        n_init=args.n_init,
        covariance_type=args.covariance_type,
    )

    panel = pd.read_parquet(Path(args.panel))
    train_labels = pd.read_parquet(train_labels_path)
    test_labels = pd.read_parquet(test_labels_path)

    train_report = regime_report(panel, train_labels, price_col=args.price_col)
    test_report = regime_report(panel, test_labels, price_col=args.price_col)

    # Save reports in reports directory
    train_report.per_regime.to_csv(reports_dir / "train_per_regime.csv", index=False)
    test_report.per_regime.to_csv(reports_dir / "test_per_regime.csv", index=False)
    train_report.overall.to_csv(reports_dir / "train_overall.csv", index=False)
    test_report.overall.to_csv(reports_dir / "test_overall.csv", index=False)
    train_report.segments.to_csv(reports_dir / "train_segments.csv", index=False)
    test_report.segments.to_csv(reports_dir / "test_segments.csv", index=False)

    print(f"train_labels -> {train_labels_path}")
    print(f"test_labels  -> {test_labels_path}")
    print(f"model        -> {model_path}")
    print(f"means        -> {centers_path}")
    print(f"reports      -> {reports_dir}")

if __name__ == "__main__":
    main()
