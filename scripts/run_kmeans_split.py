from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main() -> None:
    """
    CLI entry point for KMeans train/test split with reporting.

    This script:
    1) Loads standardized features.
    2) Fits KMeans on train and predicts test.
    3) Saves labels and model artifacts.
    4) Generates regime performance reports for train and test.

    Example
    -------
    python scripts/run_kmeans_split.py \
        --train-start 2010-01-01 --train-end 2018-12-31 \
        --test-start 2019-01-01 --test-end 2024-12-31 \
        --k 4 --price-col SPX
    """
    from src.config import FEATURES_PARQUET, PROC_DIR, PANEL_PARQUET
    from src.models.kmeans_split import run_kmeans_split
    from src.eval.report import regime_report

    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, default=str(FEATURES_PARQUET))
    ap.add_argument("--panel", type=str, default=str(PANEL_PARQUET))
    ap.add_argument("--out-dir", type=str, default=str(PROC_DIR / "kmeans_split"))
    ap.add_argument("--train-start", type=str, required=True)
    ap.add_argument("--train-end", type=str, required=True)
    ap.add_argument("--test-start", type=str, required=True)
    ap.add_argument("--test-end", type=str, required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--n-init", type=int, default=20)
    ap.add_argument("--price-col", type=str, default="SPX")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    train_labels_path, test_labels_path, model_path, centers_path, labels_csv_path = run_kmeans_split(
        features_path=Path(args.features),
        out_dir=out_dir,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        k=args.k,
        random_state=args.random_state,
        n_init=args.n_init,
    )

    panel = pd.read_parquet(Path(args.panel))
    train_labels = pd.read_parquet(train_labels_path)
    test_labels = pd.read_parquet(test_labels_path)

    train_report = regime_report(panel, train_labels, price_col=args.price_col)
    test_report = regime_report(panel, test_labels, price_col=args.price_col)

    train_report.per_regime.to_csv(out_dir / "train_per_regime.csv", index=False)
    test_report.per_regime.to_csv(out_dir / "test_per_regime.csv", index=False)
    train_report.overall.to_csv(out_dir / "train_overall.csv", index=False)
    test_report.overall.to_csv(out_dir / "test_overall.csv", index=False)
    train_report.segments.to_csv(out_dir / "train_segments.csv", index=False)
    test_report.segments.to_csv(out_dir / "test_segments.csv", index=False)

    print(f"train_labels -> {train_labels_path}")
    print(f"test_labels  -> {test_labels_path}")
    print(f"model        -> {model_path}")
    print(f"centers      -> {centers_path}")

if __name__ == "__main__":
    """
    Module execution guard.
    """
    main()
