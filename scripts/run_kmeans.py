# scripts/run_kmeans.py
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import FEATURES_PARQUET, KMEANS_LABELS_PARQUET, KMEANS_CENTERS_CSV, KMEANS_MODEL_PKL
from src.models.kmeans import run_kmeans


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for running K-Means clustering on features.
    All arguments have defaults so the script can run without any parameters.
    """
    p = argparse.ArgumentParser(description="Run K-Means clustering for market regime detection.")

    p.add_argument(
        "--features",
        type=Path,
        default=str(FEATURES_PARQUET),
        help="Path to features parquet.",
    )
    p.add_argument(
        "--out-labels",
        type=Path,
        default=str(KMEANS_LABELS_PARQUET),
        help="Path to save labeled features parquet.",
    )
    p.add_argument(
        "--out-centers",
        type=Path,
        default=str(KMEANS_CENTERS_CSV),
        help="Path to save cluster centers CSV.",
    )
    p.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of clusters.",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    p.add_argument(
        "--n-init",
        type=int,
        default=20,
        help="Number of centroid initializations.",
    )
    p.add_argument(
        "--features-cols",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of feature column names to use (space-separated). If None, use all features.",
    )
    return p.parse_args()


def main() -> None:
    """
    Execute K-Means with provided arguments (or defaults) and print output paths.
    """
    args = parse_args()
    out_labels, out_centers = run_kmeans(
        features_path=args.features,
        out_labels_path=args.out_labels,
        out_centers_path=args.out_centers,
        k=args.k,
        random_state=args.random_state,
        n_init=args.n_init,
        feature_cols=args.features_cols if args.features_cols else None,
    )
    print(f"Labeled features saved to: {out_labels}")
    print(f"Cluster centers saved to: {out_centers}")


if __name__ == "__main__":
    main()
