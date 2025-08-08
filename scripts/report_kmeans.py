# scripts/report_kmeans.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.config import PROC_DIR


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for the K-Means clustering report.
    """
    p = argparse.ArgumentParser(description="Generate an analytical report from K-Means labeled features.")
    p.add_argument(
        "--labels",
        type=Path,
        default=PROC_DIR / "kmeans_labels.parquet",
        help="Path to labeled features parquet.",
    )
    p.add_argument(
        "--centers",
        type=Path,
        default=PROC_DIR / "kmeans_centers.csv",
        help="Path to cluster centers CSV.",
    )
    return p.parse_args()


def _run_lengths(labels: np.ndarray) -> Dict[int, Tuple[int, float, int]]:
    """
    Compute run-length statistics per label: (count_runs, mean_length, max_length).
    """
    stats: Dict[int, Tuple[int, float, int]] = {}
    if labels.size == 0:
        return stats

    uniq = np.unique(labels)
    for u in uniq:
        indices = np.where(labels == u)[0]
        if indices.size == 0:
            stats[int(u)] = (0, 0.0, 0)
            continue
        breaks = np.where(np.diff(indices) > 1)[0]
        starts = np.r_[0, breaks + 1]
        ends = np.r_[breaks, indices.size - 1]
        lengths = indices[ends] - indices[starts] + 1
        count_runs = lengths.size
        mean_len = float(np.mean(lengths)) if lengths.size else 0.0
        max_len = int(np.max(lengths)) if lengths.size else 0
        stats[int(u)] = (count_runs, mean_len, max_len)
    return stats


def _transition_matrix(labels: np.ndarray) -> pd.DataFrame:
    """
    Compute the first-order transition matrix between cluster labels.
    """
    if labels.size < 2:
        return pd.DataFrame()
    prev = labels[:-1]
    curr = labels[1:]
    df = pd.crosstab(pd.Series(prev, name="prev"), pd.Series(curr, name="curr"), normalize="index")
    return df


def main() -> None:
    """
    Load labeled features and centers, then print a concise clustering report:
    - Overall coverage and date range
    - Cluster sizes
    - Cluster centers (z-scored features)
    - Regime name mapping distribution
    - Transition matrix
    - Run-length statistics (count, mean, max) per cluster
    """
    args = parse_args()
    labeled = pd.read_parquet(args.labels)
    centers = pd.read_csv(args.centers, index_col=0)

    print("=== Labeled features summary ===")
    print(f"path: {args.labels}")
    print(f"shape: {labeled.shape[0]} rows x {labeled.shape[1]} cols")
    print(f"date range: {labeled.index.min()} -> {labeled.index.max()}")
    print(f"columns: {list(labeled.columns)}\n")

    if "regime" not in labeled.columns:
        raise ValueError("Missing 'regime' column in labeled features.")
    name_col = "regime_name" if "regime_name" in labeled.columns else None

    print("=== Cluster sizes ===")
    counts = labeled["regime"].value_counts().sort_index()
    print(counts.to_frame(name="obs"))
    print()

    print("=== Cluster centers (z-scores) ===")
    print(centers)
    print()

    if name_col:
        print("=== Regime name distribution ===")
        print(labeled[name_col].value_counts())
        print()

    print("=== Transition matrix (prev -> curr, row-normalized) ===")
    tm = _transition_matrix(labeled["regime"].to_numpy())
    print(tm)
    print()

    print("=== Run-length statistics per cluster ===")
    rstats = _run_lengths(labeled["regime"].to_numpy())
    if rstats:
        rl_df = pd.DataFrame.from_dict(
            rstats, orient="index", columns=["runs", "mean_len", "max_len"]
        ).sort_index()
        print(rl_df)
    else:
        print("Insufficient data for run-length analysis.")


if __name__ == "__main__":
    main()
