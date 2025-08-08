# scripts/build_features.py
from __future__ import annotations

import argparse
from pathlib import Path

from src.features.build_features import build_features


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for the feature building script.
    """
    p = argparse.ArgumentParser(description="Build standardized features for market regime detection.")
    p.add_argument("--panel", type=Path, default='data/processed/market_panel.parquet', help="Path to processed panel parquet.")
    p.add_argument("--out", type=Path, default='data/processed/features_custom.parquet', help="Path to save features parquet.")
    return p.parse_args()


def main() -> None:
    """
    Build features from the processed panel and print the output path.
    """
    args = parse_args()
    out = build_features(panel_path=args.panel, out_path=args.out)
    print(f"Features saved to: {out}")


if __name__ == "__main__":
    main()
