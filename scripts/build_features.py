# scripts/build_features.py
from __future__ import annotations
import argparse
from pathlib import Path

def main() -> None:
    """
    CLI entry point for building standardized features.

    This script delegates to `build_features` and exposes:
    - --panel: input market panel parquet path
    - --out: output features parquet path
    - --standardize: {"rolling","global","none"}
    - --rolling-window: rolling window size for rolling z-score

    Example
    -------
    python scripts/build_features.py \
        --panel data/processed/market_panel.parquet \
        --out data/processed/features.parquet \
        --standardize rolling \
        --rolling-window 252
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=str, default="data/processed/market_panel.parquet")
    ap.add_argument("--out", type=str, default="data/processed/features.parquet")
    ap.add_argument("--standardize", type=str, choices=["rolling", "global", "none"], default="rolling")
    ap.add_argument("--rolling-window", type=int, default=252)
    args = ap.parse_args()

    from src.features.build_features import build_features

    out_path = build_features(
        panel_path=Path(args.panel),
        out_path=Path(args.out),
        standardize=args.standardize,  # type: ignore[arg-type]
        rolling_window=args.rolling_window,
    )
    print(f"Features saved to: {out_path}")

if __name__ == "__main__":
    """
    Module execution guard.
    """
    main()

