#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path

def _import_build_features():
    """
    Import `build_features` from your codebase, trying both common layouts.
    """
    try:
        from src.build_features import build_features  # type: ignore
        return build_features
    except Exception:
        from src.features.build_features import build_features  # type: ignore
        return build_features

def _print_artifact_info(path: Path) -> None:
    """
    Print a quick summary of the generated features parquet (shape + columns).
    """
    try:
        import pandas as pd
        df = pd.read_parquet(path)
        cols = ", ".join(df.columns.tolist())
        print(f"[INFO] Features shape: {df.shape[0]} rows x {df.shape[1]} cols")
        print(f"[INFO] Columns: {cols}")
    except Exception as e:
        print(f"[WARN] Could not read parquet for summary: {e}")

def main() -> None:
    """
    CLI entry point for building standardized features with robust transforms.
    """
    ap = argparse.ArgumentParser(description="Build features with bundles, winsorization, and optional rolling PCA.")
    ap.add_argument("--panel", type=str, default="data/processed/market_panel.parquet", help="Input market panel parquet path.")
    ap.add_argument("--out", type=str, default="data/processed/features.parquet", help="Output features parquet path.")
    ap.add_argument("--standardize", type=str, choices=["rolling", "global", "none"], default="rolling", help="Standardization mode for base features.")
    ap.add_argument("--window", type=int, default=252, help="Rolling window for z-scores and defaults.")
    ap.add_argument("--winsor", action="store_true", default=True, help="Enable rolling winsorization.")
    ap.add_argument("--no-winsor", dest="winsor", action="store_false", help="Disable rolling winsorization.")
    ap.add_argument("--winsor-window", type=int, default=252, help="Window for rolling winsorization.")
    ap.add_argument("--winsor-q", type=float, nargs=2, default=(0.01, 0.99), help="Lower and upper quantiles for winsorization.")
    ap.add_argument("--bundle", type=str, choices=["core", "market_plus", "macro_plus"], default="market_plus", help="Feature bundle selection.")
    ap.add_argument("--pca", action="store_true", default=False, help="Enable rolling PCA and append PC1/PC2.")
    ap.add_argument("--pca-window", type=int, default=252, help="Window for rolling PCA.")
    ap.add_argument("--pca-replace", action="store_true", default=False, help="Replace selected features with PC1/PC2 instead of appending.")
    args = ap.parse_args()

    build_features = _import_build_features()

    out_path = build_features(
        panel_path=Path(args.panel),
        out_path=Path(args.out),
        standardize=args.standardize,  # type: ignore[arg-type]
        rolling_window=int(args.window),
        enable_winsor=bool(args.winsor),
        winsor_window=int(args.winsor_window),
        winsor_q_low=float(args.winsor_q[0]),
        winsor_q_high=float(args.winsor_q[1]),
        bundle=str(args.bundle),
        enable_pca=bool(args.pca),
        pca_window=int(args.pca_window),
        pca_replace=bool(args.pca_replace),
    )
    print(f"[OK] Features saved to: {out_path}")
    _print_artifact_info(Path(out_path))

if __name__ == "__main__":
    main()
