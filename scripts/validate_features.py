#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path

def _import_validate():
    """
    Import validate_and_report from either src.validation or src.reporting.validate.
    """
    try:
        from src.validation.validation import validate_and_report  # type: ignore
        return validate_and_report
    except Exception:
        from src.reporting.validate import validate_and_report  # type: ignore
        return validate_and_report

def main() -> None:
    """
    CLI entry point for the validation and monitoring mini-report.
    """
    ap = argparse.ArgumentParser(description="Validate features, monitor drift, and export a mini-report.")
    ap.add_argument("--features", type=str, default="data/processed/features.parquet", help="Path to features parquet.")
    ap.add_argument("--out-dir", type=str, default="reports/feature_report", help="Output directory for the report.")
    ap.add_argument("--split-date", type=str, default=None, help="Calendar split date 'YYYY-MM-DD'.")
    ap.add_argument("--split-ratio", type=float, default=None, help="Train ratio in (0,1); used if --split-date is not provided.")
    ap.add_argument("--adf", action="store_true", default=False, help="Run ADF stationarity test if statsmodels is available.")
    ap.add_argument("--roll-window", type=int, default=126, help="Rolling window for OOS vs Train comparison.")
    args = ap.parse_args()

    validate_and_report = _import_validate()

    out = validate_and_report(
        features_path=Path(args.features),
        out_dir=Path(args.out_dir),
        split_date=args.split_date,
        split_ratio=args.split_ratio if args.split_date is None else None,
        run_adf=bool(args.adf),
        rolling_cmp_window=int(args.roll_window),
    )
    print("[OK] Validation report created.")
    print(f"[INFO] Summary: {out['summary']}")
    print(f"[INFO] Artifacts saved under: {args.out_dir}")

if __name__ == "__main__":
    main()
