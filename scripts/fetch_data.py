# scripts/fetch_data.py
from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from src.config import RAW_DIR, PROC_DIR, DEFAULT_START, DEFAULT_END, YAHOO_TICKERS, FRED_SERIES
from src.data.ingest import run_ingest


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for the data fetching script.
    """
    parser = argparse.ArgumentParser(description="Fetch and merge Yahoo/FRED data into a unified panel.")
    parser.add_argument("--start", type=str, default=DEFAULT_START, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, default=DEFAULT_END, help="End date (YYYY-MM-DD).")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="Directory to store raw series.")
    parser.add_argument("--processed-dir", type=Path, default=PROC_DIR, help="Directory to store processed panel.")
    return parser.parse_args()


def main() -> None:
    """
    Load environment, run ingestion, and print the output path.
    """
    load_dotenv()
    args = parse_args()
    out = run_ingest(
        yahoo_tickers=YAHOO_TICKERS,
        fred_series=FRED_SERIES,
        start=args.start,
        end=args.end,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
    )
    print(f"Unified panel saved to: {out}")


if __name__ == "__main__":
    main()
