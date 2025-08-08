# src/config.py
from __future__ import annotations

from datetime import date
from pathlib import Path

# Directories
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"

# Date range
START_DATE = "2015-01-01"
END_DATE = date.today().isoformat()

# Yahoo Finance tickers (alias: ticker)
YAHOO_TICKERS = {
    "SPX": "^GSPC",   # S&P 500 Index
    "VIX": "^VIX",    # Volatility Index
    "TLT": "TLT",     # Long-term Treasury ETF (optional)
}

# FRED series (alias: series_id)
FRED_SERIES = {
    "US10Y": "DGS10",  # 10-Year Treasury Constant Maturity Rate
    "US2Y": "DGS2",    # 2-Year Treasury Constant Maturity Rate
}
