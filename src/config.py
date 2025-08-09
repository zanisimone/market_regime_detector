# src/config.py
from __future__ import annotations
from pathlib import Path
from datetime import date

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
EXT_DIR = DATA_DIR / "external"
FIG_DIR = ROOT / "figures"

# Default date window
DEFAULT_START = "2005-01-01"
DEFAULT_END = date.today().isoformat()

"""
Mapping alias -> Yahoo ticker.
Core indices + volatility + duration, plus ETF proxies for credit/FX/commodities and breakeven.
"""
YAHOO_TICKERS = {
    "SPX": "^GSPC",
    "VIX": "^VIX",
    "TLT": "TLT",
    "HYG": "HYG",
    "LQD": "LQD",
    "UUP": "UUP",
    "GLD": "GLD",
    "USO": "USO",
    "TIP": "TIP",
    "IEF": "IEF",
}

"""
Required tickers for core pipeline; if missing, the build should fail fast.
"""
YAHOO_REQUIRED = ["SPX", "VIX", "TLT"]

"""
Optional proxies used in Step 2; if missing, features are set to NaN and a warning is logged.
"""
YAHOO_OPTIONAL = ["HYG", "LQD", "UUP", "GLD", "USO", "TIP", "IEF"]

# FRED series mapping
FRED_SERIES = {
    "US10Y": "DGS10",
    "US2Y": "DGS2",
}

# Standard artifact paths
PANEL_PARQUET = PROC_DIR / "market_panel.parquet"
FEATURES_PARQUET = PROC_DIR / "features.parquet"
KMEANS_LABELS_PARQUET = PROC_DIR / "kmeans_labels.parquet"
KMEANS_CENTERS_CSV = PROC_DIR / "kmeans_centers.csv"
KMEANS_MODEL_PKL = PROC_DIR / "kmeans_model.pkl"
