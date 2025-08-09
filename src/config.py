# src/config.py
from __future__ import annotations
from pathlib import Path
from datetime import date

# Root del progetto (puoi adeguarla se usi un layout diverso)
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
EXT_DIR = DATA_DIR / "external"
FIG_DIR = ROOT / "figures"

# Default date window (puoi cambiare)
DEFAULT_START = "2005-01-01"
DEFAULT_END = date.today().isoformat()

# Mapping alias -> ticker / serie
YAHOO_TICKERS = {
    "SPX": "^GSPC",
    "VIX": "^VIX",
    "TLT": "TLT",
}

FRED_SERIES = {
    "US10Y": "DGS10",
    "US2Y": "DGS2",
}

# File standard
PANEL_PARQUET = PROC_DIR / "market_panel.parquet"
FEATURES_PARQUET = PROC_DIR / "features.parquet"
KMEANS_LABELS_PARQUET = PROC_DIR / "kmeans_labels.parquet"
KMEANS_CENTERS_CSV = PROC_DIR / "kmeans_centers.csv"
KMEANS_MODEL_PKL = PROC_DIR / "kmeans_model.pkl"  # opzionale: salvataggio modello
