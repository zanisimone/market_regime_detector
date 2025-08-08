from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from .yahoo import fetch_yahoo
from .fred import fetch_fred
from .utils import merge_on_calendar


def run_ingest(
    yahoo_tickers: Dict[str, str],
    fred_series: Dict[str, str],
    start: str,
    end: str,
    raw_dir: Path,
    processed_dir: Path,
) -> Path:
    """
    Orchestrate the data ingestion process:
    - Download Yahoo Finance and FRED series
    - Save each raw series to disk
    - Merge into a unified daily panel
    - Save the processed panel

    Parameters
    ----------
    yahoo_tickers : dict[str, str]
        Mapping {alias: yahoo_ticker}, e.g., {"SPX": "^GSPC"}.
    fred_series : dict[str, str]
        Mapping {alias: fred_series_id}, e.g., {"US10Y": "DGS10"}.
    start, end : str
        Date range in 'YYYY-MM-DD' format.
    raw_dir : Path
        Directory to store individual raw series.
    processed_dir : Path
        Directory to store the merged panel.

    Returns
    -------
    Path
        Path to the saved processed panel file.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    dfs: Dict[str, pd.DataFrame] = {}

    # Yahoo Finance series
    for alias, ticker in yahoo_tickers.items():
        df = fetch_yahoo(ticker, start, end)
        df.to_parquet(raw_dir / f"yahoo_{alias}.parquet")
        dfs[alias] = df

    # FRED series
    for alias, series_id in fred_series.items():
        df = fetch_fred(series_id, start, end, col_name=alias)
        df.to_parquet(raw_dir / f"fred_{alias}.parquet")
        dfs[alias] = df

    merged = merge_on_calendar(dfs)
    out_path = processed_dir / "market_panel.parquet"
    merged.to_parquet(out_path)

    return out_path
