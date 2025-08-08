# src/data/yahoo.py
from __future__ import annotations

import time
from typing import Iterable, Optional, List

import pandas as pd
import yfinance as yf


_YAHOO_RENAME_MAP = {
    "Open": "Open",
    "High": "High",
    "Low": "Low",
    "Close": "Close",
    "Adj Close": "Adj Close",
    "Volume": "Volume",
}


def fetch_yahoo(
    ticker: str,
    start: str,
    end: str,
    *,
    interval: str = "1d",
    auto_adjust: bool = False,
    columns: Optional[Iterable[str]] = None,
    retries: int = 2,
    retry_wait_sec: float = 1.0,
    progress: bool = False,
) -> pd.DataFrame:
    """
    Download historical market data from Yahoo Finance for a single ticker.

    Parameters
    ----------
    ticker : str
        Yahoo ticker, e.g., '^GSPC', '^VIX', 'TLT'.
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str
        End date in 'YYYY-MM-DD' format.
    interval : str, default '1d'
        Data interval, e.g., '1d', '1wk', '1mo'.
    auto_adjust : bool, default False
        If True, prices are adjusted by Yahoo; 'Adj Close' may be omitted.
    columns : Iterable[str] | None, default None
        Subset of columns to keep. Allowed values:
        {'Open','High','Low','Close','Adj Close','Volume'}.
        If None, all available columns are returned.
    retries : int, default 2
        Number of retry attempts on transient errors or empty responses.
    retry_wait_sec : float, default 1.0
        Seconds to wait between retries.
    progress : bool, default False
        Whether to show yfinance progress bar.

    Returns
    -------
    pd.DataFrame
        Time-indexed (tz-naive), sorted, deduplicated DataFrame with standardized column names.

    Raises
    ------
    ValueError
        If no data is returned after cleaning or all retries fail.
    """
    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=auto_adjust,
                progress=progress,
                threads=True,
            )

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [" ".join([c for c in tup if c]).strip() for tup in df.columns]

            if not df.empty:
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df = df.sort_index()
                df = df[~df.index.duplicated(keep="last")]
                df = df.rename(columns=_YAHOO_RENAME_MAP)

                if columns is not None:
                    keep: List[str] = [c for c in columns if c in df.columns]
                    if len(keep) == 0:
                        raise ValueError(
                            f"No requested columns are present for {ticker}. "
                            f"Available: {list(df.columns)}"
                        )
                    df = df[keep]

                if df.empty:
                    raise ValueError(f"Empty data for {ticker} after cleaning.")

                return df

            last_err = ValueError(f"Yahoo returned empty dataframe for {ticker}.")
        except Exception as e:
            last_err = e

        if attempt < retries:
            time.sleep(retry_wait_sec)

    msg = f"Failed to download Yahoo data for {ticker} between {start} and {end}."
    if last_err:
        msg += f" Last error: {last_err}"
    raise ValueError(msg)
