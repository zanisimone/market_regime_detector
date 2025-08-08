# src/data/fred.py
from __future__ import annotations

import os
import time
from typing import Optional, Iterable, Dict

import pandas as pd
from fredapi import Fred


def _get_fred(api_key: Optional[str] = None) -> Fred:
    """
    Create a Fred API client using the provided API key or the FRED_API_KEY environment variable.

    Parameters
    ----------
    api_key : str | None
        FRED API key. If None, will attempt to load from environment variable.

    Returns
    -------
    Fred
        An instance of the Fred API client.

    Raises
    ------
    ValueError
        If no API key is provided and none is found in the environment.
    """
    key = api_key or os.getenv("FRED_API_KEY")
    if not key:
        raise ValueError("FRED_API_KEY not found. Set it in .env or pass api_key to fetch_fred().")
    return Fred(api_key=key)


def fetch_fred(
    series_id: str,
    start: str,
    end: str,
    *,
    col_name: Optional[str] = None,
    api_key: Optional[str] = None,
    retries: int = 2,
    retry_wait_sec: float = 1.0,
) -> pd.DataFrame:
    """
    Download a single series from FRED as a one-column DataFrame indexed by date.

    Parameters
    ----------
    series_id : str
        FRED series code (e.g., 'DGS10', 'DGS2').
    start, end : str
        Date range in 'YYYY-MM-DD' format.
    col_name : str | None, default None
        Column name in output DataFrame. If None, series_id is used.
    api_key : str | None
        FRED API key. If None, will attempt to load from environment.
    retries : int, default 2
        Number of retry attempts for transient errors.
    retry_wait_sec : float, default 1.0
        Seconds to wait between retries.

    Returns
    -------
    pd.DataFrame
        DataFrame with one column (col_name or series_id), tz-naive DatetimeIndex, sorted.

    Raises
    ------
    ValueError
        If the series is empty or API key is missing.
    """
    name = col_name or series_id
    last_err: Optional[Exception] = None

    for attempt in range(retries + 1):
        try:
            fred = _get_fred(api_key=api_key)
            s = fred.get_series(series_id, observation_start=start, observation_end=end)

            if s is None or len(s) == 0:
                raise ValueError(f"No FRED data for '{series_id}' between {start} and {end}.")

            df = s.to_frame(name=name)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]
            df[name] = pd.to_numeric(df[name], errors="coerce")

            if df[name].isna().all():
                raise ValueError(f"All values are NaN for '{series_id}' after numeric conversion.")

            return df

        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(retry_wait_sec)

    msg = f"Failed to download FRED series '{series_id}' between {start} and {end}."
    if last_err:
        msg += f" Last error: {last_err}"
    raise ValueError(msg)


def fetch_fred_many(
    series: Iterable[str] | Dict[str, str],
    start: str,
    end: str,
    *,
    api_key: Optional[str] = None,
    retries: int = 2,
    retry_wait_sec: float = 1.0,
) -> Dict[str, pd.DataFrame]:
    """
    Download multiple FRED series at once.

    Parameters
    ----------
    series : Iterable[str] | Dict[str, str]
        Iterable of series_id → column will have the same name as series_id.
        Dict {alias: series_id} → column will be named with the alias.
    start, end : str
        Date range in 'YYYY-MM-DD' format.
    api_key : str | None
        FRED API key (fallback to environment variable).
    retries, retry_wait_sec : see fetch_fred().

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping column_name to one-column DataFrame.
    """
    out: Dict[str, pd.DataFrame] = {}

    if isinstance(series, dict):
        for alias, sid in series.items():
            out[alias] = fetch_fred(
                sid, start, end, col_name=alias, api_key=api_key, retries=retries, retry_wait_sec=retry_wait_sec
            )
    else:
        for sid in series:
            out[sid] = fetch_fred(
                sid, start, end, col_name=sid, api_key=api_key, retries=retries, retry_wait_sec=retry_wait_sec
            )

    return out
