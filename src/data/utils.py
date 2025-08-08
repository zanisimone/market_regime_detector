from typing import Optional, Dict, Sequence, Iterable

import pandas as pd


def merge_on_calendar(
    dfs: Dict[str, pd.DataFrame],
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    freq: str = "D",
    fred_prefixes: Sequence[str] = ("US",),
    extra_macro: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Merge multiple time series onto a continuous calendar and forward-fill macro series.
    Accepts price DataFrames that have either 'Close' or 'Adj Close' columns (case-insensitive),
    or single-column DataFrames. Preference order for price selection is: 'Adj Close' -> 'Close'.
    """
    if not dfs:
        raise ValueError("No input DataFrames provided to merge_on_calendar().")

    def _select_price_col(dfi: pd.DataFrame, name: str) -> pd.DataFrame:
        """
        Select a single price column for merging, tolerating yfinance labels like
        'Adj Close ^GSPC' or 'Close ^GSPC'. Preference: 'Adj Close' -> 'Close'.
        """
        cols = list(dfi.columns)
        # try prefixes 'Adj Close' and 'Close' (case-insensitive), allowing ticker suffixes
        for pref in ("adj close", "close"):
            cand = [c for c in cols if c.lower().startswith(pref)]
            if cand:
                col = cand[0]
                return dfi[[col]].rename(columns={col: name})

        # fallback: single-column DF
        if dfi.shape[1] == 1:
            return dfi.rename(columns={dfi.columns[0]: name})

        raise ValueError(
            f"DataFrame for '{name}' must contain 'Close'/'Adj Close' (with or without ticker suffix) or be single-column. "
            f"Got: {cols}"
        )

    prepared = []
    for name, df in dfs.items():
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"DataFrame '{name}' must have a DatetimeIndex.")
        dfi = df.copy()
        dfi.index = pd.to_datetime(dfi.index).tz_localize(None)
        dfi = dfi.sort_index()
        dfi = dfi[~dfi.index.duplicated(keep="last")]
        prepared.append(_select_price_col(dfi, name))

    merged = None
    for dfx in prepared:
        merged = dfx if merged is None else merged.join(dfx, how="outer")

    cal_idx = pd.date_range(
        start=start or merged.index.min(),
        end=end or merged.index.max(),
        freq=freq
    )
    merged = merged.reindex(cal_idx)

    macro_cols = []
    for c in merged.columns:
        if any(c.upper().startswith(p.upper()) for p in fred_prefixes):
            macro_cols.append(c)
    if extra_macro:
        macro_cols = list(set(macro_cols + list(extra_macro)))

    if macro_cols:
        merged[macro_cols] = merged[macro_cols].ffill()

    return merged

def panel_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a simple quality report with coverage and NaN percentages per column.

    Parameters
    ----------
    df : pd.DataFrame
        Input panel.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['count', 'non_na', 'na', 'pct_na'] indexed by original columns.
    """
    total = len(df)
    counts = df.count()
    nas = df.isna().sum()
    pct = (nas / total * 100.0).round(2) if total > 0 else 0.0
    out = pd.DataFrame(
        {"count": total, "non_na": counts, "na": nas, "pct_na": pct},
    )
    return out

