from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence, Literal

import numpy as np
import pandas as pd

from src.config import PROC_DIR

StandardizeMode = Literal["global", "rolling", "none"]

def _zscore_global(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """
    Apply global z-score standardization on selected columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with raw features.
    cols : list of str
        Columns to standardize.

    Returns
    -------
    pd.DataFrame
        Standardized DataFrame.
    """
    out = df.copy()
    for c in cols:
        mu = out[c].mean()
        sd = out[c].std(ddof=0)
        out[c] = (out[c] - mu) / sd if sd and not np.isnan(sd) else out[c] * 0.0
    return out

def _zscore_rolling(df: pd.DataFrame, cols: Sequence[str], window: int = 252, minp: int = 126) -> pd.DataFrame:
    """
    Apply rolling z-score standardization to avoid look-ahead bias.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with raw features.
    cols : list of str
        Columns to standardize.
    window : int, default 252
        Rolling window size in business days.
    minp : int, default 126
        Minimum periods required to compute rolling stats.

    Returns
    -------
    pd.DataFrame
        Standardized DataFrame with NaNs in the initial periods without enough history.
    """
    out = df.copy()
    for c in cols:
        roll_mean = out[c].rolling(window, min_periods=minp).mean()
        roll_std = out[c].rolling(window, min_periods=minp).std(ddof=0)
        z = (out[c] - roll_mean) / roll_std
        out[c] = z
    return out

def _prepare_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the market panel for feature computation:
    - Set frequency to business days.
    - Forward-fill and backward-fill macroeconomic series.
    - Remove rows without price data.
    """
    df = df.asfreq("B")
    macro_cols = [c for c in df.columns if c.upper().startswith("US")]
    if macro_cols:
        df[macro_cols] = df[macro_cols].ffill().bfill()
    price_cols = [c for c in df.columns if c in ("SPX", "VIX", "TLT")]
    df = df[df[price_cols].notna().any(axis=1)]
    return df

def _compute_raw_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute raw (non-standardized) features:
    - 1-day and 5-day log returns of SPX.
    - 20-day rolling volatility of SPX daily returns.
    - VIX level.
    - Yield curve slope: US10Y - US2Y.
    - 20-day momentum of TLT.
    """
    df = panel.copy()

    spx = df["SPX"]
    vix = df["VIX"]
    tlt = df["TLT"]
    us10 = df["US10Y"]
    us2 = df["US2Y"]

    spx_ret_1d = np.log(spx / spx.shift(1))
    spx_ret_5d = np.log(spx / spx.shift(5))
    spx_vol_20d = spx_ret_1d.rolling(20, min_periods=15).std()

    slope_10y2y = us10 - us2
    tlt_ma20 = tlt.rolling(20, min_periods=15).mean()
    tlt_mom_20 = (tlt / tlt_ma20) - 1.0

    feats = pd.DataFrame(
        {
            "spx_ret_1d": spx_ret_1d,
            "spx_ret_5d": spx_ret_5d,
            "spx_vol_20d": spx_vol_20d,
            "vix_level": vix,
            "slope_10y2y": slope_10y2y,
            "tlt_mom_20": tlt_mom_20,
        },
        index=df.index,
    )
    return feats

def _standardize(feats: pd.DataFrame, mode: StandardizeMode, rolling_window: int = 252) -> pd.DataFrame:
    """
    Apply standardization to features.

    Parameters
    ----------
    feats : pd.DataFrame
        Raw feature DataFrame.
    mode : {"rolling", "global", "none"}
        Standardization method.
    rolling_window : int
        Rolling window size if mode is "rolling".

    Returns
    -------
    pd.DataFrame
        Standardized DataFrame.
    """
    cols = feats.columns.tolist()
    if mode == "none":
        return feats
    if mode == "global":
        return _zscore_global(feats, cols)
    if mode == "rolling":
        return _zscore_rolling(feats, cols, window=rolling_window, minp=max(rolling_window // 2, 60))
    raise ValueError(f"Unknown standardization mode: {mode}")

def build_features(
    panel_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
    *,
    standardize: StandardizeMode = "rolling",
    rolling_window: int = 252,
) -> Path:
    """
    Build and save standardized features from the market panel.

    Parameters
    ----------
    panel_path : Path, optional
        Path to the market panel parquet file.
    out_path : Path, optional
        Output path for the features parquet file.
    standardize : {"rolling", "global", "none"}, default "rolling"
        Standardization mode.
    rolling_window : int, default 252
        Rolling window for rolling z-score.

    Returns
    -------
    Path
        Path to the saved features parquet file.
    """
    panel_path = panel_path or (PROC_DIR / "market_panel.parquet")
    out_path = out_path or (PROC_DIR / "features.parquet")

    panel = pd.read_parquet(panel_path)
    panel = _prepare_panel(panel)

    feats = _compute_raw_features(panel)
    feats = feats.dropna(how="any")
    feats = _standardize(feats, mode=standardize, rolling_window=rolling_window)
    feats = feats.dropna(how="any")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out_path)
    return out_path

def main() -> None:
    """
    CLI entry point to build features with default settings.
    """
    out = build_features()
    print(f"Features saved to: {out}")

if __name__ == "__main__":
    main()
