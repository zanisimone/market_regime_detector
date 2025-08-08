# src/features/build_features.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.config import PROC_DIR


def _zscore(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """
    Apply z-score standardization column-wise to the selected columns.
    """
    out = df.copy()
    for c in cols:
        mu = out[c].mean()
        sd = out[c].std(ddof=0)
        out[c] = (out[c] - mu) / sd if sd and not np.isnan(sd) else out[c] * 0.0
    return out


def _prepare_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the input panel to a business-day frequency, forward-fill macro series,
    and filter out rows without any price information.
    """
    df = df.asfreq("B")
    macro_cols = [c for c in df.columns if c.upper().startswith("US")]
    if macro_cols:
        df[macro_cols] = df[macro_cols].ffill().bfill()
    price_cols = [c for c in df.columns if c in ("SPX", "VIX", "TLT")]
    df = df[df[price_cols].notna().any(axis=1)]
    return df


def _compute_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a minimal feature set for market regime detection:
    - spx_ret_1d, spx_ret_5d: log returns on SPX
    - spx_vol_20d: rolling 20-day std of daily SPX log returns
    - vix_level: VIX level
    - slope_10y2y: US10Y - US2Y
    - tlt_mom_20: momentum proxy via 20-day mean
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

    vix_level = vix

    slope_10y2y = us10 - us2

    tlt_ma20 = tlt.rolling(20, min_periods=15).mean()
    tlt_mom_20 = (tlt / tlt_ma20) - 1.0

    feats = pd.DataFrame(
        {
            "spx_ret_1d": spx_ret_1d,
            "spx_ret_5d": spx_ret_5d,
            "spx_vol_20d": spx_vol_20d,
            "vix_level": vix_level,
            "slope_10y2y": slope_10y2y,
            "tlt_mom_20": tlt_mom_20,
        },
        index=df.index,
    )

    feats = feats.dropna(how="any")
    feats = _zscore(feats, feats.columns.tolist())
    return feats


def build_features(
    panel_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
) -> Path:
    """
    Build and save standardized features from the processed market panel.

    Parameters
    ----------
    panel_path : Path | None
        Path to the processed panel parquet. Defaults to data/processed/market_panel.parquet.
    out_path : Path | None
        Path to save the features parquet. Defaults to data/processed/features.parquet.

    Returns
    -------
    Path
        Path to the saved features parquet.
    """
    panel_path = panel_path or (PROC_DIR / "market_panel.parquet")
    out_path = out_path or (PROC_DIR / "features.parquet")

    panel = pd.read_parquet(panel_path)
    panel = _prepare_panel(panel)
    feats = _compute_features(panel)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out_path)
    return out_path


def main() -> None:
    """
    Entry point to build features using default paths from config.
    """
    out = build_features()
    print(f"Features saved to: {out}")


if __name__ == "__main__":
    main()
