import numpy as np
import pandas as pd
from typing import Optional, Dict

def rolling_prob_metrics(prob_risk_off: pd.Series, ma_window: int = 5, z_window: int = 63) -> pd.DataFrame:
    """
    Compute rolling metrics on Risk-Off probability to support early warning.

    Parameters
    ----------
    prob_risk_off : pd.Series
        Daily probability or proxy in [0,1] that the regime is Risk-Off.
        For hard labels, use a binary series (1 if Risk-Off, else 0).
    ma_window : int
        Rolling window for short-term average.
    z_window : int
        Rolling window for z-score normalization.

    Returns
    -------
    pd.DataFrame
        Columns: prob, ma, slope, zscore.
    """
    s = prob_risk_off.astype(float).clip(0.0, 1.0)
    ma = s.rolling(ma_window, min_periods=1).mean()
    slope = ma.diff()
    mu = s.rolling(z_window, min_periods=10).mean()
    sd = s.rolling(z_window, min_periods=10).std(ddof=1)
    z = (s - mu) / (sd.replace(0.0, np.nan))
    out = pd.DataFrame({"prob": s, "ma": ma, "slope": slope, "zscore": z}, index=s.index)
    return out


def early_warning_signal(prob_risk_off: pd.Series,
                         ma_window: int = 5,
                         z_window: int = 63,
                         prob_threshold: float = 0.60,
                         z_threshold: float = 1.0,
                         slope_threshold: float = 0.02,
                         cool_off_days: int = 5) -> pd.DataFrame:
    """
    Generate early-warning heuristics for rising Risk-Off probability using multiple triggers.

    Parameters
    ----------
    prob_risk_off : pd.Series
        Daily Risk-Off probability in [0,1] or binary indicator.
    ma_window : int
        Window for short-term moving average.
    z_window : int
        Window for z-score baseline.
    prob_threshold : float
        Trigger when MA exceeds this level.
    z_threshold : float
        Trigger when z-score exceeds this level.
    slope_threshold : float
        Trigger when short-term MA slope exceeds this level.
    cool_off_days : int
        Minimum spacing between active warning signals.

    Returns
    -------
    pd.DataFrame
        Columns: prob, ma, slope, zscore, trigger_prob, trigger_z, trigger_slope, warning.
    """
    df = rolling_prob_metrics(prob_risk_off, ma_window=ma_window, z_window=z_window)
    df["trigger_prob"] = (df["ma"] >= prob_threshold).astype(int)
    df["trigger_z"] = (df["zscore"] >= z_threshold).astype(int)
    df["trigger_slope"] = (df["slope"] >= slope_threshold).astype(int)
    raw = ((df["trigger_prob"] + df["trigger_z"] + df["trigger_slope"]) >= 2).astype(int)
    warning = raw.copy()
    last_idx = None
    for i, val in enumerate(raw.values):
        if val == 1:
            if last_idx is None or (i - last_idx) > cool_off_days:
                warning.iloc[i] = 1
                last_idx = i
            else:
                warning.iloc[i] = 0
        else:
            warning.iloc[i] = 0
    df["warning"] = warning
    return df
