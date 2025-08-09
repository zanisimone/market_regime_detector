from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import numpy as np
import pandas as pd

@dataclass
class RegimeReport:
    """
    Container for regime performance outputs.

    Attributes
    ----------
    per_regime : pd.DataFrame
        Aggregated daily performance statistics by regime.
    segments : pd.DataFrame
        Regime segments with start/end, duration, and segment return.
    overall : pd.DataFrame
        Overall daily performance statistics on the aligned period.
    """
    per_regime: pd.DataFrame
    segments: pd.DataFrame
    overall: pd.DataFrame

def _max_drawdown(equity: pd.Series) -> float:
    """
    Compute maximum drawdown on an equity curve series.

    Parameters
    ----------
    equity : pd.Series
        Equity curve indexed by date.

    Returns
    -------
    float
        Maximum drawdown as a negative fraction.
    """
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min()) if len(dd) else np.nan

def _segments_from_labels(labeled: pd.DataFrame, label_col: str, name_col: Optional[str]) -> pd.DataFrame:
    """
    Convert a labeled daily series into contiguous regime segments.

    Parameters
    ----------
    labeled : pd.DataFrame
        DataFrame indexed by date with at least the label column.
    label_col : str
        Column name of the regime labels.
    name_col : str or None
        Optional column name of the human-readable regime name.

    Returns
    -------
    pd.DataFrame
        DataFrame of segments with columns: start, end, regime, regime_name (if available).
    """
    df = labeled.copy()
    df["__prev"] = df[label_col].shift(1)
    change_idx = df[df[label_col] != df["__prev"]].index
    if len(change_idx) == 0:
        return pd.DataFrame(columns=["start", "end", "regime", "regime_name"]).set_index("start")
    starts = change_idx
    ends = list(starts[1:]) + [df.index[-1]]
    rows = []
    for s, e in zip(starts, ends):
        seg = df.loc[s:e]
        reg = int(seg[label_col].iloc[0])
        nm = seg[name_col].iloc[0] if name_col and name_col in seg.columns else None
        rows.append({"start": s, "end": seg.index[-1], "regime": reg, "regime_name": nm})
    segdf = pd.DataFrame(rows).set_index("start")
    return segdf

def regime_report(
    panel: pd.DataFrame,
    labeled: pd.DataFrame,
    *,
    price_col: str = "SPX",
    label_col: str = "regime",
    name_col: str = "regime_name",
) -> RegimeReport:
    """
    Build per-regime, per-segment, and overall performance statistics.

    Parameters
    ----------
    panel : pd.DataFrame
        Market panel with a price column.
    labeled : pd.DataFrame
        DataFrame indexed by date with regime labels and optional regime names.
    price_col : str, default "SPX"
        Name of the price column in the panel.
    label_col : str, default "regime"
        Name of the regime label column in the labeled DataFrame.
    name_col : str, default "regime_name"
        Name of the human-readable regime name column in the labeled DataFrame.

    Returns
    -------
    RegimeReport
        Structured report with per-regime, per-segment, and overall statistics.
    """
    df = panel[[price_col]].join(labeled[[c for c in [label_col, name_col] if c in labeled.columns]], how="inner").dropna()
    px = df[price_col].astype(float)
    ret_d = px.pct_change().dropna()
    df = df.loc[ret_d.index]

    seg_idx = _segments_from_labels(df[[c for c in [label_col, name_col] if c in df.columns]], label_col, name_col)
    seg_rows = []
    for s, row in seg_idx.iterrows():
        e = row["end"]
        reg = row["regime"]
        nm = row.get("regime_name", None)
        seg = df.loc[s:e]
        start_px = float(seg[price_col].iloc[0])
        end_px = float(seg[price_col].iloc[-1])
        seg_rows.append({
            "start_date": s.date(),
            "end_date": e.date(),
            "regime": int(reg),
            "regime_name": nm if nm is not None else f"Regime {reg}",
            "start_price": start_px,
            "end_price": end_px,
            "ret": float(end_px / start_px - 1.0),
            "duration_days": int(len(seg)),
        })
    segments = pd.DataFrame(seg_rows)

    stats = []
    for reg, g in df.groupby(label_col):
        nm = g[name_col].iloc[0] if name_col in g.columns else f"Regime {reg}"
        r = g[price_col].pct_change().dropna()
        avg = float(r.mean())
        med = float(r.median())
        vol = float(r.std(ddof=0))
        shp = float(avg / vol) if vol and not np.isnan(vol) else np.nan
        eq = (1.0 + r).cumprod()
        mdd = _max_drawdown(eq)
        n_days = int(len(r))
        n_segments = int((segments["regime"] == reg).sum())
        stats.append({
            "regime": int(reg),
            "regime_name": nm,
            "mean_ret_d": avg,
            "median_ret_d": med,
            "vol_d": vol,
            "sharpe_d": shp,
            "max_drawdown": mdd,
            "days": n_days,
            "segments": n_segments,
        })
    per_regime = pd.DataFrame(stats).sort_values("regime").reset_index(drop=True)

    avg_all = float(ret_d.mean())
    vol_all = float(ret_d.std(ddof=0))
    shp_all = float(avg_all / vol_all) if vol_all and not np.isnan(vol_all) else np.nan
    overall = pd.DataFrame([{
        "mean_ret_d": avg_all,
        "vol_d": vol_all,
        "sharpe_d": shp_all,
        "max_drawdown": _max_drawdown((1.0 + ret_d).cumprod()),
        "days": int(len(ret_d)),
    }])

    return RegimeReport(per_regime=per_regime, segments=segments, overall=overall)
