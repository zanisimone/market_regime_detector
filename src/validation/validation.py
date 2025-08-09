from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _adf_pvalue_opt(series: pd.Series) -> Optional[float]:
    """
    Return ADF p-value if statsmodels is available; otherwise return None.
    """
    try:
        from statsmodels.tsa.stattools import adfuller  # type: ignore
    except Exception:
        return None
    x = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 30:
        return None
    try:
        return float(adfuller(x, autolag="AIC")[1])
    except Exception:
        return None


def _basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute basic descriptive statistics and NaN share per column.
    """
    stats = pd.DataFrame(
        {
            "mean": df.mean(numeric_only=True),
            "std": df.std(ddof=0, numeric_only=True),
            "min": df.min(numeric_only=True),
            "p01": df.quantile(0.01, numeric_only=True),
            "p50": df.quantile(0.50, numeric_only=True),
            "p99": df.quantile(0.99, numeric_only=True),
            "max": df.max(numeric_only=True),
            "nan_pct": df.isna().mean() * 100.0,
        }
    )
    return stats


def _rule_bounds() -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """
    Provide soft bounds for sanity checks per pattern of feature names.
    """
    return {
        "corr_": (-1.0, 1.0),
        "beta_": (None, None),
        "vol_": (0.0, None),
        "spx_vol_": (0.0, None),
        "vix_to_realized_": (0.0, None),
        "ddown_": (-1.0, 0.0),
        "_z": (-8.0, 8.0),
    }


def _check_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check columns against soft name-based bounds; report violations count and share.
    """
    rules = _rule_bounds()
    rows = []
    for c in df.columns:
        low: Optional[float] = None
        high: Optional[float] = None
        for key, (lo, hi) in rules.items():
            if c.startswith(key) or c.endswith(key):
                low = lo if lo is not None else low
                high = hi if hi is not None else high
        if low is None and high is None:
            continue
        x = df[c].replace([np.inf, -np.inf], np.nan).dropna()
        if x.empty:
            rows.append({"feature": c, "violations": 0, "viol_pct": 0.0, "low": low, "high": high})
            continue
        mask_low = np.zeros_like(x, dtype=bool)
        mask_high = np.zeros_like(x, dtype=bool)
        if low is not None:
            mask_low = x < low
        if high is not None:
            mask_high = x > high
        viol = int(mask_low.sum() + mask_high.sum())
        rows.append(
            {
                "feature": c,
                "violations": viol,
                "viol_pct": 100.0 * viol / len(x),
                "low": low,
                "high": high,
            }
        )
    return pd.DataFrame(rows).sort_values(["viol_pct", "violations"], ascending=False)


def _split_train_oos(df: pd.DataFrame, split_date: Optional[str], split_ratio: Optional[float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into train and oos using either a calendar date or a ratio.
    """
    if split_date:
        train = df.loc[:pd.to_datetime(split_date)]
        oos = df.loc[pd.to_datetime(split_date) + pd.offsets.BDay(1):]
        return train, oos
    if split_ratio is not None:
        n = len(df)
        k = int(max(1, min(n - 1, round(n * float(split_ratio)))))
        return df.iloc[:k], df.iloc[k:]
    raise ValueError("Provide either split_date or split_ratio.")


def _segment_stats(train: pd.DataFrame, oos: pd.DataFrame) -> pd.DataFrame:
    """
    Compare mean and std between train and oos, returning absolute and relative deltas.
    """
    cols = [c for c in train.columns if c in oos.columns]
    mu_tr = train[cols].mean()
    sd_tr = train[cols].std(ddof=0)
    mu_oos = oos[cols].mean()
    sd_oos = oos[cols].std(ddof=0)
    out = pd.DataFrame(
        {
            "mean_train": mu_tr,
            "mean_oos": mu_oos,
            "mean_delta": mu_oos - mu_tr,
            "mean_rel_delta": (mu_oos - mu_tr) / (sd_tr.replace(0.0, np.nan)),
            "std_train": sd_tr,
            "std_oos": sd_oos,
            "std_delta": sd_oos - sd_tr,
            "std_rel_delta": (sd_oos - sd_tr) / sd_tr.replace(0.0, np.nan),
        }
    )
    return out


def _rolling_compare(train: pd.DataFrame, oos: pd.DataFrame, window: int = 126) -> pd.DataFrame:
    """
    Compute rolling means in OOS and compare last OOS mean to train mean in SD units.
    """
    cols = [c for c in train.columns if c in oos.columns]
    mu_tr = train[cols].mean()
    sd_tr = train[cols].std(ddof=0).replace(0.0, np.nan)
    mu_oos_roll = oos[cols].rolling(window, min_periods=max(20, window // 3)).mean().iloc[-1]
    z = (mu_oos_roll - mu_tr) / sd_tr
    return pd.DataFrame({"roll_mean_oos_last": mu_oos_roll, "z_vs_train": z})


def _corr_heatmap(df: pd.DataFrame, out_png: Path) -> None:
    """
    Save a correlation heatmap PNG using matplotlib only.
    """
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.index)
    ax.set_title("Feature Correlation Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def _histograms(df: pd.DataFrame, out_dir: Path, bins: int = 50, max_plots: int = 60) -> None:
    """
    Save per-feature histograms as PNGs; cap the count to avoid excessive files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in df.columns if np.issubdtype(df[c].dropna().dtype, np.number)]
    for i, c in enumerate(cols[:max_plots]):
        x = df[c].replace([np.inf, -np.inf], np.nan).dropna()
        if x.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(x.values, bins=bins)
        ax.set_title(f"Distribution: {c}")
        ax.set_xlabel(c)
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        fig.savefig(out_dir / f"hist_{i:03d}_{c}.png", dpi=120)
        plt.close(fig)


def validate_and_report(
    features_path: Path,
    *,
    out_dir: Path,
    split_date: Optional[str] = None,
    split_ratio: Optional[float] = 0.8,
    run_adf: bool = False,
    rolling_cmp_window: int = 126,
) -> Dict[str, Any]:
    """
    Load features parquet, run sanity suite, drift monitor, and export a mini-report.
    """
    df = pd.read_parquet(features_path)
    df = df.replace([np.inf, -np.inf], np.nan)

    basic = _basic_stats(df)
    ranges = _check_ranges(df)

    adf_map: Dict[str, Optional[float]] = {}
    if run_adf:
        for c in df.columns:
            adf_map[c] = _adf_pvalue_opt(df[c])

    train, oos = _split_train_oos(df, split_date=split_date, split_ratio=split_ratio)
    seg = _segment_stats(train, oos)
    roll = _rolling_compare(train, oos, window=rolling_cmp_window)

    out_dir.mkdir(parents=True, exist_ok=True)
    basic.to_csv(out_dir / "basic_stats.csv")
    ranges.to_csv(out_dir / "range_checks.csv", index=False)
    seg.to_csv(out_dir / "drift_segment.csv")
    roll.to_csv(out_dir / "drift_rolling.csv")

    if run_adf and len(adf_map) > 0:
        pd.Series(adf_map, name="adf_pvalue").to_csv(out_dir / "adf_pvalues.csv")

    _histograms(df, out_dir=out_dir / "dists")
    _corr_heatmap(df.dropna(axis=1, how="any"), out_png=out_dir / "corr_heatmap.png")

    summary = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "nan_pct_mean": float(df.isna().mean().mean() * 100.0),
        "nan_pct_max": float(df.isna().mean().max() * 100.0),
        "violations_total": int(ranges["violations"].sum()) if not ranges.empty else 0,
    }
    return {
        "summary": summary,
        "paths": {
            "basic_stats": str(out_dir / "basic_stats.csv"),
            "range_checks": str(out_dir / "range_checks.csv"),
            "drift_segment": str(out_dir / "drift_segment.csv"),
            "drift_rolling": str(out_dir / "drift_rolling.csv"),
            "adf_pvalues": str(out_dir / "adf_pvalues.csv") if run_adf else None,
            "corr_heatmap": str(out_dir / "corr_heatmap.png"),
            "hist_dir": str(out_dir / "dists"),
        },
    }
