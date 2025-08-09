from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence, Literal, Iterable, Tuple, List, Dict
import argparse
import logging

import numpy as np
import pandas as pd

from src.config import PROC_DIR

StandardizeMode = Literal["global", "rolling", "none"]

logger = logging.getLogger(__name__)


def _zscore_global(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """
    Apply global z-score standardization on selected columns.
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
    """
    out = df.copy()
    for c in cols:
        roll_mean = out[c].rolling(window, min_periods=minp).mean()
        roll_std = out[c].rolling(window, min_periods=minp).std(ddof=0)
        out[c] = (out[c] - roll_mean) / roll_std
    return out


def _rolling_corr(x: pd.Series, y: pd.Series, window: int, minp: int) -> pd.Series:
    """
    Compute rolling Pearson correlation between two series.
    """
    return x.rolling(window, min_periods=minp).corr(y)


def _rolling_beta(y: pd.Series, x: pd.Series, window: int, minp: int) -> pd.Series:
    """
    Compute rolling beta of y on x as Cov(y,x)/Var(x).
    """
    cov = y.rolling(window, min_periods=minp).cov(x)
    var = x.rolling(window, min_periods=minp).var()
    beta = cov / var
    return beta.replace([np.inf, -np.inf], np.nan)


def _realized_vol_pct(ret: pd.Series, window: int, minp: int) -> pd.Series:
    """
    Compute annualized realized volatility in percent using daily log returns.
    """
    return ret.rolling(window, min_periods=minp).std(ddof=0) * np.sqrt(252.0) * 100.0


def _momentum_pct_ma(price: pd.Series, lookback: int, minp: int) -> pd.Series:
    """
    Momentum as price-to-moving-average minus 1 (dimensionless).
    """
    ma = price.rolling(lookback, min_periods=minp).mean()
    return (price / ma) - 1.0


def _trailing_log_return(price: pd.Series, lookback: int) -> pd.Series:
    """
    Trailing log return over lookback sessions.
    """
    return np.log(price / price.shift(lookback))


def _rolling_drawdown(price: pd.Series, window: int, minp: int) -> pd.Series:
    """
    Rolling drawdown as (price/rolling_max - 1) over a window.
    """
    rmax = price.rolling(window, min_periods=minp).max()
    return (price / rmax) - 1.0


def _first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    """
    Return the first column name that exists in the DataFrame.
    """
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the candidate columns exist: {list(candidates)}")


def _prepare_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the market panel for feature computation.
    """
    df = df.asfreq("B")
    macro_cols = [c for c in df.columns if c.upper().startswith(("US", "DGS"))]
    if macro_cols:
        df[macro_cols] = df[macro_cols].ffill().bfill()
    price_cols = [c for c in ("SPX", "VIX", "TLT") if c in df.columns]
    if price_cols:
        df = df[df[price_cols].notna().any(axis=1)]
    return df


def _compute_raw_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute market-only raw features.
    """
    df = panel.copy()

    spx = df[_first_existing(df, ("SPX",))]
    tlt = df[_first_existing(df, ("TLT",))]
    vix = df[_first_existing(df, ("VIX",))]
    c10 = df[_first_existing(df, ("US10Y", "DGS10"))] if any(c in df.columns for c in ("US10Y", "DGS10")) else pd.Series(index=df.index, dtype=float)
    c2 = df[_first_existing(df, ("US2Y", "DGS2"))] if any(c in df.columns for c in ("US2Y", "DGS2")) else pd.Series(index=df.index, dtype=float)

    spx_ret_1d = np.log(spx / spx.shift(1))
    spx_ret_5d = _trailing_log_return(spx, 5)
    tlt_ret_1d = np.log(tlt / tlt.shift(1))
    tlt_ret_5d = _trailing_log_return(tlt, 5)

    spx_mom_20 = _momentum_pct_ma(spx, 20, 15)
    tlt_mom_20 = _momentum_pct_ma(tlt, 20, 15)

    spx_vol_21d = _realized_vol_pct(spx_ret_1d, 21, 15)
    spx_vol_63d = _realized_vol_pct(spx_ret_1d, 63, 40)

    vix_to_realized_21d = vix / spx_vol_21d

    spx_ddown_63d = _rolling_drawdown(spx, 63, 40)

    corr_spx_tlt_63d = _rolling_corr(spx_ret_1d, tlt_ret_1d, 63, 40)
    beta_spx_on_tlt_252d = _rolling_beta(spx_ret_1d, tlt_ret_1d, 252, 120)

    spx_mom_63 = _momentum_pct_ma(spx, 63, 40)
    tlt_mom_63 = _momentum_pct_ma(tlt, 63, 40)
    mom_diff_spx_tlt_63d = spx_mom_63 - tlt_mom_63

    slope_candidates_present = any(c in df.columns for c in ("US10Y", "DGS10")) and any(c in df.columns for c in ("US2Y", "DGS2"))
    curve_slope = (c10 - c2) if slope_candidates_present else pd.Series(index=df.index, dtype=float)

    feats = pd.DataFrame(
        {
            "spx_ret_1d": spx_ret_1d,
            "spx_ret_5d": spx_ret_5d,
            "spx_mom_20d": spx_mom_20,
            "tlt_ret_1d": tlt_ret_1d,
            "tlt_ret_5d": tlt_ret_5d,
            "tlt_mom_20d": tlt_mom_20,
            "spx_vol_21d": spx_vol_21d,
            "spx_vol_63d": spx_vol_63d,
            "vix_to_realized_21d": vix_to_realized_21d,
            "ddown_63d": spx_ddown_63d,
            "corr_spx_tlt_63d": corr_spx_tlt_63d,
            "beta_spx_on_tlt_252d": beta_spx_on_tlt_252d,
            "mom_diff_spx_tlt_63d": mom_diff_spx_tlt_63d,
            "curve_slope": curve_slope,
            "vix_level": vix,
        },
        index=df.index,
    )
    return feats


def _compute_etf_proxy_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute optional ETF proxy features; missing tickers yield NaN with warnings.
    """
    df = panel.copy()
    out = pd.DataFrame(index=df.index)

    if "HYG" in df.columns and "LQD" in df.columns:
        out["credit_spread_ret_21d"] = _trailing_log_return(df["HYG"], 21) - _trailing_log_return(df["LQD"], 21)
    else:
        out["credit_spread_ret_21d"] = np.nan
        missing = [t for t in ("HYG", "LQD") if t not in df.columns]
        logger.warning(f"Missing proxy: {missing} -> credit_spread_ret_21d=NaN")

    for tkr, colname in (("UUP", "uup_mom_63d"), ("GLD", "gld_mom_63d"), ("USO", "uso_mom_63d")):
        if tkr in df.columns:
            out[colname] = _momentum_pct_ma(df[tkr], 63, 40)
        else:
            out[colname] = np.nan
            logger.warning(f"Missing proxy: {tkr} -> {colname}=NaN")

    if "TIP" in df.columns and "IEF" in df.columns:
        out["infl_breakeven_proxy_63d"] = _trailing_log_return(df["TIP"], 63) - _trailing_log_return(df["IEF"], 63)
    else:
        out["infl_breakeven_proxy_63d"] = np.nan
        missing = [t for t in ("TIP", "IEF") if t not in df.columns]
        logger.warning(f"Missing proxy: {missing} -> infl_breakeven_proxy_63d=NaN")

    return out


def _append_selected_rolling_zscores(feats: pd.DataFrame, window: int = 252, minp: int = 120, suffix: str = "_z") -> pd.DataFrame:
    """
    Compute and append rolling z-scores for a curated subset of features.
    """
    out = feats.copy()
    selected = [
        "spx_ret_1d",
        "spx_ret_5d",
        "spx_mom_20d",
        "tlt_mom_20d",
        "spx_vol_21d",
        "spx_vol_63d",
        "vix_to_realized_21d",
        "ddown_63d",
        "corr_spx_tlt_63d",
        "beta_spx_on_tlt_252d",
        "mom_diff_spx_tlt_63d",
        "curve_slope",
    ]
    existing = [c for c in selected if c in out.columns]
    if existing:
        zdf = _zscore_rolling(out[existing], existing, window=window, minp=minp)
        zdf = zdf.add_suffix(suffix)
        out = pd.concat([out, zdf], axis=1)
    return out


def _optional_proxy_cols() -> list[str]:
    """
    Names of optional proxy columns that must NOT trigger row drops when NaN.
    """
    return [
        "credit_spread_ret_21d",
        "uup_mom_63d",
        "gld_mom_63d",
        "uso_mom_63d",
        "infl_breakeven_proxy_63d",
    ]


def apply_winsor_rolling(s: pd.Series, win: int = 252, q: Tuple[float, float] = (0.01, 0.99), minp: int = 120) -> pd.Series:
    """
    Apply rolling winsorization by clipping values to rolling quantiles.
    """
    q_low = s.rolling(win, min_periods=minp).quantile(q[0])
    q_high = s.rolling(win, min_periods=minp).quantile(q[1])
    return s.where(s >= q_low, q_low).where(s <= q_high, q_high)


def _feature_bundles() -> Dict[str, List[str]]:
    """
    Define feature bundles: 'core', 'market_plus', 'macro_plus'.
    """
    core = [
        "spx_ret_1d",
        "spx_ret_5d",
        "spx_mom_20d",
        "tlt_ret_1d",
        "tlt_ret_5d",
        "tlt_mom_20d",
        "spx_vol_21d",
        "spx_vol_63d",
        "vix_to_realized_21d",
        "ddown_63d",
        "corr_spx_tlt_63d",
        "beta_spx_on_tlt_252d",
        "mom_diff_spx_tlt_63d",
        "curve_slope",
        "vix_level",
        "spx_ret_1d_z",
        "spx_ret_5d_z",
        "spx_mom_20d_z",
        "tlt_mom_20d_z",
        "spx_vol_21d_z",
        "spx_vol_63d_z",
        "vix_to_realized_21d_z",
        "ddown_63d_z",
        "corr_spx_tlt_63d_z",
        "beta_spx_on_tlt_252d_z",
        "mom_diff_spx_tlt_63d_z",
        "curve_slope_z",
    ]
    market_plus = core + [
        "credit_spread_ret_21d",
        "uup_mom_63d",
        "gld_mom_63d",
        "uso_mom_63d",
        "infl_breakeven_proxy_63d",
    ]
    macro_plus = market_plus
    return {"core": core, "market_plus": market_plus, "macro_plus": macro_plus}


def _select_by_bundle(feats: pd.DataFrame, bundle: str) -> pd.DataFrame:
    """
    Select a subset of features according to the requested bundle.
    """
    bundles = _feature_bundles()
    if bundle not in bundles:
        raise ValueError(f"Unknown bundle '{bundle}'. Available: {list(bundles)}")
    cols = [c for c in bundles[bundle] if c in feats.columns]
    return feats[cols]


def _export_used_features(feats: pd.DataFrame, out_dir: Path) -> Path:
    """
    Export list of actually used feature names to a text file for traceability.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "features_used.txt"
    with p.open("w", encoding="utf-8") as f:
        for c in feats.columns:
            f.write(f"{c}\n")
    return p


def _rolling_pca_matrix(X: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute PCA via SVD on a matrix with rows as observations and columns as features.
    """
    X = X - np.nanmean(X, axis=0, keepdims=True)
    std = np.nanstd(X, axis=0, keepdims=True)
    std[std == 0] = 1.0
    Xn = (X - 0.0) / std
    Xn = np.nan_to_num(Xn, nan=0.0)
    U, S, Vt = np.linalg.svd(Xn, full_matrices=False)
    comps = Vt[:n_components]
    scores = Xn @ comps.T
    return scores, comps


def rolling_pca(df: pd.DataFrame, window: int = 252, minp: int = 120, n_components: int = 2) -> pd.DataFrame:
    """
    Compute rolling PCA on columns of df, returning PC1..PCn time series.
    """
    pcs = [pd.Series(index=df.index, dtype=float) for _ in range(n_components)]
    values = df.values
    for i in range(len(df)):
        start = max(0, i - window + 1)
        Xw = values[start : i + 1, :]
        if Xw.shape[0] < minp:
            continue
        if np.isnan(Xw).any():
            continue
        sc, _ = _rolling_pca_matrix(Xw, n_components=n_components)
        for k in range(n_components):
            pcs[k].iloc[i] = sc[-1, k]
    out = pd.DataFrame({f"pca{k+1}": pcs[k] for k in range(n_components)}, index=df.index)
    return out

def _standardize(df: pd.DataFrame, mode: StandardizeMode = "rolling", rolling_window: int = 252) -> pd.DataFrame:
    """
    Standardize features according to the selected mode:
    - 'global': z-score over the whole sample
    - 'rolling': rolling z-score to avoid look-ahead bias
    - 'none': no standardization
    """
    cols = list(df.columns)
    if mode == "global":
        return _zscore_global(df, cols)
    elif mode == "rolling":
        return _zscore_rolling(df, cols, window=rolling_window, minp=max(rolling_window // 2, 60))
    elif mode == "none":
        return df
    else:
        raise ValueError(f"Unknown standardization mode: {mode}")



def build_features(
    panel_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
    *,
    standardize: StandardizeMode = "rolling",
    rolling_window: int = 252,
    enable_winsor: bool = True,
    winsor_window: int = 252,
    winsor_q_low: float = 0.01,
    winsor_q_high: float = 0.99,
    bundle: str = "market_plus",
    enable_pca: bool = False,
    pca_window: int = 252,
    pca_replace: bool = False,
) -> Path:
    """
    Build and save features with optional rolling winsorization, bundles, and rolling PCA.
    """
    panel_path = panel_path or (PROC_DIR / "market_panel.parquet")
    out_path = out_path or (PROC_DIR / "features.parquet")

    panel = pd.read_parquet(panel_path)
    panel = _prepare_panel(panel)

    core = _compute_raw_features(panel).dropna(how="any")
    proxy = _compute_etf_proxy_features(panel)
    feats = pd.concat([core, proxy], axis=1)

    feats = _append_selected_rolling_zscores(
        feats, window=rolling_window, minp=max(rolling_window // 2, 60), suffix="_z"
    )

    if enable_winsor:
        targets = [c for c in ["spx_ret_1d", "spx_vol_21d", "curve_slope"] if c in feats.columns]
        for c in targets:
            feats[c] = apply_winsor_rolling(
                feats[c], win=winsor_window, q=(winsor_q_low, winsor_q_high), minp=max(winsor_window // 2, 60)
            )

    feats = _standardize(feats, mode=standardize, rolling_window=rolling_window)

    required_cols = [c for c in feats.columns if c not in _optional_proxy_cols()]
    feats = feats.dropna(subset=required_cols, how="any")

    feats = _select_by_bundle(feats, bundle=bundle)
    _export_used_features(feats, out_dir=PROC_DIR)

    if enable_pca and feats.shape[1] >= 2:
        pca_df = rolling_pca(feats.dropna(how="any", axis=1), window=pca_window, minp=max(pca_window // 2, 60), n_components=2)
        if pca_replace:
            feats = pca_df
        else:
            feats = pd.concat([feats, pca_df], axis=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out_path)
    return out_path


def _importable_main(args: Optional[argparse.Namespace] = None) -> Path:
    """
    Internal entry to support scripts and direct module execution.
    """
    if args is None:
        ap = argparse.ArgumentParser(description="Build features with robust transformations.")
        ap.add_argument("--panel", type=str, default=str(PROC_DIR / "market_panel.parquet"))
        ap.add_argument("--out", type=str, default=str(PROC_DIR / "features.parquet"))
        ap.add_argument("--standardize", type=str, choices=["rolling", "global", "none"], default="rolling")
        ap.add_argument("--window", type=int, default=252)
        ap.add_argument("--winsor", action="store_true", default=True)
        ap.add_argument("--no-winsor", dest="winsor", action="store_false")
        ap.add_argument("--winsor-window", type=int, default=252)
        ap.add_argument("--winsor-q", type=float, nargs=2, default=(0.01, 0.99))
        ap.add_argument("--bundle", type=str, choices=["core", "market_plus", "macro_plus"], default="market_plus")
        ap.add_argument("--pca", action="store_true", default=False)
        ap.add_argument("--pca-window", type=int, default=252)
        ap.add_argument("--pca-replace", action="store_true", default=False)
        args = ap.parse_args()

    out = build_features(
        panel_path=Path(args.panel),
        out_path=Path(args.out),
        standardize=args.standardize,  # type: ignore[arg-type]
        rolling_window=int(args.window),
        enable_winsor=bool(args.winsor),
        winsor_window=int(args.winsor_window),
        winsor_q_low=float(args.winsor_q[0]) if isinstance(args.winsor_q, (list, tuple)) else 0.01,
        winsor_q_high=float(args.winsor_q[1]) if isinstance(args.winsor_q, (list, tuple)) else 0.99,
        bundle=str(args.bundle),
        enable_pca=bool(args.pca),
        pca_window=int(args.pca_window),
        pca_replace=bool(args.pca_replace),
    )
    return out


def main() -> None:
    """
    CLI entry point for direct module execution.
    """
    out = _importable_main()
    print(f"[OK] Features saved to: {out}")


if __name__ == "__main__":
    main()
