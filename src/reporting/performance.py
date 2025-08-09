import numpy as np
import pandas as pd
from typing import Dict, Optional


def max_drawdown(cum_returns: pd.Series) -> float:
    """
    Compute maximum drawdown from a cumulative return index.

    Parameters
    ----------
    cum_returns : pd.Series
        Cumulative wealth path starting at 1.0.

    Returns
    -------
    float
        Maximum drawdown as a negative number in [-1, 0].
    """
    peak = cum_returns.cummax()
    dd = cum_returns / peak - 1.0
    return float(dd.min())


def summarize_performance(returns: pd.Series,
                          turnover: Optional[pd.Series] = None,
                          freq: int = 252,
                          rf: float = 0.0) -> Dict[str, float]:
    """
    Compute global performance metrics.

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns.
    turnover : Optional[pd.Series]
        Daily turnover series; if None, turnover is reported as NaN.
    freq : int
        Periods per year for annualization.
    rf : float
        Daily risk-free rate; set 0 by default.

    Returns
    -------
    Dict[str, float]
        Metrics: CAGR, volatility, Sharpe, max_drawdown, avg_turnover.
    """
    ret = returns.fillna(0.0)
    wealth = (1.0 + ret).cumprod()
    n_years = max((len(ret) / max(freq, 1)), 1e-9)
    cagr = float(wealth.iloc[-1] ** (1.0 / n_years) - 1.0) if len(wealth) > 0 else np.nan
    vol = float(ret.std(ddof=1) * np.sqrt(freq)) if len(ret) > 1 else np.nan
    sharpe = float(((ret - rf).mean() / (ret - rf).std(ddof=1)) * np.sqrt(freq)) if ret.std(ddof=1) > 0 else np.nan
    mdd = max_drawdown(wealth)
    avg_tvr = float(turnover.mean()) if turnover is not None and len(turnover) > 0 else np.nan
    return {"CAGR": cagr, "vol": vol, "Sharpe": sharpe, "MDD": mdd, "turnover": avg_tvr}


def hit_rate_by_regime(returns: pd.Series, labels: pd.Series) -> pd.DataFrame:
    """
    Compute hit-rate (fraction of positive days) and mean return grouped by regime.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns indexed by date.
    labels : pd.Series
        Integer regimes indexed by date.

    Returns
    -------
    pd.DataFrame
        Columns: hit_rate, mean_ret, n_obs indexed by regime id.
    """
    idx = returns.index.intersection(labels.index)
    r = returns.loc[idx]
    y = labels.loc[idx].astype(int)
    grp = pd.DataFrame({"r": r, "y": y}).groupby("y")["r"]
    out = pd.DataFrame({
        "hit_rate": grp.apply(lambda s: float((s > 0).mean())),
        "mean_ret": grp.mean(),
        "n_obs": grp.size().astype(int),
    })
    return out.sort_index()


def regime_performance(returns: pd.Series, labels: pd.Series, freq: int = 252) -> pd.DataFrame:
    """
    Compute per-regime annualized mean, volatility, and pseudo-Sharpe.

    Parameters
    ----------
    returns : pd.Series
        Portfolio daily returns.
    labels : pd.Series
        Regime labels.
    freq : int
        Periods per year.

    Returns
    -------
    pd.DataFrame
        Columns: ann_mean, ann_vol, ann_sharpe by regime id.
    """
    idx = returns.index.intersection(labels.index)
    r = returns.loc[idx]
    y = labels.loc[idx].astype(int)
    def agg(s):
        mu = s.mean() * freq
        sd = s.std(ddof=1) * np.sqrt(freq)
        sh = mu / sd if sd > 0 else np.nan
        return pd.Series({"ann_mean": float(mu), "ann_vol": float(sd), "ann_sharpe": float(sh)})
    return r.groupby(y).apply(agg).sort_index()
