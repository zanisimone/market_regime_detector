import pandas as pd
import numpy as np
from typing import Tuple, Optional


def price_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert price levels to simple returns with forward-fill for missing values.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide price table indexed by date with asset columns.

    Returns
    -------
    pd.DataFrame
        Daily simple returns with NaNs filled to 0 on the first valid value per column.
    """
    px = prices.sort_index().ffill()
    rets = px.pct_change()
    for c in rets.columns:
        if rets[c].first_valid_index() is not None:
            rets.loc[rets[c].first_valid_index(), c] = 0.0
    return rets.fillna(0.0)


def vectorized_portfolio_returns(returns: pd.DataFrame,
                                 weights: pd.DataFrame,
                                 rebalance_on_change: bool = True) -> Tuple[pd.Series, pd.Series]:
    """
    Compute portfolio returns and turnover given asset returns and target weights.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset return matrix aligned by index with columns matching weights.
    weights : pd.DataFrame
        Target weights per date; must have same columns as returns.
    rebalance_on_change : bool
        If True, apply weights whenever they change day-over-day; else rebalance daily.

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        port_ret: portfolio daily returns,
        turnover: daily turnover = 0.5 * sum(|w_t - w_{t-1}|).
    """
    R = returns.reindex(weights.index).fillna(0.0)
    W = weights.copy()
    if rebalance_on_change:
        changed = (W != W.shift(1)).any(axis=1)
        W = W.where(changed, np.nan).ffill().fillna(0.0)
    port_ret = (W.shift(1).fillna(0.0) * R).sum(axis=1)
    turnover = 0.5 * (W.fillna(0.0).diff().abs().sum(axis=1).fillna(0.0))
    return port_ret, turnover


def buy_and_hold_benchmark(prices: pd.DataFrame, asset: str) -> pd.Series:
    """
    Compute a buy-and-hold benchmark return series for a single asset.

    Parameters
    ----------
    prices : pd.DataFrame
        Price levels with at least the specified asset column.
    asset : str
        Column to use as the buy-and-hold benchmark.

    Returns
    -------
    pd.Series
        Daily returns of buy-and-hold on the asset.
    """
    return price_to_returns(prices[[asset]])[asset]
