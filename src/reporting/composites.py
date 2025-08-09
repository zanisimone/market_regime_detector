import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


def normalize_price(series: pd.Series, mode: str = "rebased") -> pd.Series:
    """
    Normalize a price series.

    Parameters
    ----------
    series : pd.Series
        Price-level series indexed by date/time.
    mode : str
        "rebased": divide by value at t0 and subtract 1.0 (i.e., relative return path).
        "logret": cumulative log-returns rebased to 0 at t0.

    Returns
    -------
    pd.Series
        Normalized series with same index.
    """
    s = series.astype(float)
    if len(s) == 0:
        return s
    if mode == "rebased":
        base = s.iloc[0]
        return s / base - 1.0
    elif mode == "logret":
        r = np.log(s).diff().fillna(0.0)
        return r.cumsum() - r.cumsum().iloc[0]
    else:
        raise ValueError("mode must be 'rebased' or 'logret'")


def regime_path_composites(price: pd.Series,
                           labels: pd.Series,
                           regime_id: int,
                           lookback: int = 5,
                           lookahead: int = 20,
                           mode: str = "rebased",
                           min_run: int = 3) -> pd.DataFrame:
    """
    Compute normalized price path composites around regime-start events.

    Parameters
    ----------
    price : pd.Series
        Price-level series.
    labels : pd.Series
        Integer regime series aligned to the same index as price.
    regime_id : int
        Regime to anchor on (event = first day of a new run of this regime).
    lookback : int
        Number of bars before event to include (negative offsets).
    lookahead : int
        Number of bars after event to include (positive offsets).
    mode : str
        Normalization mode ("rebased" or "logret").
    min_run : int
        Minimum length of the regime run to count the event.

    Returns
    -------
    pd.DataFrame
        Composite table with index as offset in [-lookback, ..., +lookahead] and
        columns: mean, median, p25, p75, n, plus optionally each sample path as wide columns.
    """
    idx = price.index.intersection(labels.index)
    p = price.loc[idx]
    y = labels.loc[idx].astype(int)

    starts = []
    run_len = 0
    for t in range(len(y)):
        if t == 0 or y.iloc[t] != y.iloc[t - 1]:
            if t > 0 and y.iloc[t - 1] == regime_id and run_len >= min_run:
                pass
            run_len = 1
            if y.iloc[t] == regime_id:
                starts.append(t)
        else:
            run_len += 1

    paths = []
    for t0 in starts:
        t_start = max(0, t0 - lookback)
        t_end = min(len(p) - 1, t0 + lookahead)
        window = p.iloc[t_start : t_end + 1]
        normed = normalize_price(window, mode=mode)
        if len(normed) < lookback + lookahead + 1:
            normed = normed.reindex(range(p.index[t0].to_period("D").ordinal - lookback,
                                          p.index[t0].to_period("D").ordinal + lookahead + 1))
        offsets = np.arange(-(p.index.get_loc(p.index[t0]) - t_start), len(normed) - 1 - (t0 - t_start) + 1)
        normed.index = offsets
        paths.append(normed)

    df = pd.concat(paths, axis=1) if len(paths) else pd.DataFrame(index=np.arange(-lookback, lookahead + 1))
    df = df.reindex(np.arange(-lookback, lookahead + 1))
    stats = pd.DataFrame({
        "mean": df.mean(axis=1, skipna=True),
        "median": df.median(axis=1, skipna=True),
        "p25": df.quantile(0.25, axis=1, interpolation="linear"),
        "p75": df.quantile(0.75, axis=1, interpolation="linear"),
        "n": df.count(axis=1).astype(int),
    })
    return stats
