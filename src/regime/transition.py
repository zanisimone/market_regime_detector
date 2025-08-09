import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List

def estimate_transition_matrix(labels: pd.Series, n_states: Optional[int] = None, smoothing: float = 1e-6) -> np.ndarray:
    """
    Estimate a first-order Markov transition matrix from a discrete regime label series.

    Parameters
    ----------
    labels : pd.Series
        Sequence of integer regime labels ordered in time.
    n_states : Optional[int]
        Number of distinct regimes. If None, inferred from labels' unique values.
    smoothing : float
        Additive smoothing to avoid zero-probability transitions.

    Returns
    -------
    np.ndarray
        Row-stochastic transition matrix P with shape (n_states, n_states),
        where P[i, j] = P(S_t = j | S_{t-1} = i).
    """
    x = labels.dropna().astype(int).to_numpy()
    if n_states is None:
        n_states = int(x.max()) + 1
    counts = np.full((n_states, n_states), smoothing, dtype=float)
    for i in range(1, len(x)):
        a, b = x[i - 1], x[i]
        if 0 <= a < n_states and 0 <= b < n_states:
            counts[a, b] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    P = counts / row_sums
    return P


def stationary_distribution(P: np.ndarray, tol: float = 1e-12, max_iter: int = 10000) -> np.ndarray:
    """
    Compute the stationary distribution of a Markov chain given its transition matrix.

    Parameters
    ----------
    P : np.ndarray
        Row-stochastic transition matrix of shape (n_states, n_states).
    tol : float
        Convergence tolerance for power iteration.
    max_iter : int
        Maximum number of iterations for power iteration.

    Returns
    -------
    np.ndarray
        Stationary distribution pi with shape (n_states,), satisfying pi^T P = pi^T.
    """
    n = P.shape[0]
    pi = np.full(n, 1.0 / n, dtype=float)
    for _ in range(max_iter):
        new_pi = pi @ P
        if np.max(np.abs(new_pi - pi)) < tol:
            return new_pi
        pi = new_pi
    return pi


def dwell_times(labels: pd.Series, n_states: Optional[int] = None) -> Dict[int, List[int]]:
    """
    Compute dwell times (consecutive run lengths) for each regime.

    Parameters
    ----------
    labels : pd.Series
        Sequence of integer regime labels ordered in time.
    n_states : Optional[int]
        Number of regimes. If None, inferred from labels.

    Returns
    -------
    Dict[int, List[int]]
        Map regime -> list of dwell lengths (in number of periods).
    """
    x = labels.dropna().astype(int).to_numpy()
    if len(x) == 0:
        return {}
    if n_states is None:
        n_states = int(x.max()) + 1
    out: Dict[int, List[int]] = {k: [] for k in range(n_states)}
    run_label = x[0]
    run_len = 1
    for i in range(1, len(x)):
        if x[i] == run_label:
            run_len += 1
        else:
            out[run_label].append(run_len)
            run_label = x[i]
            run_len = 1
    out[run_label].append(run_len)
    return out


def dwell_time_summary(labels: pd.Series, n_states: Optional[int] = None) -> pd.DataFrame:
    """
    Produce summary statistics of dwell-time distributions per regime.

    Parameters
    ----------
    labels : pd.Series
        Sequence of integer regime labels ordered in time.
    n_states : Optional[int]
        Number of regimes. If None, inferred from labels.

    Returns
    -------
    pd.DataFrame
        Summary with columns: count, mean, median, std, p25, p75, max for each regime.
    """
    dt = dwell_times(labels, n_states)
    rows = []
    for k, arr in dt.items():
        a = np.array(arr, dtype=float) if len(arr) else np.array([np.nan])
        rows.append({
            "regime": k,
            "count": int(len(arr)),
            "mean": float(np.nanmean(a)),
            "median": float(np.nanmedian(a)),
            "std": float(np.nanstd(a, ddof=1)) if len(arr) > 1 else np.nan,
            "p25": float(np.nanpercentile(a, 25)) if len(arr) else np.nan,
            "p75": float(np.nanpercentile(a, 75)) if len(arr) else np.nan,
            "max": float(np.nanmax(a)) if len(arr) else np.nan,
        })
    return pd.DataFrame(rows).set_index("regime").sort_index()
