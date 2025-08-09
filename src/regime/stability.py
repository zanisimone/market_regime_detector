import numpy as np
import pandas as pd
from typing import List, Dict, Optional

def frobenius_dispersion(P_list: List[np.ndarray]) -> float:
    """
    Compute dispersion across transition matrices via mean Frobenius distance to the centroid.

    Parameters
    ----------
    P_list : List[np.ndarray]
        List of row-stochastic transition matrices from different splits/models.

    Returns
    -------
    float
        Average Frobenius norm distance to the mean transition matrix.
    """
    if len(P_list) == 0:
        return np.nan
    P_bar = np.mean(np.stack(P_list, axis=0), axis=0)
    d = [np.linalg.norm(P - P_bar, ord="fro") for P in P_list]
    return float(np.mean(d))


def spectral_gap(P: np.ndarray) -> float:
    """
    Compute spectral gap (1 - |lambda_2|) as a stability proxy for mixing/mean-reversion.

    Parameters
    ----------
    P : np.ndarray
        Transition matrix.

    Returns
    -------
    float
        Spectral gap in [0, 1], larger implies faster mixing.
    """
    evals = np.linalg.eigvals(P.T)
    evals_sorted = np.sort(np.abs(evals))[::-1]
    if len(evals_sorted) < 2:
        return np.nan
    return float(1.0 - min(1.0, np.abs(evals_sorted[1])))


def stability_kpis(P_list: List[np.ndarray]) -> Dict[str, float]:
    """
    Aggregate transition-matrix stability KPIs across splits/models.

    Parameters
    ----------
    P_list : List[np.ndarray]
        Collection of transition matrices.

    Returns
    -------
    Dict[str, float]
        KPIs: frobenius_dispersion, mean_spectral_gap, std_spectral_gap, coeff_var_P.
    """
    if len(P_list) == 0:
        return {"frobenius_dispersion": np.nan, "mean_spectral_gap": np.nan, "std_spectral_gap": np.nan, "coeff_var_P": np.nan}
    fd = frobenius_dispersion(P_list)
    gaps = [spectral_gap(P) for P in P_list]
    P_stack = np.stack(P_list, axis=0)
    coeff_var = float(P_stack.std(axis=0).mean() / (P_stack.mean(axis=0).mean() + 1e-12))
    return {
        "frobenius_dispersion": float(fd),
        "mean_spectral_gap": float(np.nanmean(gaps)),
        "std_spectral_gap": float(np.nanstd(gaps, ddof=1)) if len(gaps) > 1 else np.nan,
        "coeff_var_P": coeff_var,
    }


def dwell_stability_kpis(dwell_summaries: List[pd.DataFrame]) -> Dict[str, float]:
    """
    Measure dwell-time stability across splits via mean/std of mean dwell per regime.

    Parameters
    ----------
    dwell_summaries : List[pd.DataFrame]
        Each DataFrame is the output of dwell_time_summary for a split.

    Returns
    -------
    Dict[str, float]
        KPIs: mean_mean_dwell, std_mean_dwell, mean_max_dwell.
    """
    if len(dwell_summaries) == 0:
        return {"mean_mean_dwell": np.nan, "std_mean_dwell": np.nan, "mean_max_dwell": np.nan}
    means = []
    maxs = []
    for df in dwell_summaries:
        means.append(df["mean"].mean())
        maxs.append(df["max"].mean())
    return {
        "mean_mean_dwell": float(np.nanmean(means)),
        "std_mean_dwell": float(np.nanstd(means, ddof=1)) if len(means) > 1 else np.nan,
        "mean_max_dwell": float(np.nanmean(maxs)),
    }
