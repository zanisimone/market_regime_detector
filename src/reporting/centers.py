import numpy as np
import pandas as pd
from typing import List, Optional, Dict


def unscale_centers(centers_z: np.ndarray,
                    feature_means: Optional[np.ndarray] = None,
                    feature_stds: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Convert cluster centers from z-score space back to original feature scale.

    Parameters
    ----------
    centers_z : np.ndarray
        Array of shape (K, F) with centers expressed in standardized space.
    feature_means : Optional[np.ndarray]
        Vector of feature means in original space with shape (F,). If None, returns z-space.
    feature_stds : Optional[np.ndarray]
        Vector of feature stds in original space with shape (F,). If None, returns z-space.

    Returns
    -------
    np.ndarray
        Centers on original scale if means/stds provided, else same as input.
    """
    if feature_means is None or feature_stds is None:
        return centers_z.copy()
    return centers_z * feature_stds[np.newaxis, :] + feature_means[np.newaxis, :]


def feature_contributions(centers_z: np.ndarray,
                          feature_names: List[str],
                          top: int = 10,
                          signed: bool = True) -> pd.DataFrame:
    """
    Rank features by their signed or absolute contribution per regime using standardized centers.

    Parameters
    ----------
    centers_z : np.ndarray
        Array (K, F) of cluster centers in z-score space.
    feature_names : List[str]
        Names for the F features in column order.
    top : int
        Number of top contributors to include per regime.
    signed : bool
        If True, keep the sign; if False, rank by absolute magnitude and report abs values.

    Returns
    -------
    pd.DataFrame
        Long-form table with columns: regime, rank, feature, value, abs_value.
        Higher rank indicates stronger contribution in magnitude.
    """
    K, F = centers_z.shape
    rows = []
    for k in range(K):
        vals = centers_z[k, :]
        order = np.argsort(-np.abs(vals))[: min(top, F)]
        for r, j in enumerate(order, start=1):
            v = vals[j] if signed else np.sign(vals[j]) * np.abs(vals[j])
            rows.append({
                "regime": k,
                "rank": r,
                "feature": feature_names[j],
                "value": float(v),
                "abs_value": float(abs(vals[j])),
            })
    return pd.DataFrame(rows).sort_values(["regime", "rank"]).reset_index(drop=True)
