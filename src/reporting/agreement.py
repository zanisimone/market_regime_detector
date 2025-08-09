import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def pairwise_confusion(a: pd.Series, b: pd.Series, labels: List[int]) -> pd.DataFrame:
    """
    Build a confusion-like crosstab between two label series on their common index.

    Parameters
    ----------
    a : pd.Series
        First label series (integer regimes), indexed by date/time.
    b : pd.Series
        Second label series (integer regimes), indexed by date/time.
    labels : List[int]
        Ordered list of all regime ids to include in rows/cols.

    Returns
    -------
    pd.DataFrame
        Matrix C where C[i, j] counts occurrences with a==i and b==j on the intersection index.
        Rows correspond to 'a' labels; columns correspond to 'b' labels.
    """
    idx = a.index.intersection(b.index)
    ca = a.loc[idx].astype(int)
    cb = b.loc[idx].astype(int)
    C = pd.crosstab(ca, cb).reindex(index=labels, columns=labels, fill_value=0)
    return C


def agreement_rate(a: pd.Series, b: pd.Series) -> float:
    """
    Compute fraction of times two label series agree on the intersection of their indices.

    Parameters
    ----------
    a : pd.Series
        First label series.
    b : pd.Series
        Second label series.

    Returns
    -------
    float
        Agreement rate in [0, 1].
    """
    idx = a.index.intersection(b.index)
    if len(idx) == 0:
        return np.nan
    return float((a.loc[idx].astype(int).values == b.loc[idx].astype(int).values).mean())


def multi_model_agreement(labels_map: Dict[str, pd.Series], labels: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute model-vs-model agreement matrix and mean per-class Jaccard indices.

    Parameters
    ----------
    labels_map : Dict[str, pd.Series]
        Mapping model_name -> label series (integer regimes).
    labels : List[int]
        Ordered list of all regime ids.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A: model-by-model agreement rates in [0,1].
        J: model-by-model mean Jaccard index across classes, where for each class k:
           J_k = |{t: m1==k and m2==k}| / |{t: m1==k or m2==k}|.
    """
    models = list(labels_map.keys())
    A = pd.DataFrame(index=models, columns=models, dtype=float)
    J = pd.DataFrame(index=models, columns=models, dtype=float)
    for i, mi in enumerate(models):
        for j, mj in enumerate(models):
            ai = labels_map[mi]
            aj = labels_map[mj]
            idx = ai.index.intersection(aj.index)
            if len(idx) == 0:
                A.loc[mi, mj] = np.nan
                J.loc[mi, mj] = np.nan
                continue
            xi = ai.loc[idx].astype(int).values
            xj = aj.loc[idx].astype(int).values
            A.loc[mi, mj] = float((xi == xj).mean())
            jacc = []
            for k in labels:
                inter = np.logical_and(xi == k, xj == k).sum()
                union = np.logical_or(xi == k, xj == k).sum()
                jacc.append(inter / union if union > 0 else np.nan)
            J.loc[mi, mj] = float(np.nanmean(jacc))
    return A, J
