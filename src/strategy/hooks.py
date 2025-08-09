import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def constant_weight_map(assets: List[str],
                        risk_on_label: int,
                        risk_off_label: int,
                        on_weights: Dict[str, float],
                        off_weights: Optional[Dict[str, float]] = None,
                        cash_asset: str = "CASH") -> Dict[int, Dict[str, float]]:
    """
    Build a regime -> weights mapping for a simple two-regime allocation rule.

    Parameters
    ----------
    assets : List[str]
        List of tradable asset tickers present in the price DataFrame.
    risk_on_label : int
        Integer id used for Risk-On regime in labels.
    risk_off_label : int
        Integer id used for Risk-Off regime in labels.
    on_weights : Dict[str, float]
        Weights dictionary applied when labels == risk_on_label.
    off_weights : Optional[Dict[str, float]]
        Weights dictionary applied when labels == risk_off_label. If None, allocate to cash.
    cash_asset : str
        Synthetic cash column name expected in prices (constant 1.0 level).

    Returns
    -------
    Dict[int, Dict[str, float]]
        Mapping from regime id to weights dictionaries.
    """
    if off_weights is None:
        off_weights = {cash_asset: 1.0}
    for w in (on_weights, off_weights):
        s = sum(w.get(a, 0.0) for a in set(list(w.keys()) + assets + [cash_asset]))
        if not np.isclose(s, 1.0):
            raise ValueError("Weights must sum to 1.0")
    return {risk_on_label: on_weights, risk_off_label: off_weights}


def weights_from_labels(labels: pd.Series,
                        regime_weights: Dict[int, Dict[str, float]],
                        assets: List[str],
                        cash_asset: str = "CASH") -> pd.DataFrame:
    """
    Convert a regime label series into a time-aligned weight matrix.

    Parameters
    ----------
    labels : pd.Series
        Integer regime labels indexed by date.
    regime_weights : Dict[int, Dict[str, float]]
        Mapping regime -> weights dict.
    assets : List[str]
        Asset universe in desired column order.
    cash_asset : str
        Column name for synthetic cash.

    Returns
    -------
    pd.DataFrame
        DataFrame of weights with columns assets + [cash_asset], rows indexed by labels.index.
    """
    cols = list(assets) + [cash_asset]
    rows = []
    for r in labels.astype(int).tolist():
        w = regime_weights.get(r, {cash_asset: 1.0})
        row = [float(w.get(a, 0.0)) for a in assets] + [float(w.get(cash_asset, 0.0))]
        rows.append(row)
    W = pd.DataFrame(rows, index=labels.index, columns=cols)
    W = W.div(W.sum(axis=1).replace(0.0, 1.0), axis=0)
    return W
