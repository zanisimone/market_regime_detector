# src/models/kmeans.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.config import PROC_DIR


def fit_kmeans(
    features: pd.DataFrame,
    *,
    k: int = 3,
    random_state: int = 42,
    n_init: int = 20,
    max_iter: int = 300,
    feature_cols: Optional[Sequence[str]] = None,
) -> Tuple[KMeans, np.ndarray, pd.DataFrame]:
    """
    Fit a K-Means model on the feature matrix.

    Parameters
    ----------
    features : pd.DataFrame
        Z-scored feature DataFrame indexed by date.
    k : int, default 3
        Number of clusters.
    random_state : int, default 42
        Random seed for reproducibility.
    n_init : int, default 20
        Number of centroid initializations to run.
    max_iter : int, default 300
        Maximum number of iterations per run.
    feature_cols : Sequence[str] | None, default None
        Subset of feature columns to use. If None, all columns are used.

    Returns
    -------
    model : KMeans
        Trained K-Means model.
    labels : np.ndarray
        Cluster label for each row in `features`.
    centers : pd.DataFrame
        Cluster centers as a DataFrame with feature means per cluster.
    """
    X = features[feature_cols] if feature_cols is not None else features.copy()
    model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init, max_iter=max_iter)
    labels = model.fit_predict(X.values)
    centers = pd.DataFrame(model.cluster_centers_, columns=X.columns)
    centers.index.name = "cluster"
    return model, labels, centers


def _regime_name_from_center(center: pd.Series) -> str:
    risk_score = center.get("spx_ret_5d", 0.0) - center.get("vix_level", 0.0) - 0.5 * center.get("spx_vol_20d", 0.0)
    slope = center.get("slope_10y2y", 0.0)
    if risk_score < -0.3 and center.get("vix_level", 0.0) > 0.0:
        return "Risk-Off"
    if risk_score > 0.1:
        return "Risk-On (Inverted)" if slope < 0 else "Risk-On (Steepening)"
    return "Neutral"


def infer_regime_labels(centers: pd.DataFrame) -> Dict[int, str]:
    """
    Map each cluster index to a semantic regime label based on cluster centers.

    Parameters
    ----------
    centers : pd.DataFrame
        Cluster centers with feature columns.

    Returns
    -------
    dict[int, str]
        Mapping from cluster id to regime name.
    """
    mapping: Dict[int, str] = {}
    for idx, row in centers.iterrows():
        mapping[int(idx)] = _regime_name_from_center(row)
    return mapping


def attach_labels(
    features: pd.DataFrame,
    labels: np.ndarray,
    label_map: Dict[int, str],
    *,
    label_col: str = "regime",
    name_col: str = "regime_name",
) -> pd.DataFrame:
    """
    Attach numeric and semantic regime labels to the features DataFrame.

    Parameters
    ----------
    features : pd.DataFrame
        Input features indexed by date.
    labels : np.ndarray
        Cluster labels.
    label_map : dict[int, str]
        Mapping from cluster id to regime name.
    label_col : str, default "regime"
        Name of the numeric label column.
    name_col : str, default "regime_name"
        Name of the semantic label column.

    Returns
    -------
    pd.DataFrame
        Features with appended regime columns.
    """
    out = features.copy()
    out[label_col] = labels.astype(int)
    out[name_col] = out[label_col].map(label_map)
    return out


def run_kmeans(
    features_path: Optional[Path] = None,
    out_labels_path: Optional[Path] = None,
    out_centers_path: Optional[Path] = None,
    *,
    k: int = 3,
    random_state: int = 42,
    n_init: int = 20,
    feature_cols: Optional[Sequence[str]] = None,
) -> Tuple[Path, Path]:
    """
    Run K-Means on the saved features and persist labels and centers.

    Parameters
    ----------
    features_path : Path | None, default None
        Path to the features parquet. Defaults to data/processed/features.parquet.
    out_labels_path : Path | None, default None
        Path to save features with labels (parquet). Defaults to data/processed/kmeans_labels.parquet.
    out_centers_path : Path | None, default None
        Path to save cluster centers (csv). Defaults to data/processed/kmeans_centers.csv.
    k : int, default 3
        Number of clusters.
    random_state : int, default 42
        Random seed.
    n_init : int, default 20
        Number of centroid initializations.
    feature_cols : Sequence[str] | None, default None
        Subset of feature columns to use.

    Returns
    -------
    (Path, Path)
        Paths to the saved labels parquet and centers csv.
    """
    features_path = features_path or (PROC_DIR / "features.parquet")
    out_labels_path = out_labels_path or (PROC_DIR / "kmeans_labels.parquet")
    out_centers_path = out_centers_path or (PROC_DIR / "kmeans_centers.csv")

    feats = pd.read_parquet(features_path)
    model, labels, centers = fit_kmeans(
        feats,
        k=k,
        random_state=random_state,
        n_init=n_init,
        feature_cols=feature_cols,
    )
    label_map = infer_regime_labels(centers)
    labeled = attach_labels(feats, labels, label_map)

    out_labels_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(out_labels_path)
    centers.to_csv(out_centers_path, index=True)
    return out_labels_path, out_centers_path


def main() -> None:
    """
    Run K-Means with defaults for quick local testing.
    """
    labels_path, centers_path = run_kmeans()
    print(f"Saved labeled features to: {labels_path}")
    print(f"Saved cluster centers to: {centers_path}")


if __name__ == "__main__":
    main()
