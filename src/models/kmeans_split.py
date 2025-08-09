from __future__ import annotations
from pathlib import Path
from typing import Tuple
import pandas as pd
import joblib
from sklearn.cluster import KMeans

def run_kmeans_split(
    features_path: Path,
    out_dir: Path,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    k: int = 3,
    random_state: int = 42,
    n_init: int = 20,
) -> Tuple[Path, Path, Path, Path]:
    """
    Fit KMeans on a temporal train split and predict on the test split.

    Parameters
    ----------
    features_path : Path
        Path to standardized features parquet.
    out_dir : Path
        Output directory.
    train_start : str
        Training period start in YYYY-MM-DD.
    train_end : str
        Training period end in YYYY-MM-DD.
    test_start : str
        Test period start in YYYY-MM-DD.
    test_end : str
        Test period end in YYYY-MM-DD.
    k : int, default 3
        Number of clusters.
    random_state : int, default 42
        Random state for reproducibility.
    n_init : int, default 20
        Number of initializations.

    Returns
    -------
    Tuple[Path, Path, Path, Path]
        Paths to (train_labels.parquet, test_labels.parquet, kmeans_model.pkl, kmeans_centers.csv).
    """
    feats = pd.read_parquet(features_path)
    train_df = feats.loc[train_start:train_end].dropna()
    test_df = feats.loc[test_start:test_end].dropna()

    model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    model.fit(train_df.values)

    train_labels = pd.DataFrame({"regime": model.labels_}, index=train_df.index)
    test_labels = pd.DataFrame({"regime": model.predict(test_df.values)}, index=test_df.index)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_labels_path = out_dir / "train_labels.parquet"
    test_labels_path = out_dir / "test_labels.parquet"
    model_path = out_dir / "kmeans_model.pkl"
    centers_path = out_dir / "kmeans_centers.csv"

    train_labels.to_parquet(train_labels_path)
    test_labels.to_parquet(test_labels_path)
    joblib.dump(model, model_path)
    pd.DataFrame(model.cluster_centers_).to_csv(centers_path, index=False)

    return train_labels_path, test_labels_path, model_path, centers_path
