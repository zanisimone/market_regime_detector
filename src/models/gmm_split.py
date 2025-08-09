from __future__ import annotations
from pathlib import Path
from typing import Tuple
import pandas as pd
import joblib
from sklearn.mixture import GaussianMixture

def run_gmm_split(
    features_path: Path,
    out_dir: Path,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    k: int = 3,
    random_state: int = 42,
    n_init: int = 5,
    covariance_type: str = "full",
) -> Tuple[Path, Path, Path, Path]:
    """
    Fit Gaussian Mixture on a temporal train split and predict on the test split.

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
        Number of mixture components.
    random_state : int, default 42
        Random state for reproducibility.
    n_init : int, default 5
        Number of initializations.
    covariance_type : str, default "full"
        Covariance type passed to GaussianMixture.

    Returns
    -------
    Tuple[Path, Path, Path, Path]
        Paths to (train_labels.parquet, test_labels.parquet, gmm_model.pkl, gmm_centers.csv).
    """
    feats = pd.read_parquet(features_path)
    train_df = feats.loc[train_start:train_end].dropna()
    test_df = feats.loc[test_start:test_end].dropna()

    model = GaussianMixture(
        n_components=k,
        covariance_type=covariance_type,
        random_state=random_state,
        n_init=n_init,
    )
    model.fit(train_df.values)

    train_labels = pd.DataFrame({"regime": model.predict(train_df.values)}, index=train_df.index)
    test_labels = pd.DataFrame({"regime": model.predict(test_df.values)}, index=test_df.index)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_labels_path = out_dir / "train_labels.parquet"
    test_labels_path = out_dir / "test_labels.parquet"
    model_path = out_dir / "gmm_model.pkl"
    centers_path = out_dir / "gmm_means.csv"

    train_labels.to_parquet(train_labels_path)
    test_labels.to_parquet(test_labels_path)
    joblib.dump(model, model_path)
    pd.DataFrame(model.means_).to_csv(centers_path, index=False)

    return train_labels_path, test_labels_path, model_path, centers_path
