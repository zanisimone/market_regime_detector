from __future__ import annotations
from pathlib import Path
from typing import Tuple
import pandas as pd
import joblib

def run_hmm_split(
    features_path: Path,
    out_dir: Path,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    k: int = 3,
    random_state: int = 42,
    n_iter: int = 200,
    covariance_type: str = "full",
) -> Tuple[Path, Path, Path, Path]:
    """
    Fit Gaussian HMM on a temporal train split and predict on the test split.

    This implementation requires `hmmlearn`. Install via:
    `pip install hmmlearn`.

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
        Number of hidden states.
    random_state : int, default 42
        Random state for reproducibility.
    n_iter : int, default 200
        Maximum EM iterations.
    covariance_type : str, default "full"
        Covariance type for GaussianHMM.

    Returns
    -------
    Tuple[Path, Path, Path, Path]
        Paths to (train_labels.parquet, test_labels.parquet, hmm_model.pkl, hmm_means.csv).
    """
    from hmmlearn.hmm import GaussianHMM

    feats = pd.read_parquet(features_path)
    train_df = feats.loc[train_start:train_end].dropna()
    test_df = feats.loc[test_start:test_end].dropna()

    model = GaussianHMM(
        n_components=k,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
        init_params="stmcw",
    )
    model.fit(train_df.values)

    train_labels = pd.DataFrame({"regime": model.predict(train_df.values)}, index=train_df.index)
    test_labels = pd.DataFrame({"regime": model.predict(test_df.values)}, index=test_df.index)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_labels_path = out_dir / "train_labels.parquet"
    test_labels_path = out_dir / "test_labels.parquet"
    model_path = out_dir / "hmm_model.pkl"
    centers_path = out_dir / "hmm_means.csv"

    train_labels.to_parquet(train_labels_path)
    test_labels.to_parquet(test_labels_path)
    joblib.dump(model, model_path)
    pd.DataFrame(model.means_).to_csv(centers_path, index=False)

    return train_labels_path, test_labels_path, model_path, centers_path
