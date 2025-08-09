from __future__ import annotations
from pathlib import Path
from typing import Tuple, List
import pandas as pd
from sklearn.mixture import GaussianMixture

def run_gmm_rolling(
    features_path: Path,
    out_dir: Path,
    start: str,
    end: str,
    lookback_days: int = 504,
    oos_days: int = 21,
    step_days: int = 21,
    k: int = 3,
    random_state: int = 42,
    n_init: int = 5,
    covariance_type: str = "full",
) -> Tuple[Path, Path]:
    """
    Run a rolling backtest for Gaussian Mixture by refitting on each window.

    Parameters
    ----------
    features_path : Path
        Path to standardized features parquet.
    out_dir : Path
        Output directory.
    start : str
        Backtest window start in YYYY-MM-DD.
    end : str
        Backtest window end in YYYY-MM-DD.
    lookback_days : int, default 504
        Training window length in business days.
    oos_days : int, default 21
        Out-of-sample horizon per iteration.
    step_days : int, default 21
        Step size between refits.
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
    Tuple[Path, Path]
        Paths to (rolling_labels.parquet, rolling_schedule.csv).
    """
    feats = pd.read_parquet(features_path).sort_index()
    feats = feats.loc[start:end]
    idx = feats.index

    anchors: List[pd.Timestamp] = []
    if len(idx) > 0:
        cur = idx[0]
        while cur <= idx[-1]:
            anchors.append(cur)
            pos = idx.get_indexer([cur])[0]
            next_pos = pos + step_days
            if next_pos >= len(idx):
                break
            cur = idx[next_pos]

    rows = []
    all_labels = []

    for t0 in anchors:
        pos0 = idx.get_indexer([t0])[0]
        pos_train_start = max(0, pos0 - lookback_days + 1)
        pos_pred_end = min(len(idx) - 1, pos0 + oos_days)
        train_slice = feats.iloc[pos_train_start:pos0 + 1]
        pred_slice = feats.iloc[pos0 + 1:pos_pred_end + 1]
        if len(train_slice) < 2 or len(pred_slice) == 0:
            continue

        model = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            random_state=random_state,
            n_init=n_init,
        )
        model.fit(train_slice.values)
        pred_labels = model.predict(pred_slice.values)
        labels_df = pd.DataFrame({"regime": pred_labels}, index=pred_slice.index)
        all_labels.append(labels_df)

        rows.append({
            "anchor_date": t0.date(),
            "train_start": train_slice.index[0].date(),
            "train_end": train_slice.index[-1].date(),
            "pred_start": pred_slice.index[0].date(),
            "pred_end": pred_slice.index[-1].date(),
            "k": k,
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    schedule = pd.DataFrame(rows)
    labels = pd.concat(all_labels, axis=0).sort_index()
    labels = labels[~labels.index.duplicated(keep="last")]

    labels_path = out_dir / "rolling_labels.parquet"
    schedule_path = out_dir / "rolling_schedule.csv"

    labels.to_parquet(labels_path)
    schedule.to_csv(schedule_path, index=False)

    return labels_path, schedule_path
