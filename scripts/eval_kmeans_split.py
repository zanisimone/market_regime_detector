import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans

def internal_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute internal clustering metrics on features X and labels.
    """
    return {
        "silhouette": float(silhouette_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
    }

def temporal_metrics(df: pd.DataFrame) -> dict:
    """
    Compute temporal coherence metrics given df with columns ['date','regime'] sorted by date.
    """
    df = df.sort_values("date").reset_index(drop=True)
    switches = (df["regime"].shift(1) != df["regime"]).sum() - 1
    durations = df.groupby((df["regime"].shift() != df["regime"]).cumsum())["date"].agg(["min","max","count"])
    avg_dur = float(durations["count"].mean())
    regime_share = df["regime"].value_counts(normalize=True).to_dict()
    # transition matrix
    tr = pd.crosstab(df["regime"].shift(1), df["regime"], normalize=0).fillna(0.0)
    persistence = float(np.nanmean(np.diag(tr.values))) if tr.size else np.nan
    return {
        "switches": int(max(switches, 0)),
        "avg_duration_days": avg_dur,
        "regime_share": {int(k): float(v) for k, v in regime_share.items()},
        "persistence": persistence,
        "transition_matrix": tr.to_dict(),
    }

def economic_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-regime performance metrics. Expects columns ['date','regime','ret'] where 'ret' are daily returns.
    """
    g = df.groupby("regime")
    def sharpe(x):
        mu = np.mean(x)
        sd = np.std(x, ddof=1)
        return float(mu / sd) * np.sqrt(252) if sd > 0 else np.nan
    out = pd.DataFrame({
        "n_days": g["ret"].count(),
        "mean_daily": g["ret"].mean(),
        "vol_daily": g["ret"].std(ddof=1),
        "sharpe_annual": g["ret"].apply(sharpe),
        "hit_rate": g["ret"].apply(lambda x: float((x>0).mean())),
        "cum_return": g["ret"].sum(),
    })
    out["cagr_approx"] = (1.0 + out["mean_daily"])**252 - 1.0
    return out.reset_index()

def fit_kmeans_train_assign_test(X_train, X_test, k=3, seed=42):
    """
    Fit KMeans on X_train and assign clusters to X_test via nearest centroid.
    """
    km = KMeans(n_clusters=k, n_init=50, random_state=seed)
    km.fit(X_train)
    # assign test to nearest centroid
    dists = np.linalg.norm(X_test[:, None, :] - km.cluster_centers_[None, :, :], axis=2)
    labels_test = dists.argmin(axis=1)
    return km.labels_, labels_test, km.cluster_centers_

def stability_ari(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """
    Compute Adjusted Rand Index between two labelings for stability assessment.
    """
    return float(adjusted_rand_score(labels_a, labels_b))
