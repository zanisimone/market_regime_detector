# src/viz/plots.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def load_labeled_features(path: Path) -> pd.DataFrame:
    """
    Load the labeled features parquet and ensure a DatetimeIndex.
    """
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def load_price_series(panel_path: Path, price_col: str = "SPX") -> pd.Series:
    """
    Load a price column from the processed market panel parquet as a Series.
    """
    panel = pd.read_parquet(panel_path)
    if not isinstance(panel.index, pd.DatetimeIndex):
        panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()
    if price_col not in panel.columns:
        raise ValueError(f"Column '{price_col}' not found in panel. Available: {list(panel.columns)}")
    return panel[price_col].dropna()


def plot_price_with_regimes(
    labeled: pd.DataFrame,
    price: pd.Series,
    *,
    label_col: str = "regime",
    name_col: Optional[str] = "regime_name",
    title: str = "Price with Regimes",
    out_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Plot a price series segmented by regime labels. Each regime is drawn as a separate line segment.
    """
    df = pd.DataFrame({"price": price}).join(labeled[[label_col] + ([name_col] if name_col and name_col in labeled.columns else [])], how="inner")
    df = df.dropna(subset=["price", label_col]).copy()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")

    for regime_id, seg in df.groupby(label_col):
        ax.plot(seg.index, seg["price"], label=(seg[name_col].iloc[0] if name_col and name_col in seg.columns else f"Regime {regime_id}"))

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.legend(loc="best")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    fig.tight_layout()
    plt.show()
    return None
