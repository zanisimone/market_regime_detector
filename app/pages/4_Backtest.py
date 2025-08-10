# --- sys.path bootstrap (page) ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------

import streamlit as st
import pandas as pd
from app.utils import project_root
from app.components.kpi_cards import kpi_row

# ---------- helpers ----------
def _find_date_col(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure there is a 'date' column."""
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df
    for c in ["datetime", "Datetime", "timestamp", "Timestamp", "DATE", "Date", "dt"]:
        if c in df.columns:
            df = df.rename(columns={c: "date"})
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            return df
    # last resort: auto-detect a datetime-like column
    for c in df.columns:
        try:
            conv = pd.to_datetime(df[c], errors="coerce")
            if conv.notna().mean() > 0.9:
                df = df.rename(columns={c: "date"})
                df["date"] = conv
                return df
        except Exception:
            pass
    raise ValueError("Could not identify a date column. Expected 'date' or a datetime index.")

def _find_price_col(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Return df with a 'price' column; fallback to common names or first numeric col."""
    if "price" in df.columns:
        return df, "price"
    for c in ["close", "Close", "adj_close", "Adj Close", "close_price", "px_last"]:
        if c in df.columns:
            return df.rename(columns={c: "price"}), "price"
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in num_cols if c != "regime"]
    if num_cols:
        chosen = num_cols[0]
        if chosen != "price":
            df = df.rename(columns={chosen: "price"})
        return df, "price"
    raise ValueError("No suitable price column found. Tried: price/close/adj_close/px_last or first numeric column.")

def _load_labels_csv(labels_file: Path) -> pd.DataFrame:
    """Load labels CSV and ensure it has 'date' and 'regime'."""
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file}")
    lab = pd.read_csv(labels_file)
    if "date" not in lab.columns:
        if "index" in lab.columns:
            lab = lab.rename(columns={"index": "date"})
        else:
            # auto-detect datetime-like column
            found = False
            for c in lab.columns:
                try:
                    conv = pd.to_datetime(lab[c], errors="coerce")
                    if conv.notna().mean() > 0.9:
                        lab = lab.rename(columns={c: "date"})
                        found = True
                        break
                except Exception:
                    pass
            if not found:
                raise KeyError("Labels CSV does not contain a 'date' column and could not auto-detect one.")
    lab["date"] = pd.to_datetime(lab["date"], errors="coerce")
    if "regime" not in lab.columns:
        raise KeyError("Labels CSV must contain a 'regime' column.")
    return lab[["date", "regime"]].dropna(subset=["date"]).sort_values("date")

def _load_price_panel() -> pd.DataFrame:
    """Try market_panel.parquet first, then panel.parquet; ensure 'date' and 'price'."""
    root = project_root()
    candidates = [
        root / "data" / "processed" / "market_panel.parquet",
        root / "data" / "processed" / "panel.parquet",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_parquet(p)
            df = _find_date_col(df)
            df, _ = _find_price_col(df)
            return df[["date", "price"]].sort_values("date")
    raise FileNotFoundError(
        "Could not find a panel parquet. Looked for data/processed/market_panel.parquet and data/processed/panel.parquet."
    )

def _load_joined() -> pd.DataFrame:
    """Load joined price + labels with 'date','price','regime'."""
    root = project_root()
    price = _load_price_panel()
    labels = _load_labels_csv(root / "reports" / "kmeans_labels.csv")
    df = price.merge(labels, on="date", how="inner").sort_values("date")
    return df

def _bt_simple(df: pd.DataFrame, long_regimes: list[int]) -> pd.DataFrame:
    """Binary allocation: invested if regime ∈ long_regimes, else cash."""
    df = df.copy().sort_values("date")
    df["ret"] = df["price"].pct_change().fillna(0.0)
    df["w"] = df["regime"].isin(long_regimes).astype(float)
    df["strat_ret"] = df["w"] * df["ret"]
    df["strat_eq"] = (1 + df["strat_ret"]).cumprod()
    df["buyhold_eq"] = (1 + df["ret"]).cumprod()
    return df

def _kpis(eq: pd.Series) -> tuple[float, float, float]:
    """Return (CAGR, MaxDD, FinalEquity) for an equity series indexed by date."""
    eq = eq.copy()
    eq.index = pd.to_datetime(eq.index)
    total = eq.iloc[-1]
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = total ** (1 / years) - 1 if years > 0 and total > 0 else 0.0
    roll_max = eq.cummax()
    mdd = (eq / roll_max - 1).min()
    return cagr, mdd, total

# ---------- page ----------
def main():
    st.title("🧪 Backtest")
    try:
        df = _load_joined()
    except Exception as e:
        st.error(f"Could not load data: {e}")
        return

    # Controls
    regimes = sorted(df["regime"].dropna().unique().tolist())
    default_long = [r for r in regimes if r != 0] or regimes  # default: tutti tranne 0
    chosen = st.multiselect("Long regimes", regimes, default=default_long)

    # Date range filter
    c1, c2 = st.columns([3, 1])
    with c2:
        min_d, max_d = df["date"].min(), df["date"].max()
        date_range = st.date_input("Date range", (min_d.date(), max_d.date()))
    if isinstance(date_range, tuple):
        df = df[(df["date"] >= pd.Timestamp(date_range[0])) & (df["date"] <= pd.Timestamp(date_range[1]))]

    # Backtest
    res = _bt_simple(df.set_index("date"), chosen)
    cagr, mdd, final_eq = _kpis(res["strat_eq"])

    kpi_row([
        {"label": "Strategy CAGR", "value": f"{cagr*100:.2f}%"},
        {"label": "Max Drawdown", "value": f"{mdd*100:.2f}%"},
        {"label": "Final Equity", "value": f"{final_eq:.2f}x"},
    ])

    st.line_chart(res[["strat_eq", "buyhold_eq"]])

    st.subheader("Equity data (tail)")
    st.dataframe(res.reset_index().tail(300), use_container_width=True)

if __name__ == "__main__":
    main()
