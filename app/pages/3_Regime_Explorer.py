# --- sys.path bootstrap (page) ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------

import streamlit as st
import pandas as pd
from app.app_config import project_root
from app.utils import (
    initialize_session_state, 
    get_available_models_from_session,
    get_selected_model,
    set_selected_model,
    get_default_model_for_page3,
    load_model_labels,
    get_model_info
)
from app.components.regime_timeline import plot as plot_timeline
from app.components.kpi_cards import kpi_row
from app.components.regime_info import load_regime_catalog, show_regime_info, label_for
from app.components.model_selector import render_compact_model_selector


# --------- helpers ---------
def _find_date_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a 'date' column exists; if the index is datetime-like or a different
    date column name is present, convert to 'date' and reset index.
    """
    df = df.copy()

    # case 1: already has 'date'
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    # case 2: datetime-like index
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    # case 3: other typical date column names
    candidates = ["datetime", "Datetime", "timestamp", "Timestamp", "DATE", "Date", "dt"]
    for c in candidates:
        if c in df.columns:
            df = df.rename(columns={c: "date"})
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            return df

    # last resort: try to auto-detect the first datetime-like column
    for c in df.columns:
        try:
            converted = pd.to_datetime(df[c], errors="coerce")
            if converted.notna().mean() > 0.9:
                df = df.rename(columns={c: "date"})
                df["date"] = converted
                return df
        except Exception:
            pass

    raise ValueError("Could not identify a date column. Expected 'date' or datetime index.")


def _find_price_col(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Ensure we have a usable price column name. Prefer 'price', fallback to common names.
    Returns (df, price_col_name).
    """
    if "price" in df.columns:
        return df, "price"
    for c in ["close", "Close", "adj_close", "Adj Close", "close_price", "px_last"]:
        if c in df.columns:
            return df.rename(columns={c: "price"}), "price"
    # as a last resort, pick the first numeric column (excluding 'regime' if present)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    numeric_cols = [c for c in numeric_cols if c != "regime"]
    if numeric_cols:
        chosen = numeric_cols[0]
        if chosen != "price":
            df = df.rename(columns={chosen: "price"})
        return df, "price"
    raise ValueError("No suitable price column found. Tried: price/close/adj_close/px_last.")


def _load_labels_csv(labels_file: Path) -> pd.DataFrame:
    """
    Load labels CSV and ensure it has 'date' and 'regime'.
    """
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file}")
    lab = pd.read_csv(labels_file)
    # normalize date
    if "date" not in lab.columns:
        # if index-like column present due to previous parquet reset_index
        if "index" in lab.columns:
            lab = lab.rename(columns={"index": "date"})
        else:
            # try auto-detection
            for c in lab.columns:
                try:
                    converted = pd.to_datetime(lab[c], errors="coerce")
                    if converted.notna().mean() > 0.9:
                        lab = lab.rename(columns={c: "date"})
                        break
                except Exception:
                    pass
    if "date" not in lab.columns:
        raise KeyError("Labels CSV does not contain a 'date' column and could not auto-detect one.")
    lab["date"] = pd.to_datetime(lab["date"], errors="coerce")
    if "regime" not in lab.columns:
        raise KeyError("Labels CSV must contain a 'regime' column.")
    return lab[["date", "regime"]].dropna(subset=["date"]).sort_values("date")


def _load_price_panel() -> pd.DataFrame:
    """
    Load price panel from data/processed. Tries market_panel.parquet first, then panel.parquet.
    Ensures 'date' and 'price' columns.
    """
    root = project_root()
    cand = [
        root / "data" / "processed" / "market_panel.parquet",
        root / "data" / "processed" / "panel.parquet",
    ]
    for path in cand:
        if path.exists():
            df = pd.read_parquet(path)
            df = _find_date_col(df)
            df, price_col = _find_price_col(df)
            # keep only date and price (plus other columns for hover if you want)
            return df[["date", "price"]].sort_values("date")
    raise FileNotFoundError(
        "Could not find a panel parquet. Looked for data/processed/market_panel.parquet and panel.parquet."
    )


def _load_joined(selected_model: str) -> pd.DataFrame:
    """Load joined price + labels with 'date', 'price', 'regime' for selected model."""
    root = project_root()
    price = _load_price_panel()
    
    # Load labels using the new model management system
    labels = load_model_labels(selected_model)
    if labels is None:
        raise ValueError(f"Could not load labels for model: {selected_model}")
    
    df = price.merge(labels, on="date", how="inner").sort_values("date")
    return df


def _regime_kpis(df: pd.DataFrame, catalog):
    out = []
    df = df.copy()
    df["ret"] = df["price"].pct_change().fillna(0)
    for r, g in df.groupby("regime"):
        total = (1 + g["ret"]).prod() - 1.0
        out.append({
            "label": f"{label_for(int(r), catalog)} Total Return",
            "value": f"{total*100:.2f}%",
            "help": "Compounded return over the selected sample for days classified in this regime.",
        })
    return out


# --------- page ---------
def main():
    st.title("📊 Regime Explorer")

    # Initialize session state
    initialize_session_state()

    # Load regime catalog
    catalog = load_regime_catalog()
    
    # Sidebar with regime info
    st.sidebar.markdown("### Regimes")
    with st.sidebar:
        show_regime_info(catalog)

    # Model selection section
    st.header("🔍 Model Selection")
    
    # Get available models
    available_models = get_available_models_from_session()
    
    if not available_models:
        st.error("❌ No models found. Please run some clustering algorithms first.")
        st.info("Go to the **Clustering** page to train models.")
        return
    
    # Get default model (from page 2 or first available)
    default_model = get_default_model_for_page3()
    if default_model not in available_models:
        default_model = available_models[0]
    
    # Model selector
    selected_model = render_compact_model_selector(
        key="model_selector",
        default_model=default_model
    )
    
    # Update session state when model changes
    if selected_model and selected_model != get_selected_model():
        set_selected_model(selected_model)
    
    # Display model info
    if selected_model:
        model_info = get_model_info(selected_model)
        if model_info:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Algorithm", model_info.algorithm.upper())
            with col2:
                st.metric("Mode", model_info.mode.title())
            with col3:
                st.metric("Regimes", model_info.num_regimes)
            with col4:
                st.metric("Date Range", f"{model_info.date_range[0]} to {model_info.date_range[1]}")
    
    # Load and display data
    try:
        df = _load_joined(selected_model)
        
        if df.empty:
            st.warning("⚠️ No data available for the selected model and date range.")
            return
            
    except Exception as e:
        st.error(f"❌ Could not load data: {e}")
        return

    # Date range filter
    st.header("📅 Date Range Filter")
    col1, col2 = st.columns([3, 1])
    with col2:
        min_d, max_d = df["date"].min(), df["date"].max()
        date_range = st.date_input("Date range", (min_d.date(), max_d.date()))
    
    if isinstance(date_range, tuple):
        df = df[(df["date"] >= pd.Timestamp(date_range[0])) & (df["date"] <= pd.Timestamp(date_range[1]))]

    # Visualization section
    st.header("📈 Regime Timeline")
    plot_timeline(df, date_col="date", y_col="price", regime_col="regime")
    
    # KPI section
    st.header("📊 Performance Metrics")
    kpi_row(_regime_kpis(df, catalog))

    # Data section
    st.header("📋 Data Preview")
    
    # Download + preview
    col1, col2 = st.columns([1, 1])
    with col1:
        st.download_button(
            "📥 Download current view (CSV)", 
            df.to_csv(index=False).encode(),
            file_name=f"regime_view_{selected_model}.csv", 
            mime="text/csv"
        )
    
    with col2:
        st.info(f"Showing {len(df)} observations for model: **{selected_model}**")
    
    st.dataframe(df.tail(300), use_container_width=True)

if __name__ == "__main__":
    main()
