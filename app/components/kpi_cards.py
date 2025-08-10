import streamlit as st

def kpi_row(items):
    """Render a row of KPI cards; items is list of dicts with keys: label, value, help."""
    cols = st.columns(len(items))
    for c, it in zip(cols, items):
        with c:
            st.metric(label=it["label"], value=it["value"], help=it.get("help", ""))
