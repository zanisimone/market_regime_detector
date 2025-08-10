import pandas as pd
import plotly.express as px
import streamlit as st

def plot(df: pd.DataFrame, date_col: str = "date", y_col: str = "price", regime_col: str = "regime"):
    """Plot a colored price timeline by regime using plotly."""
    fig = px.scatter(
        df,
        x=date_col,
        y=y_col,
        color=regime_col,
        hover_data=df.columns,
        render_mode="webgl",
        opacity=0.8,
    )
    fig.update_traces(mode="lines+markers")
    st.plotly_chart(fig, use_container_width=True)
