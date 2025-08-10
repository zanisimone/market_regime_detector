import streamlit as st
from utils import load_defaults

def _init_page():
    """Set basic page config and global style."""
    st.set_page_config(page_title="Market Regime Detector", page_icon="🧭", layout="wide")
    st.title("🧭 Market Regime Detector")
    st.caption("Interactive UI over your existing pipeline (K-Means / HMM, rolling standardization, regime stats & backtests).")

def _sidebar():
    """Render the sidebar with global settings."""
    cfg = load_defaults()
    st.sidebar.header("Global Settings")
    st.sidebar.write("These act as defaults, each page can override.")
    st.sidebar.code(cfg, language="toml")
    st.sidebar.info("Use the pages on the left to navigate.")
    return cfg

def main():
    """Entry point for the Streamlit multipage app."""
    _init_page()
    _sidebar()
    st.success("Open a page from the sidebar to get started: Data & Features → Clustering → Regime Explorer → Backtest.")

if __name__ == "__main__":
    main()
