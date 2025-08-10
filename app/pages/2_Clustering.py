import streamlit as st
import pandas as pd
from pathlib import Path

# --- sys.path bootstrap (page) ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # .../market_regime_detector
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------

from app.utils import project_root

def _paths():
    """Return input/output paths relevant to clustering."""
    root = project_root()
    return {
        "features": root / "data" / "processed" / "features.parquet",
        "labels": root / "reports" / "kmeans_labels.csv",
    }

def _run_kmeans_split(train_start, train_end, test_start, test_end, n_clusters, random_state):
    """Shell out to your existing script for reproducibility."""
    import subprocess, sys
    cmd = [
        sys.executable, "-m", "scripts.run_kmeans_split",
        "--train-start", train_start, "--train-end", train_end,
        "--test-start", test_start, "--test-end", test_end,
        "--k", str(n_clusters), "--random-state", str(random_state),
        "--out-dir", "reports"
    ]
    st.info(f"Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    st.code(res.stdout)
    if res.returncode != 0:
        st.error(res.stderr)
    else:
        st.success("K-Means (split) completed.")

def main():
    """Clustering page."""
    st.title("🔍 Clustering")
    p = _paths()

    with st.form("kmeans_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            train_start = st.text_input("Train start", "2010-01-01")
            test_start = st.text_input("Test start", "2018-01-01")
        with c2:
            train_end = st.text_input("Train end", "2017-12-31")
            test_end = st.text_input("Test end", "2025-01-01")
        with c3:
            n_clusters = st.number_input("n_clusters", 2, 8, 3)
            random_state = st.number_input("random_state", 0, 10_000, 42)
        run = st.form_submit_button("Run K-Means split")

    if run:
        _run_kmeans_split(train_start, train_end, test_start, test_end, n_clusters, random_state)

    if p["labels"].exists():
        df = pd.read_csv(p["labels"], parse_dates=["date"])
        st.subheader("Labels preview")
        st.dataframe(df.tail(200), use_container_width=True)
    else:
        st.info("No labels found yet. Run K-Means above.")

if __name__ == "__main__":
    main()
