# --- sys.path bootstrap (page) ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------

import streamlit as st
import pandas as pd
import subprocess
import io
import zipfile
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date
from app.utils import project_root

def _paths():
    """Return canonical file paths used by the page."""
    root = project_root()
    return {
        "features": root / "data" / "processed" / "features.parquet",
        "panel": root / "data" / "processed" / "panel.parquet",
        "report_dir": root / "reports" / "feature_report"
    }

def _build_features_from_cli():
    """Call your script via subprocess to build features."""
    cmd = [sys.executable, "-m", "scripts.build_features"]
    st.info("Running build_features…")
    res = subprocess.run(cmd, capture_output=True, text=True)
    st.code(res.stdout or res.stderr)
    if res.returncode != 0:
        st.error("Feature building failed.")
    else:
        st.success("Features built successfully.")

def _run_validation(split_ratio=None, split_date=None):
    """Run validate_features.py with chosen split param."""
    cmd = [sys.executable, "-m", "scripts.validate_features"]
    if split_ratio is not None:
        cmd += ["--split-ratio", str(split_ratio)]
    elif split_date is not None:
        cmd += ["--split-date", str(split_date)]
    else:
        st.error("Provide either split_ratio or split_date.")
        return
    st.info(f"Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    st.code(res.stdout or res.stderr)
    if res.returncode != 0:
        st.error("Validation failed.")
    else:
        st.success("Validation completed.")

def _load_summary(report_dir: Path):
    """Try to load summary.json or summary.csv from the report dir."""
    summary_file_json = report_dir / "summary.json"
    summary_file_csv = report_dir / "summary.csv"
    if summary_file_json.exists():
        import json
        return json.loads(summary_file_json.read_text(encoding="utf-8"))
    elif summary_file_csv.exists():
        return pd.read_csv(summary_file_csv)
    return None

def _plot_nan_heatmap(df: pd.DataFrame):
    """Plot a heatmap of NaN values per column."""
    nan_matrix = df.isna().astype(int).T  # columns as rows
    plt.figure(figsize=(12, max(4, len(df.columns) * 0.3)))
    sns.heatmap(nan_matrix, cbar=False, cmap="Reds")
    plt.ylabel("Columns")
    plt.xlabel("Row index")
    plt.title("NaN Pattern Heatmap")
    st.pyplot(plt.gcf())
    plt.close()

def main():
    """Data & Features page."""
    st.title("📦 Data & Features")
    p = _paths()

    # --- Actions ---
    left, right = st.columns(2)
    with left:
        if st.button("🔄 Build/Refresh features"):
            _build_features_from_cli()

    with right:
        with st.form("validate_form"):
            mode = st.radio("Validation split", ["By ratio", "By date"], horizontal=True)
            split_ratio = None
            split_date = None
            if mode == "By ratio":
                split_ratio = st.slider("Train ratio", 0.5, 0.95, 0.80, step=0.01)
            else:
                split_date = st.date_input("Split date", date(2018, 1, 1))
            run_val = st.form_submit_button("Validate features")
        if run_val:
            _run_validation(split_ratio=split_ratio if mode == "By ratio" else None,
                            split_date=split_date if mode == "By date" else None)

    # --- Features preview ---
    if p["features"].exists():
        df = pd.read_parquet(p["features"])
        st.subheader("Preview of features.parquet")
        st.dataframe(df.head(200), use_container_width=True)
        st.caption(f"Rows: {len(df):,} · Cols: {len(df.columns)} · File: {p['features'].relative_to(project_root())}")

        with st.expander("🔍 NaN Heatmap"):
            _plot_nan_heatmap(df)
    else:
        st.warning("features.parquet not found. Click 'Build/Refresh features' first.")

    # --- Validation report preview ---
    if p["report_dir"].exists():
        st.subheader("Validation report")
        summary = _load_summary(p["report_dir"])
        if summary is not None:
            st.write("**Summary:**")
            st.write(summary)
        else:
            st.info("No summary file found in report dir.")

        # Download ZIP of report dir
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            for file in p["report_dir"].glob("**/*"):
                if file.is_file():
                    zf.write(file, arcname=file.relative_to(p["report_dir"]))
        st.download_button(
            "📥 Download full report (.zip)",
            buffer.getvalue(),
            file_name="feature_report.zip",
            mime="application/zip"
        )
    else:
        st.info("No validation report found. Run 'Validate features' first.")

if __name__ == "__main__":
    main()
