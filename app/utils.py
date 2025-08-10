import streamlit as st
from pathlib import Path

DEFAULTS = """\
[symbols]
benchmark = "SPX"
price_col = "price"

[features]
standardize = "rolling"
window = 252

[clustering]
algo = "kmeans"
n_clusters = 3
random_state = 42
"""

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]

@st.cache_data(show_spinner=False)
def load_defaults() -> str:
    cfg_path = project_root() / "configs" / "app.defaults.toml"
    if not cfg_path.exists():
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(DEFAULTS, encoding="utf-8")
    return cfg_path.read_text(encoding="utf-8")
