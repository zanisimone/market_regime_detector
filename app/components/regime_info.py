import json
import pandas as pd
import streamlit as st
from app.app_config import project_root

# Default catalog if no config is provided
DEFAULT_REGIMES = {
    0: {"name": "Risk-Off", "desc": "Defensive conditions: rising volatility, drawdowns, tighter financial conditions."},
    1: {"name": "Risk-On (Steepening)", "desc": "Pro-cyclical risk appetite: improving growth, steepening curve, broader breadth."},
    2: {"name": "Risk-On (Inverted Curve)", "desc": "Late-cycle/liquidity-led: strong risk appetite despite inversion; momentum-driven."},
}

def load_regime_catalog() -> dict[int, dict]:
    """
    Load an optional regimes catalog from configs/regimes.json.
    Expected format:
    {
      "0": {"name": "...","desc": "..."},
      "1": {"name": "...","desc": "..."}
    }
    """
    path = project_root() / "configs" / "regimes.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            out = {}
            for k, v in data.items():
                try:
                    out[int(k)] = {"name": v.get("name", f"Regime {k}"),
                                   "desc": v.get("desc", "")}
                except Exception:
                    pass
            return out or DEFAULT_REGIMES
        except Exception:
            return DEFAULT_REGIMES
    return DEFAULT_REGIMES

def label_for(regime_id: int, catalog: dict[int, dict]) -> str:
    """Return pretty label like '1 — Risk-On (Steepening)'."""
    info = catalog.get(int(regime_id), {})
    name = info.get("name", f"Regime {int(regime_id)}")
    return f"{int(regime_id)} — {name}"

def show_regime_info(catalog: dict[int, dict]):
    """
    Render a small info menu with regime descriptions.
    Uses st.popover when available; falls back to st.expander.
    """
    rows = []
    for rid, meta in sorted(catalog.items()):
        rows.append({"ID": rid, "Name": meta.get("name", f"Regime {rid}"),
                     "Description": meta.get("desc", "")})
    df = pd.DataFrame(rows)

    # Try popover (newer Streamlit); fallback to expander
    try:
        with st.popover("ℹ️ Regime info"):
            st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception:
        with st.expander("ℹ️ Regime info", expanded=False):
            st.dataframe(df, use_container_width=True, hide_index=True)
