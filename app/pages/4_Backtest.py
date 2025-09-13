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
from app.components.kpi_cards import kpi_row
from app.components.regime_info import load_regime_catalog, show_regime_info, label_for
from app.components.model_selector import render_compact_model_selector


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

def _load_joined(selected_model: str) -> pd.DataFrame:
    """Load joined price + labels with 'date','price','regime' for selected model."""
    root = project_root()
    price = _load_price_panel()
    
    # Load labels using the new model management system
    labels = load_model_labels(selected_model)
    if labels is None:
        raise ValueError(f"Could not load labels for model: {selected_model}")
    
    df = price.merge(labels, on="date", how="inner").sort_values("date")
    return df

def _bt_etf(df: pd.DataFrame, long_regimes: list[int], 
            initial_capital: float = 100000.0, 
            ter_rate: float = 0.0009) -> pd.DataFrame:
    """
    ETF-style binary allocation with TER (Total Expense Ratio).
    
    Args:
        df: DataFrame with 'date', 'price', 'regime' columns
        long_regimes: List of regime IDs for long positions
        initial_capital: Starting capital amount
        ter_rate: Annual TER rate (e.g., 0.0009 for 0.09% SPY)
    
    Returns:
        DataFrame with strategy performance including costs
    """
    df = df.copy().sort_values("date")
    df["ret"] = df["price"].pct_change().fillna(0.0)
    
    # Binary allocation weights
    df["w"] = df["regime"].isin(long_regimes).astype(float)
    
    # Calculate position changes (when weight changes)
    df["w_prev"] = df["w"].shift(1).fillna(0.0)
    df["position_change"] = df["w"] - df["w_prev"]
    
    # Calculate strategy returns before costs
    df["strat_ret_before_costs"] = df["w"] * df["ret"]
    
    # Calculate TER (daily rate from annual rate)
    # TER is applied continuously, not per trade
    daily_ter = ter_rate / 252  # Convert annual to daily
    
    # Apply TER only when invested (w > 0)
    df["ter_cost"] = df["w"] * daily_ter
    
    # Net strategy returns (after TER)
    df["strat_ret"] = df["strat_ret_before_costs"] - df["ter_cost"]
    
    # Calculate equity curves
    df["strat_eq"] = initial_capital * (1 + df["strat_ret"]).cumprod()
    df["buyhold_eq"] = initial_capital * (1 + df["ret"]).cumprod()
    
    # Calculate cumulative TER costs in dollars
    df["cumulative_costs"] = (df["ter_cost"] * initial_capital).cumsum()
    
    # Calculate position value
    df["position_value"] = df["w"] * df["strat_eq"]
    df["cash_value"] = (1 - df["w"]) * df["strat_eq"]
    
    # For compatibility with metrics calculation
    df["trading_cost"] = df["ter_cost"]
    
    return df

def _bt_futures(df: pd.DataFrame, long_regimes: list[int], 
                initial_capital: float = 100000.0, 
                commission_per_contract: float = 2.50,
                margin_per_contract: float = 14000.0) -> pd.DataFrame:
    """
    Futures ES-style binary allocation with contract-based trading.
    
    Args:
        df: DataFrame with 'date', 'price', 'regime' columns
        long_regimes: List of regime IDs for long positions
        initial_capital: Starting capital amount
        commission_per_contract: Commission per contract in dollars
        margin_per_contract: Margin required per contract
    
    Returns:
        DataFrame with strategy performance including costs
    """
    df = df.copy().sort_values("date")
    df["ret"] = df["price"].pct_change().fillna(0.0)
    
    # Calculate number of contracts based on capital and margin
    max_contracts = int(initial_capital / margin_per_contract)
    contracts = min(max_contracts, 1)  # Start with 1 contract for simplicity
    
    # Binary allocation weights (0 or 1 for contract position)
    df["w"] = df["regime"].isin(long_regimes).astype(float)
    
    # Calculate position changes (when weight changes)
    df["w_prev"] = df["w"].shift(1).fillna(0.0)
    df["position_change"] = df["w"] - df["w_prev"]
    
    # Calculate strategy returns before costs
    df["strat_ret_before_costs"] = df["w"] * df["ret"]
    
    # Calculate trading costs (fixed commission per contract)
    df["trading_cost"] = (df["position_change"] != 0).astype(float) * commission_per_contract * contracts
    
    # Net strategy returns (after costs)
    # Convert fixed costs to percentage of current equity
    df["trading_cost_pct"] = df["trading_cost"] / (initial_capital * (1 + df["strat_ret_before_costs"]).cumprod())
    df["strat_ret"] = df["strat_ret_before_costs"] - df["trading_cost_pct"]
    
    # Calculate equity curves
    df["strat_eq"] = initial_capital * (1 + df["strat_ret"]).cumprod()
    df["buyhold_eq"] = initial_capital * (1 + df["ret"]).cumprod()
    
    # Calculate cumulative trading costs in dollars
    df["cumulative_costs"] = df["trading_cost"].cumsum()
    
    # Calculate position value
    df["position_value"] = df["w"] * df["strat_eq"]
    df["cash_value"] = (1 - df["w"]) * df["strat_eq"]
    
    # Add contract information
    df["contracts"] = contracts
    df["margin_used"] = contracts * margin_per_contract
    
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

def _calculate_advanced_metrics(strat_ret: pd.Series, bench_ret: pd.Series, 
                               trading_costs: pd.Series = None, 
                               initial_capital: float = 100000.0,
                               backtest_type: str = "etf") -> dict:
    """Calculate advanced performance metrics including trading costs."""
    # Basic metrics
    strat_cagr, strat_mdd, strat_final = _kpis((1 + strat_ret).cumprod())
    bench_cagr, bench_mdd, bench_final = _kpis((1 + bench_ret).cumprod())
    
    # Additional metrics
    strat_vol = strat_ret.std() * (252 ** 0.5)  # Annualized volatility
    bench_vol = bench_ret.std() * (252 ** 0.5)
    
    # Sharpe ratio (assuming 0% risk-free rate)
    strat_sharpe = strat_ret.mean() / strat_ret.std() * (252 ** 0.5) if strat_ret.std() > 0 else 0
    bench_sharpe = bench_ret.mean() / bench_ret.std() * (252 ** 0.5) if bench_ret.std() > 0 else 0
    
    # Calmar ratio
    strat_calmar = strat_cagr / abs(strat_mdd) if strat_mdd != 0 else 0
    bench_calmar = bench_cagr / abs(bench_mdd) if bench_mdd != 0 else 0
    
    # Win rate
    strat_win_rate = (strat_ret > 0).mean()
    bench_win_rate = (bench_ret > 0).mean()
    
    # Trading cost metrics
    cost_metrics = {}
    if trading_costs is not None:
        if backtest_type == "etf":
            # ETF: costs are in percentage, convert to dollars
            total_costs = trading_costs.sum() * initial_capital
            final_equity_dollars = strat_final * initial_capital
            cost_ratio = total_costs / final_equity_dollars if final_equity_dollars > 0 else 0
            avg_daily_cost = trading_costs.mean() * initial_capital
        else:  # futures
            # Futures: costs are already in dollars
            total_costs = trading_costs.sum()
            final_equity_dollars = strat_final * initial_capital
            cost_ratio = total_costs / final_equity_dollars if final_equity_dollars > 0 else 0
            avg_daily_cost = trading_costs.mean()
        
        trades_count = (trading_costs > 0).sum()
        cost_metrics = {
            'total_costs': total_costs,
            'cost_ratio': cost_ratio,
            'avg_daily_cost': avg_daily_cost,
            'cost_per_trade': trading_costs[trading_costs > 0].mean() if trades_count > 0 else 0,
            'total_trades': trades_count
        }
    
    return {
        'strategy': {
            'cagr': strat_cagr,
            'max_dd': strat_mdd,
            'final_equity': strat_final,
            'volatility': strat_vol,
            'sharpe': strat_sharpe,
            'calmar': strat_calmar,
            'win_rate': strat_win_rate,
            **cost_metrics
        },
        'benchmark': {
            'cagr': bench_cagr,
            'max_dd': bench_mdd,
            'final_equity': bench_final,
            'volatility': bench_vol,
            'sharpe': bench_sharpe,
            'calmar': bench_calmar,
            'win_rate': bench_win_rate
        }
    }

# ---------- page ----------
def main():
    st.title("🧪 Backtest")

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
        key="backtest_model_selector",
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

    # Regime selection (common for both strategies)
    st.header("🎯 Regime Selection")
    regimes = sorted(df["regime"].dropna().unique().tolist())
    default_long = [r for r in regimes if r != 0] or regimes  # default: tutti tranne 0
    
    col1, col2 = st.columns([2, 1])
    with col1:
        chosen = st.multiselect(
            "Long Regimes", 
            regimes, 
            default=default_long,
            help="Select which regimes should trigger a long position (100% invested). Other regimes will be cash (0% invested)."
        )
    
    with col2:
        st.info(f"**Available Regimes:** {regimes}")
        if not chosen:
            st.warning("⚠️ No regimes selected. Strategy will be 100% cash.")

    # Date range filter
    st.header("📅 Date Range Filter")
    col1, col2 = st.columns([3, 1])
    with col2:
        min_d, max_d = df["date"].min(), df["date"].max()
        date_range = st.date_input("Date range", (min_d.date(), max_d.date()))
    
    if isinstance(date_range, tuple):
        df = df[(df["date"] >= pd.Timestamp(date_range[0])) & (df["date"] <= pd.Timestamp(date_range[1]))]

    if df.empty:
        st.warning("⚠️ No data available for the selected date range.")
        return

    # Create tabs for ETF vs Futures
    tab1, tab2 = st.tabs(["📈 ETF Strategy", "⚡ Futures ES Strategy"])
    
    # ETF Strategy Tab
    with tab1:
        st.header("📈 ETF Strategy")
        st.info("Simulates trading an ETF (like SPY) with TER (Total Expense Ratio) applied annually")
        
        # ETF Configuration
        st.subheader("⚙️ ETF Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            etf_capital = st.number_input(
                "Initial Capital ($)",
                min_value=1000.0,
                max_value=10000000.0,
                value=100000.0,
                step=10000.0,
                help="Starting capital amount for ETF strategy",
                key="etf_capital"
            )
        
        with col2:
            etf_ter_rate = st.number_input(
                "TER Rate (%)",
                min_value=0.0,
                max_value=1.0,
                value=0.09,
                step=0.01,
                format="%.3f",
                help="Annual TER (Total Expense Ratio) - e.g., 0.09% for SPY",
                key="etf_ter"
            ) / 100.0  # Convert to decimal
        
        # Run ETF backtest
        if st.button("🚀 Run ETF Backtest", key="etf_button"):
            with st.spinner("Running ETF backtest..."):
                etf_res = _bt_etf(df.set_index("date"), chosen, etf_capital, etf_ter_rate)
                
                # Calculate metrics
                etf_metrics = _calculate_advanced_metrics(
                    etf_res["strat_ret"], 
                    etf_res["ret"], 
                    etf_res["trading_cost"],
                    etf_capital,
                    "etf"
                )
                
                # Display ETF results
                _display_backtest_results(etf_res, etf_metrics, "ETF", etf_capital, etf_ter_rate*100, chosen)
    
    # Futures Strategy Tab
    with tab2:
        st.header("⚡ Futures ES Strategy")
        st.info("Simulates trading ES futures contracts with fixed commissions per contract")
        
        # Futures Configuration
        st.subheader("⚙️ Futures Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            futures_capital = st.number_input(
                "Initial Capital ($)",
                min_value=14000.0,
                max_value=10000000.0,
                value=100000.0,
                step=10000.0,
                help="Starting capital amount for futures strategy",
                key="futures_capital"
            )
        
        with col2:
            futures_commission = st.number_input(
                "Commission per Contract ($)",
                min_value=0.0,
                max_value=20.0,
                value=2.50,
                step=0.25,
                format="%.2f",
                help="Fixed commission per contract",
                key="futures_commission"
            )
        
        with col3:
            margin_per_contract = st.number_input(
                "Margin per Contract ($)",
                min_value=5000.0,
                max_value=50000.0,
                value=14000.0,
                step=1000.0,
                help="Margin required per ES contract",
                key="futures_margin"
            )
        
        # Calculate max contracts
        max_contracts = int(futures_capital / margin_per_contract)
        st.info(f"💰 **Maximum Contracts:** {max_contracts} (based on capital and margin)")
        
        # Run Futures backtest
        if st.button("🚀 Run Futures Backtest", key="futures_button"):
            with st.spinner("Running futures backtest..."):
                futures_res = _bt_futures(df.set_index("date"), chosen, futures_capital, futures_commission, margin_per_contract)
                
                # Calculate metrics
                futures_metrics = _calculate_advanced_metrics(
                    futures_res["strat_ret"], 
                    futures_res["ret"], 
                    futures_res["trading_cost"],
                    futures_capital,
                    "futures"
                )
                
                # Display Futures results
                _display_backtest_results(futures_res, futures_metrics, "Futures", futures_capital, futures_commission, chosen)

def _display_backtest_results(res: pd.DataFrame, metrics: dict, strategy_type: str, capital: float, commission: float, chosen: list):
    """Display backtest results in a standardized format."""
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Strategy**")
        st.metric("CAGR", f"{metrics['strategy']['cagr']*100:.2f}%")
        st.metric("Max Drawdown", f"{metrics['strategy']['max_dd']*100:.2f}%")
        st.metric("Final Equity", f"{metrics['strategy']['final_equity']:.2f}")
        st.metric("Sharpe Ratio", f"{metrics['strategy']['sharpe']:.2f}")
        st.metric("Calmar Ratio", f"{metrics['strategy']['calmar']:.2f}")
        st.metric("Volatility", f"{metrics['strategy']['volatility']*100:.2f}%")
        st.metric("Win Rate", f"{metrics['strategy']['win_rate']*100:.2f}%")
    
    with col2:
        st.markdown("**Benchmark (Buy & Hold)**")
        st.metric("CAGR", f"{metrics['benchmark']['cagr']*100:.2f}%")
        st.metric("Max Drawdown", f"{metrics['benchmark']['max_dd']*100:.2f}%")
        st.metric("Final Equity", f"{metrics['benchmark']['final_equity']:.2f}")
        st.metric("Sharpe Ratio", f"{metrics['benchmark']['sharpe']:.2f}")
        st.metric("Calmar Ratio", f"{metrics['benchmark']['calmar']:.2f}")
        st.metric("Volatility", f"{metrics['benchmark']['volatility']*100:.2f}%")
        st.metric("Win Rate", f"{metrics['benchmark']['win_rate']*100:.2f}%")
    
    with col3:
        st.markdown("**Trading Costs**")
        if 'total_costs' in metrics['strategy']:
            st.metric("Total Costs", f"${metrics['strategy']['total_costs']:,.2f}")
            st.metric("Cost Ratio", f"{metrics['strategy']['cost_ratio']*100:.2f}%")
            st.metric("Total Trades", f"{metrics['strategy']['total_trades']:,.0f}")
            st.metric("Cost per Trade", f"${metrics['strategy']['cost_per_trade']:,.2f}")
        else:
            st.info("No trading costs calculated")

    # Visualizations
    st.subheader("📈 Equity Curves")
    st.line_chart(res[["strat_eq", "buyhold_eq"]])

    # Trading costs analysis
    st.subheader("💰 Trading Costs Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.line_chart(res["cumulative_costs"])
        st.caption("Cumulative Trading Costs Over Time")
    
    with col2:
        st.line_chart(res[["position_value", "cash_value"]])
        st.caption("Position vs Cash Allocation")
    
    # Regime analysis
    st.subheader("🎯 Regime Analysis")
    
    # Calculate regime statistics
    regime_stats = res.groupby("regime").agg({
        "ret": ["count", "mean", "std"],
        "w": "mean",
        "trading_cost": ["sum", "mean", "count"]
    }).round(4)
    
    regime_stats.columns = [
        "Days", "Avg Return", "Volatility", "Avg Weight",
        "Total Costs ($)", "Avg Daily Cost ($)", "Trade Days"
    ]
    
    st.dataframe(regime_stats)
    
    # Trading frequency analysis
    st.subheader("📊 Trading Frequency")
    
    col1, col2 = st.columns(2)
    
    with col1:
        trades_per_regime = res.groupby("regime")["trading_cost"].apply(lambda x: (x > 0).sum())
        st.bar_chart(trades_per_regime)
        st.caption("Number of Trades per Regime")
    
    with col2:
        monthly_trades = res.set_index("date").resample("M")["trading_cost"].apply(lambda x: (x > 0).sum())
        st.line_chart(monthly_trades)
        st.caption("Monthly Trading Frequency")
    
    # Download results
    st.subheader("💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = res.reset_index().to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv_data,
            file_name=f"backtest_{strategy_type.lower()}.csv",
            mime="text/csv"
        )
    
    with col2:
        total_trades = (res["position_change"] != 0).sum()
        if strategy_type == "ETF":
            st.info(f"""
            **Strategy:** {strategy_type}  
            **Period:** {len(res)} days  
            **Long Regimes:** {chosen}  
            **Total Trades:** {total_trades}  
            **Capital:** ${capital:,.0f}  
            **TER:** {commission:.3f}% annually
            """)
        else:  # Futures
            st.info(f"""
            **Strategy:** {strategy_type}  
            **Period:** {len(res)} days  
            **Long Regimes:** {chosen}  
            **Total Trades:** {total_trades}  
            **Capital:** ${capital:,.0f}  
            **Commission:** ${commission:.2f} per contract
            """)

if __name__ == "__main__":
    main()
