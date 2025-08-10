# Market Regime Detector

The **Market Regime Detector** is a Python framework to **identify, analyze, and act on market regimes** (e.g., Risk-On, Risk-Off) using macro‑financial features and unsupervised models.  
It supports multiple algorithms and training modes out of the box:

- **Algorithms:** **K‑Means**, **Gaussian Mixture Models (GMM)**, **Hidden Markov Models (HMM)**.
- **Training modes:** **Split** (fit once on train, label full sample) and **Rolling** (refit on a moving window for time‑adaptive regimes).

Beyond clustering, the framework includes:

- **Regime transition analysis:** transition matrices, stationary distributions, dwell‑time distributions, and stability KPIs.
- **Early‑warning signals:** heuristics on rising Risk‑Off probability.
- **Reporting & diagnostics:** model agreement matrices, per‑regime normalized price path composites, and a feature‑contribution explorer for cluster centers.
- **Strategy hooks & backtesting:** simple allocation rules (e.g., Risk‑On → long SPX; Risk‑Off → cash/TLT), vectorized backtests with buy‑and‑hold benchmark, and performance reporting.

**Goals**
- Classify the current market context in near real time.
- Measure regime stability and transition risks across models/splits.
- Generate actionable allocation signals.
- Benchmark regime‑aware strategies versus buy‑and‑hold.


---

## Streamlit App

An interactive dashboard is included to explore **data**, **features**, **regimes**, **KPIs**, and **backtests** end-to-end.

### App pages (sidebar)
- **Home** — overview, run status, quick links.
- **1 · Data & Features** — load merged market/macro data, run feature engineering (`build_features`) directly in the app, inspect panels & distributions.
- **2 · Clustering** — run and visualize regime detection models.
- **3 · Regime Explorer** — inspect regime sequences, transitions, and statistics.
- **4 · Backtest** — apply allocation rules by regime, run a vectorized backtest, and compare against benchmarks.

---

### **MVP Limitations**
- **Data fetching is not available in the app** — you must run:
  ```bash
  python scripts/fetch_data.py
  ```
  
before launching Streamlit.

Feature engineering (`build_features.py`) can be run from inside the app (page **1 · Data & Features**).

Only **K-Means Split** is currently supported for in-app model execution.

---

### How to launch
```bash
# from the repository root
streamlit run app/app.py
# or open a specific page
streamlit run app/pages/4_Backtest.py
```

### App prerequisites
- Ensure you have run `fetch_data.py` so that `data/processed/` contains the required market/macro dataset.
- Generate at least one **labels** file (e.g., `reports/labels_kmeans_split.csv`) to unlock Regime Explorer and Backtest tabs.

### Optional configuration
Create a `.env` in repo root if you need credentials/keys (e.g., FRED API) or want to override defaults:
```bash
FRED_API_KEY=xxxxxxxxxxxxxxxx
APP_DATA_DIR=data
APP_REPORTS_DIR=reports
```

### Troubleshooting
- **ModuleNotFoundError: No module named 'src'**  
  Set `PYTHONPATH` to project root before launching:
  - PowerShell: `$env:PYTHONPATH=(Get-Location)`
  - bash/zsh: `export PYTHONPATH=$(pwd)`

---

## Motivation

Markets evolve through **distinct phases** shaped by macroeconomic conditions, monetary policy, and investor sentiment — they are not random walks.  
Identifying the current regime provides an edge for traders, portfolio managers, and researchers by enabling them to:

- Adjust **position sizing** and risk exposure dynamically.
- Select the most suitable asset classes, sectors, or strategies.
- Avoid significant drawdowns during high-risk phases.
- Capture upside in favorable market environments.

With the extended framework, you can now:

- **Choose your regime model:** K-Means, Gaussian Mixture Models (GMM), or Hidden Markov Models (HMM), in both split and rolling training modes.
- **Quantify** regime stability, transition probabilities, and dwell-time distributions.
- **Monitor** early-warning signals for deteriorating market conditions.
- **Diagnose** and compare model outputs with agreement matrices, composites, and feature contribution analysis.
- **Implement** simple allocation rules and **backtest** them against benchmarks in a vectorized, fast workflow.

---

## Supported Models & Training Modes

The framework supports three unsupervised learning algorithms for market regime detection:

### 1. K-Means
- **Type:** Hard clustering, distance-based.
- **Mechanics:** Partitions feature space into `n_clusters` by minimizing within-cluster variance.
- **Pros:** Fast, simple, interpretable centers; works well when clusters are roughly spherical in feature space.
- **Cons:** Assumes equal variance and similar cluster size; no probabilistic output.

### 2. Gaussian Mixture Models (GMM)
- **Type:** Soft clustering, probabilistic.
- **Mechanics:** Models data as a mixture of Gaussian distributions; assigns each point a probability of belonging to each regime.
- **Pros:** Flexible cluster shapes; provides regime probabilities; can capture overlapping regimes.
- **Cons:** More computationally intensive; can overfit with small sample sizes or many features.

### 3. Hidden Markov Models (HMM)
- **Type:** Probabilistic sequence model.
- **Mechanics:** Assumes the market moves through hidden states (regimes) with Markov transition probabilities; emits observed features from state-specific distributions.
- **Pros:** Captures temporal dependencies; explicit transition probabilities; well-suited for regime persistence analysis.
- **Cons:** Higher complexity; requires careful initialization; sensitive to model specification.

---

### Training Modes

**Split Mode**
- Fit the model once on a designated training window.
- Apply the trained model to label the entire dataset.
- Pros: Stable regime definitions; good for historical regime mapping.
- Cons: Does not adapt to structural changes after the training period.

**Rolling Mode**
- Refit the model on a moving window (e.g., last 2 years) at each step.
- Produces time-adaptive regimes that can capture evolving market structures.
- Pros: More responsive to new patterns; useful for live regime tracking.
- Cons: Higher computation; regime definitions can drift over time.
---

> **Current MVP limitation:**  
> From the Streamlit app, you can currently launch only the **K-Means Split** model.  
> Data fetching is **not** available directly in the app — you must run `scripts/fetch_data.py` and `scripts/build_features.py` before accessing the UI.

---

## Main regimes in the current model

The framework supports multiple algorithms — **K-Means**, **Gaussian Mixture Models (GMM)**, and **Hidden Markov Models (HMM)** — each capable of discovering market regimes from macro-financial features.  
The examples below come from the default **K-Means** configuration with three regimes, but results will vary depending on:

- The chosen algorithm and its hyperparameters.
- The training mode (**split** vs **rolling**).
- The feature set and preprocessing steps.

**Example with 3 regimes (K-Means default):**

1. **Risk-Off**  
   - Negative SPX returns, high volatility, elevated VIX.  
   - Typical of crises, sell-offs, and aggressive monetary tightening.

2. **Risk-On (Steepening)**  
   - Rising equities, low volatility, positively sloped 10y–2y yield curve.  
   - Typical of early-cycle recoveries after recessions.

3. **Risk-On (Inverted Curve)**  
   - Rising equities, low volatility, inverted 10y–2y yield curve.  
   - Typical of late-cycle rallies under restrictive monetary policy.

> Regime definitions are **model-dependent** — switching to GMM or HMM, or changing the number of clusters/states, will change the regime set and interpretation.


---

## Project Structure

```plaintext
## Project Structure

```plaintext
market_regime_detector/
│
├── app/                          # Streamlit dashboard
│   ├── app.py                     # Main Streamlit entrypoint
│   ├── utils.py                   # App-specific helper functions
│   ├── __init__.py
│   │
│   ├── components/                # Reusable UI components
│   │   ├── kpi_cards.py           # KPI display cards
│   │   ├── regime_info.py         # Info popups/descriptions for regimes
│   │   └── regime_timeline.py     # Timeline visualization of regimes
│   │
│   ├── configs/                   # App configuration files
│   │   ├── app.defaults.toml      # Default app parameters
│   │   └── regimes.json           # Human-readable regime definitions
│   │
│   └── pages/                     # Multi-page Streamlit app
│       ├── 1_Data_and_Features.py # Data inspection + run build_features
│       ├── 2_Clustering.py        # Model execution & clustering results (MVP: KMeans Split only)
│       ├── 3_Regime_Explorer.py   # Explore regime sequences, KPIs, transitions
│       └── 4_Backtest.py          # Regime-aware backtesting
│
├── data/
│   ├── raw/                       # Raw market/macro data from Yahoo, FRED, etc.
│   ├── processed/                 # Cleaned datasets & feature panels
│   └── external/                  # Optional external datasets (credit spreads, FX, etc.)
│
├── reports/                       # Outputs (labels, KPIs, composites, backtests)
│
├── notebooks/                     # Exploratory notebooks
│
├── scripts/                       # CLI scripts for pipeline steps
│   ├── fetch_data.py               # Download & merge market/macro data
│   ├── build_features.py           # Feature engineering (callable from app)
│   ├── run_kmeans_split.py         # K-Means Split mode (only model available in-app for MVP)
│   ├── run_kmeans_rolling.py       # K-Means Rolling mode
│   ├── run_gmm_split.py            # GMM Split mode
│   ├── run_gmm_rolling.py          # GMM Rolling mode
│   ├── run_hmm_split.py            # HMM Split mode
│   ├── run_hmm_rolling.py          # HMM Rolling mode
│   ├── plot_regimes.py             # Plot price series with regime coloring
│   ├── export_regimes.py           # Export regimes to CSV
│   ├── export_regime_transitions.py# Export regime change dates
│   ├── run_transition_kpis.py      # Compute transition matrix, stationary dist, dwell times
│   ├── run_early_warning.py        # Early-warning signals for Risk-Off
│   ├── run_agreement.py            # Compare model/split agreement
│   ├── run_composites.py           # Per-regime price path composites
│   ├── run_center_explorer.py      # Inspect cluster centers & top features
│   └── run_allocation_backtest.py  # Regime-aware allocation backtest
│
├── src/                            # Core Python package
│   ├── config.py
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── regime/
│   ├── alerts/
│   ├── reporting/
│   ├── backtest/
│   ├── strategy/
│   └── viz/
│
└── README.md

```


---

## End-to-End Workflow

The framework can be run entirely from the CLI scripts or via `make` targets.  
It supports **K-Means**, **GMM**, and **HMM** models in both **split** and **rolling** modes.

---

### 1. Fetch Market and Macro Data
Download historical series from:
- **Yahoo Finance**: SPX (^GSPC), VIX (^VIX), TLT.
- **FRED**: DGS10 (US 10Y), DGS2 (US 2Y).

```bash
python scripts/fetch_data.py
# or
make fetch-data
```

---

### 2. Feature Engineering
Compute derived indicators such as:
- SPX 1d and 5d returns.
- 20-day rolling volatility.
- VIX level.
- 10y–2y yield curve slope.
- 20-day momentum of TLT.

```bash
python scripts/build_features.py
# or
make build-features
```

---

### 3. Choose and Run a Clustering Model
Select algorithm and training mode:

**K-Means:**
```bash
python scripts/run_kmeans_split.py --in data/processed/panel.parquet --out reports/labels_kmeans_split.csv --n_clusters 3
python scripts/run_kmeans_rolling.py --in data/processed/panel.parquet --out reports/labels_kmeans_roll.csv --n_clusters 3
```

**GMM:**
```bash
python scripts/run_gmm_split.py --in data/processed/panel.parquet --out reports/labels_gmm_split.csv --n_clusters 3
python scripts/run_gmm_rolling.py --in data/processed/panel.parquet --out reports/labels_gmm_roll.csv --n_clusters 3
```

**HMM:**
```bash
python scripts/run_hmm_split.py --in data/processed/panel.parquet --out reports/labels_hmm_split.csv --n_states 3
python scripts/run_hmm_rolling.py --in data/processed/panel.parquet --out reports/labels_hmm_roll.csv --n_states 3
```

---

### 4. Transition & Stability KPIs
Compute:
- Transition matrix.
- Stationary distribution.
- Dwell-time statistics.
- Stability metrics across splits/models.

```bash
python scripts/run_transition_kpis.py --labels_csv reports/labels_kmeans_split.csv --out_csv reports/transition_kpis.csv
# or
make transition-kpis LABELS="reports/labels_kmeans_split.csv" OUT=reports/transition_kpis.csv
```

---

### 5. Early-Warning Signals
Generate early-warning heuristics for rising Risk-Off probability.

```bash
python scripts/run_early_warning.py --prob_csv data/risk_off_prob.csv --out_csv reports/early_warning.csv
# or
make early-warning PROB=data/risk_off_prob.csv OUT=reports/early_warning.csv
```

---

### 6. Reporting & Diagnostics
- **Agreement**: Compare regime labels between models/splits.
- **Composites**: Build per-regime normalized price path composites.
- **Center Explorer**: Inspect cluster centers and top contributing features.

```bash
make agreement LABELS="reports/labels_kmeans_split.csv reports/labels_gmm_split.csv" NAMES="KMeans GMM" OUT=reports/agreement
make composites PRICE=data/px.csv LABELS=reports/labels_kmeans_split.csv REGIME=0 OUT=reports/composite_r0.csv
make center-explorer CENTERS=reports/kmeans_centers.csv FEATCOLS="ret_1d,vol_21d" OUT=reports/centers
```

---

### 7. Strategy Hooks & Backtest
Apply a simple allocation rule based on regimes and compare against a benchmark.

```bash
python scripts/run_allocation_backtest.py \
  --prices_csv data/prices.csv --assets "SPX,TLT" --cash CASH \
  --labels_csv reports/labels_kmeans_split.csv --risk_on 1 --risk_off 0 \
  --on_weights "SPX:1.0" --off_weights "TLT:1.0" \
  --benchmark SPX --out_prefix reports/alloc_simple
# or
make allocation-backtest PRICES=data/prices.csv ASSETS="SPX,TLT" LABELS=reports/labels_kmeans_split.csv \
  RISK_ON=1 RISK_OFF=0 ON_W="SPX:1.0" OFF_W="TLT:1.0" BENCH=SPX OUT=reports/alloc_simple
```

---

## Data Sources

The framework is designed to work with freely available macro-financial datasets and can be extended to custom data providers.

**Yahoo Finance** — Daily OHLC and adjusted close prices for:
- S&P 500 Index (^GSPC)  
- CBOE Volatility Index (^VIX)  
- iShares 20+ Year Treasury Bond ETF (TLT)  
- Any additional tickers specified in the configuration

**FRED** (Federal Reserve Economic Data) — Daily macro series:
- DGS10: 10-Year Treasury Constant Maturity Rate  
- DGS2: 2-Year Treasury Constant Maturity Rate  
- Optional: credit spreads (e.g., HYG–LQD), USD Index (DXY), MOVE Index

**Custom / External Datasets** — Placed in `data/external/`:
- Proprietary macro or market features
- Alternative data (sentiment indices, commodity spreads, etc.)

> All series are aligned to a unified trading calendar with forward-fill for macro data on non-trading days.


---

## Methodology

The Market Regime Detector follows a modular pipeline — from raw data ingestion to actionable allocation signals — supporting multiple algorithms (**K-Means**, **GMM**, **HMM**) and training modes (**split** and **rolling**).

---

### Data Ingestion
- Fetch and merge market + macro series on a unified trading calendar.
- Forward-fill missing macro values for non-trading days.
- Support for Yahoo Finance, FRED, and external custom datasets.

---

### Feature Engineering
- Core features: returns, rolling volatility, momentum, yield curve slope.
- Optional macro & alternative features: credit spreads, DXY, MOVE index.
- Normalization using z-scores before clustering.
- Flexible feature configuration for different model runs.

---

### Unsupervised Modeling
- **K-Means**, **Gaussian Mixture Models (GMM)**, or **Hidden Markov Models (HMM)**.
- Configurable number of regimes/clusters/states.
- Two training modes:
  - **Split:** Fit once on a training window and label the full dataset.
  - **Rolling:** Refit on a moving window for time-adaptive regime detection.

---

### Regime Transition Mechanics
- First-order Markov transition matrix estimation.
- Stationary distribution and dwell-time distribution per regime.
- Stability KPIs across splits/models (spectral gap, Frobenius dispersion).

---

### Early-Warning Signals
- Rolling probability metrics for Risk-Off regime.
- Multi-trigger heuristics: MA level, z-score deviation, slope acceleration.
- Cool-off logic to avoid clustered false positives.

---

### Reporting & Diagnostics
- **Agreement analysis:** Confusion-like comparisons between model outputs.
- **Price path composites:** Normalized per-regime composites around start dates.
- **Feature contribution explorer:** Top contributing features per regime center.

---

### Strategy Hooks & Backtesting
- Simple allocation rules mapping regimes to portfolio weights.
- Vectorized backtest engine with turnover calculation.
- Benchmark comparison (buy-and-hold).
- Performance report including CAGR, volatility, Sharpe ratio, max drawdown, turnover, hit rate, and per-regime stats.
 

---

## Example Output

### Regime Label Dataset
Full dataset with assigned regimes and names:
| date       | price   | regime | regime_name               |
|------------|---------|--------|---------------------------|
| 2015-01-26 | 2057.09 | 1      | Risk-On (Steepening)       |
| 2015-01-27 | 2050.03 | 0      | Risk-Off                   |
| 2015-01-28 | 2029.55 | 2      | Risk-On (Inverted Curve)   |

---

### Transition Matrix
Estimated from daily regime sequence (rows = from, cols = to):

| from\to | 0     | 1     | 2     |
|---------|-------|-------|-------|
| 0       | 0.85  | 0.10  | 0.05  |
| 1       | 0.08  | 0.86  | 0.06  |
| 2       | 0.12  | 0.08  | 0.80  |

**Stationary distribution:**  
Regime 0: 32%, Regime 1: 45%, Regime 2: 23%

---

### Early-Warning Signals
Rolling Risk-Off probability metrics and triggers:

| date       | prob  | ma    | slope | zscore | trigger_prob | trigger_z | trigger_slope | warning |
|------------|-------|-------|-------|--------|--------------|-----------|---------------|---------|
| 2020-02-18 | 0.15  | 0.14  | 0.00  | -0.45  | 0            | 0         | 0             | 0       |
| 2020-02-19 | 0.22  | 0.18  | 0.04  | -0.22  | 0            | 0         | 1             | 0       |
| 2020-02-20 | 0.35  | 0.24  | 0.06  |  0.15  | 0            | 0         | 1             | 0       |
| 2020-02-21 | 0.62  | 0.33  | 0.09  |  0.85  | 1            | 0         | 1             | 1       |

---

### Per-Regime Price Path Composite (SPX, normalized)
Average normalized SPX path around regime start events (Risk-Off):

| offset | mean   | median | p25    | p75    | n   |
|--------|--------|--------|--------|--------|-----|
| -5     |  0.004 |  0.003 | -0.002 |  0.009 |  15 |
|  0     |  0.000 |  0.000 |  0.000 |  0.000 |  15 |
|  5     | -0.012 | -0.010 | -0.020 | -0.004 |  15 |
| 10     | -0.025 | -0.022 | -0.035 | -0.012 |  15 |

---

### Allocation Backtest Report
**Portfolio vs Benchmark (SPX buy-and-hold)**

**Portfolio metrics:**
- CAGR: 8.7%
- Volatility: 10.3%
- Sharpe: 0.85
- Max Drawdown: -12.4%
- Turnover: 0.18

**Benchmark metrics:**
- CAGR: 7.5%
- Volatility: 14.2%
- Sharpe: 0.53
- Max Drawdown: -33.9%

**Hit rate by regime:**
| regime | hit_rate | mean_ret  | n_obs |
|--------|----------|-----------|-------|
| 0      | 0.42     | -0.0003   | 250   |
| 1      | 0.58     |  0.0005   | 400   |
| 2      | 0.54     |  0.0004   | 320   |


---

## Potential Extensions

With the current framework already supporting K-Means, GMM, HMM, transition analysis, early-warning signals, reporting, and regime-based backtesting, future development can focus on:

- **Additional algorithms**:
  - Spectral clustering, DBSCAN for non-linear regime boundaries.
  - Bayesian HMMs with regime duration modeling.
- **Feature expansion**:
  - Global macro features (commodities, FX, global indices).
  - Alternative data (news sentiment, liquidity metrics, options-implied signals).
- **Live integration**:
  - Streaming market data for near real-time regime updates.
  - Direct integration with trading APIs for automated regime-aware allocation.
- **Advanced backtesting**:
  - Multi-asset allocation with optimization constraints.
  - Transaction cost modeling and slippage simulation.
  - Walk-forward optimization with multiple OOS splits.
- **Visualization & dashboard**:
  - Interactive dashboard (e.g., Streamlit) for exploring regimes, KPIs, and backtest results.
- **Model monitoring**:
  - Drift detection in feature distributions.
  - Automatic alerts on regime probability shifts.
 

---

## License

This project is released under the **MIT License**.

You are free to use, modify, and distribute this software for any purpose, provided that the original copyright
notice and this permission notice are included in all copies or substantial portions of the software.



