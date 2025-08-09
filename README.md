#  Market Regime Detector

The **Market Regime Detector** is a Python project designed to identify and monitor **market regimes** (e.g., Risk-On, Risk-Off) using macro-financial data and unsupervised machine learning (**K-Means Clustering**).

Its goal is to provide a robust tool to:
- Classify the current market context in near real-time.
- Analyze regime transitions and their duration.
- Integrate regime detection into trading strategies or asset allocation models.

---

##  Motivation

Markets don’t move randomly — they transition through **distinct phases** driven by macroeconomic conditions, monetary policy, and investor sentiment.  
Identifying the current regime is essential for:
- Adjusting **position sizing**.
- Selecting the most suitable asset classes and sectors.
- Avoiding large drawdowns during high-risk phases.

### Main regimes in the current model
The K-Means model currently identifies 3 main clusters:
1. **Risk-Off**  
   - Negative SPX returns, high volatility, elevated VIX.
   - Typical of crises, sell-offs, and aggressive monetary tightening.
2. **Risk-On (Steepening)**  
   - Rising equities, low volatility, positively sloped 10y–2y yield curve.
   - Typical of early-cycle recoveries after recessions.
3. **Risk-On (Inverted Curve)**  
   - Rising equities, low volatility, inverted 10y–2y yield curve.
   - Typical of late-cycle rallies under restrictive monetary policy.

---

##  Project Structure

```plaintext
market_regime_detector/
│
├── data/
│   ├── raw/               # Raw data downloaded from Yahoo Finance / FRED
│   ├── processed/         # Cleaned, merged market panel and features
│   └── external/          # Additional optional datasets
│
├── notebooks/             # Exploratory analysis & prototyping
│
├── scripts/               # CLI scripts
│   ├── fetch_data.py      # Download and merge data from Yahoo & FRED
│   ├── build_features.py  # Compute features (returns, volatility, slope, momentum)
│   ├── run_kmeans.py      # Run K-Means clustering and assign regimes
│   ├── plot_regimes.py    # Visualize price series with regime coloring
│   ├── export_regimes.py  # Export full regimes dataset to CSV
│   └── export_regime_transitions.py # Export regime change dates only
│
├── src/
│   ├── config.py          # Global path and folder configuration
│   ├── data/
│   │   ├── yahoo.py       # Yahoo Finance data fetching
│   │   ├── fred.py        # FRED macro data fetching
│   │   ├── utils.py       # Utilities for merging and validation
│   │   └── ingest.py      # Data ingestion and merging pipeline
│   ├── features/          # Feature engineering
│   │   └── features.py
│   ├── models/
│   │   └── kmeans.py      # Clustering and labeling logic
│   └── viz/
│       └── plots.py       # Visualization utilities
│
└── README.md
```

## End-to-End Workflow

### Fetch market and macro data  
Download historical series from:  
- **Yahoo Finance**: SPX (^GSPC), VIX (^VIX), TLT.  
- **FRED**: DGS10 (US 10Y), DGS2 (US 2Y).  

```bash
python scripts/fetch_data.py
```

---

### Feature Engineering  
Compute derived indicators:  
- SPX 1d and 5d returns.  
- 20-day rolling volatility.  
- VIX level.  
- 10y–2y yield curve slope.  
- 20-day momentum of TLT.  

```bash
python scripts/build_features.py
```

---

### K-Means Clustering  
Segment the dataset into clusters (market regimes).  

```bash
python scripts/run_kmeans.py
```

---

### Visualization  
Plot SPX with regimes colored according to clustering.  

```bash
python scripts/plot_regimes.py
```

---

### Export Final Regimes Dataset  

**Full dataset** (date, price, regime id, regime name):  
```bash
python scripts/export_regimes.py
```

**Only regime change dates**:  
```bash
python scripts/export_regime_transitions.py
```

---

##  Data Sources

**Yahoo Finance** — Daily OHLC and adjusted close prices for:  
- S&P 500 Index (^GSPC)  
- CBOE Volatility Index (^VIX)  
- iShares 20+ Year Treasury Bond ETF (TLT)  

**FRED** (Federal Reserve Economic Data) — Daily macro series:  
- DGS10: 10-Year Treasury Constant Maturity Rate  
- DGS2: 2-Year Treasury Constant Maturity Rate  

---

##  Methodology

### Data Ingestion
- Fetch and merge market + macro series on a unified trading calendar.  
- Forward-fill missing macro values for non-trading days.  

---

### Feature Engineering
- Return, volatility, momentum, yield curve slope.  
- Normalization using z-scores before clustering.  

---

### Unsupervised Clustering
- **K-Means** with configurable `n_clusters` (default: 3).  
- Cluster centers analyzed and mapped to human-readable regime names.  

---

### Interpretation
- Regime assignment based on market and macro conditions.  
- *Risk-On* split into **Steepening** vs **Inverted** cycles.  

---

##  Example Output
| date       | price   | regime | regime_name               |
|------------|---------|--------|---------------------------|
| 2015-01-26 | 2057.09 | 1      | Risk-On (Steepening)       |
| 2015-01-27 | 2050.03 | 0      | Risk-Off                   |
| 2015-01-28 | 2029.55 | 2      | Risk-On (Inverted Curve)   |

---

##  Potential Extensions
- Add more macro features: credit spreads (HYG–LQD), DXY, MOVE index.  
- Try other clustering algorithms (Gaussian Mixture Models, HMM).  
- Integrate with trading systems for regime-aware allocation.  
- Backtest performance of different strategies per regime.  

---

##  License
This project is released under the **MIT License**.

