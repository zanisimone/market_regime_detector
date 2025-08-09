# =========
# Variables
# =========
PYTHON ?= python

# Paths
PANEL      ?= data/processed/market_panel.parquet
FEATURES   ?= data/processed/features.parquet
PROC_DIR   ?= data/processed
LABELS     ?= $(PROC_DIR)/kmeans_labels.parquet

# Common params
PRICE_COL        ?= SPX
K_CLUSTERS       ?= 3
RANDOM_STATE     ?= 42

# Split dates
TRAIN_START ?= 2010-01-01
TRAIN_END   ?= 2018-12-31
TEST_START  ?= 2019-01-01
TEST_END    ?= 2024-12-31

# Rolling params (business days)
ROLL_START     ?= 2010-01-01
ROLL_END       ?= 2024-12-31
LOOKBACK_DAYS  ?= 504
OOS_DAYS       ?= 21
STEP_DAYS      ?= 21

# =========
# Pipelines
# =========

# Features (rolling z-score 252)
features:
	$(PYTHON) scripts/build_features.py --panel $(PANEL) --out $(FEATURES) --standardize rolling --rolling-window 252

# KMeans full-sample + report
kmeans:
	$(PYTHON) scripts/run_kmeans.py --features $(FEATURES) --labels-out $(LABELS) --centers-out $(PROC_DIR)/kmeans_centers.csv --k $(K_CLUSTERS)

report:
	$(PYTHON) scripts/report_regime_stats.py --panel $(PANEL) --labels $(LABELS) --out-prefix $(PROC_DIR)/regime_report --price-col $(PRICE_COL)

# All-in-one: features + kmeans + report
all: features kmeans report

# Export daily regimes
export-daily:
	$(PYTHON) scripts/export_regimes_daily.py --panel $(PANEL) --labels $(LABELS) --price-col $(PRICE_COL)

# =========
# KMEANS: Split & Rolling
# =========
split:
	$(PYTHON) scripts/run_kmeans_split.py \
		--features $(FEATURES) \
		--panel $(PANEL) \
		--out-dir $(PROC_DIR)/kmeans_split \
		--train-start $(TRAIN_START) --train-end $(TRAIN_END) \
		--test-start $(TEST_START) --test-end $(TEST_END) \
		--k $(K_CLUSTERS) --price-col $(PRICE_COL) --random-state $(RANDOM_STATE)

rolling:
	$(PYTHON) scripts/run_kmeans_rolling.py \
		--features $(FEATURES) \
		--panel $(PANEL) \
		--out-dir $(PROC_DIR)/kmeans_rolling \
		--start $(ROLL_START) --end $(ROLL_END) \
		--lookback-days $(LOOKBACK_DAYS) --oos-days $(OOS_DAYS) --step-days $(STEP_DAYS) \
		--k $(K_CLUSTERS) --price-col $(PRICE_COL) --random-state $(RANDOM_STATE)

# =========
# GMM: Split & Rolling
# =========
gmm-split:
	$(PYTHON) scripts/run_gmm_split.py \
		--features $(FEATURES) \
		--panel $(PANEL) \
		--out-dir $(PROC_DIR)/gmm_split \
		--train-start $(TRAIN_START) --train-end $(TRAIN_END) \
		--test-start $(TEST_START) --test-end $(TEST_END) \
		--k $(K_CLUSTERS) --price-col $(PRICE_COL) --random-state $(RANDOM_STATE) \
		--covariance-type full --n-init 5

gmm-rolling:
	$(PYTHON) scripts/run_gmm_rolling.py \
		--features $(FEATURES) \
		--panel $(PANEL) \
		--out-dir $(PROC_DIR)/gmm_rolling \
		--start $(ROLL_START) --end $(ROLL_END) \
		--lookback-days $(LOOKBACK_DAYS) --oos-days $(OOS_DAYS) --step-days $(STEP_DAYS) \
		--k $(K_CLUSTERS) --price-col $(PRICE_COL) --random-state $(RANDOM_STATE) \
		--covariance-type full --n-init 5

# =========
# HMM: Split & Rolling
# =========
hmm-split:
	$(PYTHON) scripts/run_hmm_split.py \
		--features $(FEATURES) \
		--panel $(PANEL) \
		--out-dir $(PROC_DIR)/hmm_split \
		--train-start $(TRAIN_START) --train-end $(TRAIN_END) \
		--test-start $(TEST_START) --test-end $(TEST_END) \
		--k $(K_CLUSTERS) --price-col $(PRICE_COL) --random-state $(RANDOM_STATE) \
		--covariance-type full --n-iter 200

hmm-rolling:
	$(PYTHON) scripts/run_hmm_rolling.py \
		--features $(FEATURES) \
		--panel $(PANEL) \
		--out-dir $(PROC_DIR)/hmm_rolling \
		--start $(ROLL_START) --end $(ROLL_END) \
		--lookback-days $(LOOKBACK_DAYS) --oos-days $(OOS_DAYS) --step-days $(STEP_DAYS) \
		--k $(K_CLUSTERS) --price-col $(PRICE_COL) --random-state $(RANDOM_STATE) \
		--covariance-type full --n-iter 200
