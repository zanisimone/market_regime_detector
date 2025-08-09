# Variables
PY ?= python
SHELL := /bin/bash
SCRIPTS := scripts
DATA := data
REPORTS := reports

# Tools (override as needed)
BLACK ?= black
ISORT ?= isort
RUFF ?= ruff

.PHONY: help setup fmt lint test clean \
        run-kmeans-split transition-kpis early-warning \
        agreement composites center-explorer allocation-backtest \
        features features-core features-plus features-pca \
        validate validate-adf

help:
	@echo "Targets:"
	@echo "  setup                    Install project dependencies"
	@echo "  fmt                      Run code formatters (black, isort)"
	@echo "  lint                     Run static checks (ruff)"
	@echo "  run-kmeans-split         Run K-Means split pipeline"
	@echo "  transition-kpis          Compute transition matrix, stationary dist, dwell-time KPIs"
	@echo "  early-warning            Generate early-warning signals on Risk-Off probability"
	@echo "  agreement                Confusion-like comparisons and model agreement matrices"
	@echo "  composites               Per-regime normalized price path composites"
	@echo "  center-explorer          Cluster centers & feature contributions"
	@echo "  allocation-backtest      Regime-based allocation backtest + benchmark"
	@echo "  ------------------------------------------------------------"
	@echo "  features                 Build features (Step 1–3: core + proxies + transforms)"
	@echo "     VARS: PANEL, OUT, STD, WINDOW, BUNDLE, NOWINSOR=1, WQLOW, WQHIGH, PCA=1, PCA_WINDOW, PCA_REPLACE=1"
	@echo "  features-core            Build features with bundle=core"
	@echo "  features-plus            Build features with bundle=market_plus (default)"
	@echo "  features-pca            Build features with PCA appended"
	@echo "  validate                 Validation & monitoring mini-report (Step 4)"
	@echo "     VARS: FEATURES, OUTDIR, SPLIT_DATE, RATIO, ADF=1, ROLLWIN"
	@echo "  ------------------------------------------------------------"
	@echo "  test                     Run tests (if present)"
	@echo "  clean                    Remove caches and temp files"

setup:
	$(PY) -m pip install -U pip
	@if [ -f requirements.txt ]; then $(PY) -m pip install -r requirements.txt; fi

fmt:
	$(ISORT) .
	$(BLACK) .

lint:
	$(RUFF) check .

test:
	@if [ -d tests ]; then $(PY) -m pytest -q; else echo "No tests/ directory found."; fi

# ------------------------------
# Pipelines (existing)
# ------------------------------

run-kmeans-split:
	@if [ -z "$(IN)" ]; then echo "IN is required (input parquet/csv)"; exit 1; fi
	@if [ -z "$(OUT)" ]; then echo "OUT is required (output csv)"; exit 1; fi
	@if [ -z "$(K)" ]; then echo "K is required (n_clusters)"; exit 1; fi
	$(PY) $(SCRIPTS)/run_kmeans_split.py --in $(IN) --out $(OUT) --n_clusters $(K)

transition-kpis:
	@if [ -z "$(LABELS)" ]; then echo "LABELS is required (space-separated CSV paths)"; exit 1; fi
	@if [ -z "$(OUT)" ]; then echo "OUT is required (output csv)"; exit 1; fi
	@mkdir -p $(REPORTS)
	$(PY) $(SCRIPTS)/run_transition_kpis.py \
		$(foreach f,$(LABELS),--labels_csv $(f)) \
		--risk_off_label $(if $(RISK_OFF),$(RISK_OFF),0) \
		--out_csv $(OUT)

early-warning:
	@if [ -z "$(PROB)" ]; then echo "PROB is required (input csv with probability series)"; exit 1; fi
	@if [ -z "$(OUT)" ]; then echo "OUT is required (output csv)"; exit 1; fi
	@mkdir -p $(REPORTS)
	$(PY) $(SCRIPTS)/run_early_warning.py \
		--prob_csv $(PROB) \
		--out_csv $(OUT) \
		--prob_col $(if $(COL),$(COL),prob) \
		--ma_window $(if $(MA),$(MA),5) \
		--z_window $(if $(ZW),$(ZW),63) \
		--prob_threshold $(if $(PTH),$(PTH),0.60) \
		--z_threshold $(if $(ZTH),$(ZTH),1.0) \
		--slope_threshold $(if $(SLOPE),$(SLOPE),0.02) \
		--cool_off_days $(if $(COOL),$(COOL),5)

# ------------------------------
# Reporting & Diagnostics (existing)
# ------------------------------

agreement:
	@if [ -z "$(LABELS)" ]; then echo "LABELS is required (space-separated CSV paths)"; exit 1; fi
	@if [ -z "$(NAMES)" ]; then echo "NAMES is required (space-separated model names)"; exit 1; fi
	@if [ -z "$(OUT)" ]; then echo "OUT is required (output prefix)"; exit 1; fi
	@set -e; \
	 IFS=' ' read -r -a __LBL <<< "$(LABELS)"; \
	 IFS=' ' read -r -a __NAM <<< "$(NAMES)"; \
	 if [ "$${#__LBL[@]}" -ne "$${#__NAM[@]}" ]; then echo "LABELS and NAMES length mismatch"; exit 1; fi; \
	 CMD="$(PY) $(SCRIPTS)/run_agreement.py --out_prefix $(OUT)"; \
	 for i in $${!__LBL[@]}; do CMD="$$CMD --labels_csv $${__LBL[$$i]} --name $${__NAM[$$i]}"; done; \
	 if [ -n "$(AGG_LABELS)" ]; then CMD="$$CMD --labels $(AGG_LABELS)"; fi; \
	 eval "$$CMD"

composites:
	@if [ -z "$(PRICE)" ]; then echo "PRICE is required (price csv)"; exit 1; fi
	@if [ -z "$(LABELS)" ]; then echo "LABELS is required (labels csv)"; exit 1; fi
	@if [ -z "$(REGIME)" ]; then echo "REGIME is required (int)"; exit 1; fi
	@if [ -z "$(OUT)" ]; then echo "OUT is required (output csv)"; exit 1; fi
	$(PY) $(SCRIPTS)/run_composites.py \
		--price_csv $(PRICE) \
		--labels_csv $(LABELS) \
		--regime $(REGIME) \
		--lookback $(if $(LOOKBACK),$(LOOKBACK),5) \
		--lookahead $(if $(LOOKAHEAD),$(LOOKAHEAD),20) \
		--mode $(if $(MODE),$(MODE),rebased) \
		--min_run $(if $(MINRUN),$(MINRUN),3) \
		--out_csv $(OUT)

center-explorer:
	@if [ -z "$(CENTERS)" ]; then echo "CENTERS is required (centers csv)"; exit 1; fi
	@if [ -z "$(FEATCOLS)" ]; then echo "FEATCOLS is required (comma-separated feature names)"; exit 1; fi
	@if [ -z "$(OUT)" ]; then echo "OUT is required (output prefix)"; exit 1; fi
	$(PY) $(SCRIPTS)/run_center_explorer.py \
		--centers_csv $(CENTERS) \
		--feature_cols "$(FEATCOLS)" \
		$(if $(STATS),--means_stds_csv $(STATS),) \
		--top $(if $(TOP),$(TOP),10) \
		--out_prefix $(OUT)

# ------------------------------
# Strategy Hooks & Backtest (existing)
# ------------------------------

allocation-backtest:
	@if [ -z "$(PRICES)" ]; then echo "PRICES is required (wide prices csv)"; exit 1; fi
	@if [ -z "$(ASSETS)" ]; then echo "ASSETS is required (comma-separated)"; exit 1; fi
	@if [ -z "$(LABELS)" ]; then echo "LABELS is required (labels csv)"; exit 1; fi
	@if [ -z "$(RISK_ON)" ]; then echo "RISK_ON is required (int)"; exit 1; fi
	@if [ -z "$(RISK_OFF)" ]; then echo "RISK_OFF is required (int)"; exit 1; fi
	@if [ -z "$(ON_W)" ]; then echo "ON_W is required (ASSET:W pairs)"; exit 1; fi
	@if [ -z "$(BENCH)" ]; then echo "BENCH is required (benchmark asset col)"; exit 1; fi
	@if [ -z "$(OUT)" ]; then echo "OUT is required (output prefix)"; exit 1; fi
	$(PY) $(SCRIPTS)/run_allocation_backtest.py \
		--prices_csv $(PRICES) \
		--assets "$(ASSETS)" \
		--cash $(if $(CASH),$(CASH),CASH) \
		--labels_csv $(LABELS) \
		--risk_on $(RISK_ON) \
		--risk_off $(RISK_OFF) \
		--on_weights "$(ON_W)" \
		$(if $(OFF_W),--off_weights "$(OFF_W)",) \
		--benchmark $(BENCH) \
		--out_prefix $(OUT)

# ------------------------------
# Step 1–3: Features build
# ------------------------------

features:
	@mkdir -p $(DATA)/processed
	$(PY) $(SCRIPTS)/build_features.py \
		--panel $(if $(PANEL),$(PANEL),$(DATA)/processed/market_panel.parquet) \
		--out $(if $(OUT),$(OUT),$(DATA)/processed/features.parquet) \
		--standardize $(if $(STD),$(STD),rolling) \
		--window $(if $(WINDOW),$(WINDOW),252) \
		--bundle $(if $(BUNDLE),$(BUNDLE),market_plus) \
		$(if $(NOWINSOR),--no-winsor,) \
		$(if $(PCA),--pca,) \
		$(if $(PCA_WINDOW),--pca-window $(PCA_WINDOW),) \
		$(if $(PCA_REPLACE),--pca-replace,) \
		$(if $(WQLOW),--winsor-q $(WQLOW) $(if $(WQHIGH),$(WQHIGH),0.99),)

features-core:
	@$(MAKE) features BUNDLE=core

features-plus:
	@$(MAKE) features BUNDLE=market_plus

features-pca:
	@$(MAKE) features BUNDLE=market_plus PCA=1

# ------------------------------
# Step 4: Validation & Monitoring
# ------------------------------

validate:
	@mkdir -p $(REPORTS)/feature_report
	$(PY) $(SCRIPTS)/validate_features.py \
		--features $(if $(FEATURES),$(FEATURES),$(DATA)/processed/features.parquet) \
		--out-dir $(if $(OUTDIR),$(OUTDIR),$(REPORTS)/feature_report) \
		$(if $(SPLIT_DATE),--split-date $(SPLIT_DATE),$(if $(RATIO),--split-ratio $(RATIO),--split-ratio 0.8)) \
		$(if $(ROLLWIN),--roll-window $(ROLLWIN),)

validate-adf:
	@$(MAKE) validate ADF=1

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -prune -exec rm -rf {} +
