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
        agreement composites center-explorer allocation-backtest

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
# Pipelines
# ------------------------------

# Example:
# make run-kmeans-split IN=$(DATA)/panel.parquet OUT=$(REPORTS)/labels_split1.csv K=3
run-kmeans-split:
	@if [ -z "$(IN)" ]; then echo "IN is required (input parquet/csv)"; exit 1; fi
	@if [ -z "$(OUT)" ]; then echo "OUT is required (output csv)"; exit 1; fi
	@if [ -z "$(K)" ]; then echo "K is required (n_clusters)"; exit 1; fi
	$(PY) $(SCRIPTS)/run_kmeans_split.py --in $(IN) --out $(OUT) --n_clusters $(K)

# Example:
# make transition-kpis LABELS="$(REPORTS)/labels_split1.csv $(REPORTS)/labels_split2.csv" RISK_OFF=0 OUT=$(REPORTS)/transition_kpis.csv
transition-kpis:
	@if [ -z "$(LABELS)" ]; then echo "LABELS is required (space-separated CSV paths)"; exit 1; fi
	@if [ -z "$(OUT)" ]; then echo "OUT is required (output csv)"; exit 1; fi
	@mkdir -p $(REPORTS)
	$(PY) $(SCRIPTS)/run_transition_kpis.py \
		$(foreach f,$(LABELS),--labels_csv $(f)) \
		--risk_off_label $(if $(RISK_OFF),$(RISK_OFF),0) \
		--out_csv $(OUT)

# Example:
# make early-warning PROB=$(DATA)/risk_off_prob.csv OUT=$(REPORTS)/early_warning.csv \
#   MA=5 ZW=63 PTH=0.60 ZTH=1.0 SLOPE=0.02 COOL=5 COL=prob
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
# Reporting & Diagnostics
# ------------------------------

# Example:
# make agreement LABELS="$(REPORTS)/modelA.csv $(REPORTS)/modelB.csv $(REPORTS)/modelC.csv" \
#   NAMES="A B C" OUT=$(REPORTS)/agreement AGG_LABELS="0,1,2"
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

# Example:
# make composites PRICE=$(DATA)/px.csv LABELS=$(REPORTS)/labels.csv REGIME=0 OUT=$(REPORTS)/comp_r0.csv \
#   LOOKBACK=5 LOOKAHEAD=20 MODE=rebased MINRUN=3
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

# Example:
# make center-explorer CENTERS=$(REPORTS)/kmeans_centers.csv FEATCOLS="ret_1d,vol_21d,carry,curve_slope" \
#   OUT=$(REPORTS)/centers TOP=10 STATS=$(REPORTS)/feature_stats.csv
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
# Strategy Hooks & Backtest
# ------------------------------

# Example:
# make allocation-backtest PRICES=$(DATA)/prices.csv ASSETS="SPX,TLT" CASH=CASH \
#   LABELS=$(REPORTS)/labels.csv RISK_ON=1 RISK_OFF=0 \
#   ON_W="SPX:1.0" OFF_W="TLT:1.0" BENCH=SPX OUT=$(REPORTS)/alloc_simple
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

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -prune -exec rm -rf {} +
