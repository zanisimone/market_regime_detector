import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict

from src.strategy.hooks import constant_weight_map, weights_from_labels
from src.backtest.vectorized import price_to_returns, vectorized_portfolio_returns, buy_and_hold_benchmark
from src.reporting.performance import summarize_performance, hit_rate_by_regime, regime_performance

def main():
    """
    CLI to run a simple regime-based allocation backtest and compare to buy&hold.

    Usage
    -----
    python scripts/run_allocation_backtest.py \
      --prices_csv data/prices.csv --assets "SPX,TLT" --cash CASH \
      --labels_csv reports/labels.csv --risk_on 1 --risk_off 0 \
      --on_weights "SPX:1.0" --off_weights "TLT:1.0" \
      --benchmark SPX --out_prefix reports/alloc_simple

    prices_csv schema: date,<asset1>,<asset2>,...,CASH(optional constant 1.0)
    labels_csv schema: date,regime
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices_csv", required=True, help="CSV with wide prices indexed by date")
    parser.add_argument("--assets", required=True, help="Comma-separated asset tickers present in prices")
    parser.add_argument("--cash", default="CASH", help="Cash column name (price must be constant level=1.0 if present)")
    parser.add_argument("--labels_csv", required=True, help="CSV with date,regime")
    parser.add_argument("--risk_on", type=int, required=True, help="Risk-On regime id")
    parser.add_argument("--risk_off", type=int, required=True, help="Risk-Off regime id")
    parser.add_argument("--on_weights", required=True, help='Comma-separated "ASSET:W" pairs for Risk-On')
    parser.add_argument("--off_weights", default="", help='Comma-separated "ASSET:W" pairs for Risk-Off; if empty, go to cash')
    parser.add_argument("--benchmark", required=True, help="Asset column to use for buy&hold benchmark")
    parser.add_argument("--out_prefix", required=True, help="Output prefix for CSVs")
    args = parser.parse_args()

    assets = [x.strip() for x in args.assets.split(",")]
    px = pd.read_csv(args.prices_csv, parse_dates=["date"]).set_index("date")
    if args.cash not in px.columns:
        px[args.cash] = 1.0

    lab = pd.read_csv(args.labels_csv, parse_dates=["date"]).set_index("date")["regime"].astype(int)
    px = px.reindex(lab.index.union(px.index)).ffill()
    lab = lab.reindex(px.index).ffill().bfill()

    def parse_weights(s: str) -> Dict[str, float]:
        if not s:
            return {}
        out = {}
        for kv in s.split(","):
            k, v = kv.split(":")
            out[k.strip()] = float(v)
        return out

    on_w = parse_weights(args.on_weights)
    off_w = parse_weights(args.off_weights) if args.off_weights else None

    regime_map = constant_weight_map(assets, args.risk_on, args.risk_off, on_w, off_w, cash_asset=args.cash)
    W = weights_from_labels(lab, regime_map, assets, cash_asset=args.cash)

    rets = price_to_returns(px[assets + [args.cash]])
    port_ret, tvr = vectorized_portfolio_returns(rets, W, rebalance_on_change=True)
    bench_ret = buy_and_hold_benchmark(px, args.benchmark).reindex(port_ret.index).fillna(0.0)

    port_metrics = summarize_performance(port_ret, turnover=tvr)
    bench_metrics = summarize_performance(bench_ret)

    hr = hit_rate_by_regime(port_ret, lab)
    rp = regime_performance(port_ret, lab)

    out_path = Path(args.out_prefix)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    equity = pd.DataFrame({
        "port_ret": port_ret,
        "bench_ret": bench_ret,
        "port_wealth": (1.0 + port_ret).cumprod(),
        "bench_wealth": (1.0 + bench_ret).cumprod(),
        "turnover": tvr,
        "regime": lab,
    })
    equity.to_csv(f"{args.out_prefix}_timeseries.csv", index=True)

    pd.DataFrame([port_metrics]).to_csv(f"{args.out_prefix}_metrics_portfolio.csv", index=False)
    pd.DataFrame([bench_metrics]).to_csv(f"{args.out_prefix}_metrics_benchmark.csv", index=False)
    hr.to_csv(f"{args.out_prefix}_hit_rate_by_regime.csv")
    rp.to_csv(f"{args.out_prefix}_regime_performance.csv")

if __name__ == "__main__":
    main()
