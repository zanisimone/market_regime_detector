import argparse
import pandas as pd
from pathlib import Path
from typing import List

from src.reporting.agreement import pairwise_confusion, agreement_rate, multi_model_agreement

def main():
    """
    CLI to compute model agreement diagnostics.

    Usage
    -----
    python scripts/run_agreement.py --labels_csv modelA.csv --name A --labels_csv modelB.csv --name B --labels model_ids --out_prefix reports/agreement
    CSV schema: date, regime
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_csv", action="append", required=True, help="Path(s) to CSV with columns: date, regime")
    parser.add_argument("--name", action="append", required=True, help="Model name(s) in same order as labels_csv")
    parser.add_argument("--labels", default="", help="Comma-separated regime ids (e.g., 0,1,2). If empty, inferred.")
    parser.add_argument("--out_prefix", required=True, help="Output prefix for CSV files.")
    args = parser.parse_args()

    assert len(args.labels_csv) == len(args.name), "labels_csv and name must have same length"

    series_map = {}
    all_vals = set()
    for path, nm in zip(args.labels_csv, args.name):
        df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        s = df["regime"].astype(int)
        series_map[nm] = s
        all_vals |= set(s.dropna().unique().tolist())

    labels = [int(x) for x in args.labels.split(",")] if args.labels else sorted(int(x) for x in all_vals)

    names = list(series_map.keys())
    mats = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            C = pairwise_confusion(series_map[names[i]], series_map[names[j]], labels)
            C.to_csv(f"{args.out_prefix}_confusion_{names[i]}_vs_{names[j]}.csv")
            mats.append(((names[i], names[j]), C))

    A, J = multi_model_agreement(series_map, labels)
    A.to_csv(f"{args.out_prefix}_agreement_rate.csv")
    J.to_csv(f"{args.out_prefix}_jaccard.csv")

if __name__ == "__main__":
    main()
