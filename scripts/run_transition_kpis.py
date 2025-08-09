import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from src.regime.transition import estimate_transition_matrix, stationary_distribution, dwell_time_summary
from src.regime.stability import stability_kpis, dwell_stability_kpis

def main():
    """
    CLI entry-point to compute transition matrix, stationary distribution, dwell-time stats,
    and stability KPIs across multiple model splits.

    Usage
    -----
    python scripts/run_transition_kpis.py --labels_csv data/labels_split1.csv --labels_csv data/labels_split2.csv --risk_off_label 0 --out_csv reports/transition_kpis.csv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_csv", action="append", required=True, help="Path(s) to CSV with columns: date, regime (int). Use multiple --labels_csv for splits.")
    parser.add_argument("--regime_col", default="regime", help="Column name for regime labels.")
    parser.add_argument("--risk_off_label", type=int, default=0, help="Integer id for Risk-Off regime.")
    parser.add_argument("--out_csv", required=True, help="Path to output KPIs CSV.")
    args = parser.parse_args()

    P_list = []
    dwell_summ_list = []

    for p in args.labels_csv:
        df = pd.read_csv(p, parse_dates=["date"])
        labels = df[args.regime_col].astype(int)
        P = estimate_transition_matrix(labels)
        P_list.append(P)
        dwell_summ_list.append(dwell_time_summary(labels))

    P_bar = np.mean(np.stack(P_list, axis=0), axis=0)
    pi_bar = stationary_distribution(P_bar)

    stab = stability_kpis(P_list)
    dwell_k = dwell_stability_kpis(dwell_summ_list)

    rows = {
        "frobenius_dispersion": stab["frobenius_dispersion"],
        "mean_spectral_gap": stab["mean_spectral_gap"],
        "std_spectral_gap": stab["std_spectral_gap"],
        "coeff_var_P": stab["coeff_var_P"],
        "mean_mean_dwell": dwell_k["mean_mean_dwell"],
        "std_mean_dwell": dwell_k["std_mean_dwell"],
        "mean_max_dwell": dwell_k["mean_max_dwell"],
    }
    rows.update({f"stationary_p{idx}": float(pi_bar[idx]) for idx in range(len(pi_bar))})
    out = pd.DataFrame([rows])
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()
