from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.download import VARIABLE_ORDER
from src.svar_estimator import fit_var

LABELS: dict[str, str] = {
    "wind_mw": "Wind Generation (MW)",
    "load_mw": "Electricity Consumption (MW)",
    "net_flow_mw": "Net Cross-Border Flow (MW)",
    "price_eur": "Spot Price (EUR/MWh)",
}


def main() -> None:
    data_path = PROJECT_ROOT / "data" / "fingrid_daily.csv"
    df = pd.read_csv(data_path, parse_dates=["date"], index_col="date")
    df = df[VARIABLE_ORDER].dropna()

    result = fit_var(df, n_lags=7)

    residuals = pd.DataFrame(
        result["residuals"],
        columns=result["variable_names"],
        index=df.index[result["n_lags"]:],
    )

    # --- Time series plots ---
    fig_ts, axes_ts = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    for i, var in enumerate(VARIABLE_ORDER):
        ax = axes_ts[i]
        mean = df[var].mean()
        std = df[var].std()
        ax.plot(df.index, df[var], linewidth=0.7, color="steelblue")
        ax.axhline(mean, color="grey", linewidth=0.8, linestyle="-", label="mean")
        ax.axhline(mean + std, color="grey", linewidth=0.8, linestyle="--", label="+1 σ")
        ax.axhline(mean - std, color="grey", linewidth=0.8, linestyle="--", label="−1 σ")
        ax.set_ylabel(LABELS[var], fontsize=9)
        ax.grid(True, alpha=0.3)
    axes_ts[0].set_title("Endogenous Variables — Daily Time Series", fontsize=13)
    axes_ts[-1].set_xlabel("Date")
    fig_ts.tight_layout()

    # --- Residual plots ---
    fig_res, axes_res = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    for i, var in enumerate(VARIABLE_ORDER):
        ax = axes_res[i]
        std = residuals[var].std()
        ax.plot(residuals.index, residuals[var], linewidth=0.6, color="indianred")
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.axhline(std, color="grey", linewidth=0.8, linestyle="--", label="+1 σ")
        ax.axhline(-std, color="grey", linewidth=0.8, linestyle="--", label="−1 σ")
        ax.set_ylabel(f"{LABELS[var]}\nresidual", fontsize=9)
        ax.grid(True, alpha=0.3)
    axes_res[0].set_title("VAR(7) Reduced-Form Residuals", fontsize=13)
    axes_res[-1].set_xlabel("Date")
    fig_res.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
