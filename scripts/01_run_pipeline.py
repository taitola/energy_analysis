"""
01_run_pipeline.py
==================
End-to-end execution script for the Finnish electricity market SVAR analysis.

This script runs all five stages of the analysis pipeline:

  1. LOAD DATA      — Read daily market data from a local CSV cache, or
                      download it from the Fingrid REST API if not cached.

  2. ESTIMATE VAR   — Fit a reduced-form VAR(7) model via OLS (Ordinary
                      Least Squares). This models each variable as a linear
                      function of its own 7 past daily values plus the 7
                      past values of all other variables.

  3. CHOLESKY ID    — Apply Cholesky identification to decompose correlated
                      residuals into uncorrelated structural shocks.
                      Wind is ordered first (exogenous weather), price last.

  4. IMPULSE-RESPONSE — Compute and plot orthogonalised IRFs showing how a
                      one-standard-deviation wind shock propagates through
                      the system over 30 days, with 90% bootstrap confidence
                      bands (200 replications).

  5. COUNTERFACTUAL — Simulate what would have happened if wind shocks had
                      been zero from the midpoint of the sample onward.
                      Plot actual vs counterfactual electricity prices.

Usage
-----
    python scripts/01_run_pipeline.py

Prerequisites
-------------
    - A cached CSV at data/fingrid_daily.csv, OR a valid Fingrid API key
      in the .env file (FINGRID_API_KEY=your_key_here).
    - All dependencies listed in the project's requirements.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Make sure Python can find our project modules (src/) regardless of the
# directory from which this script is launched.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.download import VARIABLE_ORDER, load_daily_data
from src.svar_estimator import bootstrap_irf, fit_var
from src.counterfactual import simulate_counterfactual


def main() -> None:
    """Run the complete SVAR analysis pipeline."""

    print("=" * 72)
    print("  energy-svar-project  |  Fingrid SVAR Pipeline")
    print("=" * 72)

    # Where to look for (or save) the cached daily data
    data_file = PROJECT_ROOT / "data" / "fingrid_daily.csv"

    # Where to save generated figures
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1/5 — LOAD OR DOWNLOAD DATA
    # ══════════════════════════════════════════════════════════════════════
    #
    # load_daily_data() first checks if a cached CSV file exists at
    # `data_file`. If it does, it reads from disk (fast). If not, it
    # downloads four time series from the Fingrid REST API, aggregates
    # them to daily frequency, and saves the result to disk for next time.
    #
    # The four variables (columns) in the resulting DataFrame are:
    #   wind_mw     — Daily mean wind power generation (MW)
    #   load_mw     — Daily total electricity consumption (MW)
    #   net_flow_mw — Daily net cross-border transmission flow (MW)
    #   price_eur   — Daily electricity spot price (EUR/MWh)
    #
    # The index is a DatetimeIndex of daily dates.

    print("\n[1/5] Loading daily market data ...")
    try:
        df = load_daily_data(
            file_path=data_file,
            start_time="2023-01-01T00:00:00Z",
            end_time="2024-01-01T00:00:00Z",
        )
    except ValueError as err:
        # This typically means the Fingrid API key is missing or invalid
        print(f"\n[!] Configuration Notice:\n{err}\n")
        print("Please configure your .env file with your Fingrid API key and re-run.")
        sys.exit(1)

    print(f"      Rows: {len(df)}  |  Date range: {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"      Variables: {list(df.columns)}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2/5 — ESTIMATE REDUCED-FORM VAR(7)
    # ══════════════════════════════════════════════════════════════════════
    #
    # fit_var() estimates the Vector Autoregression:
    #
    #   y_t = c + A_1·y_{t-1} + A_2·y_{t-2} + … + A_7·y_{t-7} + u_t
    #
    # where y_t is a 4×1 vector [wind, load, flow, price] and A_i are
    # 4×4 coefficient matrices estimated via OLS.
    #
    # It also checks stability: the companion matrix eigenvalues must all
    # have modulus < 1 (meaning shocks die out rather than explode).
    #
    # The result is a dictionary containing everything we need for the
    # subsequent steps (coefficients, residuals, Cholesky factor, etc.).

    print("\n[2/5] Estimating reduced-form VAR(7) via OLS ...")
    result = fit_var(df, n_lags=7)

    # Effective T = total observations minus 7 (the first 7 rows are
    # consumed as initial lags and cannot be predicted)
    print(f"      Effective T = {result['n_obs']}")
    print(f"      Max |eigenvalue| of companion matrix = {result['max_eigenvalue_modulus']:.6f}")
    print(f"      Stable: {result['is_stable']}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3/5 — CHOLESKY IDENTIFICATION
    # ══════════════════════════════════════════════════════════════════════
    #
    # The Cholesky factor B0_inv is a 4×4 lower-triangular matrix that
    # decomposes the residual covariance:
    #
    #   Σ_u = B0_inv @ B0_inv.T
    #
    # It maps uncorrelated structural shocks (w_t) into correlated
    # reduced-form residuals (u_t):  u_t = B0_inv @ w_t
    #
    # Because B0_inv is lower-triangular, the ordering of variables
    # determines the causal identification:
    #   - Wind (row 0) is affected only by its own shock
    #   - Load (row 1) can be affected by wind and load shocks
    #   - Flow (row 2) can be affected by wind, load, and flow shocks
    #   - Price (row 3) is affected by ALL shocks contemporaneously
    #
    # This reflects the economic logic: weather (wind) is exogenous,
    # while price reacts to everything within the same day.

    print("\n[3/5] Cholesky identification factor (B0_inv):")
    # np.array2string formats a numpy array for nice printing
    # precision=4 shows 4 decimal places, suppress_small hides tiny values
    print(np.array2string(result["B0_inv"], precision=4, suppress_small=True))

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4/5 — IMPULSE-RESPONSE FUNCTIONS (IRFs)
    # ══════════════════════════════════════════════════════════════════════
    #
    # An IRF shows how the system responds over time to a one-time
    # structural shock. Here we shock wind generation (shock_idx=0) by
    # +1 standard deviation and trace the response of all 4 variables
    # over 30 days.
    #
    # bootstrap_irf() returns:
    #   - irf_point: the point estimate (from the original estimation)
    #   - irf_lo / irf_hi: 90% bootstrap confidence bands
    #     (computed from 200 resampled datasets)
    #
    # Each has shape (31, 4) — 31 horizons (0 to 30) × 4 variables.

    print("\n[4/5] Computing orthogonalised IRFs (wind shock -> 30 days) ...")
    irf_point, irf_lo, irf_hi = bootstrap_irf(
        result,
        df,
        steps=30,          # trace the response for 30 days
        shock_idx=0,       # shock variable 0 = wind_mw
        shock_size=1.0,    # +1 standard deviation shock
        n_boot=200,        # number of bootstrap replications
        ci=0.90,           # 90% confidence interval
        seed=42,           # fixed seed for reproducibility
    )

    # --- Plot a 2×2 grid of IRF charts (one per variable) ---
    #
    # plt.subplots(2, 2) creates a figure with 4 subplots arranged in
    # a 2-row × 2-column grid. It returns:
    #   fig  — the overall figure object
    #   axes — a 2×2 numpy array of Axes (individual plot areas)
    #
    # sharex=True makes all subplots share the same x-axis range.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle("Orthogonalised IRF: +1 s.d. Wind Generation Shock", fontsize=14)

    # horizons = [0, 1, 2, …, 30] — the x-axis values (days after shock)
    horizons = np.arange(irf_point.shape[0])

    # axes.ravel() flattens the 2×2 array of axes into a 1D list of 4 axes,
    # so we can iterate over them together with the 4 variable names.
    # zip() pairs each axis with its corresponding variable name.
    for idx, (ax, var) in enumerate(zip(axes.ravel(), VARIABLE_ORDER)):
        # Plot the point estimate as a solid black line
        ax.plot(horizons, irf_point[:, idx], "k-", lw=1.5, label="Point")

        # Plot the 90% confidence band as a shaded blue region
        # fill_between() fills the area between irf_lo and irf_hi
        ax.fill_between(
            horizons,
            irf_lo[:, idx],
            irf_hi[:, idx],
            alpha=0.25,            # transparency (0 = invisible, 1 = opaque)
            color="steelblue",
            label="90% CI",
        )

        # Add a horizontal dashed line at y=0 for reference
        ax.axhline(0, color="grey", ls="--", lw=0.5)
        ax.set_title(var)
        ax.legend(fontsize=8)

    # Label the x-axis only on the bottom row
    axes[1, 0].set_xlabel("Days after shock")
    axes[1, 1].set_xlabel("Days after shock")

    # tight_layout() automatically adjusts spacing so labels don't overlap
    plt.tight_layout()

    # Save the figure to disk as a PNG image
    irf_path = figures_dir / "irf_wind_shock.png"
    fig.savefig(irf_path, dpi=150)   # dpi=150 gives good resolution
    plt.close(fig)                    # close the figure to free memory
    print(f"      Saved IRF plot -> {irf_path}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 5/5 — DYNAMIC COUNTERFACTUAL SIMULATION
    # ══════════════════════════════════════════════════════════════════════
    #
    # We ask: "What would electricity prices have looked like if wind
    # shocks had been zero from the midpoint of the sample onward?"
    #
    # simulate_counterfactual() replays the entire history through the VAR,
    # but zeroes out the wind structural shock (index 0) starting at T*.
    # All dynamic feedback (wind → load → flow → price) still operates
    # through the lagged coefficients, but there are no new wind surprises.
    #
    # The result dictionary contains:
    #   cf["actual"]          — DataFrame of the observed historical data
    #   cf["counterfactual"]  — DataFrame of the simulated "no wind shock" path
    #   cf["diff"]            — counterfactual minus actual (the causal effect)

    # Pick the midpoint of the sample as the intervention date T*
    mid_date = df.index[len(df) // 2]
    print(f"\n[5/5] Running counterfactual: zeroing wind shocks from {mid_date.date()} ...")

    cf = simulate_counterfactual(
        result,
        df,
        intervention_start=mid_date,
        zeroed_shocks=[0],    # zero out only the wind shock (index 0)
    )

    # --- Plot actual vs counterfactual electricity price ---
    fig2, ax2 = plt.subplots(figsize=(12, 5))

    # Plot the actual observed price as a solid line
    ax2.plot(
        cf["actual"].index,
        cf["actual"]["price_eur"],
        label="Actual",
        lw=1,
    )

    # Plot the counterfactual price as a dashed line
    ax2.plot(
        cf["counterfactual"].index,
        cf["counterfactual"]["price_eur"],
        label="Counterfactual (no wind shock)",
        lw=1,
        ls="--",        # dashed line style
    )

    # Draw a vertical red dotted line at the intervention date T*
    ax2.axvline(mid_date, color="red", ls=":", lw=1, label=f"T* = {mid_date.date()}")

    ax2.set_title("Counterfactual Electricity Price Path: Wind Shock -> 0")
    ax2.set_ylabel("Price (EUR/MWh)")
    ax2.legend()
    plt.tight_layout()

    # Save the counterfactual plot
    cf_path = figures_dir / "counterfactual_price.png"
    fig2.savefig(cf_path, dpi=150)
    plt.close(fig2)
    print(f"      Saved counterfactual plot -> {cf_path}")

    # --- Compute the average causal effect on price after T* ---
    #
    # cf["diff"] = counterfactual − actual
    #   Positive diff → without wind shocks, price would have been HIGHER
    #   Negative diff → without wind shocks, price would have been LOWER
    #
    # .loc[mid_date:, "price_eur"] selects the price_eur column from
    # the intervention date onward (all post-T* rows).
    # .mean() takes the average of those differences.
    mean_effect = cf["diff"].loc[mid_date:, "price_eur"].mean()
    print(f"\n      Mean counterfactual price effect (post-T*): {mean_effect:+.2f} EUR/MWh")

    print("\n" + "=" * 72)
    print("  Pipeline complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
