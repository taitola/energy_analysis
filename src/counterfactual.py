"""
counterfactual.py
=================
Dynamic historical counterfactual simulation engine.

What does this module do?
-------------------------
Imagine we could go back in time and "turn off" wind power shocks from
some date T* onward. How would electricity prices (and other variables)
have evolved differently?

This module answers that question by:
  1. Recovering the structural shocks (w_t) from the fitted SVAR model
  2. Setting chosen shocks to zero from T* onward (the "intervention")
  3. Re-simulating the entire system forward using the modified shocks
  4. Comparing the counterfactual path to the actual historical data

The difference between counterfactual and actual prices tells us the
estimated causal effect of wind power shocks on the electricity market.

Strictly functional and modular: no classes or OOP. Pure functions operating
on standard pandas DataFrames, NumPy arrays, and dictionaries.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.linalg import inv


def simulate_counterfactual(
    result: dict[str, Any],
    data: pd.DataFrame,
    intervention_start: str | pd.Timestamp,
    zeroed_shocks: list[int] | None = None,
) -> dict[str, Any]:
    """Execute a dynamic counterfactual simulation.

    Parameters
    ----------
    result : dict[str, Any]
        Fitted SVAR result dictionary from fit_var().
    data : pd.DataFrame
        Original data used for estimation, with a DatetimeIndex.
    intervention_start : str or pd.Timestamp
        Date T* from which the specified structural shocks are set to zero.
        Must lie within the effective sample (after initial lag observations).
    zeroed_shocks : list[int], optional
        Indices of structural shocks to zero. Defaults to [0] (wind shock).

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - 'actual': DataFrame of observed data aligned to simulation window.
        - 'counterfactual': DataFrame of counterfactual paths.
        - 'diff': DataFrame of (counterfactual − actual) = the causal effect.
        - 'intervention_start': pd.Timestamp of T*.
        - 'zeroed_shocks': list of zeroed shock indices.

    Raises
    ------
    ValueError
        If the VAR is not stable or intervention_start is invalid.
    """
    # --- Safety check: refuse to simulate an explosive model ---
    if not result.get("is_stable", False):
        max_mod = result.get("max_eigenvalue_modulus", 1.0)
        raise ValueError(
            f"Counterfactual simulation requires a stable VAR. Max |eigenvalue| = {max_mod:.6f}"
        )

    # Default: zero out only the wind shock (index 0)
    if zeroed_shocks is None:
        zeroed_shocks = [0]

    # --- Extract what we need from the estimation result ---
    names = result["variable_names"]       # e.g. ['wind_mw', 'load_mw', ...]
    data_aligned = data[names].copy()      # keep only the model variables
    Y = data_aligned.values                # shape: (T_full, K) — raw numpy array
    p = result["n_lags"]                   # number of lags (7)
    k = len(names)                         # number of variables (4)
    T_full = Y.shape[0]                    # total number of days

    # --- Locate the intervention date T* in the data index ---
    t_star = pd.Timestamp(intervention_start)
    date_index = data_aligned.index

    # If the exact date isn't in the index, find the next available date
    if t_star not in date_index:
        # Create a boolean mask: True where dates are >= t_star
        mask = date_index >= t_star
        if not mask.any():
            raise ValueError(
                f"intervention_start {t_star} is after the sample end ({date_index[-1]})."
            )
        # Pick the first date that is >= t_star
        t_star = date_index[mask][0]

    # Convert the date to a numeric position (row number) in the DataFrame
    t_star_pos = date_index.get_loc(t_star)

    # T* must be after the pre-sample lags (first p rows are used for initial conditions)
    if t_star_pos < p:
        raise ValueError(
            f"intervention_start must be after the first {p} dates (earliest valid: {date_index[p]})."
        )

    # --- Step 1: Recover structural shocks for the entire sample ---
    #
    # The relationship between reduced-form residuals (u_t) and structural
    # shocks (w_t) is:
    #     u_t = B0_inv @ w_t     (B0_inv is the lower-triangular Cholesky factor)
    #
    # To get w_t from u_t, we need B0 (the inverse of B0_inv):
    #     w_t = B0 @ u_t
    #
    B0_inv = result["B0_inv"]              # shape: (K, K), lower-triangular
    B0 = inv(B0_inv)                       # shape: (K, K), inverse of B0_inv
    intercept = result["intercept"]        # shape: (K,), constant terms
    A = result["A"]                        # list of p coefficient matrices, each (K, K)

    # First, compute the reduced-form residuals u_t for every time period
    # (this re-derives them from the raw data using the estimated coefficients)
    u_full = _compute_full_residuals(intercept, A, Y, k, p)  # shape: (T_eff, K)
    # T_eff = T_full - p (we lose the first p rows to lags)

    # Convert reduced-form residuals to structural shocks:
    #   w_t = B0 @ u_t   for each row t
    #
    # u_full.T has shape (K, T_eff), so B0 @ u_full.T has shape (K, T_eff)
    # .T transposes back to (T_eff, K) — each row is one day's structural shocks
    w_full = (B0 @ u_full.T).T             # shape: (T_eff, K)

    # --- Step 2: Build the counterfactual shock matrix ---
    # Copy the actual shocks, then zero out selected ones from T* onward
    w_cf = w_full.copy()

    # Convert T* from a position in the full data to a position in the
    # residual array (which starts p rows later)
    cf_start_in_resid = t_star_pos - p

    # Zero out the chosen structural shocks from T* onward
    for j in zeroed_shocks:
        w_cf[cf_start_in_resid:, j] = 0.0
    # Now w_cf is identical to w_full before T*, but has zeros in the
    # wind shock (or whichever shocks we chose) from T* onward.

    # --- Step 3: Dynamic forward simulation ---
    # Re-play the entire history using the modified shocks.
    # The first p rows are the actual data (initial conditions).
    # From row p onward, each day is simulated using the VAR equation:
    #   y_t = c + A_1·y_{t-1} + A_2·y_{t-2} + … + A_p·y_{t-p} + u_t^cf
    #
    # where u_t^cf = B0_inv @ w_t^cf  (counterfactual reduced-form residual)

    Y_cf = np.zeros((T_full, k))           # counterfactual data array
    Y_cf[:p] = Y[:p]                       # initial conditions = actual data

    for t in range(p, T_full):
        t_resid = t - p                     # index into the residual arrays

        # Start with the intercept (constant term)
        Y_cf[t] = intercept.copy()

        # Add contributions from all p lags:
        #   A_1 · y_{t-1}^cf + A_2 · y_{t-2}^cf + … + A_p · y_{t-p}^cf
        for i, Ai in enumerate(A, start=1):
            # Ai is the (K, K) coefficient matrix for lag i
            # Y_cf[t - i] is the (K,) counterfactual vector from i days ago
            # Ai @ Y_cf[t-i] does matrix-vector multiplication → (K,)
            Y_cf[t] += Ai @ Y_cf[t - i]

        # Add the counterfactual residual:
        #   u_t^cf = B0_inv @ w_t^cf
        # B0_inv @ w_cf[t_resid]: (K,K) @ (K,) → (K,)
        u_cf_t = B0_inv @ w_cf[t_resid]
        Y_cf[t] += u_cf_t

    # --- Step 4: Package results as DataFrames ---
    actual_df = data_aligned.copy()
    cf_df = pd.DataFrame(Y_cf, index=date_index, columns=names)

    # The difference shows the causal effect:
    # positive diff = counterfactual is higher than actual
    # (e.g., without wind shocks, prices would have been higher)
    diff_df = cf_df - actual_df

    return {
        "actual": actual_df,
        "counterfactual": cf_df,
        "diff": diff_df,
        "intervention_start": t_star,
        "zeroed_shocks": list(zeroed_shocks),
    }


def _compute_full_residuals(
    intercept: np.ndarray,
    A: list[np.ndarray],
    Y: np.ndarray,
    k: int,
    p: int,
) -> np.ndarray:
    """Compute reduced-form residuals u_t for all in-sample observations.

    For each time t (from p to T_full-1), the residual is:
        u_t = y_t  −  (c + A_1·y_{t-1} + A_2·y_{t-2} + … + A_p·y_{t-p})
              ^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
              actual               model's prediction (fitted value)

    Parameters
    ----------
    intercept : np.ndarray, shape (K,)
        Estimated constant terms.
    A : list of np.ndarray
        [A_1, …, A_p], each (K, K).
    Y : np.ndarray, shape (T_full, K)
        Full observed data.
    k : int
        Number of variables (K).
    p : int
        Number of lags.

    Returns
    -------
    np.ndarray, shape (T_eff, K)
        Residuals, where T_eff = T_full − p.
    """
    T_full = Y.shape[0]
    T_eff = T_full - p              # number of time periods with residuals
    residuals = np.zeros((T_eff, k))

    for t in range(p, T_full):
        # Compute the model's prediction (fitted value) for time t
        fitted = intercept.copy()   # start with the constant → shape (K,)
        for i, Ai in enumerate(A, start=1):
            # Add the contribution of lag i:
            #   Ai @ Y[t-i]  →  (K,K) @ (K,) → (K,)
            fitted += Ai @ Y[t - i]

        # Residual = actual minus predicted
        residuals[t - p] = Y[t] - fitted

    return residuals
