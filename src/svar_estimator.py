"""
svar_estimator.py
=================
Estimates a reduced-form VAR(p) model using plain OLS (Ordinary Least Squares),
identifies structural shocks via Cholesky decomposition, and computes
impulse-response functions (IRFs) with bootstrap confidence bands.

Refresher on the core idea
--------------------------
A VAR (Vector Autoregression) models K time series jointly:

    y_t = c + A_1 · y_{t-1} + A_2 · y_{t-2} + … + A_p · y_{t-p} + u_t

where:
    y_t   = (K × 1) vector of today's values (wind, load, flow, price)
    c     = (K × 1) intercept / constant vector
    A_i   = (K × K) coefficient matrix for lag i
    u_t   = (K × 1) reduced-form error / residual (correlated across variables)
    p     = number of lags (we use 7 days)

"Cholesky identification" decomposes the covariance of u_t into structural
shocks w_t that are uncorrelated (orthogonal):
    Σ_u = B₀⁻¹ (B₀⁻¹)ᵀ     where B₀⁻¹ is lower-triangular
    u_t = B₀⁻¹ · w_t         so w_t = B₀ · u_t

The ordering of variables matters: wind is first (exogenous weather),
price is last (reacts to everything contemporaneously).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from numpy.linalg import cholesky, eigvals, lstsq, solve

# ---------------------------------------------------------------------------
# Make sure Python can find our project modules when run from any directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.download import VARIABLE_ORDER
except ImportError:
    from download import VARIABLE_ORDER


# ═══════════════════════════════════════════════════════════════════════════
# 1. COMPANION MATRIX  — used to check VAR stability
# ═══════════════════════════════════════════════════════════════════════════

def assemble_companion(A_list: Sequence[np.ndarray], k: int) -> np.ndarray:
    """Build the (Kp × Kp) VAR companion matrix from coefficient matrices.

    The companion form rewrites the p-lag system as a first-order system:

        Z_t = F · Z_{t-1} + ...

    where Z_t stacks [y_t, y_{t-1}, …, y_{t-p+1}] and F is:

        ┌ A_1  A_2  ···  A_p ┐
        │ I    0    ···  0   │
        │ 0    I    ···  0   │
        └ 0    0    ···  0   ┘

    The VAR is stable (stationary) if and only if ALL eigenvalues of F
    have modulus (absolute value) strictly less than 1.

    Parameters
    ----------
    A_list : list of np.ndarray
        [A_1, A_2, …, A_p], each shaped (K, K).
    k : int
        Number of variables (K = 4 in our model).

    Returns
    -------
    np.ndarray
        Companion matrix of shape (K*p, K*p).
    """
    p = len(A_list)                               # number of lags
    size = k * p                                   # total dimension of companion matrix
    companion = np.zeros((size, size))

    # --- Top row: paste A_1, A_2, …, A_p side by side ---
    # np.hstack glues arrays horizontally: [(K,K), (K,K), ...] → (K, K*p)
    companion[:k, :] = np.hstack(A_list)

    # --- Sub-diagonal identity block ---
    # This captures the trivial identity  y_{t-1} = y_{t-1}
    # It's an (K*(p-1)) × (K*(p-1)) identity matrix placed below the top row
    if p > 1:
        companion[k:, : k * (p - 1)] = np.eye(k * (p - 1))

    return companion


# ═══════════════════════════════════════════════════════════════════════════
# 2. OLS INTERNALS  — building the regression and solving it
# ═══════════════════════════════════════════════════════════════════════════

def _build_regressor_matrix(Y: np.ndarray, p: int) -> np.ndarray:
    """Create the OLS design matrix X from the raw data Y.

    For a VAR(p), each row of X at time t contains:
        [1,  y_{t-1}',  y_{t-2}',  …,  y_{t-p}']

    where y_{t-i}' is a row of K values (transposed from the column vector).

    Example with K=2 variables, p=2 lags, and T_full=5 observations:

        Raw data Y (5 rows × 2 cols):
            row 0: [y0_a, y0_b]  ← used only as lag, not as a dependent obs
            row 1: [y1_a, y1_b]  ← used only as lag
            row 2: [y2_a, y2_b]  ← first usable observation (depends on rows 0,1)
            row 3: [y3_a, y3_b]
            row 4: [y4_a, y4_b]

        Design matrix X (3 rows × 5 cols):
            [1, y1_a, y1_b, y0_a, y0_b]   ← regressors for predicting row 2
            [1, y2_a, y2_b, y1_a, y1_b]   ← regressors for predicting row 3
            [1, y3_a, y3_b, y2_a, y2_b]   ← regressors for predicting row 4

    Parameters
    ----------
    Y : np.ndarray, shape (T_full, K)
        Full matrix of observed data (each row = one day, each col = one variable).
    p : int
        Number of lags.

    Returns
    -------
    np.ndarray, shape (T, K*p + 1)
        Regressor matrix where T = T_full - p (we lose the first p rows to lags).
    """
    T_full, k = Y.shape    # T_full = total number of days, k = number of variables
    T = T_full - p          # usable observations after reserving p rows for lags

    # Start with a column of ones (the intercept / constant term)
    # Shape: (T, K*p + 1)  — one intercept column + K values for each of p lags
    X = np.ones((T, k * p + 1))

    for lag in range(1, p + 1):
        # For lag=1: we want rows [p-1, p, ..., T_full-2] from Y
        #   these are y_{t-1} for t = p, p+1, ..., T_full-1
        # For lag=2: rows [p-2, p-1, ..., T_full-3]  (y_{t-2})
        # etc.
        #
        # Slice explanation:
        #   start = p - lag       (e.g., lag=1 → start = p-1)
        #   stop  = T_full - lag  (e.g., lag=1 → stop = T_full-1)
        #   This gives exactly T rows.
        start_row = p - lag
        end_row = T_full - lag

        # Place these K columns into the right position in X
        col_start = 1 + (lag - 1) * k   # skip the intercept column
        col_end = 1 + lag * k
        X[:, col_start:col_end] = Y[start_row:end_row]

    return X


def _ols_var_core(Y: np.ndarray, p: int) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    """Run OLS regression for the VAR and return the estimated coefficients.

    This is the mathematical heart of the estimation:

        Y_dep = X @ B_hat + residuals

    where:
        Y_dep  = (T, K) matrix — the observations we're trying to predict
        X      = (T, K*p+1) matrix — lagged values + intercept
        B_hat  = (K*p+1, K) matrix — OLS coefficient estimates
        residuals = Y_dep - X @ B_hat — what the model can't explain

    We solve for B_hat using least squares: B_hat = (X'X)⁻¹ X'Y_dep

    Parameters
    ----------
    Y : np.ndarray, shape (T_full, K)
        Full raw data matrix.
    p : int
        Number of lags.

    Returns
    -------
    intercept : np.ndarray, shape (K,)
        Estimated constant term for each variable.
    A_list : list of np.ndarray
        [A_1, A_2, …, A_p], each (K, K) — the lag coefficient matrices.
    residuals : np.ndarray, shape (T, K)
        OLS residuals (prediction errors).
    sigma_u : np.ndarray, shape (K, K)
        Estimated covariance matrix of the residuals.
    """
    T_full, k = Y.shape
    T = T_full - p                    # number of usable observations

    # Y_dep: the "dependent variable" — the rows we're trying to predict
    # We skip the first p rows because those are only used as lagged inputs
    Y_dep = Y[p:]                     # shape: (T, K)

    # X: the "regressor matrix" — lagged values + a column of 1s for the intercept
    X = _build_regressor_matrix(Y, p)  # shape: (T, K*p + 1)

    # --- Solve the OLS problem: find B_hat that minimises ||Y_dep - X @ B_hat||² ---
    # np.linalg.lstsq returns (solution, residuals_sum, rank, singular_values)
    # We only need the solution (index [0])
    # rcond=None silences a numpy deprecation warning
    B_hat = lstsq(X, Y_dep, rcond=None)[0]   # shape: (K*p + 1, K)

    # --- Extract the intercept (first row of B_hat) ---
    intercept = B_hat[0, :]                   # shape: (K,)

    # --- Extract coefficient matrices A_1, A_2, …, A_p ---
    # B_hat layout (rows):
    #   row 0          → intercept
    #   rows 1..K      → coefficients for lag 1
    #   rows K+1..2K   → coefficients for lag 2
    #   etc.
    #
    # Each block is (K, K) in B_hat, but stored as (K rows, K cols).
    # We need to transpose (.T) because the VAR equation is:
    #   y_t = c + A_1 · y_{t-1} + ...
    # where A_i multiplies a column vector, but in B_hat the coefficients
    # are arranged for the row-based regression X @ B_hat.
    A_list = []
    for i in range(p):
        row_start = 1 + i * k
        row_end = 1 + (i + 1) * k
        A_i = B_hat[row_start:row_end, :].T    # transpose: (K, K)
        A_list.append(A_i)

    # --- Compute residuals: what the model couldn't predict ---
    # X @ B_hat = predicted values (fitted values)
    # residuals = actual - predicted
    residuals = Y_dep - X @ B_hat              # shape: (T, K)

    # --- Estimate the residual covariance matrix Σ_u ---
    # This tells us how the residuals co-move across variables.
    # We divide by degrees of freedom (T minus number of parameters per equation)
    # to get an unbiased estimate.
    dof = max(T - (k * p + 1), 1)              # degrees of freedom
    # residuals.T @ residuals = (K, T) @ (T, K) = (K, K) — sum of outer products
    sigma_u = (residuals.T @ residuals) / dof   # shape: (K, K)

    return intercept, A_list, residuals, sigma_u


# ═══════════════════════════════════════════════════════════════════════════
# 3. MAIN ESTIMATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def fit_var(
    data: pd.DataFrame,
    n_lags: int = 7,
    variable_names: list[str] | None = None,
    require_stability: bool = True,
) -> dict[str, Any]:
    """Estimate a VAR(p) model and identify structural shocks via Cholesky.

    This is the main entry point. It:
      1. Extracts the numeric data from the DataFrame
      2. Runs OLS to get coefficient matrices and residuals
      3. Checks stability (all companion eigenvalues < 1 in modulus)
      4. Applies Cholesky decomposition to identify structural shocks

    Parameters
    ----------
    data : pd.DataFrame
        Daily data with a DatetimeIndex. Must contain all columns in
        VARIABLE_ORDER: ['wind_mw', 'load_mw', 'net_flow_mw', 'price_eur'].
    n_lags : int, default 7
        Number of lags (p). We use 7 for daily data (one week of memory).
    variable_names : list[str], optional
        Override variable ordering. Defaults to VARIABLE_ORDER.
    require_stability : bool, default True
        If True, raises ValueError when the VAR is explosive (unstable).

    Returns
    -------
    dict[str, Any]
        Dictionary with all estimation results. Key contents:
        - 'intercept': (K,) array — constant terms
        - 'A': list of (K,K) arrays — lag coefficient matrices [A_1, …, A_p]
        - 'sigma_u': (K,K) array — residual covariance matrix
        - 'B0_inv': (K,K) array — lower-triangular Cholesky factor of sigma_u
        - 'residuals': (T,K) array — OLS residuals
        - 'structural_shocks': (T,K) array — identified structural shocks w_t
        - 'companion_eigenvalues': (K*p,) complex array — stability check
        - 'max_eigenvalue_modulus': float — largest |eigenvalue|; must be < 1
        - 'is_stable': bool
        - 'n_obs': int — effective sample size T (= total rows − p)
        - 'n_lags': int — p
        - 'variable_names': list[str]
        - 'initial_values': (p, K) array — first p rows, used for simulation

    Raises
    ------
    ValueError
        If there are too few observations or the VAR is explosive.
    """
    # --- Step 1: Extract numeric data from the DataFrame ---
    names = list(variable_names or VARIABLE_ORDER)

    # data[names] selects only the columns we need, in the right order
    # .to_numpy(dtype=float) converts the DataFrame to a plain 2D numpy array
    Y = data[names].to_numpy(dtype=float)     # shape: (T_full, K)
    T_full, k = Y.shape                        # T_full = total days, k = 4 variables
    p = n_lags
    T = T_full - p                             # effective sample size

    # Sanity check: we need more observations than parameters
    if T <= k * p + 1:
        raise ValueError(f"Insufficient observations: T={T_full} for p={p}, K={k}.")

    # --- Step 2: Run OLS to estimate the VAR ---
    intercept, A_list, residuals, sigma_u = _ols_var_core(Y, p)

    # --- Step 3: Build companion matrix and check stability ---
    companion = assemble_companion(A_list, k)

    # eigvals() returns the eigenvalues of the companion matrix (complex numbers)
    eig = eigvals(companion)

    # np.abs() computes the modulus (absolute value) of each complex eigenvalue
    # np.max() finds the largest one
    max_mod = float(np.max(np.abs(eig)))

    # The VAR is stable if ALL eigenvalues have modulus < 1
    # (this means shocks die out over time rather than exploding)
    is_stable = max_mod < 1.0

    if require_stability and not is_stable:
        raise ValueError(f"VAR({p}) is explosive: max |eigenvalue| = {max_mod:.6f} >= 1.0.")

    # --- Step 4: Cholesky identification ---
    # cholesky() decomposes the symmetric positive-definite matrix sigma_u as:
    #   sigma_u = B0_inv @ B0_inv.T
    # where B0_inv is lower-triangular (zeros above the diagonal).
    #
    # This is the "Cholesky factor". It maps uncorrelated structural shocks
    # (w_t) into the correlated reduced-form residuals (u_t):
    #   u_t = B0_inv @ w_t
    B0_inv = cholesky(sigma_u)                # shape: (K, K), lower-triangular

    # Recover structural shocks by solving: B0_inv @ w_t = u_t  for each t
    # This is like "dividing" by B0_inv (but more numerically stable than
    # computing the inverse explicitly).
    #
    # solve(A, B) solves A @ X = B for X.
    # residuals.T has shape (K, T), so we get structural_shocks with shape (K, T),
    # then transpose back to (T, K).
    structural_shocks = solve(B0_inv, residuals.T).T   # shape: (T, K)

    # --- Package everything into a dictionary ---
    return {
        "intercept": intercept,                     # (K,)     constant terms
        "A": A_list,                                # list of (K,K) lag matrices
        "sigma_u": sigma_u,                         # (K,K)   residual covariance
        "B0_inv": B0_inv,                           # (K,K)   Cholesky factor
        "residuals": residuals,                     # (T,K)   OLS residuals u_t
        "companion_eigenvalues": eig,               # (K*p,)  eigenvalues
        "max_eigenvalue_modulus": max_mod,           # float   largest |eigenvalue|
        "is_stable": is_stable,                     # bool    True if stationary
        "structural_shocks": structural_shocks,     # (T,K)   identified w_t
        "n_obs": T,                                 # int     effective sample size
        "n_lags": p,                                # int     number of lags
        "variable_names": names,                    # list    variable names
        "initial_values": Y[:p].copy(),             # (p,K)   pre-sample rows
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. IMPULSE-RESPONSE FUNCTIONS  (IRFs)
# ═══════════════════════════════════════════════════════════════════════════

def compute_irf(
    result: dict[str, Any],
    steps: int = 30,
    shock_idx: int = 0,
    shock_size: float = 1.0,
) -> np.ndarray:
    """Compute an orthogonalised impulse-response function.

    An IRF answers: "If we give a one-time shock to variable `shock_idx`
    at time 0, how do ALL variables respond over the next `steps` periods?"

    The algorithm:
      1. Compute the "MA representation" matrices Φ_0, Φ_1, …, Φ_steps
         where Φ_s shows the effect of a reduced-form shock at horizon s.
         - Φ_0 = I  (identity — a shock at time 0 has full unit effect)
         - Φ_s = Σ_{j=1}^{min(s,p)} A_j · Φ_{s-j}   (recursive formula)
      2. The structural impulse vector at horizon 0 is the shock_idx-th
         column of B0_inv, scaled by shock_size.
      3. The response at horizon s is:  Φ_s @ impulse_vector

    Parameters
    ----------
    result : dict
        Output from fit_var().
    steps : int, default 30
        Number of periods (days) to trace the response.
    shock_idx : int, default 0
        Which structural shock to hit (0 = wind, 3 = price).
    shock_size : float, default 1.0
        Size of the shock in standard deviations.

    Returns
    -------
    np.ndarray, shape (steps + 1, K)
        Response of each variable at horizons 0, 1, …, steps.
        Row 0 is the immediate (impact) response.
    """
    A = result["A"]           # list of (K, K) coefficient matrices
    B0_inv = result["B0_inv"]  # (K, K) Cholesky factor
    p = len(A)                 # number of lags
    k = B0_inv.shape[0]        # number of variables (K = 4)

    # --- Step 1: Compute MA (moving-average) representation matrices ---
    # Phi[s] is a (K × K) matrix showing how a unit reduced-form shock
    # at time 0 propagates to time s through the VAR dynamics.
    Phi = np.zeros((steps + 1, k, k))  # shape: (steps+1, K, K)
    Phi[0] = np.eye(k)                 # at horizon 0: identity (full effect)

    for s in range(1, steps + 1):
        # Recursive formula: Φ_s = A_1·Φ_{s-1} + A_2·Φ_{s-2} + … + A_p·Φ_{s-p}
        # We only sum up to min(s, p) because Φ is zero for negative indices
        for j in range(1, min(s, p) + 1):
            # A[j-1] is the coefficient matrix for lag j (0-indexed list)
            # Phi[s - j] is the MA matrix at the earlier horizon
            # The @ operator does matrix multiplication
            Phi[s] += A[j - 1] @ Phi[s - j]

    # --- Step 2: Create the structural impulse vector ---
    # The shock_idx-th column of B0_inv tells us how a 1-s.d. structural
    # shock to variable shock_idx feeds into the reduced-form innovations
    # of ALL variables at impact (time 0).
    impulse = B0_inv[:, shock_idx] * shock_size   # shape: (K,)

    # --- Step 3: Compute IRF ---
    # For each horizon s, multiply Φ_s (K×K) by impulse (K,) to get (K,)
    # Phi has shape (steps+1, K, K), impulse has shape (K,)
    # The @ broadcast: each Phi[s] @ impulse → shape (K,)
    # Result shape: (steps+1, K)
    return Phi @ impulse


# ═══════════════════════════════════════════════════════════════════════════
# 5. BOOTSTRAP CONFIDENCE BANDS FOR IRFs
# ═══════════════════════════════════════════════════════════════════════════

def bootstrap_irf(
    result: dict[str, Any],
    data: pd.DataFrame | None = None,
    steps: int = 30,
    shock_idx: int = 0,
    shock_size: float = 1.0,
    n_boot: int = 200,
    ci: float = 0.90,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute IRF point estimate and bootstrap confidence bands.

    The bootstrap works by:
      1. Taking the OLS residuals and resampling them with replacement
         (like shuffling and re-drawing with replacement — a "residual bootstrap")
      2. Simulating new fake data from the estimated VAR using these
         reshuffled residuals
      3. Re-estimating the VAR on the fake data and computing a new IRF
      4. Repeating this n_boot times to get a distribution of IRFs
      5. Taking quantiles of this distribution as confidence bands

    Parameters
    ----------
    result : dict
        Output from fit_var().
    data : pd.DataFrame, optional
        Original data (used for initial lag values). Falls back to
        result['initial_values'] if not provided.
    steps : int, default 30
        IRF horizon in days.
    shock_idx : int, default 0
        Which structural shock (0 = wind).
    shock_size : float, default 1.0
        Shock size in standard deviations.
    n_boot : int, default 200
        Number of bootstrap replications.
    ci : float, default 0.90
        Confidence level (0.90 = 90% interval).
    seed : int, default 0
        Random seed for reproducibility.

    Returns
    -------
    point : np.ndarray, shape (steps+1, K)
        Point estimate of the IRF (from the original estimation).
    lower : np.ndarray, shape (steps+1, K)
        Lower bound of the confidence band.
    upper : np.ndarray, shape (steps+1, K)
        Upper bound of the confidence band.
    """
    # Create a reproducible random number generator
    rng = np.random.default_rng(seed)

    names = result["variable_names"]
    p = result["n_lags"]
    T = result["n_obs"]               # number of residual observations

    # De-mean the residuals (center them at zero)
    # .mean(axis=0) computes the mean of each column → shape (K,)
    # Subtracting broadcasts: each row gets the column means subtracted
    residuals = result["residuals"] - result["residuals"].mean(axis=0)

    # Get the initial p rows of data (the "pre-sample" lags)
    # These are used to start the simulated series
    if data is not None:
        Y_init = data[names].to_numpy(dtype=float)[:p]    # shape: (p, K)
    else:
        Y_init = result["initial_values"]                  # shape: (p, K)

    # The point estimate is just the IRF from the original fitted model
    point = compute_irf(result, steps, shock_idx, shock_size)

    # Storage for all bootstrap IRF replications
    boot_irfs = np.empty((n_boot, steps + 1, len(names)))

    valid = 0                     # count of successful bootstrap draws
    attempts = 0
    max_attempts = n_boot * 5     # safety limit to avoid infinite loops

    while valid < n_boot and attempts < max_attempts:
        attempts += 1

        # --- (a) Resample residuals with replacement ---
        # rng.choice(T, size=T, replace=True) draws T random integers
        # from [0, T), with replacement — some residuals will be picked
        # multiple times, others not at all. This is the bootstrap.
        idx = rng.choice(T, size=T, replace=True)
        resampled_residuals = residuals[idx]               # shape: (T, K)

        # --- (b) Simulate a new dataset using the VAR dynamics ---
        Y_star = _simulate_from_estimates(result, resampled_residuals, Y_init)

        try:
            # --- (c) Re-estimate the VAR on the simulated data ---
            _, A_b, _, sigma_u_b = _ols_var_core(Y_star, p)

            # --- (d) New Cholesky decomposition ---
            B0_inv_b = cholesky(sigma_u_b)

            # --- (e) Compute IRF for this bootstrap draw ---
            boot_irfs[valid] = compute_irf(
                {"A": A_b, "B0_inv": B0_inv_b}, steps, shock_idx, shock_size
            )
            valid += 1

        except (np.linalg.LinAlgError, ValueError):
            # Some bootstrap draws may produce non-positive-definite covariance
            # matrices or explosive VARs — we just skip those and try again
            continue

    # --- Compute confidence bands from the bootstrap distribution ---
    # For a 90% CI, alpha = 0.05, so we take the 5th and 95th percentiles
    alpha = (1.0 - ci) / 2.0                                        # e.g. 0.05
    lower = np.quantile(boot_irfs[:valid], alpha, axis=0)            # 5th percentile
    upper = np.quantile(boot_irfs[:valid], 1.0 - alpha, axis=0)     # 95th percentile

    return point, lower, upper


# ═══════════════════════════════════════════════════════════════════════════
# 6. SIMULATION HELPER  — generates synthetic data from estimated VAR
# ═══════════════════════════════════════════════════════════════════════════

def _simulate_from_estimates(
    result: dict[str, Any],
    u_star: np.ndarray,
    Y_init: np.ndarray | None = None,
) -> np.ndarray:
    """Forward-simulate a time series using estimated VAR coefficients.

    Starting from the initial p rows (pre-sample lags), this function
    generates new data one row at a time using the VAR equation:

        y_t = c + A_1 · y_{t-1} + A_2 · y_{t-2} + … + A_p · y_{t-p} + u_t

    where u_t is taken from the supplied residuals (u_star).

    Parameters
    ----------
    result : dict
        Output from fit_var() (needs 'intercept', 'A', and optionally
        'initial_values').
    u_star : np.ndarray, shape (T, K)
        Residuals to inject (typically resampled bootstrap residuals).
    Y_init : np.ndarray, shape (p, K), optional
        Starting values for the first p rows.

    Returns
    -------
    np.ndarray, shape (T + p, K)
        Simulated data. First p rows are the initial values;
        rows p, p+1, …, T+p-1 are generated.
    """
    intercept = result["intercept"]     # shape: (K,)
    A_list = result["A"]                # list of p arrays, each (K, K)
    p = len(A_list)                     # number of lags
    k = intercept.shape[0]              # number of variables (K)
    T = u_star.shape[0]                 # number of time steps to simulate

    # Allocate output array: T simulated rows + p initial rows
    Y_star = np.zeros((T + p, k))

    # Fill in the initial lags (pre-sample values)
    if Y_init is not None:
        Y_star[:p] = Y_init
    elif "initial_values" in result:
        Y_star[:p] = result["initial_values"]

    # --- Pre-stack the coefficient matrices for speed ---
    # Instead of doing p separate matrix multiplications per time step,
    # we stack [A_1 | A_2 | … | A_p] horizontally into one big (K, K*p) matrix.
    # Then a single matrix-vector multiply handles all lags at once.
    A_stacked = np.hstack(A_list)       # shape: (K, K*p)

    # --- Generate data one row at a time ---
    for t in range(p, T + p):
        # Collect the lagged values into a single long vector:
        #   [y_{t-1}, y_{t-2}, …, y_{t-p}]  flattened to shape (K*p,)
        #
        # Y_star[t-p : t] grabs the p most recent rows → shape (p, K)
        # [::-1] reverses the row order so lag 1 comes first
        #   (because Y_star[t-1] is the most recent, and it's last in the slice)
        # .ravel() flattens (p, K) into a 1D array of length K*p
        lags_flat = Y_star[t - p : t][::-1].ravel()      # shape: (K*p,)

        # Apply the VAR equation:
        #   y_t = intercept + A_stacked @ lags_flat + u_t
        #
        # A_stacked @ lags_flat: (K, K*p) @ (K*p,) → (K,)
        # This is equivalent to: A_1·y_{t-1} + A_2·y_{t-2} + … + A_p·y_{t-p}
        Y_star[t] = intercept + A_stacked @ lags_flat + u_star[t - p]

    return Y_star