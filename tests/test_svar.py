"""
test_svar.py
============
Unit tests for SVAR estimation, IRF computation, and counterfactual simulation.
All tests use simple deterministic data without classes or synthetic DGPs.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so tests can run directly or via pytest
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.download import VARIABLE_ORDER
from src.svar_estimator import assemble_companion, compute_irf, fit_var
from src.counterfactual import simulate_counterfactual

@pytest.fixture
def sample_data():
    """Deterministic, stationary daily dataset for fast, clean unit testing."""
    n_obs = 100
    dates = pd.date_range("2024-01-01", periods=n_obs, freq="D")
    rng = np.random.default_rng(42)

    # Generate stationary series with some correlation
    Y = np.zeros((n_obs, 4))
    Y[0] = [2000.0, 8000.0, 500.0, 50.0]
    for t in range(1, n_obs):
        Y[t, 0] = 0.5 * Y[t - 1, 0] + rng.normal(0, 50) + 1000.0
        Y[t, 1] = 0.4 * Y[t - 1, 1] + 0.1 * Y[t, 0] + rng.normal(0, 100) + 4000.0
        Y[t, 2] = 0.3 * Y[t - 1, 2] + rng.normal(0, 20) + 200.0
        Y[t, 3] = 0.2 * Y[t - 1, 3] - 0.01 * Y[t, 0] + rng.normal(0, 5) + 30.0

    df = pd.DataFrame(Y, index=dates, columns=VARIABLE_ORDER)
    return df


def test_companion_matrix_shape():
    """assemble_companion must return (Kp x Kp) array."""
    k, p = 4, 3
    A_list = [np.eye(k) * 0.1 for _ in range(p)]
    companion = assemble_companion(A_list, k)
    assert companion.shape == (k * p, k * p)
    # Check subdiagonal block is identity
    assert np.allclose(companion[k:, : k * (p - 1)], np.eye(k * (p - 1)))


def test_fit_var_returns_dict_and_estimates(sample_data):
    """fit_var should return a clean dictionary with all expected keys."""
    res = fit_var(sample_data, n_lags=3)

    assert isinstance(res, dict)
    expected_keys = {
        "intercept",
        "A",
        "sigma_u",
        "B0_inv",
        "residuals",
        "companion_eigenvalues",
        "max_eigenvalue_modulus",
        "is_stable",
        "structural_shocks",
        "n_obs",
        "n_lags",
        "variable_names",
    }
    assert expected_keys.issubset(res.keys())
    assert res["is_stable"] is True
    assert res["max_eigenvalue_modulus"] < 1.0
    assert res["n_lags"] == 3
    assert res["n_obs"] == len(sample_data) - 3
    assert res["residuals"].shape == (res["n_obs"], 4)
    assert res["structural_shocks"].shape == res["residuals"].shape


def test_cholesky_properties(sample_data):
    """B0_inv must be lower triangular and satisfy sigma_u = B0_inv @ B0_inv.T."""
    res = fit_var(sample_data, n_lags=2)
    B0_inv = res["B0_inv"]
    sigma_u = res["sigma_u"]

    assert np.allclose(B0_inv, np.tril(B0_inv))
    assert np.all(np.diag(B0_inv) > 0)
    assert np.allclose(B0_inv @ B0_inv.T, sigma_u, atol=1e-10)


def test_fit_var_explosive_raises():
    """Explosive VAR must raise ValueError."""
    k, p, n_obs = 4, 2, 50
    dates = pd.date_range("2024-01-01", periods=n_obs, freq="D")
    Y = np.zeros((n_obs, k))
    Y[0] = 1.0
    for t in range(1, n_obs):
        Y[t] = Y[t - 1] * 2.0  # exploding exponential

    df = pd.DataFrame(Y, index=dates, columns=VARIABLE_ORDER)
    with pytest.raises(ValueError, match="explosive"):
        fit_var(df, n_lags=p)


def test_compute_irf_shape_and_impact(sample_data):
    """IRF shape must be (steps + 1, K) and horizon 0 matches B0_inv column."""
    res = fit_var(sample_data, n_lags=2)
    irf = compute_irf(res, steps=15, shock_idx=0, shock_size=1.0)

    assert irf.shape == (16, 4)
    np.testing.assert_allclose(irf[0], res["B0_inv"][:, 0], atol=1e-12)


def test_simulate_counterfactual(sample_data):
    """Pre-T* counterfactual must equal actuals, post-T* must differ."""
    res = fit_var(sample_data, n_lags=2)
    mid_date = sample_data.index[len(sample_data) // 2]

    cf = simulate_counterfactual(
        res, sample_data, intervention_start=mid_date, zeroed_shocks=[0]
    )

    assert isinstance(cf, dict)
    actual = cf["actual"]
    counterfactual = cf["counterfactual"]
    diff = cf["diff"]

    # Pre-T* paths must match
    pre_mask = actual.index < mid_date
    np.testing.assert_allclose(
        counterfactual.loc[pre_mask].values,
        actual.loc[pre_mask].values,
        atol=1e-8,
    )

    # Post-T* diff should not be all zero
    post_diff = diff.loc[mid_date:].abs().sum().sum()
    assert post_diff > 0


def test_simulate_counterfactual_invalid_date(sample_data):
    """Invalid start date must raise ValueError."""
    res = fit_var(sample_data, n_lags=2)
    with pytest.raises(ValueError, match="after the sample end"):
        simulate_counterfactual(res, sample_data, intervention_start="2099-01-01")

    with pytest.raises(ValueError, match="after the first"):
        simulate_counterfactual(res, sample_data, intervention_start=sample_data.index[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
