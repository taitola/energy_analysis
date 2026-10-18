# AGENTS.md — energy-svar-project

## Project Overview

Production-grade Python research library that estimates a **Structural Vector
Autoregression (SVAR)** for the Finnish electricity market using Fingrid Open
Data and Nord Pool pricing. The library provides automated statistical
diagnostics and counterfactual policy simulations.

---

## Repository Layout

```
energy-svar-project/
├── .env                   # Local API keys (FINGRID_API_KEY)
├── .env.example           # Environment template
├── .gitignore             # Git ignore file for secrets and data
├── AGENTS.md              # This file — project rules & conventions
├── data/                  # Raw / processed CSV files (git-ignored)
├── outputs/               # Generated analytical figures and reports
│   └── figures/           # IRF, counterfactual, and diagnostic plots
├── src/
│   ├── __init__.py
│   ├── download.py        # Fingrid REST API downloader & daily resampler
│   ├── svar_estimator.py  # VAR(p) estimation, Cholesky ID, bootstrap IRFs
│   └── counterfactual.py  # Historical dynamic counterfactual simulator
├── scripts/
│   └── 01_run_pipeline.py # End-to-end execution script
└── tests/
    ├── conftest.py        # Pytest path setup
    ├── test_download.py   # Download & preprocessing unit tests
    └── test_svar.py       # Estimation, IRF & counterfactual unit tests
```

---

## Model Specification

### Endogenous Variables (daily frequency, ordering matters)

| Position | Variable       | Description                              | Fingrid ID |
|----------|---------------|------------------------------------------|------------|
| 0        | `wind_mw`     | Daily mean wind generation (MW)          | 75         |
| 1        | `load_mw`     | Daily total electricity consumption (MW) | 124        |
| 2        | `net_flow_mw` | Daily net cross-border flow (MW)         | 194        |
| 3        | `price_eur`   | Daily spot / imbalance price (EUR/MWh)   | 319        |

### Econometric Rules

1. **Estimator**: Reduced-form VAR(p=7) estimated in levels via OLS.
   Estimation in levels is consistent with Sims, Stock & Watson (1990) —
   OLS on a VAR in levels is consistent regardless of integration /
   cointegration properties.

2. **Companion-matrix stability check**: Every eigenvalue of the
   (K·p × K·p) companion matrix **must** have modulus strictly < 1.
   If violated, raise `ValueError`.

3. **Identification**: Recursive (Cholesky) short-run identification.
   Wind ordered first (exogenous weather), price ordered last.
   - Σ_u = B₀⁻¹ (B₀⁻¹)ᵀ  (lower-triangular Cholesky factor)
   - Structural shocks: w_t = B₀ u_t

4. **Counterfactual engine**: Dynamic historical simulation that forces
   the first structural shock (wind) to zero from a user-defined T*
   onward, propagating all dynamic feedback endogenously.

---

## Coding Conventions

- **Strictly Functional & Modular**: No classes or OOP. Functions operate on
  pandas DataFrames, NumPy arrays, and standard dictionaries.
- **Python ≥ 3.10** — use `|` union types, `from __future__ import annotations`.
- **Type hints everywhere** — all public functions fully annotated.
- **NumPy for linear algebra**, pandas for time-indexed data.
- **No global mutable state** — pure functions.
- **Docstrings**: NumPy-style (`Parameters`, `Returns`, `Raises`).
- **Testing**: pytest. Every public function has ≥ 1 unit test.
- **Reproducibility**: All RNGs accept an explicit `seed` argument.

---

## Dependencies

```
numpy >= 1.24
pandas >= 2.0
scipy >= 1.11
statsmodels >= 0.14
matplotlib >= 3.7
pytest >= 7.0
requests >= 2.28
python-dotenv >= 1.0
```

---

## Key Mathematical Notation

| Symbol          | Meaning                                         |
|-----------------|--------------------------------------------------|
| K               | Number of endogenous variables (= 4)             |
| p               | VAR lag order (default 7)                         |
| A_i             | (K × K) coefficient matrix for lag i              |
| Σ_u             | (K × K) reduced-form residual covariance          |
| B₀⁻¹           | Lower-triangular Cholesky factor of Σ_u           |
| w_t             | (K × 1) structural shock vector                   |
| y_t^{cf}        | Counterfactual endogenous vector at time t         |
