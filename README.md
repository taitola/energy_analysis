# Finnish Electricity Market SVAR Analysis (`energy-svar-project`)

Production-grade Python econometrics library that estimates a **Structural Vector Autoregression (SVAR)** for the Finnish electricity market using Fingrid Open Data. The library delivers automated companion-matrix stability diagnostics, recursive (Cholesky) structural identification, residual bootstrap impulse-response functions (IRFs), and dynamic counterfactual policy simulations.

---

## 1. Model Specification

The model operates at a **daily frequency** over a 4-variable endogenous system:

| Ordering | Variable | Fingrid ID | Units | Description & Role |
| :---: | :--- | :---: | :---: | :--- |
| **0** | `wind_mw` | 75 | MW | Daily mean wind generation (strictly predetermined weather shock) |
| **1** | `load_mw` | 124 | MW | Daily total electricity consumption |
| **2** | `net_flow_mw` | 194 | MW | Daily net cross-border transmission flow (positive = net export) |
| **3** | `price_eur` | 319 | EUR/MWh | Daily volume-weighted spot / imbalance price (fast-clearing price) |

### Econometric Methodology

* **Estimator**: Reduced-form $\text{VAR}(p=7)$ estimated in levels via Ordinary Least Squares (OLS). Estimation in levels follows *Sims, Stock & Watson (1990)*, ensuring consistency regardless of integration or cointegration properties.
* **Stability Diagnostic**: Verifies that every eigenvalue $\lambda$ of the $(K \cdot p \times K \cdot p)$ companion matrix satisfies $|\lambda| < 1.0$.
* **Identification**: Short-run recursive Cholesky factorization:
  $$\Sigma_u = B_0^{-1} (B_0^{-1})^\top, \quad w_t = B_0 u_t$$
  Wind generation is ordered first (exogenous weather phenomenon), and the electricity spot price is ordered last (absorbing all physical supply and demand shocks contemporaneously).
* **Impulse-Response Functions (IRFs)**: Vector Moving Average (VMA) representation with residual-resampling bootstrap confidence bands (e.g. 90% CI over 30-day horizon).
* **Dynamic Counterfactual Simulator**: Historical dynamic simulation that zeroes selected structural shocks (e.g., wind generation shocks $w_{1,t} = 0$) from a user-specified intervention date $T^*$ onward, propagating all lag interactions endogenously.

---

## 2. Repository Structure

```
energy-svar-project/
├── .env.example               # Environment template for Fingrid API key
├── .gitignore                 # Excludes secrets, data caches, and generated figures
├── .vscode/
│   └── settings.json          # Editor configuration (hides __pycache__ and .pytest_cache)
├── AGENTS.md                  # Project rules, conventions, and mathematical notation
├── README.md                  # Project overview and run guide
├── data/
│   ├── .gitkeep               # Preserves directory in git
│   └── fingrid_daily.csv      # Cached daily market observations (2023)
├── outputs/
│   └── figures/               # Generated analytical figures
│       ├── .gitkeep           # Preserves directory in git
│       ├── irf_wind_shock.png # Orthogonalised IRF (+1 s.d. wind generation shock)
│       └── counterfactual_price.png # Actual vs. counterfactual price paths
├── src/
│   ├── __init__.py
│   ├── download.py            # Fingrid REST API downloader, rate-limiter & resampler
│   ├── svar_estimator.py      # VAR(p) OLS, companion matrix, Cholesky ID, bootstrap IRF
│   └── counterfactual.py      # Historical dynamic counterfactual simulator
├── scripts/
│   └── 01_run_pipeline.py     # End-to-end pipeline execution script
└── tests/
    ├── conftest.py            # Pytest path and environment setup
    ├── test_download.py       # API download & data aggregation unit tests
    └── test_svar.py           # Estimation, companion stability, IRF & simulation tests
```

---

## 3. Setup & Installation

### Prerequisites
* Python $\ge 3.10$

### Dependencies
Install the required packages in your environment:
```bash
pip install numpy pandas scipy statsmodels matplotlib requests pytest python-dotenv
```

### Fingrid API Key Configuration
To download fresh observations directly from the Fingrid Open Data API:
1. Register for a free API key at [data.fingrid.fi](https://data.fingrid.fi/).
2. Create a `.env` file in the project root (or copy `.env.example`):
   ```bash
   cp .env.example .env
   ```
3. Set your API key in `.env`:
   ```env
   FINGRID_API_KEY=your_actual_api_key_here
   ```
*(Note: If `data/fingrid_daily.csv` is already present, the pipeline will load from cache automatically).*

---

## 4. Running the Project

### Running the End-to-End Pipeline
Execute the main pipeline script:
```bash
python scripts/01_run_pipeline.py
```

The pipeline executes 5 automated stages:
1. **Data Loading**: Loads cached daily data from `data/fingrid_daily.csv` (or fetches via REST API).
2. **Estimation**: Fits the reduced-form $\text{VAR}(7)$ via OLS and computes companion-matrix eigenvalues ($|\lambda_{\max}| < 1$).
3. **Identification**: Computes lower-triangular Cholesky factor $B_0^{-1}$.
4. **Impulse-Response Analysis**: Computes orthogonalised $+1\sigma$ wind generation IRFs with 90% bootstrap confidence intervals and saves the figure to:
   * `outputs/figures/irf_wind_shock.png`
5. **Counterfactual Simulation**: Simulates the electricity price trajectory assuming zero structural wind shocks from mid-sample onward ($T^* = \text{2023-07-02}$), saving the comparison to:
   * `outputs/figures/counterfactual_price.png`

### Running Unit Tests
Run the complete test suite with `pytest`:
```bash
pytest
```
Or for verbose test output:
```bash
pytest -v
```

All tests in `tests/test_download.py` and `tests/test_svar.py` run in under 2 seconds without requiring external network access.
