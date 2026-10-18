"""
energy-svar-project  --  Structural VAR for the Finnish electricity market.

Clean, modular functional API with no classes or OOP.
"""

from __future__ import annotations

from src.download import (
    DATASET_IDS,
    VARIABLE_ORDER,
    download_all_data,
    download_dataset,
    get_api_key,
    load_daily_data,
    to_daily_series,
)
from src.svar_estimator import (
    assemble_companion,
    bootstrap_irf,
    compute_irf,
    fit_var,
)
from src.counterfactual import (
    simulate_counterfactual,
)

__all__ = [
    "DATASET_IDS",
    "VARIABLE_ORDER",
    "download_dataset",
    "download_all_data",
    "load_daily_data",
    "to_daily_series",
    "get_api_key",
    "assemble_companion",
    "fit_var",
    "compute_irf",
    "bootstrap_irf",
    "simulate_counterfactual",
]
