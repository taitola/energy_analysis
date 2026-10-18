"""
test_download.py
================
Unit tests for data download and preprocessing functions in src.download.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is on sys.path so tests can run directly or via pytest
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.download import (
    DATASET_IDS,
    VARIABLE_ORDER,
    download_dataset,
    get_api_key,
    to_daily_series,
)


def test_get_api_key_raises_on_placeholder(monkeypatch):
    """Placeholder API key must raise a helpful ValueError."""
    monkeypatch.setenv("FINGRID_API_KEY", "your_fingrid_api_key_here")
    with pytest.raises(ValueError, match="Fingrid API key not found"):
        get_api_key()


def test_get_api_key_raises_on_empty(monkeypatch):
    """Empty API key must raise a ValueError."""
    monkeypatch.setenv("FINGRID_API_KEY", "")
    with pytest.raises(ValueError, match="Fingrid API key not found"):
        get_api_key()


def test_get_api_key_success(monkeypatch):
    """Valid key from parameter or env var should be returned."""
    monkeypatch.setenv("FINGRID_API_KEY", "test_env_key_123")
    assert get_api_key() == "test_env_key_123"
    assert get_api_key("explicit_key_456") == "explicit_key_456"


def test_to_daily_series_aggregation():
    """High-frequency values should correctly aggregate to daily mean."""
    timestamps = [
        "2024-01-01T00:00:00Z",
        "2024-01-01T06:00:00Z",
        "2024-01-01T12:00:00Z",
        "2024-01-01T18:00:00Z",
        "2024-01-02T00:00:00Z",
        "2024-01-02T12:00:00Z",
    ]
    values = [100.0, 200.0, 300.0, 400.0, 50.0, 150.0]
    df = pd.DataFrame({"startTime": timestamps, "value": values})

    daily = to_daily_series(df, name="wind_mw")

    assert len(daily) == 2
    assert daily.iloc[0] == 250.0  # mean of 100, 200, 300, 400
    assert daily.iloc[1] == 100.0  # mean of 50, 150
    assert daily.name == "wind_mw"
    assert pd.api.types.is_datetime64_any_dtype(daily.index)


def test_to_daily_series_empty():
    """Empty DataFrame returns an empty Series."""
    empty_df = pd.DataFrame(columns=["startTime", "value"])
    s = to_daily_series(empty_df, name="load_mw")
    assert s.empty
    assert s.name == "load_mw"


@patch("src.download.requests.get")
def test_download_dataset_mocked(mock_get):
    """download_dataset should paginate and parse JSON records into a DataFrame."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": [
            {"startTime": "2024-01-01T00:00:00Z", "endTime": "2024-01-01T01:00:00Z", "value": 1500.5},
            {"startTime": "2024-01-01T01:00:00Z", "endTime": "2024-01-01T02:00:00Z", "value": 1600.0},
        ],
        "pagination": {"total": 2},
    }
    mock_get.return_value = mock_resp

    df = download_dataset(
        dataset_id=75,
        start_time="2024-01-01T00:00:00Z",
        end_time="2024-01-02T00:00:00Z",
        api_key="mock_key",
        page_size=10,
    )

    assert len(df) == 2
    assert list(df.columns) == ["startTime", "endTime", "value"]
    assert df["value"].iloc[0] == 1500.5
    assert df["value"].iloc[1] == 1600.0
    assert mock_get.called


def test_variable_order_and_dataset_ids():
    """Ensure canonical ordering and dataset IDs match specifications."""
    assert VARIABLE_ORDER == ["wind_mw", "load_mw", "net_flow_mw", "price_eur"]
    for var in VARIABLE_ORDER:
        assert var in DATASET_IDS
        assert isinstance(DATASET_IDS[var], int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
