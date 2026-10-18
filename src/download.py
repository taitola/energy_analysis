"""
download.py
===========
Data loader for the Finnish electricity-market SVAR using Fingrid Open Data REST API.

Connects to https://data.fingrid.fi/api to download electricity market time series:
- wind_mw: Wind generation (Dataset ID 75)
- load_mw: Total consumption (Dataset ID 124)
- net_flow_mw: Net cross-border transmission flow (Dataset ID 194)
- price_eur: Electricity price (Dataset ID 319)

No classes or OOP: pure, modular functions for downloading, aggregating to daily
frequency, caching, and loading.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Canonical variable ordering (Cholesky identification order)
VARIABLE_ORDER: list[str] = ["wind_mw", "load_mw", "net_flow_mw", "price_eur"]

# Fingrid Open Data dataset IDs
DATASET_IDS: dict[str, int] = {
    "wind_mw": 75,
    "load_mw": 124,
    "net_flow_mw": 194,
    "price_eur": 319,
}

API_BASE_URL: str = "https://data.fingrid.fi/api"


def get_api_key(api_key: str | None = None) -> str:
    """Retrieve the Fingrid API key from parameter or environment.

    Parameters
    ----------
    api_key : str, optional
        Explicit API key. If None, reads from FINGRID_API_KEY environment variable.

    Returns
    -------
    str
        Valid Fingrid API key.

    Raises
    ------
    ValueError
        If no API key is provided or if it still contains the placeholder value.
    """
    key = api_key or os.getenv("FINGRID_API_KEY", "").strip()
    if not key or key == "your_fingrid_api_key_here":
        raise ValueError(
            "Fingrid API key not found or still set to placeholder.\n"
            "Please paste your API key in the .env file:\n"
            "    FINGRID_API_KEY=your_actual_key\n"
            "Register for free at https://data.fingrid.fi/ to obtain an API key."
        )
    return key


def download_dataset(
    dataset_id: int,
    start_time: str,
    end_time: str,
    api_key: str | None = None,
    page_size: int = 10000,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> pd.DataFrame:
    """Download observations for a single Fingrid dataset ID.

    Parameters
    ----------
    dataset_id : int
        Fingrid dataset identifier (e.g. 75, 124, 194, 319).
    start_time : str
        ISO-8601 start timestamp in UTC, e.g. '2023-01-01T00:00:00Z'.
    end_time : str
        ISO-8601 end timestamp in UTC, e.g. '2023-12-31T23:59:59Z'.
    api_key : str, optional
        Fingrid API key. If None, reads FINGRID_API_KEY from environment/.env.
    page_size : int, default 10000
        Number of records per pagination request.
    max_retries : int, default 3
        Number of retries on 429 rate limit or network error.
    retry_delay : float, default 2.0
        Delay in seconds between retries or rate-limit cooloff.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['startTime', 'endTime', 'value'].

    Raises
    ------
    ValueError
        If API key is missing or authentication fails (HTTP 401/403).
    RuntimeError
        If download fails after exhausting retries.
    """
    key = get_api_key(api_key)
    url = f"{API_BASE_URL}/datasets/{dataset_id}/data"
    headers = {
        "x-api-key": key,
        "Accept": "application/json",
    }

    all_records: list[dict[str, Any]] = []
    page = 1

    while True:
        params: dict[str, Any] = {
            "startTime": start_time,
            "endTime": end_time,
            "page": page,
            "pageSize": page_size,
        }

        response = None
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(url, headers=headers, params=params, timeout=30)
                if response.status_code == 200:
                    break
                elif response.status_code == 429:
                    wait_time = retry_delay * attempt
                    print(f"Rate limited (429) for dataset {dataset_id}. Waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                elif response.status_code in (401, 403):
                    raise ValueError(
                        f"Authentication failed (HTTP {response.status_code}): {response.text}\n"
                        "Check that your FINGRID_API_KEY in .env is valid and subscribed."
                    )
                else:
                    print(f"HTTP {response.status_code} on attempt {attempt}: {response.text}")
                    time.sleep(retry_delay)
            except requests.RequestException as exc:
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Failed to fetch dataset {dataset_id} page {page} after {max_retries} attempts: {exc}"
                    ) from exc
                time.sleep(retry_delay)

        if response is None or response.status_code != 200:
            raise RuntimeError(
                f"Failed to fetch dataset {dataset_id} page {page}: "
                f"HTTP {response.status_code if response else 'No response'}"
            )

        payload = response.json()
        if isinstance(payload, dict):
            records = payload.get("data", [])
        elif isinstance(payload, list):
            records = payload
        else:
            records = []

        if not records:
            break

        all_records.extend(records)

        if len(records) < page_size:
            break

        page += 1
        # Friendly pacing to respect Fingrid's 1 call / 2s limit
        time.sleep(2.0)

    if not all_records:
        return pd.DataFrame(columns=["startTime", "endTime", "value"])

    df = pd.DataFrame(all_records)
    cols = [c for c in ["startTime", "endTime", "value"] if c in df.columns]
    df = df[cols].copy()
    if "startTime" in df.columns:
        df["startTime"] = pd.to_datetime(df["startTime"], utc=True)
    if "endTime" in df.columns:
        df["endTime"] = pd.to_datetime(df["endTime"], utc=True)
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df


def to_daily_series(
    df: pd.DataFrame,
    time_col: str = "startTime",
    val_col: str = "value",
    name: str = "value",
) -> pd.Series:
    """Aggregate a high-frequency time series DataFrame to daily mean values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a timestamp column and a numeric value column.
    time_col : str, default 'startTime'
        Name of the timestamp column.
    val_col : str, default 'value'
        Name of the numeric observation column.
    name : str, default 'value'
        Name to give the resulting Series.

    Returns
    -------
    pd.Series
        Daily mean series indexed by DatetimeIndex (daily frequency, normalized).
    """
    if df.empty:
        return pd.Series(dtype=float, name=name)

    temp = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(temp[time_col]):
        temp[time_col] = pd.to_datetime(temp[time_col], utc=True)

    # Normalize to date component and aggregate daily mean
    temp["date"] = temp[time_col].dt.tz_convert("UTC").dt.floor("D")
    daily = temp.groupby("date")[val_col].mean()
    daily.index = pd.DatetimeIndex(daily.index)
    daily.name = name
    return daily


def download_all_data(
    start_time: str,
    end_time: str,
    api_key: str | None = None,
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """Download all 4 endogenous market variables and combine them into a daily DataFrame.

    Parameters
    ----------
    start_time : str
        ISO-8601 start timestamp in UTC (e.g. '2023-01-01T00:00:00Z').
    end_time : str
        ISO-8601 end timestamp in UTC (e.g. '2023-12-31T23:59:59Z').
    api_key : str, optional
        Fingrid API key. If None, reads from .env.
    save_path : str or Path, optional
        Optional file path to persist the merged daily DataFrame (e.g. 'data/fingrid_daily.csv').

    Returns
    -------
    pd.DataFrame
        Daily DataFrame indexed by date with columns matching VARIABLE_ORDER:
        ['wind_mw', 'load_mw', 'net_flow_mw', 'price_eur'].
    """
    series_dict: dict[str, pd.Series] = {}

    for var_name in VARIABLE_ORDER:
        dataset_id = DATASET_IDS[var_name]
        print(f"Downloading {var_name} (Dataset ID {dataset_id}) from Fingrid API...")
        raw_df = download_dataset(
            dataset_id=dataset_id,
            start_time=start_time,
            end_time=end_time,
            api_key=api_key,
        )
        daily_series = to_daily_series(raw_df, name=var_name)
        series_dict[var_name] = daily_series
        print(f"  -> {var_name}: {len(daily_series)} daily observations retrieved.")
        # Pause between datasets to avoid rate limiting
        time.sleep(2.0)

    merged = pd.DataFrame(series_dict)[VARIABLE_ORDER]
    merged.index.name = "date"
    merged = merged.dropna()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out)
        print(f"Saved daily dataset to {out} ({len(merged)} rows).")

    return merged


def load_daily_data(
    file_path: str | Path = "data/fingrid_daily.csv",
    start_time: str | None = None,
    end_time: str | None = None,
    api_key: str | None = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """Load daily data from local CSV file or download from Fingrid API if not found.

    Parameters
    ----------
    file_path : str or Path, default 'data/fingrid_daily.csv'
        Path to cached csv file.
    start_time : str, optional
        Start timestamp if downloading. Defaults to 2 years before today.
    end_time : str, optional
        End timestamp if downloading. Defaults to yesterday.
    api_key : str, optional
        Fingrid API key.
    force_download : bool, default False
        If True, ignores existing cached file and re-downloads.

    Returns
    -------
    pd.DataFrame
        Clean daily DataFrame with columns matching VARIABLE_ORDER.
    """
    path = Path(file_path)
    if path.exists() and not force_download:
        print(f"Loading cached daily data from {path} ...")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df[VARIABLE_ORDER]

    # Defaults for download if dates are omitted: previous 365 days
    if end_time is None:
        end_time = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT23:59:59Z")
    if start_time is None:
        start_time = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365)).strftime("%Y-%m-%dT00:00:00Z")

    print(f"Cache {path} not found. Initiating download ({start_time} to {end_time}) ...")
    return download_all_data(
        start_time=start_time,
        end_time=end_time,
        api_key=api_key,
        save_path=path,
    )
