"""
TradePulse - Step 2: Data Validation and Quality Assurance

Contains pure pipeline validation logic (no plotting or reporting):
1. Raw data invariant validation (schema, nulls, OHLC consistency, positive prices, VWAP bounds).
2. Cleaned data validation against exchange schedule (monotone timestamps, grid alignment, flag checks).
3. Data quality and fill rate summary metrics computation.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

import pandas as pd
import numpy as np
import pandas_market_calendars as mcal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing import (
    EXCHANGE_CALENDAR,
    EXCHANGE_TIMEZONE,
    build_daily_5min_grid,
    get_exchange_schedule,
    PROJECT_ROOT,
    PROCESSED_CSV_PATH,
    RAW_DATA_PATH,
)

METADATA_DIR = PROJECT_ROOT / "artifacts" / "metadata"


# ==============================================================================
# Pipeline Validation Routines
# ==============================================================================

def validate_raw_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validates structural and physical price invariants on raw market bars.
    Returns (is_valid, list_of_issues).
    """
    issues = []
    required_cols = [
        "symbol",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "vwap",
    ]

    # Required columns
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns in raw data: {missing_cols}")
        return False, issues

    # Null checks
    null_counts = df[required_cols].isna().sum()
    if null_counts.any():
        issues.append(f"Null values found in columns: {null_counts[null_counts > 0].to_dict()}")

    # Chronological monotonicity
    ts = pd.to_datetime(df["timestamp"], utc=True)
    if not ts.is_monotonic_increasing:
        issues.append("Timestamps are not monotonically increasing.")

    # Duplicate timestamps
    dups = df.duplicated(subset=["symbol", "timestamp"]).sum()
    if dups > 0:
        issues.append(f"Found {dups} duplicate (symbol, timestamp) records.")

    # Positive prices
    for col in ["open", "high", "low", "close"]:
        if (df[col] <= 0).any():
            bad_count = int((df[col] <= 0).sum())
            issues.append(f"Column '{col}' has {bad_count} non-positive values.")

    # OHLC geometry
    if not (df["high"] >= df["low"]).all():
        issues.append("Violation: high < low found.")
    if not (df["high"] >= df["open"]).all():
        issues.append("Violation: high < open found.")
    if not (df["high"] >= df["close"]).all():
        issues.append("Violation: high < close found.")
    if not (df["low"] <= df["open"]).all():
        issues.append("Violation: low > open found.")
    if not (df["low"] <= df["close"]).all():
        issues.append("Violation: low > close found.")

    # VWAP bounds (allowing 1e-4 float precision tolerance)
    vwap_violations = (df["vwap"] < df["low"] - 1e-4) | (df["vwap"] > df["high"] + 1e-4)
    if vwap_violations.any():
        issues.append(f"VWAP out of [low, high] bounds in {int(vwap_violations.sum())} bars.")

    # Non-negative volume
    if (df["volume"] < 0).any():
        issues.append("Negative volume values detected.")
    if (df["trade_count"] < 0).any():
        issues.append("Negative trade_count values detected.")

    is_valid = len(issues) == 0
    return is_valid, issues


def validate_cleaned_dataframe(
    df: pd.DataFrame,
    calendar_name: str = EXCHANGE_CALENDAR,
) -> Tuple[bool, List[str]]:
    """
    Validates that cleaned dataset matches the official NASDAQ schedule,
    has zero nulls, strictly adheres to 5-minute alignment, and correctly tags filled bars.
    Returns (is_valid, list_of_issues).
    """
    issues = []
    required_cols = [
        "symbol",
        "timestamp",
        "timestamp_ny",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "vwap",
        "is_filled",
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns in clean data: {missing_cols}")
        return False, issues

    if df.isna().any().any():
        issues.append(f"Null values present in cleaned data: {df.isna().sum().to_dict()}")

    ts = pd.to_datetime(df["timestamp"], utc=True)
    if not ts.is_monotonic_increasing:
        issues.append("Cleaned timestamps are not monotonically increasing.")

    # 5-minute alignment check
    misaligned = ts[ts.dt.minute % 5 != 0]
    if len(misaligned) > 0:
        issues.append(f"Found {len(misaligned)} bars not aligned to 5-minute intervals.")

    seconds_not_zero = ts[ts.dt.second != 0]
    if len(seconds_not_zero) > 0:
        issues.append(f"Found {len(seconds_not_zero)} bars with non-zero seconds.")

    # `is_filled` binary check
    invalid_filled = ~df["is_filled"].isin([0, 1])
    if invalid_filled.any():
        issues.append(f"Invalid `is_filled` values found: {df['is_filled'].unique()}")

    # Geometric price invariants
    if not (df["high"] >= df["low"]).all():
        issues.append("Violation: high < low in cleaned data.")
    if not (df["high"] >= df["open"]).all():
        issues.append("Violation: high < open in cleaned data.")
    if not (df["high"] >= df["close"]).all():
        issues.append("Violation: high < close in cleaned data.")
    if not (df["low"] <= df["open"]).all():
        issues.append("Violation: low > open in cleaned data.")
    if not (df["low"] <= df["close"]).all():
        issues.append("Violation: low > close in cleaned data.")

    # Check session adherence against pandas-market-calendars
    start_date = ts.min().date()
    end_date = ts.max().date()
    schedule = get_exchange_schedule(start_date, end_date, calendar_name)

    total_expected_bars = 0
    for _, session_row in schedule.iterrows():
        grid = build_daily_5min_grid(session_row["market_open"], session_row["market_close"])
        total_expected_bars += len(grid)

    if len(df) != total_expected_bars:
        issues.append(
            f"Row count mismatch: cleaned data has {len(df)} bars, but official schedule "
            f"expects {total_expected_bars} bars across {len(schedule)} sessions."
        )

    is_valid = len(issues) == 0
    return is_valid, issues


# ==============================================================================
# Quality Metrics Computation (Pure Data, No UI / Plotting)
# ==============================================================================

def compute_audit_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes data quality metrics, fill rates, and intraday return distribution statistics.
    """
    ts_ny = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(EXCHANGE_TIMEZONE)
    total_bars = len(df)
    filled_bars = int(df["is_filled"].sum())
    observed_bars = total_bars - filled_bars
    fill_rate_pct = (filled_bars / total_bars * 100.0) if total_bars > 0 else 0.0

    df_temp = df.copy()
    df_temp["hour_et"] = ts_ny.dt.hour
    hourly_summary = (
        df_temp.groupby("hour_et")
        .agg(
            total_bars=("is_filled", "count"),
            filled_bars=("is_filled", "sum"),
            avg_volume=("volume", "mean"),
            avg_trades=("trade_count", "mean"),
        )
        .reset_index()
    )
    hourly_summary["fill_rate_pct"] = (
        hourly_summary["filled_bars"] / hourly_summary["total_bars"] * 100.0
    )

    df_temp["session_date"] = ts_ny.dt.date
    df_temp["ret_5m"] = df_temp.groupby("session_date")["close"].pct_change()
    returns = df_temp["ret_5m"].dropna()

    metrics = {
        "total_bars": total_bars,
        "observed_bars": observed_bars,
        "filled_bars": filled_bars,
        "fill_rate_pct": float(round(fill_rate_pct, 4)),
        "trading_sessions_count": int(df_temp["session_date"].nunique()),
        "date_min": str(df["timestamp"].min()),
        "date_max": str(df["timestamp"].max()),
        "price_min": float(df["close"].min()),
        "price_max": float(df["close"].max()),
        "returns_mean": float(returns.mean()),
        "returns_std": float(returns.std()),
        "returns_min": float(returns.min()),
        "returns_max": float(returns.max()),
        "returns_kurtosis": float(returns.kurtosis()),
        "returns_skew": float(returns.skew()),
        "spikes_above_3pct_count": int((returns.abs() > 0.03).sum()),
        "spikes_above_2pct_count": int((returns.abs() > 0.02).sum()),
        "hourly_breakdown": hourly_summary.to_dict(orient="records"),
    }

    return metrics


# ==============================================================================
# Metadata & Data Dictionary Exporter
# ==============================================================================

def export_data_dictionary(output_json: Path, output_md: Optional[Path] = None):
    """
    Saves the canonical data dictionary for clean processed bars.
    """
    dictionary_data = {
        "dataset_name": "META_5min_clean",
        "description": "Validated and calendar-reindexed 5-minute OHLCV bars for META (NASDAQ) from Alpaca IEX feed.",
        "timezone_storage": "UTC (timestamp column) and America/New_York (timestamp_ny column)",
        "frequency": "5 minutes",
        "trading_sessions": "Regular trading sessions (9:30 AM - 4:00 PM ET; 9:30 AM - 1:00 PM ET on half-days)",
        "fields": [
            {
                "name": "symbol",
                "dtype": "string",
                "unit": "ticker",
                "description": "Equity ticker symbol ('META').",
                "missing_policy": "Non-nullable."
            },
            {
                "name": "timestamp",
                "dtype": "datetime64[ns, UTC]",
                "unit": "ISO-8601 UTC",
                "description": "Bar start timestamp in UTC.",
                "missing_policy": "Reindexed to NASDAQ calendar grid."
            },
            {
                "name": "timestamp_ny",
                "dtype": "datetime64[ns, America/New_York]",
                "unit": "ISO-8601 ET",
                "description": "Bar start timestamp in local exchange time (America/New_York).",
                "missing_policy": "Derived from timestamp."
            },
            {
                "name": "open",
                "dtype": "float64",
                "unit": "USD",
                "description": "Opening trade price in the 5-minute interval.",
                "missing_policy": "Forward-filled from prior close if is_filled=1."
            },
            {
                "name": "high",
                "dtype": "float64",
                "unit": "USD",
                "description": "Highest trade price in the 5-minute interval.",
                "missing_policy": "Forward-filled from prior close if is_filled=1."
            },
            {
                "name": "low",
                "dtype": "float64",
                "unit": "USD",
                "description": "Lowest trade price in the 5-minute interval.",
                "missing_policy": "Forward-filled from prior close if is_filled=1."
            },
            {
                "name": "close",
                "dtype": "float64",
                "unit": "USD",
                "description": "Closing trade price in the 5-minute interval.",
                "missing_policy": "Forward-filled from prior close if is_filled=1."
            },
            {
                "name": "volume",
                "dtype": "float64",
                "unit": "shares",
                "description": "Total shares executed on IEX in the 5-minute interval.",
                "missing_policy": "0.0 if is_filled=1."
            },
            {
                "name": "trade_count",
                "dtype": "float64",
                "unit": "trades",
                "description": "Total number of transactions executed on IEX in the interval.",
                "missing_policy": "0.0 if is_filled=1."
            },
            {
                "name": "vwap",
                "dtype": "float64",
                "unit": "USD",
                "description": "Volume-Weighted Average Price for the interval.",
                "missing_policy": "Forward-filled from prior vwap if is_filled=1."
            },
            {
                "name": "is_filled",
                "dtype": "int64",
                "unit": "binary flag",
                "description": "Data quality flag: 0 = observed market trade bar; 1 = synthesized missing bar.",
                "missing_policy": "Non-nullable."
            }
        ]
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(dictionary_data, f, indent=2)

    if output_md:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        md_lines = [
            "# TradePulse: Data Dictionary",
            "",
            f"**Dataset:** `{dictionary_data['dataset_name']}`  ",
            f"**Description:** {dictionary_data['description']}  ",
            f"**Frequency:** {dictionary_data['frequency']}  ",
            f"**Sessions:** {dictionary_data['trading_sessions']}  ",
            "",
            "## Schema & Specifications",
            "",
            "| Column | Type | Unit | Description | Missing / Synthetic Policy |",
            "|---|---|---|---|---|",
        ]
        for fld in dictionary_data["fields"]:
            md_lines.append(
                f"| `{fld['name']}` | `{fld['dtype']}` | {fld['unit']} | {fld['description']} | {fld['missing_policy']} |"
            )
        with open(output_md, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")


if __name__ == "__main__":
    print("[ml/validation.py] Running pipeline sanity checks...")
    raw = pd.read_csv(RAW_DATA_PATH)
    r_valid, r_issues = validate_raw_dataframe(raw)
    print(f"  Raw dataset valid: {r_valid} ({len(r_issues)} issues)")

    clean = pd.read_csv(PROCESSED_CSV_PATH)
    c_valid, c_issues = validate_cleaned_dataframe(clean)
    print(f"  Clean dataset valid: {c_valid} ({len(c_issues)} issues)")

    metrics = compute_audit_metrics(clean)
    print(f"  Total bars: {metrics['total_bars']:,} | Filled: {metrics['filled_bars']} ({metrics['fill_rate_pct']}%)")
    export_data_dictionary(
        output_json=METADATA_DIR / "data_dictionary.json",
        output_md=PROJECT_ROOT / "reports" / "data_dictionary.md",
    )
    print("  Validation pipeline check passed!")
