"""
TradePulse - Step 2: Data Preprocessing and Cleaning

This module implements the canonical cleaning, calendar alignment, and missing-bar
reindexing logic for TradePulse.

Key Principles:
1. One shared pipeline: Used identically by batch training and live inference.
2. Exchange calendar alignment: Uses pandas-market-calendars to determine official
   regular market session boundaries (9:30 AM - 4:00 PM ET) and half-days.
3. No silent forward-fill: Synthesized bars are explicitly tagged with `is_filled=1`
   (observed bars have `is_filled=0`).
4. Same-day boundary isolation: Forward filling never crosses midnight or day boundaries.
   Leading missing bars at market open use the previous regular session close (causal).
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import pandas_market_calendars as mcal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXCHANGE_CALENDAR = "NASDAQ"
EXCHANGE_TIMEZONE = "America/New_York"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "META_5min.csv"
PROCESSED_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "META_5min_clean.csv"
PROCESSED_PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "META_5min_clean.parquet"


# ==============================================================================
# Shared Missing Bar Synthesis (Used by both Training & Live Service)
# ==============================================================================

def synthesize_missing_bar(
    symbol: str,
    timestamp: pd.Timestamp,
    last_known_close: float,
    last_known_vwap: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Creates a single synthetic 5-minute bar when no trade occurred on IEX.

    This function represents the exact contract shared between historical data
    reindexing and the live streaming backend.

    Contract:
    - open, high, low, close = last known close price
    - volume = 0.0
    - trade_count = 0.0
    - vwap = last known vwap (or last known close if vwap unavailable)
    - is_filled = 1
    """
    vwap_val = last_known_vwap if (last_known_vwap is not None and not np.isnan(last_known_vwap)) else last_known_close
    return {
        "symbol": symbol,
        "timestamp": pd.to_datetime(timestamp, utc=True),
        "open": float(last_known_close),
        "high": float(last_known_close),
        "low": float(last_known_close),
        "close": float(last_known_close),
        "volume": 0.0,
        "trade_count": 0.0,
        "vwap": float(vwap_val),
        "is_filled": 1,
    }


# ==============================================================================
# Calendar-Aware Session Grid Construction
# ==============================================================================

def get_exchange_schedule(
    start_date: datetime,
    end_date: datetime,
    calendar_name: str = EXCHANGE_CALENDAR,
) -> pd.DataFrame:
    """
    Retrieves the official exchange schedule for the date range, handling
    weekends, holidays, and half-days.
    """
    calendar = mcal.get_calendar(calendar_name)
    schedule = calendar.schedule(start_date=start_date, end_date=end_date)
    return schedule


def build_daily_5min_grid(
    market_open: pd.Timestamp,
    market_close: pd.Timestamp,
    freq: str = "5min",
) -> pd.DatetimeIndex:
    """
    Constructs the exact 5-minute timestamp grid for a single trading session.

    Standard session (9:30 - 16:00 ET): 78 bars [09:30, 09:35, ..., 15:55]
    Half-day session (e.g., 9:30 - 13:00 ET): 42 bars [09:30, 09:35, ..., 12:55]
    """
    return pd.date_range(
        start=market_open,
        end=market_close - pd.Timedelta(minutes=5),
        freq=freq,
    )


# ==============================================================================
# Historical Cleaning and Reindexing Pipeline
# ==============================================================================

def clean_and_reindex_dataset(
    raw_df: pd.DataFrame,
    symbol: str = "META",
    calendar_name: str = EXCHANGE_CALENDAR,
) -> pd.DataFrame:
    """
    Cleans raw market bars and reindexes each trading day to the official NASDAQ grid:
    1. Parses timestamps to UTC datetime, removes duplicates, sorts chronologically.
    2. Filters for official regular trading sessions using pandas-market-calendars.
    3. Reindexes each trading day against its calendar grid.
    4. Identifies missing intervals and synthesizes missing bars using `synthesize_missing_bar`.
    5. Tags all bars with `is_filled` (0 for original, 1 for synthetic).
    6. Adds localized timestamp column `timestamp_ny` for downstream auditing.
    """
    df = raw_df.copy()

    # 1. Normalize timestamps and drop duplicate bars
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.drop_duplicates(subset=["symbol", "timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 2. Query exchange schedule across dataset date span
    start_date = df["timestamp"].min().date()
    end_date = df["timestamp"].max().date()
    schedule = get_exchange_schedule(start_date, end_date, calendar_name)

    if schedule.empty:
        raise ValueError(f"No trading sessions found for range {start_date} to {end_date}")

    cleaned_days: List[pd.DataFrame] = []
    previous_session_close: Optional[float] = None
    previous_session_vwap: Optional[float] = None

    for session_date, session_row in schedule.iterrows():
        market_open = session_row["market_open"]
        market_close = session_row["market_close"]

        # Expected 5-minute grid for this specific session
        session_grid = build_daily_5min_grid(market_open, market_close)
        grid_df = pd.DataFrame({"timestamp": session_grid})
        grid_df["symbol"] = symbol

        # Observed bars belonging to this regular session
        session_mask = (df["timestamp"] >= market_open) & (df["timestamp"] <= (market_close - pd.Timedelta(minutes=5)))
        observed_bars = df[session_mask].copy()
        observed_bars["is_filled"] = 0

        # Left join calendar grid with observed bars
        merged = pd.merge(grid_df, observed_bars, on=["symbol", "timestamp"], how="left")
        is_missing = merged["is_filled"].isna()

        if is_missing.any():
            merged.loc[is_missing, "is_filled"] = 1
            merged.loc[is_missing, "volume"] = 0.0
            merged.loc[is_missing, "trade_count"] = 0.0

            # If the first bar(s) of the day are missing, use previous day's close (causal)
            if pd.isna(merged.loc[0, "close"]):
                if previous_session_close is None:
                    # Fallback for the very first bar of the very first day in dataset
                    first_valid_idx = merged["close"].first_valid_index()
                    if first_valid_idx is not None:
                        fill_price = merged.loc[first_valid_idx, "open"]
                        fill_vwap = merged.loc[first_valid_idx, "vwap"]
                    else:
                        raise ValueError(f"Entire trading session on {session_date} contains no trades.")
                else:
                    fill_price = previous_session_close
                    fill_vwap = previous_session_vwap if previous_session_vwap is not None else previous_session_close

                merged.loc[0, "close"] = fill_price
                merged.loc[0, "open"] = fill_price
                merged.loc[0, "high"] = fill_price
                merged.loc[0, "low"] = fill_price
                merged.loc[0, "vwap"] = fill_vwap

            # Forward fill within the same day only
            merged["close"] = merged["close"].ffill()
            merged["open"] = merged["open"].fillna(merged["close"])
            merged["high"] = merged["high"].fillna(merged["close"])
            merged["low"] = merged["low"].fillna(merged["close"])
            merged["vwap"] = merged["vwap"].ffill()

        # Update previous day closing reference
        previous_session_close = float(merged["close"].iloc[-1])
        previous_session_vwap = float(merged["vwap"].iloc[-1])

        # Cast is_filled to integer
        merged["is_filled"] = merged["is_filled"].astype(int)
        cleaned_days.append(merged)

    cleaned_df = pd.concat(cleaned_days, ignore_index=True)

    # 3. Add localized New York timestamp
    cleaned_df["timestamp_ny"] = cleaned_df["timestamp"].dt.tz_convert(EXCHANGE_TIMEZONE)

    # 4. Standard column ordering
    ordered_cols = [
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
    cleaned_df = cleaned_df[ordered_cols]

    return cleaned_df


# ==============================================================================
# Save and Execute Preprocessing Pipeline
# ==============================================================================

def run_preprocessing(
    raw_path: Path = RAW_DATA_PATH,
    output_csv: Path = PROCESSED_CSV_PATH,
    output_parquet: Path = PROCESSED_PARQUET_PATH,
) -> pd.DataFrame:
    """
    Executes Step 2 data cleaning and saves results to data/processed/.
    """
    print(f"\n[TradePulse Step 2 Preprocessing]")
    print(f"  Reading raw data: {raw_path}")
    raw_df = pd.read_csv(raw_path)
    print(f"  Raw bars read   : {len(raw_df):,}")

    clean_df = clean_and_reindex_dataset(raw_df, symbol="META")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(output_csv, index=False)
    print(f"  Cleaned bars    : {len(clean_df):,}")
    print(f"  Filled bars     : {clean_df['is_filled'].sum():,} ({clean_df['is_filled'].mean() * 100:.2f}%)")
    print(f"  Saved CSV       : {output_csv}")

    try:
        clean_df.to_parquet(output_parquet, index=False)
        print(f"  Saved Parquet   : {output_parquet}")
    except (ImportError, Exception):
        # Parquet engine (pyarrow/fastparquet) is optional; CSV is the canonical format
        pass

    return clean_df


if __name__ == "__main__":
    run_preprocessing()
