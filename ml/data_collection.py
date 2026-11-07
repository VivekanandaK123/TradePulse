"""
TradePulse - Step 1: Historical Data Collection

Downloads historical 5-minute bars for META from Alpaca using the IEX feed (free tier).
Fetches data in monthly chunks to respect Alpaca's rate limits (200 requests/min).
Saves the raw, unedited dataset to data/raw/META_5min.csv and logs detailed metadata.
"""

import os
import sys
import time
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.enums import DataFeed, Adjustment
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


# ==============================================================================
# Configuration
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SYMBOL = "META"
TIMEFRAME = TimeFrame(5, TimeFrameUnit.Minute)
DATA_FEED = DataFeed.IEX
PRICE_ADJUSTMENT = Adjustment.RAW

# Date configuration: ~1 year of historical data
# Free IEX feed has real-time IEX trades; delayed by 15 min for safety margin
DATA_DELAY_MINUTES = 20
HISTORICAL_DAYS = 365

OUTPUT_RAW_CSV = PROJECT_ROOT / "data" / "raw" / "META_5min.csv"
METADATA_RAW_JSON = PROJECT_ROOT / "data" / "raw" / "metadata.json"
METADATA_ARTIFACT_JSON = PROJECT_ROOT / "artifacts" / "metadata" / "data_collection_metadata.json"

RATE_LIMIT_SLEEP_SECONDS = 0.5
MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 2.0


# ==============================================================================
# Alpaca Client Initialization
# ==============================================================================

def initialize_alpaca_client() -> StockHistoricalDataClient:
    """
    Load Alpaca API credentials from .env and initialize the Historical Data Client.
    """
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")

    if not api_key or not secret_key:
        raise ValueError(
            "Alpaca API credentials missing. Ensure APCA_API_KEY_ID and "
            "APCA_API_SECRET_KEY are set in the .env file."
        )

    return StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)


# ==============================================================================
# Date Range Chunking (Monthly)
# ==============================================================================

def get_next_month_start(dt: datetime) -> datetime:
    """
    Returns the first day of the calendar month immediately following `dt`.
    """
    # Jump 32 days forward, then clamp to the 1st of that month at 00:00:00 UTC
    next_month_approx = dt.replace(day=28) + timedelta(days=4)
    return next_month_approx.replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )


def generate_monthly_chunks(
    start_date: datetime, end_date: datetime
) -> List[Tuple[datetime, datetime]]:
    """
    Splits the date range [start_date, end_date] into monthly chunks.
    Example: 2025-11-01 to 2025-12-01, 2025-12-01 to 2026-01-01, etc.
    """
    chunks = []
    current_start = start_date

    while current_start < end_date:
        next_boundary = get_next_month_start(current_start)
        current_end = min(next_boundary, end_date)
        chunks.append((current_start, current_end))
        current_start = current_end

    return chunks


# ==============================================================================
# Data Extraction
# ==============================================================================

def fetch_chunk_with_retry(
    client: StockHistoricalDataClient,
    request: StockBarsRequest,
    max_retries: int = MAX_RETRIES,
) -> pd.DataFrame:
    """
    Fetches a chunk of bars with exponential backoff on failure.
    """
    attempt = 0
    backoff = 1.0

    while attempt < max_retries:
        try:
            bars_response = client.get_stock_bars(request)
            df = bars_response.df
            if df is not None and not df.empty:
                return df
            return pd.DataFrame()
        except Exception as e:
            attempt += 1
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Failed to fetch data after {max_retries} attempts: {e}"
                ) from e
            print(
                f"    [Warning] Fetch failed ({e}). Retrying in {backoff:.1f}s "
                f"(Attempt {attempt + 1}/{max_retries})..."
            )
            time.sleep(backoff)
            backoff *= RETRY_BACKOFF_FACTOR

    return pd.DataFrame()


def download_historical_bars(
    client: StockHistoricalDataClient,
    symbol: str = SYMBOL,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Downloads historical 5-minute bars in monthly chunks.
    """
    if end_date is None:
        end_date = datetime.now(timezone.utc) - timedelta(minutes=DATA_DELAY_MINUTES)
    if start_date is None:
        start_date = (end_date - timedelta(days=HISTORICAL_DAYS)).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )

    chunks = generate_monthly_chunks(start_date, end_date)
    print(f"\n[TradePulse Data Collection]")
    print(f"  Symbol       : {symbol}")
    print(f"  Timeframe    : 5-minute")
    print(f"  Feed         : {DATA_FEED.value}")
    print(f"  Start Range  : {start_date.isoformat()}")
    print(f"  End Range    : {end_date.isoformat()}")
    print(f"  Total Chunks : {len(chunks)} monthly intervals\n")

    chunk_dfs = []
    total_bars = 0

    for idx, (chunk_start, chunk_end) in enumerate(chunks, 1):
        print(
            f"  -> [{idx:02d}/{len(chunks):02d}] Fetching {chunk_start.strftime('%Y-%m-%d')} "
            f"to {chunk_end.strftime('%Y-%m-%d')}...",
            end=" ",
            flush=True,
        )

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TIMEFRAME,
            start=chunk_start,
            end=chunk_end,
            feed=DATA_FEED,
            adjustment=PRICE_ADJUSTMENT,
        )

        chunk_df = fetch_chunk_with_retry(client, request)

        if not chunk_df.empty:
            count = len(chunk_df)
            total_bars += count
            chunk_dfs.append(chunk_df)
            print(f"Done ({count} bars). Running total: {total_bars}")
        else:
            print("Done (0 bars).")

        # Sleep to strictly observe Alpaca rate limits (200 requests/minute)
        time.sleep(RATE_LIMIT_SLEEP_SECONDS)

    if not chunk_dfs:
        raise RuntimeError("No historical bars were returned across any chunk.")

    combined_df = pd.concat(chunk_dfs)

    # Flatten MultiIndex (symbol, timestamp) to ordinary columns
    combined_df = combined_df.reset_index()

    # Ensure timestamp is UTC datetime
    combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"], utc=True)

    # Remove any overlapping/duplicate bars between chunk boundaries
    initial_len = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=["symbol", "timestamp"])
    duplicates_dropped = initial_len - len(combined_df)
    if duplicates_dropped > 0:
        print(f"  Note: Dropped {duplicates_dropped} boundary duplicates.")

    # Sort strictly chronologically
    combined_df = combined_df.sort_values(by="timestamp").reset_index(drop=True)

    # Align to raw data schema: symbol, timestamp, open, high, low, close, volume, trade_count, vwap
    expected_columns = [
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
    existing_cols = [c for c in expected_columns if c in combined_df.columns]
    combined_df = combined_df[existing_cols]

    return combined_df


# ==============================================================================
# Storage and Metadata
# ==============================================================================

def compute_sha256(file_path: Path) -> str:
    """
    Computes SHA-256 hash of a file for cryptographic reproducibility.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_raw_dataset(
    df: pd.DataFrame,
    symbol: str,
    output_path: Path = OUTPUT_RAW_CSV,
) -> Dict[str, Any]:
    """
    Saves the unchanged raw DataFrame to CSV and generates metadata.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    METADATA_ARTIFACT_JSON.parent.mkdir(parents=True, exist_ok=True)

    # Save raw CSV without indexing
    df.to_csv(output_path, index=False)
    file_size_bytes = output_path.stat().st_size
    file_sha256 = compute_sha256(output_path)

    metadata: Dict[str, Any] = {
        "dataset_name": f"{symbol}_5min_raw",
        "download_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "feed": DATA_FEED.value,
        "timeframe": "5Min",
        "adjustment": PRICE_ADJUSTMENT.value,
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "earliest_timestamp_utc": df["timestamp"].min().isoformat() if not df.empty else None,
        "latest_timestamp_utc": df["timestamp"].max().isoformat() if not df.empty else None,
        "file_name": output_path.name,
        "file_path": str(output_path.relative_to(PROJECT_ROOT)),
        "file_size_bytes": file_size_bytes,
        "sha256": file_sha256,
        "summary_statistics": {
            "open_min": float(df["open"].min()),
            "open_max": float(df["open"].max()),
            "close_min": float(df["close"].min()),
            "close_max": float(df["close"].max()),
            "volume_sum": float(df["volume"].sum()),
            "total_trades": float(df["trade_count"].sum()) if "trade_count" in df.columns else None,
        },
    }

    # Save metadata in data/raw/ and artifacts/metadata/
    for meta_path in [METADATA_RAW_JSON, METADATA_ARTIFACT_JSON]:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    print(f"\n[Raw Data Stored]")
    print(f"  File Location   : {output_path}")
    print(f"  Total Rows      : {len(df):,}")
    print(f"  Date Span       : {metadata['earliest_timestamp_utc']} -> {metadata['latest_timestamp_utc']}")
    print(f"  File Size       : {file_size_bytes / (1024 * 1024):.2f} MB")
    print(f"  SHA-256 Hash    : {file_sha256}")
    print(f"  Metadata Saved  : {METADATA_RAW_JSON} & {METADATA_ARTIFACT_JSON}")

    return metadata


# ==============================================================================
# Main Runner
# ==============================================================================

def collect_historical_data() -> pd.DataFrame:
    """
    Orchestrates the entire Step 1 collection workflow.
    """
    client = initialize_alpaca_client()
    df = download_historical_bars(client, symbol=SYMBOL)
    save_raw_dataset(df, symbol=SYMBOL)
    return df


if __name__ == "__main__":
    collect_historical_data()

