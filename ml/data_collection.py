import os
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd
import pandas_market_calendars as mcal

from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.enums import DataFeed, Adjustment
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SYMBOL = "META"
TIMEFRAME = TimeFrame(5, TimeFrameUnit.Minute)

HISTORICAL_DAYS = 365
DATA_DELAY_MINUTES = 20

DATA_FEED = DataFeed.SIP
PRICE_ADJUSTMENT = Adjustment.RAW

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "META_5min.csv"
)

EXCHANGE_TIMEZONE = "America/New_York"
EXCHANGE_CALENDAR = "NASDAQ"


# ============================================================
# 2. INITIALIZE ALPACA CLIENT
# ============================================================

def initialize_client():
    """
    Load Alpaca credentials and initialize the
    historical market data client.
    """

    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")

    if not api_key or not secret_key:
        raise ValueError(
            "Alpaca API credentials not found in .env"
        )

    return StockHistoricalDataClient(
        api_key,
        secret_key
    )


# ============================================================
# 3. GENERATE MONTHLY DATE RANGES
# ============================================================

def next_month(dt):
    """
    Return the beginning of the next calendar month.
    """

    first_of_next_month = (
        dt.replace(day=28) + timedelta(days=4)
    )

    return first_of_next_month.replace(
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0
    )


def generate_date_ranges(start_date, end_date):
    """
    Split a historical date range into monthly chunks.
    """

    current_start = start_date

    while current_start < end_date:

        current_end = min(
            next_month(current_start),
            end_date
        )

        yield current_start, current_end

        current_start = current_end


# ============================================================
# 4. DOWNLOAD HISTORICAL DATA
# ============================================================

def download_historical_data(
    client,
    start_date,
    end_date
):
    """
    Download historical META candles from Alpaca
    in monthly chunks.
    """

    all_data = []

    date_ranges = list(
        generate_date_ranges(
            start_date,
            end_date
        )
    )

    for current_start, current_end in date_ranges:

        print(
            f"Downloading: "
            f"{current_start.date()} to "
            f"{current_end.date()}"
        )

        request = StockBarsRequest(
            symbol_or_symbols=SYMBOL,
            timeframe=TIMEFRAME,
            start=current_start,
            end=current_end,
            feed=DATA_FEED,
            adjustment=PRICE_ADJUSTMENT,
            limit=10000
        )

        bars = client.get_stock_bars(request)

        chunk = bars.df

        if not chunk.empty:
            all_data.append(chunk)

    if not all_data:
        raise RuntimeError(
            "No historical data was returned by Alpaca."
        )

    df = pd.concat(all_data)

    return df


# ============================================================
# 5. CLEAN AND STANDARDIZE DATA
# ============================================================

def clean_data(df):
    """
    Convert the Alpaca response into a clean DataFrame.
    """

    # Convert MultiIndex into ordinary columns
    df = df.reset_index()

    # Ensure timestamps are timezone-aware UTC
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    # Remove duplicate observations
    df = df.drop_duplicates(
        subset=["symbol", "timestamp"]
    )

    # Sort chronologically
    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return df


# ============================================================
# 6. FILTER ACTUAL NASDAQ TRADING SESSIONS
# ============================================================

def filter_trading_sessions(df):
    """
    Keep only candles belonging to actual NASDAQ
    regular trading sessions.

    Handles:
    - Exchange holidays
    - Regular trading sessions
    - Early market closures
    """

    if df.empty:
        return df.copy()

    nasdaq = mcal.get_calendar(
        EXCHANGE_CALENDAR
    )

    # Generate official NASDAQ trading schedule
    schedule = nasdaq.schedule(
        start_date=df["timestamp"].min().date(),
        end_date=df["timestamp"].max().date()
    )

    if schedule.empty:
        return df.iloc[0:0].copy()

    schedule = schedule.reset_index()

    # Convert exchange session date into a join key
    schedule["trading_date"] = pd.to_datetime(
        schedule["market_open"],
        utc=True
    ).dt.tz_convert(
        EXCHANGE_TIMEZONE
    ).dt.date

    # Identify the New York trading date of each candle
    df = df.copy()

    df["trading_date"] = (
        df["timestamp"]
        .dt.tz_convert(EXCHANGE_TIMEZONE)
        .dt.date
    )

    # Attach actual market opening and closing times
    df = df.merge(
        schedule[
            [
                "trading_date",
                "market_open",
                "market_close"
            ]
        ],
        on="trading_date",
        how="inner"
    )

    # Keep candles inside the regular trading session
    df = df[
        (df["timestamp"] >= df["market_open"]) &
        (df["timestamp"] < df["market_close"])
    ].copy()

    # Remove temporary columns
    df = df.drop(
        columns=[
            "trading_date",
            "market_open",
            "market_close"
        ]
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return df


# ============================================================
# 7. VALIDATE DATA QUALITY
# ============================================================

def validate_data(df):
    """
    Perform basic structural and OHLCV validation.
    """

    if df.empty:
        raise ValueError(
            "Dataset is empty after processing."
        )

    required_columns = [
        "symbol",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "vwap"
    ]

    missing_columns = set(
        required_columns
    ) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # Missing values
    if df[required_columns].isna().any().any():
        raise ValueError(
            "Dataset contains missing values."
        )

    # Duplicate timestamps
    if df.duplicated(
        subset=["symbol", "timestamp"]
    ).any():
        raise ValueError(
            "Duplicate timestamps detected."
        )

    # Chronological ordering
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError(
            "Timestamps are not sorted chronologically."
        )

    # OHLC validation
    if not (
        (df["high"] >= df["low"]).all()
        and (df["high"] >= df["open"]).all()
        and (df["high"] >= df["close"]).all()
        and (df["low"] <= df["open"]).all()
        and (df["low"] <= df["close"]).all()
    ):
        raise ValueError(
            "Invalid OHLC observations detected."
        )

    # Volume validation
    if (df["volume"] < 0).any():
        raise ValueError(
            "Negative volume observations detected."
        )

    print("Data validation passed!")


# ============================================================
# 8. SAVE DATASET
# ============================================================

def save_data(df, output_path=OUTPUT_PATH):
    """
    Save the processed historical dataset as CSV.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(f"Dataset saved to: {output_path}")


# ============================================================
# 9. MAIN COLLECTION PIPELINE
# ============================================================

def collect_historical_data():
    """
    Execute the complete historical data collection
    workflow.
    """

    print("\nInitializing Alpaca client...")

    client = initialize_client()

    # Define historical period
    end_date = (
        datetime.now(timezone.utc)
        - timedelta(minutes=DATA_DELAY_MINUTES)
    )

    start_date = (
        end_date
        - timedelta(days=HISTORICAL_DAYS)
    )

    print("\nHistorical data collection started")
    print(f"Symbol: {SYMBOL}")
    print(f"Timeframe: 5 minutes")
    print(f"Feed: {DATA_FEED}")
    print(f"Start: {start_date}")
    print(f"End: {end_date}")

    # Download
    df = download_historical_data(
        client,
        start_date,
        end_date
    )

    # Clean
    df = clean_data(df)

    # Filter trading sessions
    df = filter_trading_sessions(df)

    # Validate
    validate_data(df)

    # Save
    save_data(df)

    # Summary
    print("\n========== DOWNLOAD COMPLETED ==========")

    print("Total candles:", len(df))

    print(
        "Date range:",
        df["timestamp"].min(),
        "to",
        df["timestamp"].max()
    )

    print("\nDataset preview:")
    print(df.head())

    print("\nDataset information:")
    df.info()

    return df


# ============================================================
# 10. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    collect_historical_data()
