import os
from pathlib import Path
from datetime import datetime, timedelta, timezone, time

import pandas as pd
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.enums import DataFeed, Adjustment
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

import pandas_market_calendars as mcal


# 1. Load API credentials
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("APCA_API_KEY_ID")
secret_key = os.getenv("APCA_API_SECRET_KEY")

if not api_key or not secret_key:
    raise ValueError("Alpaca API credentials not found in .env")


# 2. Initialize Alpaca client
client = StockHistoricalDataClient(api_key, secret_key)

SYMBOL = "META"
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "META_5min.csv"


# 3. Define the historical period
end_date = datetime.now(timezone.utc) - timedelta(minutes=20)
start_date = end_date - timedelta(days=365)


# 4. Generate monthly date ranges
def next_month(dt):
    first_of_next_month = dt.replace(day=28) + timedelta(days=4)
    return first_of_next_month.replace(day=1, hour=0, minute=0,
                                       second=0, microsecond=0)


# 5. Download data in monthly chunks
all_data = []
current_start = start_date

while current_start < end_date:
    current_end = min(next_month(current_start), end_date)

    print(f"Downloading: {current_start.date()} to {current_end.date()}")

    request = StockBarsRequest(
        symbol_or_symbols=SYMBOL,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=current_start,
        end=current_end,
        feed=DataFeed.SIP,
        adjustment=Adjustment.RAW,
        limit=10000
    )

    bars = client.get_stock_bars(request)
    chunk = bars.df

    if not chunk.empty:
        all_data.append(chunk)

    current_start = current_end


# 6. Combine all downloaded chunks
if not all_data:
    raise RuntimeError("No historical data was returned by Alpaca.")

df = pd.concat(all_data)

# Convert the MultiIndex into ordinary columns
df = df.reset_index()

# Remove duplicate timestamps, if any
df = df.drop_duplicates(subset=["symbol", "timestamp"])

# Sort chronologically
df = df.sort_values("timestamp").reset_index(drop=True)


# 7. Filter regular trading hours (New York time)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

# Get NASDAQ trading schedule
nasdaq = mcal.get_calendar("NASDAQ")

schedule = nasdaq.schedule(
    start_date=df["timestamp"].min().date(),
    end_date=df["timestamp"].max().date()
)

# Convert schedule into a DataFrame
schedule = schedule.reset_index()

schedule['trading_date'] = pd.to_datetime(
    schedule["market_open"]
).dt.date

# Identify each candle's New York trading date
df["trading_date"] = (
    df["timestamp"]
    .dt.tz_convert("America/New_York")
    .dt.date
)

# Attach the official opening and closing times
df = df.merge(
    schedule[["trading_date", "market_open", "market_close"]],
    on="trading_date",
    how="inner"
)

# Keep candles within the actual trading session
df = df[
    (df["timestamp"] >= df["market_open"]) &
    (df["timestamp"] < df["market_close"])
].copy()

df = df.drop(
    columns=["trading_date", "market_open", "market_close"]
)

df = df.sort_values("timestamp").reset_index(drop=True)

# ny_time = df["timestamp"].dt.tz_convert("America/New_York")
# session_time = ny_time.dt.time

# df = df[
#     (session_time >= time(9, 30)) &
#     (session_time < time(16, 0))
# ].copy()


# 8. Save the dataset
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_PATH, index=False)

print("\nDownload completed!")
print("Total candles:", len(df))
print("Date range:", df["timestamp"].min(), "to", df["timestamp"].max())
print("Saved to:", OUTPUT_PATH)
print("\nDataset preview:")
print(df.head())
print("\nDataset information:")
print(df.info())
