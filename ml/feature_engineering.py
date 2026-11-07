import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

def engineer_features(
    df: pd.DataFrame,
    expected_symbol: str = "META"
) -> pd.DataFrame:
    """
    Generate initial features from preprocessed 5-minute bars.
    
    Features use current and historical closed candles only.
    Intraday calculations reset at the start of each trading day.
    
    Returns a new DataFrame wihout modifying the input.
    """
    
    if df.empty:
        raise ValueError("Input data is empty.")
    
    required_columns = {
        "symbol", "timestamp", "open", "high",
        "low", "close", "volume", "trade_count", "vwap"
    }
    
    missing = required_columns - set(df.columns)
    
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    if not df["symbol"].eq(expected_symbol).all():
        raise ValueError(f"Expected symbol {expected_symbol}.")
    
    result = df.copy()
    
    result["timestamp"] = pd.to_datetime(
        result["timestamp"], utc=True
    )
    
    result = result.sort_values("timestamp").reset_index(drop=True)
    
    # Derive trading date and time using New York timezone
    ny_time = result["timestamp"].dt.tz_convert(
        "America/New_York"
    )
    
    calendar = mcal.get_calendar("NASDAQ")

    schedule = calendar.schedule(
        start_date=ny_time.min().date(),
        end_date=ny_time.max().date()
    )
    
    result["trading_date"] = ny_time.dt.date
    result["minute_of_day"] = (
        ny_time.dt.hour * 60 + ny_time.dt.minute
    )
    
    # Convert market schedule timestamps to New York timezone
    schedule = schedule.copy()

    schedule["market_open"] = (
        schedule["market_open"].dt.tz_convert("America/New_York")
    )

    # Map each candle to its trading day's market opening time
    opening_times = schedule["market_open"]

    result["market_open"] = (
        result["trading_date"].map(opening_times)
    )

    result["minutes_since_open"] = (
        (ny_time - result["market_open"])
        .dt.total_seconds() / 60
    )
    
    grouped = result.groupby("trading_date", sort=False)
    
    
    # ------------------------------------------------
    # 1. Returns and momentum
    # ------------------------------------------------
    
    for period in [1, 3, 6, 12]:
        result[f"return_{period}"] = (
            grouped["close"]
            .pct_change(period, fill_method=None) * 100
        )
        
    # ------------------------------------------------
    # 2. Rolling volatility
    # ------------------------------------------------
    
    result["rolling_volatility_12"] = (
        result.groupby("trading_date")["return_1"]
        .transform(
            lambda x: x.rolling(
                window=12,
                min_periods=12
            ).std()
        )
    )
    
    # --------------------------------------------------
    # 3. Volume features
    # --------------------------------------------------

    result["log_volume"] = np.log1p(result["volume"])

    result["volume_change"] = (
        grouped["volume"]
        .pct_change(fill_method=None) * 100
    )

    result["volume_ma_12"] = (
        grouped["volume"]
        .transform(
            lambda x: x.rolling(
                window=12,
                min_periods=12
            ).mean()
        )
    )

    result["relative_volume_12"] = (
        result["volume"] / result["volume_ma_12"]
    )
    
    # Historical average volume for the same time of day.
    # Shift ensures the current candle is excluded.
    result["historical_time_volume"] = (
        result.groupby("minute_of_day", sort=False)["volume"]
        .transform(
            lambda x: x.shift(1).expanding(
                min_periods=5
            ).mean()
        )
    )

    result["relative_volume_time"] = (
        result["volume"] / result["historical_time_volume"]
    )
    
    # --------------------------------------------------
    # 4. VWAP features
    # --------------------------------------------------

    result["vwap_deviation"] = (
        (result["close"] - result["vwap"])
        / result["vwap"] * 100
    )

    return result
    
    
    