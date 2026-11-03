import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
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

def validate_bars(
    df: pd.DataFrame,
    expected_symbol: str = "META",
    timeframe: str = "5min",
    require_sorted: bool = True,
) -> None:
    """
    Validate market OHLCV data.
    Raises ValueError if any validation rule fails.
    Does not modify the original DataFrame.
    """
    
    # 1. Basic checks
    if df.empty:
        raise ValueError("Dataset is empty.")
    
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")
    
    # 2. Validate symbol
    if not df["symbol"].eq(expected_symbol).all():
        raise ValueError(
            f"Unexpected symbol. Expected {expected_symbol}."
        )   
        
    # 3. Validate timestamps
    timestamps = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    )
    
    if timestamps.isna().any():
        raise ValueError("Invalid or missing timestamps found.")
    
    if timestamps.duplicated().any():
        raise ValueError("Duplicate timestamps found.")
    
    if require_sorted and not timestamps.is_monotonic_increasing:
        raise ValueError("Timestamps are not chronologically sorted.")
    
    # 4. Validate numeric columns
    numeric_columns = [
        "open", "high", "low", "close",
        "volume", "trade_count", "vwap"
    ]
    
    numeric = df[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )
    
    if numeric.isna().any().any():
        raise ValueError("Invalid or missing numeric values found.")
    
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Infinite numeric values found.")
    
    
    # 5. Validate prices
    if (numeric[["open", "high", "low", "close", "vwap"]] <= 0).any().any():
        raise ValueError("Prices and VWAP must be positive.")
    
    if (numeric["high"] < numeric["low"]).any():
        raise ValueError("High cannot be lower than low.")
    
    if (numeric["high"] < numeric[["open", "close"]].max(axis=1)).any():
        raise ValueError("High must be >= open and close.")
    
    if (numeric["low"] > numeric[["open", "close"]].min(axis=1)).any():
        raise ValueError("Low must be <= open and close.")
    
    # 6. Validate volume and trade count
    if (numeric["volume"] < 0).any():
        raise ValueError("Negative volume found.")
    
    if (numeric["trade_count"] < 0).any():
        raise ValueError("Negative trade count found.")
    
    # 7. Validate candle alignment
    interval = pd.Timedelta(timeframe)
    
    if not ((timestamps.dt.minute * 60 + timestamps.dt.second) % interval.total_seconds() == 0).all():
        raise ValueError("Misaligned candle timestamps.")
    
    print("All data validation checks passed.")