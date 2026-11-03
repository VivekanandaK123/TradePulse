import pandas as pd

from ml.validation import validate_bars

def preprocess_bars(
    df: pd.DataFrame,
    expected_symbol: str = "META",
    timeframe: str = "5min"
) -> pd.DataFrame:
    """
    Preprocess raw market bars.
    
    - Validate input data.
    - Converts tiemstamps to UTC datetime.
    - Sorts canldes chronologically.
    - Resets the index.
    - Preserves all original columns.
    
    Returns a new DataFrame wihtout modifying the input.
    """
    
    # Validate raw input (sorting is handled below)
    validate_bars(
        df, 
        expected_symbol=expected_symbol,
        timeframe=timeframe,
        require_sorted=False
    )
    
    # Create a copy to preserve raw data
    processed = df.copy()
    
    numeric_columns = [
        "open", "high", "low", "close",
        "volume", "trade_count", "vwap"
    ]

    processed[numeric_columns] = processed[numeric_columns].apply(
        pd.to_numeric
    )
    
    # Convert timestamps to UTC
    processed["timestamp"] = pd.to_datetime(
        processed["timestamp"],
        utc=True
    )
    
    # Sort chronologically
    processed = processed.sort_values(by="timestamp")
    
    # Reset index
    processed = processed.reset_index(drop=True)
    
    # Validate processed output
    validate_bars(
        processed,
        expected_symbol=expected_symbol,
        timeframe=timeframe,
        require_sorted=True
    )
    
    return processed