"""
TradePulse - Step 6: Feature Engineering Pipeline

Implements pure, causal feature engineering used identically in batch training and live inference.
Key Principles:
1. 100% Causal: Uses strictly current and past bars (no shift(-k) or future information).
2. IEX Volume Normalization: Normalizes volume, trade count, and trade size against rolling averages.
3. Maximum Lookback: 35 bars (~175 minutes), defining the exact live warm-up requirement.
4. Leakage Prevention: Scalers are fit strictly on training split rows and serialized to artifacts/models/.
"""

import sys
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing import EXCHANGE_TIMEZONE

MAX_LOOKBACK_BARS = 35
SCALER_ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "models" / "feature_scaler.joblib"
METADATA_PATH = PROJECT_ROOT / "artifacts" / "metadata" / "feature_metadata.json"


# ==============================================================================
# Technical Indicator Calculations (Pure Causal Functions)
# ==============================================================================

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Computes Wilder's Relative Strength Index (RSI) causally using Exponential Moving Averages.
    Output range: [0, 100].
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing corresponds to ewm with alpha = 1 / period
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Computes Normalized MACD line, signal line, and histogram.
    Normalized by price to ensure stationarity across different price regimes.
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()

    macd_line = (ema_fast - ema_slow) / (series + 1e-6)
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    return macd_line, macd_signal, macd_hist


def compute_bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> Tuple[pd.Series, pd.Series]:
    """
    Computes Bollinger Bands %B and Bandwidth.
    %B: normalized position within bands (0 at lower, 0.5 at middle, 1 at upper).
    Bandwidth: (upper - lower) / middle (measures volatility squeeze / expansion).
    """
    middle = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std()

    upper = middle + num_std * std
    lower = middle - num_std * std

    percent_b = (series - lower) / (upper - lower + 1e-6)
    bandwidth = (upper - lower) / (middle + 1e-6)

    return percent_b, bandwidth


# ==============================================================================
# Comprehensive Feature Engineering
# ==============================================================================

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms clean 5-minute bars into a feature matrix.
    Shared contract between batch training and live inference.

    Features generated:
    1. Trailing Returns: 1, 3, 6, 12 bars
    2. Volatility & Range: 6-bar std, 12-bar std, high-low range, body size, close position
    3. VWAP Metrics: distance to VWAP, 3-bar VWAP slope
    4. Relative Volume: volume / rolling mean (12, 24), trade count ratio, trade size ratio
    5. Synthetic Fill Density: count of filled bars in last 12 bars
    6. Classic Indicators: RSI(14), Normalized MACD(12,26,9), Bollinger Bands %B & Bandwidth(20)
    7. Moving Average Distances: (close - EMA_9) / close, (close - EMA_21) / close
    8. Intraday Session Context: cyclic time of day (sin/cos), day of week, return since open, overnight gap
    """
    out = df.copy()

    # Normalize timestamp
    ts_utc = pd.to_datetime(out["timestamp"], utc=True)
    ts_ny = ts_utc.dt.tz_convert(EXCHANGE_TIMEZONE)

    if "session_date" not in out.columns:
        out["session_date"] = ts_ny.dt.date

    close = out["close"]
    open_p = out["open"]
    high = out["high"]
    low = out["low"]
    vol = out["volume"]
    trades = out["trade_count"]
    vwap = out["vwap"]
    is_filled = out["is_filled"] if "is_filled" in out.columns else pd.Series(0, index=out.index)

    # --------------------------------------------------------------------------
    # 1. Trailing Multi-Horizon Returns
    # --------------------------------------------------------------------------
    out["feat_ret_1b"] = close.pct_change(1)
    out["feat_ret_3b"] = close.pct_change(3)
    out["feat_ret_6b"] = close.pct_change(6)
    out["feat_ret_12b"] = close.pct_change(12)

    # --------------------------------------------------------------------------
    # 2. Volatility & Candle Geometry
    # --------------------------------------------------------------------------
    out["feat_vol_6b"] = out["feat_ret_1b"].rolling(6, min_periods=6).std()
    out["feat_vol_12b"] = out["feat_ret_1b"].rolling(12, min_periods=12).std()
    out["feat_bar_range"] = (high - low) / (close + 1e-6)
    out["feat_body_size"] = (close - open_p).abs() / (close + 1e-6)
    out["feat_close_pos"] = (close - low) / (high - low + 1e-6)

    # --------------------------------------------------------------------------
    # 3. VWAP Relationships
    # --------------------------------------------------------------------------
    out["feat_vwap_dist"] = (close - vwap) / (vwap + 1e-6)
    out["feat_vwap_slope_3b"] = (vwap - vwap.shift(3)) / (vwap.shift(3) + 1e-6)

    # --------------------------------------------------------------------------
    # 4. Volume & Order Flow (Relative to Avoid IEX Absolute Volume Bias)
    # --------------------------------------------------------------------------
    mean_vol_12 = vol.rolling(12, min_periods=6).mean()
    mean_vol_24 = vol.rolling(24, min_periods=12).mean()
    out["feat_rvol_12b"] = vol / (mean_vol_12 + 1e-4)
    out["feat_rvol_24b"] = vol / (mean_vol_24 + 1e-4)

    mean_trades_12 = trades.rolling(12, min_periods=6).mean()
    out["feat_rtrades_12b"] = trades / (mean_trades_12 + 1e-4)

    trade_size = vol / (trades + 1e-4)
    mean_trade_size_12 = trade_size.rolling(12, min_periods=6).mean()
    out["feat_avg_trade_size_ratio"] = trade_size / (mean_trade_size_12 + 1e-4)

    # Density of filled bars in recent lookback
    out["feat_filled_density_12b"] = is_filled.rolling(12, min_periods=1).sum()

    # --------------------------------------------------------------------------
    # 5. Technical Oscillators & Moving Averages
    # --------------------------------------------------------------------------
    out["feat_rsi_14"] = compute_rsi(close, period=14) / 100.0  # Normalized to [0, 1]

    macd_line, macd_sig, macd_hist = compute_macd(close, fast=12, slow=26, signal=9)
    out["feat_macd_line"] = macd_line
    out["feat_macd_signal"] = macd_sig
    out["feat_macd_hist"] = macd_hist

    bb_pct_b, bb_width = compute_bollinger_bands(close, window=20, num_std=2.0)
    out["feat_bb_pct_b"] = bb_pct_b
    out["feat_bb_width"] = bb_width

    ema_9 = close.ewm(span=9, adjust=False).mean()
    ema_21 = close.ewm(span=21, adjust=False).mean()
    out["feat_ema_dist_9"] = (close - ema_9) / (close + 1e-6)
    out["feat_ema_dist_21"] = (close - ema_21) / (close + 1e-6)

    # --------------------------------------------------------------------------
    # 6. Session Context & Cyclic Time of Day
    # --------------------------------------------------------------------------
    minute_of_day = ts_ny.dt.hour * 60 + ts_ny.dt.minute
    # Standard session: 9:30 AM (570 min) to 16:00 PM (960 min) = 390 min duration
    norm_tod = (minute_of_day - 570) / 390.0
    out["feat_time_sin"] = np.sin(2.0 * np.pi * norm_tod)
    out["feat_time_cos"] = np.cos(2.0 * np.pi * norm_tod)
    out["feat_day_of_week"] = ts_ny.dt.day_of_week / 4.0  # Normalized Mon=0.0 to Fri=1.0

    # Cumulative return since session open
    session_open = out.groupby("session_date")["open"].transform("first")
    out["feat_ret_since_open"] = (close - session_open) / (session_open + 1e-6)

    # Causal overnight gap (session open vs prior session close)
    session_info = out.groupby("session_date").agg(
        sess_open=("open", "first"),
        sess_close=("close", "last"),
    ).reset_index()
    session_info["prev_close"] = session_info["sess_close"].shift(1)
    session_info["gap"] = (session_info["sess_open"] - session_info["prev_close"]) / (session_info["prev_close"] + 1e-6)
    session_info["gap"] = session_info["gap"].fillna(0.0)

    gap_map = dict(zip(session_info["session_date"], session_info["gap"]))
    out["feat_overnight_gap"] = out["session_date"].map(gap_map).fillna(0.0)

    return out


def get_feature_column_names(df: pd.DataFrame) -> List[str]:
    """
    Returns list of all computed feature column names prefixed with 'feat_'.
    """
    return [c for c in df.columns if c.startswith("feat_")]


# ==============================================================================
# Scaler Fitting and Persistence (Train Partition Only)
# ==============================================================================

def fit_feature_scaler(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    scaler_path: Path = SCALER_ARTIFACT_PATH,
) -> RobustScaler:
    """
    Fits a RobustScaler strictly on the training partition features and saves to disk.
    RobustScaler uses median and interquartile range (IQR), providing resilience to
    leptokurtic financial return spikes.
    """
    scaler_path.parent.mkdir(parents=True, exist_ok=True)

    X_train = train_df[feature_cols].copy()
    scaler = RobustScaler()
    scaler.fit(X_train)

    joblib.dump(scaler, scaler_path)
    print(f"  Fitted RobustScaler on {len(X_train):,} training rows and saved to: {scaler_path}")
    return scaler


def load_feature_scaler(scaler_path: Path = SCALER_ARTIFACT_PATH) -> RobustScaler:
    """
    Loads fitted RobustScaler object from disk.
    """
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler artifact not found at {scaler_path}. Run training fit first.")
    return joblib.load(scaler_path)


def transform_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    scaler: RobustScaler,
) -> pd.DataFrame:
    """
    Applies fitted scaler to feature columns, returning scaled values with original index.
    """
    scaled_array = scaler.transform(df[feature_cols])
    scaled_df = pd.DataFrame(
        scaled_array,
        columns=[f"{col}_scaled" for col in feature_cols],
        index=df.index,
    )
    return scaled_df


# ==============================================================================
# Full Feature Engineering Pipeline Runner
# ==============================================================================

def process_features_and_split(
    warmup_bars: int = MAX_LOOKBACK_BARS,
    clean_csv_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Executes the Step 6 feature workflow:
    1. Computes features across the full continuous series.
    2. Drops the first `warmup_bars` of the entire series.
    3. Partitions into train, val, and test based on frozen calendar dates.
    4. Fits RobustScaler strictly on train rows.
    5. Saves feature metadata and feature matrices.
    """
    from ml.targets import PROCESSED_TARGETS_CSV_PATH
    from ml.split import SPLIT_CONFIG_PATH

    source_path = clean_csv_path or PROCESSED_TARGETS_CSV_PATH
    print(f"\n[TradePulse Step 6: Feature Engineering]")
    print(f"  Loading continuous dataset: {source_path}")
    raw_clean = pd.read_csv(source_path)

    # 1. Compute features causally
    df_with_features = compute_features(raw_clean)
    feature_cols = get_feature_column_names(df_with_features)
    print(f"  Generated {len(feature_cols)} technical and contextual features.")

    # 2. Drop initial warm-up bars of the entire series
    df_valid = df_with_features.iloc[warmup_bars:].copy().reset_index(drop=True)
    print(f"  Dropped first {warmup_bars} warm-up bars. Remaining rows: {len(df_valid):,}")

    # 3. Partition chronologically
    with open(SPLIT_CONFIG_PATH, "r", encoding="utf-8") as f:
        split_cfg = json.load(f)["splits"]

    train_mask = (df_valid["session_date"].astype(str) >= split_cfg["train"]["start_date"]) & (
        df_valid["session_date"].astype(str) <= split_cfg["train"]["end_date"]
    )
    val_mask = (df_valid["session_date"].astype(str) >= split_cfg["validation"]["start_date"]) & (
        df_valid["session_date"].astype(str) <= split_cfg["validation"]["end_date"]
    )
    test_mask = (df_valid["session_date"].astype(str) >= split_cfg["test"]["start_date"]) & (
        df_valid["session_date"].astype(str) <= split_cfg["test"]["end_date"]
    )

    train_features = df_valid[train_mask].copy().reset_index(drop=True)
    val_features = df_valid[val_mask].copy().reset_index(drop=True)
    test_features = df_valid[test_mask].copy().reset_index(drop=True)

    print(f"  Partitioned: Train={len(train_features):,}, Val={len(val_features):,}, Test={len(test_features):,}")

    # 4. Fit scaler strictly on train split
    scaler = fit_feature_scaler(train_features, feature_cols)

    # 5. Save feature metadata
    metadata = {
        "feature_count": len(feature_cols),
        "feature_names": feature_cols,
        "max_lookback_bars": warmup_bars,
        "live_warmup_requirement_bars": warmup_bars,
        "live_warmup_minutes": warmup_bars * 5,
        "scaler_type": "RobustScaler (fit on train rows only)",
        "scaler_path": str(SCALER_ARTIFACT_PATH.relative_to(PROJECT_ROOT)),
        "train_rows_fit": len(train_features),
        "feature_categories": {
            "returns": ["feat_ret_1b", "feat_ret_3b", "feat_ret_6b", "feat_ret_12b"],
            "volatility_and_candle": ["feat_vol_6b", "feat_vol_12b", "feat_bar_range", "feat_body_size", "feat_close_pos"],
            "vwap": ["feat_vwap_dist", "feat_vwap_slope_3b"],
            "volume_and_flow": ["feat_rvol_12b", "feat_rvol_24b", "feat_rtrades_12b", "feat_avg_trade_size_ratio", "feat_filled_density_12b"],
            "oscillators_and_averages": ["feat_rsi_14", "feat_macd_line", "feat_macd_signal", "feat_macd_hist", "feat_bb_pct_b", "feat_bb_width", "feat_ema_dist_9", "feat_ema_dist_21"],
            "session_context": ["feat_time_sin", "feat_time_cos", "feat_day_of_week", "feat_ret_since_open", "feat_overnight_gap"],
        },
    }

    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Feature metadata saved to: {METADATA_PATH}")

    # 6. Save feature datasets to data/processed/
    processed_dir = PROJECT_ROOT / "data" / "processed"
    train_features.to_csv(processed_dir / "train_features.csv", index=False)
    val_features.to_csv(processed_dir / "val_features.csv", index=False)
    test_features.to_csv(processed_dir / "test_features.csv", index=False)
    print(f"  Saved train_features.csv, val_features.csv, and test_features.csv to {processed_dir}")

    return metadata


if __name__ == "__main__":
    process_features_and_split()
