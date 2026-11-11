"""
TradePulse - Step 14: Production Packaging & Inference Pipeline

Encapsulates the complete live inference workflow:
1. Validates closed candle buffers (checking causal warm-up requirement: >= 35 bars).
2. Computes the exact 29 causal features using the shared ml.features pipeline.
3. Generates calibrated class probabilities using the production 50/50 Ensemble (RF + LR).
4. Applies the production conviction gating rule (tau = 0.2208).
5. Outputs a standardized, unmodifiable prediction payload with direction, confidence, and timestamp.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.features import compute_features, get_feature_column_names, MAX_LOOKBACK_BARS
from ml.calibration import create_ensemble_probabilities
from ml.label_expansion_experiments import compute_directional_conviction

METADATA_DIR = PROJECT_ROOT / "artifacts" / "metadata"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
MODEL_VERSION = "TradePulse-v1.0-Ensemble-fixed15bp"


class TradePulsePredictor:
    """
    Production inference engine for live closed-candle signal generation.
    Enforces shared pipeline logic: identical feature transformations and calibrated probabilities.
    """

    def __init__(
        self,
        rf_path: Optional[Path] = None,
        lr_path: Optional[Path] = None,
        calibrator_path: Optional[Path] = None,
        config_path: Optional[Path] = None,
    ):
        self.rf_path = rf_path or (MODELS_DIR / "random_forest_model_label_expansion.joblib")
        self.lr_path = lr_path or (MODELS_DIR / "logistic_regression_model_label_expansion.joblib")
        self.calibrator_path = calibrator_path or (MODELS_DIR / "probability_calibrator_label_expansion.joblib")
        self.config_path = config_path or (METADATA_DIR / "production_signal_rule_label_expansion.json")

        # Load models
        self.rf_model = joblib.load(self.rf_path)
        self.lr_model = joblib.load(self.lr_path)
        self.calibrator = joblib.load(self.calibrator_path)

        # Load production rule metadata
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.threshold = float(self.config["signal_rule"]["calibrated_conviction_threshold"])
        self.target_label = self.config["chosen_target_label"]
        self.version = MODEL_VERSION

    def validate_buffer(self, buffer_df: pd.DataFrame) -> None:
        """
        Validates that the candle buffer is sufficient for causal feature computation.
        """
        if len(buffer_df) < MAX_LOOKBACK_BARS:
            raise ValueError(
                f"Insufficient candle history for feature warm-up: buffer has {len(buffer_df)} bars, "
                f"required minimum is {MAX_LOOKBACK_BARS} bars (~175 minutes)."
            )

        required_cols = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]
        missing = [c for c in required_cols if c not in buffer_df.columns]
        if missing:
            raise ValueError(f"Missing required candle columns in input buffer: {missing}")

    def compute_candle_features(self, buffer_df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs the shared feature engineering pipeline on the candle buffer.
        """
        df = buffer_df.copy()
        if "is_filled" not in df.columns:
            df["is_filled"] = 0

        # Ensure timestamp_ny and session_date exist
        if "timestamp_ny" not in df.columns:
            if "timestamp" in df.columns:
                ts_utc = pd.to_datetime(df["timestamp"], utc=True)
                df["timestamp_ny"] = ts_utc.dt.tz_convert("America/New_York")
            else:
                raise ValueError("Buffer must contain 'timestamp' or 'timestamp_ny'.")

        if "session_date" not in df.columns:
            df["session_date"] = pd.to_datetime(df["timestamp_ny"]).dt.date.astype(str)

        # Compute features using the EXACT shared function
        feat_df = compute_features(df)
        return feat_df

    def predict_candle(self, buffer_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Predicts directional signal and calibrated confidence for the LATEST closed candle.

        Parameters:
        -----------
        buffer_df : pd.DataFrame
            Chronologically sorted buffer of historical closed 5-minute candles
            containing at least 35 bars up to and including the target candle.

        Returns:
        --------
        dict: Standardized prediction payload for logging and chart presentation.
        """
        self.validate_buffer(buffer_df)

        # 1. Compute features across buffer
        feat_df = self.compute_candle_features(buffer_df)
        feature_cols = get_feature_column_names(feat_df)

        # 2. Extract latest closed bar features
        latest_row = feat_df.iloc[-1]
        latest_ts = str(latest_row["timestamp_ny"])
        latest_close = float(latest_row["close"])
        X_latest = latest_row[feature_cols].values.astype(float).reshape(1, -1)

        # Check for NaNs
        if np.isnan(X_latest).any():
            raise ValueError("Feature vector for latest candle contains NaNs. Ensure sufficient warm-up lookback.")

        # 3. Model Predictions & Calibration
        p_rf = self.rf_model.predict_proba(X_latest)
        p_lr = self.lr_model.predict_proba(X_latest)
        p_ens_raw = create_ensemble_probabilities(p_rf, p_lr)
        p_ens_cal = self.calibrator.predict_proba(p_ens_raw)[0]

        p_down = float(p_ens_cal[0])
        p_neut = float(p_ens_cal[1])
        p_up = float(p_ens_cal[2])

        conviction = float(max(p_down, p_up))

        # 4. Gating Rule Application
        if conviction >= self.threshold:
            if p_up >= p_down:
                direction = "Up"
                signal_code = 1
                confidence = round(p_up * 100.0, 1)
            else:
                direction = "Down"
                signal_code = -1
                confidence = round(p_down * 100.0, 1)
            is_gated = True
        else:
            direction = "no clear signal"
            signal_code = 0
            confidence = round(max(p_down, p_neut, p_up) * 100.0, 1)
            is_gated = False

        return {
            "timestamp": latest_ts,
            "close_price": round(latest_close, 2),
            "direction": direction,
            "signal_code": signal_code,
            "confidence_pct": confidence,
            "conviction": round(conviction, 4),
            "threshold": self.threshold,
            "is_gated": is_gated,
            "probabilities": {
                "down": round(p_down, 4),
                "neutral": round(p_neut, 4),
                "up": round(p_up, 4),
            },
            "model_version": self.version,
        }


def get_default_predictor() -> TradePulsePredictor:
    """Factory helper to instantiate predictor with production defaults."""
    return TradePulsePredictor()
