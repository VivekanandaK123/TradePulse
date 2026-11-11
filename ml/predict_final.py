"""
TradePulse - Step 14 Final: Production Packaging & Inference Pipeline
Author: TradePulse Team (IIT Patna, IMPACT Lab)

Encapsulates the complete live inference workflow for the final refit model:
1. Validates closed candle buffers (checking causal warm-up requirement: >= 35 bars, recommended 200 bars).
2. Computes the exact 29 causal features using the shared ml.features pipeline.
3. Generates calibrated class probabilities using the final 50/50 Ensemble (RF + LR) and RobustScaler.
4. Applies the final conviction gating rule (tau = 0.2635, targeting 50.0% signal coverage).
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
from ml.calibration import MulticlassCalibrator

METADATA_DIR = PROJECT_ROOT / "artifacts" / "metadata"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
MODEL_VERSION = "TradePulse-v1.0-final"


class TradePulsePredictorFinal:
    """
    Production inference engine for live closed-candle signal generation using model_final.
    Enforces shared pipeline logic: identical feature transformations, robust scaling,
    calibrated probabilities, and pre-registered conviction gating.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None,
        calibrator_path: Optional[Path] = None,
        signal_rule_path: Optional[Path] = None,
        schema_path: Optional[Path] = None,
    ):
        self.model_path = model_path or (MODELS_DIR / "model_final.joblib")
        self.scaler_path = scaler_path or (MODELS_DIR / "scaler_final.joblib")
        self.calibrator_path = calibrator_path or (MODELS_DIR / "calibrator_final.joblib")
        self.signal_rule_path = signal_rule_path or (METADATA_DIR / "signal_rule_final.json")
        self.schema_path = schema_path or (METADATA_DIR / "feature_schema_final.json")

        # Load model bundle
        bundle = joblib.load(self.model_path)
        if isinstance(bundle, dict):
            self.rf_model = bundle["random_forest"]
            self.lr_model = bundle["logistic_regression"]
            self.scaler = bundle.get("scaler") or joblib.load(self.scaler_path)
            self.feature_names = bundle.get("feature_names")
            self.version = bundle.get("version", MODEL_VERSION)
        else:
            self.rf_model = bundle
            self.lr_model = None
            self.scaler = joblib.load(self.scaler_path)
            self.feature_names = None
            self.version = MODEL_VERSION

        # Load calibrator
        self.calibrator = joblib.load(self.calibrator_path)

        # Load signal rule metadata
        with open(self.signal_rule_path, "r", encoding="utf-8") as f:
            self.signal_rule = json.load(f)

        self.threshold = float(self.signal_rule["calibrated_conviction_threshold"])
        self.target_label = self.signal_rule.get("target_label", "label_fixed_15bp")

        # Load feature schema if available
        if self.schema_path.exists():
            with open(self.schema_path, "r", encoding="utf-8") as f:
                self.schema = json.load(f)
                if self.feature_names is None:
                    self.feature_names = self.schema.get("feature_names")

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

        # Compute features using the shared pipeline
        feat_df = compute_features(df)
        return feat_df

    def predict_candle(self, buffer_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Predicts directional signal and calibrated confidence for the LATEST closed candle.

        Parameters:
        -----------
        buffer_df : pd.DataFrame
            Chronologically sorted buffer of historical closed 5-minute candles
            containing at least 35 bars (recommend 200 bars) up to and including the target candle.

        Returns:
        --------
        dict: Standardized prediction payload for logging and chart presentation.
        """
        self.validate_buffer(buffer_df)

        # 1. Compute features across buffer
        feat_df = self.compute_candle_features(buffer_df)
        feature_cols = self.feature_names or get_feature_column_names(feat_df)

        # 2. Extract latest closed bar features
        latest_row = feat_df.iloc[-1]
        latest_ts = str(latest_row["timestamp_ny"])
        latest_close = float(latest_row["close"])
        X_latest = latest_row[feature_cols].values.astype(float).reshape(1, -1)

        # Check for NaNs
        if np.isnan(X_latest).any():
            nan_cols = [c for c, v in zip(feature_cols, X_latest[0]) if np.isnan(v)]
            raise ValueError(
                f"Feature vector for latest candle contains NaNs in columns {nan_cols}. "
                f"Ensure sufficient warm-up lookback."
            )

        # 3. Model Predictions & Calibration
        # Random Forest uses unscaled features
        p_rf = self.rf_model.predict_proba(X_latest)

        # Logistic Regression uses scaled features
        if self.lr_model is not None and self.scaler is not None:
            X_latest_sc = self.scaler.transform(X_latest)
            p_lr = self.lr_model.predict_proba(X_latest_sc)
            p_ens_raw = 0.5 * p_rf + 0.5 * p_lr
        else:
            p_ens_raw = p_rf

        # Platt Sigmoid Multiclass Calibration
        p_ens_cal = self.calibrator.predict_proba(p_ens_raw)[0]

        p_down = float(p_ens_cal[0])
        p_neut = float(p_ens_cal[1])
        p_up = float(p_ens_cal[2])

        conviction = float(max(p_down, p_up))

        # 4. Conviction Gating Rule Application (tau = 0.2635)
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


def get_default_predictor_final() -> TradePulsePredictorFinal:
    """Factory helper to instantiate final predictor with production defaults."""
    return TradePulsePredictorFinal()


if __name__ == "__main__":
    predictor = get_default_predictor_final()
    print("TradePulsePredictorFinal successfully initialized.")
    print(f"Model version: {predictor.version}")
    print(f"Features: {len(predictor.feature_names)}")
    print(f"Threshold tau: {predictor.threshold}")

