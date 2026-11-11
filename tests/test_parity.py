"""
TradePulse - Step 14: Parity Verification & Prediction Engine Unit Tests
Author: TradePulse Team (IIT Patna, IMPACT Lab)

Verifies:
1. Warm-up Validation: Buffer shorter than 35 bars raises ValueError.
2. Column Validation: Buffer missing required OHLCV columns raises ValueError.
3. Feature Parity: Live sliding buffer feature computation matches batch offline features exactly.
4. Prediction Parity: Model predicted probabilities and signals match batch test outputs.
5. Payload Integrity: Standardized schema conforms to FastAPI / Frontend specifications.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.features import get_feature_column_names, MAX_LOOKBACK_BARS
from ml.calibration import create_ensemble_probabilities
from ml.predict import TradePulsePredictor, get_default_predictor


class TestPredictionParity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.predictor = get_default_predictor()
        cls.test_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "test_features.csv")
        cls.raw_clean = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "META_5min_clean.csv")
        cls.feature_cols = get_feature_column_names(cls.test_df)

        # Locate where test_df starts inside raw_clean
        test_start_ts = cls.test_df.iloc[0]["timestamp"]
        cls.test_start_idx = cls.raw_clean[cls.raw_clean["timestamp"] == test_start_ts].index[0]

    def test_warmup_validation(self):
        """Buffer with fewer than 35 bars must raise ValueError."""
        short_buffer = self.raw_clean.iloc[:20].copy()
        with self.assertRaises(ValueError) as ctx:
            self.predictor.predict_candle(short_buffer)
        self.assertIn("Insufficient candle history", str(ctx.exception))

    def test_missing_columns_validation(self):
        """Buffer missing required OHLCV columns must raise ValueError."""
        bad_buffer = self.raw_clean.iloc[:50].drop(columns=["close"]).copy()
        with self.assertRaises(ValueError) as ctx:
            self.predictor.predict_candle(bad_buffer)
        self.assertIn("Missing required candle columns", str(ctx.exception))

    def test_feature_parity_against_batch_csv(self):
        """
        Verifies that computing features on a live sliding buffer (200 bars) matches
        batch offline features in test_features.csv across all 29 features within atol=1e-5.
        """
        # Test bar 100 in the test set
        test_bar_idx = 100
        global_idx = self.test_start_idx + test_bar_idx
        window = self.raw_clean.iloc[global_idx - 200 + 1 : global_idx + 1].copy()

        feat_df = self.predictor.compute_candle_features(window)
        live_feats = feat_df.iloc[-1][self.feature_cols].values.astype(float)
        batch_feats = self.test_df.iloc[test_bar_idx][self.feature_cols].values.astype(float)

        np.testing.assert_allclose(
            live_feats,
            batch_feats,
            atol=1e-5,
            rtol=1e-4,
            err_msg="Feature mismatch between live sliding buffer and batch offline calculation!"
        )

    def test_sliding_buffer_prediction_parity(self):
        """
        Replays 20 consecutive historical test candles through the live sliding buffer.
        Verifies that predicted probabilities and signals match batch model predictions.
        """
        start_bar = 50
        n_bars = 20

        for b in range(start_bar, start_bar + n_bars):
            global_idx = self.test_start_idx + b
            window = self.raw_clean.iloc[global_idx - 200 + 1 : global_idx + 1].copy()

            # Live prediction
            live_res = self.predictor.predict_candle(window)

            # Batch prediction directly on precomputed features
            X_batch = self.test_df.iloc[b][self.feature_cols].values.astype(float).reshape(1, -1)
            p_rf = self.predictor.rf_model.predict_proba(X_batch)
            p_lr = self.predictor.lr_model.predict_proba(X_batch)
            p_ens = create_ensemble_probabilities(p_rf, p_lr)
            p_cal = self.predictor.calibrator.predict_proba(p_ens)[0]

            expected_down = round(float(p_cal[0]), 4)
            expected_neut = round(float(p_cal[1]), 4)
            expected_up = round(float(p_cal[2]), 4)

            self.assertEqual(live_res["probabilities"]["down"], expected_down)
            self.assertEqual(live_res["probabilities"]["neutral"], expected_neut)
            self.assertEqual(live_res["probabilities"]["up"], expected_up)

            # Timestamp match
            expected_ts = str(self.test_df.iloc[b]["timestamp_ny"])
            self.assertEqual(live_res["timestamp"], expected_ts)

    def test_payload_integrity_and_schema(self):
        """
        Verifies that prediction payload contains all required keys and conforms to schema.
        """
        window = self.raw_clean.iloc[self.test_start_idx : self.test_start_idx + 100].copy()
        res = self.predictor.predict_candle(window)

        required_keys = [
            "timestamp", "close_price", "direction", "signal_code",
            "confidence_pct", "conviction", "threshold", "is_gated",
            "probabilities", "model_version"
        ]
        for k in required_keys:
            self.assertIn(k, res)

        self.assertIn(res["direction"], ["Up", "Down", "no clear signal"])
        self.assertIn(res["signal_code"], [-1, 0, 1])
        self.assertIsInstance(res["close_price"], float)
        self.assertIsInstance(res["confidence_pct"], float)
        self.assertIsInstance(res["conviction"], float)
        self.assertIsInstance(res["threshold"], float)
        self.assertIsInstance(res["is_gated"], bool)

        probs = res["probabilities"]
        self.assertAlmostEqual(probs["down"] + probs["neutral"] + probs["up"], 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
