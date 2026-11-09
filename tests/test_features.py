"""
TradePulse - Feature Engineering Unit Tests (Built-in unittest)
Author: TradePulse Team (IIT Patna, IMPACT Lab)
"""

import sys
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.features import (
    compute_features,
    get_feature_column_names,
    fit_feature_scaler,
    transform_features,
    MAX_LOOKBACK_BARS,
)


class TestFeatures(unittest.TestCase):

    def setUp(self):
        # Create a mock series of 60 bars (above 35 warm-up)
        np.random.seed(42)
        n_bars = 60
        ts = pd.date_range("2026-01-05 14:30:00+00:00", periods=n_bars, freq="5min")

        prices = 600.0 + np.cumsum(np.random.randn(n_bars) * 0.5)
        highs = prices + np.random.uniform(0.1, 0.5, n_bars)
        lows = prices - np.random.uniform(0.1, 0.5, n_bars)
        opens = prices + np.random.uniform(-0.2, 0.2, n_bars)
        volumes = np.random.uniform(1000, 5000, n_bars)
        trades = np.random.uniform(50, 200, n_bars)
        vwaps = prices + np.random.uniform(-0.1, 0.1, n_bars)

        self.mock_df = pd.DataFrame({
            "symbol": "META",
            "timestamp": ts,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": volumes,
            "trade_count": trades,
            "vwap": vwaps,
            "is_filled": [0] * n_bars,
        })

    def test_feature_generation_and_causality(self):
        # 1. Compute features on all 60 bars
        feat_full = compute_features(self.mock_df)
        cols = get_feature_column_names(feat_full)

        self.assertEqual(len(cols), 29)

        # 2. Causality Test: Compute features on first 45 bars only
        mock_subset = self.mock_df.iloc[:45].copy()
        feat_subset = compute_features(mock_subset)

        # Bar 44 (the 45th bar) should have IDENTICAL feature values in both
        for c in cols:
            val_full = feat_full.loc[44, c]
            val_subset = feat_subset.loc[44, c]
            self.assertAlmostEqual(
                val_full,
                val_subset,
                places=6,
                msg=f"Causality leak detected in feature '{c}': {val_full} vs {val_subset}",
            )

    def test_no_nulls_after_warmup(self):
        feat = compute_features(self.mock_df)
        cols = get_feature_column_names(feat)

        # After MAX_LOOKBACK_BARS (35 bars), there should be zero NaNs in any feature column
        sliced = feat.iloc[MAX_LOOKBACK_BARS:][cols]
        self.assertFalse(sliced.isna().any().any(), "NaN found after warm-up lookback.")

    def test_rsi_bounds(self):
        feat = compute_features(self.mock_df)
        rsi = feat.iloc[MAX_LOOKBACK_BARS:]["feat_rsi_14"]
        self.assertTrue((rsi >= 0.0).all() and (rsi <= 1.0).all(), "RSI values out of [0, 1] range.")

    def test_scaler_fitting_and_transform(self):
        feat = compute_features(self.mock_df)
        cols = get_feature_column_names(feat)
        valid_rows = feat.iloc[MAX_LOOKBACK_BARS:].copy()

        scaler_path = PROJECT_ROOT / "artifacts" / "models" / "test_scaler.joblib"
        scaler = fit_feature_scaler(valid_rows, cols, scaler_path=scaler_path)

        scaled = transform_features(valid_rows, cols, scaler)
        self.assertEqual(len(scaled), len(valid_rows))
        self.assertEqual(len(scaled.columns), len(cols))

        if scaler_path.exists():
            scaler_path.unlink()


if __name__ == "__main__":
    unittest.main()

