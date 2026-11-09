"""
TradePulse - Evaluation & Baseline Unit Tests (Built-in unittest)
Author: TradePulse Team (IIT Patna, IMPACT Lab)
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.evaluate import (
    PurgedWalkForwardCV,
    compute_classification_metrics,
    compute_realized_trade_metrics,
    compute_fixed_coverage_precision,
)
from ml.train import MajorityBaseline, LogisticRegressionBaseline


class TestEvaluation(unittest.TestCase):

    def test_purged_walk_forward_cv(self):
        n_bars = 1000
        df = pd.DataFrame({"x": np.arange(n_bars)})
        cv = PurgedWalkForwardCV(n_splits=5, purge_bars=3, min_train_fraction=0.40)
        splits = cv.split(df)

        self.assertEqual(len(splits), 5)
        for train_idx, val_idx in splits:
            # Check non-overlap and exact 3-bar purge gap
            self.assertEqual(val_idx[0] - train_idx[-1] - 1, 3)
            # Check chronological ordering
            self.assertLess(train_idx[-1], val_idx[0])

    def test_classification_metrics(self):
        y_true = np.array([1, -1, 0, 1, 0, -1, 0, 1, -1, 0])
        y_pred = np.array([1, 1, 0, 1, 0, -1, 0, 0, -1, 0])
        m = compute_classification_metrics(y_true, y_pred)

        self.assertIn("balanced_accuracy", m)
        self.assertIn("macro_f1", m)
        self.assertGreater(m["balanced_accuracy"], 0.70)
        self.assertGreater(m["macro_f1"], 0.70)

    def test_realized_trade_metrics(self):
        actual_r = pd.Series([0.0020, -0.0010, 0.0005])
        signals = pd.Series([1.0, -1.0, 0.0])  # Long win, Short win, Neutral
        m = compute_realized_trade_metrics(actual_r, signals, cost_levels=[0.0008])

        # Active trades: 2
        self.assertEqual(m["n_active_signals"], 2)
        # Net returns: trade 1 = 0.0020 - 0.0008 = +0.0012; trade 2 = -(-0.0010) - 0.0008 = +0.0002
        # Avg net return: (0.0012 + 0.0002) / 2 = 0.0007 = 7 bps
        self.assertAlmostEqual(m["cost_8bp_avg_trade_bps"], 7.0, places=1)
        self.assertEqual(m["cost_8bp_win_rate_pct"], 100.0)

    def test_majority_baseline(self):
        X = np.random.randn(20, 5)
        y = np.array([0] * 15 + [1] * 3 + [-1] * 2)
        base = MajorityBaseline().fit(X, y)
        preds = base.predict(X[:5])
        self.assertTrue((preds == 0).all())


if __name__ == "__main__":
    unittest.main()

