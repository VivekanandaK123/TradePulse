"""
TradePulse - Evaluation & Model Training Unit Tests (Built-in unittest)
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
from ml.train import (
    MajorityBaseline,
    LogisticRegressionBaseline,
    RandomForestModel,
    XGBoostModel,
    evaluate_model_validation,
)


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

        self.assertIn("accuracy", m)
        self.assertIn("balanced_accuracy", m)
        self.assertIn("macro_f1", m)
        self.assertGreater(m["accuracy"], 0.70)
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

    def test_random_forest_model(self):
        np.random.seed(42)
        X = np.random.randn(100, 6)
        y = np.random.choice([-1, 0, 1], size=100)
        rf = RandomForestModel(n_estimators=10, max_depth=3, min_samples_leaf=5, random_state=42)
        rf.fit(X, y)
        preds = rf.predict(X)
        probs = rf.predict_proba(X)

        self.assertEqual(len(preds), 100)
        self.assertEqual(probs.shape, (100, 3))
        self.assertEqual(len(rf.feature_importances_), 6)

    def test_xgboost_model(self):
        np.random.seed(42)
        X = np.random.randn(100, 6)
        y = np.random.choice([-1, 0, 1], size=100)
        xgb_model = XGBoostModel(n_estimators=10, max_depth=3, random_state=42)
        xgb_model.fit(X, y)
        preds = xgb_model.predict(X)
        probs = xgb_model.predict_proba(X)

        self.assertEqual(len(preds), 100)
        self.assertEqual(probs.shape, (100, 3))
        # Ensure predictions are strictly in {-1, 0, 1}
        self.assertTrue(set(np.unique(preds)).issubset({-1, 0, 1}))
        self.assertEqual(len(xgb_model.feature_importances_), 6)

    def test_evaluate_model_validation(self):
        np.random.seed(42)
        n_train, n_val = 120, 40
        features = [f"feat_{i}" for i in range(4)]
        train_df = pd.DataFrame(np.random.randn(n_train, 4), columns=features)
        train_df["label"] = np.random.choice([-1, 0, 1], size=n_train)

        val_df = pd.DataFrame(np.random.randn(n_val, 4), columns=features)
        val_df["label"] = np.random.choice([-1, 0, 1], size=n_val)
        val_df["target_return"] = np.random.normal(0, 0.002, size=n_val)

        rf = RandomForestModel(n_estimators=10, max_depth=3, min_samples_leaf=5, random_state=42)
        res = evaluate_model_validation(
            model=rf,
            train_df=train_df,
            val_df=val_df,
            feature_cols=features,
            label_col="label",
        )

        self.assertIn("accuracy", res)
        self.assertIn("balanced_accuracy", res)
        self.assertIn("macro_f1", res)
        self.assertIn("cost_8bp_avg_trade_bps", res)
        self.assertIn("fixed_10pct_cov_prec", res)


if __name__ == "__main__":
    unittest.main()
