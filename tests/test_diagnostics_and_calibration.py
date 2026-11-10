"""
TradePulse - Step 9 & 10 Diagnostics, Calibration & Gating Unit Tests
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

from ml.diagnostics import (
    TimeOfDayBaseline,
    compute_direction_only_accuracy,
    compute_dual_roc_auc,
    compute_performance_by_tod,
    paired_day_block_bootstrap,
)
from ml.calibration import (
    MulticlassCalibrator,
    compute_multiclass_brier_score,
    evaluate_calibration_quality,
    create_ensemble_probabilities,
)
from ml.gating import (
    apply_confidence_gating,
    apply_quantile_gating,
    simulate_non_overlapping_trades,
    check_filled_sensitivity,
)


class TestDiagnosticsAndCalibration(unittest.TestCase):

    def test_time_of_day_baseline(self):
        n = 100
        timestamps = pd.date_range("2026-01-05 09:30", periods=n, freq="5min", tz="America/New_York")
        y = np.random.choice([-1, 0, 1], size=n)

        tod = TimeOfDayBaseline().fit(timestamps, y)
        probs = tod.predict_proba(timestamps)
        preds = tod.predict(timestamps)

        self.assertEqual(probs.shape, (n, 3))
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0))
        self.assertEqual(len(preds), n)

    def test_direction_only_accuracy(self):
        y_true = np.array([1, -1, 0, 1, -1, 0])
        y_pred = np.array([1, 1, 0, 1, -1, 0])
        res = compute_direction_only_accuracy(y_true, y_pred)

        self.assertEqual(res["n_directional_bars"], 4)
        # Matches: y_true=[1, -1, 1, -1], y_pred=[1, 1, 1, -1] -> 3/4 = 0.75
        self.assertAlmostEqual(res["direction_accuracy"], 0.75)
        self.assertIn("ci_lower", res)
        self.assertIn("ci_upper", res)

    def test_dual_roc_auc(self):
        y_true = np.array([0, 0, 1, -1, 1, -1, 0, 1])
        probs = np.array([
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.6, 0.3, 0.1],
            [0.2, 0.1, 0.7],
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ])
        aucs = compute_dual_roc_auc(y_true, probs)
        self.assertIsNotNone(aucs["auc_move_vs_nomove"])
        self.assertIsNotNone(aucs["auc_up_vs_down"])
        self.assertGreater(aucs["auc_move_vs_nomove"], 0.70)
        self.assertGreater(aucs["auc_up_vs_down"], 0.70)

    def test_multiclass_calibrator(self):
        np.random.seed(42)
        n = 200
        probs = np.random.dirichlet([1, 2, 1], size=n)
        y = np.random.choice([-1, 0, 1], size=n)

        for method in ["sigmoid", "isotonic"]:
            cal = MulticlassCalibrator(method=method).fit(probs, y)
            cal_probs = cal.predict_proba(probs)
            self.assertEqual(cal_probs.shape, (n, 3))
            self.assertTrue(np.allclose(cal_probs.sum(axis=1), 1.0))

    def test_confidence_gating_and_simulation(self):
        probs = np.array([
            [0.60, 0.20, 0.20],  # Down (conviction 0.60)
            [0.25, 0.50, 0.25],  # Neutral
            [0.10, 0.40, 0.50],  # Up (conviction 0.50)
            [0.42, 0.38, 0.20],  # Down (conviction 0.42)
        ])
        signals_45 = apply_confidence_gating(probs, threshold=0.45)
        # Bar 0: Down (-1), Bar 1: Neutral (0), Bar 2: Up (1), Bar 3: No clear signal (0, because 0.42 < 0.45)
        np.testing.assert_array_equal(signals_45, [-1, 0, 1, 0])

        df_sim = pd.DataFrame({
            "target_return": [0.0020, -0.0010, 0.0015, -0.0005],
            "session_date": ["2026-01-05"] * 4,
            "is_filled": [0, 0, 0, 0],
        })
        sim = simulate_non_overlapping_trades(df_sim, signals_45, horizon_bars=2, cost_bps=0.0008)
        self.assertGreaterEqual(sim["n_trades"], 1)
        self.assertIn("max_drawdown_pct", sim)


if __name__ == "__main__":
    unittest.main()

