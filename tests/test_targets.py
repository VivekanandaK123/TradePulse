"""
TradePulse - Target Definition Unit Tests (Built-in unittest)
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

from ml.targets import (
    compute_forward_returns,
    compute_candidate_thresholds,
    assign_labels_for_rule,
)


class TestTargets(unittest.TestCase):

    def setUp(self):
        # Create a mock 2-day session with 10 bars per day
        timestamps_day1 = pd.date_range("2026-01-05 14:30:00+00:00", periods=10, freq="5min")
        timestamps_day2 = pd.date_range("2026-01-06 14:30:00+00:00", periods=10, freq="5min")

        ts = timestamps_day1.append(timestamps_day2)
        closes = [100.0 + i for i in range(10)] + [200.0 + i * 2 for i in range(10)]
        is_filled = [0] * 20
        # Mark bar 5 on day 1 as filled
        is_filled[5] = 1

        self.df = pd.DataFrame({
            "symbol": "META",
            "timestamp": ts,
            "close": closes,
            "is_filled": is_filled,
        })

    def test_forward_return_intraday_isolation(self):
        res = compute_forward_returns(self.df, horizon_bars=3)

        # In day 1: bar 0 close is 100.0, bar 3 close is 103.0 -> return is 3.0 / 100.0 = 0.03
        self.assertAlmostEqual(res.loc[0, "target_return"], 0.03, places=5)
        self.assertTrue(res.loc[0, "valid_target"])

        # Final 3 bars of day 1 (indices 7, 8, 9) must have NaN target
        self.assertTrue(pd.isna(res.loc[7, "target_return"]))
        self.assertTrue(pd.isna(res.loc[8, "target_return"]))
        self.assertTrue(pd.isna(res.loc[9, "target_return"]))
        self.assertFalse(res.loc[7, "valid_target"])
        self.assertFalse(res.loc[8, "valid_target"])
        self.assertFalse(res.loc[9, "valid_target"])

        # Verify bar 2 target points to bar 5 which has is_filled=1
        self.assertEqual(res.loc[2, "target_is_filled"], 1)
        self.assertEqual(res.loc[0, "target_is_filled"], 0)

    def test_candidate_thresholds_computation(self):
        res = compute_forward_returns(self.df, horizon_bars=3)
        params = compute_candidate_thresholds(res)

        self.assertIn("fixed_rules", params)
        self.assertIn("quantile_rules", params)
        self.assertIn("volatility_rules", params)
        self.assertIn("quantile_50", params["quantile_rules"])
        self.assertGreater(params["quantile_rules"]["quantile_50"]["theta"], 0.0)

    def test_assign_labels(self):
        res = compute_forward_returns(self.df, horizon_bars=3)
        params = {
            "fixed_rules": {
                "fixed_10bp": {"theta": 0.02}
            }
        }
        labels = assign_labels_for_rule(res, "fixed", "fixed_10bp", params)

        # Bar 0 has return 0.03 > 0.02 -> +1 (Up)
        self.assertEqual(labels.loc[0], 1.0)
        # Bar 7 is dropped end-of-day -> NaN label
        self.assertTrue(pd.isna(labels.loc[7]))


if __name__ == "__main__":
    unittest.main()

