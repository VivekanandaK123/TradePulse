"""
TradePulse - Dataset Split Unit Tests
Author: TradePulse Team (IIT Patna, IMPACT Lab)
"""

import sys
import unittest
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


class TestDatasetSplits(unittest.TestCase):

    def test_partition_files_exist_and_counts_match(self):
        train_path = PROCESSED_DIR / "train.csv"
        val_path = PROCESSED_DIR / "val.csv"
        test_path = PROCESSED_DIR / "test.csv"

        self.assertTrue(train_path.exists(), "train.csv does not exist.")
        self.assertTrue(val_path.exists(), "val.csv does not exist.")
        self.assertTrue(test_path.exists(), "test.csv does not exist.")

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

        self.assertEqual(len(train_df), 12720)
        self.assertEqual(len(val_df), 3276)
        self.assertEqual(len(test_df), 3666)

        # Check date monotonicity and non-overlap
        self.assertLess(train_df["session_date"].max(), "2026-07-01")
        self.assertGreater(val_df["session_date"].min(), "2026-07-01")
        self.assertLess(val_df["session_date"].max(), "2026-09-01")
        self.assertGreater(test_df["session_date"].min(), "2026-09-01")


if __name__ == "__main__":
    unittest.main()

